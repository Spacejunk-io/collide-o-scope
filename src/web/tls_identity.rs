//! Transactional, permission-safe TLS identity storage for the LAN listener.
//!
//! One bounded envelope owns the certificate, private key, canonical SAN set,
//! schema version, and digest. Publication is a same-directory atomic replace:
//! a crash can therefore expose the prior complete identity or the new complete
//! identity, never a certificate/key/SAN mix assembled from separate files.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rcgen::SanType;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const IDENTITY_VERSION: u16 = 1;
const IDENTITY_FILE: &str = "tls-identity-v1.json";
const MAX_IDENTITY_BYTES: usize = 64 * 1024;
static IDENTITY_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TlsIdentity {
    pub(crate) cert_chain: Vec<Vec<u8>>,
    pub(crate) key_der: Vec<u8>,
    pub(crate) sans: Vec<String>,
    pub(crate) digest: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityEnvelope {
    version: u16,
    cert_der: Vec<u8>,
    private_key_der: Vec<u8>,
    sans: Vec<String>,
    digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum IdentityFaultPoint {
    Generation,
    StageWrite,
    FileSync,
    Rename,
    DirectorySync,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IdentityFaults {
    fail_at: Option<IdentityFaultPoint>,
}

impl IdentityFaults {
    #[cfg(test)]
    pub(crate) const fn at(point: IdentityFaultPoint) -> Self {
        Self {
            fail_at: Some(point),
        }
    }

    fn check(self, point: IdentityFaultPoint) -> Result<(), String> {
        if self.fail_at == Some(point) {
            Err(format!("injected TLS identity fault at {point:?}"))
        } else {
            Ok(())
        }
    }
}

pub(crate) fn default_identity_dir() -> PathBuf {
    crate::host_paths::state_root().join("tls")
}

/// Load a complete identity or transactionally mint a replacement when the
/// required SAN set has changed. Malformed, partially written, unknown-version,
/// or permission-unsafe state is rejected rather than silently overwritten.
pub(crate) fn load_or_create(
    dir: &Path,
    required_sans: &[String],
    faults: IdentityFaults,
) -> Result<TlsIdentity, String> {
    let required_sans = canonical_sans(required_sans.iter().cloned())?;
    secure_identity_dir(dir)?;
    let path = dir.join(IDENTITY_FILE);

    match load(&path)? {
        Some(identity) if covers(&identity.sans, &required_sans) => return Ok(identity),
        Some(_) | None => {}
    }

    faults.check(IdentityFaultPoint::Generation)?;
    let certified = rcgen::generate_simple_self_signed(required_sans.clone())
        .map_err(|error| format!("TLS certificate generation failed: {error}"))?;
    let cert_der = certified.cert.der().to_vec();
    let key_der = certified.key_pair.serialize_der();
    let digest = identity_digest(IDENTITY_VERSION, &cert_der, &key_der, &required_sans);
    let envelope = IdentityEnvelope {
        version: IDENTITY_VERSION,
        cert_der,
        private_key_der: key_der,
        sans: required_sans,
        digest,
    };

    // Validate the generated certificate's actual SAN extension before any
    // private material is staged. Key/certificate matching is independently
    // proven by rustls configuration in the listener startup path.
    let identity = validate_envelope(envelope)?;
    publish(&path, &identity, faults)?;
    let committed = load(&path)?.ok_or_else(|| {
        "TLS identity publication completed without a readable generation".to_string()
    })?;
    if committed != identity {
        return Err("TLS identity changed while its transaction was committing".to_string());
    }

    retire_legacy_identity_files(dir);
    Ok(committed)
}

fn covers(actual: &[String], required: &[String]) -> bool {
    required
        .iter()
        .all(|required| actual.binary_search(required).is_ok())
}

fn canonical_sans<I>(sans: I) -> Result<Vec<String>, String>
where
    I: IntoIterator<Item = String>,
{
    let mut canonical = BTreeSet::new();
    for san in sans {
        if san.is_empty() || san.len() > 255 || san.chars().any(char::is_control) {
            return Err("TLS identity contains an invalid subject alternative name".to_string());
        }
        canonical.insert(san.to_ascii_lowercase());
        if canonical.len() > 16 {
            return Err("TLS identity contains too many subject alternative names".to_string());
        }
    }
    if canonical.is_empty() {
        return Err("TLS identity requires at least one subject alternative name".to_string());
    }
    Ok(canonical.into_iter().collect())
}

fn identity_digest(version: u16, cert_der: &[u8], key_der: &[u8], sans: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"collide-o-scope/tls-identity\0");
    hasher.update(version.to_be_bytes());
    update_len_prefixed(&mut hasher, cert_der);
    update_len_prefixed(&mut hasher, key_der);
    hasher.update((sans.len() as u32).to_be_bytes());
    for san in sans {
        update_len_prefixed(&mut hasher, san.as_bytes());
    }
    hex_lower(&hasher.finalize())
}

fn update_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn load(path: &Path) -> Result<Option<TlsIdentity>, String> {
    reject_symlink(path, "TLS identity file")?;
    let Some(stored) = crate::controller_profile::read_bounded_document(path, MAX_IDENTITY_BYTES)
        .map_err(|error| match error {
        crate::controller_profile::BoundedDocumentReadError::TooLarge(bytes) => {
            format!("TLS identity file is too large ({bytes} bytes)")
        }
        crate::controller_profile::BoundedDocumentReadError::Io(message) => {
            format!("TLS identity read failed: {message}")
        }
    })?
    else {
        return Ok(None);
    };
    verify_identity_file_permissions(path)?;
    let plaintext = unprotect_for_current_user(&stored)?;
    if plaintext.len() > MAX_IDENTITY_BYTES {
        return Err("TLS identity plaintext exceeds its 64 KiB limit".to_string());
    }
    let envelope: IdentityEnvelope = serde_json::from_slice(&plaintext)
        .map_err(|error| format!("TLS identity envelope is invalid: {error}"))?;
    validate_envelope(envelope).map(Some)
}

fn reject_symlink(path: &Path, label: &str) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(format!("{label} must not be a symbolic link"))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("{label} metadata inspection failed: {error}")),
    }
}

fn validate_envelope(envelope: IdentityEnvelope) -> Result<TlsIdentity, String> {
    if envelope.version != IDENTITY_VERSION {
        return Err(format!(
            "unsupported TLS identity version {}",
            envelope.version
        ));
    }
    if envelope.cert_der.is_empty() || envelope.private_key_der.is_empty() {
        return Err("TLS identity certificate or private key is empty".to_string());
    }
    if envelope.cert_der.len() + envelope.private_key_der.len() > MAX_IDENTITY_BYTES {
        return Err("TLS identity key material exceeds its 64 KiB limit".to_string());
    }
    let sans = canonical_sans(envelope.sans.clone())?;
    if sans != envelope.sans {
        return Err("TLS identity SANs are not canonical and unique".to_string());
    }
    let expected = identity_digest(
        envelope.version,
        &envelope.cert_der,
        &envelope.private_key_der,
        &sans,
    );
    if envelope.digest != expected {
        return Err("TLS identity digest mismatch".to_string());
    }

    let params = rcgen::CertificateParams::from_ca_cert_der(&envelope.cert_der.as_slice().into())
        .map_err(|error| format!("TLS identity certificate parse failed: {error}"))?;
    let parsed_sans = canonical_sans(params.subject_alt_names.into_iter().map(san_string))?;
    if parsed_sans != sans {
        return Err("TLS identity SAN manifest does not match its certificate".to_string());
    }

    Ok(TlsIdentity {
        cert_chain: vec![envelope.cert_der],
        key_der: envelope.private_key_der,
        sans,
        digest: expected,
    })
}

fn san_string(san: SanType) -> String {
    match san {
        SanType::DnsName(name) => name.as_str().to_string(),
        SanType::IpAddress(ip) => ip.to_string(),
        SanType::Rfc822Name(name) => format!("email:{name}"),
        SanType::URI(uri) => format!("uri:{uri}"),
        other => format!("unsupported:{other:?}"),
    }
}

fn publish(path: &Path, identity: &TlsIdentity, faults: IdentityFaults) -> Result<(), String> {
    let envelope = IdentityEnvelope {
        version: IDENTITY_VERSION,
        cert_der: identity.cert_chain[0].clone(),
        private_key_der: identity.key_der.clone(),
        sans: identity.sans.clone(),
        digest: identity.digest.clone(),
    };
    let plaintext = serde_json::to_vec(&envelope)
        .map_err(|error| format!("TLS identity serialization failed: {error}"))?;
    if plaintext.len() > MAX_IDENTITY_BYTES {
        return Err("TLS identity serialization exceeds its 64 KiB limit".to_string());
    }
    let stored = protect_for_current_user(&plaintext)?;
    if stored.len() > MAX_IDENTITY_BYTES {
        return Err("protected TLS identity exceeds its 64 KiB limit".to_string());
    }

    let parent = path
        .parent()
        .ok_or_else(|| "TLS identity path has no parent directory".to_string())?;
    let file_name = path
        .file_name()
        .ok_or_else(|| "TLS identity path has no file name".to_string())?;
    let mut staged = None;
    for _ in 0..16 {
        let sequence = IDENTITY_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}.tmp-{}-{sequence}",
            file_name.to_string_lossy(),
            std::process::id()
        ));
        match open_private_staging_file(&candidate) {
            Ok(file) => {
                staged = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(format!("TLS identity staging failed: {error}")),
        }
    }
    let (staged_path, mut staged_file) =
        staged.ok_or_else(|| "could not reserve a unique TLS identity staging file".to_string())?;
    let publication = (|| {
        faults.check(IdentityFaultPoint::StageWrite)?;
        staged_file
            .write_all(&stored)
            .map_err(|error| format!("TLS identity staged write failed: {error}"))?;
        faults.check(IdentityFaultPoint::FileSync)?;
        staged_file
            .sync_all()
            .map_err(|error| format!("TLS identity staged sync failed: {error}"))?;
        drop(staged_file);
        faults.check(IdentityFaultPoint::Rename)?;
        atomic_replace(&staged_path, path)
            .map_err(|error| format!("TLS identity atomic publish failed: {error}"))?;
        verify_identity_file_permissions(path)?;
        faults.check(IdentityFaultPoint::DirectorySync)?;
        sync_parent_directory(path)
            .map_err(|error| format!("TLS identity directory sync failed: {error}"))
    })();
    if publication.is_err() {
        let _ = fs::remove_file(&staged_path);
    }
    publication
}

#[cfg(unix)]
fn secure_identity_dir(dir: &Path) -> Result<(), String> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true).mode(0o700);
    builder
        .create(dir)
        .map_err(|error| format!("TLS identity directory creation failed: {error}"))?;
    reject_symlink(dir, "TLS identity directory")?;
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
        .map_err(|error| format!("TLS identity directory permissions failed: {error}"))?;
    let mode = fs::metadata(dir)
        .map_err(|error| format!("TLS identity directory inspection failed: {error}"))?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err("TLS identity directory is accessible by another Unix user".to_string());
    }
    Ok(())
}

#[cfg(windows)]
fn secure_identity_dir(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir)
        .map_err(|error| format!("TLS identity directory creation failed: {error}"))?;
    reject_symlink(dir, "TLS identity directory")
}

#[cfg(not(any(unix, windows)))]
fn secure_identity_dir(_dir: &Path) -> Result<(), String> {
    Err("TLS identity protection is unavailable on this platform".to_string())
}

#[cfg(unix)]
fn open_private_staging_file(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

#[cfg(windows)]
fn open_private_staging_file(path: &Path) -> io::Result<File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

#[cfg(not(any(unix, windows)))]
fn open_private_staging_file(_path: &Path) -> io::Result<File> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "private TLS identity files are unsupported on this platform",
    ))
}

#[cfg(unix)]
fn verify_identity_file_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .map_err(|error| format!("TLS identity file permissions failed: {error}"))?;
    let mode = fs::metadata(path)
        .map_err(|error| format!("TLS identity file inspection failed: {error}"))?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err("TLS identity file is accessible by another Unix user".to_string());
    }
    Ok(())
}

#[cfg(windows)]
fn verify_identity_file_permissions(_path: &Path) -> Result<(), String> {
    // The complete envelope, including certificate metadata, is encrypted by
    // DPAPI's current-user scope. A copied file cannot be unprotected by a
    // different Windows account, even when an inherited directory ACL is broad.
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn verify_identity_file_permissions(_path: &Path) -> Result<(), String> {
    Err("TLS identity permission verification is unavailable on this platform".to_string())
}

#[cfg(not(windows))]
fn protect_for_current_user(bytes: &[u8]) -> Result<Vec<u8>, String> {
    Ok(bytes.to_vec())
}

#[cfg(not(windows))]
fn unprotect_for_current_user(bytes: &[u8]) -> Result<Vec<u8>, String> {
    Ok(bytes.to_vec())
}

#[cfg(windows)]
fn protect_for_current_user(bytes: &[u8]) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };
    let length = u32::try_from(bytes.len())
        .map_err(|_| "TLS identity is too large for Windows DPAPI".to_string())?;
    let input = CRYPT_INTEGER_BLOB {
        cbData: length,
        pbData: bytes.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB::default();
    let result = unsafe {
        CryptProtectData(
            &input,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if result == 0 {
        return Err(format!(
            "Windows user-scope TLS identity protection failed: {}",
            io::Error::last_os_error()
        ));
    }
    let protected =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
    unsafe {
        LocalFree(output.pbData.cast());
    }
    Ok(protected)
}

#[cfg(windows)]
fn unprotect_for_current_user(bytes: &[u8]) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };
    let length = u32::try_from(bytes.len())
        .map_err(|_| "TLS identity is too large for Windows DPAPI".to_string())?;
    let input = CRYPT_INTEGER_BLOB {
        cbData: length,
        pbData: bytes.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB::default();
    let mut description = std::ptr::null_mut();
    let result = unsafe {
        CryptUnprotectData(
            &input,
            &mut description,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if result == 0 {
        return Err(format!(
            "Windows user-scope TLS identity unprotect failed: {}",
            io::Error::last_os_error()
        ));
    }
    let plaintext =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
    unsafe {
        LocalFree(output.pbData.cast());
        if !description.is_null() {
            LocalFree(description.cast());
        }
    }
    Ok(plaintext)
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };
    let source = source
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

fn sync_parent_directory(child: &Path) -> io::Result<()> {
    let Some(parent) = child.parent() else {
        return Ok(());
    };
    #[cfg(unix)]
    {
        File::open(parent)?.sync_all()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(parent)?
            .sync_all()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = parent;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "directory sync is unsupported on this platform",
        ))
    }
}

fn retire_legacy_identity_files(dir: &Path) {
    for legacy in ["cert.der", "key.der", "sans.txt"] {
        match fs::remove_file(dir.join(legacy)) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => log::warn!("Could not retire inactive legacy TLS material: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(label: &str) -> PathBuf {
        let mut random = [0_u8; 8];
        getrandom::fill(&mut random).unwrap();
        std::env::temp_dir().join(format!(
            "collide-o-scope-tls-{label}-{}-{}",
            std::process::id(),
            hex_lower(&random)
        ))
    }

    fn sans() -> Vec<String> {
        vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
            "::1".to_string(),
            "192.0.2.10".to_string(),
        ]
    }

    fn mint_identity(subject_alt_names: Vec<String>) -> TlsIdentity {
        let subject_alt_names = canonical_sans(subject_alt_names).unwrap();
        let certified = rcgen::generate_simple_self_signed(subject_alt_names.clone()).unwrap();
        let cert_der = certified.cert.der().to_vec();
        let private_key_der = certified.key_pair.serialize_der();
        let digest = identity_digest(
            IDENTITY_VERSION,
            &cert_der,
            &private_key_der,
            &subject_alt_names,
        );
        validate_envelope(IdentityEnvelope {
            version: IDENTITY_VERSION,
            cert_der,
            private_key_der,
            sans: subject_alt_names,
            digest,
        })
        .unwrap()
    }

    #[test]
    fn one_versioned_identity_round_trips_with_real_cert_sans_and_digest() {
        let dir = test_dir("round-trip");
        let identity = load_or_create(&dir, &sans(), IdentityFaults::default()).unwrap();
        assert!(covers(&identity.sans, &sans()));
        assert_eq!(identity.digest.len(), 64);
        let loaded = load_or_create(&dir, &sans(), IdentityFaults::default()).unwrap();
        assert_eq!(loaded, identity);
        assert!(dir.join(IDENTITY_FILE).is_file());
        assert!(!dir.join("cert.der").exists());
        assert!(!dir.join("key.der").exists());
        assert!(!dir.join("sans.txt").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unknown_version_digest_corruption_and_san_manifest_mismatch_are_rejected() {
        let dir = test_dir("hostile");
        let identity = load_or_create(&dir, &sans(), IdentityFaults::default()).unwrap();
        let path = dir.join(IDENTITY_FILE);
        let stored = fs::read(&path).unwrap();
        let plaintext = unprotect_for_current_user(&stored).unwrap();
        let original: serde_json::Value = serde_json::from_slice(&plaintext).unwrap();

        for (label, mutate) in [
            ("version", ("version", serde_json::json!(99))),
            ("digest", ("digest", serde_json::json!("00".repeat(32)))),
            (
                "sans",
                ("sans", serde_json::json!(["127.0.0.1", "localhost"])),
            ),
        ] {
            let mut hostile = original.clone();
            hostile[mutate.0] = mutate.1;
            let protected =
                protect_for_current_user(&serde_json::to_vec(&hostile).unwrap()).unwrap();
            fs::write(&path, protected).unwrap();
            assert!(load(&path).is_err(), "accepted hostile {label} envelope");
        }

        publish(&path, &identity, IdentityFaults::default()).unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn every_pre_rename_fault_preserves_the_prior_identity_and_post_rename_is_complete() {
        let dir = test_dir("faults");
        let prior = load_or_create(&dir, &sans(), IdentityFaults::default()).unwrap();
        let replacement = mint_identity(
            sans()
                .into_iter()
                .chain(["198.51.100.8".to_string()])
                .collect(),
        );

        for point in [
            IdentityFaultPoint::StageWrite,
            IdentityFaultPoint::FileSync,
            IdentityFaultPoint::Rename,
        ] {
            assert!(publish(
                &dir.join(IDENTITY_FILE),
                &replacement,
                IdentityFaults::at(point)
            )
            .is_err());
            assert_eq!(load(&dir.join(IDENTITY_FILE)).unwrap().unwrap(), prior);
        }

        assert!(publish(
            &dir.join(IDENTITY_FILE),
            &replacement,
            IdentityFaults::at(IdentityFaultPoint::DirectorySync),
        )
        .is_err());
        assert_eq!(
            load(&dir.join(IDENTITY_FILE)).unwrap().unwrap(),
            replacement,
            "a post-rename fault may expose the complete new identity, never a mix"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn rustls_rejects_a_private_key_from_another_identity() {
        let first = mint_identity(sans());
        let second = mint_identity(sans());
        assert!(axum_server::tls_rustls::RustlsConfig::from_der(
            first.cert_chain.clone(),
            second.key_der,
        )
        .await
        .is_err());
        assert!(
            axum_server::tls_rustls::RustlsConfig::from_der(first.cert_chain, first.key_der)
                .await
                .is_ok()
        );
    }

    #[test]
    fn generation_fault_never_publishes_a_partial_identity() {
        let dir = test_dir("generation-fault");
        assert!(load_or_create(
            &dir,
            &sans(),
            IdentityFaults::at(IdentityFaultPoint::Generation)
        )
        .is_err());
        assert!(!dir.join(IDENTITY_FILE).exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn unix_identity_directory_and_file_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = test_dir("permissions");
        load_or_create(&dir, &sans(), IdentityFaults::default()).unwrap();
        assert_eq!(
            fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(dir.join(IDENTITY_FILE))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn unix_identity_directory_and_bundle_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;

        let root = test_dir("symlinks");
        let real_dir = root.join("real");
        fs::create_dir_all(&real_dir).unwrap();
        let linked_dir = root.join("linked");
        symlink(&real_dir, &linked_dir).unwrap();
        assert!(load_or_create(&linked_dir, &sans(), IdentityFaults::default()).is_err());

        let identity = load_or_create(&real_dir, &sans(), IdentityFaults::default()).unwrap();
        let identity_path = real_dir.join(IDENTITY_FILE);
        let target = root.join("copied-identity");
        fs::copy(&identity_path, &target).unwrap();
        fs::remove_file(&identity_path).unwrap();
        symlink(&target, &identity_path).unwrap();
        assert!(load(&identity_path).is_err());
        assert!(!identity.key_der.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_identity_is_dpapi_user_scope_ciphertext() {
        let dir = test_dir("dpapi");
        let identity = load_or_create(&dir, &sans(), IdentityFaults::default()).unwrap();
        let stored = fs::read(dir.join(IDENTITY_FILE)).unwrap();
        assert!(!stored
            .windows(identity.key_der.len())
            .any(|window| window == identity.key_der));
        assert!(unprotect_for_current_user(&stored).is_ok());
        fs::remove_dir_all(dir).unwrap();
    }
}
