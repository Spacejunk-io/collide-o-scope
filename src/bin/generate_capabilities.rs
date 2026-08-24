use collide_o_scope::capability::{
    canonical_registry_document, canonical_registry_json, generated_capability_markdown,
    generated_proxy_surface_snippet, generated_readme_summary, CapabilityKey, CapabilityLimitation,
    CapabilityRegistryDocument, CapabilityStatus, CapabilitySurface,
};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const README_START: &str = "<!-- BEGIN GENERATED CAPABILITY SUMMARY -->";
const README_END: &str = "<!-- END GENERATED CAPABILITY SUMMARY -->";
const PROXY_START: &str = "<!-- BEGIN GENERATED PROXY CAPABILITY -->";
const PROXY_END: &str = "<!-- END GENERATED PROXY CAPABILITY -->";
const STALE_PROXY_CLAIM: &str = "browser panel has no proxy surface";
const STALE_HARDWARE_DECODE_CLAIM: &str = "does not add a hardware decoder";
const CAMPAIGN_SCHEMA: &str = "collide-o-scope-audit-campaign-status/1";
const CAMPAIGN_STATUS_VOCABULARY: [&str; 5] = [
    "retained",
    "implemented",
    "evaluation",
    "deferred",
    "rejected",
];
const FINAL_AUDIT_CAMPAIGNS: [(&str, &str, &str); 5] = [
    ("d1_study_motion_abi_1_1", "implemented", "complete"),
    (
        "d2_photosensitivity_advisor",
        "evaluation",
        "accessibility_legal_and_p1_gpu_timing",
    ),
    (
        "d3_portable_show_bundle",
        "retained",
        "machine_a_clean_machine_b_live_export",
    ),
    ("d4_accepted_creative_mutation", "implemented", "complete"),
    (
        "d5_straight_alpha_key_fill",
        "retained",
        "application_action_live_acquisition_and_p1_readback",
    ),
];

#[derive(Debug, Deserialize)]
struct CampaignDocument {
    schema: String,
    campaigns: Vec<CampaignRecord>,
}

#[derive(Debug, Deserialize)]
struct CampaignRecord {
    id: String,
    status: String,
    gate: String,
}

fn validate_campaign_truth(path: &Path) -> Result<(), String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("read '{}': {error}", path.display()))?;
    let document: CampaignDocument = serde_json::from_str(&source)
        .map_err(|error| format!("parse '{}': {error}", path.display()))?;
    if document.schema != CAMPAIGN_SCHEMA {
        return Err(format!(
            "{} has unsupported schema {}",
            path.display(),
            document.schema
        ));
    }
    let mut ids = BTreeSet::new();
    for campaign in &document.campaigns {
        if !ids.insert(campaign.id.as_str()) {
            return Err(format!("duplicate campaign id {}", campaign.id));
        }
        if !CAMPAIGN_STATUS_VOCABULARY.contains(&campaign.status.as_str()) {
            return Err(format!(
                "campaign {} uses unknown status {}",
                campaign.id, campaign.status
            ));
        }
    }
    for (id, status, gate) in FINAL_AUDIT_CAMPAIGNS {
        let campaign = document
            .campaigns
            .iter()
            .find(|campaign| campaign.id == id)
            .ok_or_else(|| format!("missing final audit campaign {id}"))?;
        if campaign.status != status || campaign.gate != gate {
            return Err(format!(
                "campaign {id} contradicts its final receipt: expected {status}/{gate}, found {}/{}",
                campaign.status, campaign.gate
            ));
        }
    }
    Ok(())
}

fn validate_registry_evidence(document: &CapabilityRegistryDocument) -> Result<(), String> {
    for platform in &document.platforms {
        for record in &platform.capabilities {
            if record.evidence_receipt_ids.is_empty() {
                return Err(format!(
                    "{} registry status record {} has no evidence receipt",
                    platform.platform.label(),
                    record.key.as_str()
                ));
            }
            if record
                .evidence_receipt_ids
                .iter()
                .any(|receipt| receipt.0.trim().is_empty())
            {
                return Err(format!(
                    "{} registry status record {} has an empty evidence receipt ID",
                    platform.platform.label(),
                    record.key.as_str()
                ));
            }
        }
    }
    Ok(())
}

fn validate_registry_audit_boundary() -> Result<(), String> {
    let document = canonical_registry_document();
    validate_registry_evidence(&document)?;
    for platform in document.platforms {
        for key in [
            CapabilityKey::AcceptedCreativeMutationV1,
            CapabilityKey::StudyMotionAbi11,
            CapabilityKey::TransactionalControlListeners,
            CapabilityKey::CorrelatedEngineGpuTiming,
            CapabilityKey::SourceDescriptorColorTruth,
            CapabilityKey::SupervisedGpuRecoveryPhaseA,
        ] {
            let record = platform
                .capabilities
                .iter()
                .find(|record| record.key == key)
                .ok_or_else(|| {
                    format!(
                        "{} registry omits {}",
                        platform.platform.label(),
                        key.as_str()
                    )
                })?;
            if record.status != CapabilityStatus::Implemented {
                return Err(format!(
                    "{} registry understates retained operator capability {} as {:?}",
                    platform.platform.label(),
                    key.as_str(),
                    record.status
                ));
            }
            if record.surfaces.is_empty()
                || record.evidence_receipt_ids.is_empty()
                || record.known_limitations.is_empty()
            {
                return Err(format!(
                    "{} registry has an incomplete operator capability record for {}",
                    platform.platform.label(),
                    key.as_str()
                ));
            }
        }

        let find = |key| {
            platform
                .capabilities
                .iter()
                .find(|record| record.key == key)
                .expect("validated operator capability exists")
        };
        let listeners = find(CapabilityKey::TransactionalControlListeners);
        if !listeners.surfaces.iter().any(|surface| {
            surface.surface == CapabilitySurface::PhysicalVenue
                && surface.status == CapabilityStatus::EvaluationRequired
        }) {
            return Err(
                "transactional control listeners omit the external venue proof gate".into(),
            );
        }
        let timing = find(CapabilityKey::CorrelatedEngineGpuTiming);
        if !timing.known_limitations.iter().any(|limitation| {
            limitation.code == CapabilityLimitation::EngineSubmissionIsNotPhotonTime
        }) || !timing.known_limitations.iter().any(|limitation| {
            limitation.code == CapabilityLimitation::PhysicalTimingAndPerformanceProofRequired
        }) {
            return Err(
                "correlated engine/GPU timing must retain its non-photon and physical/performance gates"
                    .into(),
            );
        }
        let descriptors = find(CapabilityKey::SourceDescriptorColorTruth);
        for surface in [
            CapabilitySurface::LiveProgram,
            CapabilitySurface::OfflineExport,
        ] {
            if !descriptors.surfaces.iter().any(|entry| {
                entry.surface == surface && entry.status == CapabilityStatus::EvaluationRequired
            }) {
                return Err(format!(
                    "source descriptor truth incorrectly upgrades stopped {surface:?} integration"
                ));
            }
        }
        let recovery = find(CapabilityKey::SupervisedGpuRecoveryPhaseA);
        if recovery
            .surfaces
            .iter()
            .any(|surface| surface.surface == CapabilitySurface::LiveProgram)
            || !recovery.known_limitations.iter().any(|limitation| {
                limitation.code == CapabilityLimitation::TransparentGpuContinuityUnavailable
            })
        {
            return Err(
                "Phase-A recovery must not advertise transparent Phase-B continuity".into(),
            );
        }
    }
    let json = canonical_registry_json();
    for unavailable in [
        "photosensitivity_advisor",
        "portable_show_bundle",
        "straight_alpha_key_fill",
        "source_display_geometry_application",
        "transparent_gpu_recovery",
    ] {
        if json.contains(unavailable) {
            return Err(format!(
                "registry advertises audit capability without an operator-visible integration: {unavailable}"
            ));
        }
    }
    Ok(())
}

fn require_document_claim(path: &Path, claim: &str) -> Result<(), String> {
    let document =
        fs::read_to_string(path).map_err(|error| format!("read '{}': {error}", path.display()))?;
    if document.contains(claim) {
        Ok(())
    } else {
        Err(format!(
            "{} contradicts final audit truth; missing `{claim}`",
            path.display()
        ))
    }
}

fn validate_rfc_truth(root: &Path) -> Result<(), String> {
    require_document_claim(
        &root.join("docs/rfcs/d1-study-motion-abi-1.1.md"),
        "ABI 1.1 implemented additively",
    )?;
    require_document_claim(
        &root.join("docs/rfcs/d2-photosensitivity-risk-advisor.md"),
        "production remains\ndeferred",
    )?;
    require_document_claim(
        &root.join("docs/rfcs/d3-portable-show-bundle.md"),
        "operator UI",
    )?;
    require_document_claim(
        &root.join("docs/rfcs/d4-accepted-creative-mutation.md"),
        "Status: **implemented",
    )?;
    require_document_claim(
        &root.join("docs/rfcs/d5-straight-alpha-export.md"),
        "cannot be reached by the current MP4 action",
    )?;
    require_document_claim(
        &root.join("docs/rfcs/d5-straight-alpha-and-key-fill-export.md"),
        "superseded; not an availability statement",
    )
}

fn with_document_line_endings(document: &str, generated: &str) -> String {
    if document.contains("\r\n") {
        generated.replace('\n', "\r\n")
    } else {
        generated.to_owned()
    }
}

fn replace_region(
    document: &str,
    start_marker: &str,
    end_marker: &str,
    generated: &str,
) -> Result<String, String> {
    let start = document
        .find(start_marker)
        .ok_or_else(|| format!("missing generated-region marker {start_marker}"))?;
    let end_start = document[start..]
        .find(end_marker)
        .map(|offset| start + offset)
        .ok_or_else(|| format!("missing generated-region marker {end_marker}"))?;
    let end = end_start + end_marker.len();
    let generated = with_document_line_endings(document, generated);
    let mut result = String::with_capacity(document.len() - (end - start) + generated.len());
    result.push_str(&document[..start]);
    result.push_str(&generated);
    result.push_str(&document[end..]);
    Ok(result)
}

fn expected_region_file(
    path: &Path,
    start: &str,
    end: &str,
    generated: &str,
) -> Result<String, String> {
    let document =
        fs::read_to_string(path).map_err(|error| format!("read '{}': {error}", path.display()))?;
    replace_region(&document, start, end, generated)
}

fn check_or_write(path: &Path, expected: &str, check: bool) -> Result<(), String> {
    let actual = fs::read_to_string(path).unwrap_or_default();
    if actual == expected {
        return Ok(());
    }
    if check {
        return Err(format!(
            "generated capability artifact is stale: {}; run `cargo run --locked --bin generate_capabilities`",
            path.display()
        ));
    }
    fs::write(path, expected).map_err(|error| format!("write '{}': {error}", path.display()))
}

fn main() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let check = match arguments.next().as_deref() {
        None => false,
        Some("--check") => true,
        Some(argument) => return Err(format!("unknown argument: {argument}")),
    };
    if let Some(argument) = arguments.next() {
        return Err(format!("unexpected argument: {argument}"));
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let readme_path = root.join("README.md");
    let precision_path = root.join("docs/precision-and-scale.md");
    let markdown_path = root.join("docs/capability-registry.md");
    let json_path = root.join("docs/capability-registry.json");
    let campaign_path = root.join("docs/campaigns/audit-campaign-status.json");

    validate_campaign_truth(&campaign_path)?;
    validate_registry_audit_boundary()?;
    validate_rfc_truth(&root)?;

    let expected_readme = expected_region_file(
        &readme_path,
        README_START,
        README_END,
        &generated_readme_summary(),
    )?;
    let expected_precision = expected_region_file(
        &precision_path,
        PROXY_START,
        PROXY_END,
        &generated_proxy_surface_snippet(),
    )?;
    if expected_precision.contains(STALE_PROXY_CLAIM) {
        return Err(format!(
            "{} still contains the contradicted proxy claim",
            precision_path.display()
        ));
    }
    if expected_precision.contains(STALE_HARDWARE_DECODE_CLAIM) {
        return Err(format!(
            "{} still denies the evaluation-only D3D11VA backend",
            precision_path.display()
        ));
    }

    check_or_write(&markdown_path, &generated_capability_markdown(), check)?;
    check_or_write(&json_path, &canonical_registry_json(), check)?;
    check_or_write(&readme_path, &expected_readme, check)?;
    check_or_write(&precision_path, &expected_precision, check)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_is_exact_and_requires_both_markers() {
        let source = "before\n<!-- BEGIN GENERATED CAPABILITY SUMMARY -->\nold\n<!-- END GENERATED CAPABILITY SUMMARY -->\nafter\n";
        let expected = "before\n<!-- BEGIN GENERATED CAPABILITY SUMMARY -->\nnew\n<!-- END GENERATED CAPABILITY SUMMARY -->\nafter\n";
        let generated = "<!-- BEGIN GENERATED CAPABILITY SUMMARY -->\nnew\n<!-- END GENERATED CAPABILITY SUMMARY -->";
        assert_eq!(
            replace_region(source, README_START, README_END, generated).unwrap(),
            expected
        );
        assert!(replace_region("none", README_START, README_END, generated).is_err());
    }

    #[test]
    fn p10_final_audit_truth_and_registry_boundary_validate() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        validate_campaign_truth(&root.join("docs/campaigns/audit-campaign-status.json")).unwrap();
        validate_registry_audit_boundary().unwrap();
        validate_rfc_truth(&root).unwrap();
    }

    #[test]
    fn generated_status_records_fail_closed_without_nonempty_evidence() {
        let mut missing = canonical_registry_document();
        missing.platforms[0].capabilities[0]
            .evidence_receipt_ids
            .clear();
        assert!(validate_registry_evidence(&missing).is_err());

        let mut empty = canonical_registry_document();
        empty.platforms[0].capabilities[0].evidence_receipt_ids[0]
            .0
            .clear();
        assert!(validate_registry_evidence(&empty).is_err());
    }
}
