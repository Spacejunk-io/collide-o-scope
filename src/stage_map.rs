//! Portable venue/output calibration, deliberately separate from artistic patches.
//!
//! A [`StageMap`] names physical endpoints and maps the program canvas into
//! bounded perspective or polygon slices.  It contains no creative layer state,
//! decoder state, or GPU resources.  Validation is atomic at the document
//! boundary, while runtime planning reports failures per endpoint so one bad or
//! disconnected output never prevents the remaining venue outputs from running.

use std::collections::HashSet;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

pub const STAGE_MAP_SCHEMA_VERSION: u16 = 1;
pub const STAGE_MAP_FILE_NAME: &str = "stage_map.yaml";
pub const MAX_STAGE_MAP_BYTES: usize = 1 << 20;
pub const MAX_OUTPUT_ENDPOINTS: usize = 16;
pub const MAX_SLICES_PER_ENDPOINT: usize = 64;
pub const MAX_STAGE_SLICES: usize = 256;
pub const MAX_POLYGON_VERTICES: usize = 8;
pub const MAX_ENDPOINT_NAME_BYTES: usize = 96;
pub const MAX_MONITOR_SELECTOR_BYTES: usize = 256;
pub const MAX_SLICE_NAME_BYTES: usize = 96;
const MAX_STAGE_MAP_STATUS_BYTES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageMapLoadStatus {
    Loaded,
    DefaultMissing,
    DefaultInvalid(String),
    DefaultIo(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageMapLoad {
    pub path: PathBuf,
    pub document: StageMap,
    pub status: StageMapLoadStatus,
}

pub fn default_stage_map_path() -> PathBuf {
    default_stage_state_dir().join(STAGE_MAP_FILE_NAME)
}

#[cfg_attr(
    test,
    allow(
        dead_code,
        reason = "App tests isolate themselves from the operator's default StageMap file"
    )
)]
pub fn load_default_stage_map() -> StageMapLoad {
    load_stage_map_or_default(&default_stage_map_path())
}

pub fn load_stage_map_or_default(path: &Path) -> StageMapLoad {
    let (document, status) = match read_bounded_stage_map(path) {
        Ok(Some(bytes)) => match StageMap::from_yaml_bytes(&bytes) {
            Ok(document) => (document, StageMapLoadStatus::Loaded),
            Err(error) => (
                StageMap::default(),
                StageMapLoadStatus::DefaultInvalid(bounded_stage_status(error.to_string())),
            ),
        },
        Ok(None) => (StageMap::default(), StageMapLoadStatus::DefaultMissing),
        Err(error) if error.kind() == io::ErrorKind::InvalidData => (
            StageMap::default(),
            StageMapLoadStatus::DefaultInvalid(bounded_stage_status(error.to_string())),
        ),
        Err(error) => (
            StageMap::default(),
            StageMapLoadStatus::DefaultIo(bounded_stage_status(error.to_string())),
        ),
    };
    StageMapLoad {
        path: path.to_path_buf(),
        document,
        status,
    }
}

#[cfg_attr(
    test,
    allow(
        dead_code,
        reason = "App tests validate publication without writing the operator's default StageMap file"
    )
)]
pub fn save_default_stage_map_atomic(document: &StageMap) -> Result<PathBuf, StageMapError> {
    let path = default_stage_map_path();
    save_stage_map_atomic(document, &path)?;
    Ok(path)
}

pub fn save_stage_map_atomic(document: &StageMap, path: &Path) -> Result<(), StageMapError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(StageMapError::Io)?;
    }
    document.save_atomic(path, true)?;
    Ok(())
}

fn default_stage_state_dir() -> PathBuf {
    if let Some(base) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(base).join("collide-o-scope").join("stage");
    }
    if let Some(base) = std::env::var_os("XDG_STATE_HOME") {
        return PathBuf::from(base).join("collide-o-scope").join("stage");
    }
    if let Some(base) = std::env::var_os("HOME") {
        return PathBuf::from(base)
            .join(".local")
            .join("state")
            .join("collide-o-scope")
            .join("stage");
    }
    PathBuf::from(".collide-o-scope").join("stage")
}

fn read_bounded_stage_map(path: &Path) -> io::Result<Option<Vec<u8>>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if file.metadata()?.len() > MAX_STAGE_MAP_BYTES as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("StageMap exceeds the {MAX_STAGE_MAP_BYTES}-byte limit"),
        ));
    }
    let mut bytes = Vec::with_capacity(MAX_STAGE_MAP_BYTES.min(64 * 1024));
    Read::by_ref(&mut file)
        .take((MAX_STAGE_MAP_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_STAGE_MAP_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("StageMap exceeds the {MAX_STAGE_MAP_BYTES}-byte limit"),
        ));
    }
    Ok(Some(bytes))
}

fn bounded_stage_status(value: String) -> String {
    value.chars().take(MAX_STAGE_MAP_STATUS_BYTES).collect()
}

/// Stable, venue-authored endpoint identity. It is intentionally a constrained
/// string so operators may use meaningful IDs without exposing arbitrary paths.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutputEndpointId(String);

impl OutputEndpointId {
    pub fn parse(value: impl Into<String>) -> Result<Self, StageMapError> {
        let value = value.into();
        if value.is_empty() || value.len() > 128 {
            return Err(StageMapError::InvalidEndpointId);
        }
        if !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(StageMapError::InvalidEndpointId);
        }
        Ok(Self(value))
    }

    pub fn legacy() -> Self {
        Self("legacy-output-1".to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for OutputEndpointId {
    fn default() -> Self {
        Self::legacy()
    }
}

impl fmt::Display for OutputEndpointId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for OutputEndpointId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for OutputEndpointId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StageSliceId(u64);

impl StageSliceId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestCardMode {
    #[default]
    Off,
    SmpteBars,
    Grid,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StageRoute {
    #[default]
    Program,
    Blackout,
    TestCard {
        mode: TestCardMode,
    },
}

/// The one predicate every editor-only native control answers from.
///
/// Single-monitor audience Output reuses the main window's already-proven
/// swapchain, so while that is happening the main surface *is* the audience
/// and every native control must disappear from it. A dedicated output has its
/// own clean surface and may leave the controls on the preview.
///
/// This lives beside [`StageSurface`] because it is a leakage decision, not a
/// window-management one, and it is a single function rather than four copies
/// of `!output_on_main` so the RECOVERY strip, the patch editor, the health
/// HUD, the native gesture surface, and the transform gizmo cannot drift into
/// disagreeing about when the preview is safe to draw on.
pub const fn native_controls_visible(output_on_main: bool) -> bool {
    !output_on_main
}

/// Surfaces against which leakage policies are decided. Only a specifically
/// selected physical endpoint may receive calibration paint.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    dead_code,
    reason = "all surfaces are retained for the endpoint presenter leakage proof"
)]
pub enum StageSurface {
    EditorPreview,
    PhysicalOutput(OutputEndpointId),
    Composite,
    Audience,
    Spout,
    Record,
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StageCalibrationDecision {
    pub substitute_with_test_card: bool,
    pub overlay_output_identification: bool,
}

/// Host-session stage tools. These controls are not creative state and are not
/// stored in PatchState. Endpoint matching is exact and fail-closed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StageToolState {
    health_hud: bool,
    /// B11: the monitoring-bay overlay toggle. Host-session like the HUD;
    /// deliberately independent of it, because an operator reading the
    /// instruments does not want the timing HUD forced on beside them.
    monitor_bay: bool,
    test_card: TestCardMode,
    test_card_endpoint: Option<OutputEndpointId>,
    output_identification: bool,
    output_identification_endpoint: Option<OutputEndpointId>,
}

impl StageToolState {
    pub fn health_hud_enabled(&self) -> bool {
        self.health_hud
    }

    pub fn set_health_hud(&mut self, enabled: bool) {
        self.health_hud = enabled;
    }

    pub fn monitor_bay_enabled(&self) -> bool {
        self.monitor_bay
    }

    pub fn set_monitor_bay(&mut self, enabled: bool) {
        self.monitor_bay = enabled;
    }

    pub fn test_card(&self) -> TestCardMode {
        self.test_card
    }

    pub fn test_card_endpoint(&self) -> Option<&OutputEndpointId> {
        self.test_card_endpoint.as_ref()
    }

    pub fn output_identification_enabled(&self) -> bool {
        self.output_identification
    }

    pub fn output_identification_endpoint(&self) -> Option<&OutputEndpointId> {
        self.output_identification_endpoint.as_ref()
    }

    pub fn set_test_card(
        &mut self,
        mode: TestCardMode,
        endpoint: Option<OutputEndpointId>,
    ) -> Result<(), StageMapError> {
        if mode != TestCardMode::Off && endpoint.is_none() {
            return Err(StageMapError::CalibrationEndpointRequired);
        }
        self.test_card = mode;
        self.test_card_endpoint = if mode == TestCardMode::Off {
            None
        } else {
            endpoint
        };
        Ok(())
    }

    pub fn set_output_identification(
        &mut self,
        enabled: bool,
        endpoint: Option<OutputEndpointId>,
    ) -> Result<(), StageMapError> {
        if enabled && endpoint.is_none() {
            return Err(StageMapError::CalibrationEndpointRequired);
        }
        self.output_identification = enabled;
        self.output_identification_endpoint = if enabled { endpoint } else { None };
        Ok(())
    }

    #[allow(
        dead_code,
        reason = "consumed when endpoint presentation applies calibration overrides"
    )]
    pub fn decision_for(&self, surface: &StageSurface) -> StageCalibrationDecision {
        let StageSurface::PhysicalOutput(endpoint) = surface else {
            return StageCalibrationDecision::default();
        };
        StageCalibrationDecision {
            substitute_with_test_card: self.test_card != TestCardMode::Off
                && self.test_card_endpoint.as_ref() == Some(endpoint),
            overlay_output_identification: self.output_identification
                && self.output_identification_endpoint.as_ref() == Some(endpoint),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum OutputBinding {
    #[default]
    Unassigned,
    Monitor {
        selector: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Default for NormalizedRect {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        }
    }
}

impl NormalizedRect {
    fn validate(self) -> Result<(), StageMapError> {
        let values = [self.x, self.y, self.width, self.height];
        if !values.into_iter().all(f32::is_finite)
            || self.x < 0.0
            || self.y < 0.0
            || self.width <= 0.0
            || self.height <= 0.0
            || self.x + self.width > 1.0
            || self.y + self.height > 1.0
        {
            return Err(StageMapError::InvalidSourceRegion);
        }
        Ok(())
    }

    fn corners(self) -> [[f32; 2]; 4] {
        [
            [self.x, self.y],
            [self.x + self.width, self.y],
            [self.x + self.width, self.y + self.height],
            [self.x, self.y + self.height],
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedQuad {
    /// Clockwise or counter-clockwise corners. Self-intersection is rejected.
    pub points: [[f32; 2]; 4],
}

impl Default for NormalizedQuad {
    fn default() -> Self {
        Self {
            points: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
    }
}

impl NormalizedQuad {
    fn validate(self) -> Result<(), StageMapError> {
        validate_convex_polygon(&self.points).map_err(|_| StageMapError::InvalidOutputRegion)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StageGeometry {
    PerspectiveQuad {
        #[serde(default)]
        source: NormalizedRect,
        #[serde(default)]
        output: NormalizedQuad,
    },
    /// Bounded convex polygon mapping. Source and output vertex counts and
    /// winding must match; rendering triangulates them as one ordered fan.
    Polygon {
        source: Vec<[f32; 2]>,
        output: Vec<[f32; 2]>,
    },
}

impl Default for StageGeometry {
    fn default() -> Self {
        Self::PerspectiveQuad {
            source: NormalizedRect::default(),
            output: NormalizedQuad::default(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StageMask {
    #[default]
    None,
    /// Edge softness ordered left, top, right, bottom in normalized output units.
    EdgeFeather { softness: [f32; 4] },
    /// A convex output-space mask, optionally inverted.
    Polygon {
        points: Vec<[f32; 2]>,
        #[serde(default)]
        invert: bool,
        #[serde(default)]
        softness: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageCalibration {
    pub opacity: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub gamma: f32,
    pub gain: [f32; 3],
    pub black_level: [f32; 3],
}

impl Default for StageCalibration {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            brightness: 1.0,
            contrast: 1.0,
            gamma: 1.0,
            gain: [1.0; 3],
            black_level: [0.0; 3],
        }
    }
}

impl StageCalibration {
    fn validate(self) -> Result<(), StageMapError> {
        if ![self.opacity, self.brightness, self.contrast, self.gamma]
            .into_iter()
            .chain(self.gain)
            .chain(self.black_level)
            .all(f32::is_finite)
            || !(0.0..=1.0).contains(&self.opacity)
            || !(0.0..=2.0).contains(&self.brightness)
            || !(0.0..=2.0).contains(&self.contrast)
            || !(0.25..=4.0).contains(&self.gamma)
            || !self
                .gain
                .into_iter()
                .all(|value| (0.0..=2.0).contains(&value))
            || !self
                .black_level
                .into_iter()
                .all(|value| (0.0..=0.25).contains(&value))
        {
            return Err(StageMapError::InvalidCalibration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageSlice {
    pub id: StageSliceId,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub geometry: StageGeometry,
    #[serde(default)]
    pub mask: StageMask,
    #[serde(default)]
    pub calibration: StageCalibration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageEndpoint {
    pub id: OutputEndpointId,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub binding: OutputBinding,
    pub output_size: [u32; 2],
    #[serde(default = "default_refresh_millihz")]
    pub refresh_millihz: u32,
    #[serde(default)]
    pub route: StageRoute,
    #[serde(default)]
    pub slices: Vec<StageSlice>,
}

impl StageEndpoint {
    #[allow(dead_code, reason = "retained for the typed native StageMap editor")]
    pub fn new(id: OutputEndpointId, name: impl Into<String>, output_size: [u32; 2]) -> Self {
        Self {
            id,
            name: name.into(),
            enabled: true,
            binding: OutputBinding::Unassigned,
            output_size,
            refresh_millihz: default_refresh_millihz(),
            route: StageRoute::Program,
            slices: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageMap {
    pub schema_version: u16,
    pub next_slice_id: u64,
    #[serde(default)]
    pub endpoints: Vec<StageEndpoint>,
}

impl Default for StageMap {
    fn default() -> Self {
        Self {
            schema_version: STAGE_MAP_SCHEMA_VERSION,
            next_slice_id: 1,
            endpoints: Vec::new(),
        }
    }
}

impl StageMap {
    #[allow(dead_code, reason = "consumed by the native venue-document picker")]
    pub fn from_yaml_bytes(bytes: &[u8]) -> Result<Self, StageMapError> {
        if bytes.len() > MAX_STAGE_MAP_BYTES {
            return Err(StageMapError::DocumentTooLarge);
        }
        let mut value: Self = serde_yaml::from_slice(bytes)
            .map_err(|error| StageMapError::Deserialize(error.to_string()))?;
        value.observe_slice_ids()?;
        value.validate()?;
        Ok(value)
    }

    #[allow(dead_code, reason = "consumed by the native venue-document picker")]
    pub fn to_yaml_bytes(&self) -> Result<Vec<u8>, StageMapError> {
        self.validate()?;
        let bytes = serde_yaml::to_string(self)
            .map_err(|error| StageMapError::Serialize(error.to_string()))?
            .into_bytes();
        if bytes.len() > MAX_STAGE_MAP_BYTES {
            return Err(StageMapError::DocumentTooLarge);
        }
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), StageMapError> {
        if self.schema_version != STAGE_MAP_SCHEMA_VERSION {
            return Err(StageMapError::UnsupportedSchema(self.schema_version));
        }
        if self.next_slice_id == 0 {
            return Err(StageMapError::InvalidNextSliceId);
        }
        if self.endpoints.len() > MAX_OUTPUT_ENDPOINTS {
            return Err(StageMapError::TooManyEndpoints);
        }

        let mut endpoint_ids = HashSet::with_capacity(self.endpoints.len());
        let mut slice_ids = HashSet::new();
        let mut slice_count = 0_usize;
        for endpoint in &self.endpoints {
            if !endpoint_ids.insert(endpoint.id.clone()) {
                return Err(StageMapError::DuplicateEndpoint(endpoint.id.clone()));
            }
            validate_name(&endpoint.name, MAX_ENDPOINT_NAME_BYTES)?;
            if let OutputBinding::Monitor { selector } = &endpoint.binding {
                if selector.is_empty() || selector.len() > MAX_MONITOR_SELECTOR_BYTES {
                    return Err(StageMapError::InvalidMonitorSelector);
                }
            }
            if endpoint.output_size[0] == 0
                || endpoint.output_size[1] == 0
                || endpoint.refresh_millihz == 0
                || endpoint.refresh_millihz > 1_000_000
            {
                return Err(StageMapError::InvalidOutputMode(endpoint.id.clone()));
            }
            if matches!(
                endpoint.route,
                StageRoute::TestCard {
                    mode: TestCardMode::Off
                }
            ) {
                return Err(StageMapError::InvalidTestCardRoute(endpoint.id.clone()));
            }
            if endpoint.slices.len() > MAX_SLICES_PER_ENDPOINT {
                return Err(StageMapError::TooManyEndpointSlices(endpoint.id.clone()));
            }
            slice_count = slice_count
                .checked_add(endpoint.slices.len())
                .ok_or(StageMapError::TooManySlices)?;
            if slice_count > MAX_STAGE_SLICES {
                return Err(StageMapError::TooManySlices);
            }
            for slice in &endpoint.slices {
                if slice.id.get() == 0 || !slice_ids.insert(slice.id) {
                    return Err(StageMapError::DuplicateOrZeroSliceId(slice.id.get()));
                }
                if slice.id.get() >= self.next_slice_id {
                    return Err(StageMapError::InvalidNextSliceId);
                }
                validate_name(&slice.name, MAX_SLICE_NAME_BYTES)?;
                validate_geometry(&slice.geometry)?;
                validate_mask(&slice.mask)?;
                slice.calibration.validate()?;
            }
        }
        Ok(())
    }

    #[allow(dead_code, reason = "retained for the typed native StageMap editor")]
    pub fn add_endpoint(&mut self, endpoint: StageEndpoint) -> Result<(), StageMapError> {
        if self.endpoints.len() >= MAX_OUTPUT_ENDPOINTS {
            return Err(StageMapError::TooManyEndpoints);
        }
        if self.endpoints.iter().any(|live| live.id == endpoint.id) {
            return Err(StageMapError::DuplicateEndpoint(endpoint.id));
        }
        let mut staged = self.clone();
        staged.endpoints.push(endpoint);
        staged.validate()?;
        *self = staged;
        Ok(())
    }

    #[allow(dead_code, reason = "retained for the typed native StageMap editor")]
    pub fn remove_endpoint(&mut self, endpoint: &OutputEndpointId) -> bool {
        let Some(index) = self.endpoints.iter().position(|live| &live.id == endpoint) else {
            return false;
        };
        self.endpoints.remove(index);
        true
    }

    #[allow(dead_code, reason = "retained for the typed native StageMap editor")]
    pub fn add_slice(
        &mut self,
        endpoint: &OutputEndpointId,
        name: impl Into<String>,
        geometry: StageGeometry,
    ) -> Result<StageSliceId, StageMapError> {
        let id = StageSliceId(self.next_slice_id);
        let next = self
            .next_slice_id
            .checked_add(1)
            .ok_or(StageMapError::SliceIdExhausted)?;
        let Some(endpoint_index) = self.endpoints.iter().position(|live| &live.id == endpoint)
        else {
            return Err(StageMapError::MissingEndpoint(endpoint.clone()));
        };
        let mut staged = self.clone();
        staged.next_slice_id = next;
        staged.endpoints[endpoint_index].slices.push(StageSlice {
            id,
            name: name.into(),
            enabled: true,
            geometry,
            mask: StageMask::None,
            calibration: StageCalibration::default(),
        });
        staged.validate()?;
        *self = staged;
        Ok(id)
    }

    #[allow(dead_code, reason = "retained for the typed native StageMap editor")]
    pub fn remove_slice(&mut self, id: StageSliceId) -> bool {
        for endpoint in &mut self.endpoints {
            if let Some(index) = endpoint.slices.iter().position(|slice| slice.id == id) {
                endpoint.slices.remove(index);
                return true;
            }
        }
        false
    }

    pub fn evaluate_isolated(
        &self,
        limits: StageDeviceLimits,
        mut endpoint_available: impl FnMut(&StageEndpoint) -> Result<(), String>,
    ) -> Vec<StageEndpointEvaluation> {
        self.endpoints
            .iter()
            .map(|endpoint| {
                let result = plan_endpoint(endpoint, limits, &mut endpoint_available);
                StageEndpointEvaluation {
                    endpoint_id: endpoint.id.clone(),
                    result,
                }
            })
            .collect()
    }

    #[allow(dead_code, reason = "consumed by the native venue-document picker")]
    pub fn save_atomic(&self, path: &Path, replace: bool) -> Result<(), StageMapError> {
        let bytes = self.to_yaml_bytes()?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let stem = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("stage-map.yaml");
        let nonce = random_nonce()?;
        let temp = parent.join(format!(".{stem}.{nonce:016x}.tmp"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(StageMapError::Io)?;
        let publication = (|| {
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            publish_file(&temp, path, replace)?;
            sync_parent(path);
            Ok::<(), io::Error>(())
        })();
        if publication.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        publication.map_err(StageMapError::Io)
    }

    fn observe_slice_ids(&mut self) -> Result<(), StageMapError> {
        let maximum = self
            .endpoints
            .iter()
            .flat_map(|endpoint| endpoint.slices.iter())
            .map(|slice| slice.id.get())
            .max()
            .unwrap_or(0);
        if maximum >= self.next_slice_id {
            self.next_slice_id = maximum
                .checked_add(1)
                .ok_or(StageMapError::SliceIdExhausted)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageDeviceLimits {
    pub max_dimension: u32,
    pub max_pixels_per_endpoint: u64,
    pub max_vertices_per_endpoint: usize,
}

impl Default for StageDeviceLimits {
    fn default() -> Self {
        Self {
            max_dimension: 8192,
            max_pixels_per_endpoint: 33_554_432,
            max_vertices_per_endpoint: 512,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageMeshVertex {
    pub source_uv: [f32; 2],
    pub output_uv: [f32; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageSlicePlan {
    pub id: StageSliceId,
    pub vertices: Vec<StageMeshVertex>,
    pub indices: Vec<u16>,
    /// Row-major projective map from output UV to source UV. Polygon slices
    /// use per-triangle interpolation and therefore leave this absent.
    pub output_to_source: Option<[f32; 9]>,
    pub mask: StageMask,
    pub calibration: StageCalibration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageEndpointPlan {
    pub id: OutputEndpointId,
    pub output_size: [u32; 2],
    pub refresh_millihz: u32,
    pub route: StageRoute,
    pub slices: Vec<StageSlicePlan>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageEndpointEvaluation {
    pub endpoint_id: OutputEndpointId,
    pub result: Result<Option<StageEndpointPlan>, StageEndpointRuntimeError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageEndpointRuntimeError {
    Unavailable(String),
    DimensionsExceedDevice,
    PixelBudgetExceeded,
    VertexBudgetExceeded,
    InvalidGeometry,
}

fn plan_endpoint(
    endpoint: &StageEndpoint,
    limits: StageDeviceLimits,
    endpoint_available: &mut impl FnMut(&StageEndpoint) -> Result<(), String>,
) -> Result<Option<StageEndpointPlan>, StageEndpointRuntimeError> {
    if !endpoint.enabled {
        return Ok(None);
    }
    endpoint_available(endpoint).map_err(StageEndpointRuntimeError::Unavailable)?;
    if endpoint.output_size[0] > limits.max_dimension
        || endpoint.output_size[1] > limits.max_dimension
    {
        return Err(StageEndpointRuntimeError::DimensionsExceedDevice);
    }
    let pixels = u64::from(endpoint.output_size[0])
        .checked_mul(u64::from(endpoint.output_size[1]))
        .ok_or(StageEndpointRuntimeError::PixelBudgetExceeded)?;
    if pixels > limits.max_pixels_per_endpoint {
        return Err(StageEndpointRuntimeError::PixelBudgetExceeded);
    }

    let mut vertex_count = 0_usize;
    let mut slices = Vec::with_capacity(endpoint.slices.len());
    for slice in endpoint.slices.iter().filter(|slice| slice.enabled) {
        let geometry =
            plan_geometry(&slice.geometry).ok_or(StageEndpointRuntimeError::InvalidGeometry)?;
        vertex_count = vertex_count
            .checked_add(geometry.vertices.len())
            .ok_or(StageEndpointRuntimeError::VertexBudgetExceeded)?;
        if vertex_count > limits.max_vertices_per_endpoint {
            return Err(StageEndpointRuntimeError::VertexBudgetExceeded);
        }
        slices.push(StageSlicePlan {
            id: slice.id,
            vertices: geometry.vertices,
            indices: geometry.indices,
            output_to_source: geometry.output_to_source,
            mask: slice.mask.clone(),
            calibration: slice.calibration,
        });
    }

    Ok(Some(StageEndpointPlan {
        id: endpoint.id.clone(),
        output_size: endpoint.output_size,
        refresh_millihz: endpoint.refresh_millihz,
        route: endpoint.route.clone(),
        slices,
    }))
}

struct PlannedGeometry {
    vertices: Vec<StageMeshVertex>,
    indices: Vec<u16>,
    output_to_source: Option<[f32; 9]>,
}

fn plan_geometry(geometry: &StageGeometry) -> Option<PlannedGeometry> {
    match geometry {
        StageGeometry::PerspectiveQuad { source, output } => {
            let source_points = source.corners();
            let vertices = source_points
                .into_iter()
                .zip(output.points)
                .map(|(source_uv, output_uv)| StageMeshVertex {
                    source_uv,
                    output_uv,
                })
                .collect();
            let output_to_source = solve_homography(output.points, source_points)?;
            Some(PlannedGeometry {
                vertices,
                indices: vec![0, 1, 2, 0, 2, 3],
                output_to_source: Some(output_to_source),
            })
        }
        StageGeometry::Polygon { source, output } => {
            if source.len() != output.len() || source.len() < 3 {
                return None;
            }
            let vertices = source
                .iter()
                .copied()
                .zip(output.iter().copied())
                .map(|(source_uv, output_uv)| StageMeshVertex {
                    source_uv,
                    output_uv,
                })
                .collect();
            let mut indices = Vec::with_capacity((source.len() - 2) * 3);
            for index in 1..source.len() - 1 {
                indices.extend_from_slice(&[0, index as u16, (index + 1) as u16]);
            }
            Some(PlannedGeometry {
                vertices,
                indices,
                output_to_source: None,
            })
        }
    }
}

fn validate_geometry(geometry: &StageGeometry) -> Result<(), StageMapError> {
    match geometry {
        StageGeometry::PerspectiveQuad { source, output } => {
            source.validate()?;
            output.validate()?;
            if solve_homography(output.points, source.corners()).is_none() {
                return Err(StageMapError::InvalidOutputRegion);
            }
        }
        StageGeometry::Polygon { source, output } => {
            if source.len() != output.len()
                || !(3..=MAX_POLYGON_VERTICES).contains(&source.len())
                || polygon_winding(source) == 0
                || polygon_winding(output) == 0
                || polygon_winding(source) != polygon_winding(output)
            {
                return Err(StageMapError::InvalidPolygonMapping);
            }
            validate_convex_polygon(source)
                .and_then(|()| validate_convex_polygon(output))
                .map_err(|_| StageMapError::InvalidPolygonMapping)?;
        }
    }
    Ok(())
}

fn validate_mask(mask: &StageMask) -> Result<(), StageMapError> {
    match mask {
        StageMask::None => Ok(()),
        StageMask::EdgeFeather { softness } => {
            if softness
                .iter()
                .all(|value| value.is_finite() && (0.0..=0.5).contains(value))
            {
                Ok(())
            } else {
                Err(StageMapError::InvalidMask)
            }
        }
        StageMask::Polygon {
            points, softness, ..
        } => {
            if !(3..=MAX_POLYGON_VERTICES).contains(&points.len())
                || !softness.is_finite()
                || !(0.0..=0.5).contains(softness)
            {
                return Err(StageMapError::InvalidMask);
            }
            validate_convex_polygon(points).map_err(|_| StageMapError::InvalidMask)
        }
    }
}

fn validate_convex_polygon(points: &[[f32; 2]]) -> Result<(), ()> {
    if points.len() < 3
        || points.len() > MAX_POLYGON_VERTICES
        || points.iter().flatten().any(|value| !value.is_finite())
        || points
            .iter()
            .flatten()
            .any(|value| !(0.0..=1.0).contains(value))
    {
        return Err(());
    }
    let mut sign = 0_i8;
    for index in 0..points.len() {
        let a = points[index];
        let b = points[(index + 1) % points.len()];
        let c = points[(index + 2) % points.len()];
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() <= 1.0e-6 {
            return Err(());
        }
        let current = if cross > 0.0 { 1 } else { -1 };
        if sign == 0 {
            sign = current;
        } else if sign != current {
            return Err(());
        }
    }
    Ok(())
}

fn polygon_winding(points: &[[f32; 2]]) -> i8 {
    if points.len() < 3 {
        return 0;
    }
    let area = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let next = points[(index + 1) % points.len()];
            point[0] * next[1] - next[0] * point[1]
        })
        .sum::<f32>();
    if !area.is_finite() || area.abs() <= 1.0e-6 {
        0
    } else if area > 0.0 {
        1
    } else {
        -1
    }
}

/// Solve a projective map from four `from` points to four `to` points.
fn solve_homography(from: [[f32; 2]; 4], to: [[f32; 2]; 4]) -> Option<[f32; 9]> {
    let mut matrix = [[0.0_f64; 9]; 8];
    for (index, (source, destination)) in from.into_iter().zip(to).enumerate() {
        let x = f64::from(source[0]);
        let y = f64::from(source[1]);
        let u = f64::from(destination[0]);
        let v = f64::from(destination[1]);
        matrix[index * 2] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        matrix[index * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for column in 0..8 {
        let pivot = (column..8).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if matrix[pivot][column].abs() <= 1.0e-10 {
            return None;
        }
        matrix.swap(column, pivot);
        let divisor = matrix[column][column];
        for value in &mut matrix[column][column..] {
            *value /= divisor;
        }
        let pivot_row = matrix[column];
        for (row_index, row) in matrix.iter_mut().enumerate() {
            if row_index == column {
                continue;
            }
            let factor = row[column];
            for (value, pivot_value) in row[column..].iter_mut().zip(&pivot_row[column..]) {
                *value -= factor * pivot_value;
            }
        }
    }
    let result = [
        matrix[0][8] as f32,
        matrix[1][8] as f32,
        matrix[2][8] as f32,
        matrix[3][8] as f32,
        matrix[4][8] as f32,
        matrix[5][8] as f32,
        matrix[6][8] as f32,
        matrix[7][8] as f32,
        1.0,
    ];
    result.into_iter().all(f32::is_finite).then_some(result)
}

fn validate_name(value: &str, max_bytes: usize) -> Result<(), StageMapError> {
    if value.trim().is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        Err(StageMapError::InvalidName)
    } else {
        Ok(())
    }
}

fn default_true() -> bool {
    true
}

fn default_refresh_millihz() -> u32 {
    60_000
}

fn random_nonce() -> Result<u64, StageMapError> {
    let mut bytes = [0_u8; 8];
    getrandom::fill(&mut bytes).map_err(|error| StageMapError::Random(error.to_string()))?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(windows)]
fn publish_file(source: &Path, destination: &Path, replace: bool) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    if !replace && destination.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "StageMap destination already exists",
        ));
    }
    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let flags = MOVEFILE_WRITE_THROUGH
        | if replace {
            MOVEFILE_REPLACE_EXISTING
        } else {
            0
        };
    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), flags) } == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn publish_file(source: &Path, destination: &Path, replace: bool) -> io::Result<()> {
    if replace {
        std::fs::rename(source, destination)
    } else {
        std::fs::hard_link(source, destination)?;
        std::fs::remove_file(source)
    }
}

fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
    }
}

#[derive(Debug)]
pub enum StageMapError {
    InvalidEndpointId,
    CalibrationEndpointRequired,
    UnsupportedSchema(u16),
    DocumentTooLarge,
    Deserialize(String),
    Serialize(String),
    InvalidNextSliceId,
    SliceIdExhausted,
    TooManyEndpoints,
    TooManyEndpointSlices(OutputEndpointId),
    TooManySlices,
    DuplicateEndpoint(OutputEndpointId),
    MissingEndpoint(OutputEndpointId),
    DuplicateOrZeroSliceId(u64),
    InvalidName,
    InvalidMonitorSelector,
    InvalidOutputMode(OutputEndpointId),
    InvalidTestCardRoute(OutputEndpointId),
    InvalidSourceRegion,
    InvalidOutputRegion,
    InvalidPolygonMapping,
    InvalidMask,
    InvalidCalibration,
    Random(String),
    Io(io::Error),
}

impl fmt::Display for StageMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEndpointId => formatter.write_str(
                "endpoint ID must be 1..128 safe ASCII letters, digits, dot, underscore, or dash",
            ),
            Self::CalibrationEndpointRequired => {
                formatter.write_str("calibration output requires one exact physical endpoint")
            }
            Self::UnsupportedSchema(version) => {
                write!(formatter, "unsupported StageMap schema {version}")
            }
            Self::DocumentTooLarge => formatter.write_str("StageMap exceeds its 1 MiB cap"),
            Self::Deserialize(error) => write!(formatter, "invalid StageMap: {error}"),
            Self::Serialize(error) => write!(formatter, "could not serialize StageMap: {error}"),
            Self::InvalidNextSliceId => formatter.write_str("invalid StageMap slice ID cursor"),
            Self::SliceIdExhausted => formatter.write_str("StageMap slice IDs are exhausted"),
            Self::TooManyEndpoints => formatter.write_str("StageMap has too many output endpoints"),
            Self::TooManyEndpointSlices(endpoint) => {
                write!(formatter, "endpoint {endpoint} has too many slices")
            }
            Self::TooManySlices => formatter.write_str("StageMap has too many total slices"),
            Self::DuplicateEndpoint(endpoint) => {
                write!(formatter, "duplicate StageMap endpoint {endpoint}")
            }
            Self::MissingEndpoint(endpoint) => write!(formatter, "missing endpoint {endpoint}"),
            Self::DuplicateOrZeroSliceId(id) => {
                write!(formatter, "duplicate or zero StageMap slice ID {id}")
            }
            Self::InvalidName => formatter.write_str("invalid StageMap display name"),
            Self::InvalidMonitorSelector => formatter.write_str("invalid monitor selector"),
            Self::InvalidOutputMode(endpoint) => {
                write!(formatter, "invalid output mode for endpoint {endpoint}")
            }
            Self::InvalidTestCardRoute(endpoint) => {
                write!(
                    formatter,
                    "endpoint {endpoint} selects an inactive test card"
                )
            }
            Self::InvalidSourceRegion => formatter.write_str("invalid normalized source region"),
            Self::InvalidOutputRegion => formatter.write_str("invalid normalized output quad"),
            Self::InvalidPolygonMapping => formatter.write_str("invalid polygon slice mapping"),
            Self::InvalidMask => formatter.write_str("invalid StageMap mask"),
            Self::InvalidCalibration => formatter.write_str("invalid StageMap calibration"),
            Self::Random(error) => {
                write!(formatter, "could not create temporary filename: {error}")
            }
            Self::Io(error) => write!(formatter, "StageMap I/O failed: {error}"),
        }
    }
}

impl std::error::Error for StageMapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_PATH_SEQUENCE: AtomicU64 = AtomicU64::new(1);

    fn test_stage_map_path(label: &str) -> PathBuf {
        let ordinal = TEST_PATH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "collide-o-scope-stage-map-{label}-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        directory.join(STAGE_MAP_FILE_NAME)
    }

    fn remove_test_stage_map(path: &Path) {
        let _ = fs::remove_file(path);
        if let Some(parent) = path.parent() {
            for entry in fs::read_dir(parent).unwrap() {
                let entry = entry.unwrap();
                assert!(
                    !entry.file_name().to_string_lossy().ends_with(".tmp"),
                    "atomic writer left a temporary file"
                );
                fs::remove_file(entry.path()).unwrap();
            }
            fs::remove_dir(parent).unwrap();
        }
    }

    fn endpoint(id: &str) -> StageEndpoint {
        StageEndpoint::new(
            OutputEndpointId::parse(id).unwrap(),
            format!("Output {id}"),
            [1920, 1080],
        )
    }

    fn map_with_endpoint(id: &str) -> StageMap {
        let mut map = StageMap::default();
        map.add_endpoint(endpoint(id)).unwrap();
        map
    }

    #[test]
    fn endpoint_ids_are_human_readable_but_path_and_control_safe() {
        for valid in ["projector-left", "wall_2", "led.main", "A9"] {
            assert_eq!(OutputEndpointId::parse(valid).unwrap().as_str(), valid);
        }
        for invalid in ["", "../stage", "white space", "slash/output", "bad\n"] {
            assert!(OutputEndpointId::parse(invalid).is_err(), "{invalid:?}");
        }
        let hostile = format!("id:{}", "x".repeat(128));
        assert!(OutputEndpointId::parse(hostile).is_err());
    }

    #[test]
    fn perspective_quad_maps_every_corner_and_rejects_bow_ties() {
        let source = NormalizedRect {
            x: 0.1,
            y: 0.2,
            width: 0.5,
            height: 0.6,
        };
        let output = NormalizedQuad {
            points: [[0.05, 0.1], [0.9, 0.0], [0.8, 0.95], [0.1, 0.8]],
        };
        let homography = solve_homography(output.points, source.corners()).unwrap();
        for (from, expected) in output.points.into_iter().zip(source.corners()) {
            let denominator = homography[6] * from[0] + homography[7] * from[1] + homography[8];
            let actual = [
                (homography[0] * from[0] + homography[1] * from[1] + homography[2]) / denominator,
                (homography[3] * from[0] + homography[4] * from[1] + homography[5]) / denominator,
            ];
            assert!((actual[0] - expected[0]).abs() < 1.0e-4, "{actual:?}");
            assert!((actual[1] - expected[1]).abs() < 1.0e-4, "{actual:?}");
        }

        let bow_tie = NormalizedQuad {
            points: [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        };
        assert!(bow_tie.validate().is_err());
    }

    #[test]
    fn polygon_slices_are_bounded_corresponding_convex_fans() {
        let mut map = map_with_endpoint("polygon");
        let geometry = StageGeometry::Polygon {
            source: vec![[0.0, 0.0], [1.0, 0.0], [0.8, 1.0], [0.2, 1.0]],
            output: vec![[0.0, 0.0], [1.0, 0.0], [0.9, 1.0], [0.1, 1.0]],
        };
        map.add_slice(
            &OutputEndpointId::parse("polygon").unwrap(),
            "Convex slice",
            geometry,
        )
        .unwrap();
        let evaluation = map.evaluate_isolated(StageDeviceLimits::default(), |_| Ok(()));
        let plan = evaluation[0].result.as_ref().unwrap().as_ref().unwrap();
        assert_eq!(plan.slices[0].vertices.len(), 4);
        assert_eq!(plan.slices[0].indices, [0, 1, 2, 0, 2, 3]);

        let invalid = StageGeometry::Polygon {
            source: vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.3], [0.0, 1.0]],
            output: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        assert!(validate_geometry(&invalid).is_err());
    }

    #[test]
    fn endpoint_failure_is_isolated_and_disabled_outputs_allocate_nothing() {
        let mut map = StageMap::default();
        map.add_endpoint(endpoint("working")).unwrap();
        map.add_endpoint(endpoint("missing")).unwrap();
        let mut disabled = endpoint("disabled");
        disabled.enabled = false;
        map.add_endpoint(disabled).unwrap();
        let missing = OutputEndpointId::parse("missing").unwrap();
        let evaluations = map.evaluate_isolated(StageDeviceLimits::default(), |endpoint| {
            if endpoint.id == missing {
                Err("monitor disconnected".to_string())
            } else {
                Ok(())
            }
        });
        assert!(matches!(evaluations[0].result, Ok(Some(_))));
        assert!(matches!(
            evaluations[1].result,
            Err(StageEndpointRuntimeError::Unavailable(_))
        ));
        assert_eq!(evaluations[2].result, Ok(None));
    }

    #[test]
    fn slice_ids_are_monotonic_and_never_reused() {
        let endpoint_id = OutputEndpointId::parse("main").unwrap();
        let mut map = map_with_endpoint("main");
        let first = map
            .add_slice(&endpoint_id, "One", StageGeometry::default())
            .unwrap();
        assert!(map.remove_slice(first));
        let second = map
            .add_slice(&endpoint_id, "Two", StageGeometry::default())
            .unwrap();
        assert_eq!(first.get(), 1);
        assert_eq!(second.get(), 2);

        let mut yaml = String::from_utf8(map.to_yaml_bytes().unwrap()).unwrap();
        yaml = yaml.replace("next_slice_id: 3", "next_slice_id: 1");
        let repaired = StageMap::from_yaml_bytes(yaml.as_bytes()).unwrap();
        assert_eq!(repaired.next_slice_id, 3);
    }

    #[test]
    fn stage_tools_can_only_paint_the_exact_selected_physical_endpoint() {
        let selected = OutputEndpointId::parse("projector-a").unwrap();
        let other = OutputEndpointId::parse("projector-b").unwrap();
        let mut tools = StageToolState::default();
        assert!(tools.set_test_card(TestCardMode::Grid, None).is_err());
        tools
            .set_test_card(TestCardMode::SmpteBars, Some(selected.clone()))
            .unwrap();
        tools
            .set_output_identification(true, Some(selected.clone()))
            .unwrap();
        assert_eq!(
            tools.decision_for(&StageSurface::PhysicalOutput(selected)),
            StageCalibrationDecision {
                substitute_with_test_card: true,
                overlay_output_identification: true,
            }
        );
        for surface in [
            StageSurface::PhysicalOutput(other),
            StageSurface::EditorPreview,
            StageSurface::Composite,
            StageSurface::Audience,
            StageSurface::Spout,
            StageSurface::Record,
            StageSurface::Export,
        ] {
            assert_eq!(
                tools.decision_for(&surface),
                StageCalibrationDecision::default(),
                "{surface:?}"
            );
        }
    }

    #[test]
    fn yaml_round_trip_preserves_venue_state_and_rejects_hostile_values() {
        let endpoint_id = OutputEndpointId::parse("front-wall").unwrap();
        let mut map = map_with_endpoint("front-wall");
        map.endpoints[0].binding = OutputBinding::Monitor {
            selector: "DisplayPort Projector".to_string(),
        };
        let slice = map
            .add_slice(&endpoint_id, "Main crop", StageGeometry::default())
            .unwrap();
        map.endpoints[0].slices[0].mask = StageMask::EdgeFeather {
            softness: [0.02, 0.0, 0.03, 0.0],
        };
        assert_eq!(map.endpoints[0].slices[0].id, slice);
        let bytes = map.to_yaml_bytes().unwrap();
        assert_eq!(StageMap::from_yaml_bytes(&bytes).unwrap(), map);

        let mut hostile = bytes;
        hostile.extend(std::iter::repeat_n(b' ', MAX_STAGE_MAP_BYTES));
        assert!(matches!(
            StageMap::from_yaml_bytes(&hostile),
            Err(StageMapError::DocumentTooLarge)
        ));
    }

    #[test]
    fn stage_map_persistence_defaults_safely_and_replaces_atomically() {
        let path = test_stage_map_path("atomic");
        fs::remove_dir(path.parent().unwrap()).unwrap();
        let missing = load_stage_map_or_default(&path);
        assert_eq!(missing.path, path);
        assert_eq!(missing.document, StageMap::default());
        assert_eq!(missing.status, StageMapLoadStatus::DefaultMissing);

        let first = map_with_endpoint("front-wall");
        save_stage_map_atomic(&first, &path).unwrap();
        let loaded = load_stage_map_or_default(&path);
        assert_eq!(loaded.document, first);
        assert_eq!(loaded.status, StageMapLoadStatus::Loaded);

        let mut replacement = first.clone();
        replacement.add_endpoint(endpoint("projector-b")).unwrap();
        save_stage_map_atomic(&replacement, &path).unwrap();
        assert_eq!(
            StageMap::from_yaml_bytes(&fs::read(&path).unwrap()).unwrap(),
            replacement
        );

        let published = fs::read(&path).unwrap();
        let mut invalid = replacement;
        invalid.schema_version = STAGE_MAP_SCHEMA_VERSION + 1;
        assert!(matches!(
            save_stage_map_atomic(&invalid, &path),
            Err(StageMapError::UnsupportedSchema(_))
        ));
        assert_eq!(fs::read(&path).unwrap(), published);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        remove_test_stage_map(&path);
    }

    #[test]
    fn stage_map_load_rejects_hostile_files_without_rewriting_them() {
        let path = test_stage_map_path("hostile");
        fs::write(&path, vec![b'x'; MAX_STAGE_MAP_BYTES + 1]).unwrap();
        let oversized_bytes = fs::read(&path).unwrap();
        let oversized = load_stage_map_or_default(&path);
        assert_eq!(oversized.document, StageMap::default());
        assert!(matches!(
            oversized.status,
            StageMapLoadStatus::DefaultInvalid(_)
        ));
        assert_eq!(fs::read(&path).unwrap(), oversized_bytes);

        fs::write(&path, [0xff, 0xfe, 0xfd]).unwrap();
        let invalid_bytes = fs::read(&path).unwrap();
        let invalid = load_stage_map_or_default(&path);
        assert_eq!(invalid.document, StageMap::default());
        assert!(matches!(
            invalid.status,
            StageMapLoadStatus::DefaultInvalid(_)
        ));
        assert_eq!(fs::read(&path).unwrap(), invalid_bytes);

        fs::write(
            &path,
            "schema_version: 1\nnext_slice_id: 1\nendpoints: []\nunexpected: true\n",
        )
        .unwrap();
        let unknown_bytes = fs::read(&path).unwrap();
        let unknown = load_stage_map_or_default(&path);
        assert_eq!(unknown.document, StageMap::default());
        assert!(matches!(
            unknown.status,
            StageMapLoadStatus::DefaultInvalid(_)
        ));
        assert_eq!(fs::read(&path).unwrap(), unknown_bytes);
        remove_test_stage_map(&path);
    }
}
