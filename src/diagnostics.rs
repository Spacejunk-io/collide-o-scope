//! Stable additive diagnostics and side-effect-free remediation descriptions.
//!
//! Planners own the mapping from their typed errors into these protocol types.
//! Presentation may use `text`, but clients must branch on `code` and must
//! never derive a repair by parsing prose.

use serde::{Deserialize, Serialize};

pub const CONSTRAINT_DIAGNOSTIC_SCHEMA: &str = "collide-o-scope-constraint-diagnostic/1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintCode {
    StableIdentityMismatch,
    DuplicateStableIdentity,
    MissingStableIdentity,
    StudyBudgetExceeded,
    ScanVertexLimitExceeded,
    FilterAvalancheLimitExceeded,
    RackInvalid,
    RouteInvalid,
    RouteCycle,
    MasterBypassOrderViolation,
    TemporalBypassOrderViolation,
    TemporalBypassVhsConflict,
    GardenBypassConflict,
    MotionRouteInvalid,
    ResourceLimitExceeded,
    GpuPlanRejected,
    SelectiveMatteTopologyUnsupported,
    RevisionMismatch,
    RemediationUnavailable,
    PreparedTransitionLimitExceeded,
    MoshDomainLimitExceeded,
    InternalPlannerError,
}

impl ConstraintCode {
    pub const ALL: [Self; 22] = [
        Self::StableIdentityMismatch,
        Self::DuplicateStableIdentity,
        Self::MissingStableIdentity,
        Self::StudyBudgetExceeded,
        Self::ScanVertexLimitExceeded,
        Self::FilterAvalancheLimitExceeded,
        Self::RackInvalid,
        Self::RouteInvalid,
        Self::RouteCycle,
        Self::MasterBypassOrderViolation,
        Self::TemporalBypassOrderViolation,
        Self::TemporalBypassVhsConflict,
        Self::GardenBypassConflict,
        Self::MotionRouteInvalid,
        Self::ResourceLimitExceeded,
        Self::GpuPlanRejected,
        Self::SelectiveMatteTopologyUnsupported,
        Self::RevisionMismatch,
        Self::RemediationUnavailable,
        Self::PreparedTransitionLimitExceeded,
        Self::MoshDomainLimitExceeded,
        Self::InternalPlannerError,
    ];

    pub const fn help_key(self) -> &'static str {
        match self {
            Self::StableIdentityMismatch => "constraints/stable-identity-mismatch",
            Self::DuplicateStableIdentity => "constraints/duplicate-stable-identity",
            Self::MissingStableIdentity => "constraints/missing-stable-identity",
            Self::StudyBudgetExceeded => "constraints/study-budget",
            Self::ScanVertexLimitExceeded => "constraints/scan-vertex-limit",
            Self::FilterAvalancheLimitExceeded => "constraints/filter-avalanche-limit",
            Self::RackInvalid => "constraints/rack-invalid",
            Self::RouteInvalid => "constraints/route-invalid",
            Self::RouteCycle => "constraints/route-cycle",
            Self::MasterBypassOrderViolation => "constraints/master-bypass-order",
            Self::TemporalBypassOrderViolation => "constraints/temporal-bypass-order",
            Self::TemporalBypassVhsConflict => "constraints/temporal-bypass-vhs",
            Self::GardenBypassConflict => "constraints/garden-bypass",
            Self::MotionRouteInvalid => "constraints/motion-route",
            Self::ResourceLimitExceeded => "constraints/resource-limit",
            Self::GpuPlanRejected => "constraints/gpu-plan",
            Self::SelectiveMatteTopologyUnsupported => "constraints/selective-matte-topology",
            Self::RevisionMismatch => "constraints/revision-mismatch",
            Self::RemediationUnavailable => "constraints/remediation-unavailable",
            Self::PreparedTransitionLimitExceeded => "constraints/prepared-transition-limit",
            Self::MoshDomainLimitExceeded => "constraints/mosh-domain-limit",
            Self::InternalPlannerError => "constraints/internal-planner-error",
        }
    }

    pub fn from_help_key(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|code| code.help_key() == value)
    }

    pub const fn title(self) -> &'static str {
        match self {
            Self::StableIdentityMismatch => "Stable identity mismatch",
            Self::DuplicateStableIdentity => "Duplicate stable identity",
            Self::MissingStableIdentity => "Missing stable identity",
            Self::StudyBudgetExceeded => "Study budget exceeded",
            Self::ScanVertexLimitExceeded => "Scan geometry budget exceeded",
            Self::FilterAvalancheLimitExceeded => "Filter Avalanche history bound exceeded",
            Self::RackInvalid => "Collision Rack is invalid",
            Self::RouteInvalid => "Image route is invalid",
            Self::RouteCycle => "Current-frame route cycle",
            Self::MasterBypassOrderViolation => "Master bypass order conflict",
            Self::TemporalBypassOrderViolation => "Temporal bypass order conflict",
            Self::TemporalBypassVhsConflict => "Temporal bypass conflicts with VHS",
            Self::GardenBypassConflict => "Refresh Garden route conflicts with bypass",
            Self::MotionRouteInvalid => "Motion route is inadmissible",
            Self::ResourceLimitExceeded => "Creative resource limit exceeded",
            Self::GpuPlanRejected => "GPU plan does not match accepted plan",
            Self::SelectiveMatteTopologyUnsupported => "Selective matte topology unsupported",
            Self::RevisionMismatch => "Authored revision is stale",
            Self::RemediationUnavailable => "Remediation preview is unavailable",
            Self::PreparedTransitionLimitExceeded => "Prepared transition bound exceeded",
            Self::MoshDomainLimitExceeded => "Mosh-domain bound exceeded",
            Self::InternalPlannerError => "Internal planner contract failed",
        }
    }

    pub const fn operator_explanation(self) -> &'static str {
        match self {
            Self::StableIdentityMismatch
            | Self::DuplicateStableIdentity
            | Self::MissingStableIdentity => {
                "Stable IDs prevent edits and routes from silently retargeting after stack or topology changes. Refresh state, then repeat the edit against the current identities."
            }
            Self::StudyBudgetExceeded => {
                "The resolved Study performs more admitted per-pixel work than its declared ceiling. Reduce texture-loading instructions; the engine will not silently clamp or rewrite the program."
            }
            Self::ScanVertexLimitExceeded => {
                "The requested Scan Processor geometry exceeds the bounded ribbon-vertex budget. Reduce the authored geometry before retrying."
            }
            Self::FilterAvalancheLimitExceeded => {
                "Each Filter Avalanche can retain full-frame history. Remove or disable enough instances to fit the closed history count."
            }
            Self::RackInvalid => {
                "A rack violates its stable node, marker, ordering, or parameter contract. The rejected transaction has not changed the live rack."
            }
            Self::RouteInvalid => {
                "The requested producer/consumer relation is unavailable or illegal at the selected timing. Choose an admitted stable producer or a previous-frame route."
            }
            Self::RouteCycle => {
                "Current-frame image dependencies must be acyclic. Retarget at least one participating edge to an earlier producer or to previous-frame timing."
            }
            Self::MasterBypassOrderViolation => {
                "Master bypass must preserve one exact dry/wet ordering. The engine refuses ambiguous partitions instead of changing blend order."
            }
            Self::TemporalBypassOrderViolation => {
                "Temporal-dry layers must form the exact top contiguous prefix so they can be restored after the shared Temporal family without recompositing the programme."
            }
            Self::TemporalBypassVhsConflict => {
                "The bounded audience path cannot restore a Temporal-dry overlay and then run final-program VHS without another CPU/GPU hop. Disable one side explicitly."
            }
            Self::GardenBypassConflict => {
                "A routed Refresh Garden gate crosses the isolated Temporal partition. Use the inline gate or disable the partition."
            }
            Self::MotionRouteInvalid => {
                "The authored Motion donor or recipient cannot be admitted under the current scope and topology. Retarget or remove the named route; zero depth remains authored topology."
            }
            Self::ResourceLimitExceeded => {
                "The immutable candidate plan exceeds a named host/device or creative budget. The engine does not silently reduce quality, raster, history, or topology."
            }
            Self::GpuPlanRejected => {
                "Executor reconciliation disagrees with the already accepted immutable plan. Programme output stays on the last safe world and the mismatch must be treated as an engine fault."
            }
            Self::SelectiveMatteTopologyUnsupported => {
                "The requested dry partition crosses a topology that cannot be decomposed exactly. Disable the partition or remove the named coupling."
            }
            Self::RevisionMismatch => {
                "Another controller changed authored state first. Refresh the current revision and deliberately reapply the intended edit."
            }
            Self::RemediationUnavailable => {
                "A remediation can be applied only while the exact planner-authored preview remains current. Refresh state and review a newly offered preview; controllers cannot invent or widen repairs."
            }
            Self::PreparedTransitionLimitExceeded => {
                "The process-wide prepared-transition reservation is full. Finish or cancel an existing transition, or author a Cut."
            }
            Self::MoshDomainLimitExceeded => {
                "At most two independent codec-history domains may be admitted. Merge or remove a domain before retrying."
            }
            Self::InternalPlannerError => {
                "A closed planner invariant failed. Keep Programme on the last safe frame, preserve the receipt, and restart only through the documented recovery path."
            }
        }
    }
}

/// Generate one same-origin help page from the closed diagnostic registry.
/// `help_key` is accepted only when it exactly names a known code, so no path
/// or user-authored text can enter the response.
pub fn operator_help_html(help_key: &str) -> Option<String> {
    let code = ConstraintCode::from_help_key(help_key)?;
    Some(format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{}</title><style>body{{font:16px/1.5 system-ui,sans-serif;max-width:52rem;margin:3rem auto;padding:0 1rem;background:#101216;color:#f4f4f5}}code{{color:#8ee8ff}}a{{color:#8ee8ff}}</style></head><body><main><p><a href=\"/#creative-panel\">← Control panel</a></p><h1>{}</h1><p>{}</p><p>Protocol key: <code>{}</code></p></main></body></html>",
        code.title(),
        code.title(),
        code.operator_explanation(),
        code.help_key(),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintInvariant {
    StableIdentityPreserved,
    StableIdentitiesUnique,
    ReferencedIdentityExists,
    StudyBudgetBounded,
    ScanGeometryBounded,
    FilterAvalancheBounded,
    RackTopologyValid,
    RouteTopologyValid,
    RouteGraphAcyclic,
    DryBypassIsCanonicalPrefix,
    TemporalBypassCompatibleWithVhs,
    GardenBypassHasInlineGate,
    MotionRouteAdmissible,
    ResourceLedgerWithinCap,
    GpuPlanMatchesAcceptedPlan,
    SelectiveMatteTopologyAdmissible,
    AuthoredRevisionCurrent,
    RemediationPreviewCurrent,
    PreparedTransitionConcurrencyBounded,
    MoshDomainCountBounded,
    PlannerContractHeld,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintSeverity {
    Info,
    Warning,
    Error,
    Fatal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintScopeKind {
    Global,
    Program,
    Master,
    Layer,
    Group,
    Node,
    Route,
    Scene,
    ClipSlot,
    Output,
    MoshDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConstraintScope {
    pub kind: ConstraintScopeKind,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_stable_id",
        deserialize_with = "deserialize_optional_stable_id"
    )]
    pub stable_id: Option<u64>,
}

fn serialize_optional_stable_id<S>(value: &Option<u64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        Some(value) => serializer.serialize_some(&value.to_string()),
        None => serializer.serialize_none(),
    }
}

fn deserialize_optional_stable_id<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StableIdWire {
        String(String),
        Number(u64),
    }

    let value = Option::<StableIdWire>::deserialize(deserializer)?;
    let value = value
        .map(|value| match value {
            StableIdWire::String(value) => value.parse::<u64>().map_err(serde::de::Error::custom),
            StableIdWire::Number(value) => Ok(value),
        })
        .transpose()?;
    if value == Some(0) {
        return Err(serde::de::Error::custom("stable ID must be non-zero"));
    }
    Ok(value)
}

impl ConstraintScope {
    pub const fn singleton(kind: ConstraintScopeKind) -> Self {
        Self {
            kind,
            stable_id: None,
        }
    }

    pub const fn stable(kind: ConstraintScopeKind, stable_id: u64) -> Self {
        Self {
            kind,
            stable_id: Some(stable_id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum DiagnosticValue {
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    Number(f64),
    Text(String),
    StableScopes(Vec<ConstraintScope>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    GpuBytes,
    HostTransferBytes,
    RetainedSurfaces,
    TextureBindings,
    ShaderOperations,
    StudyInstructions,
    ScanVertices,
    PreparedTransitions,
    MoshDomains,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceDelta {
    pub resource: ResourceKind,
    pub currently_used: u64,
    pub requested_additional: u64,
    pub resulting_total: u64,
    pub limit: u64,
}

impl ResourceDelta {
    pub fn checked(
        resource: ResourceKind,
        currently_used: u64,
        requested_additional: u64,
        limit: u64,
    ) -> Option<Self> {
        let resulting_total = currently_used.checked_add(requested_additional)?;
        Some(Self {
            resource,
            currently_used,
            requested_additional,
            resulting_total,
            limit,
        })
    }

    pub const fn exceeds_limit(&self) -> bool {
        self.resulting_total > self.limit
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RemediationCandidateId(pub u64);

impl Serialize for RemediationCandidateId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for RemediationCandidateId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            String(String),
            Number(u64),
        }

        let value = match Wire::deserialize(deserializer)? {
            Wire::String(value) => value.parse::<u64>().map_err(serde::de::Error::custom)?,
            Wire::Number(value) => value,
        };
        if value == 0 {
            return Err(serde::de::Error::custom(
                "remediation candidate ID must be non-zero",
            ));
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemediationCode {
    ReorderDryPrefix,
    DisableConflictingBypass,
    RetargetMotionRoute,
    RemoveMotionRoute,
}

/// Names the authored bypass bit a remediation may change. Keeping this in
/// the protocol prevents a controller from guessing whether the planner meant
/// the Master prefix or the independently ordered Temporal family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BypassKind {
    Master,
    Temporal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum RemediationOperation {
    ReorderScopes {
        ordered: Vec<ConstraintScope>,
    },
    SetBypass {
        scope: ConstraintScope,
        bypass: BypassKind,
        enabled: bool,
    },
    RetargetRoute {
        route: ConstraintScope,
        target: ConstraintScope,
    },
    RemoveRoute {
        route: ConstraintScope,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemediationCandidate {
    pub id: RemediationCandidateId,
    pub code: RemediationCode,
    pub base_revision: u64,
    pub description: String,
    pub operations: Vec<RemediationOperation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintDiagnostic {
    pub schema: String,
    pub code: ConstraintCode,
    pub invariant: ConstraintInvariant,
    pub severity: ConstraintSeverity,
    pub affected: Vec<ConstraintScope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<DiagnosticValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<DiagnosticValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_delta: Option<ResourceDelta>,
    pub text: String,
    pub help_key: String,
    /// Stable, same-origin operator-help route derived only from `code`.
    pub help_url: String,
    pub remediations: Vec<RemediationCandidate>,
    /// Planner-evaluated immutable candidate worlds. A candidate is not
    /// actionable unless a preview with the same ID and base revision exists.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remediation_previews: Vec<RemediationPreview>,
}

impl ConstraintDiagnostic {
    pub fn new(
        code: ConstraintCode,
        invariant: ConstraintInvariant,
        severity: ConstraintSeverity,
        affected: Vec<ConstraintScope>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            schema: CONSTRAINT_DIAGNOSTIC_SCHEMA.to_owned(),
            code,
            invariant,
            severity,
            affected,
            expected: None,
            actual: None,
            resource_delta: None,
            text: text.into(),
            help_key: code.help_key().to_owned(),
            help_url: format!("/help/{}", code.help_key()),
            remediations: Vec::new(),
            remediation_previews: Vec::new(),
        }
    }

    pub fn with_values(mut self, expected: DiagnosticValue, actual: DiagnosticValue) -> Self {
        self.expected = Some(expected);
        self.actual = Some(actual);
        self
    }

    pub fn with_resource_delta(mut self, resource_delta: ResourceDelta) -> Self {
        self.resource_delta = Some(resource_delta);
        self
    }

    pub fn with_remediations(mut self, remediations: Vec<RemediationCandidate>) -> Self {
        self.remediations = remediations;
        self
    }

    pub fn with_remediation_previews(mut self, previews: Vec<RemediationPreview>) -> Self {
        self.remediation_previews = previews;
        self
    }
}

/// Error wrapper used at transaction boundaries so `?` cannot erase the
/// planner-owned protocol identity back into a bare string.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintFailure {
    pub diagnostic: Box<ConstraintDiagnostic>,
}

impl ConstraintFailure {
    pub fn new(diagnostic: ConstraintDiagnostic) -> Self {
        Self {
            diagnostic: Box::new(diagnostic),
        }
    }

    pub fn into_diagnostic(self) -> ConstraintDiagnostic {
        *self.diagnostic
    }
}

impl std::fmt::Display for ConstraintFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.diagnostic.text)
    }
}

impl std::error::Error for ConstraintFailure {}

impl From<ConstraintFailure> for String {
    fn from(failure: ConstraintFailure) -> Self {
        failure.into_diagnostic().text
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemediationConsequence {
    pub affected: Vec<ConstraintScope>,
    pub plan: RemediationPlanConsequence,
    pub pixel_order_before: Vec<String>,
    pub pixel_order_after: Vec<String>,
    pub resource_deltas: Vec<ResourceDelta>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemediationPlanKind {
    LegacyExact,
    Advanced,
}

/// Closed, bounded planner facts returned by immutable remediation preview.
/// These are consequences of the candidate world, not controller estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemediationPlanConsequence {
    pub kind: RemediationPlanKind,
    pub topology_signature: u64,
    pub full_frame_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub retained_surface_layers: u32,
    pub creative_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemediationPreview {
    pub candidate_id: RemediationCandidateId,
    pub base_revision: u64,
    pub operations: Vec<RemediationOperation>,
    pub consequence: RemediationConsequence,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_keeps_protocol_identity_separate_from_text() {
        let layer = ConstraintScope::stable(ConstraintScopeKind::Layer, 41);
        let delta = ResourceDelta::checked(ResourceKind::GpuBytes, 900, 125, 1_000).unwrap();
        let diagnostic = ConstraintDiagnostic::new(
            ConstraintCode::ResourceLimitExceeded,
            ConstraintInvariant::ResourceLedgerWithinCap,
            ConstraintSeverity::Error,
            vec![layer.clone()],
            "friendly text may change without changing protocol identity",
        )
        .with_values(
            DiagnosticValue::Unsigned(1_000),
            DiagnosticValue::Unsigned(1_025),
        )
        .with_resource_delta(delta.clone());

        assert!(delta.exceeds_limit());
        assert_eq!(diagnostic.help_key, "constraints/resource-limit");
        assert_eq!(diagnostic.help_url, "/help/constraints/resource-limit");
        let encoded = serde_json::to_value(&diagnostic).unwrap();
        assert_eq!(encoded["code"], "resource_limit_exceeded");
        assert_eq!(encoded["affected"][0]["stable_id"], "41");
        assert_eq!(encoded["resource_delta"]["resulting_total"], 1_025);
        assert_eq!(encoded["text"], diagnostic.text);
    }

    #[test]
    fn remediation_preview_is_declarative_and_does_not_mutate_the_candidate() {
        let original_order = vec![
            ConstraintScope::stable(ConstraintScopeKind::Layer, 9),
            ConstraintScope::stable(ConstraintScopeKind::Layer, 3),
        ];
        let candidate = RemediationCandidate {
            id: RemediationCandidateId(7),
            code: RemediationCode::ReorderDryPrefix,
            base_revision: 12,
            description: "move the dry layer into the canonical prefix".to_owned(),
            operations: vec![RemediationOperation::ReorderScopes {
                ordered: original_order.iter().rev().cloned().collect(),
            }],
        };
        let before = candidate.clone();
        let preview = RemediationPreview {
            candidate_id: candidate.id,
            base_revision: candidate.base_revision,
            operations: candidate.operations.clone(),
            consequence: RemediationConsequence {
                affected: original_order,
                plan: RemediationPlanConsequence {
                    kind: RemediationPlanKind::Advanced,
                    topology_signature: 91,
                    full_frame_passes: 2,
                    logical_texture_lookups_per_pixel: 4,
                    retained_surface_layers: 3,
                    creative_bytes: 1_024,
                },
                pixel_order_before: vec!["wet".to_owned(), "dry".to_owned()],
                pixel_order_after: vec!["dry".to_owned(), "wet".to_owned()],
                resource_deltas: Vec::new(),
            },
        };
        assert_eq!(
            candidate, before,
            "preview construction changed the candidate"
        );
        assert_eq!(preview.base_revision, 12);
        let encoded = serde_json::to_value(&candidate).unwrap();
        assert_eq!(encoded["id"], "7");
        assert_eq!(
            serde_json::from_value::<RemediationCandidate>(encoded).unwrap(),
            candidate
        );
    }

    #[test]
    fn protocol_ids_reject_zero_and_preserve_values_beyond_javascript_precision() {
        let id = RemediationCandidateId(9_007_199_254_740_993);
        let encoded = serde_json::to_string(&id).unwrap();
        assert_eq!(encoded, "\"9007199254740993\"");
        assert_eq!(
            serde_json::from_str::<RemediationCandidateId>(&encoded).unwrap(),
            id
        );
        assert!(serde_json::from_str::<RemediationCandidateId>("0").is_err());
        assert!(
            serde_json::from_str::<ConstraintScope>(r#"{"kind":"layer","stable_id":"0"}"#).is_err()
        );
    }

    #[test]
    fn resource_delta_fails_closed_on_overflow() {
        assert!(ResourceDelta::checked(ResourceKind::GpuBytes, u64::MAX, 1, u64::MAX).is_none());
    }

    #[test]
    fn every_closed_code_has_one_generated_same_origin_help_page() {
        let mut keys = std::collections::BTreeSet::new();
        for code in ConstraintCode::ALL {
            assert!(keys.insert(code.help_key()), "duplicate help key");
            assert_eq!(ConstraintCode::from_help_key(code.help_key()), Some(code));
            let page = operator_help_html(code.help_key()).expect("known help code");
            assert!(page.contains(code.title()));
            assert!(page.contains(code.help_key()));
            assert!(!page.contains("C:\\"));
            assert!(!page.contains("/Users/"));
        }
        assert!(operator_help_html("constraints/not-a-code").is_none());
        assert!(operator_help_html("../index.html").is_none());
    }
}
