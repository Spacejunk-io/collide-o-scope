// collide-o-scope — web control panel

const statusEl = document.getElementById('ws-status');
const layersList = document.getElementById('layers-list');
const layersEmpty = document.getElementById('layers-empty');
const libraryGrid = document.getElementById('library-grid');
const spoutInForm = document.getElementById('spout-in-form');
const spoutInName = document.getElementById('spout-in-name');
const spoutInStatus = document.getElementById('spout-in-status');
const expertMediaToggle = document.getElementById('expert-media-toggle');
const mediaSafetyMode = document.getElementById('media-safety-mode');
const mediaSafetySummary = document.getElementById('media-safety-summary');
const mediaSafetyRationale = document.getElementById('media-safety-rationale');
const mediaSafetyStatus = document.getElementById('media-safety-status');
const newLayerFit = document.getElementById('new-layer-fit');
const proxyScale = document.getElementById('proxy-scale');
const proxyFrameRate = document.getElementById('proxy-frame-rate');
const proxyIncludeAudio = document.getElementById('proxy-include-audio');
const librarySlotTarget = document.getElementById('library-slot-target');
const librarySlotTrigger = document.getElementById('library-slot-trigger');
const slotLoadStatus = document.getElementById('slot-load-status');
const sceneList = document.getElementById('scene-list');
const sceneStatus = document.getElementById('scene-status');
const sceneCaptureForm = document.getElementById('scene-capture-form');
const sceneCaptureName = document.getElementById('scene-capture-name');
const sceneCaptureMode = document.getElementById('scene-capture-mode');
const autopilotPlanForm = document.getElementById('autopilot-plan-form');
const autopilotRepeat = document.getElementById('autopilot-repeat');
const autopilotStepList = document.getElementById('autopilot-step-list');
const autopilotAddStep = document.getElementById('autopilot-add-step');
const autopilotPlay = document.getElementById('autopilot-play');
const autopilotPause = document.getElementById('autopilot-pause');
const autopilotReset = document.getElementById('autopilot-reset');
const autopilotPhase = document.getElementById('autopilot-phase');
const autopilotCurrent = document.getElementById('autopilot-current');
const autopilotNext = document.getElementById('autopilot-next');
const autopilotBeats = document.getElementById('autopilot-beats');
const autopilotStatus = document.getElementById('autopilot-status');
let authoritativeMediaSafetyMode = 'safe';
let authoritativeNewLayerFit = 'fit';
// Mirrors the engine's host-session ProxySettings tuple. `rateKey` is the
// select's value vocabulary: 'source', a preset numerator, or 'fixed:N/D'
// for a tuple another controller authored outside the presets.
let authoritativeProxySettings = { scale: 'half', rateKey: 'source', includeAudio: true };

const LAYER_BLEND_MODES = Object.freeze([
  { key: 'normal', label: 'Normal', description: 'Normal composites the layer over the content below.' },
  { key: 'screen', label: 'Screen', description: 'Screen brightens by combining inverse color values.' },
  { key: 'multiply', label: 'Multiply', description: 'Multiply darkens by multiplying color values.' },
  { key: 'difference', label: 'Difference', description: 'Difference shows the absolute color distance between layer and below.' },
  { key: 'add', label: 'Add', description: 'Add sums layer and below for luminous accumulation.' },
  { key: 'subtract', label: 'Subtract', description: 'Subtract computes below minus layer.' },
  { key: 'darken', label: 'Darken', description: 'Darken keeps the lower value in each color channel.' },
  { key: 'lighten', label: 'Lighten', description: 'Lighten keeps the higher value in each color channel.' },
  { key: 'overlay', label: 'Overlay', description: 'Overlay combines Multiply and Screen according to the content below.' },
  { key: 'soft_light', label: 'Soft Light', description: 'Soft Light applies a restrained contrast and illumination response.' },
  { key: 'hard_light', label: 'Hard Light', description: 'Hard Light combines Multiply and Screen according to the layer.' },
  { key: 'exclusion', label: 'Exclusion', description: 'Exclusion creates a lower-contrast difference relation.' },
  { key: 'dodge', label: 'Dodge', description: 'Dodge brightens the content below toward the layer color.' },
  { key: 'burn', label: 'Burn', description: 'Burn darkens the content below toward the layer color.' },
  { key: 'alpha_cut', label: 'Alpha Cut', description: 'Alpha Cut erases accumulated content; it is a no-op without content below.' },
  { key: 'vivid_light', label: 'Vivid Light', description: 'Vivid Light burns below half layer values and dodges above them.' },
  { key: 'pin_light', label: 'Pin Light', description: 'Pin Light replaces content below only outside the doubled layer bounds.' },
  { key: 'divide', label: 'Divide', description: 'Divide brightens by dividing the content below by the layer color.' },
  { key: 'wrap_add', label: 'Wrap Add', description: 'Wrap Add sums with analogue overflow, wrapping past full scale.' },
  { key: 'xor', label: 'Xor Bits', description: 'Xor Bits combines 8-bit code values with exclusive-or.' },
  { key: 'and', label: 'And Bits', description: 'And Bits combines 8-bit code values with bitwise and.' },
  { key: 'hue', label: 'Hue', description: 'Hue takes the layer hue while keeping saturation and value below.' },
  { key: 'saturation', label: 'Saturation', description: 'Saturation takes the layer saturation while keeping hue and value below.' },
  { key: 'color', label: 'Color', description: 'Color takes the layer hue and saturation while keeping the value below.' },
  { key: 'luminosity', label: 'Luminosity', description: 'Luminosity takes the layer value while keeping hue and saturation below.' },
]);
const LAYER_BLEND_ORDER_POLICY = 'Reordering changes the content below; the saved blend choice remains unchanged.';

// The one-time query key bootstraps the HttpOnly Strict cookie. Remove only
// that secret from browser history once the document has received its cookie;
// unrelated query parameters and the fragment remain intact.
function stripBootstrapKeyFromUrl() {
  const url = new URL(window.location.href);
  if (!url.searchParams.has('key')) return;
  url.searchParams.delete('key');
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', stripBootstrapKeyFromUrl, { once: true });
} else {
  stripBootstrapKeyFromUrl();
}

spoutInForm?.addEventListener('submit', (event) => {
  event.preventDefault();
  const sender = spoutInName.value.replace(/[\u0000-\u001f\u007f]/g, '').trim();
  if (!sender) {
    spoutInStatus.textContent = 'Enter the exact Spout sender name.';
    return;
  }
  if (sendAction({ action: 'add_spout_layer', sender })) {
    spoutInStatus.textContent = `Opening ${sender}\u2026`;
  } else {
    spoutInStatus.textContent = 'Control connection is offline; try again when connected.';
  }
});

const generatorAddStatus = document.getElementById('generator-add-status');
document.getElementById('add-pattern-layer')?.addEventListener('click', () => {
  if (generatorAddStatus) {
    generatorAddStatus.textContent = sendAction({ action: 'add_pattern_layer' })
      ? 'Adding pattern synth layer…'
      : 'Control connection is offline; try again when connected.';
  } else {
    sendAction({ action: 'add_pattern_layer' });
  }
});
document.getElementById('add-text-layer')?.addEventListener('click', () => {
  if (generatorAddStatus) {
    generatorAddStatus.textContent = sendAction({ action: 'add_text_layer' })
      ? 'Adding text page layer…'
      : 'Control connection is offline; try again when connected.';
  } else {
    sendAction({ action: 'add_text_layer' });
  }
});

newLayerFit?.addEventListener('change', () => {
  const fit = ['stretch', 'fit', 'fill', 'native'].includes(newLayerFit.value)
    ? newLayerFit.value
    : authoritativeNewLayerFit;
  if (!sendAction({ action: 'set_new_layer_fit', fit })) {
    newLayerFit.value = authoritativeNewLayerFit;
  }
});

const PROXY_FRAME_RATE_PRESETS = new Set(['24', '30', '60']);

// A fixed rate outside the preset list (authored by another controller) is
// represented honestly by one dynamic option instead of being misreported as
// the nearest preset.
function setProxyFrameRateSelect(rateKey) {
  if (!proxyFrameRate) return;
  const dynamic = proxyFrameRate.querySelector('option[data-authored-elsewhere]');
  const isPreset = rateKey === 'source' || PROXY_FRAME_RATE_PRESETS.has(rateKey);
  if (isPreset) {
    if (dynamic) dynamic.remove();
    proxyFrameRate.value = rateKey;
    return;
  }
  const match = /^fixed:(\d+)\/(\d+)$/.exec(rateKey);
  if (!match) {
    if (dynamic) dynamic.remove();
    proxyFrameRate.value = 'source';
    return;
  }
  const option = dynamic || document.createElement('option');
  option.value = rateKey;
  option.textContent = `Fixed ${match[1]}/${match[2]} fps`;
  option.setAttribute('data-authored-elsewhere', '');
  if (!dynamic) proxyFrameRate.appendChild(option);
  proxyFrameRate.value = rateKey;
}

function resetProxySettingsControls() {
  if (proxyScale) proxyScale.value = authoritativeProxySettings.scale;
  setProxyFrameRateSelect(authoritativeProxySettings.rateKey);
  if (proxyIncludeAudio) proxyIncludeAudio.checked = authoritativeProxySettings.includeAudio;
}

// Every edit carries the complete absolute tuple, so one control's change can
// never silently reset another and coalescing keeps only the newest tuple.
function sendProxySettings() {
  if (!proxyScale || !proxyFrameRate || !proxyIncludeAudio) return;
  const scale = ['original', 'half', 'quarter'].includes(proxyScale.value)
    ? proxyScale.value
    : authoritativeProxySettings.scale;
  const rateKey = proxyFrameRate.value;
  let frameRate = 'source';
  if (PROXY_FRAME_RATE_PRESETS.has(rateKey)) {
    frameRate = { fixed: { numerator: Number(rateKey), denominator: 1 } };
  } else if (rateKey !== 'source') {
    const match = /^fixed:(\d+)\/(\d+)$/.exec(rateKey);
    if (match) {
      frameRate = { fixed: { numerator: Number(match[1]), denominator: Number(match[2]) } };
    }
  }
  const sent = sendAction({
    action: 'set_proxy_settings',
    scale,
    frame_rate: frameRate,
    include_audio: proxyIncludeAudio.checked,
  });
  if (!sent) resetProxySettingsControls();
}

proxyScale?.addEventListener('change', sendProxySettings);
proxyFrameRate?.addEventListener('change', sendProxySettings);
proxyIncludeAudio?.addEventListener('change', sendProxySettings);

// Declared before the socket connects so reconnect reconciliation is safe
// even when the initial connection opens immediately.
let padPointerId = null;
let padLastSend = 0;
let padLastPosition = [0.5, 0.5];
let padNeedsReconcile = false;
let beatQuantizeEnabled = false;
let layerStackRevision = 0;
let compositionRevision = 0;
let presetRevision = 0;
let latestCreative = null;
let latestConstraintDiagnostics = [];
let latestLayerIdentities = [];
let latestLayers = [];
let latestAutopilotSnapshot = {};
let latestAutopilotScenes = [];
let autopilotDraft = { repeat: 'loop', steps: [] };
let autopilotPlanDirty = false;
let autopilotPendingPlanKey = null;
let transportAuthoritativePaused = false;
let transportPendingPaused = null;
let transportRequestSequence = 0;
let mediaAuthoritativeFrozen = false;
let mediaPendingFrozen = null;
let mediaRequestSequence = 0;
let outputAuthoritativeOpen = false;
let outputAuthoritativeDisplay = '';
let outputAuthoritativeGeneration = 0;
let webAuthoredRevision = 0;
let webOperationalRevision = 0;
let webTelemetryRevision = 0;
let outputPendingOpen = null;
let outputRequestSequence = 0;

const QUANTIZABLE_ACTIONS = new Set([
  'set_param', 'set_master_transform', 'set_layer_param', 'set_layer_effect',
  'set_layer_transform', 'set_group_transform', 'set_ntsc_param', 'set_temporal',
  'set_clip_transport', 'set_clip_cue', 'set_layer_matte_param',
  'set_visual_node_param', 'set_composition_group_param',
  'set_composition_group_matte_param', 'set_composition_bus_crossfade',
  'set_composition_bus_mix',
  'set_morph', 'morph_capture', 'morph_clear', 'morph_glide',
]);

// --- WebSocket ---

let ws;
function connect() {
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${wsProto}://${location.host}/ws`);

  ws.onopen = () => {
    statusEl.classList.add('connected');
    statusEl.classList.remove('disconnected');
    statusEl.title = 'connected';
    if (padNeedsReconcile || padPointerId !== null) {
      const active = padPointerId !== null;
      if (sendAction({ action: 'pad', x: padLastPosition[0], y: padLastPosition[1], active })) {
        padNeedsReconcile = false;
      }
    }
    // B10 bend pads: re-assert every held state on reconnect so a release
    // the server missed can never stay latched, and a hold the server missed
    // resumes.
    if (typeof bendHeldLocal !== 'undefined') {
      for (let i = 0; i < 6; i++) {
        sendAction({ action: 'bend_pad', index: i, held: bendHeldLocal[i] });
      }
    }
    reconcileInterruptedHistoryGestures();
    reconcileGyroStreamConnection();
    // B11: a fresh socket is a fresh client id, so the watch declaration
    // must be re-asserted or the bay stays unarmed for this panel.
    if (typeof sendMonitorWatch === 'function') sendMonitorWatch();
  };

  ws.onclose = () => {
    // If the server saw the press but misses the eventual release, spring
    // return would otherwise remain latched off forever.
    if (padPointerId !== null) padNeedsReconcile = true;
    statusEl.classList.remove('connected');
    statusEl.classList.add('disconnected');
    statusEl.title = 'disconnected';
    transportRequestSequence += 1;
    transportPendingPaused = null;
    renderMasterTransport(transportAuthoritativePaused, false);
    mediaRequestSequence += 1;
    mediaPendingFrozen = null;
    renderMediaFreeze(mediaAuthoritativeFrozen, false);
    if (expertMediaToggle) {
      expertMediaToggle.checked = authoritativeMediaSafetyMode === 'expert';
      expertMediaToggle.toggleAttribute('aria-busy', false);
    }
    if (newLayerFit) newLayerFit.value = authoritativeNewLayerFit;
    resetProxySettingsControls();
    outputRequestSequence += 1;
    outputPendingOpen = null;
    renderOutputWindow(outputAuthoritativeOpen, false, 'Control connection is offline.');
    rememberInterruptedHistoryGestures();
    showGyroDisconnected();
    setTimeout(connect, 2000);
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) return;

    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'state') {
        webAuthoredRevision = Number(msg.authored_revision) || 0;
        webOperationalRevision = Number(msg.operational_revision) || 0;
        webTelemetryRevision = Number(msg.telemetry_revision) || 0;
        syncEffects(msg.effects);
        syncMasterTransform(msg.master_transform);
        syncNtsc(msg.ntsc);
        layerStackRevision = Number(msg.layer_stack_revision) || 0;
        compositionRevision = Number(msg.composition_revision) || 0;
        syncLayers(msg.layers);
        syncCreative(msg.creative);
        syncConstraintDiagnostics(msg.constraint_diagnostics);
        syncPerformance(msg.performance);
        syncLibrary(msg.library);
        syncMediaSafety(msg.media_safety);
        syncNewLayerFit(msg.new_layer_fit);
        syncProxySettings(msg.proxy_settings);
        syncTransport(msg.program_frozen ?? msg.paused, msg.media_frozen);
        syncExport(msg.export_progress, msg.export_error, msg.export_status, msg.export_warnings, msg.export_motion);
        syncHistory(msg.history);
        syncPresets(msg.presets);
        syncRecovery(msg.recovery_available, msg.recovery_status);
        syncPatchSave(msg.patch_save_status || '');
        syncPatchLoad(msg.patch_load_status || '');
        syncShowBundle(msg.show_bundle || {});
        syncModulation(msg.modulation);
        syncSnapshotBank(msg.morph);
        syncControlFilters(msg.modulation);
        syncAudio(msg.audio);
        syncMidi(msg.midi);
        syncControllerRuntime(msg.controller_runtime);
        syncOscRuntime(msg.osc_runtime);
        syncTemporal(msg.temporal);
        syncGesture(msg.gesture);
        syncPerformanceRecorder(msg.performance_recorder);
        syncMasterMotion(msg.master_motion);
        syncSpout(msg.spout);
        syncRemote(msg.remote_url, msg.remote_status);
        syncMorph(msg.morph);
        syncOutputWindow(msg.legacy_output_window ?? msg.output_window, msg.output_error, msg.output_display, msg.output_displays, msg.output_display_generation);
        syncRecorder(msg.recorder);
        syncStageHealth(msg.stage_health);
        syncMonitorBay(msg.monitor_bay);
        syncBlackout(msg.blackout);
        syncQuantize(msg.quantized_pending || 0);
        syncRangeEditors(document);
      } else if (msg.type === 'live') {
        const authored = Number(msg.authored_revision) || 0;
        const operationalRevision = Number(msg.operational_revision) || 0;
        const telemetryRevision = Number(msg.telemetry_revision) || 0;
        // Never render a live domain against the wrong authored graph. Closing
        // invokes the established reconnect path, whose first message is a
        // separately cached complete state generation.
        if (Number(msg.wire_version) !== 2 || authored !== webAuthoredRevision) {
          console.warn('[ws] live base mismatch; requesting a coherent reconnect');
          ws.close(4001, 'state revision mismatch');
          return;
        }
        const op = msg.operational || {};
        const telemetry = msg.telemetry || {};
        if (operationalRevision >= webOperationalRevision) {
          webOperationalRevision = operationalRevision;
          syncTransport(op.program_frozen, op.media_frozen);
          syncExport(op.export_progress, op.export_error, op.export_status, op.export_warnings, op.export_motion);
          syncRecovery(op.recovery_available, op.recovery_status);
          syncPatchSave(op.patch_save_status || '');
          syncPatchLoad(op.patch_load_status || '');
          syncShowBundle(op.show_bundle || {});
          syncControllerRuntime(op.controller_runtime);
          syncOscRuntime(op.osc_runtime);
          syncSpout(op.spout);
          syncRemote(op.remote_url, op.remote_status);
          syncOutputWindow(op.output_window, op.output_error, op.output_display, op.output_displays, op.output_display_generation);
          syncRecorder(op.recorder);
          syncBlackout(op.blackout);
          syncQuantize(op.quantized_pending || 0);
          syncConstraintDiagnostics(op.constraint_diagnostics);
        }
        if (telemetryRevision >= webTelemetryRevision) {
          webTelemetryRevision = telemetryRevision;
          syncModulation(telemetry.modulation);
          syncControlFilters(telemetry.modulation);
          syncAudio(telemetry.audio);
          syncMidi(telemetry.midi);
          syncTemporal(telemetry.temporal);
          syncGesture(telemetry.gesture);
          syncPerformanceRecorder(telemetry.performance_recorder);
          syncMasterMotion(telemetry.master_motion);
          syncMorph(telemetry.morph);
          syncStageHealth(telemetry.stage_health);
          syncMonitorBay(telemetry.monitor_bay);
        }
        syncRangeEditors(document);
      }
    } catch (err) {
      console.warn('[ws] parse error:', err);
    }
  };
}
connect();

// --- Interaction guard -------------------------------------------------
// The 30Hz state broadcast must never overwrite a control the performer's
// hand is on. Desktop focus (activeElement) isn't enough: on touch,
// dragging a slider never focuses it. Track pointer-held and recently-
// edited controls explicitly; sync code asks canSync() before writing.

const touchedControls = new Map(); // element -> Set(pointerId) | last-edit ms
const rangeControlPeers = new WeakMap(); // slider <-> editable numeric display
const activeHistoryGestures = new Map(); // pointer/key identity -> gesture record
const interruptedHistoryGestures = [];
let historyGestureSequence = 0;

function nextHistoryGestureId() {
  historyGestureSequence = historyGestureSequence >= Number.MAX_SAFE_INTEGER
    ? 1
    : historyGestureSequence + 1;
  return historyGestureSequence;
}

function beginScalarHistoryGesture(key, control) {
  if (!control || beatQuantizeEnabled) return false;
  if (activeHistoryGestures.has(key)) return true;
  // The protocol owns one scalar destination at a time. Prevent a second
  // touch/key gesture from being silently folded into the first controller's
  // transaction; the server enforces the same law across browser clients.
  if (activeHistoryGestures.size) return false;
  const gesture = { id: nextHistoryGestureId(), control, dirty: false };
  if (sendAction({ action: 'begin_history_gesture', gesture_id: gesture.id })) {
    activeHistoryGestures.set(key, gesture);
    return true;
  }
  return false;
}

function closeScalarHistoryGesture(key) {
  const gesture = activeHistoryGestures.get(key);
  if (!gesture) return;
  activeHistoryGestures.delete(key);
  sendAction(historyBoundaryAction(gesture));
}

function historyBoundaryAction(gesture) {
  if (gesture.dirty) {
    return { action: 'end_history_gesture', gesture_id: gesture.id };
  }
  return { action: 'cancel_history_gesture', gesture_id: gesture.id };
}

function rememberInterruptedHistoryGestures() {
  for (const gesture of activeHistoryGestures.values()) {
    interruptedHistoryGestures.push({ id: gesture.id, dirty: gesture.dirty });
  }
  activeHistoryGestures.clear();
}

function reconcileInterruptedHistoryGestures() {
  while (interruptedHistoryGestures.length) {
    const gesture = interruptedHistoryGestures.shift();
    if (!sendAction(historyBoundaryAction(gesture))) {
      interruptedHistoryGestures.unshift(gesture);
      break;
    }
  }
}

document.addEventListener('pointerdown', (e) => {
  const el = e.target.closest('input,select,[data-range-editor]');
  if (el) {
    const current = touchedControls.get(el);
    const held = current instanceof Set ? current : new Set();
    held.add(e.pointerId);
    touchedControls.set(el, held);
  }
  const slider = e.target.closest('input[type="range"]');
  if (slider && !beginScalarHistoryGesture(`pointer:${e.pointerId}`, slider)) {
    e.preventDefault();
    e.stopImmediatePropagation();
  }
}, true);
const releaseControl = (e) => {
  for (const [el, t] of touchedControls) {
    if (!(t instanceof Set) || !t.has(e.pointerId)) continue;
    t.delete(e.pointerId);
    touchedControls.set(el, t.size ? t : performance.now());
  }
  closeScalarHistoryGesture(`pointer:${e.pointerId}`);
};
document.addEventListener('pointerup', releaseControl, true);
document.addEventListener('pointercancel', releaseControl, true);
const releaseAllControls = () => {
  const now = performance.now();
  for (const [el, state] of touchedControls) {
    if (!el.isConnected) touchedControls.delete(el);
    else if (state instanceof Set) touchedControls.set(el, now);
  }
  for (const key of Array.from(activeHistoryGestures.keys())) {
    closeScalarHistoryGesture(key);
  }
};
window.addEventListener('blur', releaseAllControls);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) releaseAllControls();
});
document.addEventListener('input', (e) => {
  if (!(touchedControls.get(e.target) instanceof Set)) {
    touchedControls.set(e.target, performance.now());
  }
  for (const gesture of activeHistoryGestures.values()) {
    if (gesture.control === e.target) gesture.dirty = true;
  }
}, true);

const HISTORY_RANGE_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End']);
document.addEventListener('keydown', (event) => {
  if (!HISTORY_RANGE_KEYS.has(event.key) || !event.target.matches?.('input[type="range"]')) return;
  if (!beginScalarHistoryGesture('keyboard', event.target)) {
    event.preventDefault();
    event.stopImmediatePropagation();
  }
}, true);
document.addEventListener('keyup', (event) => {
  if (HISTORY_RANGE_KEYS.has(event.key)) closeScalarHistoryGesture('keyboard');
}, true);

function controlIsBusy(el) {
  if (!el) return false;
  if (document.activeElement === el) return true;
  const t = touchedControls.get(el);
  return t instanceof Set || (typeof t === 'number' && performance.now() - t < 800);
}

function canSync(el) {
  if (!el || controlIsBusy(el)) return false;
  const peer = rangeControlPeers.get(el);
  return !peer || !controlIsBusy(peer);
}

function sendAction(action) {
  const outgoing = beatQuantizeEnabled && QUANTIZABLE_ACTIONS.has(action.action)
    ? { action: 'quantized', inner: action }
    : action;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(outgoing));
    return true;
  }
  return false;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function layerBlendModeInfo(key) {
  return LAYER_BLEND_MODES.find((mode) => mode.key === key) || LAYER_BLEND_MODES[0];
}

function layerBlendTitle(key) {
  return `${layerBlendModeInfo(key).description} ${LAYER_BLEND_ORDER_POLICY}`;
}

// --- Collision Rack + one-level composition ---------------------------

const creativeRackScope = document.getElementById('creative-rack-scope');
const creativeNodeKind = document.getElementById('creative-node-kind');
const creativeNodeAdd = document.getElementById('creative-node-add');
const creativeRackNodes = document.getElementById('creative-rack-nodes');
const creativeGroups = document.getElementById('creative-groups');
const creativeRoot = document.getElementById('creative-root');
const creativeStatus = document.getElementById('creative-status');
const creativeRevision = document.getElementById('creative-revision');
const creativeBusCrossfade = document.getElementById('creative-bus-crossfade');
const creativeBusCrossfadeValue = document.getElementById('creative-bus-crossfade-value');
const creativeGroupCreate = document.getElementById('creative-group-create');
const creativeGroupName = document.getElementById('creative-group-name');
const creativeGroupMembers = document.getElementById('creative-group-members');
const creativeGroupGizmoRelease = document.getElementById('creative-group-gizmo-release');
const creativeGroupGizmoReleaseStatus = document.getElementById('creative-group-gizmo-release-status');
let creativeStructureKey = '';

const CREATIVE_NODE_INFO = Object.freeze({
  legacy_canonical: { label: 'Legacy Canonical', marker: true },
  legacy_temporal: { label: 'Legacy Temporal', marker: true },
  transform: { label: 'Transform' },
  digital_color: { label: 'Digital / Color' },
  key: { label: 'Key' },
  cellular: { label: 'Cellular' },
  shift: { label: 'Shift' },
  grain: { label: 'Grain' },
  mask: { label: 'Mask' },
  displace: { label: 'Displace' },
  symmetry: { label: 'Symmetry Field' },
  residual: { label: 'Residual' },
  study: { label: 'Study' },
  scan_processor: { label: 'Scan Processor' },
  block_dct: { label: 'Block DCT' },
  pixel_sort: { label: 'Pixel Sort' },
  avalanche: { label: 'Filter Avalanche' },
});

const enumDef = (key, label, options) => ({ key, label, type: 'enum', options });
const floatDef = (key, label, min, max, step) => ({ key, label, type: 'float', min, max, step });
const vecDef = (key, label, min, max, step, components = ['X', 'Y']) => ({ key, label, type: 'vec', min, max, step, components });
const boolDef = (key, label) => ({ key, label, type: 'bool' });
const uintDef = (key, label) => ({ key, label, type: 'uint' });

const CREATIVE_NODE_PARAMS = Object.freeze({
  transform: [
    vecDef('position', 'Position', -4, 4, 0.01),
    vecDef('scale', 'Scale', -16, 16, 0.01),
    vecDef('anchor', 'Anchor', -2, 3, 0.01),
    floatDef('rotation_deg', 'Rotation', -180, 180, 0.1),
    floatDef('skew_deg', 'Skew', -89, 89, 0.1),
    floatDef('skew_axis_deg', 'Skew axis', -180, 180, 0.1),
    enumDef('fit_mode', 'Fit', [['stretch', 'Stretch'], ['fit', 'Fit'], ['fill', 'Fill'], ['native', 'Native']]),
    floatDef('crop_left', 'Crop left', 0, 1, 0.001),
    floatDef('crop_top', 'Crop top', 0, 1, 0.001),
    floatDef('crop_right', 'Crop right', 0, 1, 0.001),
    floatDef('crop_bottom', 'Crop bottom', 0, 1, 0.001),
    enumDef('edge_mode', 'Edge', [['transparent', 'Transparent'], ['clamp', 'Clamp'], ['repeat', 'Repeat'], ['mirror', 'Mirror']]),
    enumDef('sampling', 'Sampling', [['linear', 'Linear'], ['nearest', 'Nearest']]),
  ],
  digital_color: [
    floatDef('pixelate_size', 'Pixelate', 1, 32, 0.1),
    floatDef('rgb_split', 'RGB split', 0, 30, 0.1),
    floatDef('downsample', 'Downsample', 0.05, 1, 0.01),
    floatDef('hue_shift', 'Hue', -180, 180, 0.1),
    floatDef('saturation', 'Saturation', -1, 1, 0.01),
    floatDef('brightness', 'Brightness', -1, 1, 0.01),
    floatDef('contrast', 'Contrast', -1, 1, 0.01),
    floatDef('posterize', 'Posterize', 0, 16, 0.1),
    floatDef('invert', 'Invert', 0, 1, 0.01),
    floatDef('vignette', 'Vignette', 0, 1.5, 0.01),
    floatDef('color_drift', 'Color drift', 0, 0.02, 0.0001),
  ],
  key: [
    enumDef('mode', 'Mode', [['keep_bright', 'Keep bright'], ['keep_dark', 'Keep dark'], ['remove_color', 'Remove color'], ['keep_color', 'Keep color']]),
    floatDef('threshold', 'Threshold', 0, 1, 0.001),
    floatDef('softness', 'Softness', 0, 0.5, 0.001),
    vecDef('color', 'Color', 0, 1, 0.001, ['R', 'G', 'B']),
    floatDef('tolerance', 'Tolerance', 0, 1, 0.001),
    boolDef('invert', 'Invert'),
  ],
  cellular: [
    floatDef('amount', 'Amount', 0, 1, 0.001),
    floatDef('scale', 'Scale', 2, 32, 0.1),
    floatDef('warp', 'Warp', 0, 1, 0.001),
    floatDef('speed', 'Speed', 0, 2, 0.001),
    floatDef('gap_amount', 'Gap key', 0, 1, 0.001),
    floatDef('gap_threshold', 'Gap threshold', 0, 1, 0.001),
    floatDef('gap_softness', 'Gap softness', 0, 0.5, 0.001),
    uintDef('seed', 'Seed'),
  ],
  shift: [
    floatDef('amount', 'Amount', 0, 1, 0.001),
    floatDef('block_size', 'Block size', 2, 256, 1),
    floatDef('density', 'Density', 0, 1, 0.001),
    floatDef('speed', 'Speed', 0, 20, 0.01),
    uintDef('seed', 'Seed'),
  ],
  grain: [
    floatDef('intensity', 'Intensity', 0, 0.3, 0.001),
    floatDef('size', 'Size', 1, 4, 0.01),
    enumDef('algorithm', 'Algorithm', [['gaussian', 'Gaussian'], ['perlin', 'Perlin'], ['salt_pepper', 'Salt + pepper'], ['blue', 'Blue noise']]),
    boolDef('color', 'Color'),
    uintDef('seed', 'Seed'),
  ],
  mask: [
    vecDef('rectangle_center', 'Rectangle center', -2, 3, 0.001),
    vecDef('rectangle_size', 'Rectangle size', 0, 4, 0.001),
    floatDef('rectangle_rotation_deg', 'Rectangle rotation', -180, 180, 0.1),
    floatDef('rectangle_feather', 'Rectangle feather', 0, 1, 0.001),
    boolDef('rectangle_invert', 'Rectangle invert'),
    vecDef('ellipse_center', 'Ellipse center', -2, 3, 0.001),
    vecDef('ellipse_radii', 'Ellipse radii', 0, 2, 0.001),
    floatDef('ellipse_rotation_deg', 'Ellipse rotation', -180, 180, 0.1),
    floatDef('ellipse_feather', 'Ellipse feather', 0, 1, 0.001),
    boolDef('ellipse_invert', 'Ellipse invert'),
    floatDef('image_amount', 'Image amount', 0, 1, 0.001),
    floatDef('image_threshold', 'Image threshold', 0, 1, 0.001),
    floatDef('image_softness', 'Image softness', 0, 0.5, 0.001),
  ],
  displace: [
    floatDef('amount_x', 'Amount X', -1, 1, 0.001),
    floatDef('amount_y', 'Amount Y', -1, 1, 0.001),
    enumDef('boundary', 'Boundary', [['transparent', 'Transparent'], ['mirror', 'Mirror'], ['wrap', 'Wrap'], ['hold', 'Hold']]),
  ],
  // Ranges here must equal the NODE_PARAM_DESCRIPTORS ranges the server
  // validates against. The four routes are absent on purpose: they own the
  // ordered slot-addressed topology action, not this value path.
  symmetry: [
    enumDef('symmetry_mode', 'Mode', [['cyclic', 'Cyclic Cn'], ['dihedral', 'Dihedral Dn'], ['planar_p1', 'Planar p1'], ['planar_pm', 'Planar pm'], ['planar_p2', 'Planar p2'], ['planar_pmm', 'Planar pmm'], ['log_spiral', 'Log spiral'], ['orbit', 'Orbit']]),
    enumDef('symmetry_boundary', 'Boundary', [['transparent', 'Transparent'], ['mirror', 'Mirror'], ['wrap', 'Wrap'], ['hold', 'Hold'], ['cellular_reentry', 'Cellular re-entry']]),
    floatDef('symmetry_base_folds', 'Folds', 1, 32, 0.01),
    floatDef('symmetry_fold_offset', 'Fold offset', -32, 32, 0.01),
    floatDef('symmetry_radial_phase_deg', 'Radial phase', -180, 180, 0.1),
    floatDef('symmetry_orbit_phase', 'Orbit phase', -1, 1, 0.001),
    floatDef('symmetry_planar_axis_deg', 'Lattice axis', -180, 180, 0.1),
    floatDef('symmetry_planar_phase', 'Lattice phase', -4, 4, 0.001),
    floatDef('symmetry_cell_skew', 'Cell skew', -1, 1, 0.001),
    floatDef('symmetry_spiral_scale', 'Spiral scale', -1, 1, 0.001),
    floatDef('symmetry_orbit_radius', 'Orbit radius', 0, 1, 0.001),
    floatDef('symmetry_orbit_spin_deg', 'Orbit spin', -180, 180, 0.1),
    floatDef('symmetry_motion_gain', 'Motion gain', -1, 1, 0.001),
    floatDef('symmetry_hue_span', 'Hue span', 0, 1, 0.001),
    vecDef('symmetry_center', 'Center', -1, 2, 0.001),
    boolDef('symmetry_source_carrier', 'Source · carrier'),
    boolDef('symmetry_source_donor0', 'Source · donor 0'),
    boolDef('symmetry_source_donor1', 'Source · donor 1'),
    boolDef('symmetry_source_history', 'Source · clean history'),
    boolDef('symmetry_motion_slot0', 'Motion · slot 0'),
    boolDef('symmetry_motion_slot1', 'Motion · slot 1'),
    uintDef('symmetry_seed', 'Seed'),
  ],
  residual: [
    floatDef('mix', 'Mix', 0, 1, 0.001),
    floatDef('detail_gain', 'Detail gain', 0, 4, 0.001),
    enumDef('block', 'Block', [['four', '4 px'], ['eight', '8 px'], ['sixteen', '16 px'], ['thirty_two', '32 px'], ['sixty_four', '64 px']]),
    enumDef('quantization', 'Quantization', [['off', 'Off'], ['coarse', 'Coarse'], ['medium', 'Medium'], ['fine', 'Fine']]),
    uintDef('seed', 'Seed'),
  ],
  // A Study's whole authored surface is its document digest, assigned by the
  // dedicated document action rather than a generic parameter row; the card
  // renders its own paste surface.
  study: [
  ],
  // Ranges here must equal the NODE_PARAM_DESCRIPTORS ranges the server
  // validates against. Lines/samples are plan-time geometry (number inputs,
  // engine-clamped 16-1080 / 64-512); the reversals are discrete laws.
  scan_processor: [
    uintDef('scan_lines', 'Lines'),
    uintDef('scan_samples', 'Samples'),
    floatDef('scan_amount', 'Displace', 0, 1, 0.001),
    floatDef('scan_ribbon_width', 'Beam width', 0, 1, 0.001),
    floatDef('scan_velocity_mix', 'Velocity gain', 0, 1, 0.001),
    floatDef('scan_tilt_x', 'Tilt X', -1, 1, 0.001),
    floatDef('scan_tilt_y', 'Tilt Y', -1, 1, 0.001),
    floatDef('scan_perspective', 'Perspective', 0, 1, 0.001),
    floatDef('scan_s_curve', 'S-curve', -1, 1, 0.001),
    floatDef('scan_skew', 'Raster skew', -1, 1, 0.001),
    floatDef('scan_collapse', 'Collapse', 0, 1, 0.001),
    boolDef('scan_reverse_h', 'Reverse sweep'),
    boolDef('scan_reverse_v', 'Reverse field'),
    floatDef('scan_osc_amount', 'Wobble', 0, 1, 0.001),
    floatDef('scan_osc_freq', 'Wobble freq', 0, 1, 0.001),
    floatDef('scan_osc_lock', 'Wobble lock', 0, 1, 0.001),
    floatDef('scan_lissajous', 'Lissajous', 0, 1, 0.001),
    floatDef('scan_mono', 'Mono', 0, 1, 0.001),
    floatDef('scan_hue', 'Colourise', 0, 1, 0.001),
  ],
  block_dct: [
    floatDef('dct_amount', 'Amount', 0, 1, 0.001),
    floatDef('dct_quantize', 'Quantiser', 0, 1, 0.001),
    floatDef('dct_hf_penalty', 'HF penalty', 0, 1, 0.001),
    floatDef('dct_chroma_crush', 'Chroma crush', 0, 1, 0.001),
    floatDef('dct_block', 'Block size', 0, 1, 0.001),
  ],
  pixel_sort: [
    floatDef('sort_amount', 'Amount', 0, 1, 0.001),
    floatDef('sort_threshold', 'Threshold', 0, 1, 0.001),
  ],
  avalanche: [
    floatDef('avalanche_amount', 'Amount', 0, 1, 0.001),
    floatDef('avalanche_run', 'Run', 0, 1, 0.001),
    enumDef('avalanche_axis', 'Axis', [['sub', 'Sub (row)'], ['up', 'Up (column)'], ['average', 'Average (diagonal)']]),
  ],
});

function creativeScopeWire(value = creativeRackScope?.value || 'master') {
  if (value.startsWith('layer:')) return { scope: 'layer', layer_id: value.slice(6) };
  if (value.startsWith('group:')) return { scope: 'group', group_id: value.slice(6) };
  return { scope: 'master' };
}

function creativeScopeRack(value = creativeRackScope?.value || 'master') {
  if (!latestCreative) return null;
  if (value === 'master') return latestCreative.master_rack || null;
  if (value.startsWith('layer:')) {
    const id = value.slice(6);
    return (latestCreative.layer_racks || []).find(([layerId]) => String(layerId) === id)?.[1] || null;
  }
  if (value.startsWith('group:')) {
    const id = value.slice(6);
    return (latestCreative.groups || []).find((group) => String(group.group_id) === id)?.rack || null;
  }
  return null;
}

function creativeLayerLabel(layerId) {
  const index = latestLayers.findIndex((layer) => String(layer.layer_id) === String(layerId));
  const layer = index >= 0 ? latestLayers[index] : null;
  return layer ? `Layer ${index + 1} · ${layer.filename || layer.source_kind || layerId}` : `Layer ${layerId}`;
}

function creativeSetStatus(message, isError = false) {
  if (!creativeStatus) return;
  creativeStatus.textContent = message;
  creativeStatus.classList.toggle('error', isError);
}

function constraintScopeLabel(scope) {
  if (!scope || typeof scope.kind !== 'string') return 'unknown scope';
  return scope.stable_id === undefined
    ? scope.kind
    : `${scope.kind} #${scope.stable_id}`;
}

function constraintRemediationPreview(diagnostic, candidate) {
  return (diagnostic.remediation_previews || []).find((preview) =>
    String(preview.candidate_id) === String(candidate.id)
      && Number(preview.base_revision) === Number(candidate.base_revision)
  ) || null;
}

function constraintPixelOrderLabel(order) {
  return Array.isArray(order) && order.length ? order.join(' → ') : 'unchanged';
}

function confirmConstraintRemediation(diagnostic, candidate, preview) {
  if (!diagnostic || !candidate || !preview) return;
  if (Number(candidate.base_revision) !== compositionRevision
      || Number(preview.base_revision) !== compositionRevision) {
    creativeSetStatus('Remediation preview is stale; refresh before applying it.', true);
    return;
  }
  const consequence = preview.consequence || {};
  const plan = consequence.plan || {};
  const before = constraintPixelOrderLabel(consequence.pixel_order_before);
  const after = constraintPixelOrderLabel(consequence.pixel_order_after);
  const confirmed = window.confirm(
    `${candidate.description}\n\nPixel order before:\n${before}\n\nPixel order after:\n${after}\n\nPlan: ${plan.kind || 'unknown'}, ${plan.full_frame_passes || 0} full-frame pass(es), ${plan.retained_surface_layers || 0} retained surface(s), ${plan.creative_bytes || 0} B\n\nApply this exact planner preview?`,
  );
  if (!confirmed) return;
  creativeSend({
    action: 'apply_constraint_remediation',
    candidate_id: String(candidate.id),
    composition_revision: compositionRevision,
  }, 'Applying confirmed planner remediation…');
}

/// Render planner-owned protocol fields. This deliberately never searches the
/// human `text` for a repair or severity decision.
function syncConstraintDiagnostics(diagnostics) {
  latestConstraintDiagnostics = Array.isArray(diagnostics) ? diagnostics : [];
  if (!creativeStatus) return;
  const diagnostic = latestConstraintDiagnostics[0];
  if (!diagnostic) {
    delete creativeStatus.dataset.constraintCode;
    if (latestCreative?.status) {
      creativeSetStatus(
        latestCreative.status,
        /reject|error|invalid|cycle|budget|missing|stale/i.test(latestCreative.status),
      );
    }
    return;
  }

  const scopes = (diagnostic.affected || []).map(constraintScopeLabel).join(', ');
  const delta = diagnostic.resource_delta
    ? `<span class="constraint-delta">${escapeHtml(diagnostic.resource_delta.resource)}: ${escapeHtml(diagnostic.resource_delta.resulting_total)} / ${escapeHtml(diagnostic.resource_delta.limit)}</span>`
    : '';
  const helpUrl = typeof diagnostic.help_url === 'string'
    && diagnostic.help_url.startsWith('/help/constraints/')
    ? diagnostic.help_url
    : '#creative-panel';
  const repairs = (diagnostic.remediations || []).map((candidate) => {
    const preview = constraintRemediationPreview(diagnostic, candidate);
    if (!preview) return '';
    const consequence = preview.consequence || {};
    const plan = consequence.plan || {};
    const planSummary = `${plan.kind || 'unknown'} · ${plan.full_frame_passes || 0} full-frame pass(es) · ${plan.retained_surface_layers || 0} retained surface(s) · ${plan.creative_bytes || 0} B`;
    return `<li data-candidate-id="${escapeHtml(candidate.id)}">
      <span>${escapeHtml(candidate.description)}</span>
      <span class="constraint-preview">${escapeHtml(constraintPixelOrderLabel(consequence.pixel_order_before))} → ${escapeHtml(constraintPixelOrderLabel(consequence.pixel_order_after))}</span>
      <span class="constraint-preview">${escapeHtml(planSummary)}</span>
      <button type="button" class="constraint-apply" data-candidate-id="${escapeHtml(candidate.id)}">Preview and apply…</button>
    </li>`;
  }).join('');
  creativeStatus.dataset.constraintCode = diagnostic.code || 'unknown';
  creativeStatus.classList.add('error');
  creativeStatus.innerHTML = `<span class="constraint-code">${escapeHtml(diagnostic.code || 'constraint_error')}</span>
    <span>${escapeHtml(diagnostic.text || 'Creative plan rejected.')}</span>
    ${scopes ? `<span class="constraint-scopes">Affected: ${escapeHtml(scopes)}</span>` : ''}
    ${delta}
    ${repairs ? `<ul class="constraint-remediations" aria-label="Planner remediation candidates">${repairs}</ul>` : ''}
    <a class="constraint-help" href="${escapeHtml(helpUrl)}" target="_blank" rel="noopener">Why this was refused</a>`;
  creativeStatus.querySelectorAll('.constraint-apply').forEach((button) => {
    button.addEventListener('click', () => {
      const candidate = (diagnostic.remediations || []).find((item) =>
        String(item.id) === String(button.dataset.candidateId)
      );
      const preview = candidate ? constraintRemediationPreview(diagnostic, candidate) : null;
      confirmConstraintRemediation(diagnostic, candidate, preview);
    });
  });
}

function creativeSend(action, pending) {
  const sent = sendAction(action);
  creativeSetStatus(sent ? pending : 'Control connection is offline.', !sent);
  return sent;
}

function creativeOptionHtml(options, selected) {
  return options.map(([value, label]) => `<option value="${escapeHtml(value)}" ${String(value) === String(selected) ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('');
}

function creativeControlHtml(def, value, extraClass = '') {
  const klass = `creative-control ${extraClass}`.trim();
  if (def.type === 'bool') {
    return `<label class="${klass}" data-creative-param="${escapeHtml(def.key)}"><span>${escapeHtml(def.label)}</span><input type="checkbox" ${value ? 'checked' : ''}><output>${value ? 'on' : 'off'}</output></label>`;
  }
  if (def.type === 'enum') {
    return `<label class="${klass}" data-creative-param="${escapeHtml(def.key)}"><span>${escapeHtml(def.label)}</span><select>${creativeOptionHtml(def.options, value)}</select><output></output></label>`;
  }
  if (def.type === 'uint') {
    return `<label class="${klass}" data-creative-param="${escapeHtml(def.key)}"><span>${escapeHtml(def.label)}</span><input type="number" min="0" max="4294967295" step="1" value="${Number(value) || 0}"><output></output></label>`;
  }
  if (def.type === 'vec') {
    const values = Array.isArray(value) ? value : def.components.map(() => 0);
    const controls = def.components.map((component, index) => `<label class="creative-control" data-creative-param="${escapeHtml(def.key)}" data-component="${index}"><span>${escapeHtml(def.label)} ${component}</span><input type="range" min="${def.min}" max="${def.max}" step="${def.step}" value="${Number(values[index]) || 0}"><output>${Number(values[index] || 0).toFixed(3)}</output></label>`).join('');
    return `<div class="creative-control-wide creative-vector" data-vector-param="${escapeHtml(def.key)}">${controls}</div>`;
  }
  return `<label class="${klass}" data-creative-param="${escapeHtml(def.key)}"><span>${escapeHtml(def.label)}</span><input type="range" min="${def.min}" max="${def.max}" step="${def.step}" value="${Number(value) || 0}"><output>${Number(value || 0).toFixed(3)}</output></label>`;
}

function creativeRouteToken(route) {
  const input = route?.input || { source: 'one_below' };
  switch (input.source) {
    case 'selected_layer': return `layer:${input.layer_id}:${input.stage || 'post_local_effects'}`;
    case 'missing_selected_layer': return `missing-layer:${input.saved_position || '?'}`;
    case 'group_output': return `group:${input.group_id}`;
    case 'missing_group_output': return `missing-group:${input.group_id}`;
    case 'all_below': return 'all_below';
    case 'clean_program': return 'clean_program';
    case 'gesture_canvas': return 'gesture_canvas';
    case 'program_tap': return 'program_tap';
    default: return 'one_below';
  }
}

function creativeRouteOptions(route) {
  const options = [
    ['one_below', 'One below'],
    ['all_below', 'All below'],
    ['clean_program', 'Clean program (N−1)'],
    ['gesture_canvas', 'Gesture canvas (etched field)'],
    ['program_tap', 'Program re-entry (N−1 audience)'],
  ];
  for (const layer of latestLayers) {
    const id = String(layer.layer_id);
    options.push([`layer:${id}:pre_local_effects`, `${creativeLayerLabel(id)} · pre`]);
    options.push([`layer:${id}:post_local_effects`, `${creativeLayerLabel(id)} · post`]);
  }
  for (const group of latestCreative?.groups || []) {
    options.push([`group:${group.group_id}`, `Group · ${group.name || group.group_id}`]);
  }
  const token = creativeRouteToken(route);
  if (!options.some(([value]) => value === token)) options.push([token, `${token} · missing`]);
  return options;
}

function creativeRouteFromToken(token, timing) {
  const normalizedTiming = token === 'clean_program' ? 'previous_frame' : timing;
  let input;
  if (token.startsWith('layer:')) {
    const [, layerId, stage] = token.split(':');
    input = { source: 'selected_layer', layer_id: layerId, stage };
  } else if (token.startsWith('group:')) {
    input = { source: 'group_output', group_id: token.slice(6) };
  } else if (token === 'all_below') {
    input = { source: 'all_below' };
  } else if (token === 'clean_program') {
    input = { source: 'clean_program' };
  } else if (token === 'gesture_canvas') {
    // The etched field is a master-scope singleton: no ID, no saved position,
    // and no scope ordering, so both timings are authorable.
    input = { source: 'gesture_canvas' };
  } else if (token === 'program_tap') {
    // The programme tap is the same singleton shape, and it is N-1 by
    // construction, so both timings read the same committed copy.
    input = { source: 'program_tap' };
  } else {
    input = { source: 'one_below' };
  }
  return { input, timing: normalizedTiming };
}

// `slot` tags the editor so a node owning more than one route binds each editor
// to its own submit closure and its own ordered action. Slot index is route
// identity, so it also travels on the wire; an untagged editor is a
// single-route node. `label` keeps sibling donor selects distinguishable to a
// screen reader instead of repeating one generic name.
function creativeRouteEditorHtml(route, channel = 'alpha', invert = false, fieldOnly = false, slot = '', label = 'Donor') {
  const token = creativeRouteToken(route);
  const timing = route?.timing || 'current_frame';
  const channelControls = fieldOnly ? '' : `<label>Channel <select class="creative-route-channel">${creativeOptionHtml([['alpha', 'Alpha'], ['luma', 'Luma'], ['red', 'Red'], ['green', 'Green'], ['blue', 'Blue']], channel)}</select></label>
      <label><input class="creative-route-invert" type="checkbox" ${invert ? 'checked' : ''}> Invert</label>`;
  return `<div class="creative-route-editor"${slot ? ` data-route-slot="${escapeHtml(slot)}"` : ''}>
    <div class="creative-route-row">
      <label>${escapeHtml(label)} <select class="creative-route-source">${creativeOptionHtml(creativeRouteOptions(route), token)}</select></label>
      <label>Timing <select class="creative-route-timing">
        <option value="current_frame" ${timing === 'current_frame' ? 'selected' : ''}>Current frame</option>
        <option value="previous_frame" ${timing === 'previous_frame' ? 'selected' : ''}>Previous frame N−1</option>
      </select></label>
      ${channelControls}
    </div>
    <div class="creative-node-diagnostic">Current-frame routes participate in cycle rejection. Clean Program is deliberately previous-frame only.</div>
  </div>`;
}

// A Symmetry motion slot names a whole layer, so it carries no timing, stage,
// channel, or inversion — a motion route never enters the image dependency
// graph. A retained tombstone is listed for provenance only and is refused on
// submit, exactly as the layer Faraday donor select does.
function creativeMotionRouteEditorHtml(donor, slot, label, diagnostic) {
  const kind = donor?.kind || 'none';
  const selected = kind === 'selected'
    ? String(donor.layer_id || '')
    : kind === 'missing' ? `missing:${Number(donor.saved_position || 0)}` : 'none';
  const options = [['none', 'None']];
  latestLayers.forEach((candidate, index) => options.push([String(candidate.layer_id), `Layer ${index + 1} · ${candidate.filename || 'Untitled'}`]));
  if (kind === 'missing') options.push([selected, `Missing saved layer ${Number(donor.saved_position || 0) + 1}`]);
  const note = diagnostic
    ? `<div class="creative-node-diagnostic">${escapeHtml(diagnostic)}</div>`
    : '';
  return `<div class="creative-motion-route-editor" data-route-slot="${escapeHtml(slot)}">
    <div class="creative-route-row">
      <label>${escapeHtml(label)} <select class="creative-motion-route-source">${creativeOptionHtml(options, selected)}</select></label>
    </div>
    ${note}
  </div>`;
}

function creativeRouteDiagnosticHtml(diagnostic) {
  return diagnostic ? `<div class="creative-node-diagnostic">${escapeHtml(diagnostic)}</div>` : '';
}

function wireCreativeRouteEditor(editor, onChange, selfGroupId = '') {
  const source = editor.querySelector('.creative-route-source');
  const timing = editor.querySelector('.creative-route-timing');
  const channel = editor.querySelector('.creative-route-channel');
  const invert = editor.querySelector('.creative-route-invert');
  const submit = () => {
    if (source.value === 'clean_program') timing.value = 'previous_frame';
    if (selfGroupId && source.value === `group:${selfGroupId}` && timing.value === 'current_frame') {
      timing.value = 'previous_frame';
    }
    onChange(creativeRouteFromToken(source.value, timing.value), channel?.value || 'alpha', invert?.checked || false);
  };
  source.addEventListener('change', submit);
  timing.addEventListener('change', submit);
  channel?.addEventListener('change', submit);
  invert?.addEventListener('change', submit);
}

function creativeNodeVisibleDefs(node) {
  const defs = CREATIVE_NODE_PARAMS[node.kind] || [];
  if (node.kind !== 'mask') return defs;
  const variant = node.params?.variant || 'rectangle';
  return defs.filter((def) => def.key.startsWith(`${variant}_`));
}

function creativeNodeValue(node, key) {
  if (key === 'enabled') return node.enabled;
  if (key === 'wet') return node.wet;
  if (key === 'blend') return node.blend;
  return node.params?.[key];
}

function creativeReadControlValue(row, def, container) {
  if (def.type === 'vec') {
    const vector = container.querySelector(`[data-vector-param="${def.key}"]`);
    return [...vector.querySelectorAll('input')].map((input) => Number(input.value));
  }
  const input = row.querySelector('input,select');
  if (def.type === 'bool') return input.checked;
  if (def.type === 'uint') return Math.max(0, Math.min(0xffffffff, Math.trunc(Number(input.value) || 0)));
  if (def.type === 'enum') return input.value;
  return Number(input.value);
}

function wireCreativeNodeControl(card, node, def) {
  const rows = def.type === 'vec'
    ? [...card.querySelectorAll(`[data-creative-param="${def.key}"]`)]
    : [card.querySelector(`[data-creative-param="${def.key}"]`)];
  for (const row of rows) {
    if (!row) continue;
    const input = row.querySelector('input,select');
    const eventName = def.type === 'bool' || def.type === 'enum' || def.type === 'uint' ? 'change' : 'input';
    input.addEventListener(eventName, () => {
      const value = creativeReadControlValue(row, def, card);
      creativeSend({
        action: 'set_visual_node_param',
        scope: creativeScopeWire(),
        node_id: String(node.node_id),
        node_kind: node.kind,
        param: def.key,
        value,
        composition_revision: compositionRevision,
      }, `Applying ${CREATIVE_NODE_INFO[node.kind]?.label || node.kind} ${def.label.toLowerCase()}…`);
      const output = row.querySelector('output');
      if (output) output.textContent = def.type === 'bool' ? (value ? 'on' : 'off') : (typeof value === 'number' ? value.toFixed(3) : '');
    });
  }
}

function renderCreativeRack() {
  if (!creativeRackNodes) return;
  const rack = creativeScopeRack();
  const nodes = rack?.nodes || [];
  creativeNodeAdd.disabled = !rack || nodes.length >= 8 || !compositionRevision;
  creativeRackNodes.innerHTML = nodes.length ? '' : '<div class="creative-help">Empty rack · scope passes its input unchanged.</div>';
  nodes.forEach((node, index) => {
    const info = CREATIVE_NODE_INFO[node.kind] || { label: node.kind };
    const marker = !!info.marker;
    const card = document.createElement('article');
    card.className = `creative-node${marker ? ' creative-node-marker' : ''}`;
    card.dataset.nodeId = String(node.node_id);
    card.setAttribute('role', 'listitem');
    const common = marker ? '' : [
      boolDef('enabled', 'Enabled'),
      floatDef('wet', 'Wet', 0, 1, 0.001),
      enumDef('blend', 'Node blend', LAYER_BLEND_MODES.map((mode) => [mode.key, mode.label])),
    ].map((def) => creativeControlHtml(def, creativeNodeValue(node, def.key))).join('');
    const params = marker ? '' : creativeNodeVisibleDefs(node).map((def) => creativeControlHtml(def, creativeNodeValue(node, def.key))).join('');
    const maskVariant = node.kind === 'mask' ? `<label class="creative-control creative-control-wide"><span>Mask kind</span><select class="creative-mask-variant">${creativeOptionHtml([['rectangle', 'Rectangle'], ['ellipse', 'Ellipse'], ['image', 'Image donor']], node.params?.variant || 'rectangle')}</select><output></output></label>` : '';
    // Displace routes a donor vector field, not a matte, so it reuses the
    // shared editor in field-only mode: no channel and no invert. Residual
    // owns two such routes and gets one slot-tagged editor each, so a reroute
    // can never land on the partner input.
    // The Symmetry Field owns four fixed routes. Each editor carries its own
    // slot so the two image editors and the two motion editors bind to four
    // distinct submit closures instead of the first one found.
    const symmetryRoutes = node.kind === 'symmetry'
      ? [
        creativeRouteEditorHtml(node.params?.symmetry_donor0_tap, 'alpha', false, true, 'image:0', 'Donor 0'),
        creativeRouteDiagnosticHtml(node.params?.donor0_diagnostic),
        creativeRouteEditorHtml(node.params?.symmetry_donor1_tap, 'alpha', false, true, 'image:1', 'Donor 1'),
        creativeRouteDiagnosticHtml(node.params?.donor1_diagnostic),
        creativeMotionRouteEditorHtml(node.params?.symmetry_motion0_donor, 'motion:0', 'Motion 0', node.params?.motion0_diagnostic),
        creativeMotionRouteEditorHtml(node.params?.symmetry_motion1_donor, 'motion:1', 'Motion 1', node.params?.motion1_diagnostic),
      ].join('')
      : '';
    const imageRoute = node.kind === 'mask' && node.params?.variant === 'image'
      ? creativeRouteEditorHtml(node.params.image_tap, node.params.image_channel, node.params.image_invert)
      : node.kind === 'displace'
        ? creativeRouteEditorHtml(node.params?.donor_tap, 'alpha', false, true)
        : node.kind === 'residual'
          ? `${creativeRouteEditorHtml(node.params?.structure_tap, 'alpha', false, true, 'structure', 'Structure donor')}${creativeRouteEditorHtml(node.params?.detail_tap, 'alpha', false, true, 'detail', 'Detail donor')}`
          : symmetryRoutes;
    const nodeDiagnostic = node.params?.diagnostic
      ? `<div class="creative-node-diagnostic">${escapeHtml(node.params.diagnostic)}</div>`
      : '';
    // The Study's whole authored surface is its document digest. The panel
    // pastes a document; the engine validates, compiles it into the bounded
    // host library, and the node keeps only the canonical digest.
    const studyEditor = node.kind === 'study'
      ? `<div class="creative-study-editor">
          <div class="creative-node-diagnostic creative-study-state">${node.params?.document_digest
            ? `Document ${escapeHtml(String(node.params.document_digest).slice(0, 8))}… assigned`
            : 'No document assigned · exact bypass until one is.'}</div>
          <label class="creative-control creative-control-wide"><span>Study document (JSON)</span>
            <textarea class="creative-study-document" rows="6" spellcheck="false" aria-label="Study document JSON"></textarea>
          </label>
          <div class="creative-route-row">
            <button class="creative-study-apply" type="button" aria-label="Assign study document">Assign document</button>
            <button class="creative-study-clear" type="button" ${node.params?.document_digest ? '' : 'disabled'} aria-label="Clear study document">Clear</button>
          </div>
          <div class="creative-node-diagnostic creative-study-error" role="status" aria-live="polite"></div>
        </div>`
      : '';
    card.innerHTML = `<div class="creative-node-head">
      <span class="creative-node-title">${escapeHtml(info.label)}</span>
      <span class="creative-node-id">#${escapeHtml(node.node_id)}</span>
      <button class="creative-node-up" type="button" ${index === 0 ? 'disabled' : ''} aria-label="Move ${escapeHtml(info.label)} earlier">↑</button>
      <button class="creative-node-down" type="button" ${index + 1 === nodes.length ? 'disabled' : ''} aria-label="Move ${escapeHtml(info.label)} later">↓</button>
      <button class="creative-node-remove" type="button" ${marker ? 'disabled title="Legacy execution markers are immutable"' : ''} aria-label="Remove ${escapeHtml(info.label)}">×</button>
    </div><div class="creative-node-controls">${common}${maskVariant}${params}${imageRoute}${studyEditor}</div>${nodeDiagnostic}${marker ? '<div class="creative-node-diagnostic">Frozen compatibility marker · values are supplied by the established engine path.</div>' : ''}`;
    creativeRackNodes.appendChild(card);
    card.querySelector('.creative-node-up')?.addEventListener('click', () => creativeSend({ action: 'move_visual_node', scope: creativeScopeWire(), node_id: String(node.node_id), to: index - 1, composition_revision: compositionRevision }, 'Moving rack node…'));
    card.querySelector('.creative-node-down')?.addEventListener('click', () => creativeSend({ action: 'move_visual_node', scope: creativeScopeWire(), node_id: String(node.node_id), to: index + 1, composition_revision: compositionRevision }, 'Moving rack node…'));
    if (!marker) card.querySelector('.creative-node-remove')?.addEventListener('click', () => creativeSend({ action: 'remove_visual_node', scope: creativeScopeWire(), node_id: String(node.node_id), composition_revision: compositionRevision }, 'Removing rack node…'));
    if (!marker) {
      [boolDef('enabled', 'Enabled'), floatDef('wet', 'Wet', 0, 1, 0.001), enumDef('blend', 'Node blend', [])].forEach((def) => wireCreativeNodeControl(card, node, def));
      creativeNodeVisibleDefs(node).forEach((def) => wireCreativeNodeControl(card, node, def));
    }
    card.querySelector('.creative-mask-variant')?.addEventListener('change', (event) => creativeSend({
      action: 'set_visual_node_mask_variant', scope: creativeScopeWire(), node_id: String(node.node_id), variant: event.currentTarget.value, composition_revision: compositionRevision,
    }, 'Changing mask kind…'));
    card.querySelector('.creative-study-apply')?.addEventListener('click', () => {
      const textarea = card.querySelector('.creative-study-document');
      const error = card.querySelector('.creative-study-error');
      let parsed;
      try {
        parsed = JSON.parse(textarea.value);
      } catch (parseError) {
        if (error) error.textContent = `Document is not valid JSON: ${parseError.message}`;
        return;
      }
      if (error) error.textContent = '';
      creativeSend({
        action: 'set_visual_node_study_document', scope: creativeScopeWire(),
        node_id: String(node.node_id), document: parsed,
      }, 'Compiling study document…');
    });
    card.querySelector('.creative-study-clear')?.addEventListener('click', () => creativeSend({
      action: 'set_visual_node_study_document', scope: creativeScopeWire(),
      node_id: String(node.node_id), document: null,
    }, 'Clearing study document…'));
    // A node may own more than one route, so every editor in the card is wired
    // to the action naming its own slot. Binding only the first would leave a
    // second select silently submitting nothing.
    const routeEditors = [...card.querySelectorAll('.creative-route-editor')];
    const selfGroupId = creativeScopeWire().scope === 'group' ? creativeScopeWire().group_id : '';
    const slotIndex = (editor) => Number(String(editor.dataset.routeSlot || '').split(':')[1] || 0);
    if (node.kind === 'symmetry') {
      routeEditors.forEach((editor) => wireCreativeRouteEditor(editor, (route) => creativeSend({
        action: 'set_visual_node_symmetry_route', scope: creativeScopeWire(), node_id: String(node.node_id),
        route: { slot: 'image', index: slotIndex(editor), route }, composition_revision: compositionRevision,
      }, 'Preflighting symmetry donor…'), selfGroupId));
      card.querySelectorAll('.creative-motion-route-editor').forEach((editor) => {
        const select = editor.querySelector('.creative-motion-route-source');
        select?.addEventListener('change', () => {
          const value = select.value;
          if (value.startsWith('missing:')) return;
          const layerId = value === 'none' ? null : value;
          if (layerId !== null && !/^(?:[1-9][0-9]*)$/.test(layerId)) return;
          creativeSend({
            action: 'set_visual_node_symmetry_route', scope: creativeScopeWire(), node_id: String(node.node_id),
            route: { slot: 'motion', index: slotIndex(editor), layer_id: layerId }, composition_revision: compositionRevision,
          }, 'Preflighting symmetry motion donor…');
        });
      });
    } else {
      for (const routeEditor of routeEditors) {
        const slot = routeEditor.dataset.routeSlot || '';
        if (node.kind === 'residual' && slot) {
          wireCreativeRouteEditor(routeEditor, (route) => creativeSend({
            action: 'set_visual_node_residual_route', scope: creativeScopeWire(), node_id: String(node.node_id), slot, route, composition_revision: compositionRevision,
          }, `Preflighting residual ${slot} donor…`), selfGroupId);
        } else if (node.kind === 'displace') {
          wireCreativeRouteEditor(routeEditor, (route) => creativeSend({
            action: 'set_visual_node_displace_route', scope: creativeScopeWire(), node_id: String(node.node_id), route, composition_revision: compositionRevision,
          }, 'Preflighting displace donor…'), selfGroupId);
        } else {
          wireCreativeRouteEditor(routeEditor, (route, channel, invert) => creativeSend({
            action: 'set_visual_node_route', scope: creativeScopeWire(), node_id: String(node.node_id), route, channel, invert, composition_revision: compositionRevision,
          }, 'Preflighting image route…'), selfGroupId);
        }
      }
    }
  });
}

function currentCreativeGroup(card) {
  const groupId = String(card?.dataset.groupId || '');
  if (!groupId) return null;
  return (latestCreative?.groups || []).find(
    (candidate) => String(candidate.group_id) === groupId
  ) || null;
}

function sendCurrentGroupTransform(card, action, pending) {
  const group = currentCreativeGroup(card);
  if (!group) {
    creativeSetStatus('Group transform target is no longer present.', true);
    return false;
  }
  return creativeSend({
    ...action,
    group_id: String(group.group_id),
    composition_revision: compositionRevision,
  }, pending);
}

function wireGroupControl(card, param, eventName = 'input') {
  const row = card.querySelector(`[data-group-param="${param}"]`);
  const input = row?.querySelector('input,select');
  if (!input) return;
  input.addEventListener(eventName, () => {
    const group = currentCreativeGroup(card);
    if (!group) {
      creativeSetStatus('Group target is no longer present.', true);
      return;
    }
    let value = input.type === 'checkbox' ? input.checked : (input.type === 'range' ? Number(input.value) : input.value);
    if (param === 'name') {
      value = String(value).replace(/[\u0000-\u001f\u007f]/g, '').trim();
      if (new TextEncoder().encode(value).length > 64) {
        creativeSetStatus('Group names may contain at most 64 UTF-8 bytes.', true);
        return;
      }
    }
    creativeSend({
      action: 'set_composition_group_param', group_id: String(group.group_id), param, value, composition_revision: compositionRevision,
    }, `Applying group ${param.replaceAll('_', ' ')}…`);
    const output = row.querySelector('output');
    if (output && typeof value === 'number') output.textContent = value.toFixed(3);
  });
}

function renderCreativeGroups() {
  if (!creativeGroups) return;
  const groups = latestCreative?.groups || [];
  creativeGroups.innerHTML = groups.length ? '' : '<div class="creative-help">No groups. Direct layers remain on the exact legacy stack.</div>';
  for (const group of groups) {
    const card = document.createElement('article');
    card.className = 'creative-group-card';
    card.dataset.groupId = String(group.group_id);
    card.setAttribute('role', 'listitem');
    const memberOptions = latestLayers.map((layer) => {
      const id = String(layer.layer_id);
      return `<option value="${escapeHtml(id)}" ${(group.member_layer_ids || []).map(String).includes(id) ? 'selected' : ''}>${escapeHtml(creativeLayerLabel(id))}</option>`;
    }).join('');
    const transformControls = groupTransformControlsHtml(group);
    const matte = group.matte;
    const matteBody = matte ? `${creativeRouteEditorHtml(matte.route, matte.channel, matte.invert)}
      ${creativeControlHtml(floatDef('amount', 'Matte amount', 0, 1, 0.001), matte.amount)}
      ${creativeControlHtml(floatDef('threshold', 'Matte threshold', 0, 1, 0.001), matte.threshold)}
      ${creativeControlHtml(floatDef('softness', 'Matte softness', 0, 0.5, 0.001), matte.softness)}
      ${matte.diagnostic ? `<div class="creative-node-diagnostic creative-control-wide">${escapeHtml(matte.diagnostic)}</div>` : ''}` : '<div class="creative-help creative-control-wide">Disabled · no donor is sampled.</div>';
    card.innerHTML = `<div class="creative-group-head">
      <span class="creative-group-title">${escapeHtml(group.name || `Group ${group.group_id}`)}</span>
      <span class="creative-node-id">#${escapeHtml(group.group_id)}</span>
      <button class="creative-group-rack" type="button" aria-label="Open ${escapeHtml(group.name || 'group')} Collision Rack">Rack</button>
      <button class="creative-group-remove" type="button" aria-label="Remove ${escapeHtml(group.name || 'group')}">Ungroup</button>
    </div>
    <div class="creative-group-controls">
      <label class="creative-control creative-control-wide" data-group-param="name"><span>Name</span><input maxlength="64" value="${escapeHtml(group.name || '')}"><output></output></label>
      <label class="creative-control" data-group-param="opacity"><span>Opacity</span><input type="range" min="0" max="1" step="0.001" value="${Number(group.opacity)}"><output>${Number(group.opacity).toFixed(3)}</output></label>
      <label class="creative-control" data-group-param="bus"><span>Bus</span><select>${creativeOptionHtml([['program', 'Program'], ['a', 'A'], ['b', 'B']], group.bus)}</select><output></output></label>
      <label class="creative-control" data-group-param="solo"><span>Solo</span><input type="checkbox" ${group.solo ? 'checked' : ''}><output>${group.solo ? 'on' : 'off'}</output></label>
      <label class="creative-control" data-group-param="bypass"><span>Processing bypass</span><input type="checkbox" ${group.bypass ? 'checked' : ''}><output>${group.bypass ? 'on' : 'off'}</output></label>
      <label class="creative-control creative-control-wide creative-group-members"><span>Members</span><select multiple size="3">${memberOptions}</select><output></output></label>
    </div>
    <details><summary class="creative-subtitle">GROUP TRANSFORM</summary><div class="creative-group-transform spatial-transform-body" role="region" aria-label="${escapeHtml(group.name || `Group ${group.group_id}`)} (#${escapeHtml(group.group_id)}) transform">${transformControls}</div></details>
    <details><summary class="creative-subtitle">GROUP MATTE</summary><label class="creative-control"><span>Enabled</span><input class="creative-group-matte-enabled" type="checkbox" ${matte ? 'checked' : ''}><output>${matte ? 'on' : 'off'}</output></label><div class="creative-matte-controls">${matteBody}</div></details>`;
    creativeGroups.appendChild(card);
    card.querySelector('.creative-group-rack').addEventListener('click', () => {
      creativeRackScope.value = `group:${group.group_id}`;
      creativeStructureKey = '';
      syncCreative(latestCreative);
      document.getElementById('creative-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    card.querySelector('.creative-group-remove').addEventListener('click', () => creativeSend({ action: 'remove_composition_group', group_id: String(group.group_id), composition_revision: compositionRevision }, 'Ungrouping without deleting member layers…'));
    wireGroupControl(card, 'name', 'change');
    wireGroupControl(card, 'opacity');
    wireGroupControl(card, 'bus', 'change');
    wireGroupControl(card, 'solo', 'change');
    wireGroupControl(card, 'bypass', 'change');
    wireTransformPanel(card.querySelector('.creative-group-transform'), {
      stateKey: `group:${group.group_id}`,
      getTransform: () => currentCreativeGroup(card)?.transform,
      set: (param, value) => sendCurrentGroupTransform(
        card,
        { action: 'set_group_transform', param, value },
        `Applying group ${param.replaceAll('_', ' ')}…`
      ),
      reset: () => sendCurrentGroupTransform(
        card,
        { action: 'reset_group_transform' },
        'Resetting group transform…'
      ),
      apply: (transform) => sendCurrentGroupTransform(
        card,
        { action: 'apply_group_transform', transform },
        'Applying complete group transform…'
      ),
      targetGizmo: () => sendCurrentGroupTransform(
        card,
        { action: 'target_group_transform_gizmo' },
        'Targeting this group in the preview gizmo for the current control session…'
      ),
    });
    const memberSelect = card.querySelector('.creative-group-members select');
    memberSelect.addEventListener('change', () => creativeSend({
      action: 'set_composition_group_members', group_id: String(group.group_id), member_layer_ids: [...memberSelect.selectedOptions].map((option) => option.value), composition_revision: compositionRevision,
    }, 'Validating contiguous group membership…'));
    const enabled = card.querySelector('.creative-group-matte-enabled');
    enabled.addEventListener('change', () => creativeSend({
      action: 'set_composition_group_matte_route', group_id: String(group.group_id), route: enabled.checked ? creativeRouteFromToken('one_below', 'current_frame') : null, channel: matte?.channel || 'alpha', invert: matte?.invert || false, composition_revision: compositionRevision,
    }, enabled.checked ? 'Enabling group matte…' : 'Disabling group matte…'));
    const routeEditor = card.querySelector('.creative-route-editor');
    if (routeEditor) wireCreativeRouteEditor(routeEditor, (route, channel, invert) => creativeSend({
      action: 'set_composition_group_matte_route', group_id: String(group.group_id), route, channel, invert, composition_revision: compositionRevision,
    }, 'Preflighting group matte route…'), String(group.group_id));
    for (const param of ['amount', 'threshold', 'softness']) {
      const row = card.querySelector(`.creative-matte-controls [data-creative-param="${param}"]`);
      const input = row?.querySelector('input');
      input?.addEventListener('input', () => {
        const value = Number(input.value);
        row.querySelector('output').textContent = value.toFixed(3);
        creativeSend({ action: 'set_composition_group_matte_param', group_id: String(group.group_id), param, value, composition_revision: compositionRevision }, `Applying group matte ${param}…`);
      });
    }
  }
}

function renderCreativeRoot() {
  if (!creativeRoot) return;
  const root = latestCreative?.root || [];
  creativeRoot.innerHTML = root.length ? '' : '<div class="creative-help">Empty composition root.</div>';
  root.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = 'creative-root-item';
    row.setAttribute('role', 'listitem');
    const isLayer = item.kind === 'layer';
    const id = isLayer ? String(item.layer_id) : String(item.group_id);
    const group = !isLayer ? (latestCreative.groups || []).find((candidate) => String(candidate.group_id) === id) : null;
    row.dataset.rootKind = item.kind;
    row.dataset.rootId = id;
    row.innerHTML = `<span class="creative-node-id">${index + 1}</span><span class="creative-root-label">${escapeHtml(isLayer ? creativeLayerLabel(id) : `Group · ${group?.name || id}`)}</span>
      ${isLayer ? `<label>Bus <select class="creative-root-bus">${creativeOptionHtml([['program', 'Program'], ['a', 'A'], ['b', 'B']], item.bus)}</select></label>` : ''}
      <button class="creative-root-up" type="button" ${index === 0 ? 'disabled' : ''} aria-label="Move root item backward">↑</button>
      <button class="creative-root-down" type="button" ${index + 1 === root.length ? 'disabled' : ''} aria-label="Move root item forward">↓</button>`;
    creativeRoot.appendChild(row);
    const move = (to) => creativeSend({ action: 'move_composition_root_item', item, to, composition_revision: compositionRevision }, 'Moving composition root item…');
    row.querySelector('.creative-root-up').addEventListener('click', () => move(index - 1));
    row.querySelector('.creative-root-down').addEventListener('click', () => move(index + 1));
    row.querySelector('.creative-root-bus')?.addEventListener('change', (event) => creativeSend({ action: 'set_composition_layer_bus', layer_id: id, bus: event.currentTarget.value, composition_revision: compositionRevision }, 'Assigning layer bus…'));
  });
}

function creativeSyncRow(row, value) {
  if (!row) return;
  const input = row.querySelector('input,select');
  if (!input || !canSync(input)) return;
  const component = row.dataset.component;
  const scalar = component === undefined ? value : value?.[Number(component)];
  if (input.type === 'checkbox') input.checked = !!scalar;
  else input.value = scalar ?? '';
  const output = row.querySelector('output');
  if (output) {
    if (input.type === 'checkbox') output.textContent = input.checked ? 'on' : 'off';
    else if (typeof scalar === 'number') output.textContent = scalar.toFixed(3);
  }
}

function syncCreativeRenderedValues() {
  if (!latestCreative) return;
  if (creativeBusCrossfade && canSync(creativeBusCrossfade)) creativeBusCrossfade.value = latestCreative.bus_crossfade ?? 0.5;
  if (creativeBusCrossfadeValue) creativeBusCrossfadeValue.textContent = Number(latestCreative.bus_crossfade ?? 0.5).toFixed(2);
  syncBusMixer(latestCreative.mixer);
  const rack = creativeScopeRack();
  for (const node of rack?.nodes || []) {
    const card = creativeRackNodes?.querySelector(`[data-node-id="${node.node_id}"]`);
    if (!card) continue;
    for (const param of ['enabled', 'wet', 'blend']) creativeSyncRow(card.querySelector(`[data-creative-param="${param}"]`), creativeNodeValue(node, param));
    for (const def of creativeNodeVisibleDefs(node)) {
      card.querySelectorAll(`[data-creative-param="${def.key}"]`).forEach((row) => creativeSyncRow(row, creativeNodeValue(node, def.key)));
    }
  }
  for (const group of latestCreative.groups || []) {
    const card = creativeGroups?.querySelector(`[data-group-id="${group.group_id}"]`);
    if (!card) continue;
    for (const [param, value] of [['name', group.name], ['opacity', group.opacity], ['bus', group.bus], ['solo', group.solo], ['bypass', group.bypass]]) {
      creativeSyncRow(card.querySelector(`[data-group-param="${param}"]`), value);
    }
    syncTransformPanel(card.querySelector('.creative-group-transform'), group.transform);
    if (group.matte) {
      for (const param of ['amount', 'threshold', 'softness']) creativeSyncRow(card.querySelector(`.creative-matte-controls [data-creative-param="${param}"]`), group.matte[param]);
    }
  }
  (latestCreative.root || []).forEach((item) => {
    if (item.kind !== 'layer') return;
    const select = creativeRoot?.querySelector(`[data-root-kind="layer"][data-root-id="${item.layer_id}"] .creative-root-bus`);
    if (select && canSync(select)) select.value = item.bus;
  });
}

function creativeRackStructure(rack) {
  return (rack?.nodes || []).map((node) => ({
    id: String(node.node_id), kind: node.kind, variant: node.params?.variant || '',
    // A Displace donor is structural for the same reason an image-mask route
    // is: changing it must rebuild the card's route editor, not just sync values.
    route: node.kind === 'displace'
      ? node.params?.donor_tap || null
      : node.kind === 'residual'
        ? node.params?.structure_tap || null
        : (node.params?.variant === 'image' ? node.params.image_tap : null),
    // Residual's second route needs its own fingerprint slot, or rerouting the
    // detail input alone would leave a stale select submitting the old token.
    detailRoute: node.kind === 'residual' ? node.params?.detail_tap || null : null,
    // All four Symmetry slots are structural, by slot index. Folding only the
    // first one in would leave a stale editor that then resubmits the old token.
    symmetryRoutes: node.kind === 'symmetry'
      ? [
        node.params?.symmetry_donor0_tap || null,
        node.params?.symmetry_donor1_tap || null,
        node.params?.symmetry_motion0_donor || null,
        node.params?.symmetry_motion1_donor || null,
      ]
      : null,
    channel: node.params?.image_channel || '', invert: !!node.params?.image_invert,
  }));
}

function creativeSnapshotStructure(creative, selectedScope) {
  return JSON.stringify({
    selectedScope,
    master: creativeRackStructure(creative.master_rack),
    layers: (creative.layer_racks || []).map(([id, rack]) => [String(id), creativeRackStructure(rack)]),
    groups: (creative.groups || []).map((group) => ({
      id: String(group.group_id), name: group.name || '', members: (group.member_layer_ids || []).map(String), rack: creativeRackStructure(group.rack),
      matte: group.matte ? { route: group.matte.route, channel: group.matte.channel, invert: group.matte.invert } : null,
    })),
    root: creative.root || [],
    layerLabels: latestLayers.map((layer) => [String(layer.layer_id), layer.filename || layer.source_kind || '']),
  });
}

function syncCreative(creative) {
  latestCreative = creative || { master_rack: { nodes: [] }, layer_racks: [], groups: [], root: [], bus_crossfade: 0.5 };
  const liveGroupTransformKeys = new Set(
    (latestCreative.groups || []).map((group) => `group:${group.group_id}`)
  );
  for (const key of spatialTransformUiState.keys()) {
    if (String(key).startsWith('group:') && !liveGroupTransformKeys.has(key)) {
      spatialTransformUiState.delete(key);
    }
  }
  if (creativeRevision) creativeRevision.textContent = `composition ${compositionRevision}`;
  const currentScope = creativeRackScope?.value || 'master';
  const scopeOptions = [['master', 'Master']];
  for (const [layerId] of latestCreative.layer_racks || []) scopeOptions.push([`layer:${layerId}`, creativeLayerLabel(layerId)]);
  for (const group of latestCreative.groups || []) scopeOptions.push([`group:${group.group_id}`, `Group · ${group.name || group.group_id}`]);
  const nextScope = scopeOptions.some(([value]) => value === currentScope) ? currentScope : 'master';
  if (creativeRackScope && (creativeRackScope.options.length !== scopeOptions.length || [...creativeRackScope.options].some((option, index) => option.value !== scopeOptions[index]?.[0] || option.textContent !== scopeOptions[index]?.[1]))) {
    creativeRackScope.innerHTML = creativeOptionHtml(scopeOptions, nextScope);
  }
  const availableLayers = latestLayers.map((layer) => [String(layer.layer_id), creativeLayerLabel(layer.layer_id)]);
  const availableLayerKey = JSON.stringify(availableLayers);
  if (creativeGroupMembers && !controlIsBusy(creativeGroupMembers) && creativeGroupMembers.dataset.optionsKey !== availableLayerKey) {
    creativeGroupMembers.innerHTML = creativeOptionHtml(availableLayers, '');
    creativeGroupMembers.dataset.optionsKey = availableLayerKey;
  }
  if (rerollGroup && !controlIsBusy(rerollGroup)) {
    const selected = rerollGroup.value;
    const options = (latestCreative.groups || []).map((group) => [String(group.group_id), group.name || `Group ${group.group_id}`]);
    const optionsKey = JSON.stringify(options);
    if (rerollGroup.dataset.optionsKey !== optionsKey) {
      rerollGroup.innerHTML = creativeOptionHtml(options, options.some(([value]) => value === selected) ? selected : options[0]?.[0] || '');
      rerollGroup.dataset.optionsKey = optionsKey;
    }
  }
  const structure = creativeSnapshotStructure(latestCreative, nextScope);
  if (structure !== creativeStructureKey) {
    creativeStructureKey = structure;
    renderCreativeRack();
    renderCreativeGroups();
    renderCreativeRoot();
    if (!latestCreative.groups?.length && (latestCreative.master_rack?.nodes || []).every((node, index) => ['legacy_canonical', 'legacy_temporal'][index] === node.kind)) {
      creativeSetStatus('Legacy rack · exact compatibility path');
    }
  }
  if (latestCreative.status) creativeSetStatus(latestCreative.status, /reject|error|invalid|cycle|budget|missing|stale/i.test(latestCreative.status));
  syncCreativeRenderedValues();
}

creativeRackScope?.addEventListener('change', () => {
  creativeStructureKey = '';
  syncCreative(latestCreative);
});

creativeNodeAdd?.addEventListener('click', () => {
  const rack = creativeScopeRack();
  if (!rack || !creativeNodeKind?.value) return;
  creativeSend({ action: 'insert_visual_node', scope: creativeScopeWire(), index: rack.nodes?.length || 0, node_kind: creativeNodeKind.value, composition_revision: compositionRevision }, `Inserting ${CREATIVE_NODE_INFO[creativeNodeKind.value]?.label || creativeNodeKind.value}…`);
});

creativeBusCrossfade?.addEventListener('input', () => {
  const value = Number(creativeBusCrossfade.value);
  creativeBusCrossfadeValue.textContent = value.toFixed(2);
  creativeSend({ action: 'set_composition_bus_crossfade', value }, 'Crossfading A / B…');
});

// --- B8 bus mixer: wipes, blend meet, dirty mixer, edge melt. Every control
// sends the coalescible set_composition_bus_mix value action. ---
const BUS_MIX_SNAPSHOT_PATHS = Object.freeze({
  wipe_soft: (m) => m.mix?.soft, wipe_x: (m) => m.mix?.origin_x, wipe_y: (m) => m.mix?.origin_y,
  wipe_detail: (m) => m.mix?.detail, wipe_border: (m) => m.mix?.border,
  dirt: (m) => m.dirt?.dirt, dirt_rate: (m) => m.dirt?.rate, dirt_drop: (m) => m.dirt?.drop,
  dirt_cut: (m) => m.dirt?.cut, dirt_knock: (m) => m.dirt?.knock, dirt_noise: (m) => m.dirt?.noise,
  melt: (m) => m.melt?.melt, melt_width: (m) => m.melt?.width, melt_hold: (m) => m.melt?.hold,
  melt_swirl: (m) => m.melt?.swirl, melt_chroma: (m) => m.melt?.chroma, melt_creep: (m) => m.melt?.creep,
});
const busMixSliders = Array.from(document.querySelectorAll('[data-bus-mix]'));
const busMixPattern = document.getElementById('bus-mix-wipe-pattern');
const busMixBlend = document.getElementById('bus-mix-blend');
const busMixBorderColor = document.getElementById('bus-mix-wipe-border-color');
const busMixInvert = document.getElementById('bus-mix-wipe-invert');
const busMixRep = document.getElementById('bus-mix-wipe-rep');
if (busMixBlend) {
  busMixBlend.innerHTML = LAYER_BLEND_MODES
    .filter((mode) => mode.key !== 'alpha_cut')
    .map((mode) => `<option value="${escapeHtml(mode.key)}">${escapeHtml(mode.label)}</option>`)
    .join('');
}
function busMixSend(param, value) {
  creativeSend({ action: 'set_composition_bus_mix', param, value }, `Bus mixer ${param}…`);
}
function syncBusMixer(mixer) {
  if (!mixer) return;
  for (const slider of busMixSliders) {
    const value = BUS_MIX_SNAPSHOT_PATHS[slider.dataset.busMix]?.(mixer);
    if (typeof value !== 'number') continue;
    if (canSync(slider)) slider.value = String(value);
    const output = slider.parentElement?.querySelector('output');
    if (output) output.textContent = value.toFixed(2);
  }
  if (busMixPattern && canSync(busMixPattern)) busMixPattern.value = mixer.mix?.pattern ?? 'dissolve';
  if (busMixBlend && canSync(busMixBlend)) busMixBlend.value = mixer.mix?.blend ?? 'normal';
  if (busMixBorderColor && canSync(busMixBorderColor)) busMixBorderColor.value = mixer.mix?.border_color ?? 'white';
  if (busMixInvert && canSync(busMixInvert)) busMixInvert.checked = Boolean(mixer.mix?.invert);
  if (busMixRep && canSync(busMixRep)) busMixRep.value = String(mixer.mix?.rep ?? 1);
}
for (const slider of busMixSliders) {
  const output = slider.parentElement?.querySelector('output');
  slider.addEventListener('input', () => {
    const value = Number(slider.value);
    if (output) output.textContent = value.toFixed(2);
    busMixSend(slider.dataset.busMix, value);
  });
  slider.addEventListener('dblclick', () => {
    const value = Number(slider.dataset.default ?? 0);
    slider.value = String(value);
    if (output) output.textContent = value.toFixed(2);
    busMixSend(slider.dataset.busMix, value);
  });
}
busMixPattern?.addEventListener('change', () => busMixSend('wipe_pattern', busMixPattern.value));
busMixBlend?.addEventListener('change', () => busMixSend('blend', busMixBlend.value));
busMixBorderColor?.addEventListener('change', () => busMixSend('wipe_border_color', busMixBorderColor.value));
busMixInvert?.addEventListener('change', () => busMixSend('wipe_invert', busMixInvert.checked));
busMixRep?.addEventListener('change', () => busMixSend('wipe_rep', Number(busMixRep.value)));

creativeGroupCreate?.addEventListener('submit', (event) => {
  event.preventDefault();
  const name = (creativeGroupName?.value || '').replace(/[\u0000-\u001f\u007f]/g, '').trim();
  if (new TextEncoder().encode(name).length > 64) {
    creativeSetStatus('Group names may contain at most 64 UTF-8 bytes.', true);
    return;
  }
  const memberLayerIds = [...(creativeGroupMembers?.selectedOptions || [])].map((option) => option.value);
  if (creativeSend({ action: 'create_composition_group', name, member_layer_ids: memberLayerIds, root_index: latestCreative?.root?.length || 0, composition_revision: compositionRevision }, 'Creating composition group…')) {
    creativeGroupName.value = '';
  }
});

creativeGroupGizmoRelease?.addEventListener('click', () => {
  const sent = creativeSend({
    action: 'clear_group_transform_gizmo_target',
    composition_revision: compositionRevision,
  }, 'Group gizmo target release requested for this control session; the current master or layer scope can recover.');
  if (creativeGroupGizmoReleaseStatus) {
    creativeGroupGizmoReleaseStatus.textContent = sent
      ? 'Release requested. The preview gizmo can recover the current master or layer scope for this control session.'
      : 'Control connection is offline; the group gizmo target was not released.';
    creativeGroupGizmoReleaseStatus.classList.toggle('error', !sent);
  }
});

function layerBlendOptionsHtml(selected) {
  const selectedKey = layerBlendModeInfo(selected).key;
  return LAYER_BLEND_MODES.map((mode) =>
    `<option value="${mode.key}" title="${escapeHtml(mode.description)}" ${mode.key === selectedKey ? 'selected' : ''}>${escapeHtml(mode.label)}</option>`
  ).join('');
}

function syncLayerBlendDescription(row, key) {
  const mode = layerBlendModeInfo(key);
  const select = row?.querySelector('select');
  const description = row?.querySelector('.blend-mode-description');
  if (select) select.title = layerBlendTitle(mode.key);
  if (description) description.textContent = mode.description;
}

function validSceneName(name) {
  return typeof name === 'string'
    && name === name.trim()
    && !/[\u0000-\u001f\u007f]/.test(name)
    && new TextEncoder().encode(name).length <= 128;
}

function sceneNameFrom(control) {
  const name = String(control?.value || '').trim();
  if (!validSceneName(name)) {
    sceneStatus.textContent = 'Scene names must be at most 128 UTF-8 bytes with no control characters.';
    sceneStatus.classList.add('error');
    control?.focus();
    return null;
  }
  return name;
}

function sceneTriggerModeFrom(control) {
  const mode = String(control?.value || '');
  return ['immediate', 'next_beat', 'next_bar'].includes(mode) ? mode : null;
}

function layerSelector(layer, index) {
  return { index, layer_id: layer?.layer_id || null };
}

function currentLayerContext(card, fallbackLayer, fallbackIndex) {
  const currentLayer = card?._layerState || fallbackLayer;
  const currentIndex = Number.parseInt(card?.dataset.index, 10);
  return {
    layer: currentLayer,
    index: Number.isInteger(currentIndex) ? currentIndex : fallbackIndex,
  };
}

function currentLayerSelector(card, fallbackLayer, fallbackIndex) {
  const current = currentLayerContext(card, fallbackLayer, fallbackIndex);
  return layerSelector(current.layer, current.index);
}

function stableLayerId(layer) {
  const id = String(layer?.layer_id || '');
  return /^(?:[1-9][0-9]*)$/.test(id) ? id : null;
}

function currentStableLayerId(card, fallbackLayer, fallbackIndex) {
  return stableLayerId(currentLayerContext(card, fallbackLayer, fallbackIndex).layer);
}

// Attach missing programmatic names without requiring every compact visual
// label to carry a bespoke id/for pair.
document.querySelectorAll('.param-row').forEach((row) => {
  const label = row.querySelector(':scope > label');
  if (!label) return;
  const groupLabel = row.closest('.fx-group')?.querySelector(':scope > .fx-group-header .group-label')?.textContent?.trim();
  row.querySelectorAll('input:not([aria-label]),select:not([aria-label])').forEach((control) => {
    control.setAttribute('aria-label', [groupLabel, label.textContent.trim()].filter(Boolean).join(' '));
  });
});

// --- Universal range numeric entry ------------------------------------
// Every range keeps its native slider for performance gestures and gains an
// editable value beside it. Committing the editor dispatches the slider's
// existing input event, so there is exactly one action path per control.

const rangeBindings = new WeakMap();
const activeRangeBindings = new Set();
let rangeEditorSequence = 0;

function decimalPlaces(value) {
  const text = String(value).toLowerCase();
  const [coefficient, exponentText] = text.split('e');
  const exponent = Number(exponentText || 0);
  const fraction = (coefficient.split('.')[1] || '').length;
  return Math.max(0, fraction - exponent);
}

function rangeValuePrecision(step, base = 0) {
  return Number.isFinite(step)
    ? Math.min(12, Math.max(decimalPlaces(step), decimalPlaces(base)))
    : 3;
}

function rangeBounds(slider) {
  const min = Number.parseFloat(slider.min);
  const max = Number.parseFloat(slider.max);
  const step = slider.step === 'any' ? NaN : Number.parseFloat(slider.step);
  return {
    min: Number.isFinite(min) ? min : -Infinity,
    max: Number.isFinite(max) ? max : Infinity,
    step: Number.isFinite(step) && step > 0 ? step : NaN,
  };
}

function parseRangeDraft(text) {
  const canonical = String(text)
    .trim()
    .replace(/\u2212/g, '-')
    .replace(',', '.');
  if (!canonical) return NaN;
  return Number(canonical);
}

function normalizeRangeValue(slider, rawValue) {
  let value = typeof rawValue === 'number' ? rawValue : parseRangeDraft(rawValue);
  if (!Number.isFinite(value)) return null;
  const { min, max, step } = rangeBounds(slider);
  value = Math.min(max, Math.max(min, value));
  if (Number.isFinite(step)) {
    const base = Number.isFinite(min) ? min : 0;
    value = base + Math.round((value - base) / step) * step;
    value = Math.min(max, Math.max(min, value));
    value = Number(value.toFixed(rangeValuePrecision(step, base)));
    // Decimal formatting must never round the exact crop ceiling up to 1.
    value = Math.min(max, Math.max(min, value));
  }
  return Number.isFinite(value) ? value : null;
}

function formatRangeValue(slider, rawValue) {
  const value = normalizeRangeValue(slider, rawValue);
  if (value === null) return '';
  const { min, step } = rangeBounds(slider);
  const base = Number.isFinite(min) ? min : 0;
  return value.toFixed(rangeValuePrecision(step, base));
}

function rangeControlLabel(slider) {
  const explicit = slider.getAttribute('aria-label');
  if (explicit) return explicit;
  const row = slider.closest('.param-row');
  const label = row?.querySelector(':scope > label')?.textContent?.trim();
  const layer = slider.closest('.layer-card')?.querySelector('.layer-title')?.textContent?.trim();
  const group = slider.closest('.fx-group')?.querySelector(':scope > .fx-group-header .group-label')?.textContent?.trim();
  return [layer || group, label, 'value'].filter(Boolean).join(' ') || 'Range value';
}

function setRangeValidation(binding, message = '') {
  binding.editor.classList.toggle('range-value-invalid', !!message);
  binding.editor.setAttribute('aria-invalid', String(!!message));
  binding.editor.title = message || binding.help;
}

function writeRangeEditor(binding, rawValue) {
  const value = normalizeRangeValue(binding.slider, rawValue);
  if (value === null) return;
  const textValue = formatRangeValue(binding.slider, value);
  const ariaValue = String(value);
  if (binding.editor.textContent !== textValue) binding.editor.textContent = textValue;
  if (binding.editor.getAttribute('aria-valuenow') !== ariaValue) {
    binding.editor.setAttribute('aria-valuenow', ariaValue);
  }
  if (binding.editor.getAttribute('aria-valuetext') !== textValue) {
    binding.editor.setAttribute('aria-valuetext', textValue);
  }
}

function syncRangeEditorState(slider) {
  const binding = rangeBindings.get(slider);
  if (!binding) return;
  const disabled = !!slider.disabled;
  if (binding.disabled === disabled) return;
  binding.disabled = disabled;
  binding.editor.setAttribute('contenteditable', String(!disabled));
  binding.editor.setAttribute('aria-disabled', String(disabled));
  binding.editor.tabIndex = disabled ? -1 : 0;
  binding.editor.classList.toggle('disabled', disabled);
}

function commitRangeEditor(binding) {
  if (binding.slider.disabled) return false;
  const value = normalizeRangeValue(binding.slider, binding.editor.textContent);
  if (value === null) {
    const { min, max, step } = rangeBounds(binding.slider);
    const lowerLimit = Number.isFinite(min) ? `minimum ${min}` : 'no minimum';
    const upperLimit = Number.isFinite(max) ? `maximum ${max}` : 'no maximum';
    const limits = `${lowerLimit}; ${upperLimit}`;
    const increment = Number.isFinite(step) ? ` in steps of ${step}` : '';
    writeRangeEditor(binding, binding.slider.value);
    setRangeValidation(binding, `Invalid number; enter ${limits}${increment}. Previous value restored.`);
    return false;
  }
  setRangeValidation(binding);
  binding.slider.value = String(value);
  binding.committing = true;
  binding.slider.dispatchEvent(new Event('input', { bubbles: true }));
  binding.committing = false;
  // Existing handlers may use a coarser visual formatter. Restore the exact
  // step precision after they have dispatched their normal action.
  writeRangeEditor(binding, value);
  return true;
}

function cancelRangeEditor(binding) {
  setRangeValidation(binding);
  writeRangeEditor(binding, binding.slider.value);
}

function bindRangeEditor(slider) {
  if (!slider || rangeBindings.has(slider)) return rangeBindings.get(slider);

  let editor = slider.nextElementSibling;
  if (!editor?.matches('.value, .routing-depth-val, [data-range-editor]')) {
    editor = document.createElement('span');
    const wrapper = document.createElement('span');
    wrapper.className = 'range-editor-wrap';
    slider.before(wrapper);
    wrapper.append(slider, editor);
  }

  editor.classList.add('value', 'range-value');
  editor.dataset.rangeEditor = 'true';
  editor.id ||= `range-value-${++rangeEditorSequence}`;
  editor.setAttribute('role', 'spinbutton');
  const { min, max, step } = rangeBounds(slider);
  // Some mobile decimal keyboards omit a minus key. Preserve direct signed
  // entry for controls whose declared range crosses below zero.
  editor.setAttribute('inputmode', min < 0 ? 'text' : 'decimal');
  editor.setAttribute('enterkeyhint', 'done');
  editor.setAttribute('spellcheck', 'false');
  editor.setAttribute('aria-keyshortcuts', 'ArrowUp ArrowDown PageUp PageDown Enter Escape');
  const label = rangeControlLabel(slider);
  if (!slider.hasAttribute('aria-label')) slider.setAttribute('aria-label', label);
  editor.setAttribute('aria-label', `${label} numeric entry`);
  slider.setAttribute('aria-controls', editor.id);

  if (Number.isFinite(min)) editor.setAttribute('aria-valuemin', String(min));
  if (Number.isFinite(max)) editor.setAttribute('aria-valuemax', String(max));
  const help = [
    Number.isFinite(min) && Number.isFinite(max) ? `${min} to ${max}` : '',
    Number.isFinite(step) ? `step ${step}` : '',
    'Enter or blur commits; Escape cancels',
  ].filter(Boolean).join('; ');
  const binding = { slider, editor, help, committing: false, suppressBlurCommit: false, disabled: null };
  rangeBindings.set(slider, binding);
  activeRangeBindings.add(binding);
  rangeControlPeers.set(slider, editor);
  rangeControlPeers.set(editor, slider);
  setRangeValidation(binding);
  writeRangeEditor(binding, slider.value);
  syncRangeEditorState(slider);

  slider.addEventListener('input', () => {
    if (document.activeElement !== editor || binding.committing) {
      setRangeValidation(binding);
      writeRangeEditor(binding, slider.value);
    }
  });
  editor.addEventListener('focus', () => {
    setRangeValidation(binding);
    requestAnimationFrame(() => {
      if (document.activeElement !== editor) return;
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(editor);
      selection.removeAllRanges();
      selection.addRange(range);
    });
  });
  editor.addEventListener('input', () => setRangeValidation(binding));
  editor.addEventListener('paste', (event) => {
    // contenteditable otherwise accepts rich HTML. Insert only the clipboard's
    // plain-text representation so pasted markup can never become live DOM.
    event.preventDefault();
    const plainText = event.clipboardData?.getData('text/plain') || '';
    const selection = window.getSelection();
    if (!selection?.rangeCount) {
      editor.textContent += plainText;
    } else {
      const range = selection.getRangeAt(0);
      range.deleteContents();
      const textNode = document.createTextNode(plainText);
      range.insertNode(textNode);
      range.setStartAfter(textNode);
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);
    }
    setRangeValidation(binding);
  });
  editor.addEventListener('keydown', (event) => {
    const direction = event.key === 'ArrowUp' || event.key === 'PageUp'
      ? 1
      : event.key === 'ArrowDown' || event.key === 'PageDown'
        ? -1
        : 0;
    if (direction) {
      event.preventDefault();
      const { step } = rangeBounds(slider);
      const increment = Number.isFinite(step) ? step : 1;
      const multiplier = event.key.startsWith('Page') ? 10 : 1;
      const draft = normalizeRangeValue(slider, editor.textContent);
      const current = draft ?? normalizeRangeValue(slider, slider.value) ?? 0;
      const next = normalizeRangeValue(slider, current + direction * increment * multiplier);
      if (next !== null) {
        editor.textContent = String(next);
        commitRangeEditor(binding);
      }
    } else if (event.key === 'Enter') {
      event.preventDefault();
      if (commitRangeEditor(binding)) {
        binding.suppressBlurCommit = true;
        editor.blur();
      }
    } else if (event.key === 'Escape') {
      event.preventDefault();
      cancelRangeEditor(binding);
      binding.suppressBlurCommit = true;
      editor.blur();
    }
  });
  editor.addEventListener('blur', () => {
    if (binding.suppressBlurCommit) {
      binding.suppressBlurCommit = false;
      return;
    }
    commitRangeEditor(binding);
  });
  return binding;
}

function rangesWithin(root) {
  const ranges = [];
  if (root?.matches?.('input[type="range"]')) ranges.push(root);
  root?.querySelectorAll?.('input[type="range"]').forEach((slider) => ranges.push(slider));
  return ranges;
}

// Master and layer effect defaults. Hoisted to module scope so the
// double-click reset and the B15 CHANGED filter read one table rather
// than two copies that could disagree about what 'default' means.
const MASTER_PARAM_DEFAULTS = Object.freeze({
      pixelate: 1, rgb_split: 0, hue_shift: 0, saturation: 0,
      downsample: 1,
      shift_amount: 0, shift_block_size: 8, shift_density: 0.5, shift_speed: 3,
      brightness: 0, contrast: 0, posterize: 0, grain_intensity: 0,
      grain_size: 1, vignette: 0, color_drift: 0, breathe_scale: 0,
      breathe_rotation: 0, breathe_position: 0,
      cellular_amount: 0, cellular_scale: 10, cellular_warp: 0.35,
      cellular_speed: 0.25, cellular_gap_amount: 0,
      cellular_gap_threshold: 0.65, cellular_gap_softness: 0.08,
      key_color_r: 0, key_color_g: 1, key_color_b: 0,
      key_threshold: 0.5, key_softness: 0.1, key_tolerance: 0.15,
      contour: 0, contour_bands: 10, contour_width: 1.2, contour_hue: 0,
      contour_fill: 0.25, flatten: 0, flatten_levels: 5, contour_dither: 0,
      solarize: 0, negative: 0, colourpass: 0, colourpass_hue: 0,
      colourpass_width: 0.25, edge_amount: 0, edge_hue: 0, emboss: 0,
      emboss_angle: 45, halftone: 0, halftone_pitch: 0.4, halftone_angle: 0,
      moire: 0, moire_freq: 0.4, row_smear: 0, bitcrush: 0,
      bitcrush_levels: 2, bitcrush_dither: 1, multi_grid_x: 1, multi_grid_y: 1,
      barrel: 0, chroma_aberration: 0, anamorphic_streak: 0,
    });

// VHS block defaults, shared by the reset path and the CHANGED filter.
const NTSC_PARAM_DEFAULTS = Object.freeze({
      head_switching_height: 8, tracking_noise_height: 24,
      edge_wave_speed: 0.5,
      // Bipolar controls whose default is zero while their minimum is not.
      // Without these a double-click reset authors the extreme.
      head_switching_shift: 0, composite_sharpening: 0,
    });

// Temporal defaults, shared by the reset path and the CHANGED filter.
// Any control whose default is not its slider minimum must appear here;
// `every_temporal_default_agrees_between_the_markup_and_the_reset_table`
// checks that against the value spans in index.html.
const TEMPORAL_PARAM_DEFAULTS = Object.freeze({
      fb_zoom: 1,
      fb_saturation: 1,
      fb_gain_r: 1,
      fb_gain_g: 1,
      fb_gain_b: 1,
      fb_drive: 1,
      fb_pivot: 0.5,
      key_threshold: 0.1,
      key_softness: 0.03,
      key_history: 1,
      long_exposure_frames: 12,
      loom_depth: 1,
      loom_scale: 1,
      loom_folds: 1,
      atlas_territories: 8,
      garden_threshold: 0.1,
      garden_softness: 0.03,
      garden_decay: 1,
      score_state_count: 4,
      disp_il_twitter: 0.4,
      disp_phos_r: 0.86,
      disp_phos_g: 1,
      disp_phos_b: 0.66,
      disp_beam_width: 1,
      disp_beam_shape: 0.5,
      disp_mask_dark: 0.5,
      disp_bloom_radius: 0.4,
      melt_width: 0.3,
      melt_hold: 0.6,
      melt_swirl: 0,
      melt_chroma: 0.5,
      melt_creep: 0.35,
      mosh_key_removal: 0.95,
      mosh_hold: 0.25,
      mosh_rate: 0.5,
      mosh_bitrate_starve: 0.35,
      mosh_wipe: 0,
      mosh_smear: 0,
      mosh_trail: 0,
      sync_rate: 0.35,
      sync_spread: 0.25,
      // Every bipolar control below defaults to zero while its slider minimum
      // is negative. An unlisted key falls back to that minimum, so omitting
      // any of these makes a double-click "reset" author a hard negative
      // extreme instead of the neutral value.
      fb_rotate: 0,
      fb_offset_x: 0,
      fb_offset_y: 0,
      fb_hue_rotate: 0,
      slit_angle: 0,
      loom_phase: 0,
      loom_angle: 0,
      // Explicitly zero: the fallback for an unlisted key is the slider's
      // minimum, and this control's minimum is -1, so omitting it would make
      // a double-click reset author full negative bias.
      sync_bias: 0,
    });

function bindRangeEditors(root = document) {
  rangesWithin(root).forEach(bindRangeEditor);
}

function syncRangeEditors(root = document) {
  // Newly inserted controls are normally bound by the observer. Bind an
  // explicitly scoped root for callers that need same-turn availability,
  // then walk the persistent bindings instead of querying hundreds of DOM
  // nodes on every 30 Hz state packet.
  if (root !== document) bindRangeEditors(root);
  for (const binding of activeRangeBindings) {
    const { slider } = binding;
    if (!slider.isConnected) {
      activeRangeBindings.delete(binding);
      continue;
    }
    syncRangeEditorState(slider);
    if (canSync(slider)) writeRangeEditor(binding, slider.value);
  }
}

function resetRangeOnDoubleActivation(el, fallback) {
  const reset = () => {
    el.value = String(fallback);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  el.addEventListener('dblclick', reset);
  let lastTap = 0;
  let tapStart = null;
  el.addEventListener('pointerdown', (e) => {
    if (e.pointerType !== 'mouse') tapStart = { id: e.pointerId, x: e.clientX, y: e.clientY, at: performance.now() };
  });
  el.addEventListener('pointerup', (e) => {
    if (e.pointerType === 'mouse') return;
    const now = performance.now();
    const isTap = tapStart && tapStart.id === e.pointerId && now - tapStart.at < 300 &&
      Math.hypot(e.clientX - tapStart.x, e.clientY - tapStart.y) < 8;
    tapStart = null;
    if (!isTap) {
      lastTap = 0;
      return;
    }
    if (now - lastTap < 350) {
      e.preventDefault();
      reset();
      lastTap = 0;
    } else {
      lastTap = now;
    }
  });
}

// --- Canonical spatial transform controls ------------------------------

const LEGACY_SPATIAL_TRANSFORM = Object.freeze({
  position: Object.freeze([0, 0]),
  scale: Object.freeze([1, 1]),
  anchor: Object.freeze([0.5, 0.5]),
  rotation_deg: 0,
  skew_deg: 0,
  skew_axis_deg: 0,
  fit: 'stretch',
  crop: Object.freeze([0, 0, 0, 0]),
  edge: 'transparent',
  sampling: 'linear',
});

const TRANSFORM_RANGE_SPECS = [
  ['position_x', 'Position X', -4, 4, 0.01, 0],
  ['position_y', 'Position Y', -4, 4, 0.01, 0],
  ['scale_x', 'Scale X', -16, 16, 0.01, 1],
  ['scale_y', 'Scale Y', -16, 16, 0.01, 1],
  ['anchor_x', 'Anchor X', -2, 3, 0.01, 0.5],
  ['anchor_y', 'Anchor Y', -2, 3, 0.01, 0.5],
  ['rotation_deg', 'Rotation °', -180, 180, 0.1, 0],
  ['skew_deg', 'Skew °', -89, 89, 0.1, 0],
  ['skew_axis_deg', 'Skew Axis °', -180, 180, 0.1, 0],
  ['crop_left', 'Crop Left', 0, 0.999755859375, 0.000244140625, 0],
  ['crop_top', 'Crop Top', 0, 0.999755859375, 0.000244140625, 0],
  ['crop_right', 'Crop Right', 0, 0.999755859375, 0.000244140625, 0],
  ['crop_bottom', 'Crop Bottom', 0, 0.999755859375, 0.000244140625, 0],
];

const TRANSFORM_RANGE_DEFAULTS = Object.fromEntries(
  TRANSFORM_RANGE_SPECS.map(([param, , , , , fallback]) => [param, fallback])
);
const spatialTransformUiState = new Map();
let latestMasterTransform = cloneSpatialTransform(LEGACY_SPATIAL_TRANSFORM);
let spatialTransformClipboard = null;

function finiteTransformNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalizeSpatialTransform(raw) {
  const source = raw && typeof raw === 'object' ? raw : {};
  const vector = (value, fallback) => Array.isArray(value) && value.length >= fallback.length
    ? fallback.map((item, index) => finiteTransformNumber(value[index], item))
    : [...fallback];
  const oneOf = (value, choices, fallback) => choices.includes(value) ? value : fallback;
  return {
    position: vector(source.position, LEGACY_SPATIAL_TRANSFORM.position),
    scale: vector(source.scale, LEGACY_SPATIAL_TRANSFORM.scale),
    anchor: vector(source.anchor, LEGACY_SPATIAL_TRANSFORM.anchor),
    rotation_deg: finiteTransformNumber(source.rotation_deg, 0),
    skew_deg: finiteTransformNumber(source.skew_deg, 0),
    skew_axis_deg: finiteTransformNumber(source.skew_axis_deg, 0),
    fit: oneOf(source.fit, ['stretch', 'fit', 'fill', 'native'], 'stretch'),
    crop: vector(source.crop, LEGACY_SPATIAL_TRANSFORM.crop),
    edge: oneOf(source.edge, ['transparent', 'clamp', 'repeat', 'mirror'], 'transparent'),
    sampling: oneOf(source.sampling, ['linear', 'nearest'], 'linear'),
  };
}

function cloneSpatialTransform(transform) {
  const normalized = normalizeSpatialTransform(transform);
  return {
    ...normalized,
    position: [...normalized.position],
    scale: [...normalized.scale],
    anchor: [...normalized.anchor],
    crop: [...normalized.crop],
  };
}

function transformFieldValue(transform, param) {
  const t = normalizeSpatialTransform(transform);
  const fields = {
    position_x: t.position[0], position_y: t.position[1],
    scale_x: t.scale[0], scale_y: t.scale[1],
    anchor_x: t.anchor[0], anchor_y: t.anchor[1],
    rotation_deg: t.rotation_deg, skew_deg: t.skew_deg, skew_axis_deg: t.skew_axis_deg,
    crop_left: t.crop[0], crop_top: t.crop[1], crop_right: t.crop[2], crop_bottom: t.crop[3],
    fit: t.fit, edge: t.edge, sampling: t.sampling,
  };
  return fields[param];
}

function spatialTransformPreset(name) {
  const preset = cloneSpatialTransform(LEGACY_SPATIAL_TRANSFORM);
  if (name === 'center_fit') {
    preset.fit = 'fit';
    preset.edge = 'transparent';
  } else if (name === 'center_fill') {
    preset.fit = 'fill';
    preset.edge = 'transparent';
  } else if (name === 'native') {
    preset.fit = 'native';
    preset.edge = 'transparent';
  } else if (name === 'mirror_x') {
    preset.scale[0] = -1;
  } else if (name === 'mirror_y') {
    preset.scale[1] = -1;
  } else if (name !== 'identity') {
    return null;
  }
  return preset;
}

function transformRowParam(row) {
  return row?.dataset.masterTransform || row?.dataset.layerTransform || row?.dataset.groupTransform || '';
}

function transformRow(panel, param) {
  return panel?.querySelector(
    `[data-master-transform="${param}"], [data-layer-transform="${param}"], [data-group-transform="${param}"]`
  );
}

function updateTransformRangeDisplay(slider) {
  const binding = rangeBindings.get(slider);
  if (binding) {
    writeRangeEditor(binding, slider.value);
    return;
  }
  const value = Number(slider.value);
  const display = slider.nextElementSibling;
  if (display?.classList.contains('value')) {
    display.textContent = formatValue(
      value,
      Number(slider.min),
      Number(slider.max),
      Number(slider.step)
    );
  }
}

function syncTransformPanel(panel, transform) {
  if (!panel) return;
  const normalized = normalizeSpatialTransform(transform);
  panel.querySelectorAll('[data-master-transform], [data-layer-transform], [data-group-transform]').forEach((row) => {
    const param = transformRowParam(row);
    const value = transformFieldValue(normalized, param);
    const control = row.querySelector('input[type="range"],select');
    if (!control || !canSync(control)) return;
    control.value = String(value);
    if (control.matches('input[type="range"]')) updateTransformRangeDisplay(control);
  });
}

function updateTransformPasteButtons() {
  document.querySelectorAll('.transform-paste').forEach((button) => {
    button.disabled = spatialTransformClipboard === null;
    button.setAttribute('aria-disabled', String(button.disabled));
  });
}

function setTransformPanelStatus(panel, message) {
  const status = panel?.querySelector('.transform-status');
  if (status) status.textContent = message;
}

function wireTransformPanel(panel, api) {
  if (!panel || panel.dataset.transformWired === 'true') return;
  panel.dataset.transformWired = 'true';
  const link = panel.querySelector('.transform-scale-link');
  if (api.stateKey) {
    const remembered = spatialTransformUiState.get(api.stateKey);
    link.checked = remembered?.scaleLinked
      ?? Math.abs(transformFieldValue(api.getTransform(), 'scale_x') - transformFieldValue(api.getTransform(), 'scale_y')) < 1e-6;
    link.addEventListener('change', () => {
      spatialTransformUiState.set(api.stateKey, { scaleLinked: link.checked });
    });
  }

  panel.querySelectorAll('[data-master-transform], [data-layer-transform], [data-group-transform]').forEach((row) => {
    const param = transformRowParam(row);
    const control = row.querySelector('input[type="range"],select');
    if (!control) return;
    const eventName = control.matches('input[type="range"]') ? 'input' : 'change';
    control.addEventListener(eventName, () => {
      const value = control.matches('select') ? control.value : Number(control.value);
      if (!api.set(param, value)) {
        setTransformPanelStatus(panel, 'Control connection is offline; transform was not sent.');
        return;
      }
      setTransformPanelStatus(panel, '');
      if (link?.checked && (param === 'scale_x' || param === 'scale_y')) {
        const pairedParam = param === 'scale_x' ? 'scale_y' : 'scale_x';
        const paired = transformRow(panel, pairedParam)?.querySelector('input[type="range"]');
        if (paired) {
          paired.value = String(value);
          updateTransformRangeDisplay(paired);
          api.set(pairedParam, value);
        }
      }
    });
    if (control.matches('input[type="range"]')) {
      resetRangeOnDoubleActivation(control, TRANSFORM_RANGE_DEFAULTS[param]);
    }
  });

  panel.querySelector('.transform-reset')?.addEventListener('click', () => {
    setTransformPanelStatus(
      panel,
      api.reset() ? 'Transform reset requested.' : 'Control connection is offline; reset was not sent.'
    );
  });
  panel.querySelector('.transform-copy')?.addEventListener('click', () => {
    spatialTransformClipboard = cloneSpatialTransform(api.getTransform());
    updateTransformPasteButtons();
    setTransformPanelStatus(panel, 'Transform copied for this control session.');
  });
  panel.querySelector('.transform-paste')?.addEventListener('click', () => {
    if (!spatialTransformClipboard) return;
    setTransformPanelStatus(
      panel,
      api.apply(cloneSpatialTransform(spatialTransformClipboard))
        ? 'Transform paste requested.'
        : 'Control connection is offline; paste was not sent.'
    );
  });
  panel.querySelector('.transform-preset')?.addEventListener('change', (event) => {
    const transform = spatialTransformPreset(event.currentTarget.value);
    event.currentTarget.value = '';
    if (!transform) return;
    setTransformPanelStatus(
      panel,
      api.apply(transform)
        ? 'Transform preset requested.'
        : 'Control connection is offline; preset was not sent.'
      );
  });
  panel.querySelector('.transform-target-gizmo')?.addEventListener('click', () => {
    setTransformPanelStatus(
      panel,
      api.targetGizmo?.()
        ? 'Preview gizmo target requested for this control session.'
        : 'Control connection is offline; the preview gizmo target was not changed.'
    );
  });
  bindRangeEditors(panel);
  syncTransformPanel(panel, api.getTransform());
  updateTransformPasteButtons();
}

function spatialTransformControlsHtml(scope, accessibleLabel) {
  const rowAttribute = (param) => scope === 'group'
    ? `data-group-transform="${param}"`
    : `data-layer-transform="${param}"`;
  const fitAttribute = scope === 'group' ? 'data-group-transform="fit"' : 'data-layer-transform="fit"';
  const edgeAttribute = scope === 'group' ? 'data-group-transform="edge"' : 'data-layer-transform="edge"';
  const samplingAttribute = scope === 'group' ? 'data-group-transform="sampling"' : 'data-layer-transform="sampling"';
  const ariaScope = escapeHtml(accessibleLabel);
  const targetGizmo = scope === 'group'
    ? `<button type="button" class="transform-target-gizmo" aria-label="Target ${ariaScope} in the preview transform gizmo for this control session" title="Session-only preview gizmo target">Target gizmo</button>`
    : '';
  const ranges = TRANSFORM_RANGE_SPECS.map(([param, label, min, max, step, fallback]) => `
    <div class="param-row" ${rowAttribute(param)}>
      <label>${label}</label><input type="range" min="${min}" max="${max}" step="${step}" value="${fallback}" aria-label="${ariaScope} ${label}"><span class="value">${formatValue(fallback, min, max, step)}</span>
    </div>`).join('');
  return `
    <div class="transform-toolbar" role="group" aria-label="${ariaScope} transform commands">
      <button type="button" class="transform-reset" title="Reset to the legacy full-frame identity">Reset</button>
      <button type="button" class="transform-copy">Copy</button>
      <button type="button" class="transform-paste">Paste</button>
      ${targetGizmo}
      <label class="transform-preset-label">Preset
        <select class="transform-preset" aria-label="${ariaScope} transform preset">
          <option value="">Choose…</option><option value="identity">Identity</option><option value="center_fit">Center Fit</option><option value="center_fill">Center Fill</option><option value="native">Native 1:1</option><option value="mirror_x">Mirror X</option><option value="mirror_y">Mirror Y</option>
        </select>
      </label>
    </div>
    ${ranges}
    <div class="param-row toggle-row transform-scale-link-row"><label>Link scale</label><label class="toggle"><input type="checkbox" class="transform-scale-link" aria-label="Link ${ariaScope} scale axes"><span class="toggle-slider"></span></label></div>
    <div class="param-row select-row" ${fitAttribute}><label>Fit</label><select aria-label="${ariaScope} media fit"><option value="stretch">Stretch</option><option value="fit">Fit</option><option value="fill">Fill</option><option value="native">Native</option></select></div>
    <div class="param-row select-row" ${edgeAttribute}><label>Edge</label><select aria-label="${ariaScope} edge mode"><option value="transparent">Transparent</option><option value="clamp">Clamp</option><option value="repeat">Repeat</option><option value="mirror">Mirror</option></select></div>
    <div class="param-row select-row" ${samplingAttribute}><label>Sampling</label><select aria-label="${ariaScope} sampling mode"><option value="linear">Linear</option><option value="nearest">Nearest</option></select></div>
    <div class="audio-status">Position uses composition dimensions; anchor and crop use source fractions. Zero scale fails closed to transparent.</div>
    <div class="transform-status" role="status" aria-live="polite"></div>`;
}

function layerTransformControlsHtml(index) {
  return spatialTransformControlsHtml('layer', `Layer ${index + 1}`);
}

function groupTransformControlsHtml(group) {
  const groupId = String(group?.group_id || '');
  const groupName = String(group?.name || `Group ${groupId}`);
  return spatialTransformControlsHtml('group', `${groupName} (#${groupId})`);
}

const masterTransformPanel = document.getElementById('master-transform-body');
wireTransformPanel(masterTransformPanel, {
  getTransform: () => latestMasterTransform,
  set: (param, value) => sendAction({ action: 'set_master_transform', param, value }),
  reset: () => sendAction({ action: 'reset_master_transform' }),
  apply: (transform) => sendAction({ action: 'apply_master_transform', transform }),
});

function syncMasterTransform(transform) {
  latestMasterTransform = normalizeSpatialTransform(transform);
  syncTransformPanel(masterTransformPanel, latestMasterTransform);
}

// --- Initialize sliders from DOM attributes ---

document.querySelectorAll('.param-row[data-param]').forEach((row) => {
  const param = row.dataset.param;
  const min = parseFloat(row.dataset.min);
  const max = parseFloat(row.dataset.max);
  const step = parseFloat(row.dataset.step);

  const slider = row.querySelector('input[type="range"]');
  const valueEl = row.querySelector('.value');
  const checkbox = row.querySelector('input[type="checkbox"]');
  const select = row.querySelector('select');

  if (slider) {
    slider.min = min;
    slider.max = max;
    slider.step = step;
    const declaredDefault = Number(valueEl?.textContent);
    slider.value = Number.isFinite(declaredDefault) ? declaredDefault : min;

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      sendAction({ action: 'set_param', param, value: v });
    });
    const defaults = MASTER_PARAM_DEFAULTS;
    resetRangeOnDoubleActivation(slider, defaults[param] ?? min);
  }

  if (checkbox) {
    checkbox.addEventListener('change', () => {
      sendAction({ action: 'set_param', param, value: checkbox.checked });
    });
  }

  if (select) {
    select.addEventListener('change', () => {
      sendAction({ action: 'set_param', param, value: parseInt(select.value) });
    });
  }
});

// --- Initialize NTSC/VHS sliders ---

document.querySelectorAll('.param-row[data-ntsc]').forEach((row) => {
  const param = row.dataset.ntsc;
  const min = parseFloat(row.dataset.min);
  const max = parseFloat(row.dataset.max);
  const step = parseFloat(row.dataset.step);

  const slider = row.querySelector('input[type="range"]');
  const valueEl = row.querySelector('.value');
  const checkbox = row.querySelector('input[type="checkbox"]');
  const select = row.querySelector('select');

  if (slider) {
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = min;

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      sendAction({ action: 'set_ntsc_param', param, value: v });
    });
    const defaults = NTSC_PARAM_DEFAULTS;
    resetRangeOnDoubleActivation(slider, defaults[param] ?? 0);
  }

  if (checkbox) {
    checkbox.addEventListener('change', () => {
      sendAction({ action: 'set_ntsc_param', param, value: checkbox.checked });
    });
  }

  if (select) {
    select.addEventListener('change', () => {
      sendAction({ action: 'set_ntsc_param', param, value: parseInt(select.value) });
    });
  }
});

// --- Initialize temporal (feedback/slit-scan) controls ---

document.querySelectorAll('.param-row[data-temporal]').forEach((row) => {
  const param = row.dataset.temporal;
  const min = parseFloat(row.dataset.min);
  const max = parseFloat(row.dataset.max);
  const step = parseFloat(row.dataset.step);

  const slider = row.querySelector('input[type="range"]');
  const checkbox = row.querySelector('input[type="checkbox"]');
  const number = row.querySelector('input[type="number"]');
  const valueEl = row.querySelector('.value');
  const select = row.querySelector('select');

  if (slider) {
    slider.min = min;
    slider.max = max;
    slider.step = step;
    const defaults = TEMPORAL_PARAM_DEFAULTS;
    slider.value = defaults[param] ?? min;

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      sendAction({ action: 'set_temporal', param, value: v });
    });
    resetRangeOnDoubleActivation(slider, defaults[param] ?? 0);
  }

  if (checkbox) {
    checkbox.addEventListener('change', () => {
      sendAction({ action: 'set_temporal', param, value: checkbox.checked });
    });
  }

  if (number) {
    number.addEventListener('change', () => {
      const parsed = Number(number.value);
      const minimum = Number(number.min);
      const maximum = Number(number.max);
      if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
        number.value = row.dataset.default || '0';
        return;
      }
      sendAction({ action: 'set_temporal', param, value: parsed });
    });
  }

  if (select) {
    select.addEventListener('change', () => {
      const value = param === 'key_mode' ? parseInt(select.value, 10) : select.value;
      sendAction({ action: 'set_temporal', param, value });
    });
  }
});

function syncTemporalLoopDriver(driver) {
  const select = document.getElementById('temporal-score-loop-driver');
  if (!select || !canSync(select)) return;
  const desired = driver?.kind === 'selected_layer'
    ? String(driver.layer_id || '')
    : driver?.kind === 'missing_selected_layer'
      ? `missing:${Number(driver.saved_position) || 0}`
      : 'none';

  select.replaceChildren(new Option('None', 'none'));
  latestLayers.forEach((layer, index) => {
    const id = String(layer.layer_id || '');
    if (!id) return;
    select.add(new Option(`Layer ${index + 1}: ${layer.filename || layer.source_kind || id}`, id));
  });
  if (desired.startsWith('missing:')) {
    const position = Number(desired.slice(8)) + 1;
    const missing = new Option(`Missing saved layer ${position}`, desired, true, true);
    missing.disabled = true;
    select.add(missing);
  }
  select.value = [...select.options].some(option => option.value === desired) ? desired : 'none';
}

function gardenRouteDesired(route) {
  if (route?.kind === 'selected_layer') return String(route.layer_id || '');
  if (route?.kind === 'missing_selected_layer') {
    return `missing:${Number(route.saved_position) || 0}`;
  }
  return 'none';
}

function syncGardenLayerRoute(select, route, label) {
  if (!select || !canSync(select)) return;
  const desired = gardenRouteDesired(route);
  select.replaceChildren(new Option('None', 'none'));
  latestLayers.forEach((layer, index) => {
    const id = stableLayerId(layer);
    if (!id) return;
    const name = layer.filename || layer.source_kind || id;
    select.add(new Option(`Layer ${index + 1}: ${name}`, id));
  });
  if (desired.startsWith('missing:')) {
    const position = Number(desired.slice(8)) + 1;
    const missing = new Option(`Missing saved layer ${position}`, desired, true, true);
    missing.disabled = true;
    select.add(missing);
  } else if (route?.kind === 'selected_layer'
      && ![...select.options].some(option => option.value === desired)) {
    const unavailable = new Option(`${label} layer ${desired} unavailable`, desired, true, true);
    unavailable.disabled = true;
    select.add(unavailable);
  }
  select.value = [...select.options].some(option => option.value === desired) ? desired : 'none';
}

function gardenRouteDiagnostic(label, route) {
  if (route?.kind === 'selected_layer') {
    const index = latestLayers.findIndex(layer => stableLayerId(layer) === String(route.layer_id));
    return index >= 0 ? `${label} Layer ${index + 1}` : `${label} selected layer unavailable`;
  }
  if (route?.kind === 'missing_selected_layer') {
    return `${label} missing saved layer ${(Number(route.saved_position) || 0) + 1} · gate zero`;
  }
  return `${label} not selected · gate zero`;
}

function syncRefreshGardenRoutes(garden) {
  const matteRoute = garden?.matte_route || { kind: 'none' };
  const motionRoute = garden?.motion_route || { kind: 'none' };
  const matteSelect = document.getElementById('temporal-garden-matte-route');
  const motionSelect = document.getElementById('temporal-garden-motion-route');
  const stageSelect = document.getElementById('temporal-garden-matte-stage');
  syncGardenLayerRoute(matteSelect, matteRoute, 'Matte');
  syncGardenLayerRoute(motionSelect, motionRoute, 'Motion');
  if (stageSelect) {
    if (canSync(stageSelect)) stageSelect.value = matteRoute.stage || 'post_local_effects';
    stageSelect.disabled = matteRoute.kind !== 'selected_layer';
  }
  const status = document.getElementById('temporal-garden-route-status');
  if (status) {
    status.textContent = `${gardenRouteDiagnostic('Matte', matteRoute)} · ${gardenRouteDiagnostic('Motion', motionRoute)}`;
  }
}

function syncTemporal(t) {
  if (!t) return;
  const originals = t.originals || {};
  const loom = originals.loom || {};
  const atlas = originals.atlas || {};
  const garden = originals.garden || {};
  const longExposure = originals.long_exposure || {};
  const score = originals.score || {};
  const reset = originals.reset || {};
  const rig = t.rig || {};
  const display = t.display || {};
  const masterMelt = t.melt || {};
  const mosh = t.mosh || {};
  const syncLatch = t.sync || {};
  const values = {
    feedback: t.feedback,
    fb_zoom: t.fb_zoom,
    fb_rotate: t.fb_rotate,
    fb_offset_x: rig.offset_x,
    fb_offset_y: rig.offset_y,
    fb_reflect_x: rig.reflect_x,
    fb_reflect_y: rig.reflect_y,
    fb_hue_rotate: rig.hue_rotate,
    fb_saturation: rig.saturation,
    fb_gain_r: rig.gain_r,
    fb_gain_g: rig.gain_g,
    fb_gain_b: rig.gain_b,
    fb_chroma_displace: rig.chroma_displace,
    fb_blur: rig.blur,
    fb_sharpen: rig.sharpen,
    fb_shape: rig.shape,
    fb_drive: rig.drive,
    fb_pivot: rig.pivot,
    fb_threshold: rig.threshold,
    fb_noise: rig.noise,
    fb_edge: rig.edge,
    fb_servo: rig.servo,
    fb_servo_defeated: rig.servo_defeated,
    slitscan: t.slitscan,
    slit_angle: t.slit_angle,
    slit_map: t.slit_map,
    slit_interp: t.slit_interp,
    key_mode: t.key_mode,
    key_threshold: t.key_threshold,
    key_softness: t.key_softness,
    key_history: t.key_history,
    long_exposure_amount: longExposure.amount,
    long_exposure_frames: longExposure.shutter_frames,
    loom_amount: loom.amount,
    loom_topology: loom.topology,
    loom_interpolation: loom.interpolation,
    loom_depth: loom.depth,
    loom_phase: loom.phase,
    loom_scale: loom.scale,
    loom_angle: loom.angle,
    loom_folds: loom.folds,
    loom_quantization: loom.quantization,
    atlas_amount: atlas.amount,
    atlas_seed: atlas.seed,
    atlas_territories: atlas.territories,
    atlas_collision: atlas.collision,
    garden_amount: garden.amount,
    garden_gate: garden.gate,
    garden_threshold: garden.threshold,
    garden_softness: garden.softness,
    garden_decay: garden.decay,
    garden_max_hold_ticks: garden.max_hold_ticks,
    score_enabled: score.enabled,
    score_seed: score.seed,
    score_state_count: score.state_count,
    score_trigger: score.trigger,
    reset_loop_boundary: reset.loop_boundary,
    reset_downbeat: reset.downbeat,
    disp_il_amount: display.il_amount,
    disp_il_mode: display.il_mode,
    disp_il_order: display.il_order,
    disp_il_twitter: display.il_twitter,
    disp_il_judder: display.il_judder,
    disp_phosphor: display.phosphor,
    disp_phos_r: display.phos_r,
    disp_phos_g: display.phos_g,
    disp_phos_b: display.phos_b,
    disp_model: display.model,
    disp_scanlines: display.scanlines,
    disp_beam_width: display.beam_width,
    disp_beam_shape: display.beam_shape,
    disp_mask_strength: display.mask_strength,
    disp_mask_dark: display.mask_dark,
    disp_bloom: display.bloom,
    disp_bloom_radius: display.bloom_radius,
    disp_halation: display.halation,
    disp_defocus: display.defocus,
    disp_sag: display.sag,
    melt_amount: masterMelt.melt,
    melt_width: masterMelt.width,
    melt_hold: masterMelt.hold,
    melt_swirl: masterMelt.swirl,
    melt_chroma: masterMelt.chroma,
    melt_creep: masterMelt.creep,
    mosh_amount: mosh.amount,
    mosh_key_removal: mosh.key_removal,
    mosh_hold: mosh.hold,
    mosh_drop: mosh.drop,
    mosh_shuffle: mosh.shuffle,
    mosh_rate: mosh.rate,
    mosh_bitrate_starve: mosh.bitrate_starve,
    mosh_resync: mosh.resync,
    mosh_wipe: Number(mosh.wipe ?? 0),
    mosh_smear: Number(mosh.smear ?? 0),
    mosh_trail: Number(mosh.trail ?? 0),
    mosh_recycle: mosh.recycle,
    sync_amount: syncLatch.amount,
    sync_rate: syncLatch.rate,
    sync_spread: syncLatch.spread,
    sync_bias: syncLatch.bias,
    sync_latched: syncLatch.latched,
  };
  // The latch's honest state: the switch says what was asked for, this says
  // whether the program is actually still carrying accumulated shear. The
  // fact is carried by the text as well as the colour, so it does not depend
  // on colour perception to be readable.
  const syncStatus = document.getElementById('sync-latch-status');
  if (syncStatus) {
    if (!syncStatus.dataset.baseText) {
      syncStatus.dataset.baseText = syncStatus.textContent;
    }
    const damaged = t.sync_damaged === true;
    syncStatus.classList.toggle('sync-damaged', damaged);
    syncStatus.dataset.damaged = damaged ? 'true' : 'false';
    // Blank at rest: the explanation lives in the section's ? affordance, and
    // this line exists to report the one fact a switch cannot — that damage is
    // still held.
    syncStatus.textContent = damaged
      ? 'DAMAGE HELD — release the latch to unwind it.'
      : '';
  }
  for (const [param, value] of Object.entries(values)) {
    if (value === undefined || value === null) continue;
    const row = document.querySelector(`.param-row[data-temporal="${param}"]`);
    if (!row) continue;
    const slider = row.querySelector('input[type="range"]');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const number = row.querySelector('input[type="number"]');
    const valueEl = row.querySelector('.value');
    const select = row.querySelector('select');
    if (slider && valueEl && canSync(slider)) {
      slider.value = value;
      valueEl.textContent = formatValue(
        value,
        parseFloat(row.dataset.min),
        parseFloat(row.dataset.max),
        parseFloat(row.dataset.step)
      );
    }
    if (select && canSync(select)) {
      select.value = String(value);
    }
    if (checkbox && canSync(checkbox)) checkbox.checked = Boolean(value);
    if (number && canSync(number)) number.value = String(value);
  }
  syncTemporalLoopDriver(score.loop_driver);
  syncRefreshGardenRoutes(garden);

  const telemetry = t.telemetry || {};
  const telemetryEl = document.getElementById('temporal-telemetry');
  if (telemetryEl) {
    const valid = Number(telemetry.history_valid) || 0;
    const capacity = Number(telemetry.history_capacity) || 24;
    const carrier = telemetry.carrier_valid ? 'carrier ready' : 'carrier empty';
    const hold = telemetry.freeze_hold_valid ? 'hold ready' : 'hold empty';
    const ticks = Number(telemetry.total_reference_ticks) || 0;
    const scoreState = Number(telemetry.score_state) || 0;
    const ordinal = Number(telemetry.score_event_ordinal) || 0;
    const recorded = Number(telemetry.recorded_event_points) || 0;
    const track = telemetry.event_track_truncated ? `${recorded}+ recorded` : `${recorded} recorded`;
    const staged = telemetry.frame_staged ? ' · staged' : '';
    const resetText = telemetry.last_reset ? ` · reset ${telemetry.last_reset}` : '';
    telemetryEl.textContent = `History ${valid}/${capacity} · ${carrier} · ${hold} · tick ${ticks} · Score ${scoreState} · event ${ordinal} · ${track}${staged}${resetText}`;
  }
}

// --- Gesture field etching ---
//
// The honesty law on the panel: `recorded_events` and `live_only_events` are
// published as separate fields and are rendered as separate sentences, so an
// unrecorded live gesture is never displayed as replayable. Only the armed
// state is announced; the per-frame counters live outside the live region.

const GESTURE_CANVAS_DEFAULTS = { radius: 0.12, strength: 0.5, retention: 0.99 };
let gestureRecording = false;

document.querySelectorAll('.param-row[data-gesture-canvas]').forEach((row) => {
  const param = row.dataset.gestureCanvas;
  const min = parseFloat(row.dataset.min);
  const max = parseFloat(row.dataset.max);
  const step = parseFloat(row.dataset.step);
  const slider = row.querySelector('input[type="range"]');
  const valueEl = row.querySelector('.value');
  if (!slider || !valueEl) return;
  slider.min = min;
  slider.max = max;
  slider.step = step;
  const fallback = GESTURE_CANVAS_DEFAULTS[param] ?? min;
  slider.value = fallback;
  slider.addEventListener('input', () => {
    const value = parseFloat(slider.value);
    valueEl.textContent = formatValue(value, min, max, step);
    sendAction({ action: 'set_gesture_canvas', param, value });
  });
  resetRangeOnDoubleActivation(slider, fallback);
});

document.getElementById('gesture-record-toggle')?.addEventListener('click', () => {
  sendAction({
    action: 'set_gesture_recording',
    enabled: !gestureRecording,
    layer_stack_revision: layerStackRevision,
  });
});

function syncGesture(gesture) {
  const g = gesture || {};
  const canvas = g.canvas || {};
  gestureRecording = Boolean(g.recording);

  for (const param of ['radius', 'strength', 'retention']) {
    const value = canvas[param];
    if (value === undefined || value === null) continue;
    const row = document.querySelector(`.param-row[data-gesture-canvas="${param}"]`);
    if (!row) continue;
    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    if (!slider || !valueEl || !canSync(slider)) continue;
    slider.value = value;
    valueEl.textContent = formatValue(
      value,
      parseFloat(row.dataset.min),
      parseFloat(row.dataset.max),
      parseFloat(row.dataset.step)
    );
  }

  const toggle = document.getElementById('gesture-record-toggle');
  if (toggle) {
    toggle.textContent = gestureRecording ? 'Disarm recording' : 'Arm recording';
    toggle.setAttribute('aria-pressed', gestureRecording ? 'true' : 'false');
  }

  const open = Number(g.open_strokes) || 0;
  const stateEl = document.getElementById('gesture-recording-state');
  if (stateEl) {
    const armed = gestureRecording ? 'Recording' : 'Not recording';
    const incomplete = open > 0 ? ` · ${open} open stroke(s), track explicitly incomplete` : '';
    const status = g.status ? ` · ${g.status}` : '';
    stateEl.textContent = `${armed}${incomplete}${status}`;
  }

  const telemetryEl = document.getElementById('gesture-telemetry');
  if (telemetryEl) {
    const recorded = Number(g.recorded_events) || 0;
    const liveOnly = Number(g.live_only_events) || 0;
    const truncated = g.truncated ? ' (capped)' : '';
    const width = Number(canvas.grid_width) || 0;
    const height = Number(canvas.grid_height) || 0;
    const generation = Number(canvas.generation) || 0;
    telemetryEl.textContent =
      `${recorded} recorded event(s)${truncated} · ${liveOnly} live-only sample(s) · ` +
      `canvas ${width}×${height} · gen ${generation}`;
  }

  const checksumEl = document.getElementById('gesture-checksum');
  if (checksumEl) {
    // An empty track publishes no digest, so the panel says so instead of
    // rendering an empty field that could read as a verified recording.
    checksumEl.textContent = g.checksum
      ? `Recorded track ${String(g.checksum).slice(0, 16)}… · replayable`
      : 'No recorded track';
  }
}

// --- B9 performance recorder ------------------------------------------
// Both transports are ordered barriers carrying the layer-stack revision;
// the loop flag rides the playback arm so a repeated arm only retunes it.
let performanceMode = 'off';
let performanceLoop = false;

document.getElementById('performance-record-toggle')?.addEventListener('click', () => {
  sendAction({
    action: 'set_performance_recording',
    enabled: performanceMode !== 'recording',
    layer_stack_revision: layerStackRevision,
  });
});

document.getElementById('performance-play-toggle')?.addEventListener('click', () => {
  sendAction({
    action: 'set_performance_playback',
    enabled: performanceMode !== 'playing',
    loop_playback: performanceLoop,
    layer_stack_revision: layerStackRevision,
  });
});

document.getElementById('performance-loop-toggle')?.addEventListener('click', () => {
  performanceLoop = !performanceLoop;
  const loopEl = document.getElementById('performance-loop-toggle');
  if (loopEl) loopEl.setAttribute('aria-pressed', performanceLoop ? 'true' : 'false');
  if (performanceMode === 'playing') {
    sendAction({
      action: 'set_performance_playback',
      enabled: true,
      loop_playback: performanceLoop,
      layer_stack_revision: layerStackRevision,
    });
  }
});

document.getElementById('performance-clear')?.addEventListener('click', () => {
  sendAction({ action: 'clear_performance_take' });
});

function syncPerformanceRecorder(recorder) {
  const p = recorder || {};
  performanceMode = String(p.mode || 'off');
  if (performanceMode === 'playing') performanceLoop = Boolean(p.loop_playback);

  const record = document.getElementById('performance-record-toggle');
  if (record) {
    const recording = performanceMode === 'recording';
    record.textContent = recording ? 'Disarm recording' : 'Arm recording';
    record.setAttribute('aria-pressed', recording ? 'true' : 'false');
  }
  const play = document.getElementById('performance-play-toggle');
  if (play) {
    const playing = performanceMode === 'playing';
    play.textContent = playing ? 'Stop playback' : 'Play take';
    play.setAttribute('aria-pressed', playing ? 'true' : 'false');
  }
  const loop = document.getElementById('performance-loop-toggle');
  if (loop) loop.setAttribute('aria-pressed', performanceLoop ? 'true' : 'false');

  const stateEl = document.getElementById('performance-state');
  if (stateEl) {
    const mode =
      performanceMode === 'recording'
        ? 'Recording take'
        : performanceMode === 'playing'
          ? `Playing take · tick ${Number(p.playhead_tick) || 0}`
          : 'Recorder off';
    const status = p.status ? ` · ${p.status}` : '';
    stateEl.textContent = `${mode}${status}`;
  }

  const telemetryEl = document.getElementById('performance-telemetry');
  if (telemetryEl) {
    const events = Number(p.recorded_events) || 0;
    const controls = Number(p.recorded_controls) || 0;
    const ticks = Number(p.length_ticks) || 0;
    const truncated = p.truncated ? ' (capped)' : '';
    const skipped = Number(p.unsupported_edits) || 0;
    const rejected = Number(p.rejected_edits) || 0;
    const counted =
      skipped || rejected ? ` · ${skipped + rejected} edit(s) skipped as unrecordable` : '';
    telemetryEl.textContent =
      `${events} recorded edit(s)${truncated} · ${controls} control(s) · ${ticks} tick(s)${counted}`;
  }

  const checksumEl = document.getElementById('performance-checksum');
  if (checksumEl) {
    checksumEl.textContent = p.checksum
      ? `Take ${String(p.checksum).slice(0, 16)}… · replayable`
      : 'No recorded take';
  }

  const degradedEl = document.getElementById('performance-degraded');
  if (degradedEl) {
    const degraded = Array.isArray(p.degraded) ? p.degraded : [];
    degradedEl.textContent = degraded.length
      ? `Degraded control(s): ${degraded.join(', ')}`
      : '';
  }
}

document.getElementById('temporal-clear-memory')?.addEventListener('click', () => {
  sendAction({ action: 'clear_temporal_memory' });
});

document.getElementById('temporal-clear-event-track')?.addEventListener('click', () => {
  sendAction({ action: 'clear_temporal_event_track' });
});

document.getElementById('temporal-garden-trigger')?.addEventListener('click', () => {
  sendAction({ action: 'trigger_refresh_garden' });
});

document.getElementById('temporal-garden-matte-route')?.addEventListener('change', (event) => {
  const value = String(event.currentTarget.value || 'none');
  const layerId = /^(?:[1-9][0-9]*)$/.test(value) ? value : null;
  const stage = document.getElementById('temporal-garden-matte-stage')?.value || 'post_local_effects';
  sendAction({
    action: 'set_refresh_garden_matte_route',
    layer_id: layerId,
    stage,
    layer_stack_revision: layerStackRevision,
  });
});

document.getElementById('temporal-garden-matte-stage')?.addEventListener('change', (event) => {
  const route = document.getElementById('temporal-garden-matte-route');
  const layerId = String(route?.value || '');
  if (!/^(?:[1-9][0-9]*)$/.test(layerId)) return;
  sendAction({
    action: 'set_refresh_garden_matte_route',
    layer_id: layerId,
    stage: event.currentTarget.value,
    layer_stack_revision: layerStackRevision,
  });
});

document.getElementById('temporal-garden-motion-route')?.addEventListener('change', (event) => {
  const value = String(event.currentTarget.value || 'none');
  const layerId = /^(?:[1-9][0-9]*)$/.test(value) ? value : null;
  sendAction({
    action: 'set_refresh_garden_motion_route',
    layer_id: layerId,
    layer_stack_revision: layerStackRevision,
  });
});

document.getElementById('temporal-score-trigger')?.addEventListener('click', () => {
  sendAction({ action: 'trigger_collision_score' });
});

// --- M4 Motion Fields / Faraday Transplant / Curved Shutter ---

const MOTION_PARAM_DEFAULTS = Object.freeze({
  field_source: 'auto', lattice_quality: 'live', field_scale: 0.5, field_rate: 0.25,
  stretch: 0, edge_repel: 0, vector_trash: 0, trash_block_size: 16,
  transplant_amount: 0,
  carrier: 'transparent', confidence_threshold: 0.1, confidence_softness: 0.05,
  refresh: 1, decay: 1, occlusion: 0, shutter_angle: 0, shutter_phase: 0,
  shutter_curvature: 0, shutter_chromatic_lag: 0, shutter_quality: 'sharp',
  collider_enabled: false, collider_mode: 'sum', collider_boundary: 'transparent',
});

function motionParamValue(motion = {}, param) {
  const transplant = motion.transplant || {};
  const shutter = motion.shutter || {};
  const collider = motion.collider || {};
  const procedural = motion.procedural || {};
  const shaping = motion.shaping || {};
  const values = {
    field_source: motion.field_source,
    lattice_quality: motion.lattice_quality,
    field_scale: procedural.scale,
    field_rate: procedural.rate,
    stretch: shaping.stretch,
    edge_repel: shaping.edge_repel,
    vector_trash: shaping.vector_trash,
    trash_block_size: shaping.trash_block_size,
    transplant_amount: transplant.amount,
    carrier: transplant.carrier,
    confidence_threshold: transplant.confidence_threshold,
    confidence_softness: transplant.confidence_softness,
    refresh: transplant.refresh,
    decay: transplant.decay,
    occlusion: transplant.occlusion,
    shutter_angle: shutter.angle_degrees,
    shutter_phase: shutter.phase,
    shutter_curvature: shutter.curvature,
    shutter_chromatic_lag: shutter.chromatic_lag,
    shutter_quality: shutter.quality,
    collider_enabled: collider.enabled,
    collider_mode: collider.mode,
    collider_boundary: collider.boundary,
  };
  return values[param] ?? MOTION_PARAM_DEFAULTS[param];
}

function motionColliderText(motion = {}) {
  const collider = motion.collider || {};
  if (!collider.enabled) return 'Collider off - single-donor Faraday transplant is live.';
  if (collider.diagnostic) return String(collider.diagnostic);
  return collider.admitted ? 'Collider admitted - the derived field advects the carrier.' : 'Collider inert.';
}

function motionTelemetryText(motion = {}) {
  const telemetry = motion.telemetry || {};
  const dimensions = telemetry.field_dimensions || [0, 0];
  const source = telemetry.effective_source || 'idle';
  const rendered = telemetry.rendered_source || 'none';
  const fallback = telemetry.fallback_active ? ' · lattice fallback' : '';
  const missing = telemetry.donor_missing ? ' · donor missing' : '';
  const admitted = telemetry.transplant_admitted ? ' · transplant admitted' : '';
  const carrier = telemetry.carrier_valid ? 'carrier ready' : 'carrier empty';
  const field = dimensions[0] && dimensions[1] ? ` · ${dimensions[0]}×${dimensions[1]}` : '';
  const diagnostic = telemetry.diagnostic ? ` · ${telemetry.diagnostic}` : '';
  const attachment = telemetry.field_planned
    ? (telemetry.field_attached ? ` · rendered ${rendered}` : ' · field priming/unavailable')
    : ' · no field planned';
  return `v${Number(motion.algorithm_version || 1)} · planned ${source}${fallback}${field} · ${Number(telemetry.vector_count || 0)} vectors${attachment} · ${carrier}${admitted}${missing}${diagnostic}`;
}

function wireMotionPanel(panel, scopeProvider) {
  if (!panel || panel.dataset.motionWired === 'true') return;
  panel.dataset.motionWired = 'true';
  panel.querySelectorAll('[data-motion-param]').forEach((row) => {
    const param = row.dataset.motionParam;
    const control = row.querySelector('input,select');
    if (!control) return;
    const eventName = control.type === 'range' ? 'input' : 'change';
    control.addEventListener(eventName, () => {
      const scope = scopeProvider();
      if (!scope) return;
      const value = control.type === 'range'
        ? Number(control.value)
        : control.type === 'checkbox' ? control.checked : control.value;
      if (control.type === 'range') {
        const valueEl = row.querySelector('.value');
        if (valueEl) valueEl.textContent = formatValue(value, Number(control.min), Number(control.max), Number(control.step));
      }
      sendAction({ action: 'set_motion', scope, param, value });
    });
    if (control.type === 'range') resetRangeOnDoubleActivation(control, Number(MOTION_PARAM_DEFAULTS[param]));
  });
}

function syncMotionPanel(panel, motion = {}) {
  if (!panel) return;
  panel.querySelectorAll('[data-motion-param]').forEach((row) => {
    const param = row.dataset.motionParam;
    const control = row.querySelector('input,select');
    if (!control || !canSync(control)) return;
    const value = motionParamValue(motion, param);
    if (control.type === 'checkbox') {
      control.checked = Boolean(value);
      return;
    }
    control.value = String(value);
    if (control.type === 'range') {
      const valueEl = row.querySelector('.value');
      if (valueEl) valueEl.textContent = formatValue(Number(value), Number(control.min), Number(control.max), Number(control.step));
    }
  });
  const telemetry = panel.querySelector('.motion-telemetry');
  if (telemetry) {
    telemetry.textContent = motionTelemetryText(motion);
    const diagnostic = String(motion.telemetry?.diagnostic || '').toLowerCase();
    telemetry.classList.toggle('error', !!diagnostic || !!motion.telemetry?.donor_missing);
  }
}

const masterMotionPanel = document.getElementById('master-motion-panel');
wireMotionPanel(masterMotionPanel, () => ({ scope: 'master' }));

function syncMasterMotion(motion) {
  syncMotionPanel(masterMotionPanel, motion || {});
}

document.getElementById('motion-clear-memory')?.addEventListener('click', () => {
  sendAction({ action: 'clear_motion_memory' });
});

// --- Collapsible FX groups (state remembered across reloads) ---

const groupKey = (g) => g.id || g.dataset.group || '';

function loadCollapsedState() {
  try {
    return new Set(JSON.parse(localStorage.getItem('cos-collapsed') || '[]'));
  } catch {
    return new Set();
  }
}

const collapsedState = loadCollapsedState();
if (localStorage.getItem('cos-collapsed') !== null) {
  // A saved layout exists: apply it (overriding HTML defaults).
  document.querySelectorAll('.fx-group').forEach((g) => {
    g.classList.toggle('collapsed', collapsedState.has(groupKey(g)));
  });
}

document.querySelectorAll('.fx-group-header').forEach((header) => {
  const group = header.closest('.fx-group');
  const body = group.querySelector(':scope > .fx-group-body');
  const key = groupKey(group) || `fx-group-${Array.from(document.querySelectorAll('.fx-group-header')).indexOf(header)}`;
  if (body && !body.id) body.id = `${key.replace(/[^a-zA-Z0-9_-]/g, '-')}-body`;
  const label = header.querySelector('.group-label')?.textContent?.trim() || 'effects';
  const labelElement = header.querySelector('.group-label');
  if (labelElement && !labelElement.id) labelElement.id = `${key.replace(/[^a-zA-Z0-9_-]/g, '-')}-label`;
  group.setAttribute('role', 'group');
  if (labelElement) group.setAttribute('aria-labelledby', labelElement.id);
  else group.setAttribute('aria-label', `${label} controls`);
  const chevron = header.querySelector('.chevron');
  if (chevron) {
    chevron.setAttribute('role', 'button');
    chevron.tabIndex = 0;
    chevron.setAttribute('aria-label', `Toggle ${label} controls`);
    if (body) chevron.setAttribute('aria-controls', body.id);
  }

  const syncExpanded = () => {
    chevron?.setAttribute('aria-expanded', String(!group.classList.contains('collapsed')));
  };
  const toggle = () => {
    const group = header.closest('.fx-group');
    group.classList.toggle('collapsed');
    const key = groupKey(group);
    if (key) {
      if (group.classList.contains('collapsed')) collapsedState.add(key);
      else collapsedState.delete(key);
      localStorage.setItem('cos-collapsed', JSON.stringify([...collapsedState]));
    }
    syncExpanded();
  };
  syncExpanded();
  header.addEventListener('click', (e) => {
    if (e.target.closest('.group-reset')) return;
    toggle();
  });
  chevron?.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    toggle();
  });
});

// --- Group reset buttons ---

document.querySelectorAll('.group-reset').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    sendAction({ action: 'reset_group', group: btn.dataset.group });
  });
});

// --- Transport buttons ---

document.getElementById('btn-play-all').addEventListener('click', () => {
  if (transportPendingPaused !== null) return;
  const target = !transportAuthoritativePaused;
  if (sendAction({ action: 'set_program_frozen', frozen: target })) {
    const requestSequence = ++transportRequestSequence;
    transportPendingPaused = target;
    renderMasterTransport(target, true);
    // A state snapshot can be lost during reconnect. Do not leave the only
    // transport control disabled forever; after a bounded grace period,
    // return it to the last authoritative snapshot and permit a retry.
    window.setTimeout(() => {
      if (transportPendingPaused !== null && transportRequestSequence === requestSequence) {
        transportPendingPaused = null;
        renderMasterTransport(transportAuthoritativePaused, false);
      }
    }, 2000);
  }
});

document.getElementById('btn-freeze-media').addEventListener('click', () => {
  if (mediaPendingFrozen !== null) return;
  const target = !mediaAuthoritativeFrozen;
  if (sendAction({ action: 'set_media_frozen', frozen: target })) {
    const requestSequence = ++mediaRequestSequence;
    mediaPendingFrozen = target;
    renderMediaFreeze(target, true);
    window.setTimeout(() => {
      if (mediaPendingFrozen !== null && mediaRequestSequence === requestSequence) {
        mediaPendingFrozen = null;
        renderMediaFreeze(mediaAuthoritativeFrozen, false);
      }
    }, 2000);
  }
});

document.getElementById('btn-revert-master').addEventListener('click', () => {
  sendAction({ action: 'reset_visual_program' });
});

document.getElementById('btn-blackout').addEventListener('click', () => {
  sendAction({ action: 'set_blackout', enabled: !document.getElementById('btn-blackout').classList.contains('active') });
});

function syncBlackout(on) {
  document.getElementById('btn-blackout').classList.toggle('active', !!on);
}

// --- Sync effects UI from server ---

function syncEffects(effects) {
  if (!effects) return;
  const randomStatus = document.getElementById('reroll-status');
  if (randomStatus) {
    const seed = Number(effects.random_seed) >>> 0;
    randomStatus.textContent = seed === 0
      ? 'Master seed 0 · legacy pattern'
      : `Master seed ${seed}`;
  }
  const values = { ...effects };
  if (Array.isArray(effects.key_color)) {
    [values.key_color_r, values.key_color_g, values.key_color_b] = effects.key_color;
  }
  for (const [param, value] of Object.entries(values)) {
    const row = document.querySelector(`.param-row[data-param="${param}"]`);
    if (!row) continue;

    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const select = row.querySelector('select');

    if (slider && valueEl && canSync(slider)) {
      slider.value = value;
      const min = parseFloat(row.dataset.min);
      const max = parseFloat(row.dataset.max);
      const step = parseFloat(row.dataset.step);
      valueEl.textContent = formatValue(value, min, max, step);
    }

    if (checkbox && canSync(checkbox)) {
      checkbox.checked = !!value;
    }

    if (select && canSync(select)) {
      select.value = value;
    }
  }
}

// --- Deterministic Random / Dice controls ---

const rerollScope = document.getElementById('reroll-scope');
const rerollMode = document.getElementById('reroll-mode');
const rerollSeed = document.getElementById('reroll-seed');
const rerollAmount = document.getElementById('reroll-amount');
const rerollAmountValue = document.getElementById('reroll-amount-value');
const rerollGrainControls = document.getElementById('reroll-grain-controls');
const rerollTransformControls = document.getElementById('reroll-transform-controls');
const rerollRackControls = document.getElementById('reroll-rack-controls');
const rerollGroupControls = document.getElementById('reroll-group-controls');
const rerollGroup = document.getElementById('reroll-group');
const rerollButton = document.getElementById('reroll-button');

function syncRerollModeControls() {
  const variation = rerollMode?.value === 'variation';
  const groupScope = rerollScope?.value === 'group';
  const compositionScope = groupScope || rerollScope?.value === 'all';
  document.getElementById('reroll-amount-row')?.classList.toggle('control-disabled', !variation);
  document.getElementById('reroll-grain-row')?.classList.toggle('control-disabled', !variation);
  document.getElementById('reroll-transform-row')?.classList.toggle('control-disabled', !variation);
  document.getElementById('reroll-rack-row')?.classList.toggle('control-disabled', !variation);
  document.getElementById('reroll-group-controls-row')?.classList.toggle('control-disabled', !variation || !compositionScope);
  const groupRow = document.getElementById('reroll-group-row');
  if (groupRow) groupRow.hidden = !groupScope;
  if (rerollAmount) rerollAmount.disabled = !variation;
  if (rerollGrainControls) rerollGrainControls.disabled = !variation;
  if (rerollTransformControls) rerollTransformControls.disabled = !variation;
  if (rerollRackControls) rerollRackControls.disabled = !variation;
  if (rerollGroupControls) rerollGroupControls.disabled = !variation || !compositionScope;
  if (rerollGroup) rerollGroup.disabled = !groupScope;
}

function readExactRerollSeed() {
  const text = rerollSeed?.value.trim() || '';
  if (text === '') {
    rerollSeed?.setCustomValidity('');
    return { valid: true, value: null };
  }
  const seed = Number(text);
  const valid = Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff;
  rerollSeed?.setCustomValidity(valid ? '' : 'Seed must be a whole number from 0 to 4294967295');
  if (!valid) rerollSeed?.reportValidity();
  return { valid, value: valid ? seed : null };
}

rerollMode?.addEventListener('change', syncRerollModeControls);
rerollScope?.addEventListener('change', syncRerollModeControls);
rerollAmount?.addEventListener('input', () => {
  rerollAmountValue.textContent = Number(rerollAmount.value).toFixed(2);
});
const rerollKeepSource = document.getElementById('reroll-keep-source');
const rerollKeepModulation = document.getElementById('reroll-keep-modulation');
const rerollKeepOutput = document.getElementById('reroll-keep-output');
rerollSeed?.addEventListener('input', () => readExactRerollSeed());
rerollButton?.addEventListener('click', () => {
  const seed = readExactRerollSeed();
  if (!seed.valid) return;
  const scope = ['all', 'group'].includes(rerollScope.value) ? rerollScope.value : 'master';
  const action = {
    action: 'reroll',
    scope,
    mode: rerollMode.value === 'variation' ? 'variation' : 'pattern',
    amount: Math.min(2, Math.max(0, Number(rerollAmount.value) || 0)),
    include_grain_controls: !!rerollGrainControls.checked,
    include_transform: !!rerollTransformControls.checked,
    include_rack_controls: !!rerollRackControls?.checked,
    include_group_controls: scope === 'group' && !!rerollGroupControls?.checked
      || scope === 'all' && !!rerollGroupControls?.checked,
    // B15 keep-masks. Each protects one domain from an otherwise ordinary
    // throw; all three default off, so an untouched panel sends exactly what
    // it always sent.
    keep_source: !!rerollKeepSource?.checked,
    keep_modulation: !!rerollKeepModulation?.checked,
    keep_output_chain: !!rerollKeepOutput?.checked,
  };
  if (seed.value !== null) action.seed = seed.value;
  if (scope === 'all') action.stack_revision = layerStackRevision;
  if (scope === 'group') {
    if (!rerollGroup?.value) return;
    action.group_id = rerollGroup.value;
  }
  if (sendAction(action)) {
    document.getElementById('reroll-status').textContent = seed.value === null
      ? 'Advancing deterministic seed…'
      : `Applying exact seed ${seed.value}…`;
  }
});
syncRerollModeControls();

// --- Sync NTSC/VHS UI from server ---

function syncNtsc(ntsc) {
  if (!ntsc) return;
  const ntscStatus = document.getElementById('ntsc-status');
  if (ntscStatus) ntscStatus.textContent = ntsc.error || '';
  syncNtscMetrics(ntsc.live_metrics);
  for (const [param, value] of Object.entries(ntsc)) {
    const row = document.querySelector(`.param-row[data-ntsc="${param}"]`);
    if (!row) continue;

    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const select = row.querySelector('select');

    if (slider && valueEl && canSync(slider)) {
      slider.value = value;
      const min = parseFloat(row.dataset.min);
      const max = parseFloat(row.dataset.max);
      const step = parseFloat(row.dataset.step);
      valueEl.textContent = formatValue(value, min, max, step);
    }

    if (checkbox && canSync(checkbox)) {
      checkbox.checked = !!value;
    }

    if (select && canSync(select)) {
      select.value = value;
    }
  }
}

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MiB';
  return `${(bytes / (1024 * 1024)).toFixed(bytes >= 1024 * 1024 * 1024 ? 1 : 0)} MiB`;
}

function syncNtscMetrics(metrics) {
  const el = document.getElementById('ntsc-metrics');
  if (!el) return;
  const path = metrics?.active_path || 'off';
  const global = metrics?.global || {};
  const totalObserved = Number(global.attempted || 0) + Number(global.stale || 0);
  if (path === 'off' && totalObserved === 0) {
    if (el.textContent !== 'Live worker: no samples yet') el.textContent = 'Live worker: no samples yet';
    return;
  }
  const formatBucket = (label, bucket, active) => {
    const attempted = Number(bucket?.attempted || 0);
    const accepted = Number(bucket?.accepted || 0);
    const skipped = Number(bucket?.skipped || 0);
    const unavailable = Number(bucket?.unavailable || 0);
    const stale = Number(bucket?.stale || 0);
    const rate = attempted > 0 ? ` (${(skipped * 100 / attempted).toFixed(1)}%)` : '';
    const busy = active && metrics?.busy ? ' · busy' : '';
    const failed = unavailable > 0 ? ` · ${formatCount(unavailable)} unavailable` : '';
    return `${label} · ${formatCount(accepted)}/${formatCount(attempted)} admitted · ${formatCount(skipped)} skipped${rate}${failed} · ${formatCount(stale)} stale${busy}`;
  };
  const text = formatBucket('Final-program live', global, path === 'global');
  if (el.textContent !== text) el.textContent = text;
}

function syncMediaSafety(safety) {
  if (!safety || !expertMediaToggle) return;
  const mode = safety.mode === 'expert' ? 'expert' : 'safe';
  authoritativeMediaSafetyMode = mode;
  expertMediaToggle.checked = mode === 'expert';
  expertMediaToggle.toggleAttribute('aria-busy', false);
  mediaSafetyMode.textContent = mode.toUpperCase();
  mediaSafetyMode.classList.toggle('expert', mode === 'expert');

  const pixels = formatCount(mode === 'expert' ? safety.expert_max_pixels : safety.safe_max_pixels);
  const rgba = formatBytes(mode === 'expert' ? safety.expert_max_rgba_bytes : safety.safe_max_rgba_bytes);
  const edge = formatCount(safety.device_max_texture_dimension_2d || safety.absolute_max_edge);
  mediaSafetySummary.textContent = `${pixels} pixels / ${rgba} RGBA · ${edge}px device edge`;
  const budget = formatBytes(safety.planning_budget_bytes);
  const reserved = formatBytes(safety.reserved_bytes);
  mediaSafetyRationale.textContent = mode === 'expert'
    ? `Expert source planning: ${reserved} reserved of ${budget}. Portable VRAM totals are unavailable, so texture creation may still reject a source safely.`
    : `Safe defaults cap each source to UHD area. Expert mode affects future source opens only and remains bounded by device and ${budget} host planning limits.`;
  const status = String(safety.status || '');
  mediaSafetyStatus.textContent = status;
  mediaSafetyStatus.classList.toggle(
    'error',
    /^(rejected\b|error:|expert mode unavailable)/i.test(status)
  );
}

function syncNewLayerFit(value) {
  if (!newLayerFit) return;
  const fit = ['stretch', 'fit', 'fill', 'native'].includes(value) ? value : 'fit';
  authoritativeNewLayerFit = fit;
  if (canSync(newLayerFit)) newLayerFit.value = fit;
}

function syncProxySettings(settings) {
  if (!proxyScale || !proxyFrameRate || !proxyIncludeAudio) return;
  const scale = ['original', 'half', 'quarter'].includes(settings?.scale) ? settings.scale : 'half';
  let rateKey = 'source';
  const fixed = settings?.frame_rate?.fixed;
  if (fixed) {
    const numerator = Number(fixed.numerator) || 0;
    const denominator = Number(fixed.denominator) || 0;
    if (numerator > 0 && denominator > 0) {
      rateKey = denominator === 1 && PROXY_FRAME_RATE_PRESETS.has(String(numerator))
        ? String(numerator)
        : `fixed:${numerator}/${denominator}`;
    }
  }
  const includeAudio = settings?.include_audio !== false;
  authoritativeProxySettings = { scale, rateKey, includeAudio };
  if (canSync(proxyScale)) proxyScale.value = scale;
  if (canSync(proxyFrameRate)) setProxyFrameRateSelect(rateKey);
  if (canSync(proxyIncludeAudio)) proxyIncludeAudio.checked = includeAudio;
}

expertMediaToggle?.addEventListener('change', () => {
  const mode = expertMediaToggle.checked ? 'expert' : 'safe';
  if (mode === 'expert') {
    const accepted = window.confirm(
      'Expert large-media mode can use substantially more CPU and GPU memory. It applies only to future source opens, remains bounded by this host, and does not raise the existing UHD-area export-output limit. Enable it?'
    );
    if (!accepted) {
      expertMediaToggle.checked = authoritativeMediaSafetyMode === 'expert';
      return;
    }
  }
  if (sendAction({ action: 'set_media_safety_mode', mode })) {
    expertMediaToggle.toggleAttribute('aria-busy', true);
    mediaSafetyStatus.textContent = mode === 'expert'
      ? 'Enabling bounded Expert mode…'
      : 'Returning future source opens to Safe mode…';
    mediaSafetyStatus.classList.remove('error');
  } else {
    expertMediaToggle.checked = authoritativeMediaSafetyMode === 'expert';
    mediaSafetyStatus.textContent = 'Control connection is offline; media safety mode was not changed.';
    mediaSafetyStatus.classList.add('error');
  }
});

// --- Sync layers ---

function syncLayers(layers) {
  if (!layers) return;
  latestLayers = layers;
  latestLayerIdentities = layers.map((layer) => String(layer.layer_id || ''));
  const liveDisclosureKeys = new Set(layers.map(layerDisclosureKey));
  for (const key of layerDisclosureState.keys()) {
    if (!liveDisclosureKeys.has(key)) layerDisclosureState.delete(key);
  }
  for (const key of spatialTransformUiState.keys()) {
    if (!String(key).startsWith('group:') && !liveDisclosureKeys.has(key)) {
      spatialTransformUiState.delete(key);
    }
  }
  syncExportAudioLayers(layers);
  syncLibrarySlotTargets(layers);
  layersEmpty.style.display = layers.length === 0 ? 'block' : 'none';

  const layerKey = JSON.stringify(layers.map((layer) => [
    layer.layer_id,
    layer.filename,
    layer.source_kind,
    (layer.performance?.slots || []).map((slot) => [slot.id, slot.filename]),
  ]));
  // Rebuild when identity/order changes, including same-count replacement.
  if (layersList.children.length !== layers.length || layersList.dataset.layerKey !== layerKey) {
    layersList.innerHTML = '';
    layersList.dataset.layerKey = layerKey;
    layers.forEach((layer, i) => {
      layersList.appendChild(createLayerCard(layer, i));
    });
  } else {
    layers.forEach((layer, i) => {
      updateLayerCard(layersList.children[i], layer, i);
    });
  }
}

function syncLibrarySlotTargets(layers) {
  if (!librarySlotTarget) return;
  const key = JSON.stringify(layers.map((layer) => [
    layer.layer_id,
    layer.filename,
    (layer.performance?.slots || []).map((slot) => [slot.id, slot.name, slot.filename]),
  ]));
  if (librarySlotTarget.dataset.layersKey === key) return;
  const previous = librarySlotTarget.value;
  librarySlotTarget.dataset.layersKey = key;
  librarySlotTarget.replaceChildren(new Option('Choose a stable layer…', ''));
  layers.forEach((layer, index) => {
    const layerId = stableLayerId(layer);
    if (!layerId) return;
    const group = document.createElement('optgroup');
    group.label = `Layer ${index + 1} [${layerId}] · ${layer.filename || 'Untitled'}`;
    group.appendChild(new Option('New slot', `${layerId}:new`));
    for (const slot of layer.performance?.slots || []) {
      group.appendChild(new Option(
        `Replace slot ${slot.id} · ${slot.name || slot.filename || 'Untitled'}`,
        `${layerId}:${slot.id}`,
      ));
    }
    librarySlotTarget.appendChild(group);
  });
  librarySlotTarget.value = Array.from(librarySlotTarget.options).some((option) => option.value === previous)
    ? previous
    : '';
}

const AUTOPILOT_MAX_STEPS = 128;
const AUTOPILOT_MAX_HOLD_BEATS = 256;

function validAutopilotSceneId(value) {
  const sceneId = Number(value);
  return Number.isInteger(sceneId) && sceneId >= 1 && sceneId <= 65535 ? sceneId : null;
}

function normalizeAutopilotPlan(plan = {}) {
  const repeat = plan?.repeat === 'once' ? 'once' : 'loop';
  const steps = Array.isArray(plan?.steps) ? plan.steps.slice(0, AUTOPILOT_MAX_STEPS)
    .map((step) => ({
      scene_id: validAutopilotSceneId(step?.scene_id),
      hold_beats: Number(step?.hold_beats),
    }))
    .filter((step) => step.scene_id !== null)
    .map((step) => ({
      scene_id: step.scene_id,
      hold_beats: Number.isInteger(step.hold_beats)
        && step.hold_beats >= 1
        && step.hold_beats <= AUTOPILOT_MAX_HOLD_BEATS
        ? step.hold_beats
        : 4,
    })) : [];
  return { repeat, steps };
}

function autopilotPlanKey(plan) {
  return JSON.stringify(normalizeAutopilotPlan(plan));
}

function cloneAutopilotPlan(plan) {
  const clean = normalizeAutopilotPlan(plan);
  return {
    repeat: clean.repeat,
    steps: clean.steps.map((step) => ({ ...step })),
  };
}

function autopilotSceneCatalog(scenes = latestAutopilotScenes) {
  const catalog = new Map();
  for (const scene of scenes) {
    const sceneId = validAutopilotSceneId(scene?.id);
    if (sceneId === null) continue;
    catalog.set(sceneId, scene?.name || `Scene ${sceneId}`);
  }
  return catalog;
}

function autopilotStepName(plan, index, scenes = latestAutopilotScenes) {
  const step = plan?.steps?.[index];
  if (!step) return '—';
  const sceneId = validAutopilotSceneId(step.scene_id);
  if (sceneId === null) return '—';
  const name = autopilotSceneCatalog(scenes).get(sceneId);
  return name ? `${index + 1}: ${name} [${sceneId}]` : `${index + 1}: Missing Scene ${sceneId}`;
}

function setAutopilotEditorStatus(message, error = false) {
  if (!autopilotStatus) return;
  autopilotStatus.textContent = message;
  autopilotStatus.classList.toggle('error', error);
}

function markAutopilotPlanDirty() {
  autopilotPlanDirty = true;
  autopilotPendingPlanKey = null;
  autopilotStepList?.querySelectorAll('.autopilot-step-row').forEach((row) => {
    row.classList.remove('current', 'next');
  });
  setAutopilotEditorStatus('Local sequence edits are not applied yet.');
}

function readAutopilotDraftFromControls(validate = false) {
  const repeat = autopilotRepeat?.value === 'once' ? 'once' : 'loop';
  const rows = Array.from(autopilotStepList?.querySelectorAll('.autopilot-step-row') || []);
  if (validate && rows.length > AUTOPILOT_MAX_STEPS) {
    return { error: `An Autopilot may contain at most ${AUTOPILOT_MAX_STEPS} steps.` };
  }
  const steps = [];
  for (const [index, row] of rows.entries()) {
    const sceneId = validAutopilotSceneId(row.querySelector('.autopilot-scene-select')?.value);
    const rawBeats = row.querySelector('.autopilot-hold-beats')?.value ?? '';
    const beats = Number(rawBeats);
    if (validate && sceneId === null) {
      return { error: `Step ${index + 1} must retain a valid non-zero Scene ID.` };
    }
    if (validate && (!Number.isInteger(beats) || beats < 1 || beats > AUTOPILOT_MAX_HOLD_BEATS)) {
      return { error: `Step ${index + 1} beats must be an integer from 1 to ${AUTOPILOT_MAX_HOLD_BEATS}.` };
    }
    steps.push({
      scene_id: sceneId,
      // Preserve an invalid in-progress edit across row moves. The strict
      // Apply boundary above is the only place it may approach the wire.
      hold_beats: Number.isInteger(beats) ? beats : rawBeats,
    });
  }
  return { plan: { repeat, steps } };
}

function adoptAutopilotControlsIntoDraft() {
  const read = readAutopilotDraftFromControls(false);
  if (read.plan) autopilotDraft = read.plan;
}

function moveAutopilotStep(index, delta) {
  adoptAutopilotControlsIntoDraft();
  const next = index + delta;
  if (next < 0 || next >= autopilotDraft.steps.length) return;
  [autopilotDraft.steps[index], autopilotDraft.steps[next]] = [
    autopilotDraft.steps[next],
    autopilotDraft.steps[index],
  ];
  markAutopilotPlanDirty();
  renderAutopilotPlanEditor();
  autopilotStepList?.querySelector(`[data-step-index="${next}"] .autopilot-scene-select`)?.focus();
}

function removeAutopilotStep(index) {
  adoptAutopilotControlsIntoDraft();
  autopilotDraft.steps.splice(index, 1);
  markAutopilotPlanDirty();
  renderAutopilotPlanEditor();
}

function renderAutopilotPlanEditor() {
  if (!autopilotStepList || !autopilotRepeat) return;
  const snapshot = latestAutopilotSnapshot || {};
  const current = !autopilotPlanDirty && Number.isInteger(snapshot.current_step)
    ? snapshot.current_step : null;
  const next = !autopilotPlanDirty && Number.isInteger(snapshot.next_step)
    ? snapshot.next_step : null;
  const catalog = autopilotSceneCatalog();
  const sceneKey = JSON.stringify(Array.from(catalog.entries()));
  const renderKey = JSON.stringify({
    draft: autopilotDraft,
    scenes: sceneKey,
    current,
    next,
  });
  if (autopilotStepList.dataset.renderKey === renderKey) return;
  autopilotStepList.dataset.renderKey = renderKey;
  autopilotRepeat.value = autopilotDraft.repeat === 'once' ? 'once' : 'loop';
  autopilotStepList.replaceChildren();

  autopilotDraft.steps.forEach((step, index) => {
    const sceneId = validAutopilotSceneId(step.scene_id);
    const missing = sceneId !== null && !catalog.has(sceneId);
    const row = document.createElement('article');
    row.className = `autopilot-step-row${index === current ? ' current' : ''}${index === next ? ' next' : ''}${missing ? ' missing' : ''}`;
    row.dataset.stepIndex = String(index);
    if (sceneId !== null) row.dataset.sceneId = String(sceneId);
    row.setAttribute('role', 'listitem');

    const ordinal = document.createElement('span');
    ordinal.className = 'autopilot-step-number';
    ordinal.textContent = String(index + 1);
    ordinal.setAttribute('aria-hidden', 'true');

    const sceneLabel = document.createElement('label');
    sceneLabel.className = 'visually-hidden';
    sceneLabel.htmlFor = `autopilot-scene-${index}`;
    sceneLabel.textContent = `Autopilot step ${index + 1} Scene`;
    const sceneSelect = document.createElement('select');
    sceneSelect.id = `autopilot-scene-${index}`;
    sceneSelect.className = 'autopilot-scene-select';
    sceneSelect.setAttribute('aria-label', `Autopilot step ${index + 1} Scene`);
    if (missing) {
      const tombstone = new Option(`Missing Scene ${sceneId} (kept)`, String(sceneId));
      tombstone.dataset.tombstone = '';
      sceneSelect.appendChild(tombstone);
    }
    for (const [candidateId, name] of catalog) {
      sceneSelect.appendChild(new Option(`${name} [${candidateId}]`, String(candidateId)));
    }
    if (sceneId !== null) sceneSelect.value = String(sceneId);
    sceneSelect.addEventListener('change', () => {
      const selected = validAutopilotSceneId(sceneSelect.value);
      if (selected !== null) autopilotDraft.steps[index].scene_id = selected;
      markAutopilotPlanDirty();
      renderAutopilotPlanEditor();
    });

    const beatsLabel = document.createElement('label');
    beatsLabel.className = 'autopilot-beats-label';
    beatsLabel.htmlFor = `autopilot-hold-${index}`;
    beatsLabel.textContent = 'Beats';
    const beatsInput = document.createElement('input');
    beatsInput.id = `autopilot-hold-${index}`;
    beatsInput.className = 'autopilot-hold-beats';
    beatsInput.type = 'number';
    beatsInput.min = '1';
    beatsInput.max = String(AUTOPILOT_MAX_HOLD_BEATS);
    beatsInput.step = '1';
    beatsInput.inputMode = 'numeric';
    beatsInput.value = String(step.hold_beats ?? 4);
    beatsInput.setAttribute('aria-label', `Autopilot step ${index + 1} hold beats`);
    beatsInput.addEventListener('input', () => {
      const beats = Number(beatsInput.value);
      autopilotDraft.steps[index].hold_beats = Number.isInteger(beats) ? beats : beatsInput.value;
      markAutopilotPlanDirty();
    });

    const actions = document.createElement('div');
    actions.className = 'autopilot-step-actions';
    actions.setAttribute('role', 'group');
    actions.setAttribute('aria-label', `Autopilot step ${index + 1} order commands`);
    const up = document.createElement('button');
    up.type = 'button';
    up.className = 'autopilot-step-up';
    up.textContent = '↑';
    up.disabled = index === 0;
    up.setAttribute('aria-label', `Move Autopilot step ${index + 1} up`);
    up.addEventListener('click', () => moveAutopilotStep(index, -1));
    const down = document.createElement('button');
    down.type = 'button';
    down.className = 'autopilot-step-down';
    down.textContent = '↓';
    down.disabled = index + 1 === autopilotDraft.steps.length;
    down.setAttribute('aria-label', `Move Autopilot step ${index + 1} down`);
    down.addEventListener('click', () => moveAutopilotStep(index, 1));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'autopilot-step-remove';
    remove.textContent = '×';
    remove.setAttribute('aria-label', `Remove Autopilot step ${index + 1}`);
    remove.addEventListener('click', () => removeAutopilotStep(index));
    actions.append(up, down, remove);

    row.append(ordinal, sceneLabel, sceneSelect, beatsLabel, beatsInput, actions);
    autopilotStepList.appendChild(row);
  });

  if (autopilotDraft.steps.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'autopilot-empty';
    empty.textContent = catalog.size
      ? 'No steps. Add a Scene to author the sequence.'
      : 'Capture a Scene before adding an Autopilot step.';
    autopilotStepList.appendChild(empty);
  }
}

function syncAutopilot(snapshot = {}, scenes = []) {
  latestAutopilotSnapshot = snapshot || {};
  latestAutopilotScenes = Array.isArray(scenes) ? scenes : [];
  const authoritativePlan = normalizeAutopilotPlan(snapshot?.plan);
  const authoritativeKey = autopilotPlanKey(authoritativePlan);
  if (autopilotPendingPlanKey && authoritativeKey === autopilotPendingPlanKey) {
    autopilotPlanDirty = false;
    autopilotPendingPlanKey = null;
  }
  if (!autopilotPlanDirty) autopilotDraft = cloneAutopilotPlan(authoritativePlan);
  renderAutopilotPlanEditor();

  const phase = ['stopped', 'starting', 'running', 'paused', 'stalled', 'faulted', 'complete']
    .includes(snapshot?.phase) ? snapshot.phase : 'stopped';
  const current = Number.isInteger(snapshot?.current_step) ? snapshot.current_step : null;
  const next = Number.isInteger(snapshot?.next_step) ? snapshot.next_step : null;
  if (autopilotPhase) autopilotPhase.textContent = phase.replace('_', ' ');
  if (autopilotCurrent) autopilotCurrent.textContent = autopilotStepName(authoritativePlan, current, scenes);
  if (autopilotNext) autopilotNext.textContent = autopilotStepName(authoritativePlan, next, scenes);
  if (autopilotBeats) {
    autopilotBeats.textContent = Number.isInteger(snapshot?.beats_remaining)
      ? String(snapshot.beats_remaining)
      : '—';
  }
  if (autopilotPlay) autopilotPlay.disabled = authoritativePlan.steps.length === 0;
  if (autopilotPause) autopilotPause.disabled = !['starting', 'running', 'stalled'].includes(phase);
  if (autopilotReset) autopilotReset.disabled = phase === 'stopped' && current === null;

  const currentSceneId = current === null ? null : authoritativePlan.steps[current]?.scene_id;
  const nextSceneId = next === null ? null : authoritativePlan.steps[next]?.scene_id;
  sceneList?.querySelectorAll('.scene-row').forEach((row) => {
    const sceneId = validAutopilotSceneId(row.dataset.sceneId);
    row.classList.toggle('autopilot-current', sceneId !== null && sceneId === currentSceneId);
    row.classList.toggle('autopilot-next', sceneId !== null && sceneId === nextSceneId);
  });

  if (!autopilotPlanDirty) {
    const fallback = authoritativePlan.steps.length
      ? `${authoritativePlan.steps.length} step${authoritativePlan.steps.length === 1 ? '' : 's'} · ${authoritativePlan.repeat}`
      : 'No authored Autopilot sequence';
    setAutopilotEditorStatus(snapshot?.status || fallback, phase === 'faulted');
  }
}

function syncPerformance(performanceState = {}) {
  if (!sceneList || !sceneStatus) return;
  const scenes = Array.isArray(performanceState?.scenes) ? performanceState.scenes : [];
  const key = JSON.stringify({
    layers: latestLayers.map((layer) => [layer.layer_id, layer.filename]),
    scenes: scenes.map((scene) => [
      scene.id,
      scene.name,
      scene.trigger_mode,
      scene.prepared,
      scene.pending,
      scene.status,
      scene.bindings,
    ]),
  });
  if (sceneList.dataset.sceneKey !== key) {
    sceneList.dataset.sceneKey = key;
    sceneList.replaceChildren();
    const layerNames = new Map(latestLayers.map((layer, index) => [
      stableLayerId(layer),
      `L${index + 1} [${stableLayerId(layer) || 'missing'}]`,
    ]));
    for (const scene of scenes) {
      const sceneId = Number(scene.id);
      if (!Number.isInteger(sceneId) || sceneId <= 0 || sceneId > 65535) continue;
      const row = document.createElement('article');
      row.className = `scene-row${scene.prepared ? ' prepared' : ''}${scene.pending ? ' pending' : ''}`;
      row.dataset.sceneId = String(sceneId);
      row.setAttribute('role', 'listitem');
      const bindings = (scene.bindings || []).map((binding) => {
        const donor = layerNames.get(String(binding.layer_id)) || `missing ${binding.layer_id}`;
        const cue = binding.cue_id === null || binding.cue_id === undefined ? '' : ` @ cue ${binding.cue_id}`;
        return `${donor} → slot ${binding.slot_id}${cue}`;
      }).join(' · ');
      const displayName = scene.name || `Scene ${sceneId}`;
      const authoredMode = ['immediate', 'next_beat', 'next_bar'].includes(scene.trigger_mode)
        ? scene.trigger_mode
        : 'immediate';
      row.innerHTML = `
        <div class="scene-copy">
          <label class="visually-hidden" for="scene-name-${sceneId}">Scene ${sceneId} name</label>
          <input id="scene-name-${sceneId}" class="scene-name" type="text" maxlength="128" autocomplete="off" spellcheck="false" value="${escapeHtml(scene.name || '')}" placeholder="Scene ${sceneId}" aria-label="Scene ${sceneId} name">
          <span>${escapeHtml(bindings || 'No bindings')}</span>
          ${scene.status ? `<span class="scene-diagnostic">${escapeHtml(scene.status)}</span>` : ''}
        </div>
        <div class="scene-timing">
          <label for="scene-mode-${sceneId}">Timing</label>
          <select id="scene-mode-${sceneId}" class="scene-mode" aria-label="${escapeHtml(displayName)} trigger timing">
            <option value="immediate"${authoredMode === 'immediate' ? ' selected' : ''}>Immediate</option>
            <option value="next_beat"${authoredMode === 'next_beat' ? ' selected' : ''}>Next beat</option>
            <option value="next_bar"${authoredMode === 'next_bar' ? ' selected' : ''}>Next bar</option>
          </select>
        </div>
        <div class="scene-actions" role="group" aria-label="${escapeHtml(displayName)} commands">
          <button type="button" class="scene-prepare" aria-label="Prepare ${escapeHtml(displayName)}">Prepare</button>
          <button type="button" class="scene-trigger" aria-label="Trigger ${escapeHtml(displayName)}">Trigger</button>
          <button type="button" class="scene-recapture" aria-label="Recapture ${escapeHtml(displayName)} from current active slots">Recapture</button>
          <button type="button" class="scene-remove" aria-label="Remove ${escapeHtml(displayName)}">Remove</button>
        </div>`;
      row.querySelector('.scene-prepare').addEventListener('click', () => {
        if (sendAction({ action: 'prepare_scene', scene_id: sceneId })) {
          sceneStatus.textContent = `Preparing ${displayName}…`;
          sceneStatus.classList.remove('error');
        }
      });
      row.querySelector('.scene-trigger').addEventListener('click', () => {
        const triggerMode = sceneTriggerModeFrom(row.querySelector('.scene-mode'));
        if (!triggerMode) return;
        if (sendAction({
          action: 'trigger_scene',
          scene_id: sceneId,
          trigger_mode: triggerMode,
        })) {
          sceneStatus.textContent = `Triggering ${displayName} (${triggerMode.replace('_', ' ')})…`;
          sceneStatus.classList.remove('error');
        }
      });
      row.querySelector('.scene-recapture').addEventListener('click', () => {
        const name = sceneNameFrom(row.querySelector('.scene-name'));
        const triggerMode = sceneTriggerModeFrom(row.querySelector('.scene-mode'));
        if (name === null || !triggerMode) return;
        if (sendAction({
          action: 'capture_scene',
          scene_id: sceneId,
          name,
          trigger_mode: triggerMode,
        })) {
          sceneStatus.textContent = `Recapturing ${name || `Scene ${sceneId}`} from the current active slots…`;
          sceneStatus.classList.remove('error');
        } else {
          sceneStatus.textContent = 'Control connection is offline; the scene was not recaptured.';
          sceneStatus.classList.add('error');
        }
      });
      row.querySelector('.scene-remove').addEventListener('click', () => {
        if (sendAction({ action: 'remove_scene', scene_id: sceneId })) {
          sceneStatus.textContent = `Removing ${displayName}…`;
          sceneStatus.classList.remove('error');
        } else {
          sceneStatus.textContent = 'Control connection is offline; the scene was not removed.';
          sceneStatus.classList.add('error');
        }
      });
      sceneList.appendChild(row);
    }
  }
  sceneList.hidden = scenes.length === 0;
  const diagnostics = [
    performanceState?.status,
    performanceState?.scene_staging_status,
    performanceState?.source_staging_status,
    performanceState?.image_routing_status,
  ].filter(Boolean);
  sceneStatus.textContent = diagnostics.join(' · ')
    || (scenes.length ? `${scenes.length} authored scene${scenes.length === 1 ? '' : 's'}` : 'No authored scenes');
  sceneStatus.classList.toggle('error', !!performanceState?.image_routing_status);
  syncAutopilot(performanceState?.autopilot || {}, scenes);
}

sceneCaptureForm?.addEventListener('submit', (event) => {
  event.preventDefault();
  const name = sceneNameFrom(sceneCaptureName);
  const triggerMode = sceneTriggerModeFrom(sceneCaptureMode);
  if (name === null || !triggerMode) return;
  if (latestLayers.length === 0) {
    sceneStatus.textContent = 'Add at least one live layer before capturing a scene.';
    sceneStatus.classList.add('error');
    return;
  }
  if (sendAction({ action: 'capture_scene', name, trigger_mode: triggerMode })) {
    sceneStatus.textContent = `Capturing ${name || 'a new scene'} from the current active slots…`;
    sceneStatus.classList.remove('error');
    sceneCaptureName.value = '';
  } else {
    sceneStatus.textContent = 'Control connection is offline; the scene was not captured.';
    sceneStatus.classList.add('error');
  }
});

autopilotRepeat?.addEventListener('change', () => {
  autopilotDraft.repeat = autopilotRepeat.value === 'once' ? 'once' : 'loop';
  markAutopilotPlanDirty();
});

autopilotAddStep?.addEventListener('click', () => {
  adoptAutopilotControlsIntoDraft();
  if (autopilotDraft.steps.length >= AUTOPILOT_MAX_STEPS) {
    setAutopilotEditorStatus(`An Autopilot may contain at most ${AUTOPILOT_MAX_STEPS} steps.`, true);
    return;
  }
  const firstSceneId = autopilotSceneCatalog().keys().next().value;
  if (!firstSceneId) {
    setAutopilotEditorStatus('Capture at least one Scene before adding a step.', true);
    return;
  }
  autopilotDraft.steps.push({ scene_id: firstSceneId, hold_beats: 4 });
  markAutopilotPlanDirty();
  renderAutopilotPlanEditor();
  const added = autopilotDraft.steps.length - 1;
  autopilotStepList?.querySelector(`[data-step-index="${added}"] .autopilot-scene-select`)?.focus();
});

autopilotPlanForm?.addEventListener('submit', (event) => {
  event.preventDefault();
  const read = readAutopilotDraftFromControls(true);
  if (read.error) {
    setAutopilotEditorStatus(read.error, true);
    return;
  }
  const plan = read.plan;
  if (!sendAction({ action: 'replace_autopilot_plan', plan })) {
    setAutopilotEditorStatus('Control connection is offline; the sequence was not applied.', true);
    return;
  }
  autopilotDraft = cloneAutopilotPlan(plan);
  autopilotPlanDirty = true;
  autopilotPendingPlanKey = autopilotPlanKey(plan);
  setAutopilotEditorStatus(
    plan.steps.length
      ? `Applying ${plan.steps.length} Autopilot step${plan.steps.length === 1 ? '' : 's'}…`
      : 'Clearing the authored Autopilot sequence…',
  );
});

function sendAutopilotTransport(action, pendingMessage) {
  if (action === 'autopilot_play' && autopilotPlanDirty) {
    setAutopilotEditorStatus('Apply the local sequence edits before pressing Play.', true);
    return;
  }
  if (sendAction({ action })) {
    setAutopilotEditorStatus(pendingMessage);
  } else {
    setAutopilotEditorStatus('Control connection is offline; Autopilot transport was not changed.', true);
  }
}

autopilotPlay?.addEventListener('click', () => {
  sendAutopilotTransport('autopilot_play', 'Starting or resuming Autopilot…');
});
autopilotPause?.addEventListener('click', () => {
  sendAutopilotTransport('autopilot_pause', 'Pausing Autopilot…');
});
autopilotReset?.addEventListener('click', () => {
  sendAutopilotTransport('autopilot_reset', 'Resetting the Autopilot cursor…');
});

function syncExportAudioLayers(layers) {
  const select = document.getElementById('export-audio');
  if (!select) return;
  const key = JSON.stringify(layers.map((layer) => [layer.layer_id, layer.filename, layer.source_kind]));
  if (select.dataset.layerKey === key) return;
  const previous = select.value;
  select.dataset.layerKey = key;
  select.innerHTML = '<option value="">None</option>' + layers
    .map((layer, index) => ({ layer, index }))
    .filter(({ layer }) => layer.source_kind === 'video')
    .map(({ layer, index }) => `<option value="${escapeHtml(layer.layer_id || `legacy-index:${index}`)}" data-index="${index}">Layer ${index + 1}: ${escapeHtml(layer.filename)}</option>`)
    .join('');
  if (Array.from(select.options).some((option) => option.value === previous)) {
    select.value = previous;
  } else {
    select.value = '';
  }
}

let layerDrag = null;

function clearLayerDrag() {
  document.querySelectorAll('.layer-card.reorder-target').forEach((card) => {
    card.classList.remove('reorder-target');
  });
  layerDrag = null;
}

function updateLayerDragTarget(clientX, clientY) {
  if (!layerDrag) return;
  const card = document.elementFromPoint(clientX, clientY)?.closest('.layer-card');
  if (!card || !layersList.contains(card)) return;
  const target = Number.parseInt(card.dataset.index, 10);
  if (!Number.isInteger(target)) return;
  layerDrag.to = target;
  document.querySelectorAll('.layer-card.reorder-target').forEach((candidate) => {
    candidate.classList.toggle('reorder-target', candidate === card);
  });
}

const LAYER_EFFECT_CONTROLS = [
  ['pixelate', 'Pixelate', 'range', 1, 32, 1, 1],
  ['downsample', 'Downsample', 'range', 0.05, 1, 0.01, 1],
  ['rgb_split', 'RGB Split', 'range', 0, 30, 0.5, 0],
  ['hue_shift', 'Hue', 'range', -180, 180, 1, 0],
  ['saturation', 'Saturation', 'range', -1, 1, 0.01, 0],
  ['brightness', 'Brightness', 'range', -1, 1, 0.01, 0],
  ['contrast', 'Contrast', 'range', -1, 1, 0.01, 0],
  ['posterize', 'Posterize', 'range', 0, 16, 1, 0],
  ['invert', 'Invert', 'checkbox', 0, 1, 1, false],
  ['grain_intensity', 'Grain', 'range', 0, 0.3, 0.005, 0],
  ['grain_size', 'Grain Size', 'range', 1, 4, 0.25, 1],
  ['grain_algo', 'Grain Algo', 'select', 0, 3, 1, 0],
  ['color_grain', 'Color Grain', 'checkbox', 0, 1, 1, false],
  ['vignette', 'Vignette', 'range', 0, 1.5, 0.01, 0],
  ['color_drift', 'Drift', 'range', 0, 0.02, 0.001, 0],
  ['breathe_scale', 'Bth Scale', 'range', 0, 0.05, 0.001, 0],
  ['breathe_rotation', 'Bth Rotate', 'range', 0, 2, 0.05, 0],
  ['breathe_position', 'Bth Drift', 'range', 0, 0.02, 0.001, 0],
  ['shift_amount', 'Shift Amt', 'range', 0, 1, 0.01, 0],
  ['shift_block_size', 'Shift Block', 'range', 2, 256, 1, 8],
  ['shift_density', 'Shift Density', 'range', 0, 1, 0.01, 0.5],
  ['shift_speed', 'Shift Speed', 'range', 0, 20, 0.25, 3],
  ['cellular_amount', 'Amount', 'range', 0, 1, 0.05, 0],
  ['cellular_scale', 'Scale', 'range', 2, 32, 1, 10],
  ['cellular_warp', 'Warp', 'range', 0, 1, 0.05, 0.35],
  ['cellular_speed', 'Drift', 'range', 0, 2, 0.05, 0.25],
  ['cellular_gap_amount', 'Gap Key', 'range', 0, 1, 0.01, 0],
  ['cellular_gap_threshold', 'Gap Thresh', 'range', 0, 1, 0.01, 0.65],
  ['cellular_gap_softness', 'Gap Soft', 'range', 0, 0.5, 0.01, 0.08],
  // B13 small effects. The three master-only optics are deliberately absent
  // from layer cards.
  ['contour', 'Contour', 'range', 0, 1, 0.01, 0],
  ['contour_bands', 'Bands', 'range', 2, 40, 1, 10],
  ['contour_width', 'Line Width', 'range', 0.2, 6, 0.1, 1.2],
  ['contour_hue', 'Line Hue', 'range', 0, 1, 0.01, 0],
  ['contour_fill', 'Keep Fill', 'range', 0, 1, 0.01, 0.25],
  ['flatten', 'Flatten', 'range', 0, 1, 0.01, 0],
  ['flatten_levels', 'Levels', 'range', 2, 16, 1, 5],
  ['contour_dither', 'Dither', 'range', 0, 1, 0.01, 0],
  ['solarize', 'Solarize', 'range', 0, 1, 0.01, 0],
  ['negative', 'Negative', 'range', 0, 1, 0.01, 0],
  ['negative_mode', 'Neg Mode', 'select', 0, 2, 1, 0],
  ['colourpass', 'Colourpass', 'range', 0, 1, 0.01, 0],
  ['colourpass_hue', 'Pass Hue', 'range', -180, 180, 1, 0],
  ['colourpass_width', 'Pass Width', 'range', 0, 1, 0.01, 0.25],
  ['edge_amount', 'Find Edge', 'range', 0, 1, 0.01, 0],
  ['edge_hue', 'Edge Hue', 'range', -180, 180, 1, 0],
  ['emboss', 'Emboss', 'range', 0, 1, 0.01, 0],
  ['emboss_angle', 'Emboss Dir', 'range', -180, 180, 1, 45],
  ['halftone', 'Halftone', 'range', 0, 1, 0.01, 0],
  ['halftone_pitch', 'Dot Pitch', 'range', 0, 1, 0.01, 0.4],
  ['halftone_angle', 'Dot Angle', 'range', -180, 180, 1, 0],
  ['moire', 'Moire', 'range', 0, 1, 0.01, 0],
  ['moire_freq', 'Moire Freq', 'range', 0, 1, 0.01, 0.4],
  ['row_smear', 'Row Smear', 'range', 0, 1, 0.01, 0],
  ['bitcrush', 'Bitcrush', 'range', 0, 1, 0.01, 0],
  ['bitcrush_levels', 'Crush Levels', 'range', 2, 16, 1, 2],
  ['bitcrush_dither', 'Crush Dither', 'range', 0, 1, 0.01, 1],
  ['multi_grid_x', 'Grid X', 'range', 1, 8, 1, 1],
  ['multi_grid_y', 'Grid Y', 'range', 1, 8, 1, 1],
];

const LAYER_EFFECT_SELECT_OPTIONS = {
  grain_algo: [['0', 'Gaussian'], ['1', 'Perlin'], ['2', 'Salt &amp; Pepper'], ['3', 'Blue']],
  negative_mode: [['0', 'RGB'], ['1', 'Luma'], ['2', 'Hue Flip']],
};

function layerEffectsHtml(effects, index) {
  const rowHtml = ([param, label, kind, min, max, step, fallback]) => {
    const value = effects[param] ?? fallback;
    if (kind === 'checkbox') {
      return `<div class="param-row toggle-row layer-effect-row" data-layer-effect="${param}"><label>${label}</label><label class="toggle"><input type="checkbox" ${value ? 'checked' : ''} aria-label="Layer ${index + 1} ${label}"><span class="toggle-slider"></span></label></div>`;
    }
    if (kind === 'select') {
      const options = (LAYER_EFFECT_SELECT_OPTIONS[param] || [])
        .map(([optionValue, optionLabel]) => `<option value="${optionValue}">${optionLabel}</option>`)
        .join('');
      return `<div class="param-row select-row layer-effect-row" data-layer-effect="${param}"><label>${label}</label><select aria-label="Layer ${index + 1} ${label}">${options}</select></div>`;
    }
    return `<div class="param-row layer-effect-row" data-layer-effect="${param}" data-min="${min}" data-max="${max}" data-step="${step}"><label>${label}</label><input type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="Layer ${index + 1} ${label}"><span class="value">${formatValue(Number(value), min, max, step)}</span></div>`;
  };
  const cellularIndex = LAYER_EFFECT_CONTROLS.findIndex(([param]) => param === 'cellular_amount');
  const smallFxIndex = LAYER_EFFECT_CONTROLS.findIndex(([param]) => param === 'contour');
  const ordinary = LAYER_EFFECT_CONTROLS.slice(0, cellularIndex).map(rowHtml).join('');
  const cellular = LAYER_EFFECT_CONTROLS.slice(cellularIndex, smallFxIndex).map(rowHtml).join('');
  const smallFx = LAYER_EFFECT_CONTROLS.slice(smallFxIndex).map(rowHtml).join('');
  return `${ordinary}
    <div class="layer-cellular-disclosure">
      <button class="layer-cellular-toggle" type="button" aria-expanded="false" aria-controls="layer-cellular-body-${index}">
        <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>CELLULAR</span>
      </button>
      <div class="layer-cellular-body" id="layer-cellular-body-${index}" role="region" aria-label="Layer ${index + 1} cellular effects" hidden>${cellular}</div>
    </div>
    <div class="layer-cellular-disclosure">
      <button class="layer-cellular-toggle" type="button" aria-expanded="false" aria-controls="layer-small-fx-body-${index}">
        <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>SMALL FX</span>
      </button>
      <div class="layer-cellular-body" id="layer-small-fx-body-${index}" role="region" aria-label="Layer ${index + 1} small effects" hidden>${smallFx}</div>
    </div>`;
}

function activeClipSlot(layer) {
  const slots = layer?.performance?.slots || [];
  const requested = Number(layer?.performance?.active_slot_id);
  return slots.find((slot) => Number(slot.id) === requested)
    || slots.find((slot) => slot.active)
    || slots[0]
    || null;
}

function clipSlotOptionsHtml(layer) {
  const active = activeClipSlot(layer);
  const slots = layer?.performance?.slots || [];
  if (!slots.length) return '<option value="">No prepared slots</option>';
  return slots.map((slot) => `<option value="${Number(slot.id)}" ${Number(slot.id) === Number(active?.id) ? 'selected' : ''}>Slot ${Number(slot.id)} · ${escapeHtml(slot.name || slot.filename || 'Untitled')}${slot.prepared ? '' : ' · staging'}</option>`).join('');
}

function matteInputValue(input = {}) {
  if (input.source === 'selected_layer') {
    return `selected:${input.layer_id}:${input.stage || 'post_local_effects'}`;
  }
  if (input.source === 'missing_selected_layer') {
    return `missing:${Number(input.saved_position)}:${input.stage || 'post_local_effects'}`;
  }
  if (input.source === 'group_output') return `group:${String(input.group_id)}`;
  if (input.source === 'missing_group_output') return `missing-group:${String(input.group_id)}`;
  return input.source || 'one_below';
}

function matteInputOptionsHtml(layer) {
  const current = matteInputValue(layer?.performance?.matte?.input);
  const ordinary = [
    ['one_below', 'One below'],
    ['all_below', 'All below'],
    ['program_history', 'Program history (N−1)'],
    ['clean_program', 'Clean program (same-frame cycle diagnostic)'],
  ].map(([value, label]) => `<option value="${value}" ${current === value ? 'selected' : ''}>${label}</option>`);
  const selected = [];
  latestLayers.forEach((candidate, index) => {
    const candidateId = stableLayerId(candidate);
    if (!candidateId) return;
    for (const [stage, label] of [['pre_local_effects', 'pre local FX'], ['post_local_effects', 'post local FX']]) {
      const value = `selected:${candidateId}:${stage}`;
      selected.push(`<option value="${value}" ${current === value ? 'selected' : ''}>Layer ${index + 1} [${candidateId}] · ${escapeHtml(candidate.filename || 'Untitled')} · ${label}</option>`);
    }
  });
  const groupOutputs = (latestCreative?.groups || []).map((group) => {
    const value = `group:${String(group.group_id)}`;
    return `<option value="${escapeHtml(value)}" ${current === value ? 'selected' : ''}>Group · ${escapeHtml(group.name || String(group.group_id))}</option>`;
  });
  if (String(current).startsWith('missing:')) {
    const [, position, stage] = String(current).split(':');
    ordinary.unshift(`<option value="${escapeHtml(current)}" selected disabled>Missing saved layer ${escapeHtml(position)} · ${escapeHtml(stage)}</option>`);
  } else if (String(current).startsWith('missing-group:')) {
    ordinary.unshift(`<option value="${escapeHtml(current)}" selected disabled>Missing group ${escapeHtml(String(current).slice(14))}</option>`);
  }
  return ordinary.concat(selected, groupOutputs).join('');
}

function layerPerformanceHtml(layer, index) {
  const slot = activeClipSlot(layer);
  const transport = slot?.transport || {};
  const grid = transport.beat_grid || {};
  const beatLoop = transport.beat_loop || {};
  const cues = Array.isArray(transport.cues) ? transport.cues : [];
  const matte = layer?.performance?.matte || {};
  const cueOptions = cues.length
    ? cues.map((cue) => `<option value="${Number(cue.id)}">Cue ${Number(cue.id)} · ${Number(cue.at).toFixed(3)}</option>`).join('')
    : '<option value="">No cues</option>';
  return `
    <div class="layer-performance-heading">
      <button class="layer-performance-toggle" type="button" aria-expanded="false" aria-controls="layer-performance-body-${index}">
        <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>Slots / Transport</span>
      </button>
    </div>
    <div class="layer-performance-body" id="layer-performance-body-${index}" role="region" aria-label="Layer ${index + 1} clip slots and transport" hidden>
      <div class="slot-command-row">
        <label for="clip-slot-${index}">Prepared source</label>
        <select id="clip-slot-${index}" class="clip-slot-select">${clipSlotOptionsHtml(layer)}</select>
        <select class="clip-trigger-mode" aria-label="Layer ${index + 1} slot activation timing">
          <option value="immediate">Immediate</option><option value="next_beat">Next beat</option><option value="next_bar">Next bar</option>
        </select>
        <button type="button" class="clip-activate">Activate</button>
        <button type="button" class="clip-remove" aria-label="Remove selected prepared slot">Remove</button>
      </div>
      <div class="performance-status clip-slot-status" role="status" aria-live="polite">${escapeHtml(slot?.status || (slot?.prepared ? 'Prepared' : slot ? 'Staging…' : 'No prepared source'))}</div>
      <div class="transport-grid" ${slot ? '' : 'hidden'}>
        <label>Playhead <input class="clip-seek" type="range" min="0" max="1" step="0.001" value="${Number(slot?.playhead || 0)}" aria-label="Layer ${index + 1} clip playhead"></label>
        <fieldset class="clip-timecode-controls">
          <legend>Timecode seek</legend>
          <label>HH <input class="clip-timecode-hours" type="number" min="0" max="99" step="1" value="0" aria-label="Timecode hours"></label>
          <label>MM <input class="clip-timecode-minutes" type="number" min="0" max="59" step="1" value="0" aria-label="Timecode minutes"></label>
          <label>SS <input class="clip-timecode-seconds" type="number" min="0" max="59" step="1" value="0" aria-label="Timecode seconds"></label>
          <label>FF <input class="clip-timecode-frames" type="number" min="0" max="59" step="1" value="0" aria-label="Timecode frames"></label>
          <label>Rate <select class="clip-timecode-rate" aria-label="Timecode rate">
            <option value="fps24">24</option><option value="ntsc24">23.976 NDF</option>
            <option value="fps25">25</option><option value="fps30">30</option>
            <option value="ntsc30">29.97 NDF</option><option value="ntsc30_drop">29.97 DF</option>
            <option value="fps50">50</option><option value="fps60">60</option>
            <option value="ntsc60">59.94 NDF</option><option value="ntsc60_drop">59.94 DF</option>
          </select></label>
          <button type="button" class="clip-timecode-seek">Seek</button>
        </fieldset>
        <label>Direction <select data-clip-transport="direction"><option value="forward">Forward</option><option value="reverse">Reverse</option></select></label>
        <label>End <select data-clip-transport="end_behavior"><option value="loop">Loop</option><option value="ping_pong">Ping pong</option><option value="one_shot">One shot</option><option value="hold">Hold</option></select></label>
        <label>In <input data-clip-transport="in_point" type="number" min="0" max="1" step="0.001" value="${Number(transport.in_point ?? 0)}"></label>
        <label>Out <input data-clip-transport="out_point" type="number" min="0" max="1" step="0.001" value="${Number(transport.out_point ?? 1)}"></label>
        <label>Rate <input data-clip-transport="rate" type="number" min="0" max="16" step="0.01" value="${Number(transport.rate ?? 1)}"></label>
        <label>Sample FPS <input data-clip-transport="sample_fps" type="number" min="0.25" max="480" step="0.01" value="${transport.sample_fps ?? ''}" placeholder="source"></label>
        <label><input data-clip-transport="beat_grid_enabled" type="checkbox" ${transport.beat_grid ? 'checked' : ''}> Beat grid</label>
        <label>Clip BPM <input data-clip-transport="clip_bpm" type="number" min="1" max="999" step="0.01" value="${Number(grid.bpm ?? 120)}"></label>
        <label>Length beats <input data-clip-transport="length_beats" type="number" min="0.015625" max="65536" step="0.015625" value="${grid.length_beats ?? ''}" placeholder="derive"></label>
        <label><input data-clip-transport="sync_to_program" type="checkbox" ${grid.sync_to_program ? 'checked' : ''}> Sync BPM</label>
        <label>Beats/bar <input data-clip-transport="beats_per_bar" type="number" min="1" max="32" step="1" value="${Number(grid.beats_per_bar ?? 4)}"></label>
        <label><input data-clip-transport="beat_loop_enabled" type="checkbox" ${transport.beat_loop ? 'checked' : ''}> Beat loop</label>
        <label>Loop start <input data-clip-transport="beat_loop_start" type="number" min="0" max="65536" step="0.015625" value="${Number(beatLoop.start_beat ?? 0)}"></label>
        <label>Loop length <input data-clip-transport="beat_loop_length" type="number" min="0.015625" max="64" step="0.015625" value="${Number(beatLoop.length_beats ?? 1)}"></label>
      </div>
      <fieldset class="cue-controls" ${slot ? '' : 'disabled'}>
        <legend>Cues</legend>
        <select class="clip-cue-select" aria-label="Layer ${index + 1} cues">${cueOptions}</select>
        <button type="button" class="clip-cue-trigger">Trigger</button>
        <button type="button" class="clip-cue-remove">Remove</button>
        <label>ID <input class="clip-cue-id" type="number" min="0" max="4095" step="1" value="0"></label>
        <label>At <input class="clip-cue-at" type="number" min="0" max="1" step="0.001" value="0"></label>
        <button type="button" class="clip-cue-set">Add / update</button>
      </fieldset>
    </div>
    <div class="layer-matte-heading">
      <button class="layer-matte-toggle" type="button" aria-expanded="false" aria-controls="layer-matte-body-${index}">
        <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>Matte / Image Input</span>
      </button>
    </div>
    <div class="layer-matte-body" id="layer-matte-body-${index}" role="region" aria-label="Layer ${index + 1} matte and image input" hidden>
      <div class="matte-grid">
        <label><input data-layer-matte="enabled" type="checkbox" ${matte.enabled ? 'checked' : ''}> Enabled</label>
        <label>Input <select class="matte-input">${matteInputOptionsHtml(layer)}</select></label>
        <label>Channel <select data-layer-matte="channel"><option value="alpha">Alpha</option><option value="luma">Luma</option><option value="red">Red</option><option value="green">Green</option><option value="blue">Blue</option></select></label>
        <label><input data-layer-matte="invert" type="checkbox" ${matte.invert ? 'checked' : ''}> Invert</label>
        <label>Amount <input data-layer-matte="amount" type="range" min="0" max="1" step="0.01" value="${Number(matte.amount ?? 1)}"></label>
        <label>Threshold <input data-layer-matte="threshold" type="range" min="0" max="1" step="0.01" value="${Number(matte.threshold ?? 0.5)}"></label>
        <label>Softness <input data-layer-matte="softness" type="range" min="0" max="1" step="0.01" value="${Number(matte.softness ?? 0.1)}"></label>
      </div>
      <div class="performance-status matte-status" role="status" aria-live="polite">${escapeHtml(matte.diagnostic || (matte.enabled ? 'Ready' : 'Disabled'))}</div>
    </div>`;
}

const layerDisclosureState = new Map();

function layerDisclosureKey(layer, index) {
  return layer.layer_id || `${layer.source_kind || 'video'}:${layer.filename || 'untitled'}:${index}`;
}

function setLayerDisclosure(button, body, expanded) {
  button.setAttribute('aria-expanded', String(expanded));
  body.hidden = !expanded;
}

function wireLayerDisclosures(card, layer, index) {
  const key = layerDisclosureKey(layer, index);
  const remembered = layerDisclosureState.get(key) || { transform: false, motion: false, performance: false, matte: false, effects: false, cellular: false };
  const transformButton = card.querySelector('.layer-transform-toggle');
  const transformBody = card.querySelector('.layer-transform-body');
  const motionButton = card.querySelector('.layer-motion-toggle');
  const motionBody = card.querySelector('.layer-motion-body');
  const effectsButton = card.querySelector('.layer-fx-toggle');
  const effectsBody = card.querySelector('.layer-fx-body');
  const performanceButton = card.querySelector('.layer-performance-toggle');
  const performanceBody = card.querySelector('.layer-performance-body');
  const matteButton = card.querySelector('.layer-matte-toggle');
  const matteBody = card.querySelector('.layer-matte-body');
  setLayerDisclosure(transformButton, transformBody, !!remembered.transform);
  setLayerDisclosure(motionButton, motionBody, !!remembered.motion);
  setLayerDisclosure(effectsButton, effectsBody, remembered.effects);
  // Cellular and Small FX share one disclosure idiom; each toggle pairs with
  // the body its aria-controls names and remembers its own open state.
  card.querySelectorAll('.layer-cellular-toggle').forEach((button) => {
    const controls = button.getAttribute('aria-controls') || '';
    const body = card.querySelector(`#${controls}`);
    if (!body) return;
    const stateKey = controls.includes('small-fx') ? 'small_fx' : 'cellular';
    setLayerDisclosure(button, body, !!remembered[stateKey]);
    button.addEventListener('click', () => {
      remembered[stateKey] = button.getAttribute('aria-expanded') !== 'true';
      layerDisclosureState.set(key, remembered);
      setLayerDisclosure(button, body, remembered[stateKey]);
    });
  });
  setLayerDisclosure(performanceButton, performanceBody, remembered.performance);
  setLayerDisclosure(matteButton, matteBody, remembered.matte);
  transformButton.addEventListener('click', () => {
    remembered.transform = transformButton.getAttribute('aria-expanded') !== 'true';
    layerDisclosureState.set(key, remembered);
    setLayerDisclosure(transformButton, transformBody, remembered.transform);
  });
  motionButton.addEventListener('click', () => {
    remembered.motion = motionButton.getAttribute('aria-expanded') !== 'true';
    layerDisclosureState.set(key, remembered);
    setLayerDisclosure(motionButton, motionBody, remembered.motion);
  });
  effectsButton.addEventListener('click', () => {
    remembered.effects = effectsButton.getAttribute('aria-expanded') !== 'true';
    layerDisclosureState.set(key, remembered);
    setLayerDisclosure(effectsButton, effectsBody, remembered.effects);
  });
  performanceButton.addEventListener('click', () => {
    remembered.performance = performanceButton.getAttribute('aria-expanded') !== 'true';
    layerDisclosureState.set(key, remembered);
    setLayerDisclosure(performanceButton, performanceBody, remembered.performance);
  });
  matteButton.addEventListener('click', () => {
    remembered.matte = matteButton.getAttribute('aria-expanded') !== 'true';
    layerDisclosureState.set(key, remembered);
    setLayerDisclosure(matteButton, matteBody, remembered.matte);
  });
}

function motionRangeHtml(param, label, min, max, step, value) {
  return `<div class="param-row" data-motion-param="${param}"><label>${label}</label><input type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${label}"><span class="value">${value}</span></div>`;
}

function motionCheckboxHtml(param, label) {
  return `<div class="param-row select-row" data-motion-param="${param}"><label>${label}</label><input type="checkbox" aria-label="${label}"></div>`;
}

function motionSelectHtml(param, label, options) {
  return `<div class="param-row select-row" data-motion-param="${param}"><label>${label}</label><select aria-label="${label}">${options.map(([value, text]) => `<option value="${value}">${text}</option>`).join('')}</select></div>`;
}

function layerMotionControlsHtml(layer, index) {
  const motion = layer.motion || {};
  return `
    ${motionSelectHtml('field_source', 'Field source', [['auto', 'Auto'], ['codec_vectors', 'Codec vectors'], ['lattice', 'Motion lattice'], ['procedural_curl', 'Curl field'], ['procedural_radial', 'Radial field'], ['procedural_spiral', 'Spiral field'], ['procedural_contour', 'Contour field'], ['procedural_chroma', 'Chroma field'], ['procedural_weave', 'Weave field']])}
    ${motionSelectHtml('lattice_quality', 'Lattice', [['draft', 'Draft · 16px'], ['live', 'Live · 8px'], ['high', 'High · 4px']])}
    ${motionRangeHtml('field_scale', 'Field scale', 0, 1, 0.01, Number(motion.procedural?.scale ?? 0.5))}
    ${motionRangeHtml('field_rate', 'Field rate', -2, 2, 0.01, Number(motion.procedural?.rate ?? 0.25))}
    ${motionRangeHtml('stretch', 'Stretch', 0, 1, 0.01, Number(motion.shaping?.stretch ?? 0))}
    ${motionRangeHtml('edge_repel', 'Edge repel', 0, 1, 0.01, Number(motion.shaping?.edge_repel ?? 0))}
    ${motionRangeHtml('vector_trash', 'Vector trash', 0, 1, 0.01, Number(motion.shaping?.vector_trash ?? 0))}
    ${motionRangeHtml('trash_block_size', 'Trash block', 2, 256, 1, Number(motion.shaping?.trash_block_size ?? 16))}
    <div class="param-row select-row motion-donor-row"><label for="layer-motion-donor-${index}">Faraday donor</label><select id="layer-motion-donor-${index}" class="motion-donor-select" aria-label="Layer ${index + 1} Faraday motion donor"><option value="none">None</option></select></div>
    ${motionRangeHtml('transplant_amount', 'Transplant', 0, 1, 0.01, Number(motion.transplant?.amount ?? 0))}
    ${motionSelectHtml('carrier', 'Carrier', [['transparent', 'Transparent'], ['black', 'Black'], ['first_source_frame', 'First source frame']])}
    ${motionRangeHtml('confidence_threshold', 'Confidence', 0, 1, 0.01, Number(motion.transplant?.confidence_threshold ?? 0.1))}
    ${motionRangeHtml('confidence_softness', 'Conf. soft', 0, 0.5, 0.01, Number(motion.transplant?.confidence_softness ?? 0.05))}
    ${motionRangeHtml('refresh', 'Refresh', 0, 1, 0.01, Number(motion.transplant?.refresh ?? 1))}
    ${motionRangeHtml('decay', 'Decay', 0, 1, 0.01, Number(motion.transplant?.decay ?? 1))}
    ${motionRangeHtml('occlusion', 'Occlusion', 0, 1, 0.01, Number(motion.transplant?.occlusion ?? 0))}
    ${motionRangeHtml('shutter_angle', 'Shutter angle', 0, 360, 1, Number(motion.shutter?.angle_degrees ?? 0))}
    ${motionRangeHtml('shutter_phase', 'Phase', -1, 1, 0.01, Number(motion.shutter?.phase ?? 0))}
    ${motionRangeHtml('shutter_curvature', 'Curvature', -2, 2, 0.01, Number(motion.shutter?.curvature ?? 0))}
    ${motionRangeHtml('shutter_chromatic_lag', 'Chroma lag', 0, 1, 0.01, Number(motion.shutter?.chromatic_lag ?? 0))}
    ${motionSelectHtml('shutter_quality', 'Shutter quality', [['sharp', 'Sharp · 1'], ['draft', 'Draft · 4'], ['live', 'Live · 8'], ['high', 'High · 16']])}
    ${motionCheckboxHtml('collider_enabled', 'Field Collider')}
    ${motionSelectHtml('collider_mode', 'Collide mode', [['sum', 'Sum'], ['difference', 'Difference'], ['curl', 'Curl'], ['projection', 'Projection'], ['collision_boundary', 'Collision boundary']])}
    ${motionSelectHtml('collider_boundary', 'Collide edge', [['transparent', 'Transparent'], ['mirror', 'Mirror'], ['wrap', 'Wrap'], ['hold', 'Hold']])}
    <div class="param-row select-row motion-collider-row"><label for="layer-motion-collider-a-${index}">Collider A</label><select id="layer-motion-collider-a-${index}" class="motion-collider-select" data-collider-input="a" aria-label="Layer ${index + 1} Field Collider input A"><option value="none">None</option></select></div>
    <div class="param-row select-row motion-collider-row"><label for="layer-motion-collider-b-${index}">Collider B</label><select id="layer-motion-collider-b-${index}" class="motion-collider-select" data-collider-input="b" aria-label="Layer ${index + 1} Field Collider input B"><option value="none">None</option></select></div>
    <div class="audio-status motion-collider-status" role="status" aria-live="polite">${escapeHtml(motionColliderText(motion))}</div>
    <div class="audio-status motion-telemetry" role="status" aria-live="polite">${escapeHtml(motionTelemetryText(motion))}</div>
    <div class="audio-status">One active transplant is admitted composition-wide. A missing donor stays inert and never retargets. Enabling the collider parks the single-donor recipe; disabling resumes it.</div>`;
}

function syncLayerMotion(card, layer) {
  const panel = card.querySelector('.layer-motion-body');
  if (!panel) return;
  syncMotionPanel(panel, layer.motion || {});
  const donorSelect = panel.querySelector('.motion-donor-select');
  if (!donorSelect || !canSync(donorSelect)) return;
  const donor = layer.motion?.transplant?.donor || { kind: 'none' };
  const currentValue = donor.kind === 'selected'
    ? String(donor.layer_id || '')
    : donor.kind === 'missing' ? `missing:${Number(donor.saved_position || 0)}` : 'none';
  const optionKey = JSON.stringify([String(layer.layer_id || ''), currentValue, latestLayers.map((candidate) => [candidate.layer_id, candidate.filename])]);
  if (donorSelect.dataset.optionKey !== optionKey) {
    donorSelect.dataset.optionKey = optionKey;
    const options = [new Option('None', 'none')];
    latestLayers.forEach((candidate, candidateIndex) => {
      if (String(candidate.layer_id) === String(layer.layer_id)) return;
      options.push(new Option(`Layer ${candidateIndex + 1} · ${candidate.filename || 'Untitled'}`, String(candidate.layer_id)));
    });
    if (donor.kind === 'missing') {
      const missing = new Option(`Missing saved layer ${Number(donor.saved_position || 0) + 1}`, currentValue);
      missing.disabled = true;
      options.push(missing);
    }
    donorSelect.replaceChildren(...options);
  }
  donorSelect.value = currentValue;
  syncLayerMotionCollider(panel, layer);
}

// Both collider slots are populated independently and keyed by their own slot
// token, so clearing input A never slides input B's selection into A's place.
// A collider input may legally name its own recipient layer, unlike the Faraday
// donor above, so the recipient is NOT filtered out here. The partner slot's
// current layer is filtered out because A and B may never alias; the engine
// refuses that edit anyway, and offering it would invite a rejection.
function syncLayerMotionCollider(panel, layer) {
  const collider = layer.motion?.collider || {};
  const status = panel.querySelector('.motion-collider-status');
  if (status) {
    status.textContent = motionColliderText(layer.motion || {});
    status.classList.toggle('error', Boolean(collider.enabled && collider.diagnostic));
  }
  panel.querySelectorAll('.motion-collider-select').forEach((select) => {
    if (!canSync(select)) return;
    const slot = select.dataset.colliderInput === 'b' ? 'input_b' : 'input_a';
    const partner = slot === 'input_a' ? collider.input_b : collider.input_a;
    const donor = collider[slot] || { kind: 'none' };
    const currentValue = donor.kind === 'selected'
      ? String(donor.layer_id || '')
      : donor.kind === 'missing' ? `missing:${Number(donor.saved_position || 0)}` : 'none';
    const excluded = partner?.kind === 'selected' ? String(partner.layer_id || '') : '';
    const optionKey = JSON.stringify([slot, currentValue, excluded, latestLayers.map((candidate) => [candidate.layer_id, candidate.filename])]);
    if (select.dataset.optionKey !== optionKey) {
      select.dataset.optionKey = optionKey;
      const options = [new Option('None', 'none')];
      latestLayers.forEach((candidate, candidateIndex) => {
        if (excluded && String(candidate.layer_id) === excluded) return;
        options.push(new Option(`Layer ${candidateIndex + 1} · ${candidate.filename || 'Untitled'}`, String(candidate.layer_id)));
      });
      if (donor.kind === 'missing') {
        const missing = new Option(`Missing saved layer ${Number(donor.saved_position || 0) + 1}`, currentValue);
        missing.disabled = true;
        options.push(missing);
      }
      select.replaceChildren(...options);
    }
    select.value = currentValue;
  });
}

function wireLayerMotion(card, layer, index) {
  const panel = card.querySelector('.layer-motion-body');
  wireMotionPanel(panel, () => {
    const current = currentLayerContext(card, layer, index).layer;
    const layerId = String(current?.layer_id || '');
    return /^(?:[1-9][0-9]*)$/.test(layerId) ? { scope: 'layer', layer_id: layerId } : null;
  });
  panel.querySelector('.motion-donor-select')?.addEventListener('change', (event) => {
    const current = currentLayerContext(card, layer, index).layer;
    const layerId = String(current?.layer_id || '');
    const selected = event.currentTarget.value;
    if (!/^(?:[1-9][0-9]*)$/.test(layerId) || selected.startsWith('missing:')) return;
    const donorLayerId = selected === 'none' ? null : selected;
    if (donorLayerId !== null && !/^(?:[1-9][0-9]*)$/.test(donorLayerId)) return;
    sendAction({ action: 'set_motion_donor', layer_id: layerId, donor_layer_id: donorLayerId, layer_stack_revision: layerStackRevision });
  });
  panel.querySelectorAll('.motion-collider-select').forEach((select) => {
    select.addEventListener('change', (event) => {
      const current = currentLayerContext(card, layer, index).layer;
      const layerId = String(current?.layer_id || '');
      const selected = event.currentTarget.value;
      if (!/^(?:[1-9][0-9]*)$/.test(layerId) || selected.startsWith('missing:')) return;
      const donorLayerId = selected === 'none' ? null : selected;
      if (donorLayerId !== null && !/^(?:[1-9][0-9]*)$/.test(donorLayerId)) return;
      // Deliberately never wrapped as a Quantized inner action: rewiring an
      // input rewrites the motion-field request, so it is an ordered barrier.
      sendAction({
        action: 'set_motion_collider_input',
        layer_id: layerId,
        input: event.currentTarget.dataset.colliderInput === 'b' ? 'b' : 'a',
        donor_layer_id: donorLayerId,
        layer_stack_revision: layerStackRevision,
      });
    });
  });
}

function wireLayerEffects(card, layer, index) {
  card.querySelectorAll('[data-layer-effect]').forEach((row) => {
    const param = row.dataset.layerEffect;
    const spec = LAYER_EFFECT_CONTROLS.find(([candidate]) => candidate === param);
    if (!spec) return;
    const [, , kind, , , , fallback] = spec;
    const control = row.querySelector('input,select');
    if (kind === 'select') control.value = String(layer.effects?.[param] ?? fallback);
    const send = () => {
      const value = kind === 'checkbox' ? control.checked : kind === 'select' ? parseInt(control.value, 10) : parseFloat(control.value);
      row.querySelector('.value')?.replaceChildren(document.createTextNode(formatValue(value, Number(control.min), Number(control.max), Number(control.step))));
      sendAction({ action: 'set_layer_effect', ...currentLayerSelector(card, layer, index), param, value });
    };
    control.addEventListener(kind === 'range' ? 'input' : 'change', send);
    if (kind === 'range') resetRangeOnDoubleActivation(control, fallback);
  });
}

// --- B7 pattern synth layer card section ---

const LAYER_PATTERN_SELECT_OPTIONS = {
  shape: [
    ['scan', 'Scan'], ['radial', 'Radial'], ['spiral', 'Spiral'], ['plasma', 'Plasma'],
    ['lissajous', 'Lissajous'], ['rings', 'Rings'], ['starburst', 'Starburst'], ['grid', 'Grid'],
    ['tunnel', 'Tunnel'], ['cells', 'Cells'], ['interference', 'Interference'], ['polygon', 'Polygon'],
    ['mandelbrot', 'Mandelbrot (exact z² + c)'],
    ['memory_splats', 'Memory Splats (motion trails)'],
    ['gaussian_splats', 'Gaussian Splats (anisotropic)'],
  ],
  wave: [
    ['sine', 'Sine'], ['triangle', 'Triangle'], ['saw', 'Saw'],
    ['square', 'Square'], ['pulse', 'Pulse'], ['sample_hold', 'S&H'],
  ],
  color_mode: [
    ['mono', 'Mono'], ['rgb_phase', 'RGB Phase'], ['hsv_sweep', 'HSV Sweep'],
    ['duotone', 'Duotone'], ['bands', 'Bands'],
  ],
};

const LAYER_PATTERN_CONTROLS = [
  ['shape', 'Shape', 'select', 0, 0, 0, 'scan'],
  ['wave', 'Wave', 'select', 0, 0, 0, 'sine'],
  ['color_mode', 'Colour', 'select', 0, 0, 0, 'rgb_phase'],
  ['freq_x', 'Freq X', 'range', 0, 1, 0.001, 0.18],
  ['freq_y', 'Freq Y', 'range', 0, 1, 0.001, 0.12],
  ['phase', 'Phase', 'range', -1, 1, 0.001, 0],
  ['rate', 'Rate', 'range', -1, 1, 0.001, 0.08],
  ['cross_mod', 'Cross Mod', 'range', 0, 1, 0.001, 0],
  ['wavefold', 'Wavefold', 'range', 0, 1, 0.001, 0],
  ['pulse_width', 'Pulse Width', 'range', 0, 1, 0.001, 0.5],
  ['comparator', 'Comparator', 'range', 0, 1, 0.001, 0],
  ['comp_threshold', 'Comp Thresh', 'range', 0, 1, 0.001, 0.5],
  ['comp_soft', 'Comp Soft', 'range', 0, 1, 0.001, 0.12],
  ['symmetry', 'Symmetry', 'range', 1, 16, 1, 4],
  ['zoom', 'Scale', 'range', -1, 1, 0.001, 0],
  ['rotate', 'Rotate', 'range', -1, 1, 0.001, 0],
  ['skew', 'Skew', 'range', -1, 1, 0.001, 0],
  ['center_x', 'Centre X', 'range', -1, 1, 0.001, 0],
  ['center_y', 'Centre Y', 'range', -1, 1, 0.001, 0],
  ['warp', 'Domain Warp', 'range', 0, 1, 0.001, 0],
  ['hue', 'Hue', 'range', 0, 1, 0.001, 0.55],
  ['hue_spread', 'Hue Spread', 'range', 0, 2, 0.001, 1],
  ['saturation', 'Saturation', 'range', 0, 1, 0.001, 0.9],
  ['brightness', 'Brightness', 'range', 0, 1.5, 0.001, 1],
  ['color_bands', 'Colour Bands', 'range', 2, 16, 1, 6],
];

function layerPatternHtml(pattern, index) {
  const rows = LAYER_PATTERN_CONTROLS.map(([param, label, kind, min, max, step, fallback]) => {
    const value = pattern?.[param] ?? fallback;
    if (kind === 'select') {
      const options = (LAYER_PATTERN_SELECT_OPTIONS[param] || [])
        .map(([optionValue, optionLabel]) => `<option value="${optionValue}">${optionLabel}</option>`)
        .join('');
      return `<div class="param-row select-row layer-pattern-row" data-layer-pattern="${param}"><label>${label}</label><select aria-label="Layer ${index + 1} pattern ${label}">${options}</select></div>`;
    }
    return `<div class="param-row layer-pattern-row" data-layer-pattern="${param}" data-min="${min}" data-max="${max}" data-step="${step}"><label>${label}</label><input type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="Layer ${index + 1} pattern ${label}"><span class="value">${formatValue(Number(value), min, max, step)}</span></div>`;
  }).join('');
  return `
    <div class="layer-cellular-disclosure layer-pattern-section">
      <button class="layer-cellular-toggle" type="button" aria-expanded="false" aria-controls="layer-pattern-body-${index}">
        <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>PATTERN SYNTH</span>
      </button>
      <div class="layer-cellular-body" id="layer-pattern-body-${index}" role="region" aria-label="Layer ${index + 1} pattern synth" hidden>${rows}</div>
    </div>`;
}

function wireLayerPattern(card, layer, index) {
  card.querySelectorAll('[data-layer-pattern]').forEach((row) => {
    const param = row.dataset.layerPattern;
    const spec = LAYER_PATTERN_CONTROLS.find(([candidate]) => candidate === param);
    if (!spec) return;
    const [, , kind, , , , fallback] = spec;
    const control = row.querySelector('input,select');
    if (kind === 'select') control.value = String(layer.pattern?.[param] ?? fallback);
    const send = () => {
      const value = kind === 'select' ? control.value : parseFloat(control.value);
      if (kind === 'range') {
        row.querySelector('.value')?.replaceChildren(document.createTextNode(formatValue(Number(value), Number(control.min), Number(control.max), Number(control.step))));
      }
      sendAction({ action: 'set_layer_pattern', ...currentLayerSelector(card, layer, index), param, value });
    };
    control.addEventListener(kind === 'range' ? 'input' : 'change', send);
    if (kind === 'range') resetRangeOnDoubleActivation(control, fallback);
  });
}

function syncLayerPattern(card, layer) {
  for (const [param, , kind, min, max, step, fallback] of LAYER_PATTERN_CONTROLS) {
    const row = card.querySelector(`[data-layer-pattern="${param}"]`);
    const control = row?.querySelector('input,select');
    if (!control || !canSync(control)) continue;
    const value = layer.pattern?.[param] ?? fallback;
    control.value = String(value);
    if (kind === 'range') row.querySelector('.value').textContent = formatValue(Number(value), min, max, step);
  }
}

// --- B7 text page layer card section ---

const LAYER_TEXT_SELECT_OPTIONS = {
  font: [['mono', 'Mono'], ['sans', 'Sans']],
  shape: [
    ['none', 'None'], ['circle', 'Circle'], ['ring', 'Ring'], ['rect', 'Rect'], ['tri', 'Triangle'],
    ['cross', 'Cross'], ['bars', 'Bars'], ['grid', 'Grid'], ['rings', 'Rings'], ['starburst', 'Starburst'],
  ],
};

const LAYER_TEXT_CONTROLS = [
  ['font', 'Font', 'select', 0, 0, 0, 'mono'],
  ['size', 'Size', 'range', 0.03, 0.6, 0.005, 0.2],
  ['track', 'Track', 'range', -0.1, 0.5, 0.005, 0],
  ['x', 'X', 'range', 0, 1, 0.005, 0.5],
  ['y', 'Y', 'range', 0, 1, 0.005, 0.5],
  ['rot_degrees', 'Rotate', 'range', -180, 180, 1, 0],
  ['repeat', 'Repeat', 'range', 1, 8, 1, 1],
  ['outline', 'Outline', 'range', 0, 20, 0.5, 0],
  ['shape', 'Shape', 'select', 0, 0, 0, 'none'],
  ['shape_count', 'Shape Count', 'range', 1, 24, 1, 1],
  ['shape_size', 'Shape Size', 'range', 0.02, 1, 0.005, 0.3],
  ['shape_x', 'Shape X', 'range', 0, 1, 0.005, 0.5],
  ['shape_y', 'Shape Y', 'range', 0, 1, 0.005, 0.5],
  ['shape_stroke', 'Shape Stroke', 'range', 0, 40, 0.5, 0],
];

const LAYER_TEXT_COLORS = [
  ['ink', 'Ink', ['ink_r', 'ink_g', 'ink_b'], [1, 1, 1]],
  ['bg', 'Background', ['bg_r', 'bg_g', 'bg_b'], [0, 0, 0]],
  ['shape_fill', 'Shape Fill', ['shape_fill_r', 'shape_fill_g', 'shape_fill_b'], [1, 0.184, 0.627]],
];

function rgbToHex(rgb) {
  return `#${rgb.map((c) => Math.round(Math.min(1, Math.max(0, Number(c) || 0)) * 255).toString(16).padStart(2, '0')).join('')}`;
}

function layerTextHtml(text, index) {
  const body = String(text?.body ?? '');
  const rows = LAYER_TEXT_CONTROLS.map(([param, label, kind, min, max, step, fallback]) => {
    const value = text?.[param] ?? fallback;
    if (kind === 'select') {
      const options = (LAYER_TEXT_SELECT_OPTIONS[param] || [])
        .map(([optionValue, optionLabel]) => `<option value="${optionValue}">${optionLabel}</option>`)
        .join('');
      return `<div class="param-row select-row layer-text-row" data-layer-text="${param}"><label>${label}</label><select aria-label="Layer ${index + 1} text ${label}">${options}</select></div>`;
    }
    return `<div class="param-row layer-text-row" data-layer-text="${param}" data-min="${min}" data-max="${max}" data-step="${step}"><label>${label}</label><input type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="Layer ${index + 1} text ${label}"><span class="value">${formatValue(Number(value), min, max, step)}</span></div>`;
  }).join('');
  const colors = LAYER_TEXT_COLORS.map(([key, label, , fallback]) => {
    const value = rgbToHex(Array.isArray(text?.[key]) ? text[key] : fallback);
    return `<div class="param-row layer-text-color-row" data-layer-text-color="${key}"><label>${label}</label><input type="color" value="${value}" aria-label="Layer ${index + 1} text ${label} colour"></div>`;
  }).join('');
  return `
    <div class="layer-cellular-disclosure layer-text-section">
      <button class="layer-cellular-toggle" type="button" aria-expanded="false" aria-controls="layer-text-body-${index}">
        <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>TEXT PAGE</span>
      </button>
      <div class="layer-cellular-body" id="layer-text-body-${index}" role="region" aria-label="Layer ${index + 1} text page" hidden>
        <div class="param-row layer-text-body-row"><label for="layer-text-body-input-${index}">Body</label><textarea id="layer-text-body-input-${index}" class="layer-text-body-input" maxlength="4096" rows="3" aria-label="Layer ${index + 1} page text">${body.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</textarea></div>
        ${rows}
        ${colors}
      </div>
    </div>`;
}

function wireLayerText(card, layer, index) {
  const bodyInput = card.querySelector('.layer-text-body-input');
  bodyInput?.addEventListener('change', () => {
    sendAction({ action: 'set_layer_text', ...currentLayerSelector(card, layer, index), param: 'body', value: bodyInput.value.slice(0, 4096) });
  });
  card.querySelectorAll('[data-layer-text]').forEach((row) => {
    const param = row.dataset.layerText;
    const spec = LAYER_TEXT_CONTROLS.find(([candidate]) => candidate === param);
    if (!spec) return;
    const [, , kind, , , , fallback] = spec;
    const control = row.querySelector('input,select');
    if (kind === 'select') control.value = String(layer.text_page?.[param] ?? fallback);
    const send = () => {
      const value = kind === 'select'
        ? control.value
        : (param === 'repeat' || param === 'shape_count')
          ? parseInt(control.value, 10)
          : parseFloat(control.value);
      if (kind === 'range') {
        row.querySelector('.value')?.replaceChildren(document.createTextNode(formatValue(Number(value), Number(control.min), Number(control.max), Number(control.step))));
      }
      sendAction({ action: 'set_layer_text', ...currentLayerSelector(card, layer, index), param, value });
    };
    control.addEventListener(kind === 'range' ? 'input' : 'change', send);
    if (kind === 'range') resetRangeOnDoubleActivation(control, fallback);
  });
  card.querySelectorAll('[data-layer-text-color]').forEach((row) => {
    const key = row.dataset.layerTextColor;
    const spec = LAYER_TEXT_COLORS.find(([candidate]) => candidate === key);
    if (!spec) return;
    const [, , params] = spec;
    const control = row.querySelector('input');
    control.addEventListener('change', () => {
      const hex = control.value;
      const channels = [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16) / 255);
      channels.forEach((value, channel) => {
        sendAction({ action: 'set_layer_text', ...currentLayerSelector(card, layer, index), param: params[channel], value });
      });
    });
  });
}

function syncLayerText(card, layer) {
  const bodyInput = card.querySelector('.layer-text-body-input');
  if (bodyInput && canSync(bodyInput)) bodyInput.value = String(layer.text_page?.body ?? '');
  for (const [param, , kind, min, max, step, fallback] of LAYER_TEXT_CONTROLS) {
    const row = card.querySelector(`[data-layer-text="${param}"]`);
    const control = row?.querySelector('input,select');
    if (!control || !canSync(control)) continue;
    const value = layer.text_page?.[param] ?? fallback;
    control.value = String(value);
    if (kind === 'range') row.querySelector('.value').textContent = formatValue(Number(value), min, max, step);
  }
  for (const [key, , , fallback] of LAYER_TEXT_COLORS) {
    const row = card.querySelector(`[data-layer-text-color="${key}"]`);
    const control = row?.querySelector('input');
    if (!control || !canSync(control)) continue;
    control.value = rgbToHex(Array.isArray(layer.text_page?.[key]) ? layer.text_page[key] : fallback);
  }
}

function selectedSlotId(card, layer) {
  const selected = Number(card.querySelector('.clip-slot-select')?.value);
  if (Number.isInteger(selected) && selected > 0 && selected <= 65535) return selected;
  const active = Number(activeClipSlot(layer)?.id);
  return Number.isInteger(active) && active > 0 && active <= 65535 ? active : null;
}

function displayedClipSlot(card, layer) {
  const slots = layer?.performance?.slots || [];
  const active = activeClipSlot(layer);
  const requested = Number(card?.dataset.selectedSlotId);
  const selected = slots.find((slot) => Number(slot.id) === requested);
  if (card?._slotSelectionTouched && selected && Number(active?.id) !== requested) return selected;
  if (card && card._slotSelectionTouched && Number(active?.id) === requested) {
    card._slotSelectionTouched = false;
  }
  if (card && active) card.dataset.selectedSlotId = String(active.id);
  return active || selected || null;
}

function wireLayerPerformance(card, layer, index) {
  const target = () => {
    const current = currentLayerContext(card, layer, index).layer;
    const layerId = stableLayerId(current);
    const slotId = selectedSlotId(card, current);
    return layerId && slotId ? { layer: current, layerId, slotId } : null;
  };

  card.querySelector('.clip-slot-select').addEventListener('change', (event) => {
    const selected = Number(event.currentTarget.value);
    if (!Number.isInteger(selected) || selected <= 0 || selected > 65535) return;
    card.dataset.selectedSlotId = String(selected);
    card._slotSelectionTouched = true;
    syncLayerPerformance(card, currentLayerContext(card, layer, index).layer);
  });
  const status = (message, error = false) => {
    const element = card.querySelector('.clip-slot-status');
    if (!element) return;
    element.textContent = message;
    element.classList.toggle('error', error);
  };

  card.querySelector('.clip-activate').addEventListener('click', () => {
    const selected = target();
    if (!selected) return status('A prepared stable slot is required.', true);
    const triggerMode = card.querySelector('.clip-trigger-mode').value;
    if (sendAction({ action: 'activate_clip_slot', layer_id: selected.layerId, slot_id: selected.slotId, trigger_mode: triggerMode })) {
      status(`Activation staged (${triggerMode.replace('_', ' ')}).`);
    } else status('Control connection is offline.', true);
  });
  card.querySelector('.clip-remove').addEventListener('click', () => {
    const selected = target();
    if (!selected) return status('A prepared stable slot is required.', true);
    if (!window.confirm(`Remove prepared slot ${selected.slotId}?`)) return;
    sendAction({ action: 'remove_clip_slot', layer_id: selected.layerId, slot_id: selected.slotId });
  });
  const clipSeek = card.querySelector('.clip-seek');
  const sendSeek = (event) => {
    const selected = target();
    const position = Number(event.currentTarget.value);
    if (selected && Number.isFinite(position) && position >= 0 && position <= 1) {
      sendAction({ action: 'seek_clip_slot', layer_id: selected.layerId, slot_id: selected.slotId, position });
    }
  };
  // Dragging is source-time scratching, not merely a release-time seek. The
  // bounded server queue and decoder mailbox retain only the newest adjacent
  // request for this stable layer/slot; change supplies the exact final value.
  clipSeek.addEventListener('input', sendSeek);
  clipSeek.addEventListener('change', sendSeek);
  card.querySelector('.clip-timecode-seek').addEventListener('click', () => {
    const selected = target();
    const controls = {
      hours: card.querySelector('.clip-timecode-hours'),
      minutes: card.querySelector('.clip-timecode-minutes'),
      seconds: card.querySelector('.clip-timecode-seconds'),
      frames: card.querySelector('.clip-timecode-frames'),
    };
    if (!selected || Object.values(controls).some((control) => !control.checkValidity())) {
      return status('Choose an active prepared slot and a valid bounded timecode.', true);
    }
    const timecode = Object.fromEntries(Object.entries(controls).map(([key, control]) => [key, Number(control.value)]));
    timecode.rate = card.querySelector('.clip-timecode-rate').value;
    if (sendAction({ action: 'seek_clip_slot_timecode', layer_id: selected.layerId, slot_id: selected.slotId, timecode })) {
      status('Timecode seek queued for the active source.');
    } else status('Control connection is offline.', true);
  });
  card.querySelectorAll('[data-clip-transport]').forEach((control) => {
    const param = control.dataset.clipTransport;
    control.addEventListener(control.type === 'range' ? 'input' : 'change', () => {
      const selected = target();
      if (!selected || !control.checkValidity()) return;
      let value;
      if (control.type === 'checkbox') value = control.checked;
      else if (control.tagName === 'SELECT') value = control.value;
      else if (control.value === '' && ['sample_fps', 'length_beats'].includes(param)) value = null;
      else value = Number(control.value);
      if (value !== null && typeof value === 'number' && !Number.isFinite(value)) return;
      sendAction({ action: 'set_clip_transport', layer_id: selected.layerId, slot_id: selected.slotId, param, value });
    });
  });

  const cueSelection = () => {
    const cueId = Number(card.querySelector('.clip-cue-select').value);
    return Number.isInteger(cueId) && cueId >= 0 && cueId <= 4095 ? cueId : null;
  };
  card.querySelector('.clip-cue-select').addEventListener('change', (event) => {
    const selected = target();
    const cueId = cueSelection();
    const cue = selected?.layer?.performance?.slots
      ?.find((slot) => Number(slot.id) === selected.slotId)?.transport?.cues
      ?.find((candidate) => Number(candidate.id) === cueId);
    if (!cue) return;
    card.querySelector('.clip-cue-id').value = String(cue.id);
    card.querySelector('.clip-cue-at').value = String(cue.at);
    event.currentTarget.setCustomValidity('');
  });
  card.querySelector('.clip-cue-trigger').addEventListener('click', () => {
    const selected = target();
    const cueId = cueSelection();
    if (selected && cueId !== null) sendAction({ action: 'trigger_clip_cue', layer_id: selected.layerId, slot_id: selected.slotId, cue_id: cueId });
  });
  card.querySelector('.clip-cue-remove').addEventListener('click', () => {
    const selected = target();
    const cueId = cueSelection();
    if (selected && cueId !== null) sendAction({ action: 'remove_clip_cue', layer_id: selected.layerId, slot_id: selected.slotId, cue_id: cueId });
  });
  card.querySelector('.clip-cue-set').addEventListener('click', () => {
    const selected = target();
    const idControl = card.querySelector('.clip-cue-id');
    const atControl = card.querySelector('.clip-cue-at');
    const cueId = Number(idControl.value);
    const at = Number(atControl.value);
    if (!selected || !idControl.checkValidity() || !atControl.checkValidity()
        || !Number.isInteger(cueId) || cueId < 0 || cueId > 4095
        || !Number.isFinite(at) || at < 0 || at > 1) return;
    sendAction({ action: 'set_clip_cue', layer_id: selected.layerId, slot_id: selected.slotId, cue_id: cueId, at });
  });

  card.querySelectorAll('[data-layer-matte]').forEach((control) => {
    const param = control.dataset.layerMatte;
    const eventName = control.type === 'range' ? 'input' : 'change';
    control.addEventListener(eventName, () => {
      const layerId = currentStableLayerId(card, layer, index);
      if (!layerId || !control.checkValidity()) return;
      const value = control.type === 'checkbox'
        ? control.checked
        : control.tagName === 'SELECT'
          ? control.value
          : Number(control.value);
      if (typeof value === 'number' && !Number.isFinite(value)) return;
      sendAction({ action: 'set_layer_matte_param', layer_id: layerId, param, value, composition_revision: compositionRevision });
    });
  });
  card.querySelector('.matte-input').addEventListener('change', (event) => {
    const layerId = currentStableLayerId(card, layer, index);
    if (!layerId) return;
    const value = event.currentTarget.value;
    let input;
    if (value.startsWith('selected:')) {
      const [, donorId, stage] = value.split(':');
      if (!/^(?:[1-9][0-9]*)$/.test(donorId)
          || !['pre_local_effects', 'post_local_effects'].includes(stage)) return;
      input = { source: 'selected_layer', layer_id: donorId, stage };
    } else if (value.startsWith('group:')) {
      const groupId = value.slice(6);
      if (!/^(?:[1-9][0-9]*)$/.test(groupId)) return;
      input = { source: 'group_output', group_id: groupId };
    } else if (['one_below', 'all_below', 'clean_program', 'program_history'].includes(value)) {
      input = { source: value };
    } else return;
    sendAction({ action: 'set_layer_matte_input', layer_id: layerId, input, composition_revision: compositionRevision });
  });
}

function syncLayerPerformance(card, layer) {
  const slot = displayedClipSlot(card, layer);
  const transport = slot?.transport || {};
  const grid = transport.beat_grid || {};
  const beatLoop = transport.beat_loop || {};
  const slotSelect = card.querySelector('.clip-slot-select');
  if (slotSelect && canSync(slotSelect) && slot) slotSelect.value = String(slot.id);
  const seek = card.querySelector('.clip-seek');
  if (seek && canSync(seek)) seek.value = String(slot?.playhead ?? 0);
  const values = {
    direction: transport.direction ?? 'forward',
    end_behavior: transport.end_behavior ?? 'loop',
    in_point: transport.in_point ?? 0,
    out_point: transport.out_point ?? 1,
    rate: transport.rate ?? 1,
    sample_fps: transport.sample_fps ?? '',
    beat_grid_enabled: !!transport.beat_grid,
    clip_bpm: grid.bpm ?? 120,
    length_beats: grid.length_beats ?? '',
    sync_to_program: !!grid.sync_to_program,
    beats_per_bar: grid.beats_per_bar ?? 4,
    beat_loop_enabled: !!transport.beat_loop,
    beat_loop_start: beatLoop.start_beat ?? 0,
    beat_loop_length: beatLoop.length_beats ?? 1,
  };
  for (const [param, value] of Object.entries(values)) {
    const control = card.querySelector(`[data-clip-transport="${param}"]`);
    if (!control || !canSync(control)) continue;
    if (control.type === 'checkbox') control.checked = !!value;
    else control.value = String(value);
  }
  const status = card.querySelector('.clip-slot-status');
  if (status && !controlIsBusy(status)) {
    status.textContent = slot?.status || (slot?.prepared ? 'Prepared' : slot ? 'Staging…' : 'No prepared source');
    status.classList.toggle('error', !!slot?.status && !slot?.prepared);
  }
  const cues = Array.isArray(transport.cues) ? transport.cues : [];
  const cueSelect = card.querySelector('.clip-cue-select');
  const cueKey = JSON.stringify(cues);
  if (cueSelect && cueSelect.dataset.cueKey !== cueKey && !controlIsBusy(cueSelect)) {
    const previous = cueSelect.value;
    cueSelect.dataset.cueKey = cueKey;
    cueSelect.replaceChildren(...(cues.length
      ? cues.map((cue) => new Option(`Cue ${cue.id} · ${Number(cue.at).toFixed(3)}`, String(cue.id)))
      : [new Option('No cues', '')]));
    if (Array.from(cueSelect.options).some((option) => option.value === previous)) cueSelect.value = previous;
  }

  const matte = layer?.performance?.matte || {};
  for (const [param, fallback] of [['enabled', false], ['channel', 'alpha'], ['invert', false], ['amount', 1], ['threshold', 0.5], ['softness', 0.1]]) {
    const control = card.querySelector(`[data-layer-matte="${param}"]`);
    if (!control || !canSync(control)) continue;
    const value = matte[param] ?? fallback;
    if (control.type === 'checkbox') control.checked = !!value;
    else control.value = String(value);
  }
  const matteInput = card.querySelector('.matte-input');
  if (matteInput && canSync(matteInput)) {
    const optionKey = JSON.stringify([
      matte.input,
      latestLayers.map((candidate) => [candidate.layer_id, candidate.filename]),
      (latestCreative?.groups || []).map((group) => [group.group_id, group.name]),
    ]);
    if (matteInput.dataset.optionKey !== optionKey) {
      matteInput.dataset.optionKey = optionKey;
      matteInput.innerHTML = matteInputOptionsHtml(layer);
    }
    matteInput.value = matteInputValue(matte.input);
  }
  const matteStatus = card.querySelector('.matte-status');
  if (matteStatus) {
    matteStatus.textContent = matte.diagnostic || (matte.enabled ? 'Ready' : 'Disabled');
    matteStatus.classList.toggle('error', ['missing', 'cycle'].some((word) => String(matte.diagnostic).toLowerCase().includes(word)));
  }
}

function createLayerCard(layer, index) {
  const card = document.createElement('div');
  card.className = 'layer-card expanded';
  card.dataset.index = index;
  const blendMode = layerBlendModeInfo(layer.blend_mode);
  const blendSelectId = `layer-blend-${index}`;
  const blendDescriptionId = `layer-blend-description-${index}`;
  const stableLayerToken = String(layer.layer_id || `position-${index}`).replace(/[^A-Za-z0-9_-]/g, '-');
  const moshSendId = `layer-mosh-send-${stableLayerToken}`;

  card.innerHTML = `
    <div class="layer-header">
      <button class="layer-drag-btn" title="Drag or use arrow keys to reorder" aria-label="Move layer ${index + 1}; use arrow keys to reorder" aria-keyshortcuts="ArrowUp ArrowDown Home End">&#x2630;</button>
      ${layer.source_kind === 'spout'
        ? '<span class="layer-thumb lib-placeholder" aria-hidden="true">LIVE</span>'
        : layer.source_kind === 'pattern'
          ? '<span class="layer-thumb lib-placeholder" aria-hidden="true">SYN</span>'
          : layer.source_kind === 'text'
            ? '<span class="layer-thumb lib-placeholder" aria-hidden="true">TXT</span>'
            : `<img class="layer-thumb" src="/thumb/${encodeURIComponent(layer.filename)}" alt="">`}
      <span class="layer-num">${index + 1}</span>
      <button class="layer-play-btn" title="Play/Pause" aria-label="${layer.paused ? 'Play' : 'Pause'} layer ${index + 1}">${layer.paused ? '\u25B6' : '\u25A0'}</button>
      <span class="layer-title">${escapeHtml(layer.filename || 'Untitled')}</span>
      <button class="layer-rack-btn" type="button" title="Open this layer's Collision Rack" aria-label="Open layer ${index + 1} Collision Rack">RACK</button>
      <button class="layer-vis-btn ${layer.visible ? 'visible' : ''}" title="Visibility" aria-label="${layer.visible ? 'Hide' : 'Show'} layer ${index + 1}">${layer.visible ? '\u25C9' : '\u25CB'}</button>
      <button class="layer-remove-btn" title="Remove" aria-label="Remove layer ${index + 1}">\u00D7</button>
    </div>
    <div class="layer-progress"><div class="layer-progress-fill" style="width:${(layer.progress * 100).toFixed(1)}%"></div></div>
    <div class="layer-source-status" role="status" aria-live="polite"></div>
    <div class="layer-proxy-row">
      <button class="layer-proxy-btn" type="button" title="Encode a proxy for this clip's verified content identity; completion hot-adopts it live" aria-label="Encode proxy for layer ${index + 1}">Encode proxy</button>
      <span class="layer-proxy-status" role="status" aria-live="polite"></span>
    </div>
    <div class="layer-body">
      <div class="param-row" data-layer="${index}" data-param="opacity">
        <label>Opacity</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.opacity}">
        <span class="value">${layer.opacity.toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="mosh_send" title="Scales this layer's spatial contribution to the one shared Codec Mosh result; it does not create an independent codec history.">
        <label for="${moshSendId}">Mosh Send</label>
        <input id="${moshSendId}" type="range" min="0" max="1" step="0.01" value="${Number(layer.mosh_send ?? 1)}" aria-label="Layer ${index + 1} Codec Mosh send">
        <span class="value">${Number(layer.mosh_send ?? 1).toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="speed">
        <label>Speed</label>
        <input type="range" min="0.25" max="4" step="0.25" value="${layer.speed}">
        <span class="value">${layer.speed.toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="fps">
        <label>FPS</label>
        <input type="range" min="1" max="240" step="1" value="${layer.fps || 30}" aria-label="Layer ${index + 1} source FPS">
        <span class="value">${Number(layer.fps || 30).toFixed(0)}</span>
      </div>
      <div class="param-row select-row" data-layer="${index}" data-param="blend_mode">
        <label for="${blendSelectId}">Blend</label>
        <select id="${blendSelectId}" aria-describedby="layer-blend-policy ${blendDescriptionId}" title="${escapeHtml(layerBlendTitle(blendMode.key))}">
          ${layerBlendOptionsHtml(blendMode.key)}
        </select>
        <span id="${blendDescriptionId}" class="visually-hidden blend-mode-description">${escapeHtml(blendMode.description)}</span>
      </div>
      ${layer.source_kind === 'video' ? `
      <div class="param-row select-row" data-layer="${index}" data-param="delivery" title="Decode delivery: Legacy RGBA is the exact prior software path; Managed planar delivers admitted progressive 8-bit 4:2:0 frames as planes converted on the GPU under the source's declared color truth. Sources the law does not admit fall back per frame.">
        <label for="layer-delivery-${index}">Delivery</label>
        <select id="layer-delivery-${index}" aria-label="Layer ${index + 1} decode delivery policy">
          <option value="legacy_rgba" ${(layer.delivery || 'legacy_rgba') === 'legacy_rgba' ? 'selected' : ''}>Legacy RGBA</option>
          <option value="metadata_managed" ${layer.delivery === 'metadata_managed' ? 'selected' : ''}>Managed planar</option>
        </select>
        <span class="value layer-delivery-active" role="status" aria-label="Layer ${index + 1} active delivery">${layer.delivery_active_planar ? 'planar' : 'packed'}</span>
      </div>` : ''}
      ${layerPerformanceHtml(layer, index)}
      <div class="layer-transform-heading">
        <button class="layer-transform-toggle" type="button" aria-expanded="false" aria-controls="layer-transform-body-${index}">
          <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>Transform</span>
        </button>
      </div>
      <div class="layer-transform-body" id="layer-transform-body-${index}" role="region" aria-label="Layer ${index + 1} transform" hidden>${layerTransformControlsHtml(index)}</div>
      <div class="layer-motion-heading">
        <button class="layer-motion-toggle" type="button" aria-expanded="false" aria-controls="layer-motion-body-${index}">
          <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>Motion field</span>
        </button>
      </div>
      <div class="layer-motion-body motion-authoring" id="layer-motion-body-${index}" role="region" aria-label="Layer ${index + 1} motion field, Faraday transplant, and curved shutter" hidden>${layerMotionControlsHtml(layer, index)}</div>
      <div class="param-row select-row" data-layer="${index}" data-param="key_mode">
        <label>Key</label>
        <select>
          <option value="0" ${layer.key_mode === 0 ? 'selected' : ''}>Off</option>
          <option value="1" ${layer.key_mode === 1 ? 'selected' : ''}>Keep Bright</option>
          <option value="2" ${layer.key_mode === 2 ? 'selected' : ''}>Keep Dark</option>
          <option value="3" ${layer.key_mode === 3 ? 'selected' : ''}>Remove Chroma</option>
          <option value="4" ${layer.key_mode === 4 ? 'selected' : ''}>Keep Chroma</option>
        </select>
      </div>
      <div class="param-row" data-layer="${index}" data-param="key_threshold">
        <label>Key Thresh</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.key_threshold}">
        <span class="value">${layer.key_threshold.toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="key_softness">
        <label>Key Soft</label>
        <input type="range" min="0" max="0.5" step="0.01" value="${layer.key_softness}">
        <span class="value">${layer.key_softness.toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="key_color_r">
        <label>Key Red</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.key_color?.[0] ?? layer.effects?.key_color?.[0] ?? 0}">
        <span class="value">${Number(layer.key_color?.[0] ?? layer.effects?.key_color?.[0] ?? 0).toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="key_color_g">
        <label>Key Green</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.key_color?.[1] ?? layer.effects?.key_color?.[1] ?? 1}">
        <span class="value">${Number(layer.key_color?.[1] ?? layer.effects?.key_color?.[1] ?? 1).toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="key_color_b">
        <label>Key Blue</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.key_color?.[2] ?? layer.effects?.key_color?.[2] ?? 0}">
        <span class="value">${Number(layer.key_color?.[2] ?? layer.effects?.key_color?.[2] ?? 0).toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="key_tolerance">
        <label>Chroma Tol.</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.key_tolerance ?? layer.effects?.key_tolerance ?? 0.15}">
        <span class="value">${Number(layer.key_tolerance ?? layer.effects?.key_tolerance ?? 0.15).toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="key_border">
        <label>Border</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.effects?.key_border ?? 0}">
        <span class="value">${Number(layer.effects?.key_border ?? 0).toFixed(2)}</span>
      </div>
      <div class="param-row select-row" data-layer="${index}" data-param="key_border_color">
        <label>Border Col</label>
        <select aria-label="Layer ${index + 1} key border colour">
          ${[['0', 'White'], ['1', 'Yellow'], ['2', 'Cyan'], ['3', 'Green'], ['4', 'Magenta'], ['5', 'Red'], ['6', 'Blue'], ['7', 'Black']]
            .map(([code, label]) => `<option value="${code}" ${Number(layer.effects?.key_border_color ?? 0) === Number(code) ? 'selected' : ''}>${label}</option>`)
            .join('')}
        </select>
      </div>
      <div class="param-row" data-layer="${index}" data-param="key_shadow">
        <label>Shadow</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.effects?.key_shadow ?? 0}">
        <span class="value">${Number(layer.effects?.key_shadow ?? 0).toFixed(2)}</span>
      </div>
      <div class="audio-status">This layer's key reveals layers beneath it. Chroma modes use the RGB target and tolerance.</div>
      <div class="param-row toggle-row layer-master-bypass" title="Skips inherited Digital/Analog/Cellular/Motion processing; own Layer FX/opacity/key/blend remain. VHS still finishes the complete program once; any contributing bypass links the shared Temporal family dry while history stays warm.">
        <label>Bypass Master FX</label>
        <label class="toggle">
          <input type="checkbox" ${layer.bypass_master_fx ? 'checked' : ''} aria-label="Bypass Master FX for layer ${index + 1}" aria-describedby="layer-master-bypass-help-${index}">
          <span class="toggle-slider"></span>
        </label>
        <span id="layer-master-bypass-help-${index}" class="visually-hidden">Skips inherited Digital/Analog/Cellular/Motion processing; own Layer FX/opacity/key/blend remain. VHS still finishes the complete program once. By itself, any contributing Master bypass links the shared Temporal family dry for the whole program while history stays warm; an admitted explicit Temporal bypass uses its separate isolated route.</span>
      </div>
      <div class="param-row toggle-row layer-temporal-bypass" title="Keeps this layer dry above the shared Temporal result. Available for Layer 1 or a contiguous top prefix; move it above every wet layer first. Authored independently from Bypass Master FX.">
        <label>Bypass Temporal FX</label>
        <label class="toggle">
          <input type="checkbox" ${layer.bypass_temporal_fx ? 'checked' : ''} aria-label="Bypass Temporal FX for layer ${index + 1}" aria-describedby="layer-temporal-bypass-help-${index}">
          <span class="toggle-slider"></span>
        </label>
        <span id="layer-temporal-bypass-help-${index}" class="visually-hidden">Keeps this layer out of Feedback, Slit-Scan, Long Exposure, Melt, Display Physics, and Codec Mosh by restoring it above the wet result. Available for Layer 1 or a contiguous top prefix; move it above every wet layer first. Authored independently from Bypass Master FX. Unsupported VHS, Advanced arrangements, or authored Motion modulation routes are refused without changing the picture.</span>
      </div>
      <div class="layer-random-controls" role="group" aria-label="Layer ${index + 1} deterministic random pattern">
        <label>Seed <input class="layer-random-seed seed-input" type="number" min="0" max="4294967295" step="1" inputmode="numeric" value="${Number(layer.effects?.random_seed || 0) >>> 0}" aria-label="Layer ${index + 1} pattern seed" title="Zero restores the legacy pattern"></label>
        <button class="layer-reroll" type="button" title="Advance this layer's deterministic pattern seed" aria-label="Reroll layer ${index + 1} pattern">&#x2684; Reroll</button>
        ${layer.source_kind === 'video' ? `<label class="layer-loop-reroll"><input type="checkbox" ${layer.reroll_on_loop ? 'checked' : ''}> each loop</label>` : ''}
      </div>
      ${layer.source_kind === 'pattern' ? layerPatternHtml(layer.pattern || {}, index) : ''}
      ${layer.source_kind === 'text' ? layerTextHtml(layer.text_page || {}, index) : ''}
      <div class="layer-fx-heading">
        <button class="layer-fx-toggle" type="button" aria-expanded="false" aria-controls="layer-fx-body-${index}">
          <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>Layer effects</span>
        </button>
        <button class="layer-reset-fx" type="button" title="Reset direct effects, Motion, and Mosh Send (rack, transform, opacity, and transport unchanged)" aria-label="Reset direct effects, Motion, and Mosh Send; rack, transform, opacity, and transport stay unchanged">Reset FX</button>
      </div>
      <div class="layer-fx-body" id="layer-fx-body-${index}" role="region" aria-label="Layer ${index + 1} effects" hidden>${layerEffectsHtml(layer.effects || {}, index)}</div>
    </div>
  `;

  // Toggle expand
  card.querySelector('.layer-header').addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') return;
    card.classList.toggle('expanded');
  });

  // Play/pause
  card.querySelector('.layer-play-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const current = currentLayerContext(card, layer, index);
    sendAction({ action: 'set_layer_paused', ...layerSelector(current.layer, current.index), paused: !current.layer.paused });
  });

  // Proxy request — the browser twin of the native Y key. The stable ID is
  // mandatory and authoritative: the engine owns every refusal, and this
  // action deliberately has no positional fallback.
  card.querySelector('.layer-proxy-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const id = currentStableLayerId(card, layer, index);
    if (id) sendAction({ action: 'request_layer_proxy', layer_id: id });
  });

  // Pointer-owned reorder works for mouse, pen, and touch. The list stays
  // stable throughout the gesture and commits one move on release.
  const dragBtn = card.querySelector('.layer-drag-btn');
  dragBtn.addEventListener('pointerdown', (e) => {
    if (layerDrag) return;
    const current = currentLayerContext(card, layer, index);
    layerDrag = { pointerId: e.pointerId, from: current.index, to: current.index, layerId: current.layer?.layer_id || null };
    dragBtn.setPointerCapture(e.pointerId);
    card.classList.add('reorder-target');
    e.stopPropagation();
    e.preventDefault();
  });
  dragBtn.addEventListener('pointermove', (e) => {
    if (!layerDrag || layerDrag.pointerId !== e.pointerId) return;
    updateLayerDragTarget(e.clientX, e.clientY);
    e.preventDefault();
  });
  dragBtn.addEventListener('pointerup', (e) => {
    if (!layerDrag || layerDrag.pointerId !== e.pointerId) return;
    updateLayerDragTarget(e.clientX, e.clientY);
    const { from, to, layerId } = layerDrag;
    clearLayerDrag();
    if (from !== to) sendAction({ action: 'move_layer', from, to, layer_id: layerId, stack_revision: layerStackRevision });
    e.stopPropagation();
    e.preventDefault();
  });
  dragBtn.addEventListener('pointercancel', clearLayerDrag);
  dragBtn.addEventListener('lostpointercapture', (e) => {
    if (layerDrag?.pointerId === e.pointerId) clearLayerDrag();
  });
  dragBtn.addEventListener('keydown', (e) => {
    const current = currentLayerContext(card, layer, index);
    const targets = {
      ArrowUp: current.index - 1,
      ArrowLeft: current.index - 1,
      ArrowDown: current.index + 1,
      ArrowRight: current.index + 1,
      Home: 0,
      End: layersList.children.length - 1,
    };
    if (!(e.key in targets)) return;
    e.preventDefault();
    const to = Math.max(0, Math.min(layersList.children.length - 1, targets[e.key]));
    if (to !== current.index) sendAction({ action: 'move_layer', from: current.index, to, layer_id: current.layer?.layer_id || null, stack_revision: layerStackRevision });
  });

  // Visibility
  card.querySelector('.layer-vis-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const current = currentLayerContext(card, layer, index);
    sendAction({ action: 'set_layer_visibility', ...layerSelector(current.layer, current.index), visible: !current.layer.visible });
  });

  card.querySelector('.layer-rack-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const current = currentLayerContext(card, layer, index);
    const layerId = String(current.layer?.layer_id || '');
    if (!layerId || !creativeRackScope) return;
    const panel = document.getElementById('creative-panel');
    if (panel) panel.open = true;
    creativeRackScope.value = `layer:${layerId}`;
    creativeStructureKey = '';
    syncCreative(latestCreative);
    panel?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  // Remove
  card.querySelector('.layer-remove-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    sendAction({ action: 'remove_layer', ...currentLayerSelector(card, layer, index) });
  });

  // Layer param sliders
  card.querySelectorAll('.layer-body .param-row[data-param]').forEach((row) => {
    const param = row.dataset.param;
    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const select = row.querySelector('select');

    if (slider) {
      slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        if (valueEl) valueEl.textContent = v.toFixed(2);
        sendAction({ action: 'set_layer_param', ...currentLayerSelector(card, layer, index), param, value: v });
      });
      const defaults = {
        opacity: 1, mosh_send: 1, speed: 1, fps: 30, key_threshold: 0.5, key_softness: 0.1,
        key_color_r: 0, key_color_g: 1, key_color_b: 0, key_tolerance: 0.15,
      };
      resetRangeOnDoubleActivation(slider, defaults[param] ?? parseFloat(slider.min));
    }

    if (select) {
      select.addEventListener('change', () => {
        const v = param === 'key_mode' || param === 'key_border_color' ? parseInt(select.value) : select.value;
        if (param === 'blend_mode') syncLayerBlendDescription(row, v);
        sendAction({ action: 'set_layer_param', ...currentLayerSelector(card, layer, index), param, value: v });
      });
    }
  });
  card.querySelector('.layer-reset-fx').addEventListener('click', () => {
    sendAction({ action: 'reset_layer_fx', ...currentLayerSelector(card, layer, index) });
  });
  card.querySelector('.layer-master-bypass input').addEventListener('change', (event) => {
    sendAction({ action: 'set_layer_param', ...currentLayerSelector(card, layer, index), param: 'bypass_master_fx', value: event.currentTarget.checked });
  });
  card.querySelector('.layer-temporal-bypass input').addEventListener('change', (event) => {
    sendAction({ action: 'set_layer_param', ...currentLayerSelector(card, layer, index), param: 'bypass_temporal_fx', value: event.currentTarget.checked });
  });
  card.querySelector('.layer-reroll').addEventListener('click', () => {
    sendAction({
      action: 'reroll',
      scope: 'layer',
      ...currentLayerSelector(card, layer, index),
      mode: 'pattern',
      amount: 0.7,
      include_grain_controls: false,
      include_transform: !!rerollTransformControls?.checked,
    });
  });
  card.querySelector('.layer-random-seed').addEventListener('change', (event) => {
    const seed = Number(event.currentTarget.value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      event.currentTarget.setCustomValidity('Seed must be a whole number from 0 to 4294967295');
      event.currentTarget.reportValidity();
      return;
    }
    event.currentTarget.setCustomValidity('');
    sendAction({ action: 'set_layer_effect', ...currentLayerSelector(card, layer, index), param: 'random_seed', value: seed });
  });
  card.querySelector('.layer-loop-reroll input')?.addEventListener('change', (event) => {
    sendAction({ action: 'set_layer_reroll_on_loop', ...currentLayerSelector(card, layer, index), enabled: event.currentTarget.checked });
  });
  wireLayerDisclosures(card, layer, index);
  const transformStateKey = layerDisclosureKey(layer, index);
  wireTransformPanel(card.querySelector('.layer-transform-body'), {
    stateKey: transformStateKey,
    getTransform: () => currentLayerContext(card, layer, index).layer?.transform,
    set: (param, value) => sendAction({ action: 'set_layer_transform', ...currentLayerSelector(card, layer, index), param, value }),
    reset: () => sendAction({ action: 'reset_layer_transform', ...currentLayerSelector(card, layer, index) }),
    apply: (transform) => sendAction({ action: 'apply_layer_transform', ...currentLayerSelector(card, layer, index), transform }),
  });
  wireLayerEffects(card, layer, index);
  if (layer.source_kind === 'pattern') wireLayerPattern(card, layer, index);
  if (layer.source_kind === 'text') wireLayerText(card, layer, index);
  wireLayerMotion(card, layer, index);
  wireLayerPerformance(card, layer, index);
  bindRangeEditors(card);

  updateLayerCard(card, layer, index);
  return card;
}

function updateLayerCard(card, layer, index) {
  if (!card) return;
  // Cards survive ordinary state snapshots, so callbacks must read the latest
  // immutable layer DTO instead of the one captured at card construction.
  card._layerState = layer;
  card.dataset.index = String(index);
  const playBtn = card.querySelector('.layer-play-btn');
  const title = card.querySelector('.layer-title');
  const visBtn = card.querySelector('.layer-vis-btn');
  const progressFill = card.querySelector('.layer-progress-fill');
  const sourceStatus = card.querySelector('.layer-source-status');

  if (playBtn) playBtn.textContent = layer.paused ? '\u25B6' : '\u25A0';
  if (playBtn) playBtn.setAttribute('aria-label', `${layer.paused ? 'Play' : 'Pause'} layer ${index + 1}`);
  if (title) title.textContent = layer.filename || 'Untitled';
  if (visBtn) {
    visBtn.textContent = layer.visible ? '\u25C9' : '\u25CB';
    visBtn.className = `layer-vis-btn ${layer.visible ? 'visible' : ''}`;
    visBtn.setAttribute('aria-label', `${layer.visible ? 'Hide' : 'Show'} layer ${index + 1}`);
  }
  if (progressFill) {
    progressFill.style.width = `${(layer.progress * 100).toFixed(1)}%`;
  }
  if (sourceStatus) {
    const dimensions = layer.source_width && layer.source_height
      ? ` \u00b7 ${layer.source_width}\u00d7${layer.source_height}`
      : '';
    const frames = layer.source_sequence ? ` \u00b7 frame ${layer.source_sequence}` : '';
    const isSpout = layer.source_kind === 'spout';
    sourceStatus.hidden = !isSpout && !layer.source_error;
    if (layer.source_error) {
      sourceStatus.textContent = isSpout
        ? `${layer.source_error} \u00b7 ${layer.offline_export_policy || 'offline export: black'}`
        : `video decoder: ${layer.source_error}`;
      sourceStatus.className = 'layer-source-status error';
    } else if (!isSpout) {
      // A healthy video decoder is the common case: clear any stale error
      // without adding a status line to every ordinary layer card.
      sourceStatus.textContent = '';
      sourceStatus.className = 'layer-source-status active';
    } else if (layer.source_active) {
      sourceStatus.textContent = `receiving ${layer.source_name || 'Spout'}${dimensions}${frames} \u00b7 offline export: black`;
      sourceStatus.className = 'layer-source-status active';
      if (spoutInStatus?.textContent.startsWith('Opening ')) spoutInStatus.textContent = '';
    } else {
      sourceStatus.textContent = `waiting for ${layer.source_name || 'sender'} \u00b7 offline export: black`;
      sourceStatus.className = 'layer-source-status';
    }
  }

  const proxyRow = card.querySelector('.layer-proxy-row');
  if (proxyRow) {
    const proxyBtn = proxyRow.querySelector('.layer-proxy-btn');
    const proxyStatus = proxyRow.querySelector('.layer-proxy-status');
    const isVideo = layer.source_kind === 'video';
    const backed = String(layer.proxy_backing_prefix || '');
    // Only decoded video can be proxied; the engine's ladder still owns the
    // finer refusals (verified identity, busy worker, unavailable cache).
    proxyRow.hidden = !isVideo;
    if (proxyBtn) {
      proxyBtn.disabled = !isVideo || !!backed;
      proxyBtn.setAttribute('aria-label', `Encode proxy for layer ${index + 1}`);
    }
    if (proxyStatus) {
      proxyStatus.textContent = backed
        ? `proxy active (${backed}…)`
        : String(layer.proxy_note || '');
      proxyStatus.className = `layer-proxy-status${backed ? ' active' : ''}`;
    }
  }

  // Sync layer param sliders (skip if user is actively dragging)
  const opacityRow = card.querySelector('.param-row[data-param="opacity"]');
  if (opacityRow) {
    const slider = opacityRow.querySelector('input[type="range"]');
    const valEl = opacityRow.querySelector('.value');
    if (slider && canSync(slider)) {
      slider.value = layer.opacity;
      if (valEl) valEl.textContent = layer.opacity.toFixed(2);
    }
  }

  for (const [param, digits, fallback] of [['mosh_send', 2, 1], ['speed', 2, 1], ['fps', 0, 30]]) {
    const row = card.querySelector(`.param-row[data-param="${param}"]`);
    const slider = row?.querySelector('input[type="range"]');
    const valEl = row?.querySelector('.value');
    if (slider && canSync(slider)) {
      const value = Number(layer[param] ?? fallback);
      slider.value = value;
      if (valEl) valEl.textContent = value.toFixed(digits);
    }
  }

  const blendRow = card.querySelector('.param-row[data-param="blend_mode"]');
  if (blendRow) {
    const select = blendRow.querySelector('select');
    if (select && canSync(select)) {
      select.value = layerBlendModeInfo(layer.blend_mode).key;
      syncLayerBlendDescription(blendRow, select.value);
    }
  }

  const deliveryRow = card.querySelector('.param-row[data-param="delivery"]');
  if (deliveryRow) {
    const select = deliveryRow.querySelector('select');
    if (select && canSync(select)) select.value = layer.delivery || 'legacy_rgba';
    const active = deliveryRow.querySelector('.layer-delivery-active');
    if (active) active.textContent = layer.delivery_active_planar ? 'planar' : 'packed';
  }

  const keyModeRow = card.querySelector('.param-row[data-param="key_mode"]');
  if (keyModeRow) {
    const select = keyModeRow.querySelector('select');
    if (select && canSync(select)) {
      select.value = layer.key_mode;
    }
  }

  const directKeyValues = {
    key_threshold: layer.key_threshold,
    key_softness: layer.key_softness,
    key_color_r: layer.key_color?.[0] ?? layer.effects?.key_color?.[0] ?? 0,
    key_color_g: layer.key_color?.[1] ?? layer.effects?.key_color?.[1] ?? 1,
    key_color_b: layer.key_color?.[2] ?? layer.effects?.key_color?.[2] ?? 0,
    key_tolerance: layer.key_tolerance ?? layer.effects?.key_tolerance ?? 0.15,
  };
  for (const [param, value] of Object.entries(directKeyValues)) {
    const row = card.querySelector(`.param-row[data-param="${param}"]`);
    if (row) {
      const slider = row.querySelector('input[type="range"]');
      const valEl = row.querySelector('.value');
      if (slider && canSync(slider)) {
        slider.value = value;
        if (valEl) valEl.textContent = Number(value).toFixed(2);
      }
    }
  }
  const bypassMasterFx = card.querySelector('.layer-master-bypass input');
  if (bypassMasterFx && canSync(bypassMasterFx)) {
    bypassMasterFx.checked = !!layer.bypass_master_fx;
  }
  const bypassTemporalFx = card.querySelector('.layer-temporal-bypass input');
  if (bypassTemporalFx && canSync(bypassTemporalFx)) {
    bypassTemporalFx.checked = !!layer.bypass_temporal_fx;
  }
  const layerSeed = card.querySelector('.layer-random-seed');
  if (layerSeed && canSync(layerSeed)) layerSeed.value = String(Number(layer.effects?.random_seed || 0) >>> 0);
  const loopReroll = card.querySelector('.layer-loop-reroll input');
  if (loopReroll && canSync(loopReroll)) loopReroll.checked = !!layer.reroll_on_loop;
  for (const [param, , kind, min, max, step, fallback] of LAYER_EFFECT_CONTROLS) {
    const row = card.querySelector(`[data-layer-effect="${param}"]`);
    const control = row?.querySelector('input,select');
    if (!control || !canSync(control)) continue;
    const value = layer.effects?.[param] ?? fallback;
    if (kind === 'checkbox') control.checked = !!value;
    else control.value = String(value);
    if (kind === 'range') row.querySelector('.value').textContent = formatValue(Number(value), min, max, step);
  }
  syncTransformPanel(card.querySelector('.layer-transform-body'), layer.transform);
  syncLayerMotion(card, layer);
  syncLayerPerformance(card, layer);
  if (layer.pattern) syncLayerPattern(card, layer);
  if (layer.text_page) syncLayerText(card, layer);
}

// --- Sync library ---

const activePreviewIntervals = new Set();

function loadLibraryFileIntoSelectedSlot(filename) {
  const match = /^([1-9][0-9]*):(new|[1-9][0-9]*)$/.exec(librarySlotTarget?.value || '');
  if (!match) {
    if (slotLoadStatus) slotLoadStatus.textContent = 'Choose a stable layer and destination slot first.';
    librarySlotTarget?.focus();
    return;
  }
  const [, layerId, slotText] = match;
  const triggerMode = ['immediate', 'next_beat', 'next_bar'].includes(librarySlotTrigger?.value)
    ? librarySlotTrigger.value
    : 'immediate';
  const action = {
    action: 'load_clip_into_slot',
    layer_id: layerId,
    filename,
    activate: true,
    trigger_mode: triggerMode,
    ...(slotText === 'new' ? {} : { slot_id: Number(slotText) }),
  };
  if (sendAction(action)) {
    if (slotLoadStatus) slotLoadStatus.textContent = `Staging ${filename}; the current source stays live until ready.`;
  } else if (slotLoadStatus) slotLoadStatus.textContent = 'Control connection is offline.';
}

function syncLibrary(files) {
  if (!files) return;

  // Count alone misses same-length renames/deletions. Compare a stable filename key.
  const filesKey = JSON.stringify(files);
  if (libraryGrid.dataset.filesKey === filesKey) return;
  libraryGrid.dataset.filesKey = filesKey;

  for (const interval of activePreviewIntervals) clearInterval(interval);
  activePreviewIntervals.clear();
  libraryGrid.innerHTML = '';

  if (files.length === 0) {
    libraryGrid.innerHTML = '<p class="dim" style="grid-column:1/-1;text-align:center;padding:12px;">No visual files</p>';
    return;
  }

  files.forEach((filename) => {
    const item = document.createElement('div');
    item.className = 'library-item';
    item.title = filename;
    item.setAttribute('role', 'group');
    item.setAttribute('aria-label', `${filename} library actions`);

    // Thumbnail image from server (retries if not yet generated)
    const img = document.createElement('img');
    img.dataset.retries = '0';
    const thumbUrl = `/thumb/${encodeURIComponent(filename)}`;
    img.src = thumbUrl;
    img.onload = () => {
      // A thumbnail may become available after the permanent-fallback UI was
      // installed. Always let a later successful load reclaim the image slot.
      img.style.display = '';
      item.querySelector('.lib-placeholder')?.remove();
    };
    img.onerror = () => {
      // Preview frames are opportunistic: a missing preview must not consume
      // thumbnail retries or replace a perfectly good static thumbnail.
      if (img.dataset.previewing === 'true') {
        img.dataset.previewing = 'false';
        img.src = thumbUrl;
        return;
      }
      const retries = parseInt(img.dataset.retries);
      if (retries < 5) {
        img.dataset.retries = String(retries + 1);
        setTimeout(() => { img.src = `/thumb/${encodeURIComponent(filename)}?r=${retries + 1}`; }, 1500 * (retries + 1));
      } else {
        img.style.display = 'none';
        const placeholder = document.createElement('span');
        placeholder.className = 'lib-placeholder';
        placeholder.textContent = filename.replace(/\.[^.]+$/, '');
        item.appendChild(placeholder);
      }
    };
    item.appendChild(img);

    // Hover preview animation is video-only. Stills already have their exact
    // thumbnail; probing nonexistent preview strips would otherwise issue a
    // perpetual stream of 404s and flash the image while hovered.
    let hoverInterval = null;
    let hoverFrame = 0;
    const hasAnimatedPreview = /\.(mp4|webm|mov|avi|mkv)$/i.test(filename);
    if (hasAnimatedPreview) {
      item.addEventListener('mouseenter', () => {
        const enc = encodeURIComponent(filename);
        // Start cycling through preview frames
        hoverFrame = 0;
        hoverInterval = setInterval(() => {
          hoverFrame = (hoverFrame + 1) % 8;
          img.dataset.previewing = 'true';
          img.src = `/preview/${enc}/${hoverFrame}`;
        }, 250);
        activePreviewIntervals.add(hoverInterval);
      });

      item.addEventListener('mouseleave', () => {
        if (hoverInterval) {
          clearInterval(hoverInterval);
          activePreviewIntervals.delete(hoverInterval);
          hoverInterval = null;
        }
        // Restore static thumbnail
        img.dataset.previewing = 'false';
        img.dataset.retries = '0';
        img.src = thumbUrl;
      });
    }

    // Filename label on hover
    const label = document.createElement('span');
    label.className = 'lib-label';
    label.textContent = filename.replace(/\.[^.]+$/, '');
    item.appendChild(label);

    const actions = document.createElement('div');
    actions.className = 'library-actions';
    const addLayer = document.createElement('button');
    addLayer.type = 'button';
    addLayer.className = 'lib-action lib-add-layer';
    addLayer.textContent = 'Layer';
    addLayer.title = `Add ${filename} as a new layer`;
    addLayer.setAttribute('aria-label', `Add ${filename} as a new layer`);
    addLayer.addEventListener('click', (event) => {
      event.stopPropagation();
      sendAction({ action: 'add_layer', filename });
    });
    const loadSlot = document.createElement('button');
    loadSlot.type = 'button';
    loadSlot.className = 'lib-action lib-load-slot';
    loadSlot.textContent = 'Slot';
    loadSlot.title = `Load ${filename} into the selected prepared slot`;
    loadSlot.setAttribute('aria-label', `Load ${filename} into the selected prepared slot`);
    loadSlot.addEventListener('click', (event) => {
      event.stopPropagation();
      loadLibraryFileIntoSelectedSlot(filename);
    });
    actions.append(addLayer, loadSlot);
    item.appendChild(actions);

    // Remove from library (to the Recycle Bin — recoverable)
    const del = document.createElement('button');
    del.className = 'lib-delete-btn';
    del.textContent = '×';
    del.title = 'Remove from library (moves to Recycle Bin)';
    del.setAttribute('aria-label', `Move ${filename} to the Recycle Bin`);
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!window.confirm(`Move “${filename}” to the Recycle Bin?`)) return;
      try {
        const r = await fetch(`/delete?name=${encodeURIComponent(filename)}`, { method: 'POST' });
        const text = await r.text();
        let result = null;
        try { result = JSON.parse(text); } catch { /* legacy/plain transport refusal */ }
        const detail = result?.name || result?.error || text;
        libUploadStatus.textContent = r.ok ? `${detail} → Recycle Bin` : `${filename}: ${detail}`;
      } catch {
        libUploadStatus.textContent = `${filename}: remove failed`;
      }
      setTimeout(() => { libUploadStatus.textContent = ''; }, 4000);
    });
    item.appendChild(del);

    let lastTouchAdd = 0;
    item.addEventListener('dblclick', (event) => {
      if (event.target.closest('button')) return;
      if (performance.now() - lastTouchAdd < 500) return;
      sendAction({ action: 'add_layer', filename });
    });
    let libraryTap = 0;
    let libraryTapStart = null;
    item.addEventListener('pointerdown', (e) => {
      if (e.pointerType !== 'mouse' && !e.target.closest('button')) {
        libraryTapStart = { id: e.pointerId, x: e.clientX, y: e.clientY, at: performance.now() };
      }
    });
    item.addEventListener('pointerup', (e) => {
      if (e.pointerType === 'mouse' || e.target.closest('button')) return;
      const now = performance.now();
      const isTap = libraryTapStart && libraryTapStart.id === e.pointerId &&
        now - libraryTapStart.at < 350 && Math.hypot(e.clientX - libraryTapStart.x, e.clientY - libraryTapStart.y) < 10;
      libraryTapStart = null;
      if (!isTap) { libraryTap = 0; return; }
      if (now - libraryTap < 400) {
        sendAction({ action: 'add_layer', filename });
        lastTouchAdd = now;
        libraryTap = 0;
      } else {
        libraryTap = now;
      }
    });

    libraryGrid.appendChild(item);
  });
}

// --- Library uploads ---

const libUpload = document.getElementById('lib-upload');
const libUploadStatus = document.getElementById('lib-upload-status');

libUpload.addEventListener('change', async () => {
  const files = [...libUpload.files];
  libUpload.value = '';
  for (const file of files) {
    await uploadClip(file);
  }
  setTimeout(() => { libUploadStatus.textContent = ''; }, 4000);
});

function uploadClip(file, statusElement = libUploadStatus) {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      resolve(result);
    };
    xhr.open('POST', `/upload?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        statusElement.textContent = `${file.name} — ${Math.round((e.loaded / e.total) * 100)}%`;
      }
    };
    xhr.onload = () => {
      const response = xhr.responseText.trim();
      const ok = xhr.status === 200;
      let result = null;
      try { result = JSON.parse(response); } catch { /* legacy/plain transport refusal */ }
      const detail = result?.name || result?.error || response;
      statusElement.textContent = ok
        ? `${detail} added`
        : `${file.name}: ${detail || 'upload failed'}`;
      statusElement.classList.toggle('error', !ok);
      finish({ ok, filename: ok ? detail : '', error: ok ? '' : detail });
    };
    xhr.onerror = () => {
      statusElement.textContent = `${file.name}: upload failed`;
      statusElement.classList.add('error');
      finish({ ok: false, filename: '', error: 'upload failed' });
    };
    xhr.onabort = () => {
      statusElement.textContent = `${file.name}: upload cancelled`;
      statusElement.classList.add('error');
      finish({ ok: false, filename: '', error: 'upload cancelled' });
    };
    xhr.ontimeout = () => {
      statusElement.textContent = `${file.name}: upload timed out`;
      statusElement.classList.add('error');
      finish({ ok: false, filename: '', error: 'upload timed out' });
    };
    statusElement.classList.remove('error');
    statusElement.textContent = `${file.name} — 0%`;
    xhr.send(file);
  });
}

// --- Sync transport ---

function renderMasterTransport(paused, pending) {
  const btn = document.getElementById('btn-play-all');
  btn.textContent = paused ? '\u25B6' : '\u23F8';
  btn.title = paused
    ? 'Resume the complete visual program'
    : 'Freeze the complete visual program, including clocks and histories';
  btn.dataset.paused = String(paused);
  btn.disabled = pending;
  btn.toggleAttribute('aria-busy', pending);
  btn.setAttribute('aria-label', btn.title);
  btn.setAttribute('aria-pressed', String(!!paused));
}

function renderMediaFreeze(frozen, pending) {
  const btn = document.getElementById('btn-freeze-media');
  btn.classList.toggle('active', !!frozen);
  btn.title = frozen
    ? 'Release video and Spout frames'
    : 'Hold video and Spout frames while effects and modulation continue';
  btn.disabled = pending;
  btn.toggleAttribute('aria-busy', pending);
  btn.setAttribute('aria-label', btn.title);
  btn.setAttribute('aria-pressed', String(!!frozen));
}

function syncTransport(paused, mediaFrozen = false) {
  transportAuthoritativePaused = !!paused;
  if (transportPendingPaused === transportAuthoritativePaused) {
    transportRequestSequence += 1;
    transportPendingPaused = null;
  }
  renderMasterTransport(
    transportPendingPaused ?? transportAuthoritativePaused,
    transportPendingPaused !== null,
  );
  mediaAuthoritativeFrozen = !!mediaFrozen;
  if (mediaPendingFrozen === mediaAuthoritativeFrozen) {
    mediaRequestSequence += 1;
    mediaPendingFrozen = null;
  }
  renderMediaFreeze(
    mediaPendingFrozen ?? mediaAuthoritativeFrozen,
    mediaPendingFrozen !== null,
  );
}


// --- Export / Render ---

const historyUndoButton = document.getElementById('history-undo');
const historyRedoButton = document.getElementById('history-redo');
const historyStatus = document.getElementById('history-status');
const presetNameInput = document.getElementById('preset-name');
const presetKindSelect = document.getElementById('preset-kind');
const presetTargetSelect = document.getElementById('preset-target');
const presetCaptureButton = document.getElementById('preset-capture');
const presetList = document.getElementById('preset-list');
const presetStatus = document.getElementById('preset-status');
let latestPresetSnapshotKey = '';

historyUndoButton?.addEventListener('click', () => sendAction({ action: 'undo_manual' }));
historyRedoButton?.addEventListener('click', () => sendAction({ action: 'redo_manual' }));

document.addEventListener('keydown', (event) => {
  if (!(event.ctrlKey || event.metaKey) || event.altKey || event.key.toLowerCase() !== 'z') return;
  const active = document.activeElement;
  if (active?.matches?.('input[type="text"],input[type="number"],textarea,[contenteditable="true"]')) return;
  const action = event.shiftKey ? 'redo_manual' : 'undo_manual';
  const button = event.shiftKey ? historyRedoButton : historyUndoButton;
  if (button?.disabled) return;
  event.preventDefault();
  sendAction({ action });
});

function syncHistory(history = {}) {
  if (!historyUndoButton || !historyRedoButton || !historyStatus) return;
  historyUndoButton.disabled = !history.can_undo;
  historyRedoButton.disabled = !history.can_redo;
  const undoLabel = String(history.undo_label || '');
  const redoLabel = String(history.redo_label || '');
  historyUndoButton.textContent = undoLabel ? `Undo ${undoLabel}` : 'Undo';
  historyRedoButton.textContent = redoLabel ? `Redo ${redoLabel}` : 'Redo';
  historyUndoButton.title = undoLabel || 'No manual edit to undo';
  historyRedoButton.title = redoLabel || 'No manual edit to redo';
  const status = String(history.status || '');
  historyStatus.textContent = status || `${Number(history.undo_depth || 0)} undo · ${Number(history.redo_depth || 0)} redo · ${Number(history.bytes || 0).toLocaleString()} bytes`;
  historyStatus.className = /reject|error|stale|failed/i.test(status)
    ? 'export-status error'
    : 'export-status';
}

function presetTargetSnapshot(value) {
  if (value === 'master') return { scope: 'master' };
  if (value === 'controller_profile') return { scope: 'controller_profile' };
  if (value === 'stage_map') return { scope: 'stage_map' };
  const [scope, id] = String(value).split(':');
  if (!/^[1-9][0-9]*$/.test(id || '')) return null;
  if (scope === 'layer') return { scope: 'layer', layer_id: id };
  if (scope === 'group') return { scope: 'group', group_id: id };
  return null;
}

function presetTargets(kind) {
  if (kind === 'controller_profile') {
    return [{ value: 'controller_profile', label: 'Controller Profile document' }];
  }
  if (kind === 'stage_map') {
    return [{ value: 'stage_map', label: 'Stage Map document' }];
  }
  const targets = [];
  if (kind === 'transform' || kind === 'rack') {
    targets.push({ value: 'master', label: 'Master' });
  }
  if (kind !== 'group') {
    latestLayers.forEach((layer, index) => {
      const id = stableLayerId(layer);
      if (id) targets.push({ value: `layer:${id}`, label: `Layer ${index + 1}` });
    });
  }
  if (kind === 'transform' || kind === 'rack' || kind === 'matte' || kind === 'group') {
    (latestCreative?.groups || []).forEach((group, index) => {
      const id = String(group?.group_id || '');
      if (/^[1-9][0-9]*$/.test(id)) {
        targets.push({ value: `group:${id}`, label: group.name || `Group ${index + 1}` });
      }
    });
  }
  return targets;
}

function refreshPresetTargets(kind = presetKindSelect?.value || 'transform') {
  if (!presetTargetSelect) return;
  const previous = presetTargetSelect.value;
  const targets = presetTargets(kind);
  presetTargetSelect.replaceChildren(...targets.map(target => {
    const option = document.createElement('option');
    option.value = target.value;
    option.textContent = target.label;
    return option;
  }));
  if (targets.some(target => target.value === previous)) presetTargetSelect.value = previous;
  presetTargetSelect.disabled = targets.length === 0;
  if (presetCaptureButton) presetCaptureButton.disabled = presetRevision === 0 || targets.length === 0;
}

presetKindSelect?.addEventListener('change', () => refreshPresetTargets(presetKindSelect.value));

presetCaptureButton?.addEventListener('click', () => {
  const name = String(presetNameInput?.value || '').trim();
  const kind = String(presetKindSelect?.value || '');
  const target = presetTargetSnapshot(presetTargetSelect?.value);
  if (!name || name.length > 80 || !target || presetRevision === 0 || !layerStackRevision || !compositionRevision) {
    presetStatus.textContent = 'Choose a valid name and current scope.';
    presetStatus.className = 'export-status error';
    return;
  }
  sendAction({
    action: 'capture_scoped_preset',
    name,
    kind,
    target,
    preset_revision: presetRevision,
    layer_stack_revision: layerStackRevision,
    composition_revision: compositionRevision,
  });
});

function applyPreset(preset) {
  refreshPresetTargets(preset.kind);
  const target = presetTargetSnapshot(presetTargetSelect?.value);
  if (!target || !layerStackRevision || !compositionRevision || !presetRevision) return;
  sendAction({
    action: 'apply_scoped_preset',
    preset_id: String(preset.id),
    target,
    preset_revision: presetRevision,
    layer_stack_revision: layerStackRevision,
    composition_revision: compositionRevision,
  });
}

function syncPresets(snapshot = {}) {
  presetRevision = Number(snapshot.revision) || 0;
  const presets = Array.isArray(snapshot.presets) ? snapshot.presets.slice(0, 128) : [];
  const key = JSON.stringify([presetRevision, presets, snapshot.status || '']);
  if (key !== latestPresetSnapshotKey && presetList) {
    latestPresetSnapshotKey = key;
    presetList.replaceChildren(...presets.map(preset => {
      const row = document.createElement('div');
      row.className = 'preset-entry';
      const label = document.createElement('span');
      label.className = 'preset-entry-label';
      const kind = String(preset.kind || 'preset');
      label.textContent = String(preset.name || 'Untitled');
      label.title = `${label.textContent} · ${kind}`;
      const kindTag = document.createElement('span');
      kindTag.className = 'preset-entry-kind';
      kindTag.textContent = kind;
      const apply = document.createElement('button');
      apply.className = 'btn-export';
      apply.type = 'button';
      apply.textContent = 'Apply';
      apply.addEventListener('click', () => applyPreset(preset));
      const remove = document.createElement('button');
      remove.className = 'btn-export btn-cancel';
      remove.type = 'button';
      remove.textContent = 'Delete';
      remove.setAttribute('aria-label', `Delete ${label.textContent}`);
      remove.addEventListener('click', () => sendAction({
        action: 'delete_scoped_preset',
        preset_id: String(preset.id),
        preset_revision: presetRevision,
      }));
      row.append(label, kindTag, apply, remove);
      return row;
    }));
  }
  refreshPresetTargets(presetKindSelect?.value || 'transform');
  if (presetStatus) {
    const status = String(snapshot.status || '');
    presetStatus.textContent = status || (presets.length ? `${presets.length} value preset${presets.length === 1 ? '' : 's'}` : 'No presets');
    presetStatus.className = /reject|error|invalid|stale|mismatch|failed/i.test(status)
      ? 'export-status error'
      : 'export-status';
  }
}

document.getElementById('recovery-restore')?.addEventListener('click', () => {
  sendAction({ action: 'restore_recovery_journal' });
});
document.getElementById('recovery-discard')?.addEventListener('click', () => {
  sendAction({ action: 'discard_recovery_journal' });
});

function syncRecovery(available = false, status = '') {
  const panel = document.getElementById('recovery-panel');
  const statusElement = document.getElementById('recovery-status');
  const restore = document.getElementById('recovery-restore');
  if (!panel || !statusElement || !restore) return;
  const message = String(status || '');
  panel.hidden = !available && !message;
  restore.disabled = !available;
  statusElement.textContent = message || (available ? 'A valid checkpoint is available. It will never be applied automatically.' : '');
  statusElement.className = /corrupt|truncat|error|failed|reject/i.test(message)
    ? 'export-status error'
    : 'export-status';
}

const patchCaptureButton = document.getElementById('patch-capture');
const patchSaveStatus = document.getElementById('patch-save-status');
const patchLoadSnapshotButton = document.getElementById('patch-load-snapshot');
const patchApplyLookButton = document.getElementById('patch-apply-look');
const patchLoadStatus = document.getElementById('patch-load-status');

patchCaptureButton.addEventListener('click', () => {
  if (patchCaptureButton.disabled) return;
  if (sendAction({ action: 'quick_save_patch' })) {
    // Prevent an accidental double activation until the engine's next
    // snapshot takes authoritative ownership of button and status state.
    patchCaptureButton.disabled = true;
  } else {
    patchSaveStatus.textContent = 'Not connected';
    patchSaveStatus.className = 'export-status error';
  }
});

function syncPatchSave(status = '') {
  const text = String(status || '');
  const saving = text.startsWith('Saving');
  patchCaptureButton.disabled = saving;
  patchSaveStatus.textContent = text;
  patchSaveStatus.className = text.startsWith('Error:')
    ? 'export-status error'
    : text.startsWith('Saved ')
      ? 'export-status success'
      : 'export-status';
}

function requestPatchDialog(action) {
  if (patchLoadSnapshotButton.disabled || patchApplyLookButton.disabled) return;
  if (sendAction(action)) {
    patchLoadSnapshotButton.disabled = true;
    patchApplyLookButton.disabled = true;
    patchLoadStatus.textContent = 'Choose a YAML patch in the desktop window…';
    patchLoadStatus.className = 'export-status';
  } else {
    patchLoadStatus.textContent = 'Not connected';
    patchLoadStatus.className = 'export-status error';
  }
}

patchLoadSnapshotButton.addEventListener('click', () => {
  requestPatchDialog({ action: 'open_patch_snapshot' });
});

patchApplyLookButton.addEventListener('click', () => {
  requestPatchDialog({ action: 'open_patch_look', stack_revision: layerStackRevision });
});

function syncPatchLoad(status = '') {
  patchLoadSnapshotButton.disabled = false;
  patchApplyLookButton.disabled = false;
  if (!status) return;
  patchLoadStatus.textContent = String(status);
  patchLoadStatus.className = String(status).startsWith('Error:')
    ? 'export-status error'
    : 'export-status success';
}

document.getElementById('bundle-export').addEventListener('click', () => {
  if (!sendAction({ action: 'export_show_bundle' })) {
    syncShowBundle({ status: 'Not connected' });
  }
});
document.getElementById('bundle-preview').addEventListener('click', () => {
  if (!sendAction({ action: 'preview_show_bundle' })) {
    syncShowBundle({ status: 'Not connected' });
  }
});
document.getElementById('bundle-import').addEventListener('click', () => {
  sendAction({ action: 'confirm_show_bundle_import', load: false });
});
document.getElementById('bundle-import-load').addEventListener('click', () => {
  sendAction({ action: 'confirm_show_bundle_import', load: true });
});
document.getElementById('bundle-cancel').addEventListener('click', () => {
  sendAction({ action: 'cancel_show_bundle_import' });
});

function syncShowBundle(bundle = {}) {
  const status = document.getElementById('bundle-status');
  const pendingBox = document.getElementById('bundle-pending');
  const summary = document.getElementById('bundle-pending-summary');
  const list = document.getElementById('bundle-pending-entries');
  if (!status || !pendingBox || !summary || !list) return;
  const text = bundle.status || '';
  status.textContent = text;
  status.className = text.includes('rejected') || text === 'Not connected'
    ? 'export-status error'
    : 'export-status';
  const pending = bundle.pending;
  if (!pending) {
    pendingBox.hidden = true;
    list.textContent = '';
    return;
  }
  pendingBox.hidden = false;
  const patchDigest = String(pending.patch_sha256 || '').slice(0, 12);
  summary.textContent = `${pending.path} — ${pending.entry_count} entries, ` +
    `${pending.expanded_bytes} bytes expanded, patch ${patchDigest}…`;
  list.textContent = '';
  for (const entry of pending.entries || []) {
    const item = document.createElement('li');
    let line = `${entry.kind}: ${entry.logical_name} (${entry.byte_len} bytes`;
    if (!entry.authoritative) line += ', non-authoritative';
    if (entry.license) line += `, license: ${entry.license}`;
    item.textContent = line + ')';
    list.appendChild(item);
  }
  if (pending.entries_truncated) {
    const item = document.createElement('li');
    const shown = (pending.entries || []).length;
    item.textContent = `… ${pending.entry_count - shown} more entries`;
    list.appendChild(item);
  }
}

let exportActive = false;
let exportWarningsKey = null;

document.getElementById('export-start').addEventListener('click', () => {
  if (exportActive) return;
  const [w, h] = document.getElementById('export-resolution').value.split('x').map(Number);
  const durationInput = document.getElementById('export-duration');
  const duration = Math.min(300, Math.max(1, parseFloat(durationInput.value) || 10));
  durationInput.value = duration;
  const fps = [24, 30, 60, 120, 240].includes(parseInt(document.getElementById('export-fps').value))
    ? parseInt(document.getElementById('export-fps').value) : 30;
  const requestedNtscQuality = document.getElementById('export-ntsc-quality').value;
  const ntscQuality = ['live_parity', 'native'].includes(requestedNtscQuality)
    ? requestedNtscQuality : 'live_parity';
  const requestedShutterSamples = document.getElementById('export-shutter-samples').value;
  const shutterSamples = ['authored', 'samples_1', 'samples_4', 'samples_8', 'samples_16'].includes(requestedShutterSamples)
    ? requestedShutterSamples : 'authored';
  const audioSelect = document.getElementById('export-audio');
  const audioOption = audioSelect.selectedOptions[0];
  const audioLayerId = audioSelect.value === '' || audioSelect.value.startsWith('legacy-index:')
    ? null : audioSelect.value;
  const audioLayer = audioOption?.dataset.index === undefined ? null : parseInt(audioOption.dataset.index, 10);
  const requestedAlpha = document.getElementById('export-alpha').value;
  const alpha = ['straight_png_sequence', 'fill_key_png_sequence', 'straight_png_and_fill_key', 'ffv1_rgba']
    .includes(requestedAlpha) ? requestedAlpha : null;
  exportActive = true;
  document.getElementById('export-start').style.display = 'none';
  document.getElementById('export-cancel').style.display = '';
  document.getElementById('export-progress').style.display = '';
  document.getElementById('export-status').textContent = 'Starting render…';
  syncExportWarnings([]);
  if (!sendAction({ action: 'start_export', width: w, height: h, fps, duration_secs: duration, ntsc_quality: ntscQuality, shutter_samples: shutterSamples, audio_layer: audioLayer, audio_layer_id: audioLayerId, alpha })) {
    exportActive = false;
    document.getElementById('export-start').style.display = '';
    document.getElementById('export-cancel').style.display = 'none';
    document.getElementById('export-progress').style.display = 'none';
    document.getElementById('export-status').textContent = 'Not connected';
  }
});

document.getElementById('export-cancel').addEventListener('click', () => {
  document.getElementById('export-status').textContent = 'Cancelling…';
  sendAction({ action: 'cancel_export' });
});

function syncExportWarnings(warnings = []) {
  const panel = document.getElementById('export-warnings');
  const summary = document.getElementById('export-warnings-summary');
  const list = document.getElementById('export-warnings-list');
  if (!panel || !summary || !list) return;

  const messages = Array.isArray(warnings)
    ? warnings.filter(message => typeof message === 'string' && message.trim() !== '').slice(0, 128)
    : [];
  const warningsKey = JSON.stringify(messages);
  // State arrives around 30 times per second. Do not recreate an unchanged
  // aria-live list and make assistive technology announce it repeatedly.
  if (warningsKey === exportWarningsKey) return;
  exportWarningsKey = warningsKey;
  panel.hidden = messages.length === 0;
  summary.textContent = messages.length === 0
    ? ''
    : `${messages.length} source substitution${messages.length === 1 ? '' : 's'}:`;
  list.replaceChildren(...messages.map(message => {
    const item = document.createElement('li');
    item.textContent = message;
    return item;
  }));
}

function syncExportMotion(motion = {}) {
  const element = document.getElementById('export-motion-status');
  if (!element) return;
  const scopes = Array.isArray(motion.scopes) ? motion.scopes : [];
  const accepted = motion.accepted_frame;
  if (accepted === null || accepted === undefined) {
    element.textContent = '';
    return;
  }
  const fallbackCount = scopes.filter(scope => scope.field_attached && scope.rendered_source_origin === 'lattice_fallback').length;
  const unattachedCount = scopes.filter(scope => scope.field_planned && !scope.field_attached).length;
  const admittedCount = scopes.filter(scope => scope.transplant_admitted).length;
  const suffix = motion.scopes_truncated ? ' · scope list truncated' : '';
  element.textContent = `Motion metadata v${Number(motion.schema_version || 1)} · frame ${Number(accepted)} · ${scopes.length} scope${scopes.length === 1 ? '' : 's'} · ${fallbackCount} rendered fallback · ${unattachedCount} unattached · ${admittedCount} transplant · cross-GPU pixel identity not claimed${suffix}`;
}

function syncExport(progress, error, status = '', warnings = [], motion = {}) {
  const startBtn = document.getElementById('export-start');
  const cancelBtn = document.getElementById('export-cancel');
  const progressEl = document.getElementById('export-progress');
  const fillEl = document.getElementById('export-fill');
  const textEl = document.getElementById('export-text');
  const statusEl = document.getElementById('export-status');
  const warningCount = Array.isArray(warnings) ? warnings.length : 0;
  syncExportWarnings(warnings);
  syncExportMotion(motion || {});
  progress = Math.min(1, Math.max(0, Number(progress) || 0));
  progressEl.setAttribute('aria-valuenow', String(Math.round(progress * 100)));

  const terminal = status === 'succeeded' || status === 'failed' || status === 'cancelled';
  if (terminal || (status === '' && progress >= 1)) {
    // Done
    startBtn.style.display = '';
    cancelBtn.style.display = 'none';
    progressEl.style.display = 'none';
    exportActive = false;
    if (status === 'cancelled') {
      const cleanupError = error && error !== 'export cancelled' ? error : '';
      statusEl.textContent = cleanupError ? `Render cancelled: ${cleanupError}` : 'Render cancelled';
      statusEl.className = cleanupError ? 'export-status error' : 'export-status';
    } else if (status === 'failed' || error) {
      statusEl.textContent = 'Error: ' + (error || 'render failed');
      statusEl.className = 'export-status error';
    } else {
      statusEl.textContent = warningCount > 0
        ? `Render complete with ${warningCount} source substitution${warningCount === 1 ? '' : 's'}.`
        : 'Render complete!';
      statusEl.className = 'export-status success';
    }
  } else if (status === 'cancelling') {
    exportActive = true;
    startBtn.style.display = 'none';
    cancelBtn.style.display = 'none';
    progressEl.style.display = '';
    fillEl.style.width = (progress * 100) + '%';
    textEl.textContent = Math.round(progress * 100) + '%';
    statusEl.textContent = 'Cancelling\u2026';
    statusEl.className = 'export-status';
  } else if (status === 'running' || (status === '' && progress > 0 && progress < 1)) {
    // Rendering in progress. The numeric fallback exists only for legacy
    // servers; an explicit terminal status always wins even below 100%.
    exportActive = true;
    startBtn.style.display = 'none';
    cancelBtn.style.display = '';
    progressEl.style.display = '';
    fillEl.style.width = (progress * 100) + '%';
    textEl.textContent = Math.round(progress * 100) + '%';
    statusEl.textContent = '';
    statusEl.className = 'export-status';
  } else {
    // An explicit idle snapshot is authoritative. It also recovers controls
    // if a start action was lost during a disconnect after the local click.
    if (status === 'idle') {
      exportActive = false;
      startBtn.style.display = '';
      cancelBtn.style.display = 'none';
      progressEl.style.display = 'none';
      statusEl.textContent = '';
      statusEl.className = 'export-status';
    } else if (!exportActive) {
      startBtn.style.display = '';
      cancelBtn.style.display = 'none';
      progressEl.style.display = 'none';
    }
  }
}

// --- Modulation matrix ---

const MOD_TARGETS = [
  ['pixelate', 'Pixelate'],
  ['rgb_split', 'RGB Split'],
  ['hue_shift', 'Hue'],
  ['saturation', 'Saturation'],
  ['brightness', 'Brightness'],
  ['contrast', 'Contrast'],
  ['posterize', 'Posterize'],
  ['grain_intensity', 'Grain'],
  ['grain_size', 'Grain Size'],
  ['vignette', 'Vignette'],
  ['color_drift', 'Drift'],
  ['downsample', 'Downsample'],
  ['breathe_scale', 'Bth Scale'],
  ['breathe_rotation', 'Bth Rotate'],
  ['breathe_position', 'Bth Drift'],
  ['cellular_amount', 'Cell Amount'],
  ['cellular_scale', 'Cell Scale'],
  ['cellular_warp', 'Cell Warp'],
  ['cellular_speed', 'Cell Drift'],
  ['cellular_gap_amount', 'Master Cell Gap Key'],
  ['cellular_gap_threshold', 'Master Cell Gap Threshold'],
  ['cellular_gap_softness', 'Master Cell Gap Softness'],
  ['shift_amount', 'Shift Amount'],
  ['shift_block_size', 'Shift Block Size'],
  ['shift_density', 'Shift Density'],
  ['shift_speed', 'Shift Speed'],
  ['contour', 'Contour'],
  ['contour_bands', 'Contour Bands'],
  ['contour_width', 'Contour Width'],
  ['contour_hue', 'Contour Hue'],
  ['contour_fill', 'Contour Fill'],
  ['flatten', 'Flatten'],
  ['flatten_levels', 'Flatten Levels'],
  ['contour_dither', 'Contour Dither'],
  ['solarize', 'Solarize'],
  ['negative', 'Negative'],
  ['colourpass', 'Colour Pass'],
  ['colourpass_hue', 'Colour Pass Hue'],
  ['colourpass_width', 'Colour Pass Width'],
  ['edge_amount', 'Edge Amount'],
  ['edge_hue', 'Edge Hue'],
  ['emboss', 'Emboss'],
  ['emboss_angle', 'Emboss Angle'],
  ['halftone', 'Halftone'],
  ['halftone_pitch', 'Halftone Pitch'],
  ['halftone_angle', 'Halftone Angle'],
  ['moire', 'Moiré'],
  ['moire_freq', 'Moiré Frequency'],
  ['row_smear', 'Row Smear'],
  ['bitcrush', 'Bitcrush'],
  ['bitcrush_levels', 'Bitcrush Levels'],
  ['bitcrush_dither', 'Bitcrush Dither'],
  ['multi_grid_x', 'Multi Grid X'],
  ['multi_grid_y', 'Multi Grid Y'],
  ['barrel', 'Barrel'],
  ['chroma_aberration', 'Chroma Aberration'],
  ['anamorphic_streak', 'Anamorphic Streak'],
  ['key_border', 'Key Border'],
  ['key_shadow', 'Key Shadow'],
  ['position_x', 'Position X'],
  ['position_y', 'Position Y'],
  ['scale_x', 'Scale X'],
  ['scale_y', 'Scale Y'],
  ['anchor_x', 'Anchor X'],
  ['anchor_y', 'Anchor Y'],
  ['rotation_deg', 'Rotation'],
  ['skew_deg', 'Skew'],
  ['skew_axis_deg', 'Skew Axis'],
  ['crop_left', 'Crop Left'],
  ['crop_top', 'Crop Top'],
  ['crop_right', 'Crop Right'],
  ['crop_bottom', 'Crop Bottom'],
  ['key_threshold', 'Key Threshold'],
  ['key_softness', 'Key Softness'],
  ['key_color_r', 'Key Color Red'],
  ['key_color_g', 'Key Color Green'],
  ['key_color_b', 'Key Color Blue'],
  ['key_tolerance', 'Key Chroma Tolerance'],
  ['ntsc_snow', 'VHS Snow'],
  ['ntsc_tracking_snow', 'VHS Track Snow'],
  ['ntsc_edge_wave', 'VHS Edge Wave'],
  ['ntsc_edge_wave_speed', 'VHS Edge Wave Speed'],
  ['ntsc_head_shift', 'VHS Head Shift'],
  ['ntsc_tracking_wave', 'VHS Tracking Wave'],
  ['ntsc_chroma_loss', 'VHS Chroma Loss'],
  ['ntsc_composite_noise', 'VHS Composite Noise'],
  ['ntsc_luma_noise', 'VHS Luma Noise'],
  ['ntsc_chroma_noise', 'VHS Chroma Noise'],
  ['ntsc_luma_smear', 'VHS Luma Smear'],
  ['ntsc_sharpening', 'VHS Sharpening'],
  ['temporal_feedback', 'Temporal Feedback'],
  ['temporal_slitscan', 'Temporal Slit-Scan'],
  ['temporal_fb_zoom', 'Temporal FB Zoom'],
  ['temporal_fb_rotate', 'Temporal FB Rotate'],
  ['temporal_fb_offset_x', 'Temporal FB Offset X'],
  ['temporal_fb_offset_y', 'Temporal FB Offset Y'],
  ['temporal_fb_hue_rotate', 'Temporal FB Hue Rotate'],
  ['temporal_fb_saturation', 'Temporal FB Saturation'],
  ['temporal_fb_gain_r', 'Temporal FB Gain Red'],
  ['temporal_fb_gain_g', 'Temporal FB Gain Green'],
  ['temporal_fb_gain_b', 'Temporal FB Gain Blue'],
  ['temporal_fb_chroma_displace', 'Temporal FB Chroma Displace'],
  ['temporal_fb_blur', 'Temporal FB Blur'],
  ['temporal_fb_sharpen', 'Temporal FB Sharpen'],
  ['temporal_fb_drive', 'Temporal FB Drive'],
  ['temporal_fb_pivot', 'Temporal FB Pivot'],
  ['temporal_fb_threshold', 'Temporal FB Threshold'],
  ['temporal_fb_noise', 'Temporal FB Noise'],
  ['temporal_slit_angle', 'Temporal Slit Angle'],
  ['temporal_key_threshold', 'Temporal Key Threshold'],
  ['temporal_key_softness', 'Temporal Key Softness'],
  ['temporal_key_history', 'Temporal Key History'],
  ['temporal_loom_amount', 'Temporal Loom Amount'],
  ['temporal_loom_depth', 'Temporal Loom Depth'],
  ['temporal_loom_phase', 'Temporal Loom Phase'],
  ['temporal_loom_scale', 'Temporal Loom Scale'],
  ['temporal_loom_angle', 'Temporal Loom Angle'],
  ['temporal_atlas_amount', 'Temporal Atlas Amount'],
  ['temporal_atlas_collision', 'Temporal Atlas Collision'],
  ['temporal_garden_amount', 'Refresh Garden Amount'],
  ['temporal_garden_threshold', 'Refresh Garden Threshold'],
  ['temporal_garden_softness', 'Refresh Garden Softness'],
  ['temporal_garden_decay', 'Refresh Garden Decay'],
  ['temporal_long_exposure_amount', 'Temporal Long Exposure'],
  ['display_il_amount', 'Display Interlace Amount'],
  ['display_il_twitter', 'Display Interlace Twitter'],
  ['display_il_judder', 'Display Interlace Judder'],
  ['display_phosphor', 'Display Phosphor Persistence'],
  ['display_phos_r', 'Display Phosphor Red'],
  ['display_phos_g', 'Display Phosphor Green'],
  ['display_phos_b', 'Display Phosphor Blue'],
  ['display_scanlines', 'Display Scanlines'],
  ['display_beam_width', 'Display Beam Width'],
  ['display_beam_shape', 'Display Beam Shape'],
  ['display_mask_strength', 'Display Mask Strength'],
  ['display_mask_dark', 'Display Mask Darkness'],
  ['display_bloom', 'Display Bloom'],
  ['display_bloom_radius', 'Display Bloom Radius'],
  ['display_halation', 'Display Halation'],
  ['display_defocus', 'Display Defocus'],
  ['display_sag', 'Display Sag'],
  ['melt_amount', 'Melt Amount'],
  ['melt_width', 'Melt Width'],
  ['melt_hold', 'Melt Hold'],
  ['melt_swirl', 'Melt Swirl'],
  ['melt_chroma', 'Melt Chroma'],
  ['melt_creep', 'Melt Creep'],
  ['mosh_amount', 'Mosh Amount'],
  ['mosh_key_removal', 'Mosh Key Removal'],
  ['mosh_hold', 'Mosh Delta Hold'],
  ['mosh_drop', 'Mosh Drop'],
  ['mosh_shuffle', 'Mosh Chunk Shuffle'],
  ['mosh_rate', 'Mosh Rate'],
  ['mosh_bitrate_starve', 'Mosh Bitrate Starve'],
  ['mosh_resync', 'Mosh Resync'],
  ['mosh_wipe', 'Mosh Motion Wipe'],
  ['mosh_smear', 'Mosh Vector Smear'],
  ['mosh_trail', 'Mosh Motion Trail'],
  ['sync_amount', 'Sync Amount'],
  ['sync_rate', 'Sync Rate'],
  ['sync_spread', 'Sync Spread'],
  ['sync_bias', 'Sync Bias'],
  ['motion_shutter_angle', 'Motion Shutter Angle'],
  ['motion_shutter_phase', 'Motion Shutter Phase'],
  ['motion_shutter_curvature', 'Motion Shutter Curvature'],
  ['motion_shutter_chromatic_lag', 'Motion Shutter Chroma Lag'],
  ['motion_field_scale', 'Motion Field Scale'],
  ['motion_field_rate', 'Motion Field Rate'],
  ['motion_stretch', 'Motion Stretch'],
  ['motion_edge_repel', 'Motion Edge Repel'],
  ['motion_vector_trash', 'Motion Vector Trash'],
  ['motion_trash_block_size', 'Motion Trash Block Size'],
  ['gesture_radius', 'Gesture Radius'],
  ['gesture_strength', 'Gesture Strength'],
  ['gesture_retention', 'Gesture Retention'],
  ['morph', 'Morph'],
];
const MASTER_MOD_TARGETS = MOD_TARGETS.slice();

const LAYER_FX_TARGETS = [
  ['opacity', 'Opacity'], ['speed', 'Speed'], ['fps', 'FPS'],
  ['key_threshold', 'Key Threshold'],
  ['key_color_r', 'Key Color Red'], ['key_color_g', 'Key Color Green'],
  ['key_color_b', 'Key Color Blue'], ['key_tolerance', 'Key Chroma Tolerance'],
  ['pixelate', 'Pixelate'], ['rgb_split', 'RGB Split'],
  ['hue_shift', 'Hue'], ['saturation', 'Saturation'],
  ['brightness', 'Brightness'], ['contrast', 'Contrast'],
  ['posterize', 'Posterize'], ['grain_intensity', 'Grain'],
  ['grain_size', 'Grain Size'], ['vignette', 'Vignette'],
  ['color_drift', 'Drift'], ['breathe_scale', 'Bth Scale'],
  ['breathe_rotation', 'Bth Rotate'], ['breathe_position', 'Bth Drift'],
  ['cellular_amount', 'Cell Amount'], ['cellular_scale', 'Cell Scale'],
  ['cellular_warp', 'Cell Warp'], ['cellular_speed', 'Cell Drift'],
  ['cellular_gap_amount', 'Cell Gap Key'],
  ['cellular_gap_threshold', 'Cell Gap Thresh'],
  ['cellular_gap_softness', 'Cell Gap Soft'],
  ['key_softness', 'Key Soft'], ['downsample', 'Downsample'],
  ['shift_amount', 'Shift Amount'], ['shift_block_size', 'Shift Block Size'],
  ['shift_density', 'Shift Density'], ['shift_speed', 'Shift Speed'],
  ['position_x', 'Position X'], ['position_y', 'Position Y'],
  ['scale_x', 'Scale X'], ['scale_y', 'Scale Y'],
  ['anchor_x', 'Anchor X'], ['anchor_y', 'Anchor Y'],
  ['rotation_deg', 'Rotation'], ['skew_deg', 'Skew'],
  ['skew_axis_deg', 'Skew Axis'],
  ['crop_left', 'Crop Left'], ['crop_top', 'Crop Top'],
  ['crop_right', 'Crop Right'], ['crop_bottom', 'Crop Bottom'],
  ['motion_transplant_amount', 'Motion Transplant'],
  ['motion_confidence_threshold', 'Motion Confidence'],
  ['motion_confidence_softness', 'Motion Confidence Softness'],
  ['motion_refresh', 'Motion Refresh'], ['motion_decay', 'Motion Decay'],
  ['motion_occlusion', 'Motion Occlusion'],
  ['motion_shutter_angle', 'Motion Shutter Angle'],
  ['motion_shutter_phase', 'Motion Shutter Phase'],
  ['motion_shutter_curvature', 'Motion Shutter Curvature'],
  ['motion_shutter_chromatic_lag', 'Motion Shutter Chroma Lag'],
  ['motion_field_scale', 'Motion Field Scale'],
  ['motion_field_rate', 'Motion Field Rate'],
  ['motion_stretch', 'Motion Stretch'],
  ['motion_edge_repel', 'Motion Edge Repel'],
  ['motion_vector_trash', 'Motion Vector Trash'],
  ['motion_trash_block_size', 'Motion Trash Block Size'],
  ['contour', 'Contour'], ['contour_bands', 'Contour Bands'],
  ['contour_width', 'Contour Width'], ['contour_hue', 'Contour Hue'],
  ['contour_fill', 'Contour Fill'], ['flatten', 'Flatten'],
  ['flatten_levels', 'Flatten Levels'], ['contour_dither', 'Contour Dither'],
  ['solarize', 'Solarize'], ['negative', 'Negative'],
  ['colourpass', 'Colour Pass'], ['colourpass_hue', 'Colour Pass Hue'],
  ['colourpass_width', 'Colour Pass Width'], ['edge_amount', 'Edge Amount'],
  ['edge_hue', 'Edge Hue'], ['emboss', 'Emboss'],
  ['emboss_angle', 'Emboss Angle'], ['halftone', 'Halftone'],
  ['halftone_pitch', 'Halftone Pitch'], ['halftone_angle', 'Halftone Angle'],
  ['moire', 'Moiré'], ['moire_freq', 'Moiré Frequency'],
  ['row_smear', 'Row Smear'], ['bitcrush', 'Bitcrush'],
  ['bitcrush_levels', 'Bitcrush Levels'], ['bitcrush_dither', 'Bitcrush Dither'],
  ['multi_grid_x', 'Multi Grid X'], ['multi_grid_y', 'Multi Grid Y'],
  ['key_border', 'Key Border'], ['key_shadow', 'Key Shadow'],
  ['pattern_freq_x', 'Pattern Freq X'], ['pattern_freq_y', 'Pattern Freq Y'],
  ['pattern_phase', 'Pattern Phase'], ['pattern_rate', 'Pattern Rate'],
  ['pattern_cross_mod', 'Pattern Cross Mod'], ['pattern_wavefold', 'Pattern Wavefold'],
  ['pattern_pulse_width', 'Pattern Pulse Width'], ['pattern_comparator', 'Pattern Comparator'],
  ['pattern_comp_threshold', 'Pattern Comp Thresh'], ['pattern_comp_soft', 'Pattern Comp Soft'],
  ['pattern_symmetry', 'Pattern Symmetry'], ['pattern_zoom', 'Pattern Scale'],
  ['pattern_rotate', 'Pattern Rotate'], ['pattern_skew', 'Pattern Skew'],
  ['pattern_center_x', 'Pattern Centre X'], ['pattern_center_y', 'Pattern Centre Y'],
  ['pattern_warp', 'Pattern Warp'], ['pattern_hue', 'Pattern Hue'],
  ['pattern_hue_spread', 'Pattern Hue Spread'], ['pattern_saturation', 'Pattern Saturation'],
  ['pattern_brightness', 'Pattern Brightness'], ['pattern_color_bands', 'Pattern Colour Bands'],
  ['mosh_send', 'Mosh Send'],
];

const ROUTING_CURVES = [
  ['linear', 'Linear'],
  ['exp', 'Exp'],
  ['log', 'Log'],
  ['s_curve', 'S-Curve'],
  ['steps', 'Steps'],
];

function curveUsesAmount(curve) {
  return curve === 'exp' || curve === 'log' || curve === 'steps';
}

function syncCurveAmountState(curveSelect, amountInput) {
  const enabled = curveUsesAmount(curveSelect.value);
  amountInput.disabled = !enabled;
  amountInput.title = enabled ? '' : `${curveSelect.options[curveSelect.selectedIndex]?.text || 'This curve'} has a fixed response`;
  syncRangeEditorState(amountInput);
}

const LFO_SHAPES = [
  ['sine', 'Sine'],
  ['triangle', 'Tri'],
  ['saw', 'Saw'],
  ['square', 'Sqr'],
  ['sample_hold', 'S&H'],
];
const NUM_LFOS = 8;

const LFO_RATES = [
  [16, '4 bars'],
  [8, '2 bars'],
  [4, '1 bar'],
  [2, '1/2'],
  [1, '1/4'],
  [0.5, '1/8'],
  [0.25, '1/16'],
];

const bpmInput = document.getElementById('bpm-input');
const lfoList = document.getElementById('lfo-list');
const routingList = document.getElementById('routing-list');
const beatQuantize = document.getElementById('beat-quantize');
const beatQuantizeStatus = document.getElementById('beat-quantize-status');

beatQuantize.addEventListener('change', () => {
  beatQuantizeEnabled = beatQuantize.checked;
  syncQuantize(0);
});

function syncQuantize(pending) {
  if (pending > 0) {
    beatQuantizeStatus.textContent = `${pending} pending`;
  } else {
    beatQuantizeStatus.textContent = beatQuantizeEnabled ? 'next downbeat' : '';
  }
}

bpmInput.addEventListener('change', () => {
  const v = parseFloat(bpmInput.value);
  if (!isNaN(v)) sendAction({ action: 'set_bpm', value: v });
});

document.getElementById('tap-tempo').addEventListener('click', () => {
  sendAction({ action: 'tap_tempo' });
});

document.getElementById('add-routing').addEventListener('click', () => {
  sendAction({ action: 'add_routing' });
});

function optionsHtml(pairs, selected, groups = null) {
  if (groups) {
    return groups
      .map(([label, values]) => `<optgroup label="${escapeHtml(label)}">${optionsHtml(values, selected)}</optgroup>`)
      .join('');
  }
  return pairs
    .map(([v, label]) => `<option value="${escapeHtml(v)}" ${String(v) === String(selected) ? 'selected' : ''}>${escapeHtml(label)}</option>`)
    .join('');
}

// Build the fixed engine LFO bank once. Lanes 5–8 remain neutral until a
// routing selects them, exactly like an untouched legacy lane.
for (let i = 0; i < NUM_LFOS; i++) {
  const row = document.createElement('div');
  row.className = 'lfo-row';
  row.dataset.lfo = i;
  row.innerHTML = `
    <span class="lfo-name">LFO ${i + 1}</span>
    <select class="lfo-shape">${optionsHtml(LFO_SHAPES, 'sine')}</select>
    <select class="lfo-rate">${optionsHtml(LFO_RATES, 4)}</select>
    <input class="lfo-seed seed-input" type="number" min="0" max="4294967295" step="1" inputmode="numeric" value="0" aria-label="LFO ${i + 1} sample and hold seed" title="Sample &amp; Hold seed; zero reproduces the legacy sequence" hidden>
    <div class="lfo-meter"><div class="lfo-meter-fill"></div></div>
  `;
  row.querySelector('.lfo-shape').addEventListener('change', (e) => {
    sendAction({ action: 'set_lfo', index: i, param: 'shape', value: e.target.value });
    row.querySelector('.lfo-seed').hidden = e.target.value !== 'sample_hold';
  });
  row.querySelector('.lfo-rate').addEventListener('change', (e) => {
    sendAction({ action: 'set_lfo', index: i, param: 'beats', value: parseFloat(e.target.value) });
  });
  row.querySelector('.lfo-seed').addEventListener('change', (e) => {
    const seed = Number(e.target.value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      e.target.setCustomValidity('Seed must be a whole number from 0 to 4294967295');
      e.target.reportValidity();
      return;
    }
    e.target.setCustomValidity('');
    sendAction({ action: 'set_lfo', index: i, param: 'seed', value: seed });
  });
  lfoList.appendChild(row);
}

const MOD_SOURCES = [
  ['lfo0', 'L1'],
  ['lfo1', 'L2'],
  ['lfo2', 'L3'],
  ['lfo3', 'L4'],
  ['lfo4', 'L5'],
  ['lfo5', 'L6'],
  ['lfo6', 'L7'],
  ['lfo7', 'L8'],
  ['audio_level', 'Level'],
  ['audio_bass', 'Bass'],
  ['audio_mid', 'Mid'],
  ['audio_high', 'High'],
  ['audio_band1', 'Band 1'],
  ['audio_band2', 'Band 2'],
  ['audio_band3', 'Band 3'],
  ['audio_band4', 'Band 4'],
  ['audio_band5', 'Band 5'],
  ['audio_band6', 'Band 6'],
  ['audio_band7', 'Band 7'],
  ['audio_band8', 'Band 8'],
  ['audio_onset', 'Onset'],
  ['audio_bright', 'Bright'],
  ['audio_noise', 'Noise'],
  ['midi_a', 'MIDI A'],
  ['midi_b', 'MIDI B'],
  ['midi_c', 'MIDI C'],
  ['midi_d', 'MIDI D'],
  ['gyro_yaw', 'Gyro Yaw'],
  ['gyro_pitch', 'Gyro Pitch'],
  ['gyro_roll', 'Gyro Roll'],
  ['pad_x', 'Pad X'],
  ['pad_y', 'Pad Y'],
  ['bend1', 'Bend 1'],
  ['bend2', 'Bend 2'],
  ['bend3', 'Bend 3'],
  ['bend4', 'Bend 4'],
  ['bend5', 'Bend 5'],
  ['bend6', 'Bend 6'],
  ['env1', 'Env 1'],
  ['env2', 'Env 2'],
  ['env3', 'Env 3'],
  ['env4', 'Env 4'],
  ['macro1', 'Macro 1'],
  ['macro2', 'Macro 2'],
  ['macro3', 'Macro 3'],
  ['macro4', 'Macro 4'],
  ['chaos', 'Chaos'],
  ['drift', 'Drift'],
  ['spike', 'Spike'],
  ['video_motion', 'Vid Motion'],
  ['video_brightness', 'Vid Bright'],
  ['video_cut', 'Vid Cut'],
];

// B10 perform sources: envelopes, macros, bend pads, and the generator seed.
const ENVELOPE_TRIGGERS = [
  ['bend1', 'Bend 1'], ['bend2', 'Bend 2'], ['bend3', 'Bend 3'],
  ['bend4', 'Bend 4'], ['bend5', 'Bend 5'], ['bend6', 'Bend 6'],
  ['audio_onset', 'Audio Onset'], ['scene_cut', 'Scene Cut'],
  ['beat', 'Every Beat'], ['beat2', 'Every 2 Beats'], ['bar', 'Every Bar'],
];
const ENVELOPE_MODES = [['once', 'One Shot'], ['gate', 'Gate'], ['loop', 'Loop']];

const envelopeList = document.getElementById('envelope-list');
if (envelopeList) {
  for (let i = 0; i < 4; i++) {
    const row = document.createElement('div');
    row.className = 'envelope-row';
    row.dataset.envelope = i;
    row.innerHTML = `
      <span class="lfo-name">ENV ${i + 1}</span>
      <select class="env-trigger" aria-label="Envelope ${i + 1} trigger">${optionsHtml(ENVELOPE_TRIGGERS, 'bend1')}</select>
      <select class="env-mode" aria-label="Envelope ${i + 1} mode">${optionsHtml(ENVELOPE_MODES, 'once')}</select>
      <div class="param-row" data-min="0.005" data-max="10" data-step="0.005">
        <label>Atk</label><input type="range" min="0.005" max="10" step="0.005" value="0.02" aria-label="Envelope ${i + 1} attack seconds"><span class="value">0.02</span>
      </div>
      <div class="param-row" data-min="0.02" data-max="30" data-step="0.01">
        <label>Dec</label><input type="range" min="0.02" max="30" step="0.01" value="0.5" aria-label="Envelope ${i + 1} decay seconds"><span class="value">0.5</span>
      </div>
      <div class="lfo-meter"><div class="lfo-meter-fill"></div></div>
    `;
    row.querySelector('.env-trigger').addEventListener('change', (e) => {
      sendAction({ action: 'set_envelope', index: i, param: 'trigger', value: e.target.value });
    });
    row.querySelector('.env-mode').addEventListener('change', (e) => {
      sendAction({ action: 'set_envelope', index: i, param: 'mode', value: e.target.value });
    });
    const [attackRow, decayRow] = row.querySelectorAll('.param-row');
    for (const [paramRow, param] of [[attackRow, 'attack'], [decayRow, 'decay']]) {
      const slider = paramRow.querySelector('input[type="range"]');
      slider.addEventListener('input', () => {
        const value = parseFloat(slider.value);
        paramRow.querySelector('.value').textContent = value.toFixed(3);
        sendAction({ action: 'set_envelope', index: i, param, value });
      });
    }
    envelopeList.appendChild(row);
  }
}

const macroList = document.getElementById('macro-list');
if (macroList) {
  for (let i = 0; i < 4; i++) {
    const row = document.createElement('div');
    row.className = 'param-row';
    row.dataset.macro = i;
    row.dataset.min = '0';
    row.dataset.max = '1';
    row.dataset.step = '0.001';
    row.innerHTML = `
      <label>Macro ${i + 1}</label><input type="range" min="0" max="1" step="0.001" value="0" aria-label="Macro ${i + 1}"><span class="value">0.000</span>
    `;
    const slider = row.querySelector('input[type="range"]');
    slider.addEventListener('input', () => {
      const value = parseFloat(slider.value);
      row.querySelector('.value').textContent = value.toFixed(3);
      sendAction({ action: 'set_macro', index: i, value });
    });
    macroList.appendChild(row);
  }
}

const modSeedInput = document.getElementById('mod-seed');
if (modSeedInput) {
  modSeedInput.addEventListener('change', () => {
    const seed = Number(modSeedInput.value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      modSeedInput.setCustomValidity('Seed must be a whole number from 0 to 4294967295');
      modSeedInput.reportValidity();
      return;
    }
    modSeedInput.setCustomValidity('');
    sendAction({ action: 'set_mod_seed', seed });
  });
}

// Bend pads follow the XY pad's press/release machinery literally: pointer
// capture, forced edge sends, and release on blur/hide/reconnect so a
// swallowed pointer-up can never latch a pad on.
const bendHeldLocal = [false, false, false, false, false, false];
function sendBend(index, held) {
  bendHeldLocal[index] = held;
  const pad = document.querySelector(`.bend-pad[data-bend="${index}"]`);
  if (pad) pad.setAttribute('aria-pressed', held ? 'true' : 'false');
  sendAction({ action: 'bend_pad', index, held });
}
function releaseAllBends() {
  for (let i = 0; i < 6; i++) {
    if (bendHeldLocal[i]) sendBend(i, false);
  }
}
for (const pad of document.querySelectorAll('.bend-pad')) {
  const index = Number(pad.dataset.bend);
  pad.addEventListener('pointerdown', (e) => {
    pad.setPointerCapture(e.pointerId);
    sendBend(index, true);
    e.preventDefault();
  });
  pad.addEventListener('pointerup', () => sendBend(index, false));
  pad.addEventListener('pointercancel', () => sendBend(index, false));
  pad.addEventListener('lostpointercapture', () => {
    if (bendHeldLocal[index]) sendBend(index, false);
  });
  // Keyboard accessibility: a keypress is a tap, never a hold.
  pad.addEventListener('keydown', (e) => {
    if (e.key === ' ' || e.key === 'Enter') {
      sendBend(index, true);
      setTimeout(() => sendBend(index, false), 120);
      e.preventDefault();
    }
  });
}
window.addEventListener('blur', releaseAllBends);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) releaseAllBends();
});

function currentModTargets(selected = '') {
  const groups = [['Master / Program', MASTER_MOD_TARGETS.slice()]];
  const liveLayerCount = latestLayerIdentities.length;
  for (let layer = 1; layer <= liveLayerCount; layer++) {
    const targets = [];
    for (const [suffix, label] of LAYER_FX_TARGETS) {
      targets.push([`layer${layer}_${suffix}`, `L${layer} ${label}`]);
    }
    groups.push([`Layer ${layer}`, targets]);
  }
  if (latestCreative) {
    groups[0][1].push(['composition/bus_crossfade', 'Composition · A / B Crossfade']);
    for (const [suffix, label] of [
      ['bus_wipe_soft', 'Wipe Softness'], ['bus_wipe_x', 'Wipe Center X'], ['bus_wipe_y', 'Wipe Center Y'],
      ['bus_wipe_detail', 'Wipe Detail'], ['bus_wipe_border', 'Wipe Border'],
      ['bus_dirt', 'Dirt'], ['bus_dirt_rate', 'Dirt Rate'], ['bus_dirt_drop', 'Dirt Dropout'],
      ['bus_dirt_cut', 'Dirt Cut'], ['bus_dirt_knock', 'Dirt Knock'], ['bus_dirt_noise', 'Dirt Noise'],
      ['bus_melt', 'Edge Melt'], ['bus_melt_width', 'Melt Width'], ['bus_melt_hold', 'Melt Hold'],
      ['bus_melt_swirl', 'Melt Swirl'], ['bus_melt_chroma', 'Melt Chroma'], ['bus_melt_creep', 'Melt Creep'],
    ]) {
      groups[0][1].push([`composition/${suffix}`, `Composition · ${label}`]);
    }
    const nodeTargets = (scopeKey, scopeLabel, rack) => {
      const values = [];
      for (const node of rack?.nodes || []) {
        if (CREATIVE_NODE_INFO[node.kind]?.marker) continue;
        const prefix = `${scopeLabel} · ${CREATIVE_NODE_INFO[node.kind]?.label || node.kind} #${node.node_id}`;
        values.push([`node/${scopeKey}/${node.node_id}/wet`, `${prefix} · Wet`]);
        for (const def of creativeNodeVisibleDefs(node)) {
          if (def.type === 'float') values.push([`node/${scopeKey}/${node.node_id}/${def.key}`, `${prefix} · ${def.label}`]);
          if (def.type === 'vec') {
            const suffixes = def.components.length === 3 ? ['r', 'g', 'b'] : ['x', 'y'];
            suffixes.forEach((suffix, index) => values.push([`node/${scopeKey}/${node.node_id}/${def.key}.${suffix}`, `${prefix} · ${def.label} ${def.components[index]}`]));
          }
        }
      }
      return values;
    };
    const masterNodes = nodeTargets('master', 'Master rack', latestCreative.master_rack);
    if (masterNodes.length) groups.push(['Master Collision Rack', masterNodes]);
    for (const [layerId, rack] of latestCreative.layer_racks || []) {
      const values = nodeTargets(`layer/${layerId}`, creativeLayerLabel(layerId), rack);
      if (values.length) groups.push([`${creativeLayerLabel(layerId)} Rack`, values]);
    }
    for (const group of latestCreative.groups || []) {
      const groupLabel = group.name || `Group ${group.group_id}`;
      const values = [
        [`group/${group.group_id}/opacity`, `${groupLabel} · Opacity`],
        [`group/${group.group_id}/position.x`, `${groupLabel} · Position X`],
        [`group/${group.group_id}/position.y`, `${groupLabel} · Position Y`],
        [`group/${group.group_id}/scale.x`, `${groupLabel} · Scale X`],
        [`group/${group.group_id}/scale.y`, `${groupLabel} · Scale Y`],
        [`group/${group.group_id}/anchor.x`, `${groupLabel} · Anchor X`],
        [`group/${group.group_id}/anchor.y`, `${groupLabel} · Anchor Y`],
        [`group/${group.group_id}/rotation_deg`, `${groupLabel} · Rotation`],
        [`group/${group.group_id}/skew_deg`, `${groupLabel} · Skew`],
        [`group/${group.group_id}/skew_axis_deg`, `${groupLabel} · Skew axis`],
        [`group/${group.group_id}/crop_left`, `${groupLabel} · Crop left`],
        [`group/${group.group_id}/crop_top`, `${groupLabel} · Crop top`],
        [`group/${group.group_id}/crop_right`, `${groupLabel} · Crop right`],
        [`group/${group.group_id}/crop_bottom`, `${groupLabel} · Crop bottom`],
        ...(group.matte ? [
          [`group/${group.group_id}/matte.amount`, `${groupLabel} · Matte amount`],
          [`group/${group.group_id}/matte.threshold`, `${groupLabel} · Matte threshold`],
          [`group/${group.group_id}/matte.softness`, `${groupLabel} · Matte softness`],
        ] : []),
        ...nodeTargets(`group/${group.group_id}`, groupLabel, group.rack),
      ];
      groups.push([`${groupLabel} Values + Rack`, values]);
    }
  }
  // Preserve visibility for a legacy/out-of-range selection until the engine
  // publishes its authoritative replacement instead of showing a blank menu.
  if (selected && !groups.some(([, values]) => values.some(([value]) => value === selected))) {
    groups.push(['Unknown or legacy target', [[selected, `${selected} (not recognized by this build)`]]]);
  }
  return groups;
}

function createRoutingRow(routing, index) {
  const row = document.createElement('div');
  row.className = 'routing-row';
  row.dataset.index = index;
  row.dataset.routeId = routing.route_id || '';
  const selector = () => ({ index: Number(row.dataset.index), route_id: row.dataset.routeId || null });
  row.innerHTML = `
    <select class="routing-source" aria-label="Modulation source">${optionsHtml(MOD_SOURCES, routing.source)}</select>
    <span class="routing-arrow">&#x2192;</span>
    <select class="routing-target" aria-label="Modulation target">${optionsHtml([], routing.target, currentModTargets(routing.target))}</select>
    <input type="range" class="routing-depth" min="-1" max="1" step="0.01" value="${routing.depth}" aria-label="Modulation depth">
    <span class="routing-depth-val">${routing.depth.toFixed(2)}</span>
    <button class="routing-remove" title="Remove routing" aria-label="Remove modulation routing">&#xD7;</button>
    <div class="routing-activity" role="meter" aria-label="Shaped and slewed modulation source value" aria-valuemin="-1" aria-valuemax="1" aria-valuenow="0">
      <span class="routing-activity-center" aria-hidden="true"></span>
      <span class="routing-activity-fill" aria-hidden="true"></span>
      <span class="routing-activity-value">0.00</span>
    </div>
    <div class="routing-response">
      <select class="routing-curve" aria-label="Response curve">${optionsHtml(ROUTING_CURVES, routing.curve || 'linear')}</select>
      <input type="range" class="routing-curve-amount" min="-2" max="2" step="0.05" value="${routing.curve_amount || 0}" aria-label="Curve amount">
      <label>A <input type="number" class="routing-attack" min="0" max="10" step="0.01" value="${routing.attack || 0}" aria-label="Attack seconds"></label>
      <label>R <input type="number" class="routing-release" min="0" max="10" step="0.01" value="${routing.release || 0}" aria-label="Release seconds"></label>
    </div>
  `;
  row.querySelector('.routing-source').addEventListener('change', (e) => {
    sendAction({ action: 'set_routing', ...selector(), param: 'source', value: e.target.value });
  });
  row.querySelector('.routing-target').addEventListener('change', (e) => {
    const target = e.target.value;
    const layerMatch = /^layer([1-9]\d*)_/.exec(target);
    let targetIdentity = {};
    if (layerMatch) {
      const targetLayerId = latestLayerIdentities[Number(layerMatch[1]) - 1];
      if (!targetLayerId) {
        // A positional layer target without a corresponding current identity
        // is unsafe: a concurrent add/remove could bind it to another clip.
        e.target.value = routing.target;
        return;
      }
      targetIdentity = { target_layer_id: targetLayerId };
      if (layerStackRevision > 0) targetIdentity.layer_stack_revision = layerStackRevision;
    }
    sendAction({ action: 'set_routing', ...selector(), ...targetIdentity, param: 'target', value: target });
  });
  row.querySelector('.routing-depth').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    row.querySelector('.routing-depth-val').textContent = v.toFixed(2);
    sendAction({ action: 'set_routing', ...selector(), param: 'depth', value: v });
  });
  resetRangeOnDoubleActivation(row.querySelector('.routing-depth'), 0);
  const curveSelect = row.querySelector('.routing-curve');
  const curveAmount = row.querySelector('.routing-curve-amount');
  curveSelect.addEventListener('change', (e) => {
    syncCurveAmountState(curveSelect, curveAmount);
    sendAction({ action: 'set_routing', ...selector(), param: 'curve', value: e.target.value });
  });
  curveAmount.addEventListener('input', (e) => {
    sendAction({ action: 'set_routing', ...selector(), param: 'curve_amount', value: parseFloat(e.target.value) });
  });
  resetRangeOnDoubleActivation(curveAmount, 0);
  syncCurveAmountState(curveSelect, curveAmount);
  for (const param of ['attack', 'release']) {
    row.querySelector(`.routing-${param}`).addEventListener('change', (e) => {
      const value = parseFloat(e.target.value);
      if (Number.isFinite(value)) sendAction({ action: 'set_routing', ...selector(), param, value });
    });
  }
  row.querySelector('.routing-remove').addEventListener('click', () => {
    sendAction({ action: 'remove_routing', ...selector() });
  });
  bindRangeEditors(row);
  return row;
}

function syncModulation(m) {
  if (!m) return;

  if (canSync(bpmInput)) {
    bpmInput.value = m.bpm.toFixed(1);
  }

  // Beat light: bright on the beat, fading through it; strong on downbeats.
  const light = document.getElementById('beat-light');
  if (light && typeof m.beat === 'number') {
    const phase = m.beat - Math.floor(m.beat);
    const downbeat = Math.floor(m.beat) % 4 === 0;
    light.style.opacity = (1 - phase * 0.85).toFixed(2);
    light.classList.toggle('downbeat', downbeat);
  }

  (m.lfos || []).forEach((lfo, i) => {
    const row = lfoList.children[i];
    if (!row) return;
    const shapeSel = row.querySelector('.lfo-shape');
    const rateSel = row.querySelector('.lfo-rate');
    const seedInput = row.querySelector('.lfo-seed');
    if (canSync(shapeSel)) shapeSel.value = lfo.shape;
    if (canSync(rateSel)) rateSel.value = lfo.beats;
    if (canSync(seedInput)) seedInput.value = String(Number(lfo.seed || 0) >>> 0);
    seedInput.hidden = lfo.shape !== 'sample_hold';
    // Live meter: map [-1, 1] → [0%, 100%]
    const fill = row.querySelector('.lfo-meter-fill');
    fill.style.width = `${((lfo.value + 1) * 50).toFixed(1)}%`;
  });

  syncPad(m.pad);
  syncPadConfig(m.pad_config);
  syncGyroConfig(m.gyro_config);
  syncGyroStatus(m.gyro_status);

  // Gyro meters (values come from whichever device is streaming).
  if (m.gyro) {
    for (const [id, v] of [['gm-yaw', m.gyro[0]], ['gm-pitch', m.gyro[1]], ['gm-roll', m.gyro[2]]]) {
      const el = document.getElementById(id);
      if (el) el.style.width = `${(v * 100).toFixed(1)}%`;
    }
  }

  const routings = m.routings || [];
  const routingKey = JSON.stringify({
    layers: latestLayerIdentities,
    creative: creativeStructureKey,
    routes: routings.map((routing, index) => [
      routing.route_id || `legacy-index:${index}`,
      routing.target || '',
    ]),
  });
  if (routingList.children.length !== routings.length || routingList.dataset.routingKey !== routingKey) {
    routingList.innerHTML = '';
    routingList.dataset.routingKey = routingKey;
    routings.forEach((r, i) => routingList.appendChild(createRoutingRow(r, i)));
  } else {
    routings.forEach((r, i) => {
      const row = routingList.children[i];
      const sourceSel = row.querySelector('.routing-source');
      const targetSel = row.querySelector('.routing-target');
      const depthSlider = row.querySelector('.routing-depth');
      const depthVal = row.querySelector('.routing-depth-val');
      const curveSel = row.querySelector('.routing-curve');
      const curveAmount = row.querySelector('.routing-curve-amount');
      const attack = row.querySelector('.routing-attack');
      const release = row.querySelector('.routing-release');
      row.dataset.index = i;
      row.dataset.routeId = r.route_id || '';
      if (canSync(sourceSel)) sourceSel.value = r.source;
      if (canSync(targetSel)) targetSel.value = r.target;
      if (canSync(depthSlider)) {
        depthSlider.value = r.depth;
        depthVal.textContent = r.depth.toFixed(2);
      }
      if (canSync(curveSel)) curveSel.value = r.curve || 'linear';
      if (canSync(curveAmount)) curveAmount.value = r.curve_amount || 0;
      syncCurveAmountState(curveSel, curveAmount);
      if (canSync(attack)) attack.value = r.attack || 0;
      if (canSync(release)) release.value = r.release || 0;
      const activity = row.querySelector('.routing-activity');
      const fill = row.querySelector('.routing-activity-fill');
      const activityValue = row.querySelector('.routing-activity-value');
      const value = Math.min(1, Math.max(-1, Number(r.value) || 0));
      const bipolar = /^(lfo\d|gyro_|pad_|chaos|drift)/.test(r.source || '');
      const meterMin = bipolar ? -1 : 0;
      const bounded = Math.max(meterMin, value);
      const left = bipolar ? (bounded < 0 ? 50 + bounded * 50 : 50) : 0;
      const width = bipolar ? Math.abs(bounded) * 50 : bounded * 100;
      fill.style.left = `${left.toFixed(1)}%`;
      fill.style.width = `${width.toFixed(1)}%`;
      fill.classList.toggle('negative', bounded < 0);
      activity.classList.toggle('unipolar', !bipolar);
      activityValue.textContent = bounded.toFixed(2);
      activity.setAttribute('aria-valuemin', String(meterMin));
      activity.setAttribute('aria-valuenow', bounded.toFixed(3));
      activity.setAttribute('aria-valuetext', `${bipolar ? 'bipolar' : 'unipolar'} shaped and slewed value ${bounded.toFixed(2)}`);
      activity.setAttribute('aria-label', `${MOD_SOURCES.find(([source]) => source === r.source)?.[1] || r.source} shaped and slewed modulation value`);
    });
  }

  // B10 perform sources.
  syncPerformSources(m);
}

function syncPerformSources(m) {
  const envelopes = Array.isArray(m.envelopes) ? m.envelopes : [];
  document.querySelectorAll('.envelope-row').forEach((row, i) => {
    const env = envelopes[i];
    if (!env) return;
    const trigger = row.querySelector('.env-trigger');
    const mode = row.querySelector('.env-mode');
    if (trigger && canSync(trigger)) trigger.value = env.trigger || 'bend1';
    if (mode && canSync(mode)) mode.value = env.mode || 'once';
    const [attackRow, decayRow] = row.querySelectorAll('.param-row');
    for (const [paramRow, value] of [[attackRow, env.attack], [decayRow, env.decay]]) {
      if (!paramRow || value === undefined) continue;
      const slider = paramRow.querySelector('input[type="range"]');
      if (slider && canSync(slider)) {
        slider.value = value;
        paramRow.querySelector('.value').textContent = Number(value).toFixed(3);
      }
    }
    const fill = row.querySelector('.lfo-meter-fill');
    if (fill) fill.style.width = `${(Math.max(0, Math.min(1, env.level || 0)) * 100).toFixed(1)}%`;
  });
  const macros = Array.isArray(m.macros) ? m.macros : [];
  document.querySelectorAll('#macro-list .param-row').forEach((row, i) => {
    const value = macros[i];
    if (value === undefined) return;
    const slider = row.querySelector('input[type="range"]');
    if (slider && canSync(slider)) {
      slider.value = value;
      row.querySelector('.value').textContent = Number(value).toFixed(3);
    }
  });
  const bends = Array.isArray(m.bends) ? m.bends : [];
  document.querySelectorAll('.bend-pad').forEach((pad) => {
    const index = Number(pad.dataset.bend);
    // Never fight the finger that is on it: while this client holds the pad,
    // its own state is the truth.
    if (!bendHeldLocal[index]) {
      pad.setAttribute('aria-pressed', bends[index] ? 'true' : 'false');
    }
  });
  if (modSeedInput && canSync(modSeedInput)) {
    modSeedInput.value = String(m.generator_seed || 0);
  }
}

// --- Audio input ---

const audioEnabled = document.getElementById('audio-enabled');
const audioSourceKind = document.getElementById('audio-source-kind');
const audioClip = document.getElementById('audio-clip');
const audioClipRow = document.getElementById('audio-clip-row');
const audioImport = document.getElementById('audio-import');
const audioImportButton = document.getElementById('audio-import-button');
const audioImportStatus = document.getElementById('audio-import-status');
const audioGain = document.getElementById('audio-gain');
const audioGainVal = document.getElementById('audio-gain-val');
const audioStatus = document.getElementById('audio-status');
const audioBandCount = document.getElementById('audio-band-count');
const audioBandEdges = document.getElementById('audio-band-edges');
const audioBandCeiling = document.getElementById('audio-high-edge');
const audioExtraBandMeters = document.getElementById('audio-extra-band-meters');
const audioSpectrum = document.getElementById('audio-spectrum');
for (let i = 0; i < 32; i++) audioSpectrum.appendChild(document.createElement('span'));

function sendAudioBandEdges() {
  const count = Math.min(8, Math.max(3, parseInt(audioBandCount.value, 10) || 3));
  const edges = Array.from(audioBandEdges.querySelectorAll('input')).map((input) => parseFloat(input.value));
  const ceiling = parseFloat(audioBandCeiling.value);
  if (edges.length === count - 1 && edges.every(Number.isFinite) && Number.isFinite(ceiling)) {
    sendAction({ action: 'set_audio', param: 'band_edges', value: { count, edges, ceiling } });
  }
}

let renderedAudioBandCount = 0;
function rebuildAudioBandControls(count) {
  count = Math.min(8, Math.max(3, Number(count) || 3));
  if (count === renderedAudioBandCount) return;
  renderedAudioBandCount = count;
  audioBandEdges.replaceChildren();
  for (let index = 0; index < count - 1; index++) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    const legacyIds = ['audio-bass-edge', 'audio-mid-edge'];
    input.id = legacyIds[index] || `audio-band-edge-${index + 1}`;
    input.type = 'number';
    input.min = String(20 + index * 10);
    input.max = String(20000 - (count - 1 - index) * 10);
    input.step = '10';
    input.value = String(index === 0 ? 250 : index === 1 ? 2000 : 4000 + (index - 2) * 1000);
    input.addEventListener('input', sendAudioBandEdges);
    input.addEventListener('change', sendAudioBandEdges);
    label.htmlFor = input.id;
    label.textContent = `Band ${index + 1} → ${index + 2}`;
    audioBandEdges.append(label, input);
  }

  audioExtraBandMeters.replaceChildren();
  for (let index = 3; index < count; index++) {
    const row = document.createElement('div');
    row.className = 'audio-meter-row';
    row.innerHTML = `<label>Band ${index + 1}</label><div class="lfo-meter"><div class="lfo-meter-fill" data-audio-band="${index}"></div></div>`;
    audioExtraBandMeters.appendChild(row);
  }
}

rebuildAudioBandControls(3);
audioBandCount.addEventListener('change', () => {
  const count = Math.min(8, Math.max(3, parseInt(audioBandCount.value, 10) || 3));
  rebuildAudioBandControls(count);
  sendAction({ action: 'set_audio', param: 'band_count', value: count });
});
audioBandCeiling.addEventListener('input', sendAudioBandEdges);
audioBandCeiling.addEventListener('change', sendAudioBandEdges);

audioEnabled.addEventListener('change', () => {
  sendAction({ action: 'set_audio', param: 'enabled', value: audioEnabled.checked });
});

audioSourceKind.addEventListener('change', () => {
  const fileMode = audioSourceKind.value === 'file';
  audioClipRow.hidden = !fileMode;
  audioDevice.closest('.param-row').hidden = fileMode;
  sendAction({ action: 'set_audio', param: 'source_kind', value: audioSourceKind.value });
});

const AUDIO_IMPORT_ACTION = '__import_audio__';
const MAX_AUDIO_IMPORT_BYTES = 512 * 1024 * 1024;
const SUPPORTED_AUDIO_FILE = /\.(wav|mp3|flac|ogg|opus|m4a|aac)$/i;
let selectedAudioClip = '';
let audioImportBusy = false;

function openAudioImportPicker() {
  audioImport.click();
}

audioClip.addEventListener('change', () => {
  if (audioClip.value === AUDIO_IMPORT_ACTION) {
    // Restore the authoritative selection immediately. A canceled native
    // chooser therefore cannot clear or replace the running clip.
    audioClip.value = selectedAudioClip || AUDIO_IMPORT_ACTION;
    openAudioImportPicker();
    return;
  }
  sendAction({ action: 'set_audio', param: 'clip', value: audioClip.value });
});
audioImportButton.addEventListener('click', openAudioImportPicker);

audioImport.addEventListener('change', async () => {
  if (audioImportBusy) return;
  audioImportBusy = true;
  audioImportButton.disabled = true;
  audioClip.disabled = true;
  audioImport.disabled = true;
  try {
  const files = [...audioImport.files];
  audioImport.value = '';
  let lastImported = '';
  for (const file of files) {
    if (!SUPPORTED_AUDIO_FILE.test(file.name)) {
      audioImportStatus.textContent = `${file.name}: choose WAV, MP3, FLAC, OGG, Opus, M4A, or AAC`;
      audioImportStatus.classList.add('error');
      continue;
    }
    if (file.size === 0) {
      audioImportStatus.textContent = `${file.name}: the file is empty`;
      audioImportStatus.classList.add('error');
      continue;
    }
    if (file.size > MAX_AUDIO_IMPORT_BYTES) {
      audioImportStatus.textContent = `${file.name}: exceeds the 512 MiB audio import limit`;
      audioImportStatus.classList.add('error');
      continue;
    }
    const result = await uploadClip(file, audioImportStatus);
    if (result.ok) lastImported = result.filename;
  }
  if (lastImported) {
    selectedAudioClip = lastImported;
    sendAction({ action: 'set_audio', param: 'source_kind', value: 'file' });
    sendAction({ action: 'set_audio', param: 'clip', value: lastImported });
    audioImportStatus.textContent = `${lastImported} added — loading deterministic analysis…`;
    audioImportStatus.classList.remove('error');
  }
  audioClip.value = selectedAudioClip || AUDIO_IMPORT_ACTION;
  } finally {
    audioImportBusy = false;
    audioImportButton.disabled = false;
    audioClip.disabled = false;
    audioImport.disabled = false;
  }
});

audioGain.addEventListener('input', () => {
  const v = parseFloat(audioGain.value);
  audioGainVal.textContent = v.toFixed(2);
  sendAction({ action: 'set_audio', param: 'gain', value: v });
});
resetRangeOnDoubleActivation(audioGain, 1);

const audioDevice = document.getElementById('audio-device');
audioDevice.addEventListener('change', () => {
  sendAction({ action: 'set_audio', param: 'device', value: audioDevice.value });
});

let knownDevices = '';
function syncAudioDevices(devices, playbackDevices, selected) {
  const key = `${(devices || []).join('|')}::${(playbackDevices || []).join('|')}`;
  if (key !== knownDevices) {
    knownDevices = key;
    audioDevice.replaceChildren(new Option('Default input', ''));
    for (const device of devices || []) audioDevice.add(new Option(device, device));
    audioDevice.add(new Option('[System playback] Default output', 'system-playback:default'));
    for (const device of playbackDevices || []) {
      audioDevice.add(new Option(`[System playback] ${device}`, `system-playback:${device}`));
    }
  }
  if (canSync(audioDevice)) audioDevice.value = selected || '';
}

let knownAudioClips = '';
function syncAudioClips(files, selected) {
  const key = (files || []).join('|');
  if (key !== knownAudioClips) {
    knownAudioClips = key;
    audioClip.replaceChildren(new Option('Choose imported audio…', AUDIO_IMPORT_ACTION));
    for (const file of files || []) audioClip.add(new Option(file, file));
  }
  selectedAudioClip = selected || '';
  if (canSync(audioClip)) audioClip.value = selectedAudioClip || AUDIO_IMPORT_ACTION;
}

function syncAudio(a) {
  if (!a) return;
  if (canSync(audioEnabled)) audioEnabled.checked = a.enabled;
  if (canSync(audioSourceKind)) audioSourceKind.value = a.source_kind || 'live';
  const fileMode = (a.source_kind || 'live') === 'file';
  audioClipRow.hidden = !fileMode;
  audioDevice.closest('.param-row').hidden = fileMode;
  syncAudioDevices(a.devices, a.system_playback_devices, a.selected);
  syncAudioClips(a.clip_files, a.clip_path);
  if (canSync(audioGain)) {
    audioGain.value = a.gain;
    audioGainVal.textContent = a.gain.toFixed(2);
  }
  const bandCount = Math.min(8, Math.max(3, Number(a.band_count) || 3));
  if (canSync(audioBandCount)) audioBandCount.value = String(bandCount);
  rebuildAudioBandControls(bandCount);
  if (Array.isArray(a.band_edges) && a.band_edges.length >= bandCount - 1) {
    Array.from(audioBandEdges.querySelectorAll('input')).forEach((input, index) => {
      if (canSync(input)) input.value = Math.round(a.band_edges[index]);
    });
  }
  if (canSync(audioBandCeiling)) audioBandCeiling.value = Math.round(Number(a.band_ceiling_hz) || 8000);
  if (Array.isArray(a.spectrum)) {
    Array.from(audioSpectrum.children).forEach((bar, index) => {
      const value = Math.min(1, Math.max(0, Number(a.spectrum[index]) || 0));
      bar.style.transform = `scaleY(${Math.max(0.015, value).toFixed(3)})`;
    });
  }
  for (const [id, v] of [
    ['am-level', a.level],
    ['am-bass', a.bass],
    ['am-mid', a.mid],
    ['am-high', a.high],
    ['am-onset', a.onset],
    ['am-bright', a.bright || 0],
    ['am-noise', a.noise || 0],
  ]) {
    document.getElementById(id).style.width = `${(v * 100).toFixed(1)}%`;
  }
  const bands = Array.isArray(a.bands) ? a.bands : [a.bass, a.mid, a.high];
  audioExtraBandMeters.querySelectorAll('[data-audio-band]').forEach((meter) => {
    const value = Math.min(1, Math.max(0, Number(bands[Number(meter.dataset.audioBand)]) || 0));
    meter.style.width = `${(value * 100).toFixed(1)}%`;
  });
  if (a.error) {
    audioStatus.textContent = a.error;
    audioStatus.className = 'audio-status error';
    if (fileMode && (a.clip_path || audioImportStatus.textContent)) {
      audioImportStatus.textContent = a.clip_path ? `${a.clip_path}: ${a.error}` : a.error;
      audioImportStatus.classList.add('error');
    }
  } else if (fileMode && a.clip_loading) {
    audioStatus.textContent = `loading ${a.clip_path || 'audio clip'}…`;
    audioStatus.className = 'audio-status';
  } else if (fileMode && a.clip_path) {
    const duration = Number(a.clip_duration_secs) > 0 ? ` · ${Number(a.clip_duration_secs).toFixed(2)} s loop` : '';
    audioStatus.textContent = `${a.clip_path}${duration} · deterministic program-time analysis`;
    audioStatus.className = 'audio-status';
    if (audioImportStatus.textContent.includes(a.clip_path)) {
      audioImportStatus.textContent = '';
      audioImportStatus.classList.remove('error');
    }
  } else if (a.enabled && (a.active_device || a.device)) {
    const activeDevice = a.active_device || a.device;
    audioStatus.textContent = a.using_fallback
      ? `using fallback: ${activeDevice} (requested ${a.selected || 'default'})`
      : activeDevice;
    audioStatus.className = 'audio-status';
  } else {
    audioStatus.textContent = '';
  }
}

// --- MIDI input ---

const midiEnabled = document.getElementById('midi-enabled');
const midiSlots = document.getElementById('midi-slots');
const midiStatus = document.getElementById('midi-status');
const controllerProfileImport = document.getElementById('controller-profile-import');
const controllerProfileExport = document.getElementById('controller-profile-export');
const controllerProfileFile = document.getElementById('controller-profile-file');
const CONTROLLER_PROFILE_MAX_BYTES = 256 * 1024;

midiEnabled.addEventListener('change', () => {
  sendAction({ action: 'set_midi', param: 'enabled', value: midiEnabled.checked });
});

const midiClockSync = document.getElementById('midi-clock-sync');
midiClockSync.addEventListener('change', () => {
  sendAction({ action: 'set_midi', param: 'clock_sync', value: midiClockSync.checked });
});

function setControllerProfileTransferStatus(message, error = false) {
  const status = document.getElementById('controller-runtime-status');
  if (!status) return;
  status.textContent = String(message || '').slice(0, 1024);
  status.className = error ? 'audio-status error' : 'audio-status';
}

async function postControllerProfileAction(action) {
  const body = JSON.stringify(action);
  if (new TextEncoder().encode(body).byteLength > CONTROLLER_PROFILE_MAX_BYTES + 1024) {
    throw new Error('controller profile request exceeds the 257 KiB action cap');
  }
  const response = await fetch('/controller-profile', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body,
  });
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 1024);
    throw new Error(detail || `controller profile request failed (${response.status})`);
  }
  return response;
}

controllerProfileImport?.addEventListener('click', () => controllerProfileFile?.click());
controllerProfileFile?.addEventListener('change', async () => {
  const file = controllerProfileFile.files?.[0];
  controllerProfileFile.value = '';
  if (!file) return;
  if (file.size > CONTROLLER_PROFILE_MAX_BYTES) {
    setControllerProfileTransferStatus('Controller profile exceeds the 256 KiB document cap.', true);
    return;
  }
  try {
    const documentValue = JSON.parse(await file.text());
    await postControllerProfileAction({ action: 'import', document: documentValue });
    setControllerProfileTransferStatus('Controller profile import queued for atomic validation.');
  } catch (error) {
    setControllerProfileTransferStatus(`Controller profile import rejected: ${error.message || error}`, true);
  }
});

controllerProfileExport?.addEventListener('click', async () => {
  try {
    const response = await postControllerProfileAction({ action: 'export' });
    const blob = await response.blob();
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = href;
    anchor.download = 'controller_profile.json';
    anchor.click();
    URL.revokeObjectURL(href);
    setControllerProfileTransferStatus('Controller profile exported without browser path authority.');
  } catch (error) {
    setControllerProfileTransferStatus(`Controller profile export failed: ${error.message || error}`, true);
  }
});

// Build the 4 static MIDI slot rows once.
const MIDI_SLOT_NAMES = ['A', 'B', 'C', 'D'];
for (let i = 0; i < 4; i++) {
  const row = document.createElement('div');
  row.className = 'midi-slot-row';
  row.dataset.slot = i;
  row.innerHTML = `
    <span class="lfo-name">${MIDI_SLOT_NAMES[i]}</span>
    <span class="midi-cc-label">CC</span>
    <input type="number" class="midi-cc" min="0" max="127" step="1" value="${i + 1}" aria-label="MIDI ${MIDI_SLOT_NAMES[i]} Control Change number">
    <button type="button" class="midi-learn" title="Bind the next incoming Control Change; keys and notes are ignored" aria-pressed="false">Learn CC</button>
    <div class="lfo-meter"><div class="lfo-meter-fill"></div></div>
  `;
  row.querySelector('.midi-cc').addEventListener('change', (e) => {
    const v = parseInt(e.target.value);
    if (!isNaN(v)) sendAction({ action: 'set_midi', param: `cc${i}`, value: v });
  });
  row.querySelector('.midi-learn').addEventListener('click', (e) => {
    // Clicking an already-armed slot disarms it (server treats repeat as cancel).
    const armed = e.target.classList.contains('learning');
    sendAction({ action: 'set_midi', param: 'learn', value: armed ? null : i });
  });
  midiSlots.appendChild(row);
}

function syncMidi(m) {
  if (!m) return;
  if (canSync(midiEnabled)) midiEnabled.checked = m.enabled;
  if (canSync(midiClockSync)) midiClockSync.checked = !!m.clock_sync;
  const clockInd = document.getElementById('midi-clock-indicator');
  if (m.clock_active) {
    const bpm = Number.isFinite(m.clock_bpm) ? `${m.clock_bpm.toFixed(1)} BPM` : 'active';
    clockInd.textContent = `24 PPQN | ${bpm}`;
    clockInd.title = 'Following incoming MIDI Timing Clock';
    clockInd.className = 'clock-indicator active';
  } else if (m.clock_sync) {
    clockInd.textContent = m.enabled ? 'waiting for 24 PPQN' : 'enable MIDI';
    clockInd.title = m.enabled
      ? 'Waiting for incoming 24-PPQN MIDI Timing Clock'
      : 'MIDI input must be enabled before clock can be received';
    clockInd.className = 'clock-indicator';
  } else {
    clockInd.textContent = 'off';
    clockInd.title = 'External MIDI Timing Clock sync is off';
    clockInd.className = 'clock-indicator';
  }

  (m.slots || []).forEach((slot, i) => {
    const row = midiSlots.children[i];
    if (!row) return;
    const ccInput = row.querySelector('.midi-cc');
    if (canSync(ccInput)) ccInput.value = slot.cc;
    row.querySelector('.lfo-meter-fill').style.width = `${(slot.value * 100).toFixed(1)}%`;
    const learnBtn = row.querySelector('.midi-learn');
    const isLearning = m.learning === i;
    learnBtn.classList.toggle('learning', isLearning);
    learnBtn.textContent = isLearning ? 'Cancel' : 'Learn CC';
    learnBtn.setAttribute('aria-pressed', String(isLearning));
    learnBtn.title = isLearning
      ? `MIDI ${MIDI_SLOT_NAMES[i]} is waiting for a Control Change; click to cancel`
      : 'Bind the next incoming Control Change; keys and notes are ignored';
  });

  if (m.error) {
    midiStatus.textContent = m.error;
    midiStatus.className = 'audio-status error';
  } else {
    const status = [];
    if (m.enabled && m.port) status.push(`Input: ${m.port}`);
    else if (m.enabled) status.push('MIDI input starting');
    else status.push('MIDI input off');
    if (Number.isInteger(m.learning) && m.learning >= 0 && m.learning < MIDI_SLOT_NAMES.length) {
      status.push(m.enabled
        ? `Learn ${MIDI_SLOT_NAMES[m.learning]}: waiting for CC (keys ignored)`
        : `Learn ${MIDI_SLOT_NAMES[m.learning]} armed; enable MIDI to receive CC`);
    }
    midiStatus.textContent = status.join(' | ');
    midiStatus.className = 'audio-status';
  }
}

function runtimeCount(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 0 ? Math.trunc(numeric) : 0;
}

function runtimePhaseLabel(value, fallback = 'Unavailable') {
  const phase = String(value || '').trim();
  if (!phase) return fallback;
  return phase.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function syncControllerRuntime(snapshot = {}) {
  const runtime = snapshot?.midi || {};
  const counters = runtime.counters || {};
  const profile = document.getElementById('controller-runtime-profile');
  const input = document.getElementById('midi-runtime-input');
  const output = document.getElementById('midi-runtime-output');
  const status = document.getElementById('controller-runtime-status');
  const counterLine = document.getElementById('midi-runtime-counters');
  if (!profile || !input || !output || !status || !counterLine) return;

  const revision = runtimeCount(snapshot.profile_revision);
  profile.textContent = `${String(snapshot.name || 'Unspecified profile')} · revision ${revision || 'legacy'}`;
  input.textContent = runtime.input_port
    ? `${runtime.input_port} · ${runtimeCount(runtime.available_inputs?.length)} available`
    : `None · ${runtimeCount(runtime.available_inputs?.length)} available`;
  output.textContent = runtime.output_port
    ? `${runtime.output_port} · ${runtimeCount(runtime.available_outputs?.length)} available`
    : `None · ${runtimeCount(runtime.available_outputs?.length)} available`;
  const messages = [
    runtimePhaseLabel(runtime.phase, 'Legacy runtime'),
    String(snapshot.status || ''),
    String(runtime.error || ''),
  ].filter(Boolean);
  status.textContent = messages.join(' · ');
  status.className = /error|reject|unavailable|failed/i.test(status.textContent)
    ? 'audio-status error'
    : 'audio-status';
  counterLine.textContent = [
    `raw ${runtimeCount(counters.raw_received)}`,
    `decoded ${runtimeCount(counters.decoded_events)}`,
    `malformed ${runtimeCount(counters.malformed)}`,
    `unmapped ${runtimeCount(counters.channel_or_unmapped)}`,
    `input drops ${runtimeCount(counters.input_queue_dropped)}`,
    `event drops ${runtimeCount(counters.event_queue_dropped)}`,
    `loop suppressions ${runtimeCount(counters.loop_suppressed)}`,
    `feedback queued/coalesced/sent ${runtimeCount(counters.feedback_queued)}/${runtimeCount(counters.feedback_coalesced)}/${runtimeCount(counters.feedback_sent)}`,
    `feedback drops/rate limits ${runtimeCount(counters.feedback_dropped)}/${runtimeCount(counters.feedback_rate_limited)}`,
    `scans/reconnects/disconnects ${runtimeCount(counters.scans)}/${runtimeCount(counters.reconnects)}/${runtimeCount(counters.disconnects)}`,
  ].join(' · ');
}

function syncOscRuntime(runtime = {}) {
  const counters = runtime?.counters || {};
  const bind = document.getElementById('osc-runtime-bind');
  const bound = document.getElementById('osc-runtime-bound');
  const phase = document.getElementById('osc-runtime-phase');
  const warning = document.getElementById('osc-runtime-lan-warning');
  const error = document.getElementById('osc-runtime-error');
  const counterLine = document.getElementById('osc-runtime-counters');
  if (!bind || !bound || !phase || !warning || !error || !counterLine) return;

  bind.textContent = String(runtime.bind_address || 'Not configured');
  bound.textContent = String(runtime.bound_address || 'Not listening');
  phase.textContent = `${runtimePhaseLabel(runtime.phase, 'Disabled')}${runtime.running ? ' · running' : ''}`;
  warning.textContent = runtime.lan_warning
    ? 'LAN exposure is enabled: this typed OSC listener is reachable beyond loopback. Verify the interface and firewall before performance.'
    : 'Loopback-only unless a validated LAN configuration explicitly says otherwise.';
  warning.classList.toggle('active', !!runtime.lan_warning);
  error.textContent = String(runtime.error || '');
  error.className = error.textContent ? 'audio-status error' : 'audio-status';
  counterLine.textContent = [
    `datagrams/messages ${runtimeCount(counters.datagrams_received)}/${runtimeCount(counters.messages_received)}`,
    `malformed ${runtimeCount(counters.malformed)}`,
    `rate/queue drops ${runtimeCount(counters.rate_dropped)}/${runtimeCount(counters.queue_dropped)}`,
    `loop suppressions ${runtimeCount(counters.loop_suppressed)}`,
    `feedback queued/coalesced/sent ${runtimeCount(counters.feedback_queued)}/${runtimeCount(counters.feedback_coalesced)}/${runtimeCount(counters.feedback_sent)}`,
    `feedback drops/rate limits/send errors ${runtimeCount(counters.feedback_dropped)}/${runtimeCount(counters.feedback_rate_limited)}/${runtimeCount(counters.feedback_send_errors)}`,
  ].join(' · ');
}

// --- XY performance pad ---

const xyPad = document.getElementById('xy-pad');
const xyDot = document.getElementById('xy-pad-dot');

function releasePadPointer() {
  if (padPointerId === null) return;
  if (!sendAction({ action: 'pad', x: padLastPosition[0], y: padLastPosition[1], active: false })) {
    padNeedsReconcile = true;
  }
  padPointerId = null;
}

function padPosition(x, y) {
  xyDot.style.left = `${(x * 100).toFixed(1)}%`;
  xyDot.style.top = `${((1 - y) * 100).toFixed(1)}%`;
  xyPad.setAttribute('aria-valuetext', `X ${Number(x).toFixed(2)}, Y ${Number(y).toFixed(2)}`);
}

xyPad.addEventListener('keydown', (event) => {
  const delta = event.shiftKey ? 0.1 : 0.02;
  let [x, y] = padLastPosition;
  if (event.key === 'ArrowLeft') x -= delta;
  else if (event.key === 'ArrowRight') x += delta;
  else if (event.key === 'ArrowDown') y -= delta;
  else if (event.key === 'ArrowUp') y += delta;
  else if (event.key === 'Home') [x, y] = [0.5, 0.5];
  else return;
  event.preventDefault();
  x = Math.min(1, Math.max(0, x));
  y = Math.min(1, Math.max(0, y));
  padLastPosition = [x, y];
  padPosition(x, y);
  sendAction({ action: 'pad', x, y, active: false });
});

function padSend(e, force = false, active = true) {
  const rect = xyPad.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  // Screen y grows downward; the pad's Y axis grows upward, like a fader.
  const y = Math.min(1, Math.max(0, 1 - (e.clientY - rect.top) / rect.height));
  padLastPosition = [x, y];
  padPosition(x, y);
  const now = performance.now();
  if (force || now - padLastSend >= 33) { // ~30Hz, exact on press/release
    padLastSend = now;
    sendAction({ action: 'pad', x, y, active });
  }
  return [x, y];
}

xyPad.addEventListener('pointerdown', (e) => {
  if (padPointerId !== null) return;
  padPointerId = e.pointerId;
  xyPad.setPointerCapture(e.pointerId);
  padSend(e, true, true);
  e.preventDefault();
});
xyPad.addEventListener('pointermove', (e) => {
  if (e.pointerId === padPointerId) padSend(e);
});
xyPad.addEventListener('pointerup', (e) => {
  if (e.pointerId !== padPointerId) return;
  // Final position lands exactly where the owning finger left it, and the
  // engine learns that release/spring-return may begin.
  padSend(e, true, false);
  if (!ws || ws.readyState !== WebSocket.OPEN) padNeedsReconcile = true;
  padPointerId = null;
});
xyPad.addEventListener('pointercancel', (e) => {
  if (e.pointerId !== padPointerId) return;
  releasePadPointer();
});
xyPad.addEventListener('lostpointercapture', (e) => {
  if (e.pointerId !== padPointerId) return;
  releasePadPointer();
});
window.addEventListener('blur', releasePadPointer);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) releasePadPointer();
});

function syncPad(pad) {
  if (!pad || padPointerId !== null) return; // never fight the finger that's on it
  padLastPosition = [pad[0], pad[1]];
  padPosition(pad[0], pad[1]);
}

document.querySelectorAll('[data-pad-axis][data-pad-param]').forEach((el) => {
  const send = () => {
    let value = el.type === 'number' ? parseInt(el.value) :
      el.type === 'range' ? parseFloat(el.value) : el.value;
    if (Number.isFinite(value) || typeof value === 'string') {
      sendAction({ action: 'set_pad_config', axis: el.dataset.padAxis, param: el.dataset.padParam, value });
    }
  };
  el.addEventListener(el.type === 'range' ? 'input' : 'change', send);
  if (el.dataset.padParam === 'curve') {
    el.addEventListener('change', () => {
      const amount = document.querySelector(`[data-pad-axis="${el.dataset.padAxis}"][data-pad-param="curve_amount"]`);
      syncCurveAmountState(el, amount);
    });
  }
  if (el.type === 'range') resetRangeOnDoubleActivation(el, 0);
});
for (const axis of ['x', 'y']) {
  syncCurveAmountState(
    document.querySelector(`[data-pad-axis="${axis}"][data-pad-param="curve"]`),
    document.querySelector(`[data-pad-axis="${axis}"][data-pad-param="curve_amount"]`),
  );
}

const padSpringEnabled = document.getElementById('pad-spring-enabled');
const padSpringRate = document.getElementById('pad-spring-rate');
const padSpringRateVal = document.getElementById('pad-spring-rate-val');
padSpringEnabled.addEventListener('change', () => {
  sendAction({ action: 'set_pad_config', axis: 'both', param: 'spring_enabled', value: padSpringEnabled.checked });
});
padSpringRate.addEventListener('input', () => {
  const value = parseFloat(padSpringRate.value);
  padSpringRateVal.textContent = value.toFixed(1);
  sendAction({ action: 'set_pad_config', axis: 'both', param: 'spring_rate', value });
});
resetRangeOnDoubleActivation(padSpringRate, 4);

function syncPadConfig(config) {
  if (!config) return;
  for (const axis of ['x', 'y']) {
    const cfg = config[axis];
    if (!cfg) continue;
    for (const param of ['curve', 'curve_amount', 'quantize']) {
      const el = document.querySelector(`[data-pad-axis="${axis}"][data-pad-param="${param}"]`);
      if (canSync(el)) el.value = cfg[param];
    }
    syncCurveAmountState(
      document.querySelector(`[data-pad-axis="${axis}"][data-pad-param="curve"]`),
      document.querySelector(`[data-pad-axis="${axis}"][data-pad-param="curve_amount"]`),
    );
  }
  if (canSync(padSpringEnabled)) padSpringEnabled.checked = !!config.spring_enabled;
  if (canSync(padSpringRate)) {
    padSpringRate.value = config.spring_rate;
    padSpringRateVal.textContent = Number(config.spring_rate).toFixed(1);
  }
}

// --- Gyroscope streaming (runs on the device that enables it) ---

const gyroEnabled = document.getElementById('gyro-enabled');
const gyroStatus = document.getElementById('gyro-status');
let gyroLastSend = 0;
let gyroSeenEvent = false;
let gyroLocalError = '';

function setGyroLocalError(message) {
  gyroLocalError = message;
  gyroStatus.textContent = message;
  gyroStatus.className = 'audio-status error';
}

function syncGyroStatus(status) {
  if (!status) return; // Remain compatible with snapshots from older servers.
  if (gyroLocalError) {
    setGyroLocalError(gyroLocalError);
    return;
  }

  const streamers = Math.max(0, Number(status.streamers) || 0);
  const age = Number.isFinite(status.sample_age_ms)
    ? ` · ${Math.max(0, Math.round(status.sample_age_ms))} ms`
    : '';
  if (status.active) {
    gyroStatus.textContent = `live · ${streamers} streamer${streamers === 1 ? '' : 's'}${age}`;
    gyroStatus.className = 'audio-status';
  } else if (status.stale) {
    gyroStatus.textContent = streamers > 0
      ? `sensor data stale · output centered${age}`
      : `stream stopped · output centered${age}`;
    gyroStatus.className = 'audio-status error';
  } else {
    gyroStatus.textContent = 'no active phone stream';
    gyroStatus.className = 'audio-status';
  }
}

function reconcileGyroStreamConnection() {
  if (!gyroEnabled?.checked) return;
  gyroLocalError = '';
  if (!sendAction({ action: 'gyro_stream', enabled: true })) {
    setGyroLocalError('control connection offline');
  }
}

function showGyroDisconnected() {
  if (!document.getElementById('gyro-enabled')?.checked) return;
  setGyroLocalError('control connection offline · output will center');
}

function gyroHandler(e) {
  if (![e.alpha, e.beta, e.gamma].some(Number.isFinite)) return;
  gyroSeenEvent = true;
  gyroLocalError = '';
  const now = performance.now();
  if (now - gyroLastSend < 33) return; // ~30Hz
  gyroLastSend = now;
  const screenAngleRaw = screen.orientation?.angle ?? window.orientation ?? 0;
  const screenAngle = screenAngleRaw > 180 ? screenAngleRaw - 360 : screenAngleRaw;
  const radians = screenAngle * Math.PI / 180;
  const alpha = Number.isFinite(e.alpha) ? e.alpha : 0;
  const beta = Number.isFinite(e.beta) ? e.beta : 0;
  const gamma = Number.isFinite(e.gamma) ? e.gamma : 0;
  // Express pitch/roll in the visible screen's axes, not fixed device axes.
  const pitch = beta * Math.cos(radians) + gamma * Math.sin(radians);
  const roll = gamma * Math.cos(radians) - beta * Math.sin(radians);
  if (!sendAction({
    action: 'gyro',
    alpha: alpha + screenAngle,
    beta: pitch,
    gamma: roll,
  })) {
    setGyroLocalError('control connection offline · output will center');
  }
}

gyroEnabled.addEventListener('change', async () => {
  if (gyroEnabled.checked) {
    gyroLocalError = '';
    if (typeof DeviceOrientationEvent === 'undefined') {
      setGyroLocalError('no orientation sensor in this browser');
      gyroEnabled.checked = false;
      return;
    }
    // iOS requires an explicit permission request from a user gesture.
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
      try {
        const perm = await DeviceOrientationEvent.requestPermission();
        if (perm !== 'granted') {
          setGyroLocalError('permission denied');
          gyroEnabled.checked = false;
          return;
        }
      } catch (err) {
        setGyroLocalError('sensor needs HTTPS on iOS');
        gyroEnabled.checked = false;
        return;
      }
    }
    gyroSeenEvent = false;
    window.addEventListener('deviceorientation', gyroHandler);
    if (!sendAction({ action: 'gyro_stream', enabled: true })) {
      window.removeEventListener('deviceorientation', gyroHandler);
      gyroEnabled.checked = false;
      setGyroLocalError('control connection offline');
      return;
    }
    gyroStatus.textContent = 'waiting for fresh sensor data…';
    gyroStatus.className = 'audio-status';
    setTimeout(() => {
      if (gyroEnabled.checked && !gyroSeenEvent) {
        setGyroLocalError('no sensor data (desktop browser?)');
      }
    }, 2000);
  } else {
    window.removeEventListener('deviceorientation', gyroHandler);
    sendAction({ action: 'gyro_stream', enabled: false });
    gyroLocalError = '';
    gyroStatus.textContent = 'stopping stream…';
    gyroStatus.className = 'audio-status';
  }
});

document.getElementById('gyro-calibrate').addEventListener('click', () => {
  sendAction({ action: 'gyro_calibrate' });
});

document.querySelectorAll('[data-gyro-axis][data-gyro-param]').forEach((el) => {
  const send = () => {
    const value = el.type === 'checkbox' ? el.checked : parseFloat(el.value);
    if (typeof value === 'boolean' || Number.isFinite(value)) {
      sendAction({ action: 'set_gyro_config', axis: el.dataset.gyroAxis, param: el.dataset.gyroParam, value });
    }
  };
  el.addEventListener(el.type === 'range' ? 'input' : 'change', send);
  if (el.type === 'range') {
    const resetValue = el.dataset.gyroParam === 'range'
      ? (el.dataset.gyroAxis === 'yaw' ? 180 : 90)
      : 0;
    resetRangeOnDoubleActivation(el, resetValue);
  }
});

function syncGyroConfig(config) {
  if (!config) return;
  for (const axis of ['yaw', 'pitch', 'roll']) {
    const cfg = config[axis];
    if (!cfg) continue;
    for (const param of ['range', 'expo', 'invert']) {
      const el = document.querySelector(`[data-gyro-axis="${axis}"][data-gyro-param="${param}"]`);
      if (!canSync(el)) continue;
      if (el.type === 'checkbox') el.checked = !!cfg[param];
      else el.value = cfg[param];
    }
  }
}

// --- Spout output ---

const spoutEnabled = document.getElementById('spout-enabled');
const spoutResolution = document.getElementById('spout-resolution');
const spoutStatus = document.getElementById('spout-status');

spoutEnabled.addEventListener('change', () => {
  sendAction({ action: 'set_spout', enabled: spoutEnabled.checked });
});

spoutResolution?.addEventListener('change', () => {
  const resolution = spoutResolution.value === 'native' ? 'native' : '1080p';
  sendAction({ action: 'set_spout_resolution', resolution });
});

function syncSpout(s) {
  if (!s) return;
  if (canSync(spoutEnabled)) spoutEnabled.checked = s.enabled;
  if (spoutResolution && canSync(spoutResolution)) {
    spoutResolution.value = s.resolution === 'native' ? 'native' : '1080p';
  }
  const ind = document.getElementById('spout-indicator');
  if (s.active) {
    ind.textContent = '◉ sending';
    ind.className = 'clock-indicator active';
  } else if (s.enabled) {
    ind.textContent = 'starting…';
    ind.className = 'clock-indicator';
  } else {
    ind.textContent = '';
  }
  if (s.error) {
    spoutStatus.textContent = s.error;
    spoutStatus.className = 'audio-status error';
  } else if (s.active) {
    const dimensions = Number(s.width) > 0 && Number(s.height) > 0
      ? ` · ${s.width}×${s.height}`
      : '';
    spoutStatus.textContent = `sender: collide-o-scope${dimensions}`;
    spoutStatus.className = 'audio-status';
  } else {
    spoutStatus.textContent = '';
  }
}

// --- Patch morph crossfader ---

const morphT = document.getElementById('morph-t');
const morphStatus = document.getElementById('morph-status');
const morphLaw = document.getElementById('morph-law');
const morphDuration = document.getElementById('morph-duration');

document.getElementById('morph-set-a').addEventListener('click', () => {
  if (!sendAction({ action: 'morph_capture', slot: 'a', stack_revision: layerStackRevision, composition_revision: compositionRevision })) {
    morphStatus.textContent = 'Control connection is offline; A was not captured.';
  }
});
document.getElementById('morph-set-b').addEventListener('click', () => {
  if (!sendAction({ action: 'morph_capture', slot: 'b', stack_revision: layerStackRevision, composition_revision: compositionRevision })) {
    morphStatus.textContent = 'Control connection is offline; B was not captured.';
  }
});
document.getElementById('morph-clear').addEventListener('click', () => {
  if (!sendAction({ action: 'morph_clear' })) {
    morphStatus.textContent = 'Control connection is offline; slots were not cleared.';
  }
});
morphT.addEventListener('input', () => {
  sendAction({ action: 'set_morph', value: parseFloat(morphT.value) });
});
morphLaw.addEventListener('change', () => {
  sendAction({ action: 'set_morph_law', law: morphLaw.value });
});
for (const [id, target] of [['morph-glide-a', 0], ['morph-glide-b', 1]]) {
  document.getElementById(id).addEventListener('click', () => {
    const duration = Math.min(64, Math.max(0.25, parseFloat(morphDuration.value) || 4));
    morphDuration.value = duration;
    if (!sendAction({ action: 'morph_glide', target, duration_beats: duration })) {
      morphStatus.textContent = 'Control connection is offline; glide was not started.';
    }
  });
}
resetRangeOnDoubleActivation(morphT, 0);

function syncMorph(m) {
  if (!m) return;
  if (canSync(morphT)) morphT.value = m.t;
  if (canSync(morphLaw)) morphLaw.value = m.blend_law || 'linear';
  const setA = document.getElementById('morph-set-a');
  const setB = document.getElementById('morph-set-b');
  setA.classList.toggle('set', m.has_a);
  setB.classList.toggle('set', m.has_b);
  setA.setAttribute('aria-pressed', String(!!m.has_a));
  setB.setAttribute('aria-pressed', String(!!m.has_b));
  document.getElementById('morph-label-a').classList.toggle('active', m.active && m.t < 0.5);
  document.getElementById('morph-label-b').classList.toggle('active', m.active && m.t >= 0.5);
  if (m.gliding) {
    morphStatus.textContent = `gliding to ${m.glide_target >= 0.5 ? 'B' : 'A'} — ${Number(m.glide_duration_beats).toFixed(2)} beats remaining`;
  } else if (m.active) {
    morphStatus.textContent = 'morphing — editing a captured control disengages A/B';
  } else if (m.has_a || m.has_b) {
    morphStatus.textContent = m.has_a ? 'A set — capture B to engage' : 'B set — capture A to engage';
  } else {
    morphStatus.textContent = 'capture two states, then crossfade';
  }
}

// --- Fullscreen output window ---

const outputWindow = document.getElementById('output-window');
const outputDisplay = document.getElementById('output-display');

outputDisplay?.addEventListener('change', () => {
  const displayId = outputDisplay.value;
  if (!sendAction({ action: 'set_output_display', display_id: displayId, inventory_generation: outputAuthoritativeGeneration })) {
    outputDisplay.value = outputAuthoritativeDisplay;
    renderOutputWindow(outputAuthoritativeOpen, false, 'Control connection is offline.');
  }
});

document.getElementById('output-display-rescan')?.addEventListener('click', () => {
  sendAction({ action: 'rescan_output_displays' });
});

outputWindow.addEventListener('change', () => {
  if (outputPendingOpen !== null) return;
  const enabled = outputWindow.checked;
  if (sendAction({ action: 'set_output_window', enabled })) {
    const requestSequence = ++outputRequestSequence;
    outputPendingOpen = enabled;
    renderOutputWindow(enabled, true, '');
    window.setTimeout(() => {
      if (outputPendingOpen !== null && outputRequestSequence === requestSequence) {
        outputPendingOpen = null;
        renderOutputWindow(outputAuthoritativeOpen, false, 'Output request timed out; try again.');
      }
    }, 2000);
  } else {
    outputWindow.checked = outputAuthoritativeOpen;
    renderOutputWindow(outputAuthoritativeOpen, false, 'Control connection is offline.');
  }
});

function renderOutputWindow(open, pending, error = '') {
  if (canSync(outputWindow) || pending) outputWindow.checked = !!open;
  outputWindow.disabled = !!pending;
  if (outputDisplay) outputDisplay.disabled = !!pending;
  outputWindow.setAttribute('aria-busy', String(!!pending));
  document.getElementById('output-window-hint').textContent = pending
    ? (open ? 'opening…' : 'closing…')
    : (open ? 'O or Esc closes' : '');
  document.getElementById('output-status').textContent = error || '';
}

function syncOutputDisplays(selected, displays) {
  if (!outputDisplay) return;
  const inventory = Array.isArray(displays) ? displays : [];
  const wanted = typeof selected === 'string' ? selected : '';
  const optionKey = JSON.stringify([wanted, inventory.map((display) => [display.id, display.label])]);
  if (outputDisplay.dataset.optionKey !== optionKey) {
    const options = [new Option('Automatic', '')];
    for (const display of inventory) {
      if (typeof display?.id !== 'string' || typeof display?.label !== 'string') continue;
      options.push(new Option(display.label, display.id));
    }
    if (wanted && !inventory.some((display) => display?.id === wanted)) {
      options.push(new Option('Selected display unavailable', wanted));
    }
    outputDisplay.replaceChildren(...options);
    outputDisplay.value = wanted;
    outputDisplay.dataset.optionKey = optionKey;
  }
  if (canSync(outputDisplay)) outputDisplay.value = wanted;
}

function syncOutputWindow(open, error = '', selectedDisplay = '', displays = [], inventoryGeneration = 0) {
  outputAuthoritativeOpen = !!open;
  outputAuthoritativeDisplay = typeof selectedDisplay === 'string' ? selectedDisplay : '';
  outputAuthoritativeGeneration = Number.isSafeInteger(inventoryGeneration) && inventoryGeneration >= 0
    ? inventoryGeneration
    : 0;
  syncOutputDisplays(outputAuthoritativeDisplay, displays);
  if (outputPendingOpen === outputAuthoritativeOpen || error) {
    outputRequestSequence += 1;
    outputPendingOpen = null;
  }
  renderOutputWindow(
    outputPendingOpen ?? outputAuthoritativeOpen,
    outputPendingOpen !== null,
    error,
  );
}

// --- Bounded program recorder / committed resampling ---

const recorderStart = document.getElementById('recorder-start');
const recorderFinish = document.getElementById('recorder-finish');
const recorderCancel = document.getElementById('recorder-cancel');
const recorderStatus = document.getElementById('recorder-status');
const recorderCounters = document.getElementById('recorder-counters');
const stillTarget = document.getElementById('still-target');
const resampleTarget = document.getElementById('resample-target');
const resampleDestination = document.getElementById('resample-destination');
let captureOptionKey = '';

function captureTargetPayload(value) {
  if (value === 'program') return { target: 'program' };
  const separator = value.indexOf(':');
  if (separator <= 0) return null;
  const kind = value.slice(0, separator);
  const id = value.slice(separator + 1);
  if (!/^[1-9][0-9]*$/.test(id)) return null;
  if (kind === 'layer') return { target: 'layer', layer_id: id };
  if (kind === 'group') return { target: 'group', group_id: id };
  return null;
}

function syncCaptureOptions() {
  const layerFacts = Array.isArray(latestLayers) ? latestLayers : [];
  const groups = Array.isArray(latestCreative?.groups) ? latestCreative.groups : [];
  const key = JSON.stringify([
    layerFacts.map(layer => [String(layer.layer_id || ''), String(layer.filename || '')]),
    groups.map(group => [String(group.group_id || ''), String(group.name || '')]),
  ]);
  if (key === captureOptionKey) return;
  captureOptionKey = key;
  const targetOptions = [new Option('Program', 'program')];
  for (const [index, layer] of layerFacts.entries()) {
    const id = String(layer.layer_id || '');
    if (!/^[1-9][0-9]*$/.test(id)) continue;
    targetOptions.push(new Option(`Layer ${index + 1} · ${layer.filename || id}`, `layer:${id}`));
  }
  for (const group of groups) {
    const id = String(group.group_id || '');
    if (!/^[1-9][0-9]*$/.test(id)) continue;
    targetOptions.push(new Option(`Group · ${group.name || id}`, `group:${id}`));
  }
  for (const select of [stillTarget, resampleTarget]) {
    const previous = select.value;
    select.replaceChildren(...targetOptions.map(option => option.cloneNode(true)));
    if ([...select.options].some(option => option.value === previous)) select.value = previous;
  }
  const previousDestination = resampleDestination.value;
  const destinations = [new Option('Choose layer…', '')];
  for (const [index, layer] of layerFacts.entries()) {
    const id = String(layer.layer_id || '');
    if (!/^[1-9][0-9]*$/.test(id)) continue;
    destinations.push(new Option(`Layer ${index + 1} · ${layer.filename || id}`, id));
  }
  resampleDestination.replaceChildren(...destinations);
  if ([...resampleDestination.options].some(option => option.value === previousDestination)) {
    resampleDestination.value = previousDestination;
  }
}

recorderStart.addEventListener('click', () => {
  recorderStart.disabled = true;
  if (!sendAction({
    action: 'start_program_recording',
    auto_import: document.getElementById('recorder-auto-import').checked,
  })) {
    recorderStart.disabled = false;
    recorderStatus.textContent = 'Control connection is offline.';
    recorderStatus.className = 'export-status error';
  }
});

recorderFinish.addEventListener('click', () => {
  if (sendAction({ action: 'finish_program_recording' })) recorderFinish.disabled = true;
});

recorderCancel.addEventListener('click', () => {
  if (sendAction({ action: 'cancel_program_recording' })) recorderCancel.disabled = true;
});

document.getElementById('still-capture').addEventListener('click', () => {
  const target = captureTargetPayload(stillTarget.value);
  if (!target) return;
  sendAction({
    action: 'capture_still',
    ...target,
    auto_import: document.getElementById('still-auto-import').checked,
  });
});

document.getElementById('resample-start').addEventListener('click', () => {
  const target = captureTargetPayload(resampleTarget.value);
  const destination = resampleDestination.value;
  if (!target || !/^[1-9][0-9]*$/.test(destination)) {
    recorderStatus.textContent = 'Choose a current destination layer.';
    recorderStatus.className = 'export-status error';
    return;
  }
  sendAction({
    action: 'start_resample',
    ...target,
    destination_layer_id: destination,
    activate: document.getElementById('resample-activate').checked,
  });
});

function syncRecorder(recorder = {}) {
  syncCaptureOptions();
  const status = String(recorder?.status || 'idle');
  const active = ['starting', 'recording', 'finishing'].includes(status);
  recorderStart.hidden = active;
  recorderStart.disabled = false;
  recorderFinish.hidden = !['starting', 'recording'].includes(status);
  recorderFinish.disabled = status !== 'recording';
  recorderCancel.hidden = !active;
  recorderCancel.disabled = false;
  document.getElementById('still-capture').disabled = active;
  document.getElementById('resample-start').disabled = active;
  const error = String(recorder?.error || '');
  const artifact = String(recorder?.artifact_name || '');
  if (error) {
    recorderStatus.textContent = error;
    recorderStatus.className = 'export-status error';
  } else if (status === 'starting') {
    recorderStatus.textContent = 'Preparing fixed frame pool and encoder…';
    recorderStatus.className = 'export-status';
  } else if (status === 'recording') {
    recorderStatus.textContent = 'Recording program';
    recorderStatus.className = 'export-status recording-live';
  } else if (status === 'finishing') {
    recorderStatus.textContent = 'Finishing and publishing…';
    recorderStatus.className = 'export-status';
  } else if (status === 'succeeded') {
    recorderStatus.textContent = artifact ? `Committed ${artifact}` : 'Recording committed';
    recorderStatus.className = 'export-status success';
  } else if (status === 'cancelled') {
    recorderStatus.textContent = 'Recording cancelled; temporary files removed';
    recorderStatus.className = 'export-status';
  } else {
    recorderStatus.textContent = '';
    recorderStatus.className = 'export-status';
  }
  const drops = ['dropped_not_ready', 'dropped_source_unavailable', 'dropped_pool_empty', 'dropped_queue_full']
    .reduce((sum, key) => sum + Math.max(0, Number(recorder?.[key]) || 0), 0);
  recorderCounters.textContent = `attempted ${Number(recorder?.attempted) || 0} · accepted ${Number(recorder?.accepted) || 0} · encoded ${Number(recorder?.encoded) || 0} · duplicated ${Number(recorder?.duplicated) || 0} · dropped ${drops}`;
}

// --- Preview health / exact physical-endpoint calibration ---

const stageHealthHud = document.getElementById('stage-health-hud');
const stageEndpoint = document.getElementById('stage-output-endpoint');
const stageTestCard = document.getElementById('stage-test-card');
const stageIdentification = document.getElementById('stage-output-identification');
const buildIdentity = document.getElementById('build-identity');

stageHealthHud.addEventListener('change', () => {
  sendAction({ action: 'set_stage_health_hud', enabled: stageHealthHud.checked });
});

stageTestCard.addEventListener('change', () => {
  const mode = ['off', 'smpte_bars', 'grid'].includes(stageTestCard.value)
    ? stageTestCard.value : 'off';
  sendAction({
    action: 'set_stage_test_card',
    mode,
    output_endpoint_id: mode === 'off' ? null : stageEndpoint.value,
  });
});

stageIdentification.addEventListener('change', () => {
  sendAction({
    action: 'set_output_identification',
    enabled: stageIdentification.checked,
    output_endpoint_id: stageIdentification.checked ? stageEndpoint.value : null,
  });
});

stageEndpoint.addEventListener('change', () => {
  // Moving the selector while a calibration tool is active retargets that
  // exact tool explicitly. No generic audience/program route is implied.
  if (stageTestCard.value !== 'off') {
    sendAction({
      action: 'set_stage_test_card',
      mode: stageTestCard.value,
      output_endpoint_id: stageEndpoint.value,
    });
  }
  if (stageIdentification.checked) {
    sendAction({
      action: 'set_output_identification',
      enabled: true,
      output_endpoint_id: stageEndpoint.value,
    });
  }
});

function boundedMetric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, number) : 0;
}

function budgetText(label, budget = {}) {
  const used = Number(budget?.used);
  const limit = Number(budget?.limit);
  const unit = String(budget?.unit || '').slice(0, 256);
  const detail = String(budget?.detail || 'unknown').slice(0, 256);
  if (Number.isFinite(used) && Number.isFinite(limit)) {
    return `${label} ${used}/${limit} ${unit}`.trim();
  }
  return `${label} ${detail}`;
}

function ensureStageEndpoint(endpointId, label = endpointId) {
  if (!/^[A-Za-z0-9._-]{1,128}$/.test(String(endpointId || ''))) return;
  if ([...stageEndpoint.options].some(option => option.value === endpointId)) return;
  // StageMap admits at most sixteen endpoints. Keep the legacy adapter plus a
  // bounded rolling set even if a hostile/stale snapshot invents identities.
  while (stageEndpoint.options.length >= 16) stageEndpoint.remove(1);
  stageEndpoint.add(new Option(String(label || endpointId).slice(0, 256), endpointId));
}

function boundedIdentityField(value, maxLength = 512) {
  return String(value ?? 'unreported').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, '')
    .slice(0, maxLength);
}

function syncBuildIdentity(identity = {}) {
  const fields = [
    ['version', `${boundedIdentityField(identity.package_name, 80)} ${boundedIdentityField(identity.version, 80)}`],
    ['git', `${boundedIdentityField(identity.git_sha, 80)}${identity.git_dirty ? ' (dirty)' : ' (clean)'}`],
    ['artifact', identity.published_artifact ? 'published' : 'local/unpublished'],
    ['target', boundedIdentityField(identity.target, 160)],
    ['profile', boundedIdentityField(identity.profile, 80)],
    ['features', boundedIdentityField(identity.enabled_features, 512)],
    ['rustc', boundedIdentityField(identity.rustc_vv, 2048)],
    ['cargo', boundedIdentityField(identity.cargo_version, 256)],
    ['linker', boundedIdentityField(identity.linker_identity, 512)],
    ['sdk', boundedIdentityField(identity.sdk_identity, 512)],
    ['FFmpeg libs', boundedIdentityField(identity.ffmpeg_libraries, 1024)],
    ['ffmpeg binary', `${boundedIdentityField(identity.ffmpeg_binary_version, 512)} · ${boundedIdentityField(identity.ffmpeg_binary_sha256, 80)}`],
    ['ffprobe binary', `${boundedIdentityField(identity.ffprobe_binary_version, 512)} · ${boundedIdentityField(identity.ffprobe_binary_sha256, 80)}`],
    ['shaders', boundedIdentityField(identity.shader_bundle_sha256, 80)],
    ['Cargo.lock', boundedIdentityField(identity.cargo_lock_sha256, 80)],
    ['identity', boundedIdentityField(identity.identity_sha256, 80)],
  ];
  buildIdentity.textContent = fields.map(([label, value]) => `${label}: ${value}`).join('\n');
}

function syncStageHealth(health = {}) {
  syncBuildIdentity(health?.build_identity || {});
  const tools = health?.tools || {};
  if (canSync(stageHealthHud)) stageHealthHud.checked = !!tools.health_hud_enabled;
  if (canSync(stageTestCard)) stageTestCard.value = ['off', 'smpte_bars', 'grid'].includes(tools.test_card)
    ? tools.test_card : 'off';
  if (canSync(stageIdentification)) {
    stageIdentification.checked = !!tools.output_identification_enabled;
  }
  const output = health?.output || {};
  const endpointId = /^[A-Za-z0-9._-]{1,128}$/.test(String(output.endpoint_id || ''))
    ? String(output.endpoint_id) : 'legacy-output-1';
  const endpointLabel = String(output.identity || endpointId).slice(0, 256);
  ensureStageEndpoint(endpointId, endpointLabel);
  ensureStageEndpoint(tools.test_card_endpoint_id);
  ensureStageEndpoint(tools.output_identification_endpoint_id);
  const selected = tools.test_card_endpoint_id || tools.output_identification_endpoint_id || endpointId;
  if (canSync(stageEndpoint) && [...stageEndpoint.options].some(option => option.value === selected)) {
    stageEndpoint.value = selected;
  }
  const fps = boundedMetric(health?.fps);
  const p50 = boundedMetric(health?.frame_time_p50_us) / 1000;
  const p95 = boundedMetric(health?.frame_time_p95_us) / 1000;
  const p99 = boundedMetric(health?.frame_time_p99_us) / 1000;
  const latenessP50 = boundedMetric(health?.schedule_lateness_p50_us) / 1000;
  const latenessP95 = boundedMetric(health?.schedule_lateness_p95_us) / 1000;
  const latenessP99 = boundedMetric(health?.schedule_lateness_p99_us) / 1000;
  const skippedTicks = boundedMetric(
    health?.skipped_program_ticks ?? health?.missed_deadlines,
  );
  document.getElementById('stage-health-summary').textContent =
    `${fps.toFixed(1)} fps · frame ${p50.toFixed(2)}/${p95.toFixed(2)}/${p99.toFixed(2)} ms p50/p95/p99 · lateness ${latenessP50.toFixed(3)}/${latenessP95.toFixed(3)}/${latenessP99.toFixed(3)} ms · skipped ${skippedTicks} · ${boundedMetric(output.width)}×${boundedMetric(output.height)} @ ${(boundedMetric(output.refresh_millihz) / 1000).toFixed(3)} Hz`;
  const gpuTiming = health?.gpu_timing || {};
  const actionTiming = health?.action_timing || {};
  const flightRecorder = health?.flight_recorder || {};
  const latencyTriple = value => [value?.p50_us, value?.p95_us, value?.p99_us]
    .map(micros => (boundedMetric(micros) / 1000).toFixed(3)).join('/');
  const gpuText = gpuTiming.supported
    ? `GPU p95 ms source ${(boundedMetric(gpuTiming?.source_prepare?.p95_us) / 1000).toFixed(3)} · creative ${(boundedMetric(gpuTiming?.creative_composition?.p95_us) / 1000).toFixed(3)} · temporal ${(boundedMetric(gpuTiming?.temporal_motion?.p95_us) / 1000).toFixed(3)} · Mosh/VHS ${(boundedMetric(gpuTiming?.mosh_vhs?.p95_us) / 1000).toFixed(3)} · resolve ${(boundedMetric(gpuTiming?.audience_resolve?.p95_us) / 1000).toFixed(3)} · submit ${(boundedMetric(gpuTiming?.submission?.p95_us) / 1000).toFixed(3)}`
    : 'GPU timestamps unsupported on this adapter/backend';
  const flightText = flightRecorder.enabled
    ? `private flight recorder ${boundedMetric(flightRecorder.rotation_seconds)} s × ${boundedMetric(flightRecorder.retained_rotations)} · ${boundedMetric(flightRecorder.queued_events)} queued · ${boundedMetric(flightRecorder.dropped_full)} pressure drops`
    : 'private flight recorder unavailable';
  document.getElementById('stage-health-timing').textContent =
    `${gpuText} · action ms p50/p95/p99 ingress→apply ${latencyTriple(actionTiming?.ingress_to_apply)} · apply→submit ${latencyTriple(actionTiming?.apply_to_submit)} · sequence ${boundedMetric(actionTiming?.last_presented_sequence)} · generation ${boundedMetric(actionTiming?.last_submission_generation)} · ${flightText}`;
  const budgets = health?.budgets || {};
  document.getElementById('stage-health-budgets').textContent = [
    budgetText('GPU', budgets.gpu), budgetText('media', budgets.media),
    budgetText('Mosh send', budgets.ntsc), budgetText('motion', budgets.motion),
  ].join(' · ');
  const rows = (Array.isArray(health?.layers) ? health.layers : []).slice(0, 256).map(layer => {
    const row = document.createElement('div');
    row.className = 'stage-health-layer';
    const age = layer.decoded_age_ms == null ? 'n/a' : `${boundedMetric(layer.decoded_age_ms)} ms`;
    row.textContent = `${String(layer.name || layer.layer_id || 'layer').slice(0, 256)} · decoded ${age} · pending ${boundedMetric(layer.pending_frames)} · drops ${boundedMetric(layer.dropped_frames)} · ${String(layer.status || '').slice(0, 256)}`;
    return row;
  });
  document.getElementById('stage-health-layers').replaceChildren(...rows);
}

// --- B11 Monitoring bay ---
//
// The instruments run only while an observer watches. This panel declares
// itself a watcher exactly while its MONITORING BAY section is expanded and
// the tab is visible; the declaration is re-asserted on a heartbeat so a
// silently discarded tab expires server-side instead of pinning the
// readback armed.

const monitorBayGroup = document.getElementById('monitor-bay-group');
const monitorBayNative = document.getElementById('monitor-bay-native');
const monitorBayProbe = document.getElementById('monitor-bay-probe');
const monitorBayWaveform = document.getElementById('monitor-bay-waveform');
const monitorBayScope = document.getElementById('monitor-bay-scope');
const monitorBayStatus = document.getElementById('monitor-bay-status');
const monitorBaySources = document.getElementById('monitor-bay-sources');
let monitorBayLastSample = -1;
let monitorBayWatching = false;

function monitorBayIsWatching() {
  return !document.hidden && !monitorBayGroup.classList.contains('collapsed');
}

function sendMonitorWatch() {
  const watching = monitorBayIsWatching();
  // A repeated `true` is the heartbeat; a repeated `false` is elided so a
  // closed section costs no traffic at all.
  if (watching || monitorBayWatching) {
    sendAction({ action: 'monitor_watch', enabled: watching });
  }
  monitorBayWatching = watching;
}

monitorBayGroup.querySelector('.fx-group-header').addEventListener('click', () => {
  // The collapse handler toggles the class on this same click; observe the
  // final state after it has run.
  setTimeout(sendMonitorWatch, 0);
});
document.addEventListener('visibilitychange', sendMonitorWatch);
setInterval(() => {
  if (monitorBayIsWatching()) sendMonitorWatch();
}, 4000);

monitorBayNative.addEventListener('change', () => {
  sendAction({ action: 'set_monitor_bay', enabled: monitorBayNative.checked });
});

monitorBayProbe.addEventListener('change', () => {
  const probe = ['program', 'program_tap', 'gesture_canvas', 'ntsc_line_state', 'melt_band_mask', 'motion_field'].includes(monitorBayProbe.value)
    ? monitorBayProbe.value : 'program';
  sendAction({ action: 'set_monitor_probe', probe });
});

// The six 75% colour-bar targets, presentation-only: the engine's law module
// derives the authoritative positions from the same projection.
function monitorScopeTargets() {
  const bars = [[0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.75, 0.75, 0], [0, 0.75, 0.75], [0.75, 0, 0.75]];
  return bars.map(([r, g, b]) => {
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const u = (b - y) * 0.565;
    const v = (r - y) * 0.713;
    return [0.5 + u * 1.4 * 0.5, 0.5 - v * 1.4 * 0.5];
  });
}

function drawMonitorBitmap(canvas, b64, width, height, decorate) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (b64 && width > 0 && height > 0 && canvas.width === width && canvas.height === height) {
    let bytes;
    try {
      bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    } catch {
      bytes = null;
    }
    if (bytes && bytes.length >= width * height) {
      const image = ctx.createImageData(width, height);
      for (let i = 0; i < width * height; i++) {
        const v = bytes[i];
        image.data[i * 4] = v / 5 | 0;
        image.data[i * 4 + 1] = v;
        image.data[i * 4 + 2] = v / 3 | 0;
        image.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(image, 0, 0);
    }
  }
  if (decorate) decorate(ctx);
}

function syncMonitorBay(bay = {}) {
  if (!monitorBayGroup) return;
  if (canSync(monitorBayNative)) monitorBayNative.checked = !!bay.native_overlay;
  const probe = ['program', 'program_tap', 'gesture_canvas', 'ntsc_line_state', 'melt_band_mask', 'motion_field'].includes(bay.probe) ? bay.probe : 'program';
  if (canSync(monitorBayProbe)) monitorBayProbe.value = probe;
  if (!bay.active) {
    if (monitorBayLastSample !== -1) {
      monitorBayLastSample = -1;
      drawMonitorBitmap(monitorBayWaveform, '', 0, 0);
      drawMonitorBitmap(monitorBayScope, '', 0, 0);
      monitorBaySources.replaceChildren();
    }
    monitorBayStatus.textContent = 'Instruments run only while this section is open and the tab is visible.';
    return;
  }
  const status = String(bay.probe_status || '').slice(0, 64);
  monitorBayStatus.textContent = status ? `Probe ${probe}: ${status} — instruments hold.` : `Probe ${probe} · sample ${Math.max(0, Number(bay.sample) || 0)}`;
  const sample = Math.max(0, Number(bay.sample) || 0);
  if (sample === monitorBayLastSample) return;
  monitorBayLastSample = sample;
  drawMonitorBitmap(
    monitorBayWaveform,
    String(bay.waveform_b64 || ''),
    Math.max(0, Number(bay.waveform_width) || 0),
    Math.max(0, Number(bay.waveform_height) || 0),
    (ctx) => {
      ctx.strokeStyle = 'rgba(160,160,160,0.6)';
      ctx.beginPath();
      ctx.moveTo(0, 0.5);
      ctx.lineTo(monitorBayWaveform.width, 0.5);
      ctx.moveTo(0, monitorBayWaveform.height - 0.5);
      ctx.lineTo(monitorBayWaveform.width, monitorBayWaveform.height - 0.5);
      ctx.stroke();
    },
  );
  drawMonitorBitmap(
    monitorBayScope,
    String(bay.scope_b64 || ''),
    Math.max(0, Number(bay.scope_size) || 0),
    Math.max(0, Number(bay.scope_size) || 0),
    (ctx) => {
      const size = monitorBayScope.width;
      ctx.strokeStyle = 'rgba(140,140,140,0.5)';
      for (const radius of [0.33, 0.66, 1.0]) {
        ctx.beginPath();
        ctx.arc(size / 2, size / 2, radius * size / 2, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.strokeStyle = 'rgba(200,200,200,0.8)';
      for (const [x, y] of monitorScopeTargets()) {
        ctx.strokeRect(x * (size - 1) - 1.5, y * (size - 1) - 1.5, 3, 3);
      }
    },
  );
  const sources = Array.isArray(bay.sources) ? bay.sources.slice(0, 64) : [];
  monitorBaySources.textContent = sources
    .map((s) => `${String(s.name || '').slice(0, 24)} ${(Number(s.value) || 0).toFixed(2)}`)
    .join(' · ');
}

// --- Remote / QR ---

function syncRemote(url, status) {
  const el = document.getElementById('remote-url');
  const qr = document.getElementById('qr-img');
  const available = typeof url === 'string' && url.length > 0;
  if (qr) {
    qr.hidden = !available;
    qr.setAttribute('aria-hidden', available ? 'false' : 'true');
  }
  if (el) {
    const next = available
      ? url
      : (typeof status === 'string' && status.length > 0 ? status : 'LAN unavailable');
    if (el.textContent !== next) el.textContent = next;
  }
}

// --- Helpers ---

function formatValue(v, min, max, step) {
  if (step >= 1) return v.toFixed(0);
  const exactPrecision = rangeValuePrecision(Number(step), Number(min));
  if (exactPrecision > 3) return v.toFixed(exactPrecision);
  if (max <= 1 && min >= -1) return v.toFixed(2);
  if (step >= 0.01) return v.toFixed(1);
  return v.toFixed(3);
}

// Bind the static panel after all of its native input handlers exist. The
// observer is a future-proof fallback for any later generated range control.
bindRangeEditors(document);
const rangeEditorObserver = new MutationObserver((records) => {
  for (const record of records) {
    record.addedNodes.forEach((node) => {
      if (node.nodeType === Node.ELEMENT_NODE) bindRangeEditors(node);
    });
  }
});
rangeEditorObserver.observe(document.body, { childList: true, subtree: true });

// ===== B15: control search, filters, and help ============================
// Entirely client-side over data the panel already holds: no new wire action,
// no engine round trip, nothing asked of the render thread. A filtered-out
// control is *hidden*, never disabled — clearing the query restores every row
// exactly as it was, and a hidden control that a route drives keeps being
// driven.
//
// Cost discipline: the index walks the DOM only when the filter criteria
// change, or on a snapshot while a filter is actually engaged. With no filter
// active the 30 Hz state packet does no extra work at all.

const CONTROL_SEARCH = { query: '', moving: false, changed: false };

// The three families that carry help. Every other row is still indexed and
// still filtered — it simply has no help sentence to search over, which is
// better than leaving it on screen while the operator is trying to narrow
// the view.
const CONTROL_HELP_SCOPES = {
  param: 'master',
  temporal: 'temporal',
  ntsc: 'ntsc',
};

// Dataset keys that describe a row's layout rather than name its control.
const CONTROL_LAYOUT_KEYS = new Set(['min', 'max', 'step', 'default', 'group']);

const CONTROL_DEFAULT_TABLES = {
  master: () => MASTER_PARAM_DEFAULTS,
  temporal: () => TEMPORAL_PARAM_DEFAULTS,
  ntsc: () => NTSC_PARAM_DEFAULTS,
};

let controlRouteTargets = [];

function controlHelpFor(scope, param) {
  const table = (window.CONTROL_HELP || {})[scope];
  return (table && table[param]) || '';
}

// The engine's modulation target naming, transcribed as three rules rather
// than a two-hundred-entry table. A target these rules cannot map simply
// lights nothing — they never light the wrong row, because every rule is an
// exact string equality.
function routeDrivesControl(target, scope, param) {
  if (!scope) return false;
  if (target === param) return true;
  if (scope !== 'temporal') return false;
  if (target === `temporal_${param}`) return true;
  return param.startsWith('disp_') && target === `display_${param.slice(5)}`;
}

// The row's default, or null when the panel genuinely does not know it. A
// null is reported as "not changed" rather than guessed: a filter that
// invents differences is worse than one that admits a blind spot.
function controlDefaultFor(entry) {
  const table = entry.scope ? CONTROL_DEFAULT_TABLES[entry.scope]?.() : null;
  if (table && table[entry.param] !== undefined) return Number(table[entry.param]);
  const authored = entry.row.querySelector('[data-default]');
  if (authored) return Number(authored.dataset.default);
  if (table) {
    // A known family with an unlisted key falls back to the slider minimum,
    // exactly as the double-click reset does.
    const slider = entry.row.querySelector('input[type="range"]');
    if (slider) return Number(entry.row.dataset.min ?? slider.min ?? 0);
  }
  return null;
}

function controlRowIsChanged(entry) {
  const slider = entry.row.querySelector('input[type="range"]');
  if (slider) {
    const fallback = controlDefaultFor(entry);
    if (fallback === null || Number.isNaN(fallback)) return false;
    return Math.abs(Number(slider.value) - fallback) > 1e-6;
  }
  const checkbox = entry.row.querySelector('input[type="checkbox"]');
  if (checkbox) {
    // The authored default is what the markup ships checked, exactly as a
    // select's default is its `selected` option. Assuming "off" would report
    // every checkbox that ships on as permanently changed.
    const table = entry.scope ? CONTROL_DEFAULT_TABLES[entry.scope]?.() : null;
    const listed = table ? table[entry.param] : undefined;
    const fallback = listed !== undefined
      ? Boolean(listed)
      : checkbox.hasAttribute('checked');
    return checkbox.checked !== fallback;
  }
  const select = entry.row.querySelector('select');
  if (select) {
    const authored = select.querySelector('option[selected]');
    return authored ? select.value !== authored.value : false;
  }
  const number = entry.row.querySelector('input[type="number"]');
  if (number) {
    const fallback = controlDefaultFor(entry);
    if (fallback === null || Number.isNaN(fallback)) return false;
    return Math.abs(Number(number.value) - fallback) > 1e-6;
  }
  return false;
}

function buildControlIndex() {
  const rows = [];
  document.querySelectorAll('#master-fx .param-row').forEach((row) => {
    // A row is named by its first dataset key that is not a layout hint, so
    // families the help table does not cover are still indexed and filtered.
    let family = null;
    let param = null;
    for (const key of Object.keys(row.dataset)) {
      if (CONTROL_LAYOUT_KEYS.has(key)) continue;
      if (!row.dataset[key]) continue;
      family = key;
      param = row.dataset[key];
      break;
    }
    // Rows bound by element id rather than a data attribute (Audio Gain, the
    // Dice scope selectors, Morph Law) carry no identity, but they are real
    // labelled controls and must still filter. They match on their label and
    // section, and they never claim MOVING or CHANGED, because without an
    // identity the panel genuinely cannot tell.
    if (!family) param = '';
    const scope = family ? CONTROL_HELP_SCOPES[family] || null : null;
    const group = row.closest('.fx-group');
    const section = row.closest('details.temporal-study');
    const groupLabel =
      group?.querySelector('.group-label')?.textContent
      || group?.querySelector('summary')?.textContent
      || '';
    const sectionLabel = section?.querySelector('summary')?.textContent || '';
    const label = row.querySelector('label')?.textContent || '';
    const help = scope ? controlHelpFor(scope, param) : '';
    // The help sentence rides the row as a native tooltip. It costs no layout
    // and screen readers announce it, which an inline expander would not do
    // for a control the operator is already focused on.
    if (help && row.title !== help) row.title = help;
    rows.push({
      row,
      group,
      scope,
      param,
      // Two corpora, deliberately matched differently. Identity text takes a
      // plain substring so "phos" finds Persistence Red; help prose takes a
      // word-start match, because a substring over sentences is noise —
      // "gain" would otherwise match "against".
      identity: `${label} ${groupLabel} ${sectionLabel} ${param}`.toLowerCase(),
      help: help.toLowerCase(),
    });
  });
  return rows;
}

function controlFilterEngaged() {
  return (
    CONTROL_SEARCH.query.trim() !== ''
    || CONTROL_SEARCH.moving
    || CONTROL_SEARCH.changed
  );
}

function applyControlFilter() {
  const countEl = document.getElementById('control-search-count');
  const engaged = controlFilterEngaged();
  const index = buildControlIndex();
  const groups = new Set();

  if (!engaged) {
    for (const entry of index) {
      entry.row.classList.remove('control-hidden');
      if (entry.group) groups.add(entry.group);
    }
    for (const group of groups) group.classList.remove('control-hidden');
    if (countEl && countEl.textContent !== '') countEl.textContent = '';
    return;
  }

  const query = CONTROL_SEARCH.query.trim().toLowerCase();
  // One regex per pass, not one per row: the word-start test runs over every
  // indexed control on every keystroke.
  const helpPattern = query
    ? new RegExp(`\\b${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`)
    : null;
  let shown = 0;
  for (const entry of index) {
    let visible = true;
    if (query) {
      visible = entry.identity.includes(query)
        || (entry.help !== '' && helpPattern.test(entry.help));
    }
    if (visible && CONTROL_SEARCH.moving) {
      visible = controlRouteTargets.some((target) =>
        routeDrivesControl(target, entry.scope, entry.param));
    }
    if (visible && CONTROL_SEARCH.changed) visible = controlRowIsChanged(entry);
    entry.row.classList.toggle('control-hidden', !visible);
    if (visible) shown += 1;
    if (entry.group) groups.add(entry.group);
  }
  // A group with nothing left is hidden too, so the column does not become a
  // list of empty headings.
  for (const group of groups) {
    group.classList.toggle('control-hidden', !group.querySelector('.param-row:not(.control-hidden)'));
  }
  const text = `${shown}`;
  if (countEl && countEl.textContent !== text) countEl.textContent = text;
}

// Refresh the compiled route list from the snapshot the panel already
// receives. MOVING is derived from this and nothing else.
function syncControlFilters(modulation) {
  const routings = (modulation && modulation.routings) || [];
  const targets = routings
    .map((routing) => routing && routing.target)
    .filter((target) => typeof target === 'string');
  const changed =
    targets.length !== controlRouteTargets.length
    || targets.some((target, i) => target !== controlRouteTargets[i]);
  if (changed) controlRouteTargets = targets;
  // Values move constantly, so CHANGED has to re-evaluate on every packet —
  // but only while it is actually engaged.
  if (CONTROL_SEARCH.changed || (changed && CONTROL_SEARCH.moving)) applyControlFilter();
}

(function bindControlSearch() {
  const input = document.getElementById('control-search');
  const moving = document.getElementById('filter-moving');
  const changed = document.getElementById('filter-changed');
  if (!input) return;

  input.addEventListener('input', () => {
    CONTROL_SEARCH.query = input.value;
    applyControlFilter();
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      // Escape clears the query rather than bubbling: while a search is open
      // that is what the key plainly means here.
      event.stopPropagation();
      if (input.value !== '') {
        input.value = '';
        CONTROL_SEARCH.query = '';
        applyControlFilter();
      } else {
        input.blur();
      }
    }
  });

  const toggle = (button, key) => {
    button?.addEventListener('click', () => {
      CONTROL_SEARCH[key] = !CONTROL_SEARCH[key];
      button.setAttribute('aria-pressed', CONTROL_SEARCH[key] ? 'true' : 'false');
      applyControlFilter();
    });
  };
  toggle(moving, 'moving');
  toggle(changed, 'changed');

  // Build the index once at startup: filtering is lazy, but the help
  // tooltips it attaches should be there from the first hover, not only
  // after the operator happens to search for something.
  applyControlFilter();

  // `/` focuses the search, unless the operator is already typing somewhere.
  document.addEventListener('keydown', (event) => {
    if (event.key !== '/' || event.ctrlKey || event.metaKey || event.altKey) return;
    const active = document.activeElement;
    const typing =
      active
      && (active.tagName === 'INPUT'
        || active.tagName === 'TEXTAREA'
        || active.tagName === 'SELECT'
        || active.isContentEditable);
    if (typing) return;
    event.preventDefault();
    input.focus();
    input.select();
  });
})();

// ===== B15: descriptive prose folds into a ? beside each section ===
// A column of controls should be controls. The static grey notes that explain
// a section are moved into a `?` affordance in that section's own header, so
// the explanation is one hover away instead of occupying a paragraph between
// every group. Notes carrying live truth — counters, statuses, error text —
// are identified by having an `id` the engine writes to, and are left exactly
// where they are, because those have to be readable without hovering.
(function foldStaticNotesIntoHeaders() {
  document.querySelectorAll('.audio-status:not([id])').forEach((note) => {
    const text = note.textContent.trim();
    if (!text) return;
    const details = note.closest('details');
    const host = details
      ? details.querySelector('summary')
      : note.closest('.fx-group')?.querySelector('.fx-group-header');
    if (!host) return;
    const mark = document.createElement('button');
    mark.type = 'button';
    mark.className = 'group-help';
    mark.textContent = '?';
    mark.title = text;
    mark.setAttribute('aria-label', `About this section: ${text}`);
    // The header and the summary are both click targets of their own, so the
    // question mark must not collapse the very section it explains.
    mark.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    host.appendChild(mark);
    note.remove();
  });
})();
// ===== B15 snapshot bank ==================================================
// Eight whole-rig slots and one glide time. Recall does not invent a second
// way to interpolate a rig: the engine loads the slot into the existing
// Morph A/B pair and glides, so ownership transfer, midpoint discretes, and
// wrapped hues are the laws that already exist. The panel only stores, names,
// and recalls.
//
// A save and a recall both carry the two revision barriers a Morph capture
// carries, because both capture the live rig and would otherwise attach
// positional slot data to a stack that has since changed.
const SNAPSHOT_BANK_SLOTS = 8;

function snapshotBankRevisions() {
  return {
    stack_revision: layerStackRevision,
    composition_revision: compositionRevision,
  };
}

(function buildSnapshotBank() {
  const host = document.getElementById('snapshot-bank-slots');
  if (!host) return;
  for (let index = 0; index < SNAPSHOT_BANK_SLOTS; index += 1) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'snapshot-bank-slot';
    button.dataset.slot = String(index);
    button.textContent = String(index + 1);
    button.title =
      `Slot ${index + 1}. Click to recall (loads the slot into the Morph pair and glides); `
      + 'shift-click to store the current rig; alt-click to empty it.';
    button.setAttribute('aria-label', `Snapshot bank slot ${index + 1}`);
    button.addEventListener('click', (event) => {
      if (event.altKey) {
        sendAction({ action: 'snapshot_bank_clear', slot: index });
        return;
      }
      if (event.shiftKey) {
        sendAction({ action: 'snapshot_bank_save', slot: index, ...snapshotBankRevisions() });
        return;
      }
      sendAction({ action: 'snapshot_bank_recall', slot: index, ...snapshotBankRevisions() });
    });
    host.appendChild(button);
  }
})();

const snapshotBankGlide = document.getElementById('snapshot-bank-glide');
snapshotBankGlide?.addEventListener('input', () => {
  const beats = Math.min(64, Math.max(0, Number(snapshotBankGlide.value) || 0));
  const readout = document.getElementById('snapshot-bank-glide-value');
  if (readout) readout.textContent = String(beats);
  sendAction({ action: 'set_snapshot_bank_glide', beats });
});

function syncSnapshotBank(morph) {
  const filled = Array.isArray(morph?.bank_filled) ? morph.bank_filled : [];
  document.querySelectorAll('.snapshot-bank-slot').forEach((button) => {
    const index = Number(button.dataset.slot);
    const isFilled = filled[index] === true;
    button.classList.toggle('filled', isFilled);
    // The label carries the state too, so a filled slot is not distinguished
    // by colour alone.
    button.textContent = isFilled ? `[${index + 1}]` : String(index + 1);
  });
  if (snapshotBankGlide && canSync(snapshotBankGlide)
      && typeof morph?.bank_glide_beats === 'number') {
    snapshotBankGlide.value = String(morph.bank_glide_beats);
    const readout = document.getElementById('snapshot-bank-glide-value');
    if (readout) readout.textContent = String(morph.bank_glide_beats);
  }
  const status = document.getElementById('snapshot-bank-status');
  if (status) {
    const count = filled.filter(Boolean).length;
    const text = count === 0
      ? 'shift-click a slot to store the whole rig'
      : `${count} of ${SNAPSHOT_BANK_SLOTS} slots stored`;
    if (status.textContent !== text) status.textContent = text;
  }
}
