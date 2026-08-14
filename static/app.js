// collide-o-scope — web control panel

const statusEl = document.getElementById('ws-status');
const layersList = document.getElementById('layers-list');
const layersEmpty = document.getElementById('layers-empty');
const libraryGrid = document.getElementById('library-grid');
const spoutInForm = document.getElementById('spout-in-form');
const spoutInName = document.getElementById('spout-in-name');
const spoutInStatus = document.getElementById('spout-in-status');

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

// Declared before the socket connects so reconnect reconciliation is safe
// even when the initial connection opens immediately.
let padPointerId = null;
let padLastSend = 0;
let padLastPosition = [0.5, 0.5];
let padNeedsReconcile = false;
let beatQuantizeEnabled = false;
let layerStackRevision = 0;
let latestLayerIdentities = [];
let transportAuthoritativePaused = false;
let transportPendingPaused = null;
let transportRequestSequence = 0;
let outputAuthoritativeOpen = false;
let outputPendingOpen = null;
let outputRequestSequence = 0;

const QUANTIZABLE_ACTIONS = new Set([
  'set_param', 'set_layer_param', 'set_layer_effect', 'set_ntsc_param', 'set_temporal',
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
    reconcileGyroStreamConnection();
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
    outputRequestSequence += 1;
    outputPendingOpen = null;
    renderOutputWindow(outputAuthoritativeOpen, false, 'Control connection is offline.');
    showGyroDisconnected();
    setTimeout(connect, 2000);
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) return;

    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'state') {
        syncEffects(msg.effects);
        syncNtsc(msg.ntsc);
        layerStackRevision = Number(msg.layer_stack_revision) || 0;
        syncLayers(msg.layers);
        syncLibrary(msg.library);
        syncTransport(msg.paused);
        syncExport(msg.export_progress, msg.export_error, msg.export_status);
        syncPatchSave(msg.patch_save_status || '');
        syncModulation(msg.modulation);
        syncAudio(msg.audio);
        syncMidi(msg.midi);
        syncTemporal(msg.temporal);
        syncSpout(msg.spout);
        syncRemote(msg.remote_url);
        syncMorph(msg.morph);
        syncOutputWindow(msg.output_window, msg.output_error);
        syncBlackout(msg.blackout);
        syncQuantize(msg.quantized_pending || 0);
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
document.addEventListener('pointerdown', (e) => {
  const el = e.target.closest('input,select,[data-range-editor]');
  if (el) {
    const current = touchedControls.get(el);
    const held = current instanceof Set ? current : new Set();
    held.add(e.pointerId);
    touchedControls.set(el, held);
  }
}, true);
const releaseControl = (e) => {
  for (const [el, t] of touchedControls) {
    if (!(t instanceof Set) || !t.has(e.pointerId)) continue;
    t.delete(e.pointerId);
    touchedControls.set(el, t.size ? t : performance.now());
  }
};
document.addEventListener('pointerup', releaseControl, true);
document.addEventListener('pointercancel', releaseControl, true);
const releaseAllControls = () => {
  const now = performance.now();
  for (const [el, state] of touchedControls) {
    if (!el.isConnected) touchedControls.delete(el);
    else if (state instanceof Set) touchedControls.set(el, now);
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

function layerSelector(layer, index) {
  return { index, layer_id: layer?.layer_id || null };
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
    const precision = Math.min(12, Math.max(decimalPlaces(step), decimalPlaces(base)));
    value = Number(value.toFixed(precision));
  }
  return Number.isFinite(value) ? value : null;
}

function formatRangeValue(slider, rawValue) {
  const value = normalizeRangeValue(slider, rawValue);
  if (value === null) return '';
  const { step } = rangeBounds(slider);
  const precision = Number.isFinite(step) ? Math.min(6, decimalPlaces(step)) : 3;
  return value.toFixed(precision);
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
    const defaults = {
      pixelate: 1, rgb_split: 0, hue_shift: 0, saturation: 0,
      downsample: 1,
      brightness: 0, contrast: 0, posterize: 0, grain_intensity: 0,
      grain_size: 1, vignette: 0, color_drift: 0, breathe_scale: 0,
      breathe_rotation: 0, breathe_position: 0,
      cellular_amount: 0, cellular_scale: 10, cellular_warp: 0.35,
      cellular_speed: 0.25, cellular_gap_amount: 0,
      cellular_gap_threshold: 0.65, cellular_gap_softness: 0.08,
      key_color_r: 0, key_color_g: 1, key_color_b: 0,
      key_threshold: 0.5, key_softness: 0.1, key_tolerance: 0.15,
    };
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
    const defaults = {
      head_switching_height: 8, tracking_noise_height: 24,
      edge_wave_speed: 0.5,
    };
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
  const valueEl = row.querySelector('.value');
  const select = row.querySelector('select');

  if (slider) {
    slider.min = min;
    slider.max = max;
    slider.step = step;
    const defaults = {
      fb_zoom: 1,
      key_threshold: 0.1,
      key_softness: 0.03,
      key_history: 1,
    };
    slider.value = defaults[param] ?? min;

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      sendAction({ action: 'set_temporal', param, value: v });
    });
    resetRangeOnDoubleActivation(slider, defaults[param] ?? 0);
  }

  if (select) {
    select.addEventListener('change', () => {
      sendAction({ action: 'set_temporal', param, value: parseInt(select.value) });
    });
  }
});

function syncTemporal(t) {
  if (!t) return;
  for (const [param, value] of Object.entries(t)) {
    const row = document.querySelector(`.param-row[data-temporal="${param}"]`);
    if (!row) continue;
    const slider = row.querySelector('input[type="range"]');
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
      select.value = Math.round(value);
    }
  }
}

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
  if (sendAction({ action: 'set_master_paused', paused: target })) {
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

document.getElementById('btn-revert-master').addEventListener('click', () => {
  sendAction({ action: 'reset_fx' });
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

// --- Sync NTSC/VHS UI from server ---

function syncNtsc(ntsc) {
  if (!ntsc) return;
  const ntscStatus = document.getElementById('ntsc-status');
  if (ntscStatus) ntscStatus.textContent = ntsc.error || '';
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

// --- Sync layers ---

function syncLayers(layers) {
  if (!layers) return;
  latestLayerIdentities = layers.map((layer) => String(layer.layer_id || ''));
  const liveDisclosureKeys = new Set(layers.map(layerDisclosureKey));
  for (const key of layerDisclosureState.keys()) {
    if (!liveDisclosureKeys.has(key)) layerDisclosureState.delete(key);
  }
  syncExportAudioLayers(layers);
  layersEmpty.style.display = layers.length === 0 ? 'block' : 'none';

  const layerKey = JSON.stringify(layers.map((layer) => [layer.layer_id, layer.filename, layer.source_kind]));
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
  ['cellular_amount', 'Amount', 'range', 0, 1, 0.05, 0],
  ['cellular_scale', 'Scale', 'range', 2, 32, 1, 10],
  ['cellular_warp', 'Warp', 'range', 0, 1, 0.05, 0.35],
  ['cellular_speed', 'Drift', 'range', 0, 2, 0.05, 0.25],
  ['cellular_gap_amount', 'Gap Key', 'range', 0, 1, 0.01, 0],
  ['cellular_gap_threshold', 'Gap Thresh', 'range', 0, 1, 0.01, 0.65],
  ['cellular_gap_softness', 'Gap Soft', 'range', 0, 0.5, 0.01, 0.08],
];

function layerEffectsHtml(effects, index) {
  const rowHtml = ([param, label, kind, min, max, step, fallback]) => {
    const value = effects[param] ?? fallback;
    if (kind === 'checkbox') {
      return `<div class="param-row toggle-row layer-effect-row" data-layer-effect="${param}"><label>${label}</label><label class="toggle"><input type="checkbox" ${value ? 'checked' : ''} aria-label="Layer ${index + 1} ${label}"><span class="toggle-slider"></span></label></div>`;
    }
    if (kind === 'select') {
      return `<div class="param-row select-row layer-effect-row" data-layer-effect="${param}"><label>${label}</label><select aria-label="Layer ${index + 1} ${label}"><option value="0">Gaussian</option><option value="1">Perlin</option><option value="2">Salt &amp; Pepper</option><option value="3">Blue</option></select></div>`;
    }
    return `<div class="param-row layer-effect-row" data-layer-effect="${param}" data-min="${min}" data-max="${max}" data-step="${step}"><label>${label}</label><input type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="Layer ${index + 1} ${label}"><span class="value">${formatValue(Number(value), min, max, step)}</span></div>`;
  };
  const cellularIndex = LAYER_EFFECT_CONTROLS.findIndex(([param]) => param === 'cellular_amount');
  const ordinary = LAYER_EFFECT_CONTROLS.slice(0, cellularIndex).map(rowHtml).join('');
  const cellular = LAYER_EFFECT_CONTROLS.slice(cellularIndex).map(rowHtml).join('');
  return `${ordinary}
    <div class="layer-cellular-disclosure">
      <button class="layer-cellular-toggle" type="button" aria-expanded="false" aria-controls="layer-cellular-body-${index}">
        <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>CELLULAR</span>
      </button>
      <div class="layer-cellular-body" id="layer-cellular-body-${index}" role="region" aria-label="Layer ${index + 1} cellular effects" hidden>${cellular}</div>
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
  const remembered = layerDisclosureState.get(key) || { effects: false, cellular: false };
  const effectsButton = card.querySelector('.layer-fx-toggle');
  const effectsBody = card.querySelector('.layer-fx-body');
  const cellularButton = card.querySelector('.layer-cellular-toggle');
  const cellularBody = card.querySelector('.layer-cellular-body');
  setLayerDisclosure(effectsButton, effectsBody, remembered.effects);
  setLayerDisclosure(cellularButton, cellularBody, remembered.cellular);
  effectsButton.addEventListener('click', () => {
    remembered.effects = effectsButton.getAttribute('aria-expanded') !== 'true';
    layerDisclosureState.set(key, remembered);
    setLayerDisclosure(effectsButton, effectsBody, remembered.effects);
  });
  cellularButton.addEventListener('click', () => {
    remembered.cellular = cellularButton.getAttribute('aria-expanded') !== 'true';
    layerDisclosureState.set(key, remembered);
    setLayerDisclosure(cellularButton, cellularBody, remembered.cellular);
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
      sendAction({ action: 'set_layer_effect', ...layerSelector(layer, index), param, value });
    };
    control.addEventListener(kind === 'range' ? 'input' : 'change', send);
    if (kind === 'range') resetRangeOnDoubleActivation(control, fallback);
  });
}

function createLayerCard(layer, index) {
  const card = document.createElement('div');
  card.className = 'layer-card expanded';
  card.dataset.index = index;

  card.innerHTML = `
    <div class="layer-header">
      <button class="layer-drag-btn" title="Drag or use arrow keys to reorder" aria-label="Move layer ${index + 1}; use arrow keys to reorder" aria-keyshortcuts="ArrowUp ArrowDown Home End">&#x2630;</button>
      ${layer.source_kind === 'spout'
        ? '<span class="layer-thumb lib-placeholder" aria-hidden="true">LIVE</span>'
        : `<img class="layer-thumb" src="/thumb/${encodeURIComponent(layer.filename)}" alt="">`}
      <span class="layer-num">${index + 1}</span>
      <button class="layer-play-btn" title="Play/Pause" aria-label="${layer.paused ? 'Play' : 'Pause'} layer ${index + 1}">${layer.paused ? '\u25B6' : '\u25A0'}</button>
      <span class="layer-title">${escapeHtml(layer.filename || 'Untitled')}</span>
      <button class="layer-vis-btn ${layer.visible ? 'visible' : ''}" title="Visibility" aria-label="${layer.visible ? 'Hide' : 'Show'} layer ${index + 1}">${layer.visible ? '\u25C9' : '\u25CB'}</button>
      <button class="layer-remove-btn" title="Remove" aria-label="Remove layer ${index + 1}">\u00D7</button>
    </div>
    <div class="layer-progress"><div class="layer-progress-fill" style="width:${(layer.progress * 100).toFixed(1)}%"></div></div>
    <div class="layer-source-status" role="status" aria-live="polite"></div>
    <div class="layer-body">
      <div class="param-row" data-layer="${index}" data-param="opacity">
        <label>Opacity</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.opacity}">
        <span class="value">${layer.opacity.toFixed(2)}</span>
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
        <label>Blend</label>
        <select>
          <option value="normal" ${layer.blend_mode === 'normal' ? 'selected' : ''}>Normal</option>
          <option value="screen" ${layer.blend_mode === 'screen' ? 'selected' : ''}>Screen</option>
          <option value="multiply" ${layer.blend_mode === 'multiply' ? 'selected' : ''}>Multiply</option>
          <option value="difference" ${layer.blend_mode === 'difference' ? 'selected' : ''}>Difference</option>
        </select>
      </div>
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
      <div class="audio-status">This layer's key reveals layers beneath it. Chroma modes use the RGB target and tolerance.</div>
      <div class="param-row toggle-row layer-master-bypass" title="Skips Digital/Analog/Cellular/Motion/VHS master processing; own Layer FX/opacity/key/blend remain; Temporal still affects the final program.">
        <label>Bypass Master FX</label>
        <label class="toggle">
          <input type="checkbox" ${layer.bypass_master_fx ? 'checked' : ''} aria-label="Bypass Master FX for layer ${index + 1}" aria-describedby="layer-master-bypass-help-${index}">
          <span class="toggle-slider"></span>
        </label>
        <span id="layer-master-bypass-help-${index}" class="visually-hidden">Skips Digital/Analog/Cellular/Motion/VHS master processing; own Layer FX/opacity/key/blend remain; Temporal still affects the final program.</span>
      </div>
      <div class="layer-fx-heading">
        <button class="layer-fx-toggle" type="button" aria-expanded="false" aria-controls="layer-fx-body-${index}">
          <span class="layer-disclosure-chevron" aria-hidden="true">&#x25B6;</span><span>Layer effects</span>
        </button>
        <button class="layer-reset-fx" type="button" title="Reset layer effects (opacity and transport unchanged)" aria-label="Reset layer effects (opacity and transport unchanged)">Reset FX</button>
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
    const current = card._layerState || layer;
    sendAction({ action: 'set_layer_paused', ...layerSelector(current, index), paused: !current.paused });
  });

  // Pointer-owned reorder works for mouse, pen, and touch. The list stays
  // stable throughout the gesture and commits one move on release.
  const dragBtn = card.querySelector('.layer-drag-btn');
  dragBtn.addEventListener('pointerdown', (e) => {
    if (layerDrag) return;
    layerDrag = { pointerId: e.pointerId, from: index, to: index };
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
    const { from, to } = layerDrag;
    clearLayerDrag();
    if (from !== to) sendAction({ action: 'move_layer', from, to, layer_id: layer.layer_id || null, stack_revision: layerStackRevision });
    e.stopPropagation();
    e.preventDefault();
  });
  dragBtn.addEventListener('pointercancel', clearLayerDrag);
  dragBtn.addEventListener('lostpointercapture', (e) => {
    if (layerDrag?.pointerId === e.pointerId) clearLayerDrag();
  });
  dragBtn.addEventListener('keydown', (e) => {
    const targets = {
      ArrowUp: index - 1,
      ArrowLeft: index - 1,
      ArrowDown: index + 1,
      ArrowRight: index + 1,
      Home: 0,
      End: layersList.children.length - 1,
    };
    if (!(e.key in targets)) return;
    e.preventDefault();
    const to = Math.max(0, Math.min(layersList.children.length - 1, targets[e.key]));
    if (to !== index) sendAction({ action: 'move_layer', from: index, to, layer_id: layer.layer_id || null, stack_revision: layerStackRevision });
  });

  // Visibility
  card.querySelector('.layer-vis-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const current = card._layerState || layer;
    sendAction({ action: 'set_layer_visibility', ...layerSelector(current, index), visible: !current.visible });
  });

  // Remove
  card.querySelector('.layer-remove-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    sendAction({ action: 'remove_layer', ...layerSelector(layer, index) });
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
        sendAction({ action: 'set_layer_param', ...layerSelector(layer, index), param, value: v });
      });
      const defaults = {
        opacity: 1, speed: 1, fps: 30, key_threshold: 0.5, key_softness: 0.1,
        key_color_r: 0, key_color_g: 1, key_color_b: 0, key_tolerance: 0.15,
      };
      resetRangeOnDoubleActivation(slider, defaults[param] ?? parseFloat(slider.min));
    }

    if (select) {
      select.addEventListener('change', () => {
        const v = param === 'key_mode' ? parseInt(select.value) : select.value;
        sendAction({ action: 'set_layer_param', ...layerSelector(layer, index), param, value: v });
      });
    }
  });
  card.querySelector('.layer-reset-fx').addEventListener('click', () => {
    sendAction({ action: 'reset_layer_fx', ...layerSelector(layer, index) });
  });
  card.querySelector('.layer-master-bypass input').addEventListener('change', (event) => {
    sendAction({ action: 'set_layer_param', ...layerSelector(layer, index), param: 'bypass_master_fx', value: event.currentTarget.checked });
  });
  wireLayerDisclosures(card, layer, index);
  wireLayerEffects(card, layer, index);
  bindRangeEditors(card);

  updateLayerCard(card, layer, index);
  return card;
}

function updateLayerCard(card, layer, index) {
  if (!card) return;
  // Cards survive ordinary state snapshots, so callbacks must read the latest
  // immutable layer DTO instead of the one captured at card construction.
  card._layerState = layer;
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

  for (const [param, digits] of [['speed', 2], ['fps', 0]]) {
    const row = card.querySelector(`.param-row[data-param="${param}"]`);
    const slider = row?.querySelector('input[type="range"]');
    const valEl = row?.querySelector('.value');
    if (slider && canSync(slider)) {
      slider.value = layer[param];
      if (valEl) valEl.textContent = Number(layer[param]).toFixed(digits);
    }
  }

  const blendRow = card.querySelector('.param-row[data-param="blend_mode"]');
  if (blendRow) {
    const select = blendRow.querySelector('select');
    if (select && canSync(select)) {
      select.value = layer.blend_mode;
    }
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
  for (const [param, , kind, min, max, step, fallback] of LAYER_EFFECT_CONTROLS) {
    const row = card.querySelector(`[data-layer-effect="${param}"]`);
    const control = row?.querySelector('input,select');
    if (!control || !canSync(control)) continue;
    const value = layer.effects?.[param] ?? fallback;
    if (kind === 'checkbox') control.checked = !!value;
    else control.value = String(value);
    if (kind === 'range') row.querySelector('.value').textContent = formatValue(Number(value), min, max, step);
  }
}

// --- Sync library ---

const activePreviewIntervals = new Set();

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
    item.tabIndex = 0;
    item.setAttribute('role', 'button');
    item.setAttribute('aria-label', `Add ${filename} as a layer`);

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
        libUploadStatus.textContent = r.ok ? `${text} → Recycle Bin` : `${filename}: ${text}`;
      } catch {
        libUploadStatus.textContent = `${filename}: remove failed`;
      }
      setTimeout(() => { libUploadStatus.textContent = ''; }, 4000);
    });
    item.appendChild(del);

    let lastTouchAdd = 0;
    item.addEventListener('dblclick', () => {
      if (performance.now() - lastTouchAdd < 500) return;
      sendAction({ action: 'add_layer', filename });
    });
    item.addEventListener('keydown', (e) => {
      if (e.target !== item || (e.key !== 'Enter' && e.key !== ' ')) return;
      e.preventDefault();
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
      statusElement.textContent = ok
        ? `${response} added`
        : `${file.name}: ${response || 'upload failed'}`;
      statusElement.classList.toggle('error', !ok);
      finish({ ok, filename: ok ? response : '', error: ok ? '' : response });
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
  btn.title = paused ? 'Resume visual program' : 'Pause visual program';
  btn.dataset.paused = String(paused);
  btn.disabled = pending;
  btn.toggleAttribute('aria-busy', pending);
  btn.setAttribute('aria-label', btn.title);
}

function syncTransport(paused) {
  transportAuthoritativePaused = !!paused;
  if (transportPendingPaused === transportAuthoritativePaused) {
    transportRequestSequence += 1;
    transportPendingPaused = null;
  }
  renderMasterTransport(
    transportPendingPaused ?? transportAuthoritativePaused,
    transportPendingPaused !== null,
  );
}


// --- Export / Render ---

const patchCaptureButton = document.getElementById('patch-capture');
const patchSaveStatus = document.getElementById('patch-save-status');

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

let exportActive = false;

document.getElementById('export-start').addEventListener('click', () => {
  if (exportActive) return;
  const [w, h] = document.getElementById('export-resolution').value.split('x').map(Number);
  const durationInput = document.getElementById('export-duration');
  const duration = Math.min(300, Math.max(1, parseFloat(durationInput.value) || 10));
  durationInput.value = duration;
  const fps = [24, 30, 60, 120, 240].includes(parseInt(document.getElementById('export-fps').value))
    ? parseInt(document.getElementById('export-fps').value) : 30;
  const audioSelect = document.getElementById('export-audio');
  const audioOption = audioSelect.selectedOptions[0];
  const audioLayerId = audioSelect.value === '' || audioSelect.value.startsWith('legacy-index:')
    ? null : audioSelect.value;
  const audioLayer = audioOption?.dataset.index === undefined ? null : parseInt(audioOption.dataset.index, 10);
  exportActive = true;
  document.getElementById('export-start').style.display = 'none';
  document.getElementById('export-cancel').style.display = '';
  document.getElementById('export-progress').style.display = '';
  document.getElementById('export-status').textContent = 'Starting render…';
  if (!sendAction({ action: 'start_export', width: w, height: h, fps, duration_secs: duration, audio_layer: audioLayer, audio_layer_id: audioLayerId })) {
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

function syncExport(progress, error, status = '') {
  const startBtn = document.getElementById('export-start');
  const cancelBtn = document.getElementById('export-cancel');
  const progressEl = document.getElementById('export-progress');
  const fillEl = document.getElementById('export-fill');
  const textEl = document.getElementById('export-text');
  const statusEl = document.getElementById('export-status');
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
      statusEl.textContent = 'Render complete!';
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
  ['temporal_slit_angle', 'Temporal Slit Angle'],
  ['temporal_key_threshold', 'Temporal Key Threshold'],
  ['temporal_key_softness', 'Temporal Key Softness'],
  ['temporal_key_history', 'Temporal Key History'],
  ['morph', 'Morph'],
];
const MASTER_MOD_TARGETS = MOD_TARGETS.slice();

const MAX_MOD_LAYERS = 16;
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

// Build the 4 static LFO rows once.
for (let i = 0; i < 4; i++) {
  const row = document.createElement('div');
  row.className = 'lfo-row';
  row.dataset.lfo = i;
  row.innerHTML = `
    <span class="lfo-name">LFO ${i + 1}</span>
    <select class="lfo-shape">${optionsHtml(LFO_SHAPES, 'sine')}</select>
    <select class="lfo-rate">${optionsHtml(LFO_RATES, 4)}</select>
    <div class="lfo-meter"><div class="lfo-meter-fill"></div></div>
  `;
  row.querySelector('.lfo-shape').addEventListener('change', (e) => {
    sendAction({ action: 'set_lfo', index: i, param: 'shape', value: e.target.value });
  });
  row.querySelector('.lfo-rate').addEventListener('change', (e) => {
    sendAction({ action: 'set_lfo', index: i, param: 'beats', value: parseFloat(e.target.value) });
  });
  lfoList.appendChild(row);
}

const MOD_SOURCES = [
  ['lfo0', 'L1'],
  ['lfo1', 'L2'],
  ['lfo2', 'L3'],
  ['lfo3', 'L4'],
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
];

function currentModTargets(selected = '') {
  const groups = [['Master / Program', MASTER_MOD_TARGETS.slice()]];
  const liveLayerCount = Math.min(MAX_MOD_LAYERS, latestLayerIdentities.length);
  for (let layer = 1; layer <= liveLayerCount; layer++) {
    const targets = [];
    for (const [suffix, label] of LAYER_FX_TARGETS) {
      targets.push([`layer${layer}_${suffix}`, `L${layer} ${label}`]);
    }
    groups.push([`Layer ${layer}`, targets]);
  }
  // Preserve visibility for a legacy/out-of-range selection until the engine
  // publishes its authoritative replacement instead of showing a blank menu.
  if (selected && !groups.some(([, values]) => values.some(([value]) => value === selected))) {
    groups.push(['Unavailable', [[selected, `${selected} (unavailable)`]]]);
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
    const layerMatch = /^layer([1-9]|1[0-6])_/.exec(target);
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
    if (canSync(shapeSel)) shapeSel.value = lfo.shape;
    if (canSync(rateSel)) rateSel.value = lfo.beats;
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
      const bipolar = /^(lfo\d|gyro_|pad_)/.test(r.source || '');
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

midiEnabled.addEventListener('change', () => {
  sendAction({ action: 'set_midi', param: 'enabled', value: midiEnabled.checked });
});

const midiClockSync = document.getElementById('midi-clock-sync');
midiClockSync.addEventListener('change', () => {
  sendAction({ action: 'set_midi', param: 'clock_sync', value: midiClockSync.checked });
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
const spoutStatus = document.getElementById('spout-status');

spoutEnabled.addEventListener('change', () => {
  sendAction({ action: 'set_spout', enabled: spoutEnabled.checked });
});

function syncSpout(s) {
  if (!s) return;
  if (canSync(spoutEnabled)) spoutEnabled.checked = s.enabled;
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
    spoutStatus.textContent = 'sender: collide-o-scope';
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
  if (!sendAction({ action: 'morph_capture', slot: 'a', stack_revision: layerStackRevision })) {
    morphStatus.textContent = 'Control connection is offline; A was not captured.';
  }
});
document.getElementById('morph-set-b').addEventListener('click', () => {
  if (!sendAction({ action: 'morph_capture', slot: 'b', stack_revision: layerStackRevision })) {
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
    morphStatus.textContent = 'morphing — sliders follow the crossfade';
  } else if (m.has_a || m.has_b) {
    morphStatus.textContent = m.has_a ? 'A set — capture B to engage' : 'B set — capture A to engage';
  } else {
    morphStatus.textContent = 'capture two states, then crossfade';
  }
}

// --- Fullscreen output window ---

const outputWindow = document.getElementById('output-window');
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
  outputWindow.setAttribute('aria-busy', String(!!pending));
  document.getElementById('output-window-hint').textContent = pending
    ? (open ? 'opening…' : 'closing…')
    : (open ? 'O or Esc closes' : '');
  document.getElementById('output-status').textContent = error || '';
}

function syncOutputWindow(open, error = '') {
  outputAuthoritativeOpen = !!open;
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

// --- Remote / QR ---

function syncRemote(url) {
  const el = document.getElementById('remote-url');
  if (el && url && el.textContent !== url) {
    el.textContent = url;
  }
}

// --- Helpers ---

function formatValue(v, min, max, step) {
  if (step >= 1) return v.toFixed(0);
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
