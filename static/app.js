// collide-o-scope — web control panel

const statusEl = document.getElementById('ws-status');
const layersList = document.getElementById('layers-list');
const layersEmpty = document.getElementById('layers-empty');
const libraryGrid = document.getElementById('library-grid');

// --- WebSocket ---

let ws;
function connect() {
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${wsProto}://${location.host}/ws`);

  ws.onopen = () => {
    statusEl.classList.add('connected');
    statusEl.classList.remove('disconnected');
    statusEl.title = 'connected';
  };

  ws.onclose = () => {
    statusEl.classList.remove('connected');
    statusEl.classList.add('disconnected');
    statusEl.title = 'disconnected';
    setTimeout(connect, 2000);
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) return;

    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'state') {
        syncEffects(msg.effects);
        syncNtsc(msg.ntsc);
        syncLayers(msg.layers);
        syncLibrary(msg.library);
        syncTransport(msg.paused);
        syncExport(msg.export_progress, msg.export_error);
        syncModulation(msg.modulation);
        syncAudio(msg.audio);
        syncMidi(msg.midi);
        syncTemporal(msg.temporal);
        syncSpout(msg.spout);
        syncRemote(msg.remote_url);
        syncMorph(msg.morph);
        syncOutputWindow(msg.output_window);
        syncBlackout(msg.blackout);
      }
    } catch (err) {
      console.warn('[ws] parse error:', err);
    }
  };
}
connect();

function sendAction(action) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.log('[ws] send:', JSON.stringify(action));
    ws.send(JSON.stringify(action));
  } else {
    console.warn('[ws] not connected, dropping:', action);
  }
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
    slider.value = min;

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      sendAction({ action: 'set_param', param, value: v });
    });
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
    slider.value = param === 'fb_zoom' ? 1 : min;

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      sendAction({ action: 'set_temporal', param, value: v });
    });
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
    if (slider && valueEl && document.activeElement !== slider) {
      slider.value = value;
      valueEl.textContent = formatValue(
        value,
        parseFloat(row.dataset.min),
        parseFloat(row.dataset.max),
        parseFloat(row.dataset.step)
      );
    }
    if (select && document.activeElement !== select) {
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
  header.addEventListener('click', (e) => {
    if (e.target.classList.contains('group-reset')) return;
    const group = header.closest('.fx-group');
    group.classList.toggle('collapsed');
    const key = groupKey(group);
    if (key) {
      if (group.classList.contains('collapsed')) collapsedState.add(key);
      else collapsedState.delete(key);
      localStorage.setItem('cos-collapsed', JSON.stringify([...collapsedState]));
    }
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
  sendAction({ action: 'toggle_master_pause' });
});

document.getElementById('btn-stop').addEventListener('click', () => {
  sendAction({ action: 'reset_fx' });
});

document.getElementById('btn-blackout').addEventListener('click', () => {
  sendAction({ action: 'toggle_blackout' });
});

function syncBlackout(on) {
  document.getElementById('btn-blackout').classList.toggle('active', !!on);
}

// --- Sync effects UI from server ---

function syncEffects(effects) {
  if (!effects) return;
  for (const [param, value] of Object.entries(effects)) {
    const row = document.querySelector(`.param-row[data-param="${param}"]`);
    if (!row) continue;

    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const select = row.querySelector('select');

    if (slider && valueEl && document.activeElement !== slider) {
      slider.value = value;
      const min = parseFloat(row.dataset.min);
      const max = parseFloat(row.dataset.max);
      const step = parseFloat(row.dataset.step);
      valueEl.textContent = formatValue(value, min, max, step);
    }

    if (checkbox) {
      checkbox.checked = !!value;
    }

    if (select) {
      select.value = value;
    }
  }
}

// --- Sync NTSC/VHS UI from server ---

function syncNtsc(ntsc) {
  if (!ntsc) return;
  for (const [param, value] of Object.entries(ntsc)) {
    const row = document.querySelector(`.param-row[data-ntsc="${param}"]`);
    if (!row) continue;

    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const select = row.querySelector('select');

    if (slider && valueEl && document.activeElement !== slider) {
      slider.value = value;
      const min = parseFloat(row.dataset.min);
      const max = parseFloat(row.dataset.max);
      const step = parseFloat(row.dataset.step);
      valueEl.textContent = formatValue(value, min, max, step);
    }

    if (checkbox) {
      checkbox.checked = !!value;
    }

    if (select && document.activeElement !== select) {
      select.value = value;
    }
  }
}

// --- Sync layers ---

function syncLayers(layers) {
  if (!layers) return;
  layersEmpty.style.display = layers.length === 0 ? 'block' : 'none';

  // Rebuild if count changed
  if (layersList.children.length !== layers.length) {
    layersList.innerHTML = '';
    layers.forEach((layer, i) => {
      layersList.appendChild(createLayerCard(layer, i));
    });
  } else {
    layers.forEach((layer, i) => {
      updateLayerCard(layersList.children[i], layer, i);
    });
  }
}

function createLayerCard(layer, index) {
  const card = document.createElement('div');
  card.className = 'layer-card expanded';
  card.dataset.index = index;

  card.innerHTML = `
    <div class="layer-header">
      <img class="layer-thumb" src="/thumb/${encodeURIComponent(layer.filename)}" alt="">
      <span class="layer-num">${index + 1}</span>
      <button class="layer-play-btn" title="Play/Pause">${layer.paused ? '\u25B6' : '\u25A0'}</button>
      <span class="layer-title">${layer.filename || 'Untitled'}</span>
      <button class="layer-vis-btn ${layer.visible ? 'visible' : ''}" title="Visibility">${layer.visible ? '\u25C9' : '\u25CB'}</button>
      <button class="layer-remove-btn" title="Remove">\u00D7</button>
    </div>
    <div class="layer-progress"><div class="layer-progress-fill" style="width:${(layer.progress * 100).toFixed(1)}%"></div></div>
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
          <option value="1" ${layer.key_mode === 1 ? 'selected' : ''}>Bright</option>
          <option value="2" ${layer.key_mode === 2 ? 'selected' : ''}>Dark</option>
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
    console.log('[layer] play/pause clicked, index:', index);
    sendAction({ action: 'toggle_layer_pause', index });
  });

  // Visibility
  card.querySelector('.layer-vis-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    console.log('[layer] visibility clicked, index:', index);
    sendAction({ action: 'toggle_visibility', index });
  });

  // Remove
  card.querySelector('.layer-remove-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    sendAction({ action: 'remove_layer', index });
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
        sendAction({ action: 'set_layer_param', index, param, value: v });
      });
    }

    if (select) {
      select.addEventListener('change', () => {
        const v = param === 'key_mode' ? parseInt(select.value) : select.value;
        sendAction({ action: 'set_layer_param', index, param, value: v });
      });
    }
  });

  return card;
}

function updateLayerCard(card, layer, index) {
  if (!card) return;
  const playBtn = card.querySelector('.layer-play-btn');
  const title = card.querySelector('.layer-title');
  const visBtn = card.querySelector('.layer-vis-btn');
  const progressFill = card.querySelector('.layer-progress-fill');

  if (playBtn) playBtn.textContent = layer.paused ? '\u25B6' : '\u25A0';
  if (title) title.textContent = layer.filename || 'Untitled';
  if (visBtn) {
    visBtn.textContent = layer.visible ? '\u25C9' : '\u25CB';
    visBtn.className = `layer-vis-btn ${layer.visible ? 'visible' : ''}`;
  }
  if (progressFill) {
    progressFill.style.width = `${(layer.progress * 100).toFixed(1)}%`;
  }

  // Sync layer param sliders (skip if user is actively dragging)
  const opacityRow = card.querySelector('.param-row[data-param="opacity"]');
  if (opacityRow) {
    const slider = opacityRow.querySelector('input[type="range"]');
    const valEl = opacityRow.querySelector('.value');
    if (slider && document.activeElement !== slider) {
      slider.value = layer.opacity;
      if (valEl) valEl.textContent = layer.opacity.toFixed(2);
    }
  }

  const speedRow = card.querySelector('.param-row[data-param="speed"]');
  if (speedRow) {
    const slider = speedRow.querySelector('input[type="range"]');
    const valEl = speedRow.querySelector('.value');
    if (slider && document.activeElement !== slider) {
      slider.value = layer.speed;
      if (valEl) valEl.textContent = layer.speed.toFixed(2);
    }
  }

  const blendRow = card.querySelector('.param-row[data-param="blend_mode"]');
  if (blendRow) {
    const select = blendRow.querySelector('select');
    if (select && document.activeElement !== select) {
      select.value = layer.blend_mode;
    }
  }

  const keyModeRow = card.querySelector('.param-row[data-param="key_mode"]');
  if (keyModeRow) {
    const select = keyModeRow.querySelector('select');
    if (select && document.activeElement !== select) {
      select.value = layer.key_mode;
    }
  }

  for (const param of ['key_threshold', 'key_softness']) {
    const row = card.querySelector(`.param-row[data-param="${param}"]`);
    if (row) {
      const slider = row.querySelector('input[type="range"]');
      const valEl = row.querySelector('.value');
      if (slider && document.activeElement !== slider) {
        slider.value = layer[param];
        if (valEl) valEl.textContent = layer[param].toFixed(2);
      }
    }
  }
}

// --- Sync library ---

// Cache for preview frame availability: filename → frame count (0 = not checked, -1 = unavailable)
const previewCache = new Map();

function syncLibrary(files) {
  if (!files) return;

  // Only rebuild if changed
  const currentCount = libraryGrid.querySelectorAll('.library-item').length;
  if (currentCount === files.length) return;

  libraryGrid.innerHTML = '';

  if (files.length === 0) {
    libraryGrid.innerHTML = '<p class="dim" style="grid-column:1/-1;text-align:center;padding:12px;">No video files</p>';
    return;
  }

  files.forEach((filename) => {
    const item = document.createElement('div');
    item.className = 'library-item';
    item.title = filename;

    // Thumbnail image from server (retries if not yet generated)
    const img = document.createElement('img');
    img.dataset.retries = '0';
    const thumbUrl = `/thumb/${encodeURIComponent(filename)}`;
    img.src = thumbUrl;
    img.onerror = () => {
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

    // Hover preview animation
    let hoverInterval = null;
    let hoverFrame = 0;

    item.addEventListener('mouseenter', () => {
      const enc = encodeURIComponent(filename);
      // Start cycling through preview frames
      hoverFrame = 0;
      hoverInterval = setInterval(() => {
        hoverFrame = (hoverFrame + 1) % 8;
        img.src = `/preview/${enc}/${hoverFrame}`;
      }, 250);
    });

    item.addEventListener('mouseleave', () => {
      if (hoverInterval) {
        clearInterval(hoverInterval);
        hoverInterval = null;
      }
      // Restore static thumbnail
      img.src = thumbUrl;
    });

    // Filename label on hover
    const label = document.createElement('span');
    label.className = 'lib-label';
    label.textContent = filename.replace(/\.[^.]+$/, '');
    item.appendChild(label);

    item.addEventListener('dblclick', () => {
      sendAction({ action: 'add_layer', filename });
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

function uploadClip(file) {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/upload?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        libUploadStatus.textContent = `${file.name} — ${Math.round((e.loaded / e.total) * 100)}%`;
      }
    };
    xhr.onload = () => {
      libUploadStatus.textContent = xhr.status === 200
        ? `${xhr.responseText} added`
        : `${file.name}: ${xhr.responseText || 'upload failed'}`;
      resolve();
    };
    xhr.onerror = () => {
      libUploadStatus.textContent = `${file.name}: upload failed`;
      resolve();
    };
    libUploadStatus.textContent = `${file.name} — 0%`;
    xhr.send(file);
  });
}

// --- Sync transport ---

function syncTransport(paused) {
  const btn = document.getElementById('btn-play-all');
  btn.textContent = paused ? '\u25B6' : '\u23F8';
  btn.title = paused ? 'Play All' : 'Pause All';
}


// --- Export / Render ---

let exportActive = false;

document.getElementById('export-start').addEventListener('click', () => {
  const [w, h] = document.getElementById('export-resolution').value.split('x').map(Number);
  const duration = parseFloat(document.getElementById('export-duration').value) || 10;
  const fps = parseInt(document.getElementById('export-fps').value) || 30;
  sendAction({ action: 'start_export', width: w, height: h, fps, duration_secs: duration });
});

document.getElementById('export-cancel').addEventListener('click', () => {
  sendAction({ action: 'cancel_export' });
});

function syncExport(progress, error) {
  const startBtn = document.getElementById('export-start');
  const cancelBtn = document.getElementById('export-cancel');
  const progressEl = document.getElementById('export-progress');
  const fillEl = document.getElementById('export-fill');
  const textEl = document.getElementById('export-text');
  const statusEl = document.getElementById('export-status');

  if (progress > 0 && progress < 1) {
    // Rendering in progress
    exportActive = true;
    startBtn.style.display = 'none';
    cancelBtn.style.display = '';
    progressEl.style.display = '';
    fillEl.style.width = (progress * 100) + '%';
    textEl.textContent = Math.round(progress * 100) + '%';
    statusEl.textContent = '';
  } else if (progress >= 1) {
    // Done
    if (exportActive) {
      startBtn.style.display = '';
      cancelBtn.style.display = 'none';
      progressEl.style.display = 'none';
      if (error) {
        statusEl.textContent = 'Error: ' + error;
        statusEl.className = 'export-status error';
      } else {
        statusEl.textContent = 'Render complete!';
        statusEl.className = 'export-status success';
      }
      exportActive = false;
    }
  } else {
    // Idle
    if (!exportActive) {
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
  ['vignette', 'Vignette'],
  ['color_drift', 'Drift'],
  ['breathe_scale', 'Bth Scale'],
  ['breathe_rotation', 'Bth Rotate'],
  ['breathe_position', 'Bth Drift'],
  ['ntsc_snow', 'VHS Snow'],
  ['ntsc_tracking_snow', 'VHS Track Snow'],
  ['ntsc_edge_wave', 'VHS Edge Wave'],
  ['ntsc_head_shift', 'VHS Head Shift'],
  ['ntsc_chroma_loss', 'VHS Chroma Loss'],
  ['ntsc_luma_noise', 'VHS Luma Noise'],
  ['layer1_opacity', 'L1 Opacity'],
  ['layer2_opacity', 'L2 Opacity'],
  ['layer3_opacity', 'L3 Opacity'],
  ['layer4_opacity', 'L4 Opacity'],
  ['layer1_speed', 'L1 Speed'],
  ['layer2_speed', 'L2 Speed'],
  ['layer3_speed', 'L3 Speed'],
  ['layer4_speed', 'L4 Speed'],
  ['layer1_key', 'L1 Key'],
  ['layer2_key', 'L2 Key'],
  ['layer3_key', 'L3 Key'],
  ['layer4_key', 'L4 Key'],
];

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

function optionsHtml(pairs, selected) {
  return pairs
    .map(([v, label]) => `<option value="${v}" ${String(v) === String(selected) ? 'selected' : ''}>${label}</option>`)
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

function createRoutingRow(routing, index) {
  const row = document.createElement('div');
  row.className = 'routing-row';
  row.dataset.index = index;
  row.innerHTML = `
    <select class="routing-source">${optionsHtml(MOD_SOURCES, routing.source)}</select>
    <span class="routing-arrow">&#x2192;</span>
    <select class="routing-target">${optionsHtml(MOD_TARGETS, routing.target)}</select>
    <input type="range" class="routing-depth" min="-1" max="1" step="0.01" value="${routing.depth}">
    <span class="routing-depth-val">${routing.depth.toFixed(2)}</span>
    <button class="routing-remove" title="Remove">&#xD7;</button>
  `;
  row.querySelector('.routing-source').addEventListener('change', (e) => {
    sendAction({ action: 'set_routing', index, param: 'source', value: e.target.value });
  });
  row.querySelector('.routing-target').addEventListener('change', (e) => {
    sendAction({ action: 'set_routing', index, param: 'target', value: e.target.value });
  });
  row.querySelector('.routing-depth').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    row.querySelector('.routing-depth-val').textContent = v.toFixed(2);
    sendAction({ action: 'set_routing', index, param: 'depth', value: v });
  });
  row.querySelector('.routing-remove').addEventListener('click', () => {
    sendAction({ action: 'remove_routing', index });
  });
  return row;
}

function syncModulation(m) {
  if (!m) return;

  if (document.activeElement !== bpmInput) {
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
    if (document.activeElement !== shapeSel) shapeSel.value = lfo.shape;
    if (document.activeElement !== rateSel) rateSel.value = lfo.beats;
    // Live meter: map [-1, 1] → [0%, 100%]
    const fill = row.querySelector('.lfo-meter-fill');
    fill.style.width = `${((lfo.value + 1) * 50).toFixed(1)}%`;
  });

  syncPad(m.pad);

  // Gyro meters (values come from whichever device is streaming).
  if (m.gyro) {
    for (const [id, v] of [['gm-yaw', m.gyro[0]], ['gm-pitch', m.gyro[1]], ['gm-roll', m.gyro[2]]]) {
      const el = document.getElementById(id);
      if (el) el.style.width = `${(v * 100).toFixed(1)}%`;
    }
  }

  const routings = m.routings || [];
  if (routingList.children.length !== routings.length) {
    routingList.innerHTML = '';
    routings.forEach((r, i) => routingList.appendChild(createRoutingRow(r, i)));
  } else {
    routings.forEach((r, i) => {
      const row = routingList.children[i];
      const sourceSel = row.querySelector('.routing-source');
      const targetSel = row.querySelector('.routing-target');
      const depthSlider = row.querySelector('.routing-depth');
      const depthVal = row.querySelector('.routing-depth-val');
      if (document.activeElement !== sourceSel) sourceSel.value = r.source;
      if (document.activeElement !== targetSel) targetSel.value = r.target;
      if (document.activeElement !== depthSlider) {
        depthSlider.value = r.depth;
        depthVal.textContent = r.depth.toFixed(2);
      }
    });
  }
}

// --- Audio input ---

const audioEnabled = document.getElementById('audio-enabled');
const audioGain = document.getElementById('audio-gain');
const audioGainVal = document.getElementById('audio-gain-val');
const audioStatus = document.getElementById('audio-status');

audioEnabled.addEventListener('change', () => {
  sendAction({ action: 'set_audio', param: 'enabled', value: audioEnabled.checked });
});

audioGain.addEventListener('input', () => {
  const v = parseFloat(audioGain.value);
  audioGainVal.textContent = v.toFixed(2);
  sendAction({ action: 'set_audio', param: 'gain', value: v });
});

const audioDevice = document.getElementById('audio-device');
audioDevice.addEventListener('change', () => {
  sendAction({ action: 'set_audio', param: 'device', value: audioDevice.value });
});

let knownDevices = '';
function syncAudioDevices(devices, selected) {
  const key = (devices || []).join('|');
  if (key !== knownDevices) {
    knownDevices = key;
    audioDevice.innerHTML = '<option value="">Default</option>' +
      (devices || []).map(d => `<option value="${d.replace(/"/g, '&quot;')}">${d}</option>`).join('');
  }
  if (document.activeElement !== audioDevice) audioDevice.value = selected || '';
}

function syncAudio(a) {
  if (!a) return;
  if (document.activeElement !== audioEnabled) audioEnabled.checked = a.enabled;
  syncAudioDevices(a.devices, a.selected);
  if (document.activeElement !== audioGain) {
    audioGain.value = a.gain;
    audioGainVal.textContent = a.gain.toFixed(2);
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
  if (a.error) {
    audioStatus.textContent = a.error;
    audioStatus.className = 'audio-status error';
  } else if (a.enabled && a.device) {
    audioStatus.textContent = a.device;
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
    <input type="number" class="midi-cc" min="0" max="127" step="1" value="${i + 1}">
    <button class="midi-learn" title="Twist a knob to bind">Learn</button>
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
  if (document.activeElement !== midiEnabled) midiEnabled.checked = m.enabled;
  if (document.activeElement !== midiClockSync) midiClockSync.checked = !!m.clock_sync;
  const clockInd = document.getElementById('midi-clock-indicator');
  if (m.clock_active) {
    clockInd.textContent = '◉ ext clock';
    clockInd.className = 'clock-indicator active';
  } else if (m.clock_sync) {
    clockInd.textContent = 'waiting…';
    clockInd.className = 'clock-indicator';
  } else {
    clockInd.textContent = '';
  }

  (m.slots || []).forEach((slot, i) => {
    const row = midiSlots.children[i];
    if (!row) return;
    const ccInput = row.querySelector('.midi-cc');
    if (document.activeElement !== ccInput) ccInput.value = slot.cc;
    row.querySelector('.lfo-meter-fill').style.width = `${(slot.value * 100).toFixed(1)}%`;
    const learnBtn = row.querySelector('.midi-learn');
    const isLearning = m.learning === i;
    learnBtn.classList.toggle('learning', isLearning);
    learnBtn.textContent = isLearning ? '...' : 'Learn';
  });

  if (m.error) {
    midiStatus.textContent = m.error;
    midiStatus.className = 'audio-status error';
  } else if (m.enabled && m.port) {
    midiStatus.textContent = m.port;
    midiStatus.className = 'audio-status';
  } else {
    midiStatus.textContent = '';
  }
}

// --- XY performance pad ---

const xyPad = document.getElementById('xy-pad');
const xyDot = document.getElementById('xy-pad-dot');
let padTouching = false;
let padLastSend = 0;

function padPosition(x, y) {
  xyDot.style.left = `${(x * 100).toFixed(1)}%`;
  xyDot.style.top = `${((1 - y) * 100).toFixed(1)}%`;
}

function padSend(e) {
  const rect = xyPad.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  // Screen y grows downward; the pad's Y axis grows upward, like a fader.
  const y = Math.min(1, Math.max(0, 1 - (e.clientY - rect.top) / rect.height));
  padPosition(x, y);
  const now = performance.now();
  if (now - padLastSend >= 33) { // ~30Hz
    padLastSend = now;
    sendAction({ action: 'pad', x, y });
  }
  return [x, y];
}

xyPad.addEventListener('pointerdown', (e) => {
  padTouching = true;
  xyPad.setPointerCapture(e.pointerId);
  padSend(e);
  e.preventDefault();
});
xyPad.addEventListener('pointermove', (e) => {
  if (padTouching) padSend(e);
});
xyPad.addEventListener('pointerup', (e) => {
  if (padTouching) {
    // Final position lands exactly where the finger left it.
    const [x, y] = padSend(e);
    sendAction({ action: 'pad', x, y });
  }
  padTouching = false;
});
xyPad.addEventListener('pointercancel', () => { padTouching = false; });

function syncPad(pad) {
  if (!pad || padTouching) return; // never fight the finger that's on it
  padPosition(pad[0], pad[1]);
}

// --- Gyroscope streaming (runs on the device that enables it) ---

const gyroEnabled = document.getElementById('gyro-enabled');
const gyroStatus = document.getElementById('gyro-status');
let gyroLastSend = 0;
let gyroSeenEvent = false;

function gyroHandler(e) {
  gyroSeenEvent = true;
  const now = performance.now();
  if (now - gyroLastSend < 33) return; // ~30Hz
  gyroLastSend = now;
  sendAction({
    action: 'gyro',
    alpha: e.alpha || 0,
    beta: e.beta || 0,
    gamma: e.gamma || 0,
  });
}

gyroEnabled.addEventListener('change', async () => {
  if (gyroEnabled.checked) {
    if (typeof DeviceOrientationEvent === 'undefined') {
      gyroStatus.textContent = 'no orientation sensor in this browser';
      gyroStatus.className = 'audio-status error';
      gyroEnabled.checked = false;
      return;
    }
    // iOS requires an explicit permission request from a user gesture.
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
      try {
        const perm = await DeviceOrientationEvent.requestPermission();
        if (perm !== 'granted') {
          gyroStatus.textContent = 'permission denied';
          gyroStatus.className = 'audio-status error';
          gyroEnabled.checked = false;
          return;
        }
      } catch (err) {
        gyroStatus.textContent = 'sensor needs HTTPS on iOS';
        gyroStatus.className = 'audio-status error';
        gyroEnabled.checked = false;
        return;
      }
    }
    gyroSeenEvent = false;
    window.addEventListener('deviceorientation', gyroHandler);
    gyroStatus.textContent = 'streaming…';
    gyroStatus.className = 'audio-status';
    setTimeout(() => {
      if (gyroEnabled.checked && !gyroSeenEvent) {
        gyroStatus.textContent = 'no sensor data (desktop browser?)';
        gyroStatus.className = 'audio-status error';
      }
    }, 2000);
  } else {
    window.removeEventListener('deviceorientation', gyroHandler);
    gyroStatus.textContent = 'enable on the phone that should steer';
    gyroStatus.className = 'audio-status';
  }
});

// --- Spout output ---

const spoutEnabled = document.getElementById('spout-enabled');
const spoutStatus = document.getElementById('spout-status');

spoutEnabled.addEventListener('change', () => {
  sendAction({ action: 'set_spout', enabled: spoutEnabled.checked });
});

function syncSpout(s) {
  if (!s) return;
  if (document.activeElement !== spoutEnabled) spoutEnabled.checked = s.enabled;
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

document.getElementById('morph-set-a').addEventListener('click', () => {
  sendAction({ action: 'morph_capture', slot: 'a' });
});
document.getElementById('morph-set-b').addEventListener('click', () => {
  sendAction({ action: 'morph_capture', slot: 'b' });
});
document.getElementById('morph-clear').addEventListener('click', () => {
  sendAction({ action: 'morph_clear' });
});
morphT.addEventListener('input', () => {
  sendAction({ action: 'set_morph', value: parseFloat(morphT.value) });
});

function syncMorph(m) {
  if (!m) return;
  if (document.activeElement !== morphT) morphT.value = m.t;
  document.getElementById('morph-set-a').classList.toggle('set', m.has_a);
  document.getElementById('morph-set-b').classList.toggle('set', m.has_b);
  document.getElementById('morph-label-a').classList.toggle('active', m.active && m.t < 0.5);
  document.getElementById('morph-label-b').classList.toggle('active', m.active && m.t >= 0.5);
  if (m.active) {
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
  sendAction({ action: 'toggle_output_window' });
});

function syncOutputWindow(open) {
  if (document.activeElement !== outputWindow) outputWindow.checked = !!open;
  document.getElementById('output-window-hint').textContent =
    open ? 'O or Esc closes' : '';
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
