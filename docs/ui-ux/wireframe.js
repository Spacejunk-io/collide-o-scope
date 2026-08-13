// Collide-o-Scope Interactive UI Wireframe
// Mirrors the Svelte/WebGL proof-of-concept layout and interactions

const state = {
  videos: [
    'kaleidoscope-loop.mp4',
    'neon-grid.mp4',
    'analog-static.mp4',
    'fractal-zoom.mp4',
    'glitch-burst.mp4',
    'ocean-waves.mp4',
    'tunnel-fly.mp4',
    'retro-bars.mp4',
    'smoke-drift.mp4',
  ],
  layers: [],
  selectedLayerIndex: -1,
  isPlaying: false,
  master: {
    feedback: 0,
    rgbSplit: 0,
    pixelate: 1,
    posterize: 0,
    strobeHz: 0,
    scanLines: false,
    rotation: 0,
    waveAmount: 0,
    waveFreq: 5,
    shift: { mode: 'block', intensity: 0, blockSize: 8, density: 50, speed: 3, direction: 0, spread: 50, repeats: 1 },
    echo: { intensity: 0, blend: 'source-over', count: 3, spacing: 20, direction: 0, speed: 3 },
    morph: { mode: 'wave', intensity: 0, waveFreqX: 5, waveFreqY: 5, waveSpeedX: 1, waveSpeedY: 1, rippleFreq: 8, rippleSpeed: 2, swirlAmount: 0, turbulenceX: 10, turbulenceY: 10 },
    noise: { grainIntensity: 0, grainSize: 1, grainAlgo: 0, colorGrain: false, breatheScale: 0, breatheRotation: 0, breathePosition: 0, vignetteStrength: 0, colorDrift: 0 },
    color: { hue: 0, saturation: 0, brightness: 0, contrast: 0, temperature: 0, tint: 0, invert: false },
  },
  panels: { transport: true, effects: false, chroma: false, color: false, master: true, noise: false, stack: false },
};

function createLayer(filename) {
  return {
    filename,
    type: filename.match(/\.(png|jpg|jpeg|gif|webp)$/i) ? 'image' : 'video',
    speed: 1.0,
    fps: 0,
    feedback: 0,
    isReverse: false,
    isLayerPlaying: true,
    opacity: 1,
    blend: 'source-over',
    rgbSplit: 0,
    pixelate: 1,
    posterize: 0,
    strobeHz: 0,
    scanLines: false,
    rotation: 0,
    chromaKey: { enabled: false, color: [0, 1, 0], tolerance: 0.3, softness: 0.1, spillSuppress: 0.5 },
    color: { hue: 0, saturation: 0, brightness: 0, contrast: 0, temperature: 0, tint: 0, invert: false },
  };
}

function stripExt(filename) {
  return filename.replace(/\.(mp4|webm|mov|png|jpg|jpeg|gif|webp)$/i, '');
}

// Render file list
function renderFileList() {
  const el = document.getElementById('file-list');
  el.innerHTML = '';
  const group = document.createElement('div');
  group.className = 'file-group';

  state.videos.forEach((filename, i) => {
    const inUse = state.layers.some(l => l.filename === filename);
    const btn = document.createElement('button');
    btn.className = 'file-item' + (inUse ? ' in-use' : '');
    btn.innerHTML = `
      <span class="file-thumb"><span class="file-thumb-inner"></span></span>
      <span class="file-name">${stripExt(filename)}</span>
      ${inUse ? '<span class="file-badge">active</span>' : ''}
    `;
    btn.onclick = () => handleFileClick(filename);
    group.appendChild(btn);
  });

  el.appendChild(group);
}

function handleFileClick(filename) {
  const idx = state.layers.findIndex(l => l.filename === filename);
  if (idx !== -1) {
    state.selectedLayerIndex = idx;
  } else {
    state.layers.push(createLayer(filename));
    state.selectedLayerIndex = state.layers.length - 1;
    state.isPlaying = true;
  }
  renderAll();
}

// Render settings panel
function renderSettings() {
  const el = document.getElementById('settings-panel');
  el.innerHTML = '';

  // Active layers section
  if (state.layers.length > 0) {
    const layerGroup = document.createElement('div');
    layerGroup.className = 'file-group';
    layerGroup.innerHTML = '<div class="file-group-label">Active Layers</div>';

    state.layers.forEach((layer, i) => {
      const row = document.createElement('div');
      row.className = 'layer-row';
      const isSelected = i === state.selectedLayerIndex;
      row.innerHTML = `
        <button class="file-item active${isSelected ? ' selected' : ''}" data-layer="${i}">
          <span class="file-thumb"><span class="file-thumb-inner"></span></span>
          <span class="file-name">${stripExt(layer.filename)}</span>
        </button>
        <div class="layer-actions">
          ${i > 0 ? `<button class="layer-btn" data-move-up="${i}" title="Move up">&#9650;</button>` : ''}
          ${i < state.layers.length - 1 ? `<button class="layer-btn" data-move-down="${i}" title="Move down">&#9660;</button>` : ''}
          <button class="layer-btn remove" data-remove="${i}" title="Remove">&times;</button>
        </div>
      `;
      layerGroup.appendChild(row);
    });

    el.appendChild(layerGroup);

    // Attach layer events
    layerGroup.querySelectorAll('[data-layer]').forEach(btn => {
      btn.onclick = () => { state.selectedLayerIndex = parseInt(btn.dataset.layer); renderAll(); };
    });
    layerGroup.querySelectorAll('[data-move-up]').forEach(btn => {
      btn.onclick = (e) => { e.stopPropagation(); moveLayer(parseInt(btn.dataset.moveUp), -1); };
    });
    layerGroup.querySelectorAll('[data-move-down]').forEach(btn => {
      btn.onclick = (e) => { e.stopPropagation(); moveLayer(parseInt(btn.dataset.moveDown), 1); };
    });
    layerGroup.querySelectorAll('[data-remove]').forEach(btn => {
      btn.onclick = (e) => { e.stopPropagation(); removeLayer(parseInt(btn.dataset.remove)); };
    });
  }

  const selectedLayer = state.selectedLayerIndex >= 0 ? state.layers[state.selectedLayerIndex] : null;

  if (selectedLayer) {
    // Per-layer panels
    el.appendChild(createPanel('Transport', 'transport', () => renderTransport(selectedLayer)));
    el.appendChild(createPanel('Effects', 'effects', () => renderEffects(selectedLayer, true)));
    el.appendChild(createPanel('Chroma Key', 'chroma', () => renderChromaKey(selectedLayer)));
    el.appendChild(createPanel('Color', 'color', () => renderColorAdjust(selectedLayer.color)));
  } else {
    // Master panels
    el.appendChild(createPanel('Master', 'master', () => renderMasterEffects()));
    el.appendChild(createPanel('Color', 'color', () => renderColorAdjust(state.master.color)));
    el.appendChild(createPanel('Noise / Analog', 'noise', () => renderNoise()));
  }
}

function createPanel(label, key, renderContent) {
  const panel = document.createElement('div');
  panel.className = 'panel';

  const toggle = document.createElement('button');
  toggle.className = 'panel-toggle';
  toggle.innerHTML = `
    <span class="toggle-arrow${state.panels[key] ? ' open' : ''}">&#9654;</span>
    <span>${label}</span>
  `;
  toggle.onclick = () => { state.panels[key] = !state.panels[key]; renderAll(); };
  panel.appendChild(toggle);

  if (state.panels[key]) {
    const content = document.createElement('div');
    content.className = 'panel-content';
    content.appendChild(renderContent());
    panel.appendChild(content);
  }

  return panel;
}

function renderTransport(layer) {
  const frag = document.createElement('div');
  frag.style.display = 'flex';
  frag.style.flexDirection = 'column';
  frag.style.gap = '5px';

  if (layer.type !== 'image') {
    frag.innerHTML += `
      <div class="transport-row">
        <div class="transport-buttons">
          <button title="Back 5s">&#9198;</button>
          <button class="${state.isPlaying ? 'active' : ''}" title="Play/Pause">${state.isPlaying ? '&#9646;&#9646;' : '&#9654;'}</button>
          <button title="Fwd 5s">&#9197;</button>
        </div>
        <div class="direction-buttons">
          <button class="${!layer.isReverse ? 'active' : ''}">FWD</button>
          <button class="${layer.isReverse ? 'active' : ''}">REV</button>
        </div>
      </div>
      ${sliderRow('Speed', 0.01, 4, 0.01, layer.speed, layer.speed.toFixed(2) + 'x')}
      ${sliderRow('FPS', 0, 30, 1, layer.fps, layer.fps === 0 ? 'Full' : layer.fps)}
    `;
  }

  frag.innerHTML += sliderRow('Trail', 0, 99, 1, layer.feedback * 100, Math.round(layer.feedback * 100) + '%');
  return frag;
}

function renderEffects(source, showBlend) {
  const frag = document.createElement('div');
  frag.style.display = 'flex';
  frag.style.flexDirection = 'column';
  frag.style.gap = '5px';

  let html = '';
  if (showBlend) {
    html += sliderRow('Opac', 0, 100, 1, source.opacity * 100, Math.round(source.opacity * 100) + '%');
    html += `
      <div class="fx-row">
        <span class="fx-label">Blend</span>
        <select>
          <option>normal</option><option>screen</option><option>multiply</option>
          <option>difference</option><option>overlay</option><option>lighten</option>
          <option>darken</option><option>color-dodge</option><option>exclusion</option>
        </select>
      </div>
    `;
  }
  html += sliderRow('RGB', 0, 30, 1, source.rgbSplit, source.rgbSplit || '-');
  html += sliderRow('Pixel', 1, 32, 1, source.pixelate, source.pixelate);
  html += sliderRow('Poster', 0, 16, 1, source.posterize, source.posterize > 1 ? source.posterize : '-');
  html += sliderRow('Strobe', 0, 20, 1, source.strobeHz, source.strobeHz > 0 ? source.strobeHz : '-');
  html += `
    <div class="fx-row">
      <span class="fx-label">Scan</span>
      <button class="${source.scanLines ? 'active' : ''}">${source.scanLines ? 'ON' : 'OFF'}</button>
    </div>
  `;
  html += sliderRow('Rotate', 0, 360, 1, source.rotation || 0, (source.rotation || 0) + '\u00B0');

  frag.innerHTML = html;
  return frag;
}

function renderChromaKey(layer) {
  const frag = document.createElement('div');
  frag.style.display = 'flex';
  frag.style.flexDirection = 'column';
  frag.style.gap = '5px';

  const ck = layer.chromaKey;
  let html = `
    <div class="section-header">
      <span class="section-label">Chroma Key</span>
      <label class="toggle-label">
        <input type="checkbox" ${ck.enabled ? 'checked' : ''}>
        <span>${ck.enabled ? 'ON' : 'OFF'}</span>
      </label>
    </div>
  `;

  if (ck.enabled) {
    html += `
      <div class="fx-row">
        <span class="fx-label">Color</span>
        <div class="color-presets">
          <button class="color-btn" style="background:#00ff00" title="Green"></button>
          <button class="color-btn" style="background:#0000ff" title="Blue"></button>
          <button class="color-btn" style="background:#000000" title="Black"></button>
          <button class="color-btn" style="background:#ffffff" title="White"></button>
          <input type="color" value="#00ff00" style="width:20px;height:16px;border:none;padding:0;cursor:pointer;background:none;">
        </div>
      </div>
    `;
    html += sliderRow('Tol', 0.05, 0.8, 0.01, ck.tolerance, ck.tolerance.toFixed(2));
    html += sliderRow('Soft', 0, 0.5, 0.01, ck.softness, ck.softness.toFixed(2));
    html += sliderRow('Spill', 0, 1, 0.05, ck.spillSuppress, ck.spillSuppress.toFixed(1));
  }

  frag.innerHTML = html;

  // Toggle chroma key on checkbox change
  const checkbox = frag.querySelector('input[type="checkbox"]');
  if (checkbox) {
    checkbox.onchange = () => { layer.chromaKey.enabled = checkbox.checked; renderAll(); };
  }

  return frag;
}

function renderColorAdjust(color) {
  const frag = document.createElement('div');
  frag.style.display = 'flex';
  frag.style.flexDirection = 'column';
  frag.style.gap = '5px';

  let html = `
    <div class="section-header">
      <span class="section-label">Color</span>
      <button class="section-btn">Reset</button>
    </div>
  `;
  html += sliderRow('Hue', 0, 360, 1, color.hue, color.hue > 0 ? color.hue + '\u00B0' : '-');
  html += sliderRow('Sat', -100, 100, 1, color.saturation, color.saturation !== 0 ? color.saturation : '-');
  html += sliderRow('Brt', -100, 100, 1, color.brightness, color.brightness !== 0 ? color.brightness : '-');
  html += sliderRow('Cntr', -100, 100, 1, color.contrast, color.contrast !== 0 ? color.contrast : '-');
  html += sliderRow('Temp', -100, 100, 1, color.temperature, color.temperature !== 0 ? color.temperature : '-');
  html += sliderRow('Tint', -100, 100, 1, color.tint, color.tint !== 0 ? color.tint : '-');
  html += `
    <div class="fx-row">
      <span class="fx-label">Inv</span>
      <label class="toggle-label">
        <input type="checkbox" ${color.invert ? 'checked' : ''}>
        <span>${color.invert ? 'ON' : 'OFF'}</span>
      </label>
    </div>
  `;

  frag.innerHTML = html;
  return frag;
}

function renderMasterEffects() {
  const frag = document.createElement('div');
  frag.style.display = 'flex';
  frag.style.flexDirection = 'column';
  frag.style.gap = '5px';

  const m = state.master;
  let html = '';
  html += sliderRow('Trail', 0, 99, 1, m.feedback * 100, Math.round(m.feedback * 100) + '%');
  html += sliderRow('RGB', 0, 30, 1, m.rgbSplit, m.rgbSplit || '-');
  html += sliderRow('Pixel', 1, 32, 1, m.pixelate, m.pixelate);
  html += sliderRow('Poster', 0, 16, 1, m.posterize, m.posterize > 1 ? m.posterize : '-');
  html += sliderRow('Strobe', 0, 20, 1, m.strobeHz, m.strobeHz > 0 ? m.strobeHz : '-');
  html += `
    <div class="fx-row">
      <span class="fx-label">Scan</span>
      <button class="${m.scanLines ? 'active' : ''}">${m.scanLines ? 'ON' : 'OFF'}</button>
    </div>
  `;
  html += sliderRow('Wave', 0, 50, 1, m.waveAmount, m.waveAmount > 0 ? m.waveAmount : '-');

  // Shift section
  html += `
    <div style="border-top:1px solid #333;padding-top:8px;margin-top:4px;">
      <div class="section-header">
        <span class="section-label">Shift</span>
        <div class="section-buttons">
          <button class="section-btn">Dice</button>
          <button class="section-btn">Off</button>
        </div>
      </div>
      ${sliderRow('Mode', 0, 0, 1, 0, 'block')}
      ${sliderRow('Amt', 0, 100, 1, m.shift.intensity, m.shift.intensity > 0 ? m.shift.intensity : '-')}
    </div>
  `;

  // Echo section
  html += `
    <div style="border-top:1px solid #333;padding-top:8px;margin-top:4px;">
      <div class="section-header">
        <span class="section-label">Echo</span>
        <div class="section-buttons">
          <button class="section-btn">Off</button>
        </div>
      </div>
      ${sliderRow('Amt', 0, 100, 1, m.echo.intensity, m.echo.intensity > 0 ? m.echo.intensity : '-')}
    </div>
  `;

  // Morph section
  html += `
    <div style="border-top:1px solid #333;padding-top:8px;margin-top:4px;">
      <div class="section-header">
        <span class="section-label">Morph</span>
        <div class="section-buttons">
          <button class="section-btn">Dice</button>
          <button class="section-btn">Off</button>
        </div>
      </div>
      ${sliderRow('Mode', 0, 0, 1, 0, 'wave')}
      ${sliderRow('Amt', 0, 100, 1, m.morph.intensity, m.morph.intensity > 0 ? m.morph.intensity : '-')}
    </div>
  `;

  frag.innerHTML = html;
  return frag;
}

function renderNoise() {
  const frag = document.createElement('div');
  frag.style.display = 'flex';
  frag.style.flexDirection = 'column';
  frag.style.gap = '5px';

  const n = state.master.noise;
  let html = `
    <div class="section-header">
      <span class="section-label">Noise / Analog</span>
      <button class="section-btn">Reset</button>
    </div>
  `;
  html += sliderRow('Grain', 0, 30, 0.5, n.grainIntensity * 100, n.grainIntensity > 0 ? (n.grainIntensity * 100).toFixed(0) : '-');
  html += sliderRow('Brth', 0, 5, 0.2, n.breatheScale * 100, n.breatheScale > 0 ? (n.breatheScale * 100).toFixed(1) : '-');
  html += sliderRow('Vign', 0, 1.5, 0.05, n.vignetteStrength, n.vignetteStrength > 0 ? n.vignetteStrength.toFixed(1) : '-');
  html += sliderRow('CDrft', 0, 20, 1, n.colorDrift * 1000, n.colorDrift > 0 ? (n.colorDrift * 1000).toFixed(0) : '-');

  frag.innerHTML = html;
  return frag;
}

// Helper: slider row HTML
function sliderRow(label, min, max, step, value, display) {
  return `
    <div class="fx-row">
      <span class="fx-label">${label}</span>
      <input type="range" min="${min}" max="${max}" step="${step}" value="${value}">
      <span class="fx-val">${display}</span>
    </div>
  `;
}

// Layer management
function moveLayer(idx, dir) {
  const newIdx = idx + dir;
  if (newIdx < 0 || newIdx >= state.layers.length) return;
  [state.layers[idx], state.layers[newIdx]] = [state.layers[newIdx], state.layers[idx]];
  if (state.selectedLayerIndex === idx) state.selectedLayerIndex = newIdx;
  else if (state.selectedLayerIndex === newIdx) state.selectedLayerIndex = idx;
  renderAll();
}

function removeLayer(idx) {
  state.layers.splice(idx, 1);
  if (state.selectedLayerIndex >= state.layers.length) {
    state.selectedLayerIndex = state.layers.length - 1;
  }
  if (state.layers.length === 0) state.isPlaying = false;
  renderAll();
}

// Canvas preview (animated placeholder)
let animFrame;
function renderCanvas() {
  const canvas = document.getElementById('preview-canvas');
  const msg = document.getElementById('no-video-msg');

  if (state.layers.length === 0) {
    canvas.style.display = 'none';
    msg.style.display = '';
    if (animFrame) cancelAnimationFrame(animFrame);
    return;
  }

  canvas.style.display = 'block';
  msg.style.display = 'none';

  const ctx = canvas.getContext('2d');
  let t = 0;

  function draw() {
    t += 0.02;
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, 1280, 720);

    // Draw animated placeholder per layer
    state.layers.forEach((layer, i) => {
      const hue = (i * 60 + t * 30) % 360;
      ctx.save();
      ctx.globalAlpha = layer.opacity;
      ctx.translate(640, 360);
      ctx.rotate((layer.rotation || 0) * Math.PI / 180);

      // Animated rectangles
      for (let j = 0; j < 5; j++) {
        const s = 100 + j * 60;
        ctx.strokeStyle = `hsl(${(hue + j * 30) % 360}, 70%, 50%)`;
        ctx.lineWidth = 2;
        ctx.strokeRect(-s / 2 + Math.sin(t + j) * 20, -s / 2 + Math.cos(t + j) * 20, s, s);
      }

      ctx.restore();
    });

    if (state.isPlaying) {
      animFrame = requestAnimationFrame(draw);
    }
  }

  if (animFrame) cancelAnimationFrame(animFrame);
  if (state.isPlaying) draw();
  else {
    // Draw one static frame
    draw();
    cancelAnimationFrame(animFrame);
  }
}

// Deselect layer on center click
document.getElementById('col-center').onclick = (e) => {
  if (e.target.id === 'col-center' || e.target.classList.contains('canvas-placeholder') || e.target.id === 'no-video-msg') {
    state.selectedLayerIndex = -1;
    renderAll();
  }
};

// Kebab menu
document.getElementById('kebab-btn').onclick = (e) => {
  e.stopPropagation();
  document.getElementById('kebab-dropdown').classList.toggle('open');
};
document.addEventListener('click', () => {
  document.getElementById('kebab-dropdown').classList.remove('open');
});

// Resize handles with collapse support
const COLLAPSE_THRESHOLD = 60;
const DEFAULT_WIDTH = { left: 260, right: 240 };
const collapsedState = { left: false, right: false };

function initResize(handleId, colId, direction) {
  const handle = document.getElementById(handleId);

  // Add expand tab to handle
  const tab = document.createElement('button');
  tab.className = 'expand-tab';
  tab.innerHTML = direction === 'left' ? '&#9654;' : '&#9664;';
  tab.onclick = (e) => { e.stopPropagation(); expandPanel(colId, direction); };
  handle.appendChild(tab);

  handle.addEventListener('mousedown', (e) => {
    if (e.target === tab) return;
    e.preventDefault();
    const startX = e.clientX;
    const col = document.getElementById(colId);
    const wasCollapsed = collapsedState[direction];
    const startWidth = wasCollapsed ? 0 : col.offsetWidth;

    // Disable transition during drag
    col.style.transition = 'none';
    if (wasCollapsed) {
      col.classList.remove('collapsed');
      col.style.width = '0px';
    }

    document.body.style.userSelect = 'none';

    function onMove(ev) {
      const delta = direction === 'left' ? (ev.clientX - startX) : (startX - ev.clientX);
      const newWidth = startWidth + delta;
      col.style.width = Math.max(0, Math.min(500, newWidth)) + 'px';
    }

    function onUp(ev) {
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);

      col.style.transition = '';
      const finalWidth = col.offsetWidth;

      if (finalWidth < COLLAPSE_THRESHOLD) {
        // Collapse
        col.classList.add('collapsed');
        col.style.width = '';
        collapsedState[direction] = true;
        handle.classList.add('show-tab');
      } else {
        // Keep open
        col.classList.remove('collapsed');
        col.style.width = Math.max(180, Math.min(500, finalWidth)) + 'px';
        collapsedState[direction] = false;
        handle.classList.remove('show-tab');
      }
    }

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  });
}

function expandPanel(colId, direction) {
  const col = document.getElementById(colId);
  const handle = document.getElementById(direction === 'left' ? 'resize-left' : 'resize-right');
  col.classList.remove('collapsed');
  col.style.width = DEFAULT_WIDTH[direction] + 'px';
  collapsedState[direction] = false;
  handle.classList.remove('show-tab');
}

initResize('resize-left', 'col-left', 'left');
initResize('resize-right', 'col-right', 'right');

// Keyboard shortcuts (matching the app)
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

  const layer = state.selectedLayerIndex >= 0 ? state.layers[state.selectedLayerIndex] : null;

  switch (e.key) {
    case ' ':
      e.preventDefault();
      state.isPlaying = !state.isPlaying;
      renderAll();
      break;
    case 'Tab':
      e.preventDefault();
      if (state.layers.length > 0) {
        if (e.shiftKey) {
          state.selectedLayerIndex = (state.selectedLayerIndex - 1 + state.layers.length) % state.layers.length;
        } else {
          state.selectedLayerIndex = (state.selectedLayerIndex + 1) % state.layers.length;
        }
        renderAll();
      }
      break;
    case 'x': case 'X':
      if (state.layers.length > 0) removeLayer(state.layers.length - 1);
      break;
    default:
      const num = parseInt(e.key);
      if (num >= 1 && num <= 9 && num <= state.videos.length) {
        handleFileClick(state.videos[num - 1]);
      }
      break;
  }
});

// Render everything
function renderAll() {
  renderFileList();
  renderSettings();
  renderCanvas();
}

// Initial render
renderAll();
