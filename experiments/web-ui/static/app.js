// collide-o-scope WebSocket control panel

const ws = new WebSocket(`ws://${location.host}/ws`);

// --- WebSocket ---

ws.onopen = () => console.log('[ws] connected');
ws.onclose = () => console.log('[ws] disconnected');

ws.onmessage = (e) => {
  try {
    const state = JSON.parse(e.data);
    syncUI(state);
  } catch (err) {
    console.warn('[ws] bad message:', e.data);
  }
};

function send(param, value) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ param, value }));
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
    slider.value = min; // will be synced on connect

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      send(param, v);
    });
  }

  if (checkbox) {
    checkbox.addEventListener('change', () => {
      send(param, checkbox.checked);
    });
  }

  if (select) {
    select.addEventListener('change', () => {
      send(param, parseInt(select.value));
    });
  }
});

// --- Tabs ---

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// --- Sync UI from server state ---

function syncUI(state) {
  for (const [param, value] of Object.entries(state)) {
    const row = document.querySelector(`.param-row[data-param="${param}"]`);
    if (!row) continue;

    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const select = row.querySelector('select');

    if (slider && valueEl) {
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

function formatValue(v, min, max, step) {
  if (step >= 1) return v.toFixed(0);
  if (max <= 1 && min >= -1) return v.toFixed(2);
  if (step >= 0.01) return v.toFixed(1);
  return v.toFixed(3);
}
