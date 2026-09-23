/**
 * app.js — screen orchestration and the API client.
 *
 * Flow: setup → (design fetched) → task → (trials posted) → report.
 * The design is generated and stored server-side; the browser only ever
 * executes it. That is what keeps counterbalancing an auditable fact rather
 * than something a front-end bug could quietly unbalance.
 */

import { TrialEngine } from './engine.js';
import { Timing } from './timing.js';
import { renderReport } from './dashboard.js';

const $ = (id) => document.getElementById(id);

const screens = {
  setup: $('screen-setup'),
  task: $('screen-task'),
  working: $('screen-working'),
  results: $('screen-results'),
};

const state = {
  meta: null,
  studies: [],
  preset: 'standard',
  token: null,
  design: null,
  engine: null,
  calibration: null,
};

// Debug handle. Exposes the live session state so a session can be driven or
// inspected from the console, which is how the end-to-end browser test scripts
// the 190-trial task without a human pressing keys 190 times. Read-only in
// practice: nothing in the app reads back from it.
window.__implicitlab = state;

function show(name) {
  Object.entries(screens).forEach(([k, el]) => { el.hidden = k !== name; });
  if (name !== 'task') $('overlay').hidden = true;
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); detail = j.detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

// ---------------------------------------------------------------- setup

async function boot() {
  try {
    const [meta, studies] = await Promise.all([api('/api/meta'), api('/api/studies')]);
    state.meta = meta;
    state.studies = studies.studies;
    renderSetup();
  } catch (err) {
    $('metaGrid').innerHTML =
      `<p style="color:var(--bad)">Could not reach the API: ${escapeHtml(err.message)}</p>`;
  }

  // Calibrate the display while the participant reads the instructions. By the
  // time they press Start the refresh rate is already measured.
  Timing.calibrate().then((c) => { state.calibration = c; renderMeta(); });
}

function renderSetup() {
  const sel = $('studySelect');
  sel.innerHTML = state.studies
    .map((s) => `<option value="${s.id}">${escapeHtml(s.name)} — ${escapeHtml(s.sector)}</option>`)
    .join('');
  sel.addEventListener('change', updateBlurb);
  updateBlurb();

  const dims = state.meta.dimensions || {};
  $('dimSelect').innerHTML = Object.entries(dims)
    .map(([k, v]) => `<option value="${k}">${escapeHtml(v)}</option>`).join('');

  const presets = state.meta.presets || {};
  $('presetChoices').innerHTML = Object.entries(presets).map(([key, p]) => `
    <label class="choice ${key === state.preset ? 'choice--on' : ''}" data-preset="${key}">
      <input type="radio" name="preset" value="${key}" ${key === state.preset ? 'checked' : ''} />
      <span class="choice__body">
        <strong>${escapeHtml(p.label)} · ${escapeHtml(p.minutes)}</strong>
        <span>${escapeHtml(p.note)}</span>
      </span>
    </label>`).join('');
  $('presetChoices').addEventListener('change', (e) => {
    state.preset = e.target.value;
    document.querySelectorAll('.choice').forEach((c) =>
      c.classList.toggle('choice--on', c.dataset.preset === state.preset));
  });

  $('versionTag').textContent = `v${state.meta.version}`;
  renderMeta();

  $('consentBox').addEventListener('change', (e) => { $('startBtn').disabled = !e.target.checked; });
  $('startBtn').addEventListener('click', start);
  $('simulateBtn').addEventListener('click', runDemo);
  $('againBtn').addEventListener('click', () => { show('setup'); });
}

function updateBlurb() {
  const s = state.studies.find((x) => x.id === $('studySelect').value);
  if (!s) return;
  const kind = s.instrument === 'sciat'
    ? 'Single-category test (SC-IAT) — one brand, no comparator.'
    : 'Seven-block IAT — a comparative measure between two brands.';
  const provenance = s.fictitious
    ? 'Brands are invented.'
    : 'Uses real brand names.';
  $('studyBlurb').innerHTML =
    `${escapeHtml(s.blurb)}<br /><strong>${escapeHtml(kind)}</strong> Attribute dimension: ${escapeHtml(s.dimension)}. ${escapeHtml(provenance)}`;
}

function renderMeta() {
  const m = state.meta;
  if (!m) return;
  const c = state.calibration || {};
  const cells = [
    ['Scoring', 'Greenwald 2003 D', 'improved algorithm, built-in error penalty'],
    ['Insight engine', m.llm.configured ? (m.llm.deployment || 'Azure OpenAI') : 'deterministic',
      m.llm.configured ? 'downstream of the statistics, numerically verified' : 'no model configured — rules-based summary'],
    ['Your display', c.refreshHz ? `${c.refreshHz} Hz` : 'measuring…',
      c.onsetUncertaintyMs ? `stimulus onset known to ±${c.onsetUncertaintyMs.toFixed(1)} ms` : 'calibrating from animation frames'],
    ['Sessions collected', String(m.corpus.usable_sessions ?? 0),
      `${m.corpus.trials ?? 0} trials in the database`],
  ];
  $('metaGrid').innerHTML = cells.map(([l, v, s]) => `
    <div class="stat">
      <div class="stat__label">${escapeHtml(l)}</div>
      <div class="stat__value">${escapeHtml(v)}</div>
      <div class="stat__sub">${escapeHtml(s)}</div>
    </div>`).join('');
}

// ---------------------------------------------------------------- running

async function start() {
  const body = {
    study_id: $('studySelect').value,
    preset: state.preset,
    include_calibration: true,
  };
  const seed = $('seedInput').value.trim();
  if (seed !== '') body.seed = Number(seed);
  const a = $('brandA').value.trim();
  if (a) {
    body.custom_brand_a = a;
    body.custom_brand_b = $('brandB').value.trim() || null;
    body.custom_dimension = $('dimSelect').value;
  }

  $('startBtn').disabled = true;
  try {
    const { token, design } = await api('/api/sessions', {
      method: 'POST', body: JSON.stringify(body),
    });
    state.token = token;
    state.design = design;
    await runTask(design);
  } catch (err) {
    alert(`Could not start the session: ${err.message}`);
    $('startBtn').disabled = false;
  }
}

async function runTask(design) {
  show('task');
  const dom = {
    stimulus: $('stimulus'),
    feedback: $('feedback'),
    cueLeft: $('cueLeft'),
    cueRight: $('cueRight'),
    overlay: $('overlay'),
  };

  const engine = new TrialEngine(design, dom, {
    onProgress: (done, total) => {
      $('taskFill').style.width = `${(done / total) * 100}%`;
      $('taskProgress').textContent = `${done} / ${total} trials`;
    },
    onBlockStart: (block) => {
      const label = block.kind === 'motor' ? 'Calibration — movement speed'
        : block.kind === 'reading' ? 'Calibration — reading speed'
        : block.role ? `Block ${block.index} — scored`
        : `Block ${block.index} — practice`;
      $('taskBlockLabel').textContent = label;
    },
  });
  state.engine = engine;

  const onEsc = (ev) => {
    if (ev.key === 'Escape') {
      engine.abort();
      window.removeEventListener('keydown', onEsc, true);
      show('setup');
      $('startBtn').disabled = false;
    }
  };
  window.addEventListener('keydown', onEsc, true);

  const records = await engine.run();
  window.removeEventListener('keydown', onEsc, true);
  // A session abandoned earlier can finish unwinding after a new one has
  // started; it must not reset the screen out from under the live session.
  if (state.engine !== engine) return;
  if (engine.aborted) { show('setup'); $('startBtn').disabled = false; return; }

  await submit(records, engine);
}

async function submit(records, engine) {
  show('working');
  steps([
    ['Applying the latency cut-offs', 'active'],
    ['Computing D', 'idle'],
    ['Bootstrapping the interval', 'idle'],
    ['Writing the summary', 'idle'],
  ]);

  const cal = Timing.summary();
  const dispatches = records.map((r) => r.dispatch_delay_ms).filter((v) => v != null).sort((a, b) => a - b);

  const clientMeta = {
    refresh_hz: cal.refreshHz,
    frame_interval_ms: cal.frameIntervalMs,
    frame_jitter_ms: cal.frameJitterMs,
    onset_uncertainty_ms: cal.onsetUncertaintyMs,
    clock_resolution_note: cal.clockResolutionNote,
    used_event_timestamp: cal.eventTimeStampUsable,
    median_dispatch_delay_ms: dispatches.length ? dispatches[Math.floor(dispatches.length / 2)] : null,
    focus_losses: engine ? engine.focusLosses : 0,
    degraded_onsets: records.filter((r) => r.onset_degraded).length,
    user_agent: navigator.userAgent,
    screen: `${window.screen.width}x${window.screen.height}`,
    simulated: false,
  };

  // Nudge the progress list along so the wait reads as work rather than a hang.
  const timers = [
    setTimeout(() => steps([['Applying the latency cut-offs', 'done'], ['Computing D', 'active'], ['Bootstrapping the interval', 'idle'], ['Writing the summary', 'idle']]), 350),
    setTimeout(() => steps([['Applying the latency cut-offs', 'done'], ['Computing D', 'done'], ['Bootstrapping the interval', 'active'], ['Writing the summary', 'idle']]), 900),
    setTimeout(() => steps([['Applying the latency cut-offs', 'done'], ['Computing D', 'done'], ['Bootstrapping the interval', 'done'], ['Writing the summary', 'active']]), 3200),
  ];

  try {
    const report = await api('/api/results', {
      method: 'POST',
      body: JSON.stringify({
        token: state.token,
        trials: records,
        client_meta: clientMeta,
        consent: true,
        run_llm: true,
      }),
    });
    timers.forEach(clearTimeout);
    // Reveal before rendering. A canvas inside a display:none section has zero
    // measured size, so a chart constructed there is built at 0x0 and only
    // recovers if a ResizeObserver happens to fire afterwards. Showing first
    // makes correct sizing deterministic rather than incidental.
    show('results');
    renderReport(report);
  } catch (err) {
    timers.forEach(clearTimeout);
    alert(`Scoring failed: ${err.message}`);
    show('setup');
    $('startBtn').disabled = false;
  }
}

async function runDemo() {
  show('working');
  steps([
    ['Generating a synthetic respondent', 'active'],
    ['Computing D', 'idle'],
    ['Bootstrapping the interval', 'idle'],
    ['Writing the summary', 'idle'],
  ]);
  const t = setTimeout(() => steps([
    ['Generating a synthetic respondent', 'done'],
    ['Computing D', 'done'],
    ['Bootstrapping the interval', 'active'],
    ['Writing the summary', 'idle'],
  ]), 700);
  try {
    const report = await api('/api/demo', {
      method: 'POST',
      body: JSON.stringify({
        study_id: $('studySelect').value,
        preset: state.preset,
      }),
    });
    clearTimeout(t);
    show('results');
    renderReport(report);
  } catch (err) {
    clearTimeout(t);
    alert(`Could not run the example: ${err.message}`);
    show('setup');
  }
}

function steps(list) {
  $('workSteps').innerHTML = list.map(([label, st]) => `
    <div class="step step--${st === 'done' ? 'done' : st === 'active' ? 'active' : 'idle'}">
      <span class="step__mark">${st === 'done' ? '✓' : st === 'active' ? '▸' : '·'}</span>
      <span>${escapeHtml(label)}</span>
    </div>`).join('');
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

boot();
