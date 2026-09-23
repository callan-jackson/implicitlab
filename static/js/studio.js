/**
 * studio.js — the cohort analysis dashboard.
 *
 * One rule shapes this file: **the server computes, the page renders.** Every
 * number shown here comes from GET /analysis, which is the same function the
 * Excel and PowerPoint exports call with the same parameters. The page never
 * derives a statistic of its own, so the screen and the deliverable cannot
 * drift apart.
 *
 * Both statistical methods arrive in every response. The method switch is
 * therefore purely a view change — it re-renders from the report already held
 * in memory, which is what makes it instant and what guarantees that flipping
 * back and forth shows the same numbers every time.
 *
 * Colour carries exactly one meaning per chart, and the same meaning across
 * charts: blue = leans towards the first brand, orange = leans towards the
 * second. Pairings in the RT distributions get their own pair (aqua / violet)
 * precisely so they cannot be mistaken for a brand direction. Nothing is
 * encoded by colour alone: sign, marker fill and text labels repeat it.
 */

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- palette
// Validated with the dataviz palette checker against the #14161e card surface:
// blue/orange adjacent CVD ΔE 26.8, aqua/violet 17.3; all ≥ 3:1 contrast.
const POS = '#3987e5';      // leans target A
const NEG = '#d95926';      // leans target B
const MID = '#383835';      // neutral midpoint
const CON = '#199e70';      // congruent pairing
const INC = '#9085e9';      // incongruent pairing
const CARD = '#14161e';
const ACCENT = '#9085e9';
const GRID = 'rgba(255,255,255,0.06)';
const TICK = '#8a90a0';
const INK = '#f2f3f7';

const DEFAULTS = { lower_ms: 0, upper_ms: 10000, fast_ms: 300, fast_limit: 0.10, min_accuracy: 0.75 };

const state = {
  batches: [],
  batchId: 'demo',
  batch: null,
  report: null,
  method: 'permutation',
  segmentBy: null,
  distAttr: null,
  scale: 'raw',
  params: { ...DEFAULTS },
  inflight: null,
  timer: null,
  participants: null,
  participantsKey: null,
};
window.__studio = state;

const charts = {};
function destroy(key) { if (charts[key]) { charts[key].destroy(); delete charts[key]; } }

// ---------------------------------------------------------------- helpers

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const fmtD = (v, dp = 2) => (v == null ? '—' : `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(dp)}`);
const fmtInt = (v) => (v == null ? '—' : Number(v).toLocaleString('en-GB'));
const fmtPct = (v, dp = 0) => (v == null ? '—' : `${(v * 100).toFixed(dp)}%`);
function fmtP(p) {
  if (p == null) return '—';
  if (p < 0.001) return '< .001';
  return p.toFixed(3).replace(/^0/, '');
}
const brand = (s) => (s && s === s.toUpperCase() ? s.charAt(0) + s.slice(1).toLowerCase() : (s || ''));

function mix(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ch = (p, s) => (p >> s) & 255;
  const c = [16, 8, 0].map((s) => Math.round(ch(pa, s) + (ch(pb, s) - ch(pa, s)) * t));
  return `rgb(${c.join(',')})`;
}
function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
/** Diverging fill centred on zero; saturates at |D| = 0.6. */
function diverge(d) {
  if (d == null) return MID;
  const t = Math.min(1, Math.abs(d) / 0.6);
  return mix(MID, d >= 0 ? POS : NEG, 0.15 + 0.85 * t);
}
const dirColour = (d) => (d == null ? TICK : d >= 0 ? POS : NEG);

function segLabel(key) {
  const s = (state.batch?.segments || {})[key];
  return s ? s.label : key;
}
function attrInfo(key) { return (state.report?.batch?.attributes || {})[key] || {}; }

// ---------------------------------------------------------------- tooltip

const tip = $('tip');
function showTip(html, ev) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() { tip.hidden = true; }

// ---------------------------------------------------------------- API

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); detail = j.detail || detail; } catch { /* not JSON */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function paramQuery() {
  const q = new URLSearchParams();
  const p = state.params;
  q.set('lower_ms', p.lower_ms);
  q.set('upper_ms', p.upper_ms);
  q.set('fast_ms', p.fast_ms);
  q.set('fast_limit', p.fast_limit.toFixed(2));
  q.set('min_accuracy', p.min_accuracy.toFixed(2));
  if (state.segmentBy) q.set('segment_by', state.segmentBy);
  return q;
}

// ---------------------------------------------------------------- boot

async function boot() {
  readHash();
  wireControls();
  try {
    const { batches } = await api('/api/cohort/batches');
    state.batches = batches;
    if (!batches.some((b) => b.id === state.batchId)) state.batchId = 'demo';
    renderBatchSelect();
    await loadBatch(state.batchId, { keepParams: true });
  } catch (err) {
    showError(`Could not reach the cohort API: ${err.message}`);
  }
}

function renderBatchSelect() {
  $('batchSelect').innerHTML = state.batches.map((b) => {
    const when = b.built_in ? 'synthetic demo' : new Date(b.created_at).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' });
    return `<option value="${esc(b.id)}" ${b.id === state.batchId ? 'selected' : ''}>
      ${esc(b.name)} — n = ${fmtInt(b.n_participants)} · ${esc(when)}</option>`;
  }).join('');
}

async function loadBatch(id, { keepParams = false } = {}) {
  state.batchId = id;
  state.participants = null;
  const { batch, ingest_report: report } = await api(`/api/cohort/batches/${encodeURIComponent(id)}`);
  state.batch = batch;
  if (!keepParams) state.params = { ...DEFAULTS };

  const segs = Object.keys(batch.segments || {});
  if (!segs.includes(state.segmentBy)) state.segmentBy = segs[0] || null;
  const attrs = Object.keys(batch.attributes || {});
  if (!attrs.includes(state.distAttr)) state.distAttr = attrs[0] || null;

  $('batchTitle').textContent = batch.name || 'Batch';
  $('batchNote').textContent = batch.note || '';
  $('rawLink').href = `/api/cohort/batches/${encodeURIComponent(id)}/raw.csv`;

  $('segmentSelect').innerHTML = segs.length
    ? segs.map((k) => `<option value="${esc(k)}" ${k === state.segmentBy ? 'selected' : ''}>${esc(segLabel(k))}</option>`).join('')
    : '<option value="">No participant variables</option>';
  $('segmentSelect').disabled = !segs.length;
  $('distAttr').innerHTML = attrs.map((k) =>
    `<option value="${esc(k)}" ${k === state.distAttr ? 'selected' : ''}>${esc(batch.attributes[k].label || k)}</option>`).join('');

  renderIngest(report);
  syncControls();
  await analyse();
}

// ---------------------------------------------------------------- ingest

function renderIngest(r) {
  if (!r) return;
  const rej = r.rows_rejected || 0;
  $('ingestSummary').textContent =
    `${fmtInt(r.rows_accepted)} of ${fmtInt(r.rows_read)} rows accepted · ${fmtInt(r.participants)} participants`
    + (rej ? ` · ${fmtInt(rej)} rejected` : '');

  const reasons = Object.entries(r.rejection_reasons || {});
  const skipped = Object.entries(r.skipped_columns || {});
  const renamed = Object.entries(r.columns_renamed || {});
  const segs = (r.segment_variables || []).map((k) => `<span class="tag">${esc(segLabel(k))}</span>`).join(' ');
  const cells = [
    ['Rows read', fmtInt(r.rows_read)],
    ['Accepted', fmtInt(r.rows_accepted)],
    ['Rejected', fmtInt(rej)],
    ['Duplicates removed', fmtInt(r.duplicates_removed)],
    ['Participants', fmtInt(r.participants)],
    ['Attributes', (r.attributes || []).length],
  ];
  $('ingestBody').innerHTML = `
    <div class="ingest-grid">${cells.map(([l, v]) => `
      <div class="stat stat--sm"><div class="stat__label">${esc(l)}</div><div class="stat__value">${esc(v)}</div></div>`).join('')}
    </div>
    <p class="ingest-line"><strong>Segment variables detected:</strong> ${segs || '<span class="muted">none</span>'}</p>
    ${reasons.length ? `<p class="ingest-line"><strong>Rejected rows by reason:</strong></p>
      <ul class="ingest-list">${reasons.map(([k, v]) => `<li>${esc(k)} — <span class="mono">${fmtInt(v)}</span></li>`).join('')}</ul>` : ''}
    ${skipped.length ? `<p class="ingest-line"><strong>Columns not used for segmentation:</strong></p>
      <ul class="ingest-list">${skipped.map(([k, v]) => `<li><code>${esc(k)}</code> — ${esc(v)}</li>`).join('')}</ul>` : ''}
    ${renamed.length ? `<p class="ingest-line"><strong>Columns renamed:</strong> ${renamed.map(([a, b]) => `<code>${esc(a)}</code> → <code>${esc(b)}</code>`).join(', ')}</p>` : ''}
    ${(r.warnings || []).map((w) => `<div class="banner banner--warn"><span class="banner__icon">!</span><span>${esc(w)}</span></div>`).join('')}
  `;
}

async function upload(file) {
  const status = $('uploadStatus');
  status.innerHTML = banner('info', '⇪', `Uploading and validating <strong>${esc(file.name)}</strong> (${(file.size / 1024 / 1024).toFixed(1)} MB)…`);
  $('uploadLabel').classList.add('btn--busy');
  try {
    const q = new URLSearchParams({ filename: file.name, name: file.name.replace(/\.[^.]+$/, '') });
    const res = await api(`/api/cohort/batches?${q}`, { method: 'POST', body: file });
    const { batches } = await api('/api/cohort/batches');
    state.batches = batches;
    state.batchId = res.batch.id;
    renderBatchSelect();
    const r = res.ingest_report;
    status.innerHTML = banner('good', '✓',
      `<strong>Ingested ${esc(file.name)}.</strong> ${fmtInt(r.rows_accepted)} of ${fmtInt(r.rows_read)} rows accepted, `
      + `${fmtInt(r.participants)} participants, ${(r.segment_variables || []).length} segment variable(s) detected. `
      + `The ingest report below lists anything that was rejected and why.`);
    $('ingestDisclosure').open = true;
    await loadBatch(res.batch.id);
  } catch (err) {
    status.innerHTML = banner('bad', '✗', `<strong>Upload rejected.</strong> ${esc(err.message)}`);
  } finally {
    $('uploadLabel').classList.remove('btn--busy');
    $('uploadInput').value = '';
  }
}

function banner(kind, icon, html) {
  return `<div class="banner banner--${kind}"><span class="banner__icon">${icon}</span><span>${html}</span></div>`;
}
function showError(msg) { $('errorBox').innerHTML = msg ? banner('bad', '✗', esc(msg)) : ''; }

// ---------------------------------------------------------------- controls

function wireControls() {
  $('batchSelect').addEventListener('change', (e) => loadBatch(e.target.value).catch((err) => showError(err.message)));
  $('uploadInput').addEventListener('change', (e) => { if (e.target.files[0]) upload(e.target.files[0]); });

  const slider = (id, key, scale = 1) => $(id).addEventListener('input', (e) => {
    state.params[key] = Number(e.target.value) * scale;
    syncControls();
    schedule();
  });
  slider('lowerSlider', 'lower_ms');
  slider('upperSlider', 'upper_ms');
  slider('fastSlider', 'fast_ms');
  slider('fastLimitSlider', 'fast_limit', 0.01);
  slider('accSlider', 'min_accuracy', 0.01);

  $('segmentSelect').addEventListener('change', (e) => { state.segmentBy = e.target.value || null; syncControls(); schedule(0); });
  $('resetBtn').addEventListener('click', () => { state.params = { ...DEFAULTS }; syncControls(); schedule(0); });

  document.querySelectorAll('[data-method]').forEach((b) => b.addEventListener('click', () => {
    state.method = b.dataset.method;
    syncControls();
    renderMethodDependent();
  }));
  document.querySelectorAll('[data-scale]').forEach((b) => b.addEventListener('click', () => {
    state.scale = b.dataset.scale;
    document.querySelectorAll('[data-scale]').forEach((x) => x.classList.toggle('seg__btn--on', x === b));
    renderDistribution();
  }));
  $('distAttr').addEventListener('change', (e) => { state.distAttr = e.target.value; renderDistribution(); });
  $('auditFilter').addEventListener('change', renderAudit);
  $('xlsxBtn').addEventListener('click', (e) => { e.preventDefault(); download($('xlsxBtn'), 'Excel workbook'); });
  $('pptxBtn').addEventListener('click', (e) => { e.preventDefault(); download($('pptxBtn'), 'PowerPoint deck'); });
  $('auditDisclosure').addEventListener('toggle', () => { if ($('auditDisclosure').open) loadParticipants(); });
}

function syncControls() {
  const p = state.params;
  $('lowerSlider').value = p.lower_ms;
  $('upperSlider').value = p.upper_ms;
  $('fastSlider').value = p.fast_ms;
  $('fastLimitSlider').value = Math.round(p.fast_limit * 100);
  $('accSlider').value = Math.round(p.min_accuracy * 100);

  $('lowerOut').textContent = p.lower_ms > 0 ? `${fmtInt(p.lower_ms)} ms` : 'off';
  $('upperOut').textContent = `${fmtInt(p.upper_ms)} ms`;
  $('fastOut').innerHTML = `&gt;${Math.round(p.fast_limit * 100)}% under ${fmtInt(p.fast_ms)} ms`;
  $('accOut').textContent = `${Math.round(p.min_accuracy * 100)}%`;

  const hint = $('lowerHint');
  if (p.lower_ms > 0) {
    hint.textContent = 'Greenwald et al. (2003) recommend no lower trim: fast trials are signal, and the participant screen handles fast responders.';
    hint.classList.add('ctl__hint--warn');
  } else {
    hint.textContent = 'Off — the Greenwald (2003) default.';
    hint.classList.remove('ctl__hint--warn');
  }
  const seg = (state.batch?.segments || {})[state.segmentBy];
  $('segmentNote').textContent = seg?.note || '';

  document.querySelectorAll('[data-method]').forEach((b) => {
    const on = b.dataset.method === state.method;
    b.classList.toggle('seg__btn--on', on);
    b.setAttribute('aria-checked', String(on));
  });
  $('methodHint').textContent = state.method === 'permutation'
    ? 'Distribution-free: sign-flip and label-shuffle tests on D.'
    : 't-tests on per-respondent log-RT differences (Welch between segments).';

  const isDefault = Object.entries(DEFAULTS).every(([k, v]) => Math.abs(p[k] - v) < 1e-9);
  $('resetBtn').disabled = isDefault;
  updateExportLinks();
  writeHash();
}

function updateExportLinks() {
  const q = paramQuery();
  q.set('method', state.method);
  const base = `/api/cohort/batches/${encodeURIComponent(state.batchId)}`;
  $('xlsxBtn').href = `${base}/export.xlsx?${q}`;
  $('pptxBtn').href = `${base}/export.pptx?${q}`;
}

/**
 * Exports are built on demand and take a few seconds for a full panel, so the
 * button fetches the file itself: it can show that work is happening, and a
 * server-side failure becomes a readable message instead of a JSON error page.
 * The href stays a real URL so "copy link" and middle-click still work.
 */
async function download(btn, what) {
  if (btn.classList.contains('btn--busy')) return;
  const label = btn.textContent;
  btn.classList.add('btn--busy');
  btn.textContent = `Building ${what.toLowerCase()}…`;
  // On the free-tier host a full-panel workbook takes tens of seconds (about
  // three on a laptop). A counting button reads as work; a frozen one as a hang.
  const started = Date.now();
  const tick = setInterval(() => {
    const secs = Math.round((Date.now() - started) / 1000);
    btn.textContent = `Building ${what.toLowerCase()}… ${secs}s`;
    if (secs === 6) {
      $('uploadStatus').innerHTML = banner('info', 'ⓘ',
        `Building the ${esc(what.toLowerCase())} from all ${state.report ? state.report.sample.recruited : ''} `
        + 'respondents. The hosted demo runs on a shared free-tier CPU, so this can take up to a minute; '
        + 'on a normal machine it takes a few seconds.');
    }
  }, 1000);
  try {
    const res = await fetch(btn.href);
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
      throw new Error(detail);
    }
    const blob = await res.blob();
    const cd = res.headers.get('Content-Disposition') || '';
    const name = (cd.match(/filename="([^"]+)"/) || [])[1] || `implicitlab_export`;
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), { href: url, download: name });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    $('uploadStatus').innerHTML = banner('good', '✓',
      `<strong>${esc(name)}</strong> — ${(blob.size / 1024).toFixed(0)} KB, built from exactly the parameters shown `
      + `(${state.method} tests${state.segmentBy ? `, segmented by ${esc(segLabel(state.segmentBy).toLowerCase())}` : ''}).`);
  } catch (err) {
    $('uploadStatus').innerHTML = banner('bad', '✗', `<strong>Could not build the ${esc(what)}.</strong> ${esc(err.message)}`);
  } finally {
    clearInterval(tick);
    btn.classList.remove('btn--busy');
    btn.textContent = label;
  }
}

function writeHash() {
  const q = paramQuery();
  q.set('method', state.method);
  q.set('batch', state.batchId);
  history.replaceState(null, '', `#${q}`);
}
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(DEFAULTS)) {
    if (q.has(k) && Number.isFinite(Number(q.get(k)))) state.params[k] = Number(q.get(k));
  }
  if (q.get('method') === 'parametric') state.method = 'parametric';
  if (q.get('segment_by')) state.segmentBy = q.get('segment_by');
  if (q.get('batch')) state.batchId = q.get('batch');
}

function schedule(delay = 250) {
  clearTimeout(state.timer);
  state.timer = setTimeout(() => analyse().catch(() => {}), delay);
}

// ---------------------------------------------------------------- analysis

async function analyse() {
  if (state.inflight) state.inflight.abort();
  const ctrl = new AbortController();
  state.inflight = ctrl;
  const calc = $('calcStatus');
  calc.className = 'calc calc--busy';
  calc.textContent = 'recomputing…';
  const t0 = performance.now();
  try {
    const report = await api(
      `/api/cohort/batches/${encodeURIComponent(state.batchId)}/analysis?${paramQuery()}`,
      { signal: ctrl.signal },
    );
    if (ctrl.signal.aborted) return;
    state.report = report;
    state.segmentBy = report.segment_by;
    showError('');
    renderAll();
    const rt = Math.round(performance.now() - t0);
    calc.className = 'calc calc--ok';
    calc.textContent = `recomputed ${fmtInt(report.sample.recruited)} respondents in ${Math.round(report.compute_ms)} ms · ${rt} ms round trip`;
    if ($('auditDisclosure').open) loadParticipants();
  } catch (err) {
    if (err.name === 'AbortError') return;
    calc.className = 'calc calc--bad';
    calc.textContent = 'failed';
    showError(`Analysis failed: ${err.message}`);
  } finally {
    if (state.inflight === ctrl) state.inflight = null;
  }
}

function renderAll() {
  const steps = [renderKpis, renderFunnel, renderMethodDependent, renderDistribution, renderDHists, renderMethodNote];
  for (const fn of steps) {
    try { fn(); } catch (err) { console.error(`[studio] ${fn.name} failed`, err); }
  }
}

function renderMethodDependent() {
  if (!state.report) return;
  for (const fn of [renderKpis, renderTakeaways, renderHeatmap, renderForest, renderComparisons]) {
    try { fn(); } catch (err) { console.error(`[studio] ${fn.name} failed`, err); }
  }
  if (state.participants) renderAudit();
}

// ---------------------------------------------------------------- KPIs

function renderKpis() {
  const r = state.report;
  const s = r.sample;
  const tr = r.trimming || {};
  const ag = r.agreement || {};
  const removed = (tr.removed_upper || 0) + (tr.removed_lower || 0);
  const tiles = [
    ['Recruited', fmtInt(s.recruited), `${fmtInt(s.trial_rows)} trial rows ingested`, ''],
    ['Final analytic sample', fmtInt(s.included), `${fmtPct(s.included_pct)} of recruited · ${fmtInt(s.recruited - s.included)} excluded`, 'kpi--accent'],
    ['Trials trimmed', fmtPct(tr.removed_pct, 2), `${fmtInt(removed)} of ${fmtInt(tr.combined_trials)} scored-block trials`, ''],
    ['Method agreement', `${ag.n_agree ?? '—'} / ${ag.n_tests ?? '—'}`,
      ag.n_agree === ag.n_tests ? 'every significance call survives either test' : 'some calls depend on the test — see ⚠ below',
      ag.n_agree === ag.n_tests ? '' : 'kpi--warn'],
  ];
  $('kpis').innerHTML = tiles.map(([l, v, sub, cls]) => `
    <div class="kpi ${cls}">
      <div class="kpi__label">${esc(l)}</div>
      <div class="kpi__value">${esc(v)}</div>
      <div class="kpi__sub">${esc(sub)}</div>
    </div>`).join('');
}

// ---------------------------------------------------------------- funnel

function renderFunnel() {
  const f = state.report.funnel || [];
  const total = f[0]?.n || 1;
  $('funnelTag').textContent = `${fmtPct(state.report.sample.included_pct)} retained`;
  $('funnel').innerHTML = f.map((s, i) => {
    const prev = i === 0 ? s.n : f[i - 1].n;
    const keep = (s.n / total) * 100;
    const lost = s.key === 'final' || s.key === 'recruited' ? 0 : ((prev - s.n) / total) * 100;
    const isEnd = s.key === 'final' || s.key === 'recruited';
    return `
      <div class="funnel__row ${isEnd ? 'funnel__row--end' : ''}" data-reason="${esc(s.reason)}">
        <div class="funnel__label">
          <span class="funnel__name">${esc(s.label)}</span>
          ${isEnd ? '' : `<span class="funnel__why">${esc(s.reason)}</span>`}
        </div>
        <div class="funnel__track">
          <div class="funnel__keep" style="width:${keep}%"></div>
          ${lost > 0 ? `<div class="funnel__lost" style="left:${keep}%;width:${lost}%"></div>` : ''}
        </div>
        <div class="funnel__nums">
          <span class="funnel__n">${fmtInt(s.n)}</span>
          <span class="funnel__removed">${isEnd ? '' : (s.removed ? `−${fmtInt(s.removed)}` : '±0')}</span>
        </div>
      </div>`;
  }).join('');
}

// ---------------------------------------------------------------- takeaways

function renderTakeaways() {
  const list = (state.report.takeaways || {})[state.method] || [];
  $('takeawayTag').textContent = state.method;
  $('takeaways').innerHTML = list.map((t) => `
    <li class="takeaway takeaway--${esc(t.kind)}">
      <span class="takeaway__head">${esc(t.headline)}</span>
      <span class="takeaway__detail">${esc(t.detail)}</span>
    </li>`).join('');
}

// ---------------------------------------------------------------- heatmap

function cellTip(attr, cell, colLabel) {
  const m = state.method;
  const other = m === 'permutation' ? 'parametric' : 'permutation';
  const row = (name, x) => `<tr><td>${name}</td><td class="num">${fmtP(x?.p)}</td><td class="num">${fmtP(x?.q)}</td><td>${x?.sig ? '● sig.' : '○ n.s.'}</td></tr>`;
  return `
    <div class="tip__title">${esc(attr.label || attr.key)} · ${esc(colLabel)}</div>
    <div class="tip__big">D = ${fmtD(cell.mean_d, 3)}</div>
    <div>95% CI [${fmtD(cell.ci_low, 3)}, ${fmtD(cell.ci_high, 3)}] · n = ${fmtInt(cell.n)}${cell.low_base ? ' · <strong>low base</strong>' : ''}</div>
    <div>${fmtPct(cell.pct_positive)} of respondents with D &gt; 0 · incongruent pairing ${cell.slowdown_pct == null ? '—' : `${Math.abs(cell.slowdown_pct * 100).toFixed(1)}% ${cell.slowdown_pct >= 0 ? 'slower' : 'faster'}`}</div>
    <table class="tip__table"><thead><tr><th></th><th>p</th><th>q</th><th></th></tr></thead><tbody>
      ${row(m === 'permutation' ? 'Permutation ◂' : 'Parametric ◂', cell[m])}
      ${row(other === 'permutation' ? 'Permutation' : 'Parametric', cell[other])}
    </tbody></table>`;
}

function renderHeatmap() {
  const r = state.report;
  const m = state.method;
  const levels = r.levels || [];
  const inc = ((r.sample.by_segment || {})[r.segment_by] || {}).included || {};
  const cols = [{ key: null, label: 'All respondents', n: r.sample.included }, ...levels.map((lv) => ({ key: lv, label: lv, n: inc[lv] }))];
  $('heatTag').textContent = r.segment_by ? `by ${segLabel(r.segment_by).toLowerCase()} · ${m}` : m;

  const first = r.attributes[0] || {};
  const a = brand(first.target_a || 'Target A');
  const b = brand(first.target_b || 'Target B');
  $('heatLegend').innerHTML = `
    <span class="heat-legend__end">◂ leans <strong>${esc(b)}</strong></span>
    <span class="heat-legend__bar" style="background:linear-gradient(90deg, ${diverge(-0.6)}, ${MID}, ${diverge(0.6)})"></span>
    <span class="heat-legend__end">leans <strong>${esc(a)}</strong> ▸</span>
    <span class="heat-legend__note">D &gt; 0: ${esc(a)} more associated with the attribute than ${esc(b)} · colour saturates at |D| = 0.6</span>`;

  const grid = $('heatmap');
  grid.style.gridTemplateColumns = `minmax(190px, 1.3fr) repeat(${cols.length}, minmax(110px, 1fr))`;
  let html = `<div class="heat__corner">Attribute${r.segment_by ? ` <span class="heat__coln">× ${esc(segLabel(r.segment_by))}</span>` : ''}</div>`;
  html += cols.map((c, i) => `
    <div class="heat__col ${i === 0 ? 'heat__col--all' : ''}">
      <span>${esc(c.label)}</span><span class="heat__coln">n = ${fmtInt(c.n)}</span>
    </div>`).join('');
  r.attributes.forEach((attr, ai) => {
    html += `<div class="heat__row">
      <span class="heat__attr">${esc(attr.label || attr.key)}</span>
      <span class="heat__dim">${esc(brand(attr.target_a))} vs ${esc(brand(attr.target_b))}</span>
    </div>`;
    const cells = [attr.overall, ...attr.levels];
    cells.forEach((cell, ci) => {
      const sig = cell[m]?.sig;
      const d = cell.mean_d;
      html += `<div class="heat__cell ${cell.low_base ? 'heat__cell--low' : ''} ${ci === 0 ? 'heat__cell--all' : ''}"
          style="background:${diverge(d)}" data-a="${ai}" data-c="${ci}" tabindex="0"
          aria-label="${esc(attr.label)} ${esc(cols[ci].label)}: D ${fmtD(d)}, ${sig ? 'significant' : 'not significant'}">
        <span class="heat__d">${fmtD(d)}</span>
        <span class="heat__meta"><span class="heat__sig">${sig ? '● sig.' : '○ n.s.'}</span> · n ${fmtInt(cell.n)}${cell.low_base ? ' · low base' : ''}</span>
      </div>`;
    });
  });
  grid.innerHTML = html;
  grid.querySelectorAll('.heat__cell').forEach((el) => {
    const attr = r.attributes[Number(el.dataset.a)];
    const ci = Number(el.dataset.c);
    const cell = ci === 0 ? attr.overall : attr.levels[ci - 1];
    el.addEventListener('mousemove', (ev) => showTip(cellTip(attr, cell, cols[ci].label), ev));
    el.addEventListener('mouseleave', hideTip);
  });
}

// ---------------------------------------------------------------- forest

function chartReady() { return typeof Chart !== 'undefined'; }

function renderForest() {
  const r = state.report;
  const m = state.method;
  const rows = [];
  r.attributes.forEach((attr) => {
    rows.push({ attr, cell: attr.overall, label: `${attr.label || attr.key} — all`, head: true });
    attr.levels.forEach((c) => rows.push({ attr, cell: c, label: `${c.level}`, head: false }));
  });
  $('forestTag').textContent = `${rows.length} estimates · ${m}`;
  $('forestLegend').innerHTML = `
    <span><i style="background:${POS}"></i> leans ${esc(brand(r.attributes[0]?.target_a))}</span>
    <span><i style="background:${NEG}"></i> leans ${esc(brand(r.attributes[0]?.target_b))}</span>
    <span><b class="dot dot--fill"></b> significant (q &lt; .05)</span>
    <span><b class="dot"></b> not significant</span>`;
  $('forestBox').style.height = `${Math.max(240, rows.length * 30 + 70)}px`;

  if (!chartReady()) return;
  destroy('forest');
  const lo = Math.min(-0.2, ...rows.map((x) => x.cell.ci_low ?? 0));
  const hi = Math.max(0.2, ...rows.map((x) => x.cell.ci_high ?? 0));
  const pad = 0.08;
  const colours = rows.map((x) => dirColour(x.cell.mean_d));

  charts.forest = new Chart($('chartForest'), {
    data: {
      labels: rows.map((x) => x.label),
      datasets: [
        {
          type: 'bar',
          label: '95% CI',
          data: rows.map((x) => [x.cell.ci_low, x.cell.ci_high]),
          backgroundColor: colours.map((c) => rgba(c, 0.55)),
          borderWidth: 0,
          barThickness: 4,
          borderRadius: 2,
          borderSkipped: false,
          order: 2,
        },
        {
          type: 'line',
          label: 'Mean D',
          data: rows.map((x) => x.cell.mean_d),
          showLine: false,
          pointRadius: rows.map((x) => (x.head ? 7 : 6)),
          pointHoverRadius: 9,
          pointBorderWidth: 2,
          pointBorderColor: colours,
          pointBackgroundColor: rows.map((x, i) => (x.cell[m]?.sig ? colours[i] : CARD)),
          order: 1,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false, axis: 'y' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0c0e13', borderColor: '#2c3040', borderWidth: 1,
          titleColor: INK, bodyColor: '#c3c7d2', padding: 10, displayColors: false,
          filter: (item) => item.datasetIndex === 1,
          callbacks: {
            title: (items) => {
              const x = rows[items[0].dataIndex];
              return `${x.attr.label} · ${x.head ? 'All respondents' : x.cell.level}`;
            },
            label: (item) => {
              const c = rows[item.dataIndex].cell;
              return [
                `Mean D ${fmtD(c.mean_d, 3)}  (n = ${fmtInt(c.n)}${c.low_base ? ', low base' : ''})`,
                `95% CI [${fmtD(c.ci_low, 3)}, ${fmtD(c.ci_high, 3)}]`,
                `Permutation p ${fmtP(c.permutation?.p)} · q ${fmtP(c.permutation?.q)}`,
                `Parametric  p ${fmtP(c.parametric?.p)} · q ${fmtP(c.parametric?.q)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          min: Math.floor((lo - pad) * 10) / 10,
          max: Math.ceil((hi + pad) * 10) / 10,
          grid: {
            color: (ctx) => (ctx.tick && Math.abs(ctx.tick.value) < 1e-9 ? 'rgba(255,255,255,0.45)' : GRID),
            lineWidth: (ctx) => (ctx.tick && Math.abs(ctx.tick.value) < 1e-9 ? 1.5 : 1),
          },
          ticks: { color: TICK, font: { size: 12 }, callback: (v) => fmtD(Number(v), 1) },
          title: { display: true, text: 'Mean D (95% bootstrap CI)', color: TICK, font: { size: 12 } },
        },
        y: {
          grid: { color: (ctx) => (rows[ctx.index]?.head && ctx.index > 0 ? 'rgba(255,255,255,0.10)' : 'transparent') },
          ticks: {
            color: (ctx) => (rows[ctx.index]?.head ? INK : '#b4b9c6'),
            font: (ctx) => ({ size: 13, weight: rows[ctx.index]?.head ? '600' : '400' }),
            padding: 8,
          },
        },
      },
    },
  });
}

// ---------------------------------------------------------------- distributions

function renderDistribution() {
  const r = state.report;
  if (!r || !r.distributions) return;
  const attr = state.distAttr in r.distributions ? state.distAttr : Object.keys(r.distributions)[0];
  const dist = r.distributions[attr];
  const info = attrInfo(attr);
  const a = brand(info.target_a || 'Target A');
  const conLab = `${a} + ${info.pole_a || 'pole A'}`;
  const incLab = `${a} + ${info.pole_b || 'pole B'}`;
  $('distLegend').innerHTML = `
    <span><i style="background:${CON}"></i> ${esc(conLab)} (congruent)</span>
    <span><i style="background:${INC}"></i> ${esc(incLab)} (incongruent)</span>
    <span><i class="dash"></i> normal curve, same mean &amp; SD</span>`;

  const shape = dist.shape || {};
  const c = shape.congruent || {}, i = shape.incongruent || {};
  const f2 = (v) => (v == null ? '—' : v.toFixed(2));
  $('distShape').innerHTML = `
    <table class="data data--compact">
      <thead><tr><th></th><th>Median</th><th>Skew<br />raw</th><th>Kurtosis<br />raw</th><th>Skew<br />log</th><th>Kurtosis<br />log</th><th>Beyond<br />mean+2SD</th></tr></thead>
      <tbody>
        <tr><td><i class="sw" style="background:${CON}"></i> Congruent</td><td class="num">${fmtInt(Math.round(c.median_ms ?? 0))} ms</td>
          <td class="num">${f2(c.skew)}</td><td class="num">${f2(c.excess_kurtosis)}</td><td class="num">${f2(c.log_skew)}</td><td class="num">${f2(c.log_excess_kurtosis)}</td><td class="num">${fmtPct(c.beyond_2sd, 1)}</td></tr>
        <tr><td><i class="sw" style="background:${INC}"></i> Incongruent</td><td class="num">${fmtInt(Math.round(i.median_ms ?? 0))} ms</td>
          <td class="num">${f2(i.skew)}</td><td class="num">${f2(i.excess_kurtosis)}</td><td class="num">${f2(i.log_skew)}</td><td class="num">${f2(i.log_excess_kurtosis)}</td><td class="num">${fmtPct(i.beyond_2sd, 1)}</td></tr>
      </tbody>
    </table>`;
  $('distNote').textContent =
    `Correct-trial latencies pooled over ${fmtInt(r.sample.included)} included respondents. Raw RTs are right-skewed `
    + `(skew ≈ ${f2(c.skew)}; a normal distribution has 0) with ${fmtPct(c.beyond_2sd, 1)} of trials beyond mean + 2 SD against 2.3% for a normal — `
    + `so a t-test on raw latencies is not defensible. Logging pulls skew to ≈ ${f2(c.log_skew)}, which is why the parametric option tests log-RT differences, `
    + `and why the default is a permutation test that assumes no shape at all.`;

  if (!chartReady()) return;
  destroy('dist');
  const d = state.scale === 'log' ? dist.log : dist.raw;
  const pts = (ys) => (ys || []).map((y, k) => ({ x: d.grid[k], y }));
  const line = (label, ys, colour, dashed) => ({
    label, data: pts(ys), borderColor: dashed ? rgba(colour, 0.7) : colour,
    borderWidth: dashed ? 1.5 : 2, borderDash: dashed ? [5, 4] : [],
    pointRadius: 0, pointHoverRadius: dashed ? 0 : 4, tension: 0.25, fill: false,
  });
  const isLog = state.scale === 'log';
  charts.dist = new Chart($('chartDist'), {
    type: 'line',
    data: {
      datasets: [
        line('Congruent', d.congruent, CON, false),
        line('Incongruent', d.incongruent, INC, false),
        line('Congruent — normal fit', d.congruent_normal, CON, true),
        line('Incongruent — normal fit', d.incongruent_normal, INC, true),
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 200 },
      interaction: { mode: 'nearest', intersect: false, axis: 'x' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0c0e13', borderColor: '#2c3040', borderWidth: 1,
          titleColor: INK, bodyColor: '#c3c7d2', padding: 10,
          filter: (item) => item.datasetIndex < 2,
          callbacks: {
            title: (items) => (isLog ? `≈ ${Math.round(Math.exp(items[0].parsed.x))} ms` : `${Math.round(items[0].parsed.x)} ms`),
            label: (item) => `${item.dataset.label}: density ${item.parsed.y.toExponential(2)}`,
          },
        },
      },
      scales: {
        x: {
          type: 'linear', min: d.grid[0], max: d.grid[d.grid.length - 1],
          grid: { color: GRID },
          // On the log axis, tick at round millisecond values rather than at
          // round log values, so the labels read as latencies.
          afterBuildTicks: isLog
            ? (axis) => { axis.ticks = [200, 300, 500, 700, 1000, 1500, 2000, 3000].map((v) => ({ value: Math.log(v) })); }
            : undefined,
          ticks: {
            color: TICK, font: { size: 12 }, maxTicksLimit: 9,
            callback: (v) => (isLog ? `${Math.round(Math.exp(v))}` : `${Math.round(v)}`),
          },
          title: { display: true, color: TICK, font: { size: 12 },
            text: isLog ? 'Latency, log scale (tick labels in ms)' : 'Latency (ms)' },
        },
        y: {
          grid: { color: GRID }, beginAtZero: true,
          ticks: { display: false },
          title: { display: true, text: 'Density', color: TICK, font: { size: 12 } },
        },
      },
    },
  });
}

// ---------------------------------------------------------------- D histograms

function renderDHists() {
  const r = state.report;
  const box = $('dHists');
  const hs = r.d_histograms || {};
  box.innerHTML = r.attributes.map((a) => `
    <div class="multiple">
      <div class="multiple__head"><span>${esc(a.label || a.key)}</span>
        <span class="mono">mean ${fmtD(a.overall.mean_d)} · ${fmtPct(a.overall.pct_positive)} &gt; 0</span></div>
      <div class="multiple__box"><canvas id="dh-${esc(a.key)}"></canvas></div>
    </div>`).join('');
  if (!chartReady()) return;
  // Small multiples share one y-scale, or their heights cannot be compared.
  const ymax = Math.max(1, ...Object.values(hs).flatMap((h) => h.counts));
  r.attributes.forEach((a) => {
    const h = hs[a.key];
    if (!h) return;
    destroy(`dh-${a.key}`);
    const centres = h.counts.map((_, k) => (h.edges[k] + h.edges[k + 1]) / 2);
    charts[`dh-${a.key}`] = new Chart($(`dh-${a.key}`), {
      type: 'bar',
      data: {
        labels: centres.map((c) => c.toFixed(2)),
        datasets: [{
          data: h.counts,
          backgroundColor: centres.map((c) => (c > 0 ? rgba(POS, 0.85) : rgba(NEG, 0.85))),
          borderColor: CARD, borderWidth: { left: 1, right: 1 }, borderRadius: 2, categoryPercentage: 1, barPercentage: 1,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 200 },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0c0e13', borderColor: '#2c3040', borderWidth: 1, displayColors: false,
            callbacks: {
              title: (items) => {
                const k = items[0].dataIndex;
                return `D ${fmtD(h.edges[k], 1)} to ${fmtD(h.edges[k + 1], 1)}`;
              },
              label: (item) => `${item.parsed.y} respondents`,
            },
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: TICK, font: { size: 10 }, maxTicksLimit: 7, maxRotation: 0,
            callback(v) { const c = Number(this.getLabelForValue(v)); return Math.abs(c * 2 - Math.round(c * 2)) < 0.11 ? fmtD(Math.round(c * 2) / 2, 1) : ''; } } },
          y: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 10 }, precision: 0, maxTicksLimit: 4 }, beginAtZero: true, max: Math.ceil(ymax / 10) * 10 },
        },
      },
    });
  });
}

// ---------------------------------------------------------------- comparisons

const pq = (x) => `p ${fmtP(x?.p)}<br /><span class="muted">q ${fmtP(x?.q)}</span>`;

function renderComparisons() {
  const r = state.report;
  const m = state.method;
  const rows = r.comparisons || [];
  const nsig = rows.filter((c) => c[m]?.sig).length;
  $('cmpTag').textContent = rows.length ? `${nsig} of ${rows.length} significant · ${m}` : 'none';
  if (!rows.length) {
    $('cmpTable').innerHTML = '<tbody><tr><td class="muted">Choose a segment variable with at least two levels to compare.</td></tr></tbody>';
    $('cmpNote').textContent = '';
    return;
  }
  const hl = (k) => (k === m ? 'col--on' : '');
  $('cmpTable').innerHTML = `
    <thead><tr>
      <th>Attribute</th><th>Comparison</th><th>n</th><th>Mean D</th>
      <th>Difference<br />[95% CI]</th><th>Hedges'<br />g</th>
      <th class="${hl('parametric')}">Parametric<br />p / q</th>
      <th class="${hl('permutation')}">Permutation<br />p / q</th>
      <th>Call</th>
    </tr></thead>
    <tbody>${rows.map((c) => {
      const a = attrInfo(c.attribute);
      const sig = c[m]?.sig;
      return `
      <tr class="${sig ? 'row--sig' : ''}">
        <td><strong>${esc(a.label || c.attribute)}</strong></td>
        <td>${esc(c.level_a)} <span class="muted">vs</span> ${esc(c.level_b)}${c.low_base ? ' <span class="tag tag--warn">low base</span>' : ''}</td>
        <td class="num">${fmtInt(c.n_a)} / ${fmtInt(c.n_b)}</td>
        <td class="num">${fmtD(c.mean_a)} / ${fmtD(c.mean_b)}</td>
        <td class="num"><span class="diff" style="--c:${dirColour(c.diff)}">${fmtD(c.diff)}</span><br />
          <span class="muted">[${fmtD(c.ci_low)}, ${fmtD(c.ci_high)}]</span></td>
        <td class="num">${c.hedges_g == null ? '—' : c.hedges_g.toFixed(2)}</td>
        <td class="num pq ${hl('parametric')}">${pq(c.parametric)}</td>
        <td class="num pq ${hl('permutation')}">${pq(c.permutation)}</td>
        <td>${sig ? '<span class="call call--sig">● significant</span>' : '<span class="call">○ n.s.</span>'}
          ${c.agree ? '<span class="agree" title="Both methods reach the same call">✓</span>'
            : '<span class="agree agree--warn" title="The other method reaches a different call — borderline">⚠ borderline</span>'}</td>
      </tr>`;
    }).join('')}</tbody>`;
  $('cmpNote').textContent =
    `Difference = first level minus second. Significance on Benjamini–Hochberg q < .05 across all ${rows.length} comparisons, `
    + `under the ${m} method (highlighted column). ✓ = the other method makes the same call; ⚠ = it does not, so treat the result as borderline.`;
}

// ---------------------------------------------------------------- audit

async function loadParticipants() {
  const key = `${state.batchId}?${paramQuery()}`;
  if (state.participantsKey === key && state.participants) { renderAudit(); return; }
  $('auditTag').textContent = 'loading…';
  try {
    const data = await api(`/api/cohort/batches/${encodeURIComponent(state.batchId)}/participants?${paramQuery()}`);
    state.participants = data;
    state.participantsKey = key;
    renderAudit();
  } catch (err) {
    $('auditTag').textContent = 'failed';
    $('auditTable').innerHTML = `<tbody><tr><td>${esc(err.message)}</td></tr></tbody>`;
  }
}

function renderAudit() {
  const data = state.participants;
  if (!data) return;
  const cols = data.columns;
  const si = cols.indexOf('status');
  const counts = {};
  data.rows.forEach((r) => { counts[r[si]] = (counts[r[si]] || 0) + 1; });
  const excluded = data.rows.length - (counts.Included || 0);
  $('auditTag').textContent = `${fmtInt(data.rows.length)} participants · ${fmtInt(excluded)} excluded`;
  $('auditCounts').innerHTML = Object.entries(counts).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<span class="tag ${k === 'Included' ? 'tag--good' : 'tag--warn'}">${esc(k)} · ${fmtInt(v)}</span>`).join(' ');

  const filter = $('auditFilter').value;
  const rows = data.rows.filter((r) => filter === 'all'
    || (filter === 'included' ? r[si] === 'Included' : r[si] !== 'Included'));
  // Keep the audit readable: the headline columns plus one D per attribute.
  const keep = cols.map((c, k) => k).filter((k) => {
    const c = cols[k];
    return !c.startsWith('D practice') && !c.startsWith('D test') && !c.startsWith('log-RT') && !c.startsWith('Trials analysed');
  });
  const fmtCell = (c, v) => {
    if (v == null) return '<span class="muted">—</span>';
    if (c === 'accuracy' || c === 'fast_trial_share') return fmtPct(v, 1);
    if (c.startsWith('D ')) return fmtD(v, 3);
    return esc(v);
  };
  const header = (c) => (c === 'fast_trial_share' ? 'Fast trials' : c === 'participant_id' ? 'Participant'
    : (state.batch?.segments || {})[c] ? segLabel(c) : c.charAt(0).toUpperCase() + c.slice(1));
  $('auditTable').innerHTML = `
    <thead><tr>${keep.map((k) => `<th>${esc(header(cols[k]))}</th>`).join('')}</tr></thead>
    <tbody>${rows.map((r) => `<tr class="${r[si] === 'Included' ? '' : 'row--excluded'}">${keep.map((k) => {
      const c = cols[k];
      const numeric = c === 'accuracy' || c === 'fast_trial_share' || c.startsWith('D ');
      return `<td class="${numeric ? 'num' : ''}">${fmtCell(c, r[k])}</td>`;
    }).join('')}</tr>`).join('')}</tbody>`;
}

// ---------------------------------------------------------------- method note

function renderMethodNote() {
  const n = state.report.method_note || {};
  const labels = {
    unit: 'Unit of analysis', ci: 'Intervals', parametric: 'Parametric', permutation: 'Permutation',
    fdr: 'Multiple comparisons', low_base: 'Low base',
  };
  const items = Object.entries(n).map(([k, v]) => `<dt>${esc(labels[k] || k)}</dt><dd>${esc(v)}</dd>`);
  items.push(`<dt>Reading D</dt><dd>${esc(state.report.interpretation_note || '')}</dd>`);
  $('methodNote').innerHTML = items.join('');
}

boot();
