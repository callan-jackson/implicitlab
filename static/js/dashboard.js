/**
 * dashboard.js — renders the report.
 *
 * Two rules run through this file.
 *
 * **Show the interval, not a verdict.** The IAT's test-retest reliability is
 * around r = .50, and Cummins & Hussey (2026) report that a D of exactly zero
 * carries a 95% confidence interval of roughly ±0.38 — wider than the gaps
 * between the conventional labels. So the headline is a number with an interval
 * drawn around it, and the words next to it are hedged to match. A confident
 * per-respondent verdict is the single most common way this instrument gets
 * misused, and the interface is where that misuse either happens or doesn't.
 *
 * **Both conditions share an axis.** Every paired chart uses identical bins,
 * identical scales and a fixed colour for each pairing. Two histograms with
 * different bin widths cannot be compared by eye, however carefully they are
 * labelled.
 */

const $ = (id) => document.getElementById(id);
const CON = '#a78bfa';
const INC = '#fb923c';
const GRID = 'rgba(255,255,255,0.06)';
const TICK = '#6f7585';

const charts = {};

function destroy(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

/** Charts are an aid, not the report. If the library is missing, say so in the
 *  panel and let the numbers and tables carry the result. */
function chartsAvailable(canvasId) {
  if (typeof Chart !== 'undefined') return true;
  const canvas = $(canvasId);
  if (canvas && canvas.parentElement) {
    canvas.parentElement.innerHTML =
      '<p style="color:var(--ink-dim);font-size:0.85rem;display:grid;place-items:center;height:100%">'
      + 'Chart library unavailable — the figures in the tables above are unaffected.</p>';
  }
  return false;
}

const baseOptions = () => ({
  responsive: true,
  maintainAspectRatio: false,
  interaction: { intersect: false, mode: 'index' },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#14161e',
      borderColor: '#23262f',
      borderWidth: 1,
      titleColor: '#f2f3f7',
      bodyColor: '#a9aebd',
      padding: 10,
      displayColors: true,
    },
  },
  scales: {
    x: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 10 } } },
    y: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 10 } }, beginAtZero: true },
  },
});

// -------------------------------------------------------------- entry point

export function renderReport(report) {
  const simulated = !!(report.client_meta_simulated || report.simulated);
  const steps = [
    ['banners', () => renderBanners(report, simulated)],
    ['verdict', () => renderVerdict(report)],
    ['headline', () => renderHeadlineStats(report)],
    ['sensitivity', () => renderSensitivity(report)],
    ['distribution', () => renderDistribution(report)],
    ['series', () => renderSeries(report)],
    ['null', () => renderNull(report)],
    ['blocks', () => renderBlocks(report)],
    ['summary', () => renderSummary(report)],
    ['quality', () => renderQuality(report)],
    ['timing', () => renderTiming(report)],
    ['repro', () => renderRepro(report)],
    ['norms', () => renderNorms(report)],
  ];
  // Each panel renders independently. One panel throwing must not cost the
  // reader the other twelve — a chart failing is a cosmetic problem, whereas a
  // blank report after someone has done 190 trials is not.
  for (const [name, fn] of steps) {
    console.log(`[render] ${name}`);
    try {
      fn();
    } catch (err) {
      console.error(`[render] ${name} failed`, err);
    }
  }
  console.log('[render] done');
}

// -------------------------------------------------------------- banners

function renderBanners(report, simulated) {
  const out = [];
  if (simulated) {
    out.push(banner('info', '◆',
      '<strong>Simulated respondent.</strong> These trials were generated from an ' +
      'ex-Gaussian latency model with a known effect built in, then put through the ' +
      'identical scoring pipeline a real session uses. Nobody took this test.'));
  }
  const score = report.score;
  if (score.excluded) {
    out.push(banner('bad', '✗',
      '<strong>This session was excluded and no D-score is reported.</strong> ' +
      escapeHtml(score.exclusion_reasons.join(' ')) +
      ' Producing a number anyway would be the error, so the tool declines to.'));
  } else if (report.inference && report.inference.ci_low != null &&
             report.inference.ci_low <= 0 && report.inference.ci_high >= 0) {
    out.push(banner('warn', '!',
      '<strong>The interval includes zero.</strong> This session did not produce an ' +
      'effect distinguishable from no effect. The direction below is the point ' +
      'estimate, not a finding.'));
  }
  (score.warnings || []).forEach((w) => out.push(banner('warn', '!', escapeHtml(w))));
  $('resultBanners').innerHTML = out.join('');
}

function banner(kind, icon, html) {
  return `<div class="banner banner--${kind}"><span class="banner__icon">${icon}</span><span>${html}</span></div>`;
}

// -------------------------------------------------------------- verdict

function renderVerdict(report) {
  const s = report.score;
  const inf = report.inference || {};
  const study = report.study || {};

  $('verdictStudy').textContent = study.name || 'Session';
  $('verdictD').textContent = s.d == null ? '—' : fmt(s.d, 3, true);
  $('verdictD').style.color = s.d == null ? 'var(--ink-dim)'
    : s.d > 0 ? CON : s.d < 0 ? INC : 'var(--ink)';

  if (s.d == null) {
    $('verdictCI').textContent = 'No interval — the session did not meet the scoring criteria.';
  } else if (inf.ci_low != null) {
    $('verdictCI').innerHTML =
      `95% CI <span class="mono">[${fmt(inf.ci_low, 3, true)}, ${fmt(inf.ci_high, 3, true)}]</span>` +
      (inf.p_permutation != null ? ` · permutation p = ${fmt(inf.p_permutation, 3)}` : '');
  } else {
    $('verdictCI').textContent = 'Too few trials for an interval.';
  }

  $('verdictReading').textContent = s.interpretation || '';
  $('verdictCaveat').textContent = report.single_session_note || '';

  // Position the marker and the interval on a −1…+1 scale.
  const pct = (v) => `${Math.min(100, Math.max(0, ((v + 1) / 2) * 100))}%`;
  if (s.d == null) {
    $('scaleMarker').style.display = 'none';
    $('scaleCI').style.display = 'none';
  } else {
    $('scaleMarker').style.display = '';
    $('scaleMarker').style.left = pct(s.d);
    if (inf.ci_low != null) {
      const lo = Math.min(100, Math.max(0, ((inf.ci_low + 1) / 2) * 100));
      const hi = Math.min(100, Math.max(0, ((inf.ci_high + 1) / 2) * 100));
      $('scaleCI').style.display = '';
      $('scaleCI').style.left = `${lo}%`;
      $('scaleCI').style.width = `${Math.max(0.6, hi - lo)}%`;
    } else {
      $('scaleCI').style.display = 'none';
    }
  }
}

function renderHeadlineStats(report) {
  const s = report.score;
  const d = report.descriptives || {};
  const con = d.congruent || {};
  const inc = d.incongruent || {};
  const labels = d.labels || {};
  const gap = (con.mean_ms != null && inc.mean_ms != null) ? inc.mean_ms - con.mean_ms : null;

  const cells = [
    ['Practice-block D', s.d_practice == null ? '—' : fmt(s.d_practice, 3, true),
     'blocks 3 and 6'],
    ['Test-block D', s.d_test == null ? '—' : fmt(s.d_test, 3, true),
     'blocks 4 and 7'],
    ['Median latency', con.median_ms == null ? '—' : `${Math.round(con.median_ms)} ms`,
     labels.congruent || 'congruent'],
    ['Median latency', inc.median_ms == null ? '—' : `${Math.round(inc.median_ms)} ms`,
     labels.incongruent || 'incongruent'],
    ['Raw gap', gap == null ? '—' : `${gap > 0 ? '+' : ''}${Math.round(gap)} ms`,
     'mean difference before standardising'],
    ['Trials scored', String(s.n_trials_scored ?? 0),
     `${s.n_trials_dropped ?? 0} removed by the cut-offs`],
  ];
  $('headlineStats').innerHTML = cells.map(([l, v, sub]) => `
    <div class="stat">
      <div class="stat__label">${escapeHtml(l)}</div>
      <div class="stat__value">${escapeHtml(v)}</div>
      <div class="stat__sub">${escapeHtml(sub)}</div>
    </div>`).join('');
}

function renderSensitivity(report) {
  const rows = report.sensitivity || [];
  const primary = report.score.d;
  if (!rows.length) {
    $('sensTable').innerHTML =
      '<tbody><tr><td style="color:var(--ink-dim)">A single-category test has one published scoring procedure, so there is nothing to vary.</td></tr></tbody>';
    return;
  }
  $('sensTable').innerHTML = `
    <thead><tr><th>Procedure</th><th>D</th><th>Δ</th></tr></thead>
    <tbody>
      <tr>
        <td><strong>Greenwald 2003, built-in error penalty</strong><br />
          <span style="color:var(--ink-dim);font-size:0.8rem">Scores time-to-correct-response. This deployment's default.</span></td>
        <td class="num">${primary == null ? '—' : fmt(primary, 3, true)}</td>
        <td class="num" style="color:var(--ink-dim)">—</td>
      </tr>
      ${rows.map((r) => `
        <tr>
          <td>${escapeHtml(r.label)}<br />
            <span style="color:var(--ink-dim);font-size:0.8rem">${escapeHtml(r.why)}</span></td>
          <td class="num">${r.d == null ? '—' : fmt(r.d, 3, true)}</td>
          <td class="num" style="color:${Math.abs(r.delta_from_primary || 0) > 0.1 ? 'var(--warn)' : 'var(--ink-dim)'}">
            ${r.delta_from_primary == null ? '—' : fmt(r.delta_from_primary, 3, true)}</td>
        </tr>`).join('')}
    </tbody>`;
}

// -------------------------------------------------------------- charts

function renderDistribution(report) {
  const labels = (report.descriptives || {}).labels || {};
  $('legendCon').textContent = labels.congruent || 'congruent';
  $('legendInc').textContent = labels.incongruent || 'incongruent';

  const h = (report.charts || {}).histograms || {};
  const con = h.congruent || { edges: [], counts: [] };
  const inc = h.incongruent || { edges: [], counts: [] };
  const edges = con.edges.length ? con.edges : inc.edges;

  // Trim empty tails so the interesting range fills the axis.
  let last = 0;
  edges.forEach((_, i) => {
    if ((con.counts[i] || 0) + (inc.counts[i] || 0) > 0) last = i;
  });
  const upto = Math.min(edges.length, last + 3);

  if (!chartsAvailable('chartDist')) return;
  destroy('dist');
  const ctx = $('chartDist');
  if (!ctx) return;
  charts.dist = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: edges.slice(0, upto).map((e) => `${e}`),
      datasets: [
        { label: labels.congruent || 'congruent', data: con.counts.slice(0, upto),
          backgroundColor: hex(CON, 0.62), borderColor: CON, borderWidth: 1 },
        { label: labels.incongruent || 'incongruent', data: inc.counts.slice(0, upto),
          backgroundColor: hex(INC, 0.62), borderColor: INC, borderWidth: 1 },
      ],
    },
    options: {
      ...baseOptions(),
      scales: {
        x: {
          grid: { color: GRID }, stacked: false,
          ticks: { color: TICK, font: { size: 10 }, maxTicksLimit: 10,
                   callback(v) { return `${this.getLabelForValue(v)}` } },
          title: { display: true, text: 'Response latency (ms)', color: TICK, font: { size: 11 } },
        },
        y: {
          grid: { color: GRID }, beginAtZero: true,
          ticks: { color: TICK, font: { size: 10 }, precision: 0 },
          title: { display: true, text: 'Trials', color: TICK, font: { size: 11 } },
        },
      },
    },
  });
}

function renderSeries(report) {
  const series = (report.charts || {}).series || [];
  const labels = (report.descriptives || {}).labels || {};
  if (!chartsAvailable('chartSeries')) return;
  destroy('series');
  const ctx = $('chartSeries');
  if (!ctx) return;
  charts.series = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: labels.congruent || 'congruent',
          data: series.filter((p) => p.pairing === 'congruent').map((p) => ({ x: p.index, y: p.latency_ms, c: p.correct })),
          backgroundColor: (c) => (c.raw && c.raw.c === false ? '#f87171' : hex(CON, 0.85)),
          pointRadius: 3, pointHoverRadius: 5,
        },
        {
          label: labels.incongruent || 'incongruent',
          data: series.filter((p) => p.pairing === 'incongruent').map((p) => ({ x: p.index, y: p.latency_ms, c: p.correct })),
          backgroundColor: (c) => (c.raw && c.raw.c === false ? '#f87171' : hex(INC, 0.85)),
          pointRadius: 3, pointHoverRadius: 5,
        },
      ],
    },
    options: {
      ...baseOptions(),
      plugins: {
        ...baseOptions().plugins,
        tooltip: {
          ...baseOptions().plugins.tooltip,
          callbacks: {
            label: (c) => `${c.dataset.label}: ${Math.round(c.parsed.y)} ms${c.raw.c === false ? ' (error)' : ''}`,
          },
        },
      },
      scales: {
        x: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 10 } },
             title: { display: true, text: 'Trial number', color: TICK, font: { size: 11 } } },
        y: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 10 } },
             title: { display: true, text: 'Latency (ms)', color: TICK, font: { size: 11 } } },
      },
    },
  });
}

function renderNull(report) {
  const inf = report.inference || {};
  const nulls = inf.null_distribution || [];
  const p = inf.p_permutation;

  $('pTag').textContent = p == null ? 'not computed' : `p = ${fmt(p, 3)}`;
  $('pTag').className = `tag ${p == null ? '' : p < 0.05 ? 'tag--good' : 'tag--warn'}`;

  $('nullNote').textContent = nulls.length
    ? `The grey distribution is what D looks like when the pairing labels are shuffled `
      + `within each block pair — that is, when the labels carry no information. The marker `
      + `is the observed D. The p-value is the proportion of shuffles that produced a D at `
      + `least this far from zero, which is a question answerable from one participant's own `
      + `trials because the trials are the exchangeable units.`
    : 'Too few trials to build a permutation distribution.';

  if (!chartsAvailable('chartNull')) return;
  destroy('null');
  const ctx = $('chartNull');
  if (!ctx || !nulls.length) return;

  const lo = Math.min(-1, Math.min(...nulls), report.score.d ?? 0);
  const hi = Math.max(1, Math.max(...nulls), report.score.d ?? 0);
  const bins = 40;
  const width = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  nulls.forEach((v) => {
    const i = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / width)));
    counts[i] += 1;
  });
  const centres = counts.map((_, i) => lo + width * (i + 0.5));
  const d = report.score.d;
  const observedBin = d == null ? -1 : Math.min(bins - 1, Math.max(0, Math.floor((d - lo) / width)));

  charts.null = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: centres.map((c) => c.toFixed(2)),
      datasets: [{
        data: counts,
        backgroundColor: counts.map((_, i) => (i === observedBin ? CON : 'rgba(160,166,182,0.35)')),
        borderColor: counts.map((_, i) => (i === observedBin ? CON : 'rgba(160,166,182,0.5)')),
        borderWidth: 1,
      }],
    },
    options: {
      ...baseOptions(),
      plugins: {
        ...baseOptions().plugins,
        tooltip: {
          ...baseOptions().plugins.tooltip,
          callbacks: {
            title: (items) => `D ≈ ${items[0].label}`,
            label: (c) => `${c.parsed.y} of ${nulls.length} shuffles`,
          },
        },
      },
      scales: {
        x: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 9 }, maxTicksLimit: 9 },
             title: { display: true, text: 'D under the null (labels shuffled)', color: TICK, font: { size: 11 } } },
        y: { grid: { color: GRID }, beginAtZero: true, ticks: { color: TICK, font: { size: 10 } } },
      },
    },
  });
}

function renderNorms(report) {
  const norms = report.norms || {};
  const card = $('normsCard');
  if (!norms.d_values || norms.d_values.length < 5) { card.hidden = true; return; }
  card.hidden = false;

  $('normsTag').textContent = `n = ${norms.n}`;
  $('normsNote').textContent =
    (norms.percentile != null
      ? `This session sits at the ${norms.percentile}th percentile of the sessions collected so far. `
      : '')
    + norms.note;

  const lo = -1.5, hi = 1.5, bins = 30, width = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  norms.d_values.forEach((v) => {
    const i = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / width)));
    counts[i] += 1;
  });
  const d = report.score.d;
  const mine = d == null ? -1 : Math.min(bins - 1, Math.max(0, Math.floor((d - lo) / width)));

  if (!chartsAvailable('chartNorms')) return;
  destroy('norms');
  charts.norms = new Chart($('chartNorms'), {
    type: 'bar',
    data: {
      labels: counts.map((_, i) => (lo + width * (i + 0.5)).toFixed(2)),
      datasets: [{
        data: counts,
        backgroundColor: counts.map((_, i) => (i === mine ? CON : 'rgba(160,166,182,0.35)')),
        borderColor: counts.map((_, i) => (i === mine ? CON : 'rgba(160,166,182,0.5)')),
        borderWidth: 1,
      }],
    },
    options: {
      ...baseOptions(),
      scales: {
        x: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 9 }, maxTicksLimit: 9 },
             title: { display: true, text: 'D across collected sessions', color: TICK, font: { size: 11 } } },
        y: { grid: { color: GRID }, beginAtZero: true, ticks: { color: TICK, font: { size: 10 }, precision: 0 } },
      },
    },
  });
}

// -------------------------------------------------------------- tables

function renderBlocks(report) {
  const rows = report.score.blocks || [];
  const labels = (report.descriptives || {}).labels || {};
  $('blockTable').innerHTML = `
    <thead><tr>
      <th>Block</th><th>Pairing</th><th>n</th><th>Errors</th>
      <th>Mean correct</th><th>Median</th>
    </tr></thead>
    <tbody>${rows.map((b) => `
      <tr>
        <td>${b.block_index > 0 ? `B${b.block_index}` : '—'}
          <span class="tag tag--${b.block_role === 'test' ? 'test' : 'practice'}" style="margin-left:.4rem">${escapeHtml(b.block_role)}</span></td>
        <td style="color:${b.pairing === 'congruent' ? CON : INC}">
          ${escapeHtml(b.pairing === 'congruent' ? (labels.congruent || 'congruent') : (labels.incongruent || 'incongruent'))}</td>
        <td class="num">${b.n_retained}</td>
        <td class="num">${b.n_errors} <span style="color:var(--ink-dim)">(${pct(b.error_rate)})</span></td>
        <td class="num">${b.mean_correct_ms == null ? '—' : `${Math.round(b.mean_correct_ms)} ms`}</td>
        <td class="num">${b.median_ms == null ? '—' : `${Math.round(b.median_ms)} ms`}</td>
      </tr>`).join('')}
    </tbody>`;
}

function renderQuality(report) {
  const q = report.quality || {};
  const tag = $('qualityTag');
  tag.textContent = q.overall === 'pass' ? 'all checks passed'
    : q.overall === 'warn' ? 'passed with warnings' : 'failed';
  tag.className = `tag tag--${q.overall === 'pass' ? 'good' : q.overall === 'warn' ? 'warn' : 'bad'}`;

  $('qualityChecks').innerHTML = (q.checks || []).map((c) => `
    <div class="check">
      <span class="check__dot check__dot--${c.severity}"></span>
      <span>
        <span class="check__label">${escapeHtml(c.label)}</span>
        <span class="check__detail">${escapeHtml(c.detail)}</span>
      </span>
      <span class="check__value">${c.value == null ? '' : escapeHtml(String(c.value))}</span>
    </div>`).join('');
}

function renderTiming(report) {
  const meta = report.session || {};
  const cal = report.calibration || {};
  const rel = report.reliability || {};

  const cells = [
    ['Refresh rate', meta.refresh_hz ? `${Number(meta.refresh_hz).toFixed(1)} Hz` : '—',
     'measured from animation frames'],
    ['Onset uncertainty', meta.onset_uncertainty_ms ? `±${Number(meta.onset_uncertainty_ms).toFixed(1)} ms` : '—',
     'half a frame — cancels in a difference score'],
    ['Motor baseline', cal.motor_median_ms ? `${Math.round(cal.motor_median_ms)} ms` : '—',
     'median cued keypress, calibration block'],
    ['Reading slope', cal.reading_slope_ms_per_word ? `${Math.round(cal.reading_slope_ms_per_word)} ms/word` : '—',
     'from the reading calibration block'],
    ['Split-half D', rel.d_odd_trials == null ? '—'
      : `${fmt(rel.d_odd_trials, 2, true)} / ${fmt(rel.d_even_trials, 2, true)}`,
     rel.stable ? 'odd vs even trials — stable' : 'odd vs even trials — unstable'],
    ['Trials in D', String(report.score.n_trials_scored ?? 0),
     meta.preset_label || ''],
  ];
  $('timingStats').innerHTML = cells.map(([l, v, s]) => `
    <div class="stat">
      <div class="stat__label">${escapeHtml(l)}</div>
      <div class="stat__value">${escapeHtml(v)}</div>
      <div class="stat__sub">${escapeHtml(s)}</div>
    </div>`).join('');

  $('timingNote').textContent =
    'Stimulus onset is timestamped from a nested requestAnimationFrame callback and the '
    + 'response from the browser\'s own event.timeStamp, so the delay between the keypress '
    + 'and this page\'s handler running is measured rather than absorbed into the reaction '
    + 'time. Constant display lag does not affect D, because D is a difference between two '
    + 'conditions measured on the same display.';
}

function renderRepro(report) {
  const cb = (report.session || {}).counterbalance || {};
  const sid = report.session_id;
  const rows = [
    ['Session ID', sid || 'not stored'],
    ['Seed', report.seed ?? '—'],
    ['Pairing presented first', cb.congruent_pairing_first == null ? '—'
      : (cb.congruent_pairing_first ? 'congruent' : 'incongruent')],
    ['Target A started on', cb.target_a_starts_left == null ? '—'
      : (cb.target_a_starts_left ? 'the left key (E)' : 'the right key (I)')],
    ['Preset', (report.session || {}).preset_label || '—'],
    ['Algorithm', 'Greenwald, Nosek & Banaji (2003) improved D'],
  ];
  $('reproTable').innerHTML = `<tbody>${rows.map(([k, v]) => `
    <tr><td>${escapeHtml(k)}</td><td class="num">${escapeHtml(String(v))}</td></tr>`).join('')}</tbody>`;

  const csv = $('csvBtn');
  const json = $('jsonBtn');
  if (sid) {
    csv.href = `/api/sessions/${sid}/export.csv`;
    json.href = `/api/sessions/${sid}/export.json`;
    csv.style.display = json.style.display = '';
  } else {
    csv.style.display = json.style.display = 'none';
  }
}

// -------------------------------------------------------------- summary

function renderSummary(report) {
  const ins = report.insight;
  if (!ins) {
    $('summaryBody').innerHTML = '<p style="color:var(--ink-dim)">No summary was generated.</p>';
    return;
  }

  const engine = $('engineTag');
  engine.textContent = ins.engine;
  engine.className = `tag ${ins.engine === 'azure-openai' ? 'tag--accent' : 'tag--warn'}`;

  const v = ins.verification || {};
  const verify = $('verifyTag');
  verify.textContent = v.verified
    ? `${v.numbers_checked} figures verified`
    : `${(v.unverified_numbers || []).length} figures unverified`;
  verify.className = `tag ${v.verified ? 'tag--good' : 'tag--bad'}`;

  $('summaryBody').innerHTML = markdown(ins.summary_markdown || '');

  const notes = (ins.notes || []).join(' ');
  $('llmNote').innerHTML =
    'AI-generated from computed statistics — not human-reviewed. The model receives a fixed '
    + 'block of numbers that Python has already computed and writes prose about them. It never '
    + 'sees a raw latency and is never the source of a figure. '
    + (notes ? `<br /><strong>${escapeHtml(notes)}</strong>` : '');

  const p = ins.prompt_shown || {};
  $('promptText').textContent =
    `SYSTEM\n${'─'.repeat(60)}\n${p.system || ''}\n\n\nUSER\n${'─'.repeat(60)}\n${p.user || ''}`;

  if (ins.rejected_draft) {
    $('rejectedDisclosure').hidden = false;
    $('rejectedWhy').textContent = v.note || '';
    $('rejectedText').textContent = ins.rejected_draft;
  } else {
    $('rejectedDisclosure').hidden = true;
  }
}

/**
 * A deliberately small Markdown subset: headings, bullets, bold, italics, code.
 * Input is escaped first, so nothing the model emits can inject markup. Pulling
 * in a full Markdown library to render five constructs would be a larger
 * attack surface than the feature is worth.
 */
function markdown(src) {
  const esc = escapeHtml(src);
  const lines = esc.split('\n');
  const out = [];
  let inList = false;

  const inline = (s) => s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }
    if (inList) { out.push('</ul>'); inList = false; }

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { out.push(`<h3>${inline(h[2])}</h3>`); continue; }
    if (!line.trim()) continue;
    out.push(`<p>${inline(line)}</p>`);
  }
  if (inList) out.push('</ul>');
  return out.join('\n');
}

// -------------------------------------------------------------- helpers

function fmt(v, dp = 3, signed = false) {
  if (v == null || Number.isNaN(v)) return '—';
  const s = Number(v).toFixed(dp);
  return signed && Number(v) >= 0 ? `+${s}` : s;
}

function pct(v) {
  return v == null ? '—' : `${(v * 100).toFixed(0)}%`;
}

function hex(h, alpha) {
  const n = parseInt(h.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
