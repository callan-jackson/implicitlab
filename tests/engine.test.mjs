/**
 * Headless tests for the trial engine.
 *
 * The engine is the one component that cannot be exercised from Python, and it
 * owns the two things most likely to be silently wrong in an implicit-testing
 * tool: which key counts as correct, and what gets recorded when the
 * participant gets it wrong. A scoring bug shows up as an odd number; an engine
 * bug shows up as a perfectly plausible number computed from the wrong data.
 *
 * Rather than pull in jsdom, this stubs the handful of browser APIs the engine
 * actually touches. The stub is small enough to read, which matters: a test
 * harness you cannot audit is not much better than no test.
 *
 * Run with:  node --test tests/engine.test.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

// ---------------------------------------------------------------- DOM stub

function makeElement(id = '') {
  const el = {
    id,
    textContent: '',
    innerHTML: '',
    className: '',
    hidden: false,
    style: {},
    _handlers: {},
    addEventListener(type, cb) { (this._handlers[type] ||= []).push(cb); },
    removeEventListener(type, cb) {
      this._handlers[type] = (this._handlers[type] || []).filter((h) => h !== cb);
    },
    querySelector() { return this._child ||= makeElement('blockGo'); },
    click() { (this._handlers.click || []).forEach((h) => h()); },
  };
  return el;
}

function installBrowserStubs() {
  const listeners = {};
  const win = {
    addEventListener(type, cb) { (listeners[type] ||= []).push(cb); },
    removeEventListener(type, cb) {
      listeners[type] = (listeners[type] || []).filter((h) => h !== cb);
    },
    dispatchEvent(ev) {
      // Capture-phase listeners run in registration order here, which is enough
      // for the engine: it only ever registers on window.
      [...(listeners[ev.type] || [])].forEach((h) => h(ev));
      return true;
    },
    _listeners: listeners,
  };

  const doc = {
    body: { offsetHeight: 0 },
    hidden: false,
    fonts: { ready: Promise.resolve() },
    addEventListener() {},
    removeEventListener() {},
  };

  globalThis.window = win;
  globalThis.document = doc;
  globalThis.performance = globalThis.performance || { now: () => Date.now() };
  // A frame every millisecond: fast enough that a 90-trial session runs in
  // under a second, and real enough that the nested-rAF onset logic is
  // genuinely exercised rather than bypassed by the stall watchdog.
  globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 1);
  globalThis.KeyboardEvent = class {
    constructor(type, init = {}) {
      this.type = type;
      this.key = init.key;
      this.code = init.code;
      this.timeStamp = performance.now();
      this.defaultPrevented = false;
    }
    preventDefault() { this.defaultPrevented = true; }
  };
  return { win, doc };
}

const { win } = installBrowserStubs();
const { TrialEngine } = await import('../static/js/engine.js');

// ---------------------------------------------------------------- fixtures

function design({ nTrials = 8, responseWindowMs = null, blocks: blockCount = 1 } = {}) {
  const blocks = [];
  const trials = [];
  let idx = 0;
  for (let b = 1; b <= blockCount; b += 1) {
    blocks.push({
      index: b,
      n_trials: nTrials,
      kind: 'combined',
      role: 'test',
      pairing: 'congruent',
      instruction: 'Sort them.',
      left: [{ key: 'target_a', label: 'AURELIA', colour: '#fff' }],
      right: [{ key: 'negative', label: 'Everyday', colour: '#0f0' }],
      response_window_ms: responseWindowMs,
      scored: true,
    });
    for (let i = 0; i < nTrials; i += 1) {
      const onLeft = i % 2 === 0;
      trials.push({
        index: idx++,
        block_index: b,
        block_role: 'test',
        pairing: 'congruent',
        stimulus: onLeft ? 'AURELIA' : 'Ordinary',
        stimulus_category: onLeft ? 'target_a' : 'negative',
        correct_key: onLeft ? 'E' : 'I',
        correct_side: onLeft ? 'left' : 'right',
        word_count: 1,
      });
    }
  }
  return { blocks, trials, keys: { left: 'E', right: 'I' } };
}

function makeDom() {
  return {
    stimulus: makeElement('stimulus'),
    feedback: makeElement('feedback'),
    cueLeft: makeElement('cueLeft'),
    cueRight: makeElement('cueRight'),
    overlay: makeElement('overlay'),
  };
}

const press = (key) => win.dispatchEvent(new KeyboardEvent('keydown', { key }));
const tick = (ms = 5) => new Promise((r) => setTimeout(r, ms));

/**
 * Build an engine plus a driver that plays the session to completion.
 *
 * The driver follows the *engine's own* progress callback rather than counting
 * its own keypresses. Counting locally desynchronises the moment the harness
 * gets ahead — a key pressed during an inter-trial interval is dropped, and
 * from then on every press answers the wrong trial. That is a harness bug that
 * looks exactly like an engine bug, so it is worth not having.
 *
 * @param {(trial: object) => string[]} keysFor  keys to press for a trial, in order
 */
function driven(session, dom, keysFor, opts = {}) {
  const progress = { done: 0 };
  const engine = new TrialEngine(session, dom, {
    ...opts,
    onProgress: (done, total, block) => {
      progress.done = done;
      if (opts.onProgress) opts.onProgress(done, total, block);
    },
  });

  const byIndex = new Map(session.trials.map((t) => [t.index, t]));

  async function play({ timeoutMs = 20000, stopAfter = Infinity, beforeKeys = null } = {}) {
    const started = Date.now();
    while (progress.done < Math.min(session.trials.length, stopAfter)
           && Date.now() - started < timeoutMs) {
      if (!dom.overlay.hidden && dom.overlay._child) {
        dom.overlay._child.click();
        await tick();
        continue;
      }
      const txt = (dom.stimulus.textContent || '').trim();
      if (!txt || ['3', '2', '1', 'Ready', 'Ready.'].includes(txt)) { await tick(); continue; }

      const expected = progress.done;
      const trial = byIndex.get(expected);
      if (!trial) { await tick(); continue; }

      // Wait until the engine is actually listening. It arms its key handler a
      // microtask after the stimulus is painted, so a harness that dispatches
      // the instant the text changes can beat it — and the dropped keypress
      // then looks like a scoring bug. (A human cannot respond inside that
      // window; only a synthetic driver can.)
      const armed = Date.now() + 3000;
      while (!engine._resolveKey && Date.now() < armed) await tick(2);

      if (beforeKeys) await beforeKeys();
      for (const k of keysFor(trial)) { press(k); await tick(4); }

      // Wait for the engine to actually record the trial before moving on.
      const deadline = Date.now() + 3000;
      while (progress.done === expected && Date.now() < deadline) await tick(4);
    }
  }

  return { engine, play, progress };
}

// ---------------------------------------------------------------- tests

test('records one row per trial, all correct', async () => {
  const session = design({ nTrials: 8 });
  const dom = makeDom();
  const { engine, play } = driven(session, dom, (t) => [t.correct_key]);
  const run = engine.run();
  await play();
  const records = await run;

  assert.equal(records.length, 8);
  for (const r of records) {
    assert.equal(r.correct, true, `trial ${r.index} should be correct`);
    assert.equal(r.n_corrections, 0);
    assert.ok(r.latency_ms > 0, 'latency must be positive');
    assert.equal(r.latency_to_correct_ms, r.latency_ms,
      'a correct trial has the same value on both clocks');
    assert.equal(r.timed_out, false);
  }
});

test('a wrong key is recorded as an error and does not advance the trial', async () => {
  const session = design({ nTrials: 6 });
  const dom = makeDom();
  const { engine, play } = driven(session, dom, (t) => {
    const wrong = t.correct_key === 'E' ? 'I' : 'E';
    return [wrong, t.correct_key];
  });
  const run = engine.run();
  await play();
  const records = await run;

  assert.equal(records.length, 6);
  for (const r of records) {
    assert.equal(r.correct, false, 'first response was wrong, so the trial is an error');
    assert.equal(r.n_corrections, 1, 'exactly one further keypress corrected it');
    assert.ok(r.latency_to_correct_ms > r.latency_ms,
      'time-to-correct must exceed time-to-first-response on an error trial');
    assert.notEqual(r.response_key, r.correct_key,
      'response_key records what was actually pressed first');
  }
});

test('two wrong keys before the correct one are both counted', async () => {
  const session = design({ nTrials: 4 });
  const dom = makeDom();
  const { engine, play } = driven(session, dom, (t) => {
    const wrong = t.correct_key === 'E' ? 'I' : 'E';
    return [wrong, wrong, t.correct_key];
  });
  const run = engine.run();
  await play();
  const records = await run;
  for (const r of records) {
    assert.equal(r.correct, false);
    assert.equal(r.n_corrections, 2);
  }
});

test('keys other than E and I are ignored', async () => {
  const session = design({ nTrials: 4 });
  const dom = makeDom();
  const { engine, play } = driven(session, dom, (t) => ['A', 'Z', '5', t.correct_key]);
  const run = engine.run();
  await play();
  const records = await run;
  assert.equal(records.length, 4);
  for (const r of records) {
    assert.equal(r.correct, true, 'stray keys must not be treated as responses');
    assert.equal(r.n_corrections, 0);
  }
});

test('the response window times out and the trial is flagged', async () => {
  const session = design({ nTrials: 3, responseWindowMs: 60 });
  const dom = makeDom();
  // Wait past the window before responding at all.
  const { engine, play } = driven(session, dom, (t) => [t.correct_key]);
  const run = engine.run();
  await play({ beforeKeys: () => tick(110) });
  const records = await run;

  assert.equal(records.length, 3);
  for (const r of records) {
    assert.equal(r.timed_out, true, 'a response after the window must be flagged');
    assert.equal(r.correct, false, 'a timed-out trial is not scored as correct');
  }
});

test('carries the condition labels through from the design untouched', async () => {
  const session = design({ nTrials: 4 });
  const dom = makeDom();
  const { engine, play } = driven(session, dom, (t) => [t.correct_key]);
  const run = engine.run();
  await play();
  const records = await run;

  for (const r of records) {
    const truth = session.trials[r.index];
    assert.equal(r.block_index, truth.block_index);
    assert.equal(r.block_role, truth.block_role);
    assert.equal(r.pairing, truth.pairing);
    assert.equal(r.stimulus, truth.stimulus);
    assert.equal(r.stimulus_category, truth.stimulus_category);
    assert.equal(r.correct_key, truth.correct_key);
  }
});

test('runs several blocks in order and shows an instruction screen for each', async () => {
  const session = design({ nTrials: 4, blocks: 3 });
  const dom = makeDom();
  const seen = [];
  const { engine, play } = driven(session, dom, (t) => [t.correct_key],
    { onBlockStart: (b) => seen.push(b.index) });
  const run = engine.run();
  await play();
  const records = await run;

  assert.equal(records.length, 12);
  assert.deepEqual(seen, [1, 2, 3]);
  assert.deepEqual([...new Set(records.map((r) => r.block_index))], [1, 2, 3]);
  // Trials must stay in presentation order.
  assert.deepEqual(records.map((r) => r.index), [...Array(12).keys()]);
});

test('progress is reported once per trial', async () => {
  const session = design({ nTrials: 6 });
  const dom = makeDom();
  const seen = [];
  const { engine, play } = driven(session, dom, (t) => [t.correct_key],
    { onProgress: (done, total) => seen.push([done, total]) });
  const run = engine.run();
  await play();
  await run;
  assert.deepEqual(seen, [[1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6]]);
});

test('abort stops the session and returns what was collected', async () => {
  const session = design({ nTrials: 20 });
  const dom = makeDom();
  const { engine, play } = driven(session, dom, (t) => [t.correct_key]);
  const run = engine.run();
  await play({ stopAfter: 4 });
  engine.abort();
  const records = await run;

  assert.equal(engine.aborted, true);
  assert.ok(records.length >= 4 && records.length < 20,
    `expected a partial session, got ${records.length}`);
});

test('the keydown listener is removed when the session ends', async () => {
  const session = design({ nTrials: 3 });
  const dom = makeDom();
  const before = (win._listeners.keydown || []).length;
  const { engine, play } = driven(session, dom, (t) => [t.correct_key]);
  const run = engine.run();
  await play();
  await run;
  const after = (win._listeners.keydown || []).length;
  assert.equal(after, before, 'a finished session must not leave a global key handler behind');
});

test('calibration blocks are labelled for participants, never with their internal index', async () => {
  const session = design({ nTrials: 2 });
  // Calibration runs before B1 and carries negative indices internally.
  session.blocks.unshift(
    { ...session.blocks[0], index: -2, kind: 'motor', role: null, pairing: null },
    { ...session.blocks[0], index: -1, kind: 'reading', role: null, pairing: null },
  );
  const dom = makeDom();
  const engine = new TrialEngine(session, dom, {});
  const shown = [];
  for (const block of session.blocks) {
    const done = engine._instructionScreen(block);
    shown.push(dom.overlay.innerHTML);
    engine._resolveOverlay();
    await done;
  }
  assert.match(shown[0], /Calibration 1 of 2/);
  assert.match(shown[1], /Calibration 2 of 2/);
  assert.match(shown[2], /Block 1 of 1/);
  for (const html of shown) assert.doesNotMatch(html, /[Bb]lock -\d/);
});
