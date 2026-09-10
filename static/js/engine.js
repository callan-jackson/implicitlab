/**
 * engine.js — the trial runner.
 *
 * Runs the enumerated trial list the server generated, one block at a time,
 * and returns a flat array of trial records. It makes no decisions about the
 * design: the sequence, the counterbalancing and the key mapping all arrive
 * pre-determined from the server so that the browser cannot accidentally
 * unbalance a session.
 *
 * Error handling follows the standard IAT procedure: a wrong keypress shows a
 * red X and the trial does not advance until the correct key is pressed. That
 * matters for scoring. Greenwald, Nosek & Banaji (2003) describe two ways to
 * penalise errors — a fixed +600 ms added during analysis, or a "built-in"
 * penalty where the participant must correct and the correction time is part of
 * the recorded latency. This engine records *both* clocks per trial:
 *
 *   latencyMs           time to the first keypress, whatever it was
 *   latencyToCorrectMs  time to the correct keypress
 *
 * so the analysis can compute either variant from the same raw file and check
 * that they agree, rather than baking one choice into the data collection.
 */

import { Timing } from './timing.js';

const KEY_LEFT = 'E';
const KEY_RIGHT = 'I';
const ITI_MS = 300;
const ERROR_HOLD_MS = 150;

export class TrialEngine {
  constructor(session, dom, opts = {}) {
    this.session = session;
    this.dom = dom;
    this.onProgress = opts.onProgress || (() => {});
    this.onBlockStart = opts.onBlockStart || (() => {});
    this.records = [];
    this.aborted = false;
    this.focusLosses = 0;
    this._resolveKey = null;
    this._resolveOverlay = null;

    this._onKeyDown = this._onKeyDown.bind(this);
  }

  _onKeyDown(ev) {
    const key = (ev.key || '').toUpperCase();
    if (key !== KEY_LEFT && key !== KEY_RIGHT) return;
    ev.preventDefault();
    if (!this._resolveKey) return;
    const t = Timing.eventTime(ev);
    const resolve = this._resolveKey;
    this._resolveKey = null;
    resolve({ key, ...t });
  }

  _awaitKey(timeoutMs) {
    if (this.aborted) return Promise.resolve(null);
    return new Promise((resolve) => {
      this._resolveKey = resolve;
      if (timeoutMs) {
        setTimeout(() => {
          if (this._resolveKey === resolve) {
            this._resolveKey = null;
            resolve(null);
          }
        }, timeoutMs);
      }
    });
  }

  async run() {
    window.addEventListener('keydown', this._onKeyDown, { capture: true });
    Timing.watchVisibility(() => { this.focusLosses += 1; });

    await Timing.warmUp(this.dom.stimulus);

    const blocks = this.session.blocks;
    const trialsByBlock = new Map();
    for (const t of this.session.trials) {
      if (!trialsByBlock.has(t.block_index)) trialsByBlock.set(t.block_index, []);
      trialsByBlock.get(t.block_index).push(t);
    }

    let done = 0;
    const total = this.session.trials.length;

    for (const block of blocks) {
      if (this.aborted) break;
      // A tab that is not being rendered receives no animation frames, so
      // onset cannot be timed. Wait for it to come back rather than collecting
      // trials whose timing is meaningless.
      // eslint-disable-next-line no-await-in-loop
      await Timing.whenVisible();
      this._renderKeyMap(block);
      this.onBlockStart(block);
      await this._instructionScreen(block);
      if (this.aborted) break;
      await this._countdown();

      const trials = trialsByBlock.get(block.index) || [];
      for (const trial of trials) {
        if (this.aborted) break;
        // eslint-disable-next-line no-await-in-loop
        const rec = await this._runTrial(trial, block);
        if (rec === null) break;   // aborted mid-trial
        this.records.push(rec);
        done += 1;
        this.onProgress(done, total, block);
      }
      this.dom.stimulus.textContent = '';
    }

    window.removeEventListener('keydown', this._onKeyDown, { capture: true });
    return this.records;
  }

  abort() {
    this.aborted = true;
    if (this._resolveKey) {
      const r = this._resolveKey;
      this._resolveKey = null;
      r(null);
    }
    if (this._resolveOverlay) {
      const r = this._resolveOverlay;
      this._resolveOverlay = null;
      r();
    }
  }

  _renderKeyMap(block) {
    const fmt = (cats) => cats.map((c) =>
      `<span class="cue__cat" style="--cue:${c.colour}">${c.label}</span>`).join('<span class="cue__or">or</span>');
    this.dom.cueLeft.innerHTML = `<span class="cue__key">E</span>${fmt(block.left)}`;
    this.dom.cueRight.innerHTML = `<span class="cue__key">I</span>${fmt(block.right)}`;
  }

  _instructionScreen(block) {
    return new Promise((resolve) => {
      this._resolveOverlay = resolve;
      const roleTag = block.role
        ? `<span class="tag tag--${block.role}">${block.role === 'test' ? 'Scored block' : block.role === 'practice' ? 'Scored practice' : 'Test block'}</span>`
        : '<span class="tag">Practice — not scored</span>';
      this.dom.overlay.innerHTML = `
        <div class="overlay__card">
          <div class="overlay__head">
            <span class="overlay__block">Block ${block.index} of ${this.session.blocks.length}</span>
            ${roleTag}
          </div>
          <p class="overlay__text">${block.instruction}</p>
          <div class="overlay__map">
            <div class="overlay__side"><kbd>E</kbd><div>${block.left.map((c) => `<span style="color:${c.colour}">${c.label}</span>`).join(' <em>or</em> ')}</div></div>
            <div class="overlay__side"><kbd>I</kbd><div>${block.right.map((c) => `<span style="color:${c.colour}">${c.label}</span>`).join(' <em>or</em> ')}</div></div>
          </div>
          ${block.response_window_ms ? `<p class="overlay__note">Respond within ${block.response_window_ms} ms. The prompt will tell you if you are too slow.</p>` : ''}
          <p class="overlay__note">${block.n_trials} trials. Mistakes show a red ✗ — correct them to continue.</p>
          <button class="btn btn--primary" id="blockGo">Start block ${block.index} &nbsp;→</button>
          <p class="overlay__hint">or press <kbd>Space</kbd></p>
        </div>`;
      this.dom.overlay.hidden = false;

      const go = () => {
        window.removeEventListener('keydown', onSpace, true);
        this._resolveOverlay = null;
        this.dom.overlay.hidden = true;
        resolve();
      };
      const onSpace = (ev) => {
        if (ev.code === 'Space' || ev.key === ' ') { ev.preventDefault(); go(); }
      };
      this.dom.overlay.querySelector('#blockGo').addEventListener('click', go, { once: true });
      window.addEventListener('keydown', onSpace, true);
    });
  }

  async _countdown() {
    for (const n of ['3', '2', '1']) {
      // eslint-disable-next-line no-await-in-loop
      await Timing.showAndTimestamp(() => {
        this.dom.stimulus.textContent = n;
        this.dom.stimulus.className = 'stimulus stimulus--count';
      });
      // eslint-disable-next-line no-await-in-loop
      await sleep(400);
    }
    await Timing.showAndTimestamp(() => { this.dom.stimulus.textContent = ''; });
    await sleep(200);
  }

  async _runTrial(trial, block) {
    const colour = this._colourFor(trial.stimulus_category, block);
    const focusBefore = this.focusLosses;

    const onset = await Timing.showAndTimestamp(() => {
      this.dom.stimulus.textContent = trial.stimulus;
      this.dom.stimulus.className = 'stimulus';
      this.dom.stimulus.style.color = colour;
      this.dom.feedback.textContent = '';
      this.dom.feedback.className = 'feedback';
    });
    // Recorded per trial rather than per session: a tab that was backgrounded
    // for ten trials should lose those ten, not the whole session.
    const onsetDegraded = Timing.state.lastOnsetDegraded;

    const window_ms = block.response_window_ms || 0;
    const first = await this._awaitKey(window_ms || null);
    if (this.aborted) return null;

    // --- No response inside the SC-IAT window ---------------------------
    if (!first) {
      this.dom.feedback.textContent = 'Respond faster';
      this.dom.feedback.className = 'feedback feedback--slow';
      const late = await this._awaitKey(null);
      if (this.aborted) return null;
      await Timing.showAndTimestamp(() => { this.dom.stimulus.textContent = ''; });
      await sleep(ITI_MS);
      return this._record(trial, {
        onset,
        latencyMs: late ? late.timestamp - onset : window_ms,
        latencyToCorrectMs: null,
        correct: false,
        timedOut: true,
        responseKey: late ? late.key : '',
        dispatchDelayMs: late ? late.dispatchDelayMs : null,
        usedEventTimeStamp: late ? late.usedEventTimeStamp : null,
        focusLost: this.focusLosses !== focusBefore,
        nCorrections: 0,
        onsetDegraded,
      });
    }

    const latencyMs = first.timestamp - onset;
    const correct = first.key === trial.correct_key;

    if (correct) {
      await Timing.showAndTimestamp(() => { this.dom.stimulus.textContent = ''; });
      await sleep(ITI_MS);
      return this._record(trial, {
        onset,
        latencyMs,
        latencyToCorrectMs: latencyMs,
        correct: true,
        timedOut: false,
        responseKey: first.key,
        dispatchDelayMs: first.dispatchDelayMs,
        usedEventTimeStamp: first.usedEventTimeStamp,
        focusLost: this.focusLosses !== focusBefore,
        nCorrections: 0,
        onsetDegraded,
      });
    }

    // --- Error: hold the stimulus until the correct key is pressed -------
    this.dom.feedback.textContent = '✗';
    this.dom.feedback.className = 'feedback feedback--error';
    let corrections = 0;
    let corrected = null;
    while (!this.aborted) {
      // eslint-disable-next-line no-await-in-loop
      const next = await this._awaitKey(null);
      if (this.aborted) return null;
      if (!next) break;
      corrections += 1;
      if (next.key === trial.correct_key) { corrected = next; break; }
    }

    await sleep(ERROR_HOLD_MS);
    await Timing.showAndTimestamp(() => {
      this.dom.stimulus.textContent = '';
      this.dom.feedback.textContent = '';
      this.dom.feedback.className = 'feedback';
    });
    await sleep(ITI_MS);

    return this._record(trial, {
      onset,
      latencyMs,
      latencyToCorrectMs: corrected ? corrected.timestamp - onset : null,
      correct: false,
      timedOut: false,
      responseKey: first.key,
      dispatchDelayMs: first.dispatchDelayMs,
      usedEventTimeStamp: first.usedEventTimeStamp,
      focusLost: this.focusLosses !== focusBefore,
      nCorrections: corrections,
      onsetDegraded,
    });
  }

  _colourFor(categoryKey, block) {
    const all = [...block.left, ...block.right];
    const hit = all.find((c) => c.key === categoryKey);
    return hit ? hit.colour : '#f4f4f5';
  }

  _record(trial, m) {
    const frame = Timing.state.frameIntervalMs;
    return {
      index: trial.index,
      block_index: trial.block_index,
      block_role: trial.block_role,
      pairing: trial.pairing,
      stimulus: trial.stimulus,
      stimulus_category: trial.stimulus_category,
      correct_key: trial.correct_key,
      response_key: m.responseKey,
      // Rounded to 3 dp: the underlying clock is coarsened by the browser
      // anyway, and carrying 15 significant figures implies a precision the
      // measurement does not have.
      latency_ms: round3(m.latencyMs),
      latency_to_correct_ms: m.latencyToCorrectMs === null ? null : round3(m.latencyToCorrectMs),
      correct: m.correct,
      timed_out: m.timedOut,
      n_corrections: m.nCorrections,
      onset_uncertainty_ms: frame ? round3(frame / 2) : 0,
      dispatch_delay_ms: m.dispatchDelayMs === null ? null : round3(m.dispatchDelayMs),
      used_event_timestamp: m.usedEventTimeStamp,
      focus_lost: m.focusLost,
      onset_degraded: !!m.onsetDegraded,
    };
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function round3(v) {
  return Math.round(v * 1000) / 1000;
}
