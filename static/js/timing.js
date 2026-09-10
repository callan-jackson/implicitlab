/**
 * timing.js — millisecond measurement in a browser, done honestly.
 *
 * A reaction time is the interval between two physical events: light changing
 * on a display, and a switch closing under a finger. A browser can observe
 * neither directly. What it can observe is when it *handed a frame to the
 * compositor* and when the OS *delivered an input event*. Everything below is
 * about making those two proxies as tight as possible, and then reporting the
 * error that remains instead of pretending it is zero.
 *
 * Three techniques do most of the work:
 *
 * 1. STIMULUS ONSET VIA NESTED requestAnimationFrame.
 *    A rAF callback runs *before* the paint of the frame it belongs to. So
 *    mutating the DOM inside rAF #1 means the change is painted in that frame,
 *    and the timestamp handed to rAF #2 marks the start of the following frame
 *    — which is the first moment the browser can tell us that the previous
 *    frame went out. Timestamping with performance.now() at the moment we set
 *    textContent instead would be wrong by up to a full frame plus layout time,
 *    systematically shortening every latency.
 *
 * 2. RESPONSE TIME VIA event.timeStamp, NOT performance.now().
 *    For trusted events the browser stamps timeStamp when it creates the event,
 *    on the same monotonic clock as performance.now(). Reading performance.now()
 *    inside the handler instead measures when the event loop got round to us,
 *    which adds however long the main thread was busy. The gap between the two
 *    is real and variable, so this file records it per trial as
 *    `dispatchDelayMs` — it is a direct measurement of how much timing error
 *    the naive approach would have introduced.
 *
 * 3. REFRESH-RATE CALIBRATION.
 *    Onset can only be known to within the frame it was painted in. On a 60 Hz
 *    panel that is a 16.67 ms window, so the irreducible uncertainty on any
 *    single onset is about ±8 ms. This is measured at startup rather than
 *    assumed, because 120 Hz and 144 Hz displays are common and would otherwise
 *    silently change the error term.
 *
 * None of this removes display lag — the panel's own processing delay, which is
 * typically 5-20 ms and is invisible to software. That is a constant offset, so
 * it cancels in any within-participant difference score such as D. It would not
 * cancel in an absolute latency claim, which is one reason D is defined as a
 * difference in the first place.
 */

export const Timing = (() => {
  const state = {
    frameIntervalMs: null,
    refreshHz: null,
    frameJitterMs: null,
    supportsHighRes: typeof performance !== 'undefined' && typeof performance.now === 'function',
    eventTimeStampUsable: null,
    calibrationSamples: 0,
    lastOnsetDegraded: false,
    degradedFrames: 0,
  };

  /**
   * Measure the display's frame interval by timing a run of animation frames.
   * The median is used rather than the mean: a single dropped frame doubles one
   * interval and would drag a mean upward, whereas it leaves the median alone.
   */
  function calibrate(sampleFrames = 90) {
    return new Promise((resolve) => {
      const stamps = [];
      // If the tab is not being rendered no frames arrive at all, and a
      // calibration that never resolves would block the whole page. Give up
      // after five seconds and report that the refresh rate is unknown.
      const giveUp = setTimeout(() => {
        state.calibrationSamples = Math.max(0, stamps.length - 1);
        resolve(summary());
      }, 5000);
      function tick(ts) {
        stamps.push(ts);
        if (stamps.length <= sampleFrames) {
          requestAnimationFrame(tick);
        } else {
          clearTimeout(giveUp);
          const deltas = [];
          for (let i = 1; i < stamps.length; i += 1) deltas.push(stamps[i] - stamps[i - 1]);
          deltas.sort((a, b) => a - b);
          const median = deltas[Math.floor(deltas.length / 2)];
          // Median absolute deviation: robust to the occasional dropped frame.
          const devs = deltas.map((d) => Math.abs(d - median)).sort((a, b) => a - b);
          state.frameIntervalMs = median;
          state.frameJitterMs = devs[Math.floor(devs.length / 2)];
          state.refreshHz = median > 0 ? 1000 / median : null;
          state.calibrationSamples = deltas.length;
          resolve(summary());
        }
      }
      requestAnimationFrame(tick);
    });
  }

  function summary() {
    return {
      frameIntervalMs: state.frameIntervalMs,
      refreshHz: state.refreshHz ? Math.round(state.refreshHz * 10) / 10 : null,
      frameJitterMs: state.frameJitterMs,
      onsetUncertaintyMs: state.frameIntervalMs ? state.frameIntervalMs / 2 : null,
      supportsHighRes: state.supportsHighRes,
      eventTimeStampUsable: state.eventTimeStampUsable,
      calibrationSamples: state.calibrationSamples,
      // performance.now() resolution is deliberately coarsened by browsers as a
      // Spectre mitigation. Chrome clamps to 100 microseconds, Firefox to 1 ms
      // by default. Worth reporting because it bounds what "millisecond
      // precision" can mean here.
      clockResolutionNote: detectClockResolution(),
    };
  }

  /** Empirically probe the clock's granularity by looking for the smallest non-zero step. */
  function detectClockResolution() {
    if (!state.supportsHighRes) return 'performance.now() unavailable; timing is unreliable.';
    let smallest = Infinity;
    let last = performance.now();
    for (let i = 0; i < 20000; i += 1) {
      const now = performance.now();
      const delta = now - last;
      if (delta > 0 && delta < smallest) smallest = delta;
      last = now;
    }
    if (!isFinite(smallest)) return 'Clock step below measurement threshold.';
    return `Clock granularity ≈ ${smallest.toFixed(3)} ms.`;
  }

  /**
   * Paint a stimulus and resolve with the best available onset timestamp.
   *
   * @param {() => void} paint  Mutates the DOM to show the stimulus. Called
   *                            inside an animation frame so the change lands in
   *                            a known frame.
   * @param {{stallMs?: number}} opts
   *
   * A watchdog is essential rather than defensive. Browsers stop delivering
   * animation frames to a tab that is not being rendered — backgrounded,
   * minimised, or occluded by another window — and without a timeout the whole
   * task simply stops, mid-trial, with no error and no way for the participant
   * to tell what happened. When the watchdog fires, the stimulus is painted
   * anyway and the onset is timestamped from the clock instead, with
   * `degraded` set so the trial can be flagged rather than quietly scored as
   * if its onset were known to within half a frame.
   */
  function showAndTimestamp(paint, { stallMs = 1200 } = {}) {
    return new Promise((resolve) => {
      let settled = false;
      const finish = (ts, degraded) => {
        if (settled) return;
        settled = true;
        clearTimeout(watchdog);
        state.lastOnsetDegraded = degraded;
        if (degraded) state.degradedFrames += 1;
        resolve(ts);
      };

      const watchdog = setTimeout(() => {
        // rAF never ran, so the paint never happened either. Do it now.
        try { paint(); } catch { /* the caller will surface it */ }
        finish(performance.now(), true);
      }, stallMs);

      requestAnimationFrame(() => {
        paint();
        // Forcing a synchronous layout read here guarantees the style change is
        // committed within this frame rather than being coalesced into the next.
        void document.body.offsetHeight;
        requestAnimationFrame((ts) => finish(ts, false));
      });
    });
  }

  /** Resolve once the tab is actually being rendered again. */
  function whenVisible() {
    if (!document.hidden) return Promise.resolve();
    return new Promise((resolve) => {
      const check = () => {
        if (document.hidden) return;
        document.removeEventListener('visibilitychange', check);
        resolve();
      };
      document.addEventListener('visibilitychange', check);
    });
  }

  /**
   * Normalise a keyboard event into a monotonic timestamp comparable with
   * performance.now().
   *
   * Legacy engines expose event.timeStamp as a Unix epoch value; those are an
   * order of magnitude too large to be a performance timeline value, which is
   * how we detect and reject them.
   */
  function eventTime(ev) {
    const fallback = performance.now();
    const raw = ev.timeStamp;
    const plausible = typeof raw === 'number' && raw > 0 && raw < fallback + 1000;
    if (state.eventTimeStampUsable === null) state.eventTimeStampUsable = plausible;
    return {
      timestamp: plausible ? raw : fallback,
      dispatchDelayMs: plausible ? fallback - raw : null,
      usedEventTimeStamp: plausible,
    };
  }

  /** Keep the tab awake-ness of a session visible; tabbing away invalidates trials. */
  function watchVisibility(onHidden) {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) onHidden();
    });
  }

  /**
   * Warm up the rendering path before the first real trial.
   *
   * The first paint of a new font, or the first style recalculation on a fresh
   * element, is measurably slower than the thousandth. Without a warm-up the
   * first few trials of every session carry an inflated latency, and because
   * block 3 always comes before block 6 that inflation is not symmetric across
   * conditions.
   */
  async function warmUp(el, frames = 20) {
    if (document.fonts && document.fonts.ready) {
      try { await document.fonts.ready; } catch { /* non-fatal */ }
    }
    const original = el.textContent;
    for (let i = 0; i < frames; i += 1) {
      // eslint-disable-next-line no-await-in-loop
      await showAndTimestamp(
        () => { el.textContent = i % 2 ? 'Ready' : 'Ready.'; },
        { stallMs: 120 },
      );
      // No frames are arriving, so there is nothing to warm up and looping
      // would just burn seconds before the first trial.
      if (state.lastOnsetDegraded) break;
    }
    el.textContent = original;
  }

  return {
    calibrate, summary, showAndTimestamp, eventTime, watchVisibility,
    whenVisible, warmUp, state,
  };
})();
