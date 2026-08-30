/**
 * CrowdSentinel — main.js
 *
 * TWO QUANTITIES, TWO MEANINGS. Do not let them merge again.
 *
 *   COUNT     how many people the board's NPU saw. A number, UNKNOWN, or 0.
 *             It is a census, not an alarm.
 *
 *   PRESSURE  P = density x fps^2 x Var(velocity), in s^-2. The alarm
 *             quantity. Headcount enters only through density and is
 *             MULTIPLIED by velocity variance, so 200 people standing still
 *             measure ~0 (NORMAL) while 15 shoving in a doorway can measure
 *             CRITICAL. Any UI that says "risk = how many people" is lying.
 *
 * The risk LEVEL is decided server-side by the SF-CPI state machine and
 * arrives on /count_feed as `level`. This file never recomputes it. The old
 * calcRisk(count) thresholded headcount at 50 and has been deleted; keeping a
 * second, disagreeing risk number in the browser would let this panel and the
 * SNS alert page contradict each other.
 *
 * THE THIRD STATE IS THE POINT. Both quantities can be UNKNOWN, and unknown
 * is never rendered as 0, never as a calm colour, never as a low percentage.
 * "0 people" and "0.0000 s^-2" read as a safe, empty scene — precisely the
 * wrong thing to say when the truth is that we have stopped being able to
 * see. Unknown renders as an em dash and a hatched fill, everywhere.
 */

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const CONFIG = {
    // NOTE: there is no crowd/headcount threshold here any more, and none
    // should be added. The only thresholds on this page are pressures, and
    // they come from the server so the UI and the state machine cannot
    // disagree about where ELEVATED starts.
    HISTORY_LIMIT: 300,          // ~2.5 min of timeline at the 2 Hz feed
    SCALE_HEADROOM: 1.25,        // threshold scale runs 0 .. 1.25 x critical
    ENDPOINTS: {
        START:  '/start_stream',
        STOP:   '/stop_stream',
        VIDEO:  '/video_feed',
        STATUS: '/status',
        SSE:    '/count_feed',
    },
};

const RUNNING_STATUSES = ['connecting', 'live', 'no_video'];

// The server's vocabulary, used verbatim. There is deliberately no
// translation table (the old one mapped normal->low, elevated->moderate):
// two names for one level is how a UI drifts out of sync with its backend.
const LEVELS = ['unknown', 'normal', 'elevated', 'high', 'critical'];

const LEVEL_COLOR = {
    unknown:  'var(--lv-unknown)',
    normal:   'var(--lv-normal)',
    elevated: 'var(--lv-elevated)',
    high:     'var(--lv-high)',
    critical: 'var(--lv-critical)',
};

const LEVEL_COPY = {
    unknown:  'Pressure unmeasured — the state machine is sensor-blind.',
    normal:   'Normal — pressure is below the elevated threshold.',
    elevated: 'Elevated — pressure has reached the elevated band. Watch the hot cells.',
    high:     'High — sustained pressure in the crowd. Consider crowd management.',
    critical: 'Critical — turbulent crowd pressure. Act now.',
};

// ─── PURE HELPERS ────────────────────────────────────────────────────────────
// Everything above the DOM marker is side-effect free and testable in node.

/** A pressure, or null. Never coerces null/NaN into a number. */
function toPressure(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Four decimals: the whole decision range (0.010 .. 0.040) lives in the
 *  third and fourth places, so fewer digits would hide the signal. */
function formatPressure(value) {
    const p = toPressure(value);
    return p === null ? '—' : p.toFixed(4);
}

/** Thresholds, or null if the payload did not carry a usable set. */
function readThresholds(thresholds) {
    if (!thresholds) return null;
    const e = toPressure(thresholds.elevated);
    const h = toPressure(thresholds.high);
    const c = toPressure(thresholds.critical);
    if (e === null || h === null || c === null) return null;
    if (!(e < h && h < c) || c <= 0) return null;
    return { elevated: e, high: h, critical: c };
}

/** Pressure as a percentage of the critical threshold — for DISPLAY only.
 *  Returns null when there is nothing to show: an unknown pressure must not
 *  come back as 0, which would read as "calm" on every dial it feeds. */
function pressureToPercent(pressure, thresholds) {
    const p = toPressure(pressure);
    const t = readThresholds(thresholds);
    if (p === null || t === null) return null;
    return Math.max(0, Math.round((p / t.critical) * 100));
}

/** Position on the threshold scale, 0-100, or null. The scale runs to
 *  1.25 x critical so the critical tick sits inboard of the end and an
 *  over-critical reading still has somewhere to go. */
function scalePosition(pressure, thresholds) {
    const p = toPressure(pressure);
    const t = readThresholds(thresholds);
    if (p === null || t === null) return null;
    const full = t.critical * CONFIG.SCALE_HEADROOM;
    return Math.max(0, Math.min(100, (p / full) * 100));
}

// The pressure ramp. Perceptually ordered (lightness increases monotonically)
// and ANCHORED ON THE THRESHOLDS, so a change of hue is a change of level
// rather than a decorative gradient. Deliberately not a rainbow: a rainbow
// invents boundaries the data does not have.
const RAMP_STOPS = [
    [22, 32, 42],      // measured, calm
    [58, 42, 110],     // elevated onset
    [140, 47, 107],    // high onset
    [217, 84, 43],     // critical onset
    [255, 233, 168],   // well over critical: white-hot
];

function rampAnchors(t) {
    return [0, t.elevated / t.critical, t.high / t.critical, 1, CONFIG.SCALE_HEADROOM * 1.2];
}

/** [r,g,b] for a pressure, or null when the cell was not measured. null is
 *  the caller's cue to hatch the cell instead of colouring it. */
function rampColor(pressure, thresholds) {
    const p = toPressure(pressure);
    const t = readThresholds(thresholds);
    if (p === null || t === null) return null;

    const x = Math.max(0, p / t.critical);
    const anchors = rampAnchors(t);
    const last = anchors.length - 1;
    if (x >= anchors[last]) return RAMP_STOPS[last].slice();

    for (let i = 0; i < last; i++) {
        const a = anchors[i];
        const b = anchors[i + 1];
        if (x <= b) {
            const span = b - a;
            const f = span > 0 ? (x - a) / span : 0;
            const c0 = RAMP_STOPS[i];
            const c1 = RAMP_STOPS[i + 1];
            return [
                Math.round(c0[0] + (c1[0] - c0[0]) * f),
                Math.round(c0[1] + (c1[1] - c0[1]) * f),
                Math.round(c0[2] + (c1[2] - c0[2]) * f),
            ];
        }
    }
    return RAMP_STOPS[last].slice();
}

/** Cell alpha. Calm cells stay translucent so the video reads through them;
 *  hot cells go opaque. In "field only" mode every measured cell is solid. */
function cellAlpha(pressure, thresholds, fieldOnly) {
    if (fieldOnly) return 1;
    const p = toPressure(pressure);
    const t = readThresholds(thresholds);
    if (p === null || t === null) return 0;
    const x = Math.max(0, Math.min(1, p / t.critical));
    return 0.16 + 0.64 * x;
}

/** Identity of an alert, for change detection. `last_alert` is re-sent on
 *  every snapshot and is never cleared server-side, so the only way to know
 *  a NEW alert fired is that this key changed. */
function alertKey(alert) {
    if (!alert) return '';
    return [alert.timestamp, alert.level, alert.previous_level, alert.reason].join('|');
}

/** "12s" / "4m 20s" / "1h 06m" from milliseconds. */
function formatElapsed(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
    return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

/**
 * How long ago the alert arrived.
 *
 * `event.timestamp` is the SF-CPI stream clock (seconds since the frame
 * source started, or a monotonic reading) — NOT a wall-clock epoch, so it
 * cannot be differenced against Date.now(). What the browser can honestly
 * say is when IT received the alert, so that is what this reports. A null
 * receivedAt means the alert was already on the first snapshot: it predates
 * this page load and its age is genuinely unknown.
 */
function formatAlertAge(receivedAt, now) {
    if (receivedAt === null || receivedAt === undefined) return 'before this session';
    return `received ${formatElapsed(now - receivedAt)} ago`;
}

// ─── DOM ─────────────────────────────────────────────────────────────────────
// Everything below this marker touches the document.

const $ = id => document.getElementById(id);
const dom = {
    // rail
    systemTime:   $('systemTime'),
    liveBadge:    $('liveBadge'),
    connText:     $('connText'),
    feedDot:      $('feedDot'),
    btnStream:    $('btnStreamToggle'),
    // stage
    stage:        $('stage'),
    videoFeed:    $('videoFeed'),
    lattice:      $('lattice'),
    gridNote:     $('gridNote'),
    btnOverlay:   $('btnOverlay'),
    btnFieldOnly: $('btnFieldOnly'),
    stageDot:     $('stageDot'),
    feedStatusText: $('feedStatusText'),
    peakCellTag:  $('peakCellTag'),
    legendMax:    $('legendMax'),
    // pressure card
    pressureCard: $('pressureCard'),
    pressureVal:  $('pressureVal'),
    levelWord:    $('levelWord'),
    scaleTrack:   $('scaleTrack'),
    scaleFill:    $('scaleFill'),
    scaleNeedle:  $('scaleNeedle'),
    scaleTicks:   $('scaleTicks'),
    meanPressure: $('meanPressure'),
    coverageVal:  $('coverageVal'),
    // count card
    personCount:  $('personCount'),
    countUnit:    $('countUnit'),
    countReason:  $('countReason'),
    peakVal:      $('peakVal'),
    statAvg:      $('statAvg'),
    // last alert card
    alertEmpty:   $('alertEmpty'),
    alertBody:    $('alertBody'),
    alertLevel:   $('alertLevel'),
    alertPrev:    $('alertPrev'),
    alertReason:  $('alertReason'),
    alertAgo:     $('alertAgo'),
    alertMessage: $('alertMessage'),
    // banner
    alertStrip:   $('alertStrip'),
    alertIcon:    $('alertIcon'),
    alertMsg:     $('alertMsg'),
    alertVal:     $('alertVal'),
    // session strip
    statPeakP:    $('statPeakP'),
    statAlerts:   $('statAlerts'),
    statFrames:   $('statFrames'),
    statUptime:   $('statUptime'),
    // klaxon
    criticalOverlay: $('criticalOverlay'),
    klaxonSub:    $('klaxonSub'),
    klaxonDismiss: $('klaxonDismiss'),
};

// ─── STATE ───────────────────────────────────────────────────────────────────
const state = {
    isRunning:   false,
    status:      'idle',
    known:       false,
    count:       null,
    peakCount:   null,
    peakPressure: null,
    history:     [],          // known counts only
    alerts:      0,
    startTime:   null,
    criticalAck: false,
    sse:         null,
    videoAttached: false,
    fieldOnly:   false,
    // lattice
    gridRows:    0,
    gridCols:    0,
    cellEls:     [],
    latticeGeom: '',
    // last_alert change detection. `undefined` means "no snapshot seen yet",
    // which is different from `''` ("a snapshot said there is no alert").
    alertKey:    undefined,
    alertReceivedAt: null,
    lastAlertTimestamp: null,
    thresholds:  null,
    ticksDrawn:  '',
};

// ─── SYSTEM CONTROL ──────────────────────────────────────────────────────────
async function toggleSystem() {
    if (state.isRunning) await stopSystem();
    else await startSystem();
}
window.toggleSystem = toggleSystem;

async function startSystem() {
    dom.feedStatusText.textContent = 'Connecting to KVS…';
    dom.btnStream.disabled = true;
    try {
        // Returns immediately — the connect takes seconds in a worker thread.
        // Real progress arrives over the SSE status.
        const res = await fetch(CONFIG.ENDPOINTS.START, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'started' || data.status === 'already_running') {
            state.startTime = Date.now();
            attachVideo();
        }
    } catch (e) {
        console.error('Failed to start:', e);
        dom.feedStatusText.textContent = 'Could not reach the server';
    } finally {
        dom.btnStream.disabled = false;
    }
}

async function stopSystem() {
    dom.btnStream.disabled = true;
    try {
        await fetch(CONFIG.ENDPOINTS.STOP, { method: 'POST' });
        state.startTime = null;
        detachVideo();
        // Standby means both quantities are UNKNOWN, not zero: the board is
        // simply no longer being listened to. Showing "0 people" or
        // "0.0000 s^-2" here would claim a calm scene we have not observed.
        renderUnknownCount('stream stopped');
        renderPressureUnavailable('standby');
    } catch (e) {
        console.error('Error stopping:', e);
    } finally {
        dom.btnStream.disabled = false;
    }
}

function attachVideo() {
    if (state.videoAttached) return;
    dom.videoFeed.src = `${CONFIG.ENDPOINTS.VIDEO}?t=${Date.now()}`;
    state.videoAttached = true;
    dom.stage.classList.add('is-live');
}

function detachVideo() {
    dom.videoFeed.removeAttribute('src');
    state.videoAttached = false;
    dom.stage.classList.remove('is-live');
}

// ─── THE PRESSURE LATTICE (signature element) ────────────────────────────────
//
// SF-CPI computes a pressure per grid cell. A single global number throws
// away WHERE the crowd is being squeezed, which is the operationally useful
// part — so the lattice is drawn over the frame it was measured on.

function setFieldOnly(on) {
    state.fieldOnly = on;
    dom.stage.classList.toggle('is-field', on);
    dom.btnOverlay.classList.toggle('is-on', !on);
    dom.btnFieldOnly.classList.toggle('is-on', on);
    dom.btnOverlay.setAttribute('aria-pressed', String(!on));
    dom.btnFieldOnly.setAttribute('aria-pressed', String(on));
}

function buildLattice(rows, cols) {
    if (rows === state.gridRows && cols === state.gridCols) return;
    dom.lattice.textContent = '';
    state.cellEls = [];
    state.gridRows = rows;
    state.gridCols = cols;
    if (!rows || !cols) {
        dom.gridNote.textContent = 'grid —';
        return;
    }
    dom.lattice.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    dom.lattice.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
    const frag = document.createDocumentFragment();
    for (let i = 0; i < rows * cols; i++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        frag.appendChild(cell);
        state.cellEls.push(cell);
    }
    dom.lattice.appendChild(frag);
    dom.gridNote.textContent = `grid ${cols} × ${rows}`;
}

/**
 * The lattice must sit on the pixels it was measured on.
 *
 * The video panel publishes the FULL frame, but the pipeline crops it down to
 * a whole number of cells before measuring (`frame.image[:h, :w]` in
 * server.py, with w,h rounded down to a multiple of the cell size). A naive
 * 100%/100% overlay is therefore off by up to one cell — enough to blame the
 * wrong doorway. Cells are square, so the cell size is recoverable from the
 * frame's natural dimensions and the grid shape.
 */
function positionLattice() {
    const img = dom.videoFeed;
    const nw = img.naturalWidth || 0;
    const nh = img.naturalHeight || 0;
    const cols = state.gridCols;
    const rows = state.gridRows;

    let geom = '100% 100% 16/9';
    let w = '100%', h = '100%', aspect = '';

    if (nw > 0 && nh > 0) {
        aspect = `${nw} / ${nh}`;
        if (cols > 0 && rows > 0) {
            const cell = Math.min(Math.floor(nw / cols), Math.floor(nh / rows));
            if (cell > 0) {
                w = ((cols * cell / nw) * 100).toFixed(3) + '%';
                h = ((rows * cell / nh) * 100).toFixed(3) + '%';
            }
        }
        geom = `${w} ${h} ${aspect}`;
    }

    if (geom === state.latticeGeom) return;
    state.latticeGeom = geom;
    dom.lattice.style.width = w;
    dom.lattice.style.height = h;
    if (aspect) dom.stage.style.setProperty('--stage-aspect', aspect);
}

/** Paint the lattice. `cells` null (or absent) means the whole field is
 *  unmeasured — every cell is hatched, none is coloured. */
function renderLattice(cells, gridShape, thresholds) {
    let rows = 0, cols = 0;
    if (Array.isArray(gridShape) && gridShape.length === 2) {
        rows = Number(gridShape[0]) || 0;
        cols = Number(gridShape[1]) || 0;
    } else if (Array.isArray(cells) && cells.length && Array.isArray(cells[0])) {
        rows = cells.length;
        cols = cells[0].length;
    }

    // No shape has ever been reported: draw nothing rather than invent a
    // grid. The pressure card is already saying UNKNOWN in words.
    if (!rows || !cols) {
        if (state.gridRows || state.gridCols) buildLattice(0, 0);
        dom.peakCellTag.textContent = 'peak cell —';
        return;
    }

    buildLattice(rows, cols);
    positionLattice();

    const haveCells = Array.isArray(cells) && cells.length === rows;
    let peakIdx = -1;
    let peakVal = null;

    for (let r = 0; r < rows; r++) {
        const row = haveCells && Array.isArray(cells[r]) ? cells[r] : null;
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const el = state.cellEls[idx];
            if (!el) continue;
            const p = row ? toPressure(row[c]) : null;
            const rgb = rampColor(p, thresholds);
            if (rgb === null) {
                // Unmeasured: hatch it. Colour on this page means a pressure
                // was actually computed, so an unknown cell never gets one.
                el.style.backgroundColor = '';
                if (!el.classList.contains('cell--unmeasured')) {
                    el.classList.add('cell--unmeasured');
                }
            } else {
                if (el.classList.contains('cell--unmeasured')) {
                    el.classList.remove('cell--unmeasured');
                }
                const a = cellAlpha(p, thresholds, state.fieldOnly).toFixed(3);
                el.style.backgroundColor = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
            }
            if (p !== null && (peakVal === null || p > peakVal)) {
                peakVal = p;
                peakIdx = idx;
            }
            el.classList.remove('cell--peak');
        }
    }

    if (peakIdx >= 0 && state.cellEls[peakIdx]) {
        state.cellEls[peakIdx].classList.add('cell--peak');
        const r = Math.floor(peakIdx / cols) + 1;
        const c = (peakIdx % cols) + 1;
        dom.peakCellTag.textContent =
            `peak cell r${r} c${c} · ${formatPressure(peakVal)} s⁻²`;
    } else {
        dom.peakCellTag.textContent = 'peak cell — · field unmeasured';
    }
}

// ─── THRESHOLD SCALE ─────────────────────────────────────────────────────────
/** Draw the tick marks. Only the RISE thresholds are sent, so only they are
 *  drawn; the note in the markup explains that a level is held until pressure
 *  falls below a lower release threshold, which is why the needle can sit
 *  under a tick while the level stays put. */
function drawTicks(thresholds) {
    const t = readThresholds(thresholds);
    if (!t) return;
    const key = `${t.elevated}|${t.high}|${t.critical}`;
    if (key === state.ticksDrawn) return;
    state.ticksDrawn = key;

    const full = t.critical * CONFIG.SCALE_HEADROOM;
    dom.scaleTicks.textContent = '';
    [['elevated', t.elevated], ['high', t.high], ['critical', t.critical]]
        .forEach(([name, value]) => {
            const el = document.createElement('span');
            el.className = 'tick';
            el.style.left = ((value / full) * 100).toFixed(2) + '%';
            const v = document.createElement('span');
            v.className = 'tick__v tnum';
            v.textContent = value.toFixed(3);
            const k = document.createElement('span');
            k.className = 'tick__k';
            k.textContent = name;
            el.appendChild(v);
            el.appendChild(k);
            dom.scaleTicks.appendChild(el);
        });

    dom.legendMax.innerHTML = `${t.critical.toFixed(3)} s&#8315;&sup2;`;
}

// ─── PRESSURE / LEVEL RENDERING ──────────────────────────────────────────────
//
// The RISK half of the page. It is driven by the server's `level` and
// `pressure` and never by the count — the two are rendered from separate
// functions on purpose, so that a blind sensor on one side cannot silently
// blank or fake the other.

function setLevelColor(el, level) {
    el.style.setProperty('--lv', LEVEL_COLOR[level] || LEVEL_COLOR.unknown);
}

function renderPressure(snap) {
    const thresholds = snap.thresholds;
    const t = readThresholds(thresholds);
    if (t) {
        state.thresholds = t;
        drawTicks(thresholds);
    }

    const running = RUNNING_STATUSES.includes(snap.status);
    const level = LEVELS.includes(snap.level) ? snap.level : 'unknown';
    const peak = toPressure(snap.max_pressure);
    const mean = toPressure(snap.pressure);

    // --- the number -------------------------------------------------------
    dom.pressureVal.textContent = formatPressure(peak);
    dom.pressureVal.classList.toggle('is-unknown', peak === null);

    // --- the word ---------------------------------------------------------
    // At standby the server parks `level` at "normal" as housekeeping, not as
    // a measurement. Printing NORMAL over a dash would be a reassuring lie,
    // so a stopped stream says STANDBY and a running one with no reading yet
    // says so explicitly.
    let word, wordLevel;
    if (!running) {
        word = 'Standby';
        wordLevel = 'unknown';
    } else if (level === 'unknown') {
        word = 'Unknown';
        wordLevel = 'unknown';
    } else if (peak === null) {
        word = `${level} · no reading`;
        wordLevel = 'unknown';
    } else {
        word = level;
        wordLevel = level;
    }
    dom.levelWord.textContent = word;
    setLevelColor(dom.levelWord, wordLevel);
    setLevelColor(dom.pressureCard, wordLevel);

    // --- the scale --------------------------------------------------------
    const pos = scalePosition(peak, thresholds);
    if (pos === null) {
        // Hatched track, no fill, no needle. A zero-width fill would read as
        // "measured, and calm".
        dom.scaleTrack.classList.add('is-unknown');
        dom.scaleFill.style.width = '0%';
        dom.scaleNeedle.classList.add('is-hidden');
    } else {
        dom.scaleTrack.classList.remove('is-unknown');
        dom.scaleFill.style.width = pos.toFixed(2) + '%';
        dom.scaleNeedle.classList.remove('is-hidden');
        dom.scaleNeedle.style.left = pos.toFixed(2) + '%';
    }

    // --- supporting facts -------------------------------------------------
    dom.meanPressure.textContent = mean === null ? '—' : formatPressure(mean);
    const cov = toPressure(snap.coverage);
    dom.coverageVal.textContent = cov === null ? '—' : `${Math.round(cov * 100)}%`;

    if (peak !== null && (state.peakPressure === null || peak > state.peakPressure)) {
        state.peakPressure = peak;
        dom.statPeakP.textContent = formatPressure(peak);
    }

    renderLattice(snap.cells, snap.grid_shape, thresholds);
    renderBanner(running, level, peak, thresholds);
    renderKlaxon(running, level, peak, thresholds);
}

/** No pressure at all — standby, or the server went quiet. Distinct from
 *  renderUnknownCount: the field and the census fail independently. */
function renderPressureUnavailable(reason) {
    dom.pressureVal.textContent = '—';
    dom.pressureVal.classList.add('is-unknown');
    dom.levelWord.textContent = reason === 'standby' ? 'Standby' : 'Unknown';
    setLevelColor(dom.levelWord, 'unknown');
    setLevelColor(dom.pressureCard, 'unknown');

    dom.scaleTrack.classList.add('is-unknown');
    dom.scaleFill.style.width = '0%';
    dom.scaleNeedle.classList.add('is-hidden');

    dom.meanPressure.textContent = '—';
    dom.coverageVal.textContent = '—';

    renderLattice(null, null, null);

    dom.alertStrip.className = 'banner is-unknown';
    setLevelColor(dom.alertStrip, 'unknown');
    dom.alertIcon.textContent = '■';
    dom.alertMsg.textContent = reason === 'standby'
        ? 'Standby — the pressure field is not being measured.'
        : 'No contact with the server — pressure is unknown.';
    dom.alertVal.textContent = '';

    dom.criticalOverlay.hidden = true;
}

function renderBanner(running, level, peak, thresholds) {
    let cls, msg, mark, lv;

    if (!running) {
        lv = 'unknown';
        cls = 'banner is-unknown';
        mark = '■';
        msg = 'Standby — the pressure field is not being measured.';
    } else if (level === 'unknown') {
        lv = 'unknown';
        cls = 'banner is-unknown';
        mark = '■';
        msg = LEVEL_COPY.unknown;
    } else if (peak === null) {
        // Never pair a null pressure with copy that claims active monitoring.
        lv = 'unknown';
        cls = 'banner is-unknown';
        mark = '■';
        msg = 'No pressure reading yet — waiting for enough frames to measure the field.';
    } else {
        lv = level;
        cls = `banner is-${level}`;
        mark = level === 'normal' ? '■' : '▲';
        msg = LEVEL_COPY[level] || LEVEL_COPY.unknown;
    }

    dom.alertStrip.className = cls;
    setLevelColor(dom.alertStrip, lv);
    dom.alertIcon.textContent = mark;
    dom.alertMsg.textContent = msg;

    // The evidence, always: the value and where it sits relative to critical.
    const pct = pressureToPercent(peak, thresholds);
    const t = readThresholds(thresholds);
    dom.alertVal.textContent = (pct === null || t === null)
        ? 'no pressure reading'
        : `P ${formatPressure(peak)} s⁻² · ${pct}% of critical ${t.critical.toFixed(3)}`;
}

function renderKlaxon(running, level, peak, thresholds) {
    const t = readThresholds(thresholds);
    // A klaxon is only ever raised on a real, measured CRITICAL.
    const critical = running && level === 'critical' && peak !== null;
    if (critical && !state.criticalAck) {
        if (dom.criticalOverlay.hidden) {
            dom.criticalOverlay.hidden = false;
            if (dom.klaxonDismiss) dom.klaxonDismiss.focus();
        }
        dom.klaxonSub.textContent = t
            ? `Peak cell pressure is ${formatPressure(peak)} s⁻², over the critical threshold of ${t.critical.toFixed(3)} s⁻². This is crowd turbulence, not a headcount.`
            : `Peak cell pressure is ${formatPressure(peak)} s⁻².`;
    }
    if (!critical) {
        state.criticalAck = false;
        dom.criticalOverlay.hidden = true;
    }
}

window.dismissCritical = () => {
    state.criticalAck = true;
    dom.criticalOverlay.hidden = true;
    if (dom.btnStream) dom.btnStream.focus();
};

// ─── LAST ALERT ──────────────────────────────────────────────────────────────
function renderLastAlert(alert) {
    const key = alertKey(alert);

    if (state.alertKey === undefined) {
        // First snapshot of this page load. `last_alert` is never cleared
        // server-side, so anything present here may predate the session: seed
        // the key WITHOUT counting it as an alert this session fired, and
        // report its age as unknown rather than inventing "0s ago".
        state.alertKey = key;
        state.alertReceivedAt = null;
    } else if (key !== state.alertKey) {
        state.alertKey = key;
        if (alert) {
            state.alertReceivedAt = Date.now();
            state.alerts++;
            dom.statAlerts.textContent = state.alerts;
        } else {
            state.alertReceivedAt = null;
        }
    }

    if (!alert) {
        state.lastAlertTimestamp = null;
        dom.alertEmpty.hidden = false;
        dom.alertBody.hidden = true;
        return;
    }

    dom.alertEmpty.hidden = true;
    dom.alertBody.hidden = false;

    state.lastAlertTimestamp = alert.timestamp;

    const lvl = LEVELS.includes(alert.level) ? alert.level : 'unknown';
    dom.alertLevel.textContent = lvl;
    setLevelColor(dom.alertLevel, lvl);
    dom.alertPrev.textContent = alert.previous_level || '—';
    dom.alertReason.textContent = alert.reason || '—';
    dom.alertMessage.textContent = alert.message || '';
    refreshAlertAge();
}

/** Re-rendered once a second by the clock so the age does not go stale
 *  between the 2 Hz snapshots. */
function refreshAlertAge() {
    if (dom.alertBody.hidden) return;
    const ts = toPressure(state.lastAlertTimestamp);
    const age = formatAlertAge(state.alertReceivedAt, Date.now());
    dom.alertAgo.textContent = ts === null ? age : `${age} · stream t=${ts.toFixed(1)}s`;
}

// ─── COUNT RENDERING ─────────────────────────────────────────────────────────
//
// The CENSUS half of the page. Nothing here colours anything by level, sets a
// risk state, or raises an alert. The count used to turn red past 50 and drive
// a "CROWD DENSITY / SPARSE-DENSE-CRITICAL" gauge; both were headcount dressed
// up as risk, and the gauge's job is now done properly by the pressure
// lattice, which measures density where it actually matters.

function renderKnownCount(count) {
    state.known = true;
    state.count = count;

    dom.personCount.textContent = count;
    dom.personCount.classList.remove('is-unknown');
    dom.countUnit.textContent = count === 1 ? 'person' : 'people';
    dom.countReason.textContent = 'board NPU · live';

    if (state.peakCount === null || count > state.peakCount) {
        state.peakCount = count;
        dom.peakVal.textContent = count;
    }

    state.history.push(count);
    if (state.history.length > CONFIG.HISTORY_LIMIT) state.history.shift();
    const avg = state.history.reduce((s, n) => s + n, 0) / state.history.length;
    dom.statAvg.textContent = avg.toFixed(1);
}

/** No usable count. Count-derived UI is blanked rather than zeroed, and
 *  nothing is recorded into the history or the average. This function must
 *  NOT touch the pressure card, the lattice or the banner: "the board stopped
 *  reporting a count" and "the state machine has no pressure" are different
 *  failures, and blanking one because of the other hides a live risk. */
function renderUnknownCount(reason) {
    state.known = false;
    state.count = null;

    dom.personCount.textContent = '—';
    dom.personCount.classList.add('is-unknown');
    dom.countUnit.textContent = 'no data';
    dom.countReason.textContent = reason || 'no board metadata';
}

// ─── TIMELINE ────────────────────────────────────────────────────────────────
const chartCtx = document.getElementById('timelineChart').getContext('2d');

const RULE = (color) => ({
    data: [], borderColor: color, borderWidth: 1, borderDash: [5, 4],
    pointRadius: 0, fill: false, yAxisID: 'p', spanGaps: true, order: 3,
});

const chartData = {
    labels: [],
    datasets: [
        {   // the alarm quantity, on its own axis, in the ramp's ember
            label: 'Peak cell pressure (s⁻²)',
            data: [], borderColor: '#d9542b',
            backgroundColor: 'rgba(217,84,43,0.10)',
            borderWidth: 1.6, fill: true, tension: 0.25, pointRadius: 0,
            spanGaps: false,       // an unknown pressure is a GAP, not a dip to 0
            yAxisID: 'p', order: 1,
        },
        {   // the census, greyscale, secondary axis — it is context, not alarm
            label: 'People in frame',
            data: [], borderColor: 'rgba(169,179,185,0.55)',
            borderWidth: 1, fill: false, tension: 0.25, pointRadius: 0,
            spanGaps: false,       // an unknown count is a GAP, not a dip to 0
            yAxisID: 'n', order: 2,
        },
        RULE('rgba(139,120,230,0.45)'),   // elevated
        RULE('rgba(228,113,63,0.45)'),    // high
        RULE('rgba(217,84,43,0.75)'),     // critical
    ],
};

const chart = new Chart(chartCtx, {
    type: 'line',
    data: chartData,
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#14181b',
                borderColor: '#232a2e',
                borderWidth: 1,
                titleFont: { family: "'IBM Plex Mono'", size: 10 },
                bodyFont: { family: "'IBM Plex Mono'", size: 11 },
                filter: item => item.datasetIndex < 2,
            },
        },
        scales: {
            x: { display: false },
            p: {
                position: 'left',
                beginAtZero: true,
                grid: { color: 'rgba(255,255,255,0.035)' },
                ticks: {
                    color: '#79838a',
                    font: { family: "'IBM Plex Mono'", size: 9 },
                    callback: v => Number(v).toFixed(3),
                },
            },
            n: {
                position: 'right',
                beginAtZero: true,
                grid: { display: false },
                ticks: { color: '#4d565c', font: { family: "'IBM Plex Mono'", size: 9 } },
            },
        },
    },
});

/** One sample per snapshot. Either value may be null, and null is plotted as
 *  a literal gap — drawing it as 0 would put a reassuring dip in the timeline
 *  at exactly the moment the system stopped being able to see. */
function recordSample(count, pressure) {
    if (!state.isRunning && count === null && pressure === null) return;

    const t = state.thresholds;
    chartData.labels.push(new Date().toLocaleTimeString());
    chartData.datasets[0].data.push(toPressure(pressure));
    chartData.datasets[1].data.push(typeof count === 'number' ? count : null);
    chartData.datasets[2].data.push(t ? t.elevated : null);
    chartData.datasets[3].data.push(t ? t.high : null);
    chartData.datasets[4].data.push(t ? t.critical : null);

    if (chartData.labels.length > CONFIG.HISTORY_LIMIT) {
        chartData.labels.shift();
        chartData.datasets.forEach(d => d.data.shift());
    }

    // Keep the threshold rules on screen during a calm session, or a 0.0005
    // trace autoscales them off the top and the operator loses the reference.
    chart.options.scales.p.suggestedMax = t ? t.critical * 1.2 : undefined;
    chart.update('none');
}

// ─── CLOCK ───────────────────────────────────────────────────────────────────
function tickClock() {
    const now = new Date();
    dom.systemTime.textContent = now.toLocaleTimeString('en-IN', {
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    });

    if (state.isRunning && state.startTime) {
        const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
        const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const ss = String(elapsed % 60).padStart(2, '0');
        dom.statUptime.textContent = `${mm}:${ss}`;
    }
    refreshAlertAge();
}
setInterval(tickClock, 1000);
tickClock();

// ─── STATUS ──────────────────────────────────────────────────────────────────
const BADGE_TEXT = {
    idle:       'Offline',
    connecting: 'Connecting',
    live:       'Live',
    no_video:   'No video',
    error:      'Error',
};

/** A new session starts with clean session statistics. Carrying the previous
 *  run's peak pressure into a fresh one would attribute an old crowd's crush
 *  to this one. */
function resetSessionStats() {
    state.peakCount = null;
    state.peakPressure = null;
    state.history = [];
    state.alerts = 0;
    dom.peakVal.textContent = '\u2014';
    dom.statAvg.textContent = '\u2014';
    dom.statPeakP.textContent = '\u2014';
    dom.statAlerts.textContent = '0';
    chartData.labels.length = 0;
    chartData.datasets.forEach(d => { d.data.length = 0; });
}

function applySnapshot(snap) {
    state.status = snap.status;
    const running = RUNNING_STATUSES.includes(snap.status);

    if (running && !state.isRunning) resetSessionStats();
    if (running && !state.startTime) state.startTime = Date.now();
    state.isRunning = running;

    dom.btnStream.textContent = running ? 'Stop monitoring' : 'Start monitoring';

    dom.liveBadge.classList.toggle('is-live', snap.status === 'live');
    dom.connText.textContent = BADGE_TEXT[snap.status] || String(snap.status).toUpperCase();
    dom.feedDot.classList.toggle('is-on', snap.status === 'live');
    dom.stageDot.classList.toggle('is-on', snap.status === 'live');

    dom.feedStatusText.textContent = snap.detail
        ? `${snap.status_text} — ${snap.detail}`
        : snap.status_text;

    dom.statFrames.textContent = `${snap.frames ?? 0} / ${snap.dropped ?? 0}`;

    if (running) attachVideo(); else if (snap.status === 'idle') detachVideo();

    // Two independent renders, in this order, from two different fields.
    // Neither is derived from the other.
    if (snap.known && snap.count !== null && snap.count !== undefined) {
        renderKnownCount(snap.count);
    } else {
        renderUnknownCount(snap.count_reason);
    }

    renderPressure(snap);
    renderLastAlert(snap.last_alert || null);

    recordSample(
        snap.known && typeof snap.count === 'number' ? snap.count : null,
        toPressure(snap.max_pressure),
    );
}

// ─── SSE ─────────────────────────────────────────────────────────────────────
function connectSSE() {
    if (state.sse) state.sse.close();
    state.sse = new EventSource(CONFIG.ENDPOINTS.SSE);

    state.sse.onmessage = (event) => {
        let snap;
        try {
            snap = JSON.parse(event.data);
        } catch (err) {
            console.error('bad SSE payload', err);
            return;   // an unparseable payload is unknown, never zero
        }
        applySnapshot(snap);
    };

    state.sse.onerror = () => {
        // The server itself has gone quiet. Both quantities are unknown —
        // and the risk half must say so out loud rather than keep showing
        // the last level as if it were still being measured.
        renderUnknownCount('dashboard lost contact with the server');
        renderPressureUnavailable('disconnected');
        dom.feedStatusText.textContent = 'Server unreachable';
        if (state.sse) state.sse.close();
        setTimeout(connectSSE, 2000);
    };
}

// ─── VIEW SWITCH ─────────────────────────────────────────────────────────────
dom.btnOverlay.addEventListener('click', () => setFieldOnly(false));
dom.btnFieldOnly.addEventListener('click', () => setFieldOnly(true));

// Escape acknowledges the klaxon, so a keyboard operator is not trapped.
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !dom.criticalOverlay.hidden) window.dismissCritical();
});

// The MJPEG stream reports its natural size only once a frame has decoded;
// re-measure then, so the lattice lands on the pixels it was computed from.
dom.videoFeed.addEventListener('load', () => {
    state.latticeGeom = '';
    positionLattice();
});

// ─── BOOTSTRAP ───────────────────────────────────────────────────────────────
// The feed runs from page load, not from the Start button: standby is a real
// state, with a real (unknown) count and a real (unmeasured) field.
setFieldOnly(false);
renderUnknownCount('waiting for the first status');
renderPressureUnavailable('standby');
connectSSE();
