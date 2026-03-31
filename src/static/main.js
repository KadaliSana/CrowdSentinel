/**
 * CrowdSentinel — main.js
 * Integrated with Python Flask Backend
 */

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const CONFIG = {
    CROWD_THRESHOLD: 50,
    RISK_CRITICAL:  80,
    HISTORY_LIMIT:  60,
    ENDPOINTS: {
        START: '/start_stream',
        STOP: '/stop_stream',
        UPLOAD: '/upload',
        VIDEO: '/video_feed',
        SSE: '/count_feed'
    }
};

// ─── STATE ────────────────────────────────────────────────────────────────────
const state = {
    isRunning:   false,
    count:       0,
    peak:        0,
    history:     [], 
    alerts:      0,
    startTime:   null, // Set when stream starts
    criticalAck: false,
    sse:         null,
};

// ─── DOM REFS ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const dom = {
    personCount:  $('personCount'),
    countBar:     $('countBar'),
    threshVal:    $('threshVal'),
    peakVal:      $('peakVal'),
    riskRingFill: $('riskRingFill'),
    riskPct:      $('riskPct'),
    riskLabel:    $('riskLabel'),
    alertStrip:   $('alertStrip'),
    alertIcon:    $('alertIcon'),
    alertMsg:     $('alertMsg'),
    statAvg:      $('statAvg'),
    statMax:      $('statMax'),
    statAlerts:   $('statAlerts'),
    statUptime:   $('statUptime'),
    systemTime:   $('systemTime'),
    feedDot:      $('feedDot'),
    feedStatusText: $('feedStatusText'),
    criticalOverlay: $('criticalOverlay'),
    liveBadge:    $('liveBadge'),
    videoFeed:    $('videoFeed'),
    btnStream:    $('btnStreamToggle'),
    densityCols:  Array.from({ length: 8 }, (_, i) => $(`dc${i}`)),
};

// ─── CORE SYSTEM FUNCTIONS ───────────────────────────────────────────────────

async function toggleSystem() {
    if (state.isRunning) {
        stopSystem();
    } else {
        startSystem();
    }
}

async function startSystem() {
    dom.feedStatusText.textContent = 'INITIALIZING...';
    
    try {
        const res = await fetch(CONFIG.ENDPOINTS.START, { method: 'POST' });
        const data = await res.json();
        
        if (data.status === 'started' || data.status === 'already_running') {
            state.isRunning = true;
            state.startTime = Date.now();
            
            // Update UI
            dom.btnStream.textContent = "TERMINATE FEED";
            dom.btnStream.classList.add('danger');
            dom.liveBadge.style.opacity = '1';
            dom.liveBadge.querySelector('.live-text').textContent = 'LIVE';
            
            // Start Video
            // Add timestamp to bust cache
            dom.videoFeed.src = `${CONFIG.ENDPOINTS.VIDEO}?t=${Date.now()}`;
            dom.videoFeed.classList.add('active');
            
            // Connect SSE
            connectSSE();
            
            dom.feedStatusText.textContent = 'SYSTEM ONLINE';
            dom.feedDot.classList.add('active');
        }
    } catch (e) {
        console.error("Failed to start:", e);
        dom.feedStatusText.textContent = 'CONNECTION ERROR';
    }
}

async function stopSystem() {
    try {
        await fetch(CONFIG.ENDPOINTS.STOP, { method: 'POST' });
        
        state.isRunning = false;
        state.startTime = null;

        // Update UI
        dom.btnStream.textContent = "INITIALIZE SYSTEM";
        dom.btnStream.classList.remove('danger');
        dom.liveBadge.style.opacity = '0.3';
        dom.liveBadge.querySelector('.live-text').textContent = 'OFFLINE';

        // Stop Video
        dom.videoFeed.src = ""; // Stop buffering
        dom.videoFeed.classList.remove('active');

        // Kill SSE
        if (state.sse) {
            state.sse.close();
            state.sse = null;
        }

        dom.feedStatusText.textContent = 'SYSTEM STANDBY';
        dom.feedDot.classList.remove('active');
        dom.personCount.textContent = "0";
        dom.alertMsg.textContent = "SYSTEM STANDBY";
        dom.alertStrip.className = 'alert-strip'; // remove colors
        
    } catch (e) {
        console.error("Error stopping:", e);
    }
}

// ─── FILE UPLOAD HANDLER ─────────────────────────────────────────────────────
$('imgUpload').addEventListener('change', async (e) => {
    if (e.target.files.length === 0) return;

    // Pause live stream if active to show static result
    if (state.isRunning) {
        await stopSystem(); 
        dom.feedStatusText.textContent = 'ANALYZING IMAGE...';
    }

    const formData = new FormData();
    // Python expects 'file'
    formData.append('file', e.target.files[0]);

    try {
        const res = await fetch(CONFIG.ENDPOINTS.UPLOAD, { method: 'POST', body: formData });
        const data = await res.json();
        
        if (data.detected_count !== undefined) {
            // Update UI manually with static data
            handleNewCount(data.detected_count);
            dom.feedStatusText.textContent = `STATIC ANALYSIS: ${data.filename}`;
            dom.alertMsg.textContent = `STATIC IMAGE ANALYZED`;
        }
    } catch (err) {
        dom.feedStatusText.textContent = 'UPLOAD FAILED';
        console.error(err);
    }
});

// ─── RISK ENGINE ─────────────────────────────────────────────────────────────
function calcRisk(count) {
    const t = CONFIG.CROWD_THRESHOLD;
    const ratio = Math.min(count / t, 2.0); 
    const risk = Math.min(100, Math.round(
        ratio < 1
        ? (ratio ** 1.8) * 60          
        : 60 + (ratio - 1) * 80        
    ));
    return Math.min(risk, 100);
}

function riskCategory(pct) {
    if (pct < 30)  return 'low';
    if (pct < 60)  return 'moderate';
    if (pct < 80)  return 'high';
    return 'critical';
}

const RISK_COLORS = {
    low:      '#39d98a',
    moderate: '#f5a623',
    high:     '#ff8c00',
    critical: '#e63946',
};

const RISK_MESSAGES = {
    low:      'SYSTEM NOMINAL — MONITORING ACTIVE',
    moderate: 'ELEVATED DENSITY — MONITOR CLOSELY',
    high:     'HIGH RISK — CONSIDER CROWD MANAGEMENT',
    critical: 'CRITICAL RISK — IMMEDIATE ACTION REQUIRED',
};

// ─── UI UPDATERS ─────────────────────────────────────────────────────────────
function updateCount(count) {
    dom.personCount.textContent = count;
    dom.personCount.classList.toggle('danger', count > CONFIG.CROWD_THRESHOLD);

    const barPct = Math.min((count / CONFIG.CROWD_THRESHOLD) * 100, 100);
    dom.countBar.style.width = barPct + '%';

    if (count > state.peak) {
        state.peak = count;
        dom.peakVal.textContent = count;
        dom.statMax.textContent = count;
    }
    dom.threshVal.textContent = CONFIG.CROWD_THRESHOLD;
}

function updateRisk(pct) {
    const circumference = 314; 
    const offset = circumference - (pct / 100) * circumference;
    dom.riskRingFill.style.strokeDashoffset = offset;

    const cat = riskCategory(pct);
    const color = RISK_COLORS[cat];

    dom.riskRingFill.style.stroke = color;
    dom.riskPct.textContent = pct + '%';
    dom.riskLabel.textContent = cat.toUpperCase();
    dom.riskLabel.style.color = color;

    document.querySelectorAll('.zone').forEach(el => {
        el.className = 'zone'; // reset
        if (el.classList.contains(`zone--${cat === 'moderate' ? 'mod' : cat}`)) {
             el.classList.add(`active-${cat}`);
        }
    });
    // Manual fix for mod class mapping
    if(cat === 'moderate') document.querySelector('.zone--mod').classList.add('active-moderate');
}

function updateAlertStrip(cat, pct) {
    dom.alertStrip.className = 'alert-strip';
    if (cat === 'moderate' || cat === 'high') dom.alertStrip.classList.add('warning');
    if (cat === 'critical') dom.alertStrip.classList.add('danger');

    dom.alertIcon.textContent = cat === 'low' ? '●' : '⚠';
    dom.alertMsg.textContent = RISK_MESSAGES[cat] + ` [RISK: ${pct}%]`;

    if ((cat === 'high' || cat === 'critical') && pct > 0) {
        if (!dom.alertStrip._lastCat || dom.alertStrip._lastCat !== cat) {
            state.alerts++;
            dom.statAlerts.textContent = state.alerts;
        }
    }
    dom.alertStrip._lastCat = cat;
}

function updateDensityGauge(count) {
    const max = CONFIG.CROWD_THRESHOLD * 1.5;
    const filled = Math.min(Math.round((count / max) * 8), 8);

    dom.densityCols.forEach((col, i) => {
        const active = i < filled;
        let h;
        if (!active) { h = 4; }
        else {
            const frac = (i + 1) / 8;
            h = Math.round(10 + frac * 50);
        }
        col.style.height = h + 'px';

        const pct = (i + 1) / 8;
        if (pct < 0.4)       col.style.background = 'rgba(57,217,138,0.5)';
        else if (pct < 0.7)  col.style.background = 'rgba(245,166,35,0.6)';
        else                 col.style.background = 'rgba(230,57,70,0.7)';

        if (!active) col.style.background = 'rgba(255,255,255,0.04)';
    });
}

function updateCriticalOverlay(cat) {
    if (cat === 'critical' && !state.criticalAck) {
        dom.criticalOverlay.classList.add('show');
    }
    if (cat !== 'critical') {
        state.criticalAck = false;
        dom.criticalOverlay.classList.remove('show');
    }
}

window.dismissCritical = () => {
    state.criticalAck = true;
    dom.criticalOverlay.classList.remove('show');
};

// ─── CHART ───────────────────────────────────────────────────────────────────
const chartCtx = document.getElementById('timelineChart').getContext('2d');
const chartData = {
    labels:   [],
    datasets: [{
        label: 'Person Count',
        data: [],
        borderColor: '#f5a623',
        backgroundColor: 'rgba(245,166,35,0.08)',
        borderWidth: 1.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
    }, {
        label: 'Threshold',
        data: [],
        borderColor: 'rgba(230,57,70,0.4)',
        borderWidth: 1,
        borderDash: [6, 4],
        fill: false,
        pointRadius: 0,
    }],
};

const chart = new Chart(chartCtx, {
    type: 'line',
    data: chartData,
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 }, // Disable animation for performance
        plugins: { legend: { display: false } },
        scales: {
            x: { display: false }, // Hide X axis for cleaner look
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255,255,255,0.03)' },
                ticks: { color: '#3a4a5a', font: { family: "'Share Tech Mono'" } }
            },
        },
    },
});

function pushToChart(count) {
    if(!state.isRunning && count === 0) return; // Don't fill chart with 0s if stopped

    const now = new Date();
    const label = now.toLocaleTimeString();

    chartData.labels.push(label);
    chartData.datasets[0].data.push(count);
    chartData.datasets[1].data.push(CONFIG.CROWD_THRESHOLD);

    if (chartData.labels.length > CONFIG.HISTORY_LIMIT) {
        chartData.labels.shift();
        chartData.datasets[0].data.shift();
        chartData.datasets[1].data.shift();
    }

    chart.update();
}

// ─── STATS ────────────────────────────────────────────────────────────────────
function updateStats(count) {
    state.history.push({ time: Date.now(), count });
    if (state.history.length > CONFIG.HISTORY_LIMIT) state.history.shift();
    const avg = Math.round(state.history.reduce((s, d) => s + d.count, 0) / state.history.length);
    dom.statAvg.textContent = avg;
}

// ─── CLOCK & UPTIME ──────────────────────────────────────────────────────────
function tickClock() {
    const now = new Date();
    dom.systemTime.textContent = now.toLocaleTimeString('en-IN', {
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    });

    if(state.isRunning && state.startTime) {
        const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
        const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const ss = String(elapsed % 60).padStart(2, '0');
        dom.statUptime.textContent = `${mm}:${ss}`;
    }
}
setInterval(tickClock, 1000);
tickClock();

// ─── MASTER UPDATE ───────────────────────────────────────────────────────────
function handleNewCount(count) {
    state.count = count;
    const risk = calcRisk(count);
    const cat  = riskCategory(risk);

    updateCount(count);
    updateRisk(risk);
    updateAlertStrip(cat, risk);
    updateDensityGauge(count);
    updateCriticalOverlay(cat);
    pushToChart(count);
    updateStats(count);
}

// ─── SSE CONNECTION ──────────────────────────────────────────────────────────
function connectSSE() {
    if (state.sse) state.sse.close();

    state.sse = new EventSource(CONFIG.ENDPOINTS.SSE);

    state.sse.onopen = () => {
        // Feed is definitely active
    };

    state.sse.onmessage = (event) => {
        const count = parseInt(event.data, 10);
        if (!isNaN(count)) handleNewCount(count);
    };

    state.sse.onerror = (e) => {
        // If stream was stopped manually, don't try to reconnect
        if(state.isRunning) {
            state.sse.close();
            console.log("SSE dropped, retrying...");
            setTimeout(connectSSE, 2000);
        }
    };
}