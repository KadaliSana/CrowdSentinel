import pytest
from sfcpi.risk.levels import RiskLevel, Thresholds
from sfcpi.risk.machine import RiskStateMachine

T = Thresholds()

def _m(**kw):
    kw.setdefault("min_dwell_s", 2.0)
    kw.setdefault("min_realert_s", 60.0)
    kw.setdefault("blind_alert_s", 10.0)
    return RiskStateMachine(thresholds=T, **kw)

def test_nan_becomes_unknown_and_alerts_sensor_blind():
    """A blind sensor at a mass gathering is itself an incident."""
    m = _m()
    m.update(0.0, 0.0)
    assert m.update(1.0, float("nan")) is None       # dwell pending
    # 4s elapsed: past min_dwell_s (2.0) but short of blind_alert_s (10.0).
    # Still None here pins that entry to UNKNOWN is gated by blind_alert_s,
    # not by the shorter min_dwell_s -- swapping them would fire here.
    assert m.update(5.0, float("nan")) is None
    ev = m.update(11.5, float("nan"))                # 10.5s elapsed: blind_alert_s met
    assert ev is not None
    assert ev.level is RiskLevel.UNKNOWN
    assert ev.reason == "sensor-blind"
    assert ev.pressure is None

def test_unknown_never_reports_as_normal():
    m = _m()
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    ev = m.update(11.5, float("nan"))
    assert ev.level is not RiskLevel.NORMAL

def test_recovery_from_blind_is_reported():
    """Recovery FROM UNKNOWN uses the normal dwell (min_dwell_s), not
    blind_alert_s -- the extra dwell only applies to going blind, not to
    coming back. min_realert_s is short here so the recovery-to-NORMAL
    gate (see test_recovery_to_normal_is_gated_below) does not swallow it --
    that gate is pinned separately."""
    m = _m(min_realert_s=1.0)
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    m.update(11.5, float("nan"))                     # -> UNKNOWN (blind_alert_s met)
    m.update(12.0, 0.0)
    # 2.5s elapsed: past min_dwell_s (2.0) but well short of blind_alert_s
    # (10.0). Firing here pins that recovery uses min_dwell_s, not
    # blind_alert_s -- swapping them would still be None at this point.
    ev = m.update(14.5, 0.0)
    assert ev is not None and ev.level is RiskLevel.NORMAL
    assert ev.reason == "recovery"

def test_recovery_to_normal_is_gated_by_the_realert_interval():
    """Recovery to NORMAL is the resolution of the prior blind alert -- it
    goes through the same re-alert gate as any other all-clear."""
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    m.update(11.5, float("nan"))                     # -> UNKNOWN, sensor-blind fires
    m.update(12.0, 0.0)
    assert m.update(14.5, 0.0) is None                # recovered, but too soon to re-alert

def test_recovery_to_danger_bypasses_the_realert_interval():
    """A sensor coming back and immediately showing danger must be heard --
    there is no pre-blind severity to compare against, so this is not gated
    the way recovery-to-NORMAL is."""
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    m.update(11.5, float("nan"))                     # -> UNKNOWN, sensor-blind fires
    m.update(12.0, 0.025)                             # candidate HIGH
    ev = m.update(14.0, 0.025)                        # min_dwell_s met
    assert ev is not None
    assert ev.level is RiskLevel.HIGH
    assert ev.reason == "recovery"

def test_escalation_bypasses_the_realert_interval():
    """Getting worse must always be heard immediately."""
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, 0.025); first = m.update(3.0, 0.025)      # -> HIGH
    assert first is not None
    m.update(4.0, 0.05); second = m.update(6.5, 0.05)       # -> CRITICAL, still inside 600s
    assert second is not None and second.level is RiskLevel.CRITICAL

def test_de_escalation_respects_the_realert_interval():
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, 0.025); m.update(3.0, 0.025)              # -> HIGH, alert fired
    m.update(4.0, 0.0)
    assert m.update(6.5, 0.0) is None                        # calmer, but too soon to re-alert


# --- C2: a flapping camera must not page once per flap -----------------------

def _flap(machine, duration, period=3.0, dt=0.5):
    """Alternate `period` seconds blind / `period` seconds fine."""
    events, t = [], 0.0
    while t < duration:
        blind = int(t // period) % 2 == 0
        ev = machine.update(t, float("nan") if blind else 0.0)
        if ev is not None:
            events.append(ev)
        t += dt
    return events


def test_flapping_blind_sensor_is_bounded_by_the_realert_interval():
    """A camera flapping blind/fine paged once per flap, forever.

    Leaving UNKNOWN de-escalates to NORMAL, that all-clear is gated, so
    _last_alerted_level STAYS UNKNOWN and the next blind period bypassed the
    re-alert gate again. Before the fix this produced 10 pages in 60s (and 20
    in 120s) -- bounded by the flap period, not by min_realert_s.
    """
    duration, realert = 60.0, 60.0
    m = _m(min_dwell_s=2.0, min_realert_s=realert, blind_alert_s=2.0)
    events = _flap(m, duration)
    blinds = [e for e in events if e.reason == "sensor-blind"]
    flap_periods = int(duration // 6.0)          # 10 blind periods in 60s
    bound = 1 + int(duration // realert)         # what min_realert_s allows
    assert len(blinds) <= bound, (
        f"expected at most {bound} blind pages in {duration}s at "
        f"min_realert_s={realert}, got {len(blinds)}"
    )
    assert len(blinds) < flap_periods, "page count still tracks the flap period"


def test_flap_page_count_does_not_grow_with_flap_duration():
    """Doubling the flap duration must not double the pages."""
    short = _flap(_m(min_dwell_s=2.0, min_realert_s=600.0, blind_alert_s=2.0), 60.0)
    long = _flap(_m(min_dwell_s=2.0, min_realert_s=600.0, blind_alert_s=2.0), 120.0)
    assert len([e for e in short if e.reason == "sensor-blind"]) == 1
    assert len([e for e in long if e.reason == "sensor-blind"]) == 1


def test_flapping_after_a_danger_alert_does_not_page_via_recovery():
    """C2, hole #4: the repeat-blind gate RELOCATED the flap spam onto
    `recovery`.

    Gating the second and later sensor-blind alerts leaves
    `_last_alerted_level` at whatever was last actually paged -- a DANGER
    level, not UNKNOWN -- while `self.level` is UNKNOWN. The old
    recovery-to-NORMAL gate keyed on `_last_alerted_level is previous`
    (i.e. UNKNOWN), so that identity check failed and every all-clear
    bypassed `min_realert_s` outright. Worse, firing set
    `_last_alerted_level = NORMAL`, which is *also* not UNKNOWN -- so the
    bypass re-armed itself and the flap paged forever.

    Priming matters: the two older C2 tests start from a pure flap (where
    `_last_alerted_level` stays UNKNOWN, so the identity check happened to
    hold) and then filter to `reason == "sensor-blind"`, which discards
    exactly the pages this leak produces. So: prime with one blind period
    and a recovery straight to HIGH, then flap, then count EVERY reason.

    Measured on the pre-fix machine: 18 pages in 113s at min_realert_s=600.
    """
    realert = 600.0
    m = _m(min_dwell_s=2.0, min_realert_s=realert, blind_alert_s=2.0)

    # Prime: _last_alerted_level must end up a DANGER level, not UNKNOWN.
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    assert m.update(3.5, float("nan")) is not None          # -> UNKNOWN, blind fires
    m.update(4.0, 0.025)
    primed = m.update(6.5, 0.025)                            # -> HIGH, recovery fires
    assert primed is not None and primed.level is RiskLevel.HIGH
    assert m._last_alerted_level is RiskLevel.HIGH

    # 3s blind / 3s fine, for ~113s -- well inside a single 600s window.
    duration, t0 = 113.0, 6.5
    events, t = [], t0
    while t < t0 + duration:
        blind = int((t - t0) // 3.0) % 2 == 0
        ev = m.update(t, float("nan") if blind else 0.0)
        if ev is not None:
            events.append(ev)
        t += 0.5

    # Count ALL reasons. Filtering to "sensor-blind" is what hid this.
    bound = 1 + int(duration // realert)
    assert len(events) <= bound, (
        f"expected at most {bound} page(s) of ANY reason in {duration}s at "
        f"min_realert_s={realert}, got {len(events)}: "
        f"{[e.reason for e in events]}"
    )


def test_the_first_blind_alert_is_never_swallowed():
    """Gating subsequent blinds must not mute the first one: a sensor dying
    right after a danger alert is exactly when its silence matters."""
    m = _m(min_dwell_s=2.0, min_realert_s=600.0, blind_alert_s=2.0)
    m.update(0.0, 0.0)
    m.update(1.0, 0.025)
    assert m.update(3.0, 0.025) is not None            # -> HIGH, alert fired
    m.update(4.0, float("nan"))
    ev = m.update(6.5, float("nan"))                   # sensor dies 3.5s later
    assert ev is not None, "first sensor-blind must fire even inside min_realert_s"
    assert ev.reason == "sensor-blind"


def test_danger_after_a_gated_blind_recovery_is_still_heard():
    """After a blind alert whose recovery-to-NORMAL was gated,
    _last_alerted_level is a stale UNKNOWN. severity(UNKNOWN) is None, which
    must not be read as 'nothing has alerted yet' -- but a genuine danger
    level is new information to the operator and must still be heard."""
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    assert m.update(11.5, float("nan")) is not None    # -> UNKNOWN, blind fires
    m.update(12.0, 0.0)
    assert m.update(14.5, 0.0) is None                 # recovery gated: stale UNKNOWN
    m.update(15.0, 0.025)
    ev = m.update(17.0, 0.025)
    assert ev is not None and ev.level is RiskLevel.HIGH


# --- C3: dwell must be satisfied by evidence FOR the level adopted -----------

def test_unscorable_frames_are_not_evidence_for_a_danger_level():
    """2.1s of NaN followed by ONE frame at 0.05 used to be adopted as an
    immediate CRITICAL 'escalation': dwell counted time since the last frame
    that agreed with the adopted level, then adopted the LAST candidate. The
    dwell exists to kill single-frame spikes -- unscorable frames must not
    act as evidence for a danger level."""
    m = _m(min_dwell_s=2.0, blind_alert_s=10.0)
    m.update(0.0, 0.0)
    for i in range(1, 22):                             # 2.1s of NaN
        assert m.update(i * 0.1, float("nan")) is None
    assert m.update(2.2, 0.05) is None, "one frame cannot complete the dwell"
    assert m.level is RiskLevel.NORMAL
    # ...and the spike still escalates once it is genuinely sustained.
    ev = m.update(4.3, 0.05)
    assert ev is not None and ev.level is RiskLevel.CRITICAL


def test_dwell_adopts_the_level_every_frame_supports():
    """Candidates straddling a boundary settle at the level ALL the evidence
    supports, not at whichever side the frame completing the dwell landed on."""
    m = _m(min_dwell_s=2.0)
    m.update(0.0, 0.0)
    m.update(1.0, 0.019)                               # -> ELEVATED candidate
    m.update(2.0, 0.025)                               # -> HIGH candidate
    ev = m.update(3.0, 0.025)
    assert ev is not None
    assert ev.level is RiskLevel.ELEVATED


# --- C5: low coverage is unscorable, not calm --------------------------------

def test_low_coverage_is_unknown_never_normal():
    """A face-detector dropout is a legitimate pressure~=0 reading, so scoring
    it as NORMAL turns the dense-crowd signature into silence."""
    m = _m(min_dwell_s=2.0, blind_alert_s=2.0, min_coverage=0.5)
    m.update(0.0, 0.0, coverage=0.9)
    assert m.update(1.0, 0.0, coverage=0.1) is None    # blind dwell pending
    ev = m.update(3.5, 0.0, coverage=0.1)
    assert ev is not None
    assert ev.level is RiskLevel.UNKNOWN
    assert ev.reason == "sensor-blind"
    assert ev.pressure is None                          # never reported as 0.0
    assert ev.coverage == pytest.approx(0.1)


def test_coverage_at_or_above_min_coverage_is_scored_normally():
    m = _m(min_dwell_s=2.0, min_coverage=0.5)
    m.update(0.0, 0.0, coverage=0.9)
    m.update(1.0, 0.025, coverage=0.5)                  # exactly at the floor
    ev = m.update(3.0, 0.025, coverage=0.9)
    assert ev is not None
    assert ev.level is RiskLevel.HIGH
    assert ev.reason == "escalation"
    assert ev.pressure == pytest.approx(0.025)


def test_min_coverage_zero_is_disabled_and_changes_nothing():
    """0.0 must preserve current behaviour exactly, even at coverage 0.0."""
    m = _m(min_dwell_s=2.0, blind_alert_s=2.0)          # default min_coverage=0.0
    assert m.min_coverage == 0.0
    m.update(0.0, 0.0, coverage=0.0)
    for i in range(1, 21):
        assert m.update(float(i), 0.0, coverage=0.0) is None
    assert m.level is RiskLevel.NORMAL


def test_coverage_is_ignored_when_absent():
    m = _m(min_dwell_s=2.0, blind_alert_s=2.0, min_coverage=0.9)
    m.update(0.0, 0.0)
    m.update(1.0, 0.025)                                # no coverage supplied
    ev = m.update(3.0, 0.025)
    assert ev is not None and ev.level is RiskLevel.HIGH


def test_interleaved_dropout_frames_surface_instead_of_going_silent():
    """The C5 target: alternating danger frames and detector-dropout frames.
    Scored naively the dropouts read as NORMAL, reset the dwell every frame
    and the pattern is silent forever. With coverage honoured it must produce
    SOMETHING -- an escalation or a sensor-blind, but not silence."""
    m = _m(min_dwell_s=2.0, blind_alert_s=10.0, min_realert_s=600.0, min_coverage=0.5)
    events, t = [], 0.0
    while t < 30.0:
        if int(t * 2) % 2 == 0:
            ev = m.update(t, 0.030, coverage=0.9)       # dense crowd
        else:
            ev = m.update(t, 0.0, coverage=0.1)         # detector dropped out
        if ev is not None:
            events.append(ev)
        t += 0.5
    assert events, "alternating danger/dropout must not be silent"
    assert events[0].level is not RiskLevel.NORMAL
    assert events[0].reason in ("escalation", "sensor-blind")


# --- C3 (cycle 3): the re-alert invariant, ENUMERATED over the reason space --
#
# Five unbounded-page defects have been found in this gate. Three consecutive
# fixes each RELOCATED the spam rather than removing it, because each shipped
# with a test written to that fix instead of to the property. The most recent
# gated recovery->NORMAL and counted all reasons -- but drove a "fine" pressure
# of 0.0, so it could only ever reach recovery->NORMAL and was structurally
# blind to recovery->danger, which is where the spam had gone.
#
# These tests therefore enumerate rather than sample: every reason the machine
# can emit is driven, by name, and the "fine" value is a parameter covering
# both 0.0 (recovery -> NORMAL) and 0.025 (recovery -> HIGH).

BLIND = float("nan")

# Every reason RiskEvent documents. `sustained` is unreachable by construction
# (equal severities cannot be different levels) and is marked pragma: no cover
# in the machine; it is listed here so that adding a reason without extending
# these tests fails loudly.
DOCUMENTED_REASONS = {
    "escalation", "de-escalation", "sustained", "sensor-blind", "recovery",
}
REACHABLE_REASONS = DOCUMENTED_REASONS - {"sustained"}

# An emission may legitimately bypass min_realert_s by outranking the last
# level the operator was told. That ladder is finite and monotone within a
# window: NORMAL -> ELEVATED -> HIGH -> CRITICAL, plus the one blindness alert.
# It is what separates "bounded by the clock" from "silent".
LADDER_BYPASSES = 4


def _drive(machine, duration, a, b, period=3.0, dt=0.5, t0=0.0):
    """Alternate `period` seconds of input `a` / `period` seconds of `b`.

    Returns (events, adoptions). `adoptions` counts every time the machine
    ADOPTED a new level, whether or not it paged. It is the anti-vacuity
    guard: it proves the drive genuinely traversed the transitions under test,
    so a parameterisation that can never reach the interesting case -- the
    exact defect the last three regression tests shipped with -- fails loudly
    instead of passing empty.
    """
    events, adoptions, t = [], 0, t0
    while t < t0 + duration:
        before = machine.level
        ev = machine.update(t, a if int((t - t0) // period) % 2 == 0 else b)
        if machine.level is not before:
            adoptions += 1
        if ev is not None:
            events.append(ev)
        t += dt
    return events, adoptions


def _by_reason(events):
    counts = {}
    for e in events:
        counts[e.reason] = counts.get(e.reason, 0) + 1
    return counts


def _max_pages_in_any_window(events, window):
    """Most pages falling in any half-open interval of length `window`.

    This is the LOCAL form of the invariant and the one worth asserting: a
    global count over a long run legitimately grows with the number of
    re-alert windows, which hides a per-flap leak inside a loose bound. The
    per-window maximum does not grow with either the run length or the flap
    rate, so it separates the two directly.
    """
    ts = [e.timestamp for e in events]
    return max(
        (sum(1 for x in ts if start <= x < start + window) for start in ts),
        default=0,
    )


# (a, b, expected-per-reason counts) for a single 600s re-alert window.
FLAP_CASES = [
    pytest.param(
        BLIND, 0.0, {"sensor-blind": 1},
        id="blind-flap-recovering-to-normal",
    ),
    pytest.param(
        # THE CASE THREE PREVIOUS TEST-WRITERS MISSED: a genuinely dense crowd
        # with a flapping camera -- the likeliest real pairing. Every recovery
        # lands on HIGH, not NORMAL, so `previous is UNKNOWN` -> reason
        # "recovery" -> adopted is not NORMAL -> the old branch chain fired
        # unconditionally, every single flap. Pre-fix: 19 pages in 113s at
        # min_realert_s=600 (18 recovery + 1 sensor-blind), self-sustaining at
        # period blind_alert_s + min_dwell_s.
        BLIND, 0.025, {"sensor-blind": 1, "recovery": 1},
        id="blind-flap-recovering-to-danger",
    ),
    pytest.param(
        0.025, 0.0, {"escalation": 1},
        id="measured-full-swing-high-to-normal",
    ),
    pytest.param(
        0.05, 0.025, {"escalation": 1},
        id="measured-swing-critical-to-high",
    ),
]


@pytest.mark.parametrize("a,b,expected", FLAP_CASES)
def test_no_reason_pages_more_than_the_realert_interval_allows(a, b, expected):
    """Within ONE min_realert_s window, a flapping input pages a bounded
    number of times -- bounded by the clock and the finite severity ladder,
    never by the flap period.

    Asserted per-reason as well as in total, so a future relocation of the
    spam onto a different reason is caught BY NAME rather than hidden inside
    an aggregate that happens to stay under a loose bound.
    """
    duration, realert = 113.0, 600.0
    m = _m(min_dwell_s=2.0, min_realert_s=realert, blind_alert_s=2.0)
    events, adoptions = _drive(m, duration, a, b)
    counts = _by_reason(events)
    labels = [e.reason for e in events]

    # The drive must actually have flapped -- otherwise this test proves
    # nothing, which is precisely how the previous regression tests passed.
    assert adoptions >= 15, (
        f"drive adopted only {adoptions} levels in {duration}s; it is not "
        "exercising the flap and this assertion would be vacuous"
    )

    assert counts == expected, (
        f"per-reason page counts changed: expected {expected}, got {counts} "
        f"in {duration}s at min_realert_s={realert} (labels: {labels})"
    )
    assert len(events) == sum(expected.values()), (
        f"total pages {len(events)} != {sum(expected.values())}: {labels}"
    )
    assert len(events) <= (1 + int(duration // realert)) + LADDER_BYPASSES, labels
    # The whole point: pages track the clock, not the flap.
    assert len(events) < adoptions, (
        f"{len(events)} pages for {adoptions} adoptions -- still paging per flap"
    )


@pytest.mark.parametrize("a,b,expected", FLAP_CASES)
def test_page_count_is_independent_of_how_long_the_flap_lasts(a, b, expected):
    """Quadrupling the flap duration inside one re-alert window must not
    change the page count at all. This is the truest statement of "bounded by
    min_realert_s rather than by the flap period"."""
    realert = 600.0
    short, short_adoptions = _drive(
        _m(min_dwell_s=2.0, min_realert_s=realert, blind_alert_s=2.0), 113.0, a, b
    )
    long, long_adoptions = _drive(
        _m(min_dwell_s=2.0, min_realert_s=realert, blind_alert_s=2.0), 452.0, a, b
    )
    assert long_adoptions > short_adoptions, "the longer drive must flap more"
    assert _by_reason(short) == _by_reason(long) == expected, (
        f"page count grew with duration: {_by_reason(short)} -> {_by_reason(long)}"
    )


def test_every_reachable_reason_is_bounded_by_the_realert_interval():
    """Enumerate the reason space, do not sample it.

    A short min_realert_s over many windows so that EVERY reachable reason --
    escalation, de-escalation, sensor-blind and recovery -- is actually
    emitted at least once and can be bounded by name. The 600s tests above
    pin the single-window behaviour; this one pins that no reason grows with
    the flap once the clock does open.
    """
    duration, realert = 300.0, 20.0
    observed = {}
    for a, b in ((BLIND, 0.0), (BLIND, 0.025), (0.025, 0.0), (0.05, 0.025)):
        m = _m(min_dwell_s=2.0, min_realert_s=realert, blind_alert_s=2.0)
        events, adoptions = _drive(m, duration, a, b)
        counts = _by_reason(events)
        windows = 1 + int(duration // realert)
        # Per window the clock may open once, and the severity ladder may be
        # re-climbed from whatever the operator was last told. Across the run
        # that budget repeats per window -- which is the definition of "gated
        # by min_realert_s" -- so the tight assertion is the per-window one
        # below, not this global cap.
        per_window = 1 + LADDER_BYPASSES
        assert set(counts) <= DOCUMENTED_REASONS, (
            f"undocumented reason emitted: {set(counts) - DOCUMENTED_REASONS}"
        )
        assert _max_pages_in_any_window(events, realert) <= per_window, (
            f"input ({a}, {b}) paged "
            f"{_max_pages_in_any_window(events, realert)} times inside a single "
            f"{realert}s window (cap {per_window}): {[e.reason for e in events]}"
        )
        for reason, n in counts.items():
            assert n <= windows * per_window, (
                f"reason {reason!r} paged {n} times in {duration}s at "
                f"min_realert_s={realert} for input ({a}, {b})"
            )
        assert len(events) <= windows * per_window, (
            f"{len(events)} total pages for input ({a}, {b}): "
            f"{[e.reason for e in events]}"
        )
        assert len(events) < adoptions, "pages still track the flap period"
        for reason, n in counts.items():
            observed[reason] = observed.get(reason, 0) + n

    missing = REACHABLE_REASONS - set(observed)
    assert not missing, (
        f"these reasons were never exercised, so nothing above bounds them: "
        f"{sorted(missing)}. Add a case that reaches them rather than "
        "narrowing the enumeration."
    )
