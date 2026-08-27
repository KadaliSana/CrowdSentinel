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
