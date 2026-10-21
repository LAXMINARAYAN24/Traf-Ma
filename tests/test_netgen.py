import math

from conftest import needs_sumo
from trafma import APPROACHES
from trafma.sensors import approach_signal_char, time_to_state_change

pytestmark = needs_sumo


def test_twelve_links_three_per_approach(topo):
    assert topo.n_links == 12
    all_links = sorted(i for a in APPROACHES for i in topo.approaches[a].link_indices)
    assert all_links == list(range(12))
    assert all(len(topo.approaches[a].link_indices) == 3 for a in APPROACHES)


def test_signal_never_gives_two_approaches_green(topo):
    for ph in topo.signal_phases:
        live = [a for a in APPROACHES if approach_signal_char(ph["state"], topo.approaches[a].link_indices) != "r"]
        assert len(live) <= 1, ph


def test_every_approach_gets_green_and_cycle(topo):
    greens = {a: sum(p["duration"] for p in topo.signal_phases
                     if approach_signal_char(p["state"], topo.approaches[a].link_indices) == "G") for a in APPROACHES}
    assert set(greens.values()) == {25.0}
    assert topo.cycle_s == 120.0


def test_one_rsu_per_road_beside_its_own_stop_line(topo):
    pos = {a: topo.approaches[a].rsu_xy for a in APPROACHES}
    assert len(set(pos.values())) == 4
    for a, g in topo.approaches.items():
        d = math.dist(g.rsu_xy, g.stop_xy)
        assert 7.9 < d < 8.1
        others = [math.dist(g.rsu_xy, o.stop_xy) for b, o in topo.approaches.items() if b != a]
        assert min(others) > d            # nearest stop line is its own


def test_lane_fits_zone(topo):
    assert all(g.in_len >= 150 and g.out_len >= 100 for g in topo.approaches.values())


def test_time_to_state_change(topo):
    # phase 0 = N green with 24 s left: N changes in 24 s; E stays red through N's yellow+allred (24+3+2)
    assert time_to_state_change(topo, "N", 0, 24.0) == 24.0
    assert time_to_state_change(topo, "E", 0, 24.0) == 29.0
