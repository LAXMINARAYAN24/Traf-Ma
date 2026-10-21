"""Scope fixtures: an RSU cannot see another approach; a vehicle cannot read the whole world."""
import pytest
from conftest import needs_sumo, veh, world
from trafma.sensors import queue_chain, rsu_view, vehicle_view, VEHICLE_MASKED

pytestmark = needs_sumo


def L(topo, a):
    return topo.approaches[a].in_len


def test_rsu_sees_only_its_own_approach(topo, cfg):
    w = world([veh("a", "N_in", L(topo, "N") - 20), veh("b", "E_in", L(topo, "E") - 20),
               veh("c", "S_in", L(topo, "S") - 20)], topo)
    n = rsu_view(w, topo, "N", cfg)
    assert n["vehicle_count"] == 1 and n["_detected"] == ["a"]
    assert rsu_view(w, topo, "W", cfg)["vehicle_count"] == 0


def test_rsu_zone_boundary(topo, cfg):
    inside = veh("in", "N_in", L(topo, "N") - 149.0)
    outside = veh("out", "N_in", L(topo, "N") - 151.0)
    v = rsu_view(world([inside, outside], topo), topo, "N", cfg)
    assert v["_detected"] == ["in"]


def test_empty_region_is_unavailable_not_zero(topo, cfg):
    v = rsu_view(world([], topo), topo, "N", cfg)
    assert v["vehicle_count"] == 0 and v["queue_length_m"] == 0.0
    assert v["mean_speed_mps"] is None and v["max_waiting_time_s"] is None
    assert set(v["unavailable"]) == {"mean_speed_mps", "mean_waiting_time_s", "max_waiting_time_s"}


def test_rsu_counts_outbound_road_of_its_own_road_only(topo, cfg):
    w = world([veh("x", "N_out", 30.0, 8.0), veh("y", "E_out", 30.0, 8.0), veh("z", "N_out", 250.0, 8.0)], topo)
    assert rsu_view(w, topo, "N", cfg)["outbound_count"] == 1


def test_rsu_signal_matches_its_road(topo, cfg):
    w = world([], topo, phase_index=0, remaining=10)
    assert rsu_view(w, topo, "N", cfg)["signal_state"] == "G"
    assert rsu_view(w, topo, "E", cfg)["signal_state"] == "r"


def test_queue_chain_rules(cfg):
    # halted chain from the stop line, 7.5 m pitch
    items = [(2.0, 0, 5), (9.5, 0, 5), (17.0, 0, 5), (24.5, 0, 5)]
    assert queue_chain(items, cfg) == (29.5, 4)
    # gap > queue_gap breaks the chain
    assert queue_chain([(2, 0, 5), (40, 0, 5)], cfg) == (7.0, 1)
    # first halted vehicle too far from the stop line -> no queue
    assert queue_chain([(60, 0, 5)], cfg) == (0.0, 0)
    # moving vehicles never form a queue
    assert queue_chain([(2, 5.0, 5), (9.5, 5.0, 5)], cfg) == (0.0, 0)
    assert queue_chain([], cfg) == (0.0, 0)


def test_vehicle_on_own_approach_sees_only_own_edge_neighbours(topo, cfg):
    p = L(topo, "N") - 80
    w = world([veh("me", "N_in", p, 5.0), veh("ahead", "N_in", p + 30, 0.0), veh("behind", "N_in", p - 40, 6.0),
               veh("far", "N_in", p - 150, 6.0), veh("other", "E_in", p, 3.0)], topo)
    v = vehicle_view(w, topo, "me", "N", cfg)
    assert v["scope"] == "incoming_local"
    assert v["neighbor_count"] == 2 and v["neighbor_ahead_count"] == 1 and v["neighbor_behind_count"] == 1
    assert v["leader_gap_m"] == pytest.approx(25.0) and v["neighbor_halting_count"] == 1
    assert v["dist_to_stop_line_m"] == pytest.approx(80.0)


def test_vehicle_cannot_read_the_world(topo, cfg):
    p = L(topo, "N") - 80
    w = world([veh("me", "N_in", p, 5.0)] + [veh(f"o{i}", "E_in", 100 + i * 10, 3.0) for i in range(5)], topo)
    v = vehicle_view(w, topo, "me", "N", cfg)
    assert v["neighbor_count"] == 0 and v["neighbor_mean_speed_mps"] is None and "leader_gap_m" in v["unavailable"]
    assert not any(isinstance(x, list) and x and isinstance(x[0], str) and x[0].startswith("o") for x in v.values())


def test_signal_only_readable_within_radius(topo, cfg):
    near = vehicle_view(world([veh("me", "N_in", L(topo, "N") - 50)], topo), topo, "me", "N", cfg)
    far = vehicle_view(world([veh("me", "N_in", L(topo, "N") - 200)], topo), topo, "me", "N", cfg)
    assert near["signal_state"] == "G" and far["signal_state"] is None and "signal_state" in far["unavailable"]


@pytest.mark.parametrize("edge,scope", [("S_out", "out_of_coverage"), (":C_3", "in_junction"), ("E_in", "out_of_coverage")])
def test_out_of_coverage_is_masked(topo, cfg, edge, scope):
    w = world([veh("me", edge, 5.0, 4.0), veh("n", edge, 10.0, 0.0)], topo)
    v = vehicle_view(w, topo, "me", "N", cfg)
    assert v["scope"] == scope
    assert all(v[k] is None for k in VEHICLE_MASKED) and set(VEHICLE_MASKED) == set(v["unavailable"])
    assert v["speed_mps"] == 4.0          # own kinematics stay available


def test_unknown_vehicle_returns_none(topo, cfg):
    assert vehicle_view(world([], topo), topo, "ghost", "N", cfg) is None
