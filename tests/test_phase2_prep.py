"""Phase 2 preparation: new RSU fields (mean accel / mean distance), schema version 1.1, signal broadcast flag."""
import pytest
from conftest import needs_sumo, world
from trafma import MSG_VERSION
from trafma.agents import RsuAgent
from trafma.config import ConfigError, config_from_dict, load_config
from trafma.messages import PAYLOAD_FIELDS, RSU_SUMMARY, VEHICLE_REPORT, validate
from trafma.sensors import VEHICLE_MASKED, rsu_view, time_to_state_change, vehicle_view
from trafma.world import VehicleState


def vs(vid, edge, pos, speed=0.0, accel=0.0, waiting=0.0):
    return VehicleState(vid, edge, pos, speed, accel, waiting, 0.0, 0.0, 5.0)


def L(topo, a):
    return topo.approaches[a].in_len


# ------------------------------------------------------------------ schema / version (no SUMO needed)
def test_schema_version_bumped_and_old_messages_still_valid(cfg):
    assert MSG_VERSION == "1.1"
    rsu = RsuAgent("N", cfg)
    d = rsu.summarise({"approach": "N"}, 1.0).to_dict()
    # the validator checks the MAJOR version only, so v1.0 logs/messages remain acceptable
    d["version"] = "1.0"
    assert not [p for p in validate(d) if "version" in p]


def test_new_fields_are_part_of_the_payload_contract():
    assert {"mean_accel_mps2", "mean_dist_to_stop_m"} <= set(PAYLOAD_FIELDS[RSU_SUMMARY])
    assert "signal_remaining_s" in PAYLOAD_FIELDS[VEHICLE_REPORT]
    assert "signal_remaining_s" in VEHICLE_MASKED


def test_signal_broadcast_config_default_and_validation():
    assert load_config().agents.signal_broadcast is False
    assert load_config(overrides={"agents": {"signal_broadcast": True}}).agents.signal_broadcast is True
    for bad in ("yes", 1, None):
        with pytest.raises(ConfigError):
            config_from_dict({"agents": {"signal_broadcast": bad}})


# ------------------------------------------------------------------ RSU mean accel / distance
def test_rsu_mean_accel_and_distance_over_zone_vehicles_only(topo, cfg):
    n = L(topo, "N")
    w = world([vs("a", "N_in", n - 20.0, accel=1.0),
               vs("b", "N_in", n - 40.0, accel=-2.0),
               vs("far", "N_in", n - 200.0, accel=5.0),     # outside the 150 m zone
               vs("other", "E_in", L(topo, "E") - 30.0, accel=9.0)], topo)   # another approach
    v = rsu_view(w, topo, "N", cfg)
    assert v["vehicle_count"] == 2
    assert v["mean_accel_mps2"] == pytest.approx(-0.5)
    assert v["mean_dist_to_stop_m"] == pytest.approx(30.0)
    assert "mean_accel_mps2" not in v["unavailable"] and "mean_dist_to_stop_m" not in v["unavailable"]


def test_rsu_new_fields_unavailable_not_zero_when_zone_empty(topo, cfg):
    v = rsu_view(world([], topo), topo, "N", cfg)
    assert v["mean_accel_mps2"] is None and v["mean_dist_to_stop_m"] is None
    assert {"mean_accel_mps2", "mean_dist_to_stop_m"} <= set(v["unavailable"])


def test_measured_zero_accel_is_a_value_not_unavailable(topo, cfg):
    v = rsu_view(world([vs("a", "N_in", L(topo, "N") - 10.0, accel=0.0)], topo), topo, "N", cfg)
    assert v["mean_accel_mps2"] == 0.0 and "mean_accel_mps2" not in v["unavailable"]


def test_rsu_summary_with_new_fields_validates(topo, cfg):
    for vehicles in ([], [vs("a", "N_in", L(topo, "N") - 25.0, 3.0, 0.5)]):
        view = rsu_view(world(vehicles, topo), topo, "N", cfg)
        view.pop("_detected")
        env = RsuAgent("N", cfg).summarise(view, 1.0)
        assert validate(env.to_dict()) == []
        assert "mean_accel_mps2" in env.payload and "mean_dist_to_stop_m" in env.payload


# ------------------------------------------------------------------ signal broadcast to vehicles
def test_default_vehicle_never_sees_the_timer(topo, cfg):
    near = vehicle_view(world([vs("me", "N_in", L(topo, "N") - 50)], topo), topo, "me", "N", cfg)
    far = vehicle_view(world([vs("me", "N_in", L(topo, "N") - 200)], topo), topo, "me", "N", cfg)
    assert near["signal_state"] == "G" and near["signal_remaining_s"] is None
    assert far["signal_state"] is None and far["signal_remaining_s"] is None
    assert {"signal_state", "signal_remaining_s"} <= set(far["unavailable"])
    assert "signal_remaining_s" in near["unavailable"] and "signal_state" not in near["unavailable"]


def test_broadcast_gives_state_and_timer_anywhere_on_the_approach(topo):
    cfg = load_config(overrides={"agents": {"signal_broadcast": True}})
    w = world([vs("me", "N_in", L(topo, "N") - 200)], topo, phase_index=0, remaining=10.0)
    v = vehicle_view(w, topo, "me", "N", cfg)
    assert v["signal_state"] == "G"
    assert v["signal_remaining_s"] == pytest.approx(time_to_state_change(topo, "N", 0, 10.0))
    assert "signal_state" not in v["unavailable"] and "signal_remaining_s" not in v["unavailable"]
    # a red approach gets the time until green
    e = vehicle_view(world([vs("me", "E_in", L(topo, "E") - 200)], topo, 0, 10.0), topo, "me", "E", cfg)
    assert e["signal_state"] == "r" and e["signal_remaining_s"] > 10.0


@pytest.mark.parametrize("edge,scope", [("S_out", "out_of_coverage"), (":C_3", "in_junction")])
def test_broadcast_does_not_extend_coverage_beyond_the_incoming_road(topo, edge, scope):
    cfg = load_config(overrides={"agents": {"signal_broadcast": True}})
    v = vehicle_view(world([vs("me", edge, 5.0, 4.0)], topo), topo, "me", "N", cfg)
    assert v["scope"] == scope and v["signal_state"] is None and v["signal_remaining_s"] is None


def test_broadcast_does_not_change_other_vehicle_fields(topo):
    p = L(topo, "N") - 80
    w = world([vs("me", "N_in", p, 5.0), vs("ahead", "N_in", p + 30), vs("behind", "N_in", p - 40, 6.0)], topo)
    off = vehicle_view(w, topo, "me", "N", load_config())
    on = vehicle_view(w, topo, "me", "N", load_config(overrides={"agents": {"signal_broadcast": True}}))
    keep = lambda d: {k: v for k, v in d.items() if k not in ("signal_remaining_s", "unavailable")}
    assert keep(off) == keep(on)


# ------------------------------------------------------------------ whole-simulation guarantees
@needs_sumo
def test_sensor_changes_never_alter_traffic_and_logs_carry_new_fields(tmp_path):
    import json
    from trafma.runner import run_simulation

    def go(rid, bcast):
        ov = {"run": {"output_dir": str(tmp_path), "run_id": rid, "duration_s": 250, "vehicles": 30},
              "agents": {"signal_broadcast": bcast}}
        return run_simulation(load_config(overrides=ov), quiet=True)

    a, b = go("off", False), go("on", True)
    assert a.manifest["trace_sha256"] == b.manifest["trace_sha256"]          # same traffic, only observations differ
    recs = [json.loads(x) for x in open(a.run_dir / "agents" / "rsu_N.jsonl")]
    assert all("mean_accel_mps2" in r["view"] and "mean_dist_to_stop_m" in r["view"] for r in recs)
    assert any(r["view"]["mean_accel_mps2"] is not None for r in recs)
    seen = {}
    for rd in (a.run_dir, b.run_dir):
        sig = [json.loads(x)["view"]["signal_remaining_s"] for p in (rd / "agents").glob("veh_*.jsonl")
               for x in open(p)]
        seen[rd.name] = any(s is not None for s in sig)
    assert seen == {"off": False, "on": True}
