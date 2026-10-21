import copy

import pytest
from trafma import APPROACHES
from trafma.agents import AgentRegistry, RsuAgent, VehicleAgent
from trafma.channel import Channel
from trafma.config import LinkCfg, load_config
from trafma.coordinator import Coordinator
from trafma.messages import validate

VIEW = {"scope": "incoming_local", "edge_id": "N_in", "speed_mps": 5.0, "accel_mps2": 0.0, "lane_pos_m": 100.0,
        "waiting_time_s": 0.0, "dist_to_stop_line_m": 190.0, "signal_state": None, "leader_gap_m": None,
        "leader_speed_mps": None, "neighbor_count": 0, "neighbor_ahead_count": 0, "neighbor_behind_count": 0,
        "neighbor_halting_count": 0, "neighbor_mean_speed_mps": None}
RVIEW = {"approach": "N", "zone_m": 150.0, "coverage_m": 150.0, "vehicle_count": 2, "halting_count": 1,
         "queue_length_m": 7.0, "queue_vehicle_count": 1, "mean_speed_mps": 2.0, "density_veh_per_km": 13.3,
         "mean_waiting_time_s": 1.0, "max_waiting_time_s": 2.0, "outbound_count": 0, "signal_state": "r",
         "signal_remaining_s": 5.0, "signal_phase_index": 3}


@pytest.fixture
def cfg():
    return load_config()


def setup(cfg):
    reg = AgentRegistry(cfg)
    reg.on_depart("veh_0", "N", 1.0)
    rsu = reg.rsus["N"]
    return reg, rsu, reg.vehicles["veh_0"], Coordinator(cfg)


def relay(rsu, v, now, coord):
    env = v.report(VIEW, now).to_dict()
    ok, why = rsu.receive_from_child(env, now)
    b = rsu.flush_relay(now)
    return env, ok, why, b


# ------------------------------------------------------------------ validation
def test_valid_messages(cfg):
    _, rsu, v, _ = setup(cfg)
    assert validate(v.report(VIEW, 1.0).to_dict()) == []
    assert validate(rsu.summarise(RVIEW, 1.0).to_dict()) == []


@pytest.mark.parametrize("mut,frag", [
    (lambda d: d.pop("sim_time"), "missing"),
    (lambda d: d.update(version="2.0"), "version"),
    (lambda d: d.update(schema="other"), "schema"),
    (lambda d: d.update(msg_type="mystery"), "msg_type"),
    (lambda d: d.update(seq=-1), "seq"),
    (lambda d: d.update(sim_time=float("nan")), "sim_time"),
    (lambda d: d.update(msg_id="forged#9"), "msg_id"),
    (lambda d: d["payload"].pop("speed_mps"), "payload missing"),
    (lambda d: d["payload"].update(leader_gap_m=None, unavailable=[]), "unavailable"),
    (lambda d: d.update(evidence="direct"), "evidence"),
])
def test_malformed_messages_rejected(cfg, mut, frag):
    _, _, v, _ = setup(cfg)
    d = copy.deepcopy(v.report(VIEW, 1.0).to_dict())
    d["payload"]["unavailable"] = sorted(d["payload"]["unavailable"])
    mut(d)
    probs = validate(d)
    assert probs and any(frag in p for p in probs), probs


def test_validate_never_raises_on_garbage():
    for g in (None, 5, "x", [], {}, {"schema": 1}):
        assert validate(g)


def test_reserved_fields_explicitly_unavailable(cfg):
    _, rsu, _, _ = setup(cfg)
    assert rsu.summarise(RVIEW, 1).reserved == {"trust": "unavailable", "signature": "unavailable"}


# ------------------------------------------------------------------ RSU relay
def test_rsu_rejects_duplicate_stale_and_foreign(cfg):
    reg, rsu, v, _ = setup(cfg)
    env = v.report(VIEW, 10.0).to_dict()
    assert rsu.receive_from_child(env, 10.0) == (True, "accepted")
    assert rsu.receive_from_child(env, 10.0) == (False, "duplicate")
    old = v.report(VIEW, 10.0).to_dict()                       # seq 1 but sim_time old
    old["sim_time"] = 1.0
    assert rsu.receive_from_child(old, 10.0)[1] == "stale"
    reg.on_depart("veh_9", "E", 2.0)                            # child of RSU-E, sent to RSU-N
    foreign = reg.vehicles["veh_9"].report(VIEW, 10.0).to_dict()
    assert rsu.receive_from_child(foreign, 10.0)[1] == "not_my_child"
    ooo = v.report(VIEW, 10.0).to_dict(); v.report(VIEW, 10.0)
    later = v.report(VIEW, 10.0).to_dict()
    assert rsu.receive_from_child(later, 10.0)[0] is True
    assert rsu.receive_from_child(ooo, 10.0)[1] == "out_of_order"


def test_relay_adds_parent_to_hop_path(cfg):
    _, rsu, v, _ = setup(cfg)
    _, ok, _, b = relay(rsu, v, 1.0, None)
    assert ok and b.payload["reports"][0]["hops"] == ["veh_0", "rsu_N"]


# ------------------------------------------------------------------ coordinator
def test_vehicle_to_coordinator_path_and_evidence_kinds(cfg):
    _, rsu, v, co = setup(cfg)
    _, _, _, b = relay(rsu, v, 1.0, co)
    s = rsu.summarise(RVIEW, 1.0)
    assert co.ingest(s.to_dict(), 1.0)[0][1] == "accepted"
    assert co.ingest(b.to_dict(), 1.0)[0][1] == "accepted"
    n = co.snapshot(1.0)["approaches"]["N"]
    assert n["direct"]["observation"]["vehicle_count"] == 2              # RSU's own sensing
    assert n["relayed_child"]["reporting_vehicles"] == 1                  # kept separate, never summed
    assert n["status"] == "ok"
    assert co.snapshot(1.0)["approaches"]["E"]["status"] == "no_data"
    assert co.snapshot(1.0)["approaches"]["E"]["direct"] is None          # unknown is None, not zeros


def test_duplicate_bundle_not_double_counted(cfg):
    _, rsu, v, co = setup(cfg)
    _, _, _, b = relay(rsu, v, 1.0, co)
    assert co.ingest(b.to_dict(), 1.0)[0][1] == "accepted"
    assert co.ingest(copy.deepcopy(b.to_dict()), 1.0)[0][2] == "duplicate"
    assert co.totals()["relayed_accepted"] == 1 and co.totals()["rejected"]["duplicate"] == 1


def test_stale_summary_rejected_and_marked(cfg):
    _, rsu, _, co = setup(cfg)
    s = rsu.summarise(RVIEW, 1.0).to_dict()
    assert co.ingest(s, 20.0)[0][2] == "stale"
    ok = rsu.summarise(RVIEW, 20.0).to_dict()
    co.ingest(ok, 20.0)
    snap = co.snapshot(30.0)["approaches"]["N"]
    assert snap["status"] == "stale" and snap["direct"]["stale"] is True


def test_forged_provenance_rejected(cfg):
    _, rsu, v, co = setup(cfg)
    _, _, _, b = relay(rsu, v, 1.0, co)
    bd = copy.deepcopy(b.to_dict())
    bd["payload"]["reports"][0]["hops"] = ["veh_0", "rsu_E"]            # claims a different parent
    assert co.ingest(bd, 1.0)[0][2].startswith("bad_provenance")
    bd2 = copy.deepcopy(b.to_dict()); bd2["sender_id"] = "rsu_E"        # sender != declared rsu
    assert co.ingest(bd2, 1.0)[0][2].startswith("bad_provenance")
    assert co.ingest({"junk": 1}, 1.0)[0][1] == "rejected"


def test_rsu_summary_for_wrong_approach_rejected(cfg):
    _, rsu, _, co = setup(cfg)
    bad = rsu.summarise(dict(RVIEW, approach="E"), 1.0).to_dict()        # rsu_N claiming to be E
    assert co.ingest(bad, 1.0)[0][2].startswith("bad_provenance")


def test_departed_vehicle_drops_from_live_view(cfg):
    _, rsu, v, co = setup(cfg)
    _, _, _, b = relay(rsu, v, 1.0, co)
    co.ingest(b.to_dict(), 1.0)
    assert co.snapshot(2.0)["approaches"]["N"]["relayed_child"]["reporting_vehicles"] == 1
    assert co.snapshot(60.0)["approaches"]["N"]["relayed_child"]["reporting_vehicles"] == 0


def test_registry_rules(cfg):
    reg = AgentRegistry(cfg)
    assert sorted(reg.rsus) == sorted(APPROACHES) and len(reg.rsus) == 4
    reg.on_depart("a", "S", 1.0)
    with pytest.raises(RuntimeError):
        reg.on_depart("a", "S", 2.0)
    reg.on_arrive("a", 9.0)
    assert reg.parent_of("a") == "rsu_S" and "a" not in reg.vehicles
    with pytest.raises(RuntimeError):
        reg.on_arrive("ghost", 9.0)


# ------------------------------------------------------------------ channel
def test_perfect_channel_default():
    ch = Channel("x", LinkCfg(), 1)
    assert ch.send("m", 1.0) == ["queued"] and ch.deliver_due(1.0) == [("m", "orig")]


def test_channel_delay_loss_dup_deterministic():
    ch = Channel("x", LinkCfg(delay_s=2.0), 1)
    ch.send("m", 1.0)
    assert ch.deliver_due(2.0) == [] and ch.deliver_due(3.0) == [("m", "orig")]
    assert Channel("x", LinkCfg(loss_prob=1.0), 1).send("m", 0) == ["lost"]
    d = Channel("x", LinkCfg(duplicate_prob=1.0), 1)
    assert d.send("m", 0) == ["queued", "duplicated"] and len(d.deliver_due(0)) == 2
    seq = lambda: [Channel("y", LinkCfg(loss_prob=0.5), 5).send(i, 0) for i in range(30)]
    assert seq() == seq()
