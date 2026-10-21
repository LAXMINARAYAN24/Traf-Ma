from collections import Counter

import pytest
from trafma.config import ConfigError, load_config
from trafma.demand import generate_demand, write_routes
from trafma.netgen import movement_dest


def C(**run):
    return load_config(overrides={"run": run})


def test_same_seed_same_schedule_and_bytes(tmp_path):
    a, b = generate_demand(C(seed=7, vehicles=50)), generate_demand(C(seed=7, vehicles=50))
    assert a == b
    write_routes(a, C(seed=7), tmp_path / "a.xml"); write_routes(b, C(seed=7), tmp_path / "b.xml")
    assert (tmp_path / "a.xml").read_bytes() == (tmp_path / "b.xml").read_bytes()


def test_different_seed_differs():
    assert generate_demand(C(seed=1, vehicles=50)) != generate_demand(C(seed=2, vehicles=50))


@pytest.mark.parametrize("n", [0, 1, 20, 100])
def test_n_controls_population(n):
    assert len(generate_demand(C(vehicles=n))) == n


def test_departures_sorted_and_within_window():
    s = generate_demand(C(vehicles=200, duration_s=600))
    t = [x.depart_s for x in s]
    assert t == sorted(t) and t[-1] <= 600 - 120 + 1e-6


def test_balanced_uses_all_four_approaches():
    assert set(Counter(x.origin for x in generate_demand(C(vehicles=200)))) == set("NESW")


def test_asymmetric_skews_demand():
    cfg = load_config(overrides={"run": {"vehicles": 800}, "demand": {"profile": "asymmetric"}})
    c = Counter(x.origin for x in generate_demand(cfg))
    assert c["N"] + c["S"] > 3 * (c["E"] + c["W"])


def test_turn_geometry():
    assert movement_dest("N", "straight") == "S"
    assert (movement_dest("N", "left"), movement_dest("N", "right")) == ("E", "W")
    assert (movement_dest("E", "left"), movement_dest("E", "right")) == ("S", "N")
    assert (movement_dest("S", "left"), movement_dest("S", "right")) == ("W", "E")
    assert (movement_dest("W", "left"), movement_dest("W", "right")) == ("N", "S")


def test_no_departure_window_rejected():
    with pytest.raises(ConfigError):
        generate_demand(load_config(overrides={"run": {"duration_s": 100, "vehicles": 5}}))
