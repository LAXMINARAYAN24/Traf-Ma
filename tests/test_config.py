import pytest
from trafma.config import ConfigError, load_config, parse_set_overrides, config_from_dict


def test_defaults_valid():
    c = load_config()
    assert c.run.vehicles == 20 and c.agents.rsu_zone_m == 150 and c.agents.vehicle_radius_m == 100


@pytest.mark.parametrize("n", [-1, 1.5, "20", True, None])
def test_invalid_n_rejected(n):
    with pytest.raises(ConfigError):
        load_config(overrides={"run": {"vehicles": n}})


def test_n_cap():
    with pytest.raises(ConfigError):
        load_config(overrides={"run": {"vehicles": 10_000}})


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(overrides={"agents": {"camera_zone_m": 100}})


def test_malformed_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("run: [unclosed")
    with pytest.raises(ConfigError, match="malformed YAML"):
        load_config(p)


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("nope.yaml")


def test_not_a_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigError):
        load_config(p)


@pytest.mark.parametrize("ov", [
    {"run": {"report_step_s": 0.7}},                    # not a multiple of step
    {"run": {"duration_s": 0}},
    {"run": {"step_s": float("nan")}},
    {"agents": {"rsu_zone_m": 400}},                    # does not fit on the arm
    {"demand": {"turn_probs": {"straight": .5, "left": .2, "right": .2}}},
    {"demand": {"profile": "custom"}},                  # custom needs weights
    {"demand": {"approach_weights": {"N": 1, "E": 1}}},
    {"comms": {"v2i": {"loss_prob": 1.5}}},
    {"labels": {"congestion_thresholds": [0.5, 0.2]}},
    {"network": {"lanes_per_direction": 2}},
])
def test_malformed_values_rejected(ov):
    with pytest.raises(ConfigError):
        load_config(overrides=ov)


def test_set_override_parsing():
    assert parse_set_overrides(["agents.rsu_zone_m=120", "run.gui=true"]) == {
        "agents": {"rsu_zone_m": 120}, "run": {"gui": True}}
    with pytest.raises(ConfigError):
        parse_set_overrides(["novalue"])
