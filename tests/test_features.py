import pytest

from trafma.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FeatureContext,
    FeatureError,
    adapt_observation,
)


@pytest.fixture
def context():
    return FeatureContext(
        zone_m=150.0,
        arm_length_m=300.0,
        incoming_edge_length_m=292.8,
        vehicle_radius_m=100.0,
        vehicle_length_m=5.0,
        min_gap_m=2.5,
        max_speed_mps=13.89,
        max_accel_mps2=2.6,
        max_decel_mps2=4.5,
        halting_speed_mps=0.1,
        signal_cycle_s=120.0,
    )


def test_feature_schema_order_and_dimension():
    assert FEATURE_DIM == 22
    assert len(FEATURE_NAMES) == 22
    assert FEATURE_NAMES[0] == "local_mean_speed_mps"
    assert FEATURE_NAMES[10] == "local_mean_speed_mps_available"
    assert FEATURE_NAMES[20:] == ("client_vehicle", "client_rsu")


def test_rsu_empty_zone_preserves_missing_means_and_measured_zeroes(context):
    view = {
        "zone_m": 150.0,
        "vehicle_count": 0,
        "halting_count": 0,
        "mean_speed_mps": None,
        "mean_accel_mps2": None,
        "mean_dist_to_stop_m": None,
        "mean_waiting_time_s": None,
        "signal_state": "G",
        "signal_remaining_s": 24.0,
        "unavailable": [
            "mean_speed_mps", "mean_accel_mps2", "mean_dist_to_stop_m", "mean_waiting_time_s",
        ],
    }

    vector = adapt_observation(view, "rsu", context)

    assert len(vector) == FEATURE_DIM
    assert vector[:7] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pytest.approx(0.2)]
    assert vector[7:10] == [0.0, 0.0, 1.0]
    assert vector[10:20] == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert vector[20:] == [0.0, 1.0]


def test_vehicle_mean_speed_includes_self_once_and_capacity_is_clipped(context):
    view = {
        "scope": "incoming_local",
        "speed_mps": 13.89,
        "accel_mps2": 0.0,
        "dist_to_stop_line_m": 287.7,
        "waiting_time_s": 0.0,
        "lane_pos_m": 5.1,
        "neighbor_count": 1,
        "neighbor_mean_speed_mps": 7.0,
        "neighbor_halting_count": 0,
        "signal_state": None,
        "signal_remaining_s": None,
        "unavailable": ["signal_state", "signal_remaining_s"],
    }

    vector = adapt_observation(view, "vehicle", context)

    assert vector[0] == pytest.approx(((13.89 + 7.0) / 2) / 13.89)
    visible_length = min(292.8, 5.1 + 100.0) - max(0.0, 5.1 - 100.0)
    assert vector[3] == pytest.approx(2 / (visible_length / 7.5))
    assert vector[7:10] == [0.0, 0.0, 0.0]
    assert vector[17:20] == [0.0, 0.0, 0.0]
    assert vector[20:] == [1.0, 0.0]


def test_vehicle_out_of_scope_and_inconsistent_missingness_are_rejected(context):
    with pytest.raises(FeatureError, match="incoming_local"):
        adapt_observation({"scope": "in_junction"}, "vehicle", context)

    view = {
        "scope": "incoming_local",
        "speed_mps": 0.0,
        "accel_mps2": 0.0,
        "dist_to_stop_line_m": 1.0,
        "waiting_time_s": 0.0,
        "lane_pos_m": 291.8,
        "neighbor_count": 0,
        "neighbor_mean_speed_mps": None,
        "neighbor_halting_count": 0,
        "signal_state": None,
        "signal_remaining_s": None,
        "unavailable": ["neighbor_mean_speed_mps", "signal_remaining_s"],
    }
    with pytest.raises(FeatureError, match="signal_state is null"):
        adapt_observation(view, "vehicle", context)
