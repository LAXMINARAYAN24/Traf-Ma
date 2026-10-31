import numpy as np
import pytest
import torch

from trafma.features import FEATURE_DIM, FEATURE_SCHEMA_HASH, SEQUENCE_LENGTH, FeatureWindow
from trafma.federation import (
    ModelUpdate,
    aggregate_flat,
    aggregate_hierarchical,
    model_hash,
)
from trafma.models import GRUPredictor, multitask_loss
from trafma.federated_training import _partition_iid_windows


def _update(client, tensors, base, n=1, **overrides):
    fields = {
        "client_id": client,
        "client_type": "vehicle",
        "round_id": 3,
        "base_model_version": 7,
        "base_model_hash": model_hash(base),
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "sample_count": n,
        "tensors": tensors,
        "received_at_s": 9.0,
        "group_id": "rsu_N",
        "training_time_s": 0.1,
        "training_metadata": {"epochs": 1.0, "training_loss": 0.4},
    }
    fields.update(overrides)
    return ModelUpdate(**fields)


def test_gru_has_both_heads_expected_shapes_and_finite_gradients():
    model = GRUPredictor()
    features = torch.zeros(4, SEQUENCE_LENGTH, FEATURE_DIM)
    output = model(features)
    assert output["queue_ratio"].shape == (4,)
    assert output["class_logits"].shape == (4, 3)
    assert torch.all(output["queue_ratio"] >= 0)
    loss, parts = multitask_loss(
        output, torch.tensor([0.0, 0.1, 0.4, 1.0]), torch.tensor([0, 0, 2, 2])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert set(parts) == {"regression", "classification"}


def test_gru_rejects_bad_tensor_shapes():
    with pytest.raises(ValueError, match="expected input"):
        GRUPredictor()(torch.zeros(2, 9, FEATURE_DIM))


def test_flat_fedavg_is_sample_weighted_and_validates_rejections():
    base = {"weight": np.array([0.0, 0.0], dtype=np.float32)}
    updates = [
        _update("small", {"weight": np.array([10.0, 4.0], dtype=np.float32)}, base, n=1),
        _update("large", {"weight": np.array([2.0, 8.0], dtype=np.float32)}, base, n=3),
        _update("late", {"weight": np.array([99.0, 99.0], dtype=np.float32)}, base,
                n=100, received_at_s=11.0),
        _update("wrong", {"weight": np.array([1.0, 1.0], dtype=np.float32)}, base,
                n=1, base_model_version=6),
        _update("nan", {"weight": np.array([np.nan, 1.0], dtype=np.float32)}, base, n=1),
    ]

    result = aggregate_flat(updates, base, round_id=3, model_version=7, deadline_s=10.0)

    np.testing.assert_allclose(result.tensors["weight"], [4.0, 7.0])
    assert result.sample_mass == 4
    assert result.accepted_clients == ["small", "large"]
    assert result.rejected_clients == {
        "late": "late",
        "wrong": "wrong_base_model",
        "nan": "non_finite_tensor",
    }
    assert result.serialized_bytes["small"] > 0


def test_duplicate_client_submissions_are_not_double_counted():
    base = {"weight": np.array([0.0], dtype=np.float32)}
    update = _update("same", {"weight": np.array([5.0], dtype=np.float32)}, base, n=4)
    result = aggregate_flat([update, update], base, round_id=3, model_version=7, deadline_s=10)
    assert result.sample_mass == 0
    assert result.skipped
    assert result.rejected_clients["same"] == "duplicate_client"
    np.testing.assert_array_equal(result.tensors["weight"], base["weight"])


def test_hierarchical_fedavg_equals_flat_with_unequal_nonempty_groups():
    base = {"weight": np.array([0.0, 0.0], dtype=np.float64)}
    updates = [
        _update("n1", {"weight": np.array([4.0, 1.0])}, base, n=2, group_id="rsu_N"),
        _update("n2", {"weight": np.array([0.0, 7.0])}, base, n=1, group_id="rsu_N"),
        _update("e1", {"weight": np.array([10.0, 3.0])}, base, n=8, group_id="rsu_E"),
        _update("s1", {"weight": np.array([999.0, 999.0])}, base, n=1, group_id="rsu_INVALID"),
    ]

    flat = aggregate_flat(updates[:3], base, round_id=3, model_version=7, deadline_s=10)
    hierarchical = aggregate_hierarchical(
        updates, base, round_id=3, model_version=7, deadline_s=10
    )

    np.testing.assert_allclose(hierarchical.tensors["weight"], flat.tensors["weight"], rtol=0, atol=1e-12)
    assert hierarchical.sample_mass == 11
    assert hierarchical.groups["rsu_N"]["sample_mass"] == 3
    assert hierarchical.groups["rsu_E"]["sample_mass"] == 8
    assert hierarchical.groups["rsu_S"]["sample_mass"] == 0
    assert hierarchical.groups["rsu_W"]["sample_mass"] == 0
    assert hierarchical.rejected_clients["s1"] == "invalid_group"


def test_hierarchical_fedavg_counts_persistent_rsu_sensor_update_once():
    base = {"weight": np.array([0.0], dtype=np.float64)}
    updates = [
        _update("vehicle_a", {"weight": np.array([2.0])}, base, n=2, group_id="rsu_N"),
        _update("rsu_N", {"weight": np.array([8.0])}, base, n=3,
                client_type="rsu", group_id="rsu_N"),
        _update("vehicle_e", {"weight": np.array([5.0])}, base, n=5, group_id="rsu_E"),
        _update("rsu_E", {"weight": np.array([4.0])}, base, n=2,
                client_type="rsu", group_id="rsu_E"),
        _update("misidentified_rsu", {"weight": np.array([100.0])}, base, n=20,
                client_type="rsu", group_id="rsu_N"),
    ]

    flat = aggregate_flat(updates[:4], base, round_id=3, model_version=7, deadline_s=10.0)
    result = aggregate_hierarchical(
        updates, base, round_id=3, model_version=7, deadline_s=10.0
    )

    np.testing.assert_allclose(result.tensors["weight"], [61.0 / 12.0])
    np.testing.assert_allclose(result.tensors["weight"], flat.tensors["weight"], rtol=0, atol=1e-12)
    assert result.sample_mass == 12
    assert result.groups["rsu_N"]["sample_mass"] == 5
    assert result.groups["rsu_N"]["client_ids"] == ["vehicle_a", "rsu_N"]
    assert result.groups["rsu_E"]["sample_mass"] == 7
    assert result.groups["rsu_E"]["client_ids"] == ["vehicle_e", "rsu_E"]
    assert result.rejected_clients["misidentified_rsu"] == "invalid_rsu_learner_identity"
    assert result.serialized_bytes["misidentified_rsu"] > 0


def test_synthetic_iid_partition_balances_class_distribution_deterministically():
    samples = [
        FeatureWindow(
            "dataset", f"run_{index}", "train", "scenario", f"run_{index}/veh_1",
            "vehicle", "N", float(index), 150.0, [], None, float(label), label,
        )
        for label in range(3)
        for index in range(12)
    ]

    first = _partition_iid_windows(samples, 4, seed=7)
    second = _partition_iid_windows(samples, 4, seed=7)

    assert first == second
    assert set(first) == {f"iid_client_{index:03d}" for index in range(4)}
    for partition in first.values():
        assert [sum(sample.congestion_class == label for sample in partition)
                for label in range(3)] == [3, 3, 3]


def test_empty_federated_round_keeps_prior_weights():
    base = {"weight": np.array([1.0, 2.0], dtype=np.float32)}
    result = aggregate_flat([], base, round_id=3, model_version=7, deadline_s=10)
    assert result.sample_mass == 0
    assert result.skipped
    np.testing.assert_array_equal(result.tensors["weight"], base["weight"])


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"sample_count": 0}, "invalid_sample_count"),
        ({"feature_schema_hash": "wrong"}, "wrong_feature_schema"),
        ({"tensors": {"other": np.array([1.0], dtype=np.float32)}}, "tensor_signature_mismatch"),
        ({"tensors": {"weight": np.array([np.inf], dtype=np.float32)}}, "non_finite_tensor"),
        ({"training_metadata": {"raw_examples": 1.0}}, "disallowed_training_metadata"),
        ({"received_at_s": float("nan")}, "invalid_received_time"),
    ],
)
def test_flat_fedavg_rejects_invalid_update_metadata_and_tensors(overrides, expected_reason):
    base = {"weight": np.array([0.0], dtype=np.float32)}
    fields = dict(overrides)
    tensors = fields.pop("tensors", {"weight": np.array([1.0], dtype=np.float32)})
    update = _update("invalid", tensors, base, **fields)

    result = aggregate_flat([update], base, round_id=3, model_version=7, deadline_s=10.0)

    assert result.sample_mass == 0
    assert result.skipped
    assert result.rejected_clients == {"invalid": expected_reason}
    np.testing.assert_array_equal(result.tensors["weight"], base["weight"])
