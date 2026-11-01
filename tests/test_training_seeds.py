from pathlib import Path

import pytest
import yaml

from trafma.training import TrainingError, _validate_training


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", [400, 401, 402, 403, 404])
def test_reserved_training_seeds_are_rejected(seed):
    config = yaml.safe_load((ROOT / "configs" / "learning.yaml").read_text(encoding="utf-8"))
    config["training"]["seed"] = seed

    with pytest.raises(TrainingError, match="reserved and must remain unused"):
        _validate_training(config)
