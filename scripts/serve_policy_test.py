import pytest
import tyro

from openpi.training import config as _config
from scripts import serve_policy


@pytest.mark.parametrize("config_name", ["pi05_piper_full_finetune", "pi05_piper_lora_finetune"])
@pytest.mark.parametrize("image_size", [None, 224, 336, 448])
def test_checkpoint_image_resolution(tmp_path, config_name, image_size):
    checkpoint = serve_policy.Checkpoint(
        config=config_name, dir=str(tmp_path), asset_id="local/piper", image_size=image_size
    )
    config, _ = serve_policy._prepare_checkpoint(checkpoint)  # noqa: SLF001
    base = _config.get_config(config_name)
    expected = image_size or 224
    assert config.model.image_resolution == (expected, expected)
    assert base.model.image_resolution == (224, 224)
    assert config.model.paligemma_variant == base.model.paligemma_variant
    if image_size is None:
        assert config.model is base.model


def test_checkpoint_image_size_cli(tmp_path):
    args = tyro.cli(
        serve_policy.Args,
        args=[
            "policy:checkpoint",
            "--policy.config=pi05_piper_lora_finetune",
            f"--policy.dir={tmp_path}",
            "--policy.asset-id=local/piper",
            "--policy.image-size=448",
        ],
    )
    config, _ = serve_policy._prepare_checkpoint(args.policy)  # noqa: SLF001
    assert config.model.image_resolution == (448, 448)


@pytest.mark.parametrize("image_size", [0, -14, 225])
def test_checkpoint_rejects_invalid_image_size(tmp_path, image_size):
    checkpoint = serve_policy.Checkpoint(
        config="pi05_piper_lora_finetune", dir=str(tmp_path), asset_id="local/piper", image_size=image_size
    )
    with pytest.raises(ValueError, match="positive dimensions divisible by 14"):
        serve_policy._prepare_checkpoint(checkpoint)  # noqa: SLF001
