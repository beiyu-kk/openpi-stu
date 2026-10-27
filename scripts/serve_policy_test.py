import pytest
import tyro

from openpi.models import pi0_fast
from openpi.training import config as _config
from scripts import serve_policy


@pytest.mark.parametrize(
    ("config_name", "image_resolution"),
    [
        ("pi05_piper_full_finetune", (224, 224)),
        ("pi05_piper_lora_finetune", (224, 224)),
        ("pi05_piper_full_finetune_image448", (448, 448)),
        ("pi05_piper_lora_finetune_image448", (448, 448)),
        ("pi05_piper_full_finetune_image640", (640, 640)),
        ("pi05_piper_lora_finetune_image640", (640, 640)),
        ("pi0_fast_piper_full_finetune", (224, 224)),
        ("pi0_fast_piper_lora_finetune", (224, 224)),
        ("pi0_fast_piper_full_finetune_image448", (448, 448)),
        ("pi0_fast_piper_lora_finetune_image448", (448, 448)),
        ("pi0_fast_piper_full_finetune_image640", (640, 640)),
        ("pi0_fast_piper_lora_finetune_image640", (640, 640)),
        ("pi0_fast_piper_full_finetune_image640_pad_only", (640, 640)),
        ("pi05_piper_lora_finetune_image640_pad_only", (640, 640)),
    ],
)
def test_checkpoint_uses_registered_image_resolution(tmp_path, config_name, image_resolution):
    checkpoint = serve_policy.Checkpoint(config=config_name, dir=str(tmp_path), asset_id="local/piper")
    config, _ = serve_policy._prepare_checkpoint(checkpoint)  # noqa: SLF001
    base = _config.get_config(config_name)
    assert config.model.image_resolution == image_resolution
    assert base.model.image_resolution == image_resolution
    assert config.model.paligemma_variant == base.model.paligemma_variant
    assert config.model is base.model
    assert config.model.image_resize_mode == ("pad_only" if config_name.endswith("pad_only") else "resize_with_pad")
    if config_name.startswith("pi0_fast"):
        assert isinstance(config.model, pi0_fast.Pi0FASTConfig)


def test_legacy_image_size_cli_argument_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        tyro.cli(
            serve_policy.Args,
            args=[
                "policy:checkpoint",
                "--policy.config=pi0_fast_piper_lora_finetune",
                f"--policy.dir={tmp_path}",
                "--policy.asset-id=local/piper",
                "--policy.image-size=448",
            ],
        )
