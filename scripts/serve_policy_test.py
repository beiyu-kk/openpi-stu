import pytest

from openpi.training import config as _config
from scripts import serve_policy


@pytest.mark.parametrize(
    ("config_name", "expected"),
    [
        ("pi05_piper_full_finetune", 224),
        ("pi05_piper_lora_finetune", 224),
        ("pi05_piper_full_finetune_336", 336),
        ("pi05_piper_lora_finetune_336", 336),
        ("pi05_piper_full_finetune_448", 448),
        ("pi05_piper_lora_finetune_448", 448),
    ],
)
def test_checkpoint_image_resolution(tmp_path, config_name, expected):
    checkpoint = serve_policy.Checkpoint(config=config_name, dir=str(tmp_path), asset_id="local/piper")
    config, _ = serve_policy._prepare_checkpoint(checkpoint)  # noqa: SLF001
    assert config.model is _config.get_config(config_name).model
    assert config.model.image_resolution == (expected, expected)
