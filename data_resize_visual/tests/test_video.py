from fractions import Fraction
from pathlib import Path

import pytest

from data_resize_visual.video import _encoder_command, _parse_fps, _validate_arguments


def test_parse_fractional_fps() -> None:
    assert _parse_fps("30000/1001") == Fraction(30000, 1001)
    assert _parse_fps("0/0") is None


def test_validate_rejects_input_output_alias(tmp_path: Path) -> None:
    input_path = tmp_path / "video.mp4"
    input_path.touch()

    with pytest.raises(ValueError, match="must be different"):
        _validate_arguments(input_path, input_path, 224, 224, 16, False, False)


def test_compatible_encoder_uses_yuv420p_with_bt709() -> None:
    command = _encoder_command(
        Path("input.mp4"),
        Path("output.mp4"),
        Fraction(30, 1),
        224,
        224,
        True,
        False,
    )

    assert "libx264" in command
    assert "libx264rgb" not in command
    assert "yuv420p" in command
    assert "bt709" in command


def test_lossless_encoder_is_explicit() -> None:
    command = _encoder_command(
        Path("input.mp4"),
        Path("output.mp4"),
        Fraction(30, 1),
        224,
        224,
        True,
        True,
    )

    assert "libx264rgb" in command
    assert "yuv420p" not in command
