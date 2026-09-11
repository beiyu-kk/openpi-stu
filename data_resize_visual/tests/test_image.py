from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_resize_visual import cli
from data_resize_visual.image import process_image
from data_resize_visual.resize import resize_with_pad


def test_cli_image_matches_resize_core_without_ffmpeg(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("PATH", "")
    pixels = np.arange(19 * 31 * 3, dtype=np.uint8).reshape(19, 31, 3)
    input_path = tmp_path / "photo.PNG"
    Image.fromarray(pixels).save(input_path)

    assert cli.main([str(input_path), "--height", "15", "--width", "23"]) == 0

    output_path = tmp_path / "photo_resized_23x15.png"
    with Image.open(output_path) as actual:
        assert actual.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(actual), resize_with_pad(pixels, 15, 23))
    assert str(output_path) in capsys.readouterr().err


def test_jpeg_exif_orientation_and_custom_output(tmp_path: Path, capsys) -> None:
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    pixels[:, :20] = (240, 10, 30)
    pixels[:, 20:] = (10, 240, 30)
    source = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6
    input_path = tmp_path / "phone.jpg"
    source.save(input_path, exif=exif)
    with Image.open(input_path) as decoded:
        expected = np.rot90(np.asarray(decoded), k=-1)
    output_path = tmp_path / "nested" / "upright.png"

    assert cli.main([str(input_path), "-o", str(output_path), "--width", "20", "--height", "40", "--quiet"]) == 0
    with Image.open(output_path) as actual:
        np.testing.assert_array_equal(np.asarray(actual), expected)
        assert actual.getexif().get(274, 1) == 1
    assert not capsys.readouterr().err


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_image_modes_convert_to_rgb(tmp_path: Path, mode: str) -> None:
    input_path = tmp_path / "photo.png"
    Image.new(mode, (32, 16)).save(input_path)
    output_path = process_image(input_path, tmp_path / "out.png")
    with Image.open(output_path) as actual:
        assert actual.size == (224, 224)
        assert actual.mode == "RGB"


def test_overwrite_protection_and_input_alias(tmp_path: Path) -> None:
    input_path = tmp_path / "photo.png"
    Image.new("RGB", (32, 16), "red").save(input_path)
    output_path = tmp_path / "out.png"
    output_path.write_bytes(b"keep existing output")
    with pytest.raises(FileExistsError):
        process_image(input_path, output_path)
    assert output_path.read_bytes() == b"keep existing output"
    process_image(input_path, output_path, overwrite=True)
    with Image.open(output_path) as actual:
        assert actual.size == (224, 224)
    with pytest.raises(ValueError, match="must be different"):
        process_image(input_path, input_path, overwrite=True)


def test_invalid_image_returns_error_without_output(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "broken.jpg"
    input_path.write_bytes(b"not an image")
    assert cli.main([str(input_path)]) == 1
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "broken_resized_224x224.png").exists()


@pytest.mark.parametrize("arguments", [["--width", "0"], ["--height", "-1"], ["--lossless-rgb"]])
def test_invalid_image_options(tmp_path: Path, arguments: list[str], capsys) -> None:
    input_path = tmp_path / "photo.png"
    Image.new("RGB", (32, 16)).save(input_path)
    assert cli.main([str(input_path), *arguments]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_keeps_video_dispatch(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "process_video", lambda *args, **kwargs: calls.append((args, kwargs)))
    input_path = tmp_path / "clip.mp4"
    assert cli.main([str(input_path), "--batch-size", "4", "--no-audio"]) == 0
    args, kwargs = calls[0]
    assert args == (input_path, tmp_path / "clip_resized_224x224.mp4")
    assert kwargs == {
        "height": 224,
        "width": 224,
        "batch_size": 4,
        "keep_audio": False,
        "lossless_rgb": False,
        "overwrite": False,
        "quiet": False,
    }
