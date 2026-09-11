"""Contract tests for the STL integration, without GPU weights."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from openpi.models.stl import STLConfig
from openpi.models.stl import locator
from openpi.models.stl.loader import load_preprocessing, resolve_model_path
from openpi.models.stl.parsing import parse_boxes, parse_points
from openpi.models.stl.tasks import run_task
from openpi.models.stl.visualization import draw_predictions


@pytest.mark.parametrize(
    ("task", "query", "categories", "expected"),
    [
        (
            "detect",
            None,
            ["book", "cup"],
            "Locate all the instances that matches the following description: book</c>cup.",
        ),
        (
            "ground-single",
            "book",
            None,
            "Locate a single instance that matches the following description: book.",
        ),
        (
            "ground-multi",
            "book",
            None,
            "Locate all the instances that match the following description: book.",
        ),
        ("ground-text", "title", None, "Please locate the text referred as title."),
        ("detect-text", None, None, "Detect all the text in box format."),
        (
            "gui-box",
            "button",
            None,
            "Locate the region that matches the following description: button.",
        ),
        ("gui-point", "button", None, "Point to: button."),
        ("point", "book", None, "Point to: book."),
        ("custom", "Keep this prompt.", None, "Keep this prompt."),
    ],
)
def test_task_prompts_match_release(task, query, categories, expected):
    worker = object.__new__(locator.LocateAnythingWorker)
    worker.predict = lambda image, question, **kwargs: (image, question, kwargs)
    image = Image.new("RGB", (32, 16))
    assert run_task(
        worker, image, task, query=query, categories=categories, temperature=0.7
    ) == (
        image,
        expected,
        {"temperature": 0.7},
    )


@pytest.mark.parametrize(
    ("task", "kwargs"),
    [
        ("unknown", {}),
        ("detect", {}),
        ("ground-text", {}),
        ("custom", {"query": ""}),
    ],
)
def test_invalid_tasks_fail_before_inference(task, kwargs):
    with pytest.raises(ValueError):
        run_task(None, None, task, **kwargs)


def test_original_coordinate_rules_and_empty_results():
    answer = "<ref>title</ref><box><820><516><875><662></box><box><500><250></box>"
    assert parse_boxes(answer, 2000, 1000) == [
        {"x1": 1640.0, "y1": 516.0, "x2": 1750.0, "y2": 662.0}
    ]
    assert parse_points(answer, 2000, 1000) == [{"x": 1000.0, "y": 250.0}]
    assert parse_boxes("<box><1><2><3></box>", 100, 100) == []
    assert parse_points("No matching text.", 100, 100) == []
    assert parse_boxes("<box><0><0><1001><1000></box>", 100, 100)[0]["x2"] == 100.1


def test_render_preserves_input_and_draws_both_geometry_types():
    image = Image.new("RGB", (200, 100), "white")
    rendered = draw_predictions(
        image, [{"x1": 10, "y1": 10, "x2": 80, "y2": 80}], [{"x": 150, "y": 50}]
    )
    assert image.getpixel((10, 10)) == (255, 255, 255)
    assert rendered.getpixel((10, 10)) == (255, 0, 0)
    assert rendered.getpixel((150, 50)) == (0, 255, 0)
    assert draw_predictions(image, [], []).tobytes() == image.tobytes()


def test_in_memory_api_reuses_worker_and_preserves_pixels(monkeypatch):
    constructed = []
    seen = []

    class Worker:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

        def ground_text(self, image, query, **kwargs):
            seen.append(np.asarray(image).copy())
            return {"answer": "<box><100><200><900><800></box>"}

    monkeypatch.setattr(locator, "LocateAnythingWorker", Worker)
    stl = locator.SceneTextLocator(STLConfig(model_path="weights"))
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 240
    for _ in range(2):
        result = stl.locate(image, query="title")
        assert result.image_size == (200, 100)
        assert result.boxes == [{"x1": 20.0, "y1": 20.0, "x2": 180.0, "y2": 80.0}]
    assert len(constructed) == 1
    np.testing.assert_array_equal(seen[0], image)
    for invalid in (
        image.astype(np.float32) / 255,
        image.transpose(2, 0, 1),
        image[:, :, 0],
    ):
        with pytest.raises(ValueError, match="HWC RGB uint8"):
            stl.locate(invalid, query="title")


def test_local_preprocessor_retains_checkpoint_settings(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<IMG_CONTEXT>": 0, "[UNK]": 1}, unk_token="[UNK]")
        )
    )
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "in_token_limit": 25600,
                "patch_size": 14,
                "merge_kernel_size": [2, 2],
                "auto_map": {"AutoImageProcessor": "must_not_execute.Code"},
            }
        )
    )
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"image_start_token": "<img>"})
    )
    (tmp_path / "chat_template.json").write_text(
        json.dumps({"chat_template": "test template"})
    )
    tokenizer, processor = load_preprocessing(tmp_path)
    assert processor.image_processor.in_token_limit == 25600
    assert processor.chat_template == "test template"
    assert type(processor).__module__.startswith("openpi.models.stl.")
    assert type(processor.image_processor).__module__.startswith("openpi.models.stl.")
    assert not hasattr(processor.image_processor, "auto_map")
    assert resolve_model_path(str(tmp_path)) == tmp_path
    with pytest.raises(FileNotFoundError):
        resolve_model_path(str(tmp_path / "missing"))


def test_import_does_not_load_models_or_optional_data_backends():
    code = (
        "import sys; import openpi.models.stl; "
        "assert 'torch' not in sys.modules; "
        "import openpi.models.stl.locate_anything.processing_locateanything; "
        "assert 'decord' not in sys.modules; assert 'lmdb' not in sys.modules; "
        "assert not any(n.startswith('transformers_modules.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_batch_configuration_cannot_silently_replace_loaded_model(monkeypatch):
    from openpi.models.stl.locate_anything.batch_utils import (
        engine_hybrid,
        hybrid_runtime,
    )

    monkeypatch.setattr(hybrid_runtime, "_model", None)
    for name in (
        "MODEL",
        "DEV",
        "DT",
        "ATTN_MODE",
        "REMOTE_ATTN_MODE",
        "VISION_ATTN_MODE",
        "STRICT_ATTN",
    ):
        monkeypatch.setattr(hybrid_runtime, name, getattr(hybrid_runtime, name))
    hybrid_runtime.configure("weights", device="cpu", attn="sdpa", vision_attn="sdpa")
    assert engine_hybrid.runtime.DEV == "cpu"
    monkeypatch.setattr(hybrid_runtime, "_model", SimpleNamespace())
    hybrid_runtime.configure("weights", device="cpu", attn="sdpa", vision_attn="sdpa")
    with pytest.raises(RuntimeError, match="one model configuration"):
        hybrid_runtime.configure("other", device="cpu")
