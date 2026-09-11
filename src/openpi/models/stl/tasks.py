"""Task dispatch shared by the CLI and in-memory STL interface."""

TASKS = (
    "detect",
    "ground-single",
    "ground-multi",
    "detect-text",
    "ground-text",
    "gui-box",
    "gui-point",
    "point",
    "custom",
)
TASKS_REQUIRING_QUERY = frozenset(TASKS) - {"detect", "detect-text"}


def run_task(worker, image, task, *, query=None, categories=None, **generation_args):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if task == "detect" and not categories:
        raise ValueError("categories is required for detect")
    if task in TASKS_REQUIRING_QUERY and not query:
        raise ValueError(f"query is required for {task}")
    if task == "detect":
        return worker.detect(image, categories, **generation_args)
    if task == "ground-single":
        return worker.ground_single(image, query, **generation_args)
    if task == "ground-multi":
        return worker.ground_multi(image, query, **generation_args)
    if task == "detect-text":
        return worker.detect_text(image, **generation_args)
    if task == "ground-text":
        return worker.ground_text(image, query, **generation_args)
    if task in {"gui-box", "gui-point"}:
        return worker.ground_gui(
            image, query, output_type=task.removeprefix("gui-"), **generation_args
        )
    if task == "point":
        return worker.point(image, query, **generation_args)
    return worker.predict(image, query, **generation_args)
