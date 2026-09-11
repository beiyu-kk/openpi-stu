"""Scene text localization with project-owned LocateAnything source."""

from .config import STLConfig

__all__ = ["LocateAnythingWorker", "SceneTextLocator", "STLConfig", "STLResult"]


def __getattr__(name):
    if name in {"LocateAnythingWorker", "SceneTextLocator"}:
        from . import locator

        return getattr(locator, name)
    if name == "STLResult":
        from .types import STLResult

        return STLResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
