import importlib
import sys
from pathlib import Path


def test_app_module_imports():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mod = importlib.import_module("app")
    assert mod is not None
    assert hasattr(mod, "main")
