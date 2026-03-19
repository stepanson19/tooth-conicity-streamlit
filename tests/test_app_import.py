import importlib
import sys
from pathlib import Path


class _DummySessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, item, value):
        self[item] = value


class _DummyStreamlit:
    def __init__(self):
        self.session_state = _DummySessionState(
            analysis_output={"status": "ok"},
            analysis_image="old-image",
        )


def test_app_module_imports():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mod = importlib.import_module("app")
    assert mod is not None
    assert hasattr(mod, "main")


def test_clear_analysis_result_resets_session_state():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    app = importlib.import_module("app")

    st = _DummyStreamlit()

    app._clear_analysis_result(st)

    assert st.session_state.analysis_output is None
    assert st.session_state.analysis_image is None
