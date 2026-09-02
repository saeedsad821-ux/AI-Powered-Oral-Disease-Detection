"""Compatibility shim: preserve `app/app.py` path while loading the real app logic from `app/main.py`.

Streamlit entrypoints cannot be named `app` when the package root is also named
`app`, so this shim executes `app/main.py` by path and clears any conflicting
`app` module alias before loading package imports.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Remove a conflicting `app` module entry if it exists as a script module.
if "app" in sys.modules and not getattr(sys.modules["app"], "__path__", None):
    del sys.modules["app"]

MAIN_PATH = Path(__file__).resolve().with_name("main.py")
spec = importlib.util.spec_from_file_location("__streamlit_main__", MAIN_PATH)
module = importlib.util.module_from_spec(spec)
module.__file__ = str(MAIN_PATH)
sys.modules["__main__"] = module
sys.modules[spec.name] = module
spec.loader.exec_module(module)
