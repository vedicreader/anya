__version__ = "0.0.1"

from .vision import *
from .core import *
from .core import Model, Pred, Preds
from .tasks import *

# Runtimes are optional extras (`pip install 'anya[onnx]'`, `[litert]`, `[coreml]`), so importing one
# eagerly here would make `import anya` fail on a machine that only has the others. `Model(name)`
# imports the runtime it actually needs, lazily, via `core.get_runtime`. `anya.onnx` and friends
# still work as ordinary submodule imports; this only stops them being *required*.
_lazy = ('onnx', 'litert', 'apple', 'hub', 'tools')

def __getattr__(name):
    "Import a runtime or helper submodule on first attribute access (`anya.litert` without a hard dependency)."
    if name in _lazy:
        from importlib import import_module
        return import_module(f'.{name}', __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__(): return sorted(list(globals()) + list(_lazy))
