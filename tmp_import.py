import importlib
import traceback

try:
    importlib.import_module('execution.live_scheduler')
    print('import succeeded')
except Exception:
    traceback.print_exc()
