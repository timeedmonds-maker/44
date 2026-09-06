from __future__ import annotations

import json
import runpy

import numpy as np

_old_default = json.JSONEncoder.default


def _numpy_safe_default(self, obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _old_default(self, obj)


json.JSONEncoder.default = _numpy_safe_default
runpy.run_module("freeze_spin.prove_frame_c_left_above_rim_metric_camera_v37", run_name="__main__")
