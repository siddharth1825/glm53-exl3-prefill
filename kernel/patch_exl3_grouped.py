#!/usr/bin/env python3
"""Runtime patch (run by gw-start.sh before vllm starts): hook the grouped prefill into vllm's exl3 overlay.
Appends an env-gated install block to the installed exl3.py; idempotent."""
from pathlib import Path

target = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py")
marker = "# --- exl3 grouped prefill hook (design 1) ---"
text = target.read_text()
if marker in text:
    print("exl3 grouped prefill hook already present")
else:
    block = f'''

{marker}
import os as _gp_os, sys as _gp_sys
if _gp_os.environ.get("EXL3_GROUPED_PREFILL", "0") != "0":
    _gp_dir = _gp_os.path.dirname(_gp_os.environ.get("EXL3_GROUPED_SRC", "/opt/glm53/grouped/exl3_grouped_prefill.cu"))
    if _gp_dir not in _gp_sys.path:
        _gp_sys.path.insert(0, _gp_dir)
    import exl3_grouped_runtime as _gp_rt
    _gp_rt.install(_gp_sys.modules[__name__])
'''
    target.write_text(text + block)
    print("exl3 grouped prefill hook appended to", target)
