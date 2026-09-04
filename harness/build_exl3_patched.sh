#!/usr/bin/env bash
# Build the M-tiled exllamav3_ext inside the MiaAI EXL3 image, as an overlay.
#
# Mirrors the image's own recipe (Dockerfile L410-427): pinned tarball, the
# aarch64 stub patch, CUDA headers from nvidia/cu13, TORCH_CUDA_ARCH_LIST=12.1a.
# Then copies our patched .cu/.cuh files over the tree and builds. Output: the
# rebuilt extension .so, exported to $OUT on the host for bind-mounting over
# the image's copy at serve time. Nothing in the running image is modified.
#
# Run ON spark-head:  bash build_exl3_patched.sh
set -euo pipefail

IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3
COMMIT=c5d9c657966ffeeaa9353f0cc899f18629da4a13
PATCH_DIR=$HOME/glm-opt/exl3patch          # exllamav3/exllamav3_ext/quant/... (from make_exl3_patch.py)
OUT=$HOME/glm-opt/exl3build                 # rebuilt .so + build log land here
mkdir -p "$OUT"

docker run --rm --gpus all \
  -v "$PATCH_DIR:/patch:ro" \
  -v "$OUT:/out" \
  --entrypoint bash "$IMAGE" -c '
set -eux
mkdir -p /tmp/exllamav3
curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/'"$COMMIT"'.tar.gz" | tar -xz -C /tmp/exllamav3 --strip-components=1
python3 /opt/glm53/patch_exl3_ext_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext
# overlay our patched files (exact tree layout)
cp -rv /patch/exllamav3/. /tmp/exllamav3/exllamav3/
# record what we are building
( cd /tmp/exllamav3 && find exllamav3/exllamav3_ext/quant -name "*m64*" ; grep -n "MOE_TM" exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh | head -3 )
export CPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CPATH"
export C_INCLUDE_PATH="$CPATH"
cd /tmp/exllamav3
# build in place (no install): produces exllamav3_ext*.so under build/
TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=8 python3 setup.py build_ext --inplace 2>&1 | tee /out/build.log | grep -E "error|Error|warning: .*regist|spill|ptxas info.*exl3_moe_kernel|^copying|Finished|-- Build" || true
SO=$(find /tmp/exllamav3 -name "exllamav3_ext*.so" | head -1)
test -n "$SO"
cp -v "$SO" /out/
ORIG=$(python3 -c "import exllamav3_ext; print(exllamav3_ext.__file__)")
echo "image .so: $ORIG"; ls -la "$ORIG" /out/*.so
# smoke: the new instance symbols exist in the rebuilt extension
python3 - <<EOF
import ctypes, subprocess
so = "$SO"
syms = subprocess.run(["nm", "-DC", so], capture_output=True, text=True).stdout
for s in ("exl3_moe_kernel_k4_n128_m64", "exl3_moe_kernel_k0_n128_m64", "exl3_moe_prefill_kernel"):
    print("  symbol", s, "present" if s in syms else "MISSING")
EOF
'
echo "build artefacts in $OUT:"; ls -la "$OUT"
