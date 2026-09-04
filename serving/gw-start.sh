#!/bin/bash
# Boot script for the GLM-5.3-Flash EXL3 lane (MiaAI-Lab image) under the
# gateway's multi-node custom-container mode. One script for every rank: the
# gateway tells each container where it stands (GW_NODE_RANK, GW_NNODES,
# GW_HEAD_IP, GW_NODE_IP, GW_MASTER_PORT); rank 0 serves HTTP, the others run
# --headless. Everything else is MiaAI-Lab's inner script, unchanged in
# substance: apply the runtime patches, then exec vllm serve.
#
# Knobs (all env, all optional — defaults are MiaAI-Lab's production values):
#   MODEL_DIR             weights directory (required)
#   SERVED_MODEL_NAME     model id at /v1                 (GLM-5.3-Flash-EXL3)
#   PORT                  HTTP port on rank 0             (8888)
#   TP                    tensor parallel size            (= GW_NNODES)
#   QUANTIZATION          exl3
#   KV_CACHE_DTYPE        fp8
#   MAX_MODEL_LEN         1000000
#   GPU_MEM_UTIL          0.87
#   MAX_NUM_SEQS          4
#   MAX_NUM_BATCHED_TOKENS 2048
#   ENFORCE_EAGER         0 (CUDA graphs on)
#   SPEC_METHOD           mtp | dflash | none             (mtp)
#   MTP_TOKENS            2
#   DFLASH_MODEL_DIR / DFLASH_TOKENS / DFLASH_DRAFT_TP    (dflash only)
#   CHAT_TEMPLATE         /opt/glm53/chat_template.jinja
#   LANGUAGE_MODEL_ONLY   0 (vision on) | 1
#   LIMIT_MM              {"image":4,"video":1}
#   SKIP_MM_PROFILING     1
#   ABLIT, ABLIT_METHOD, ABLIT_DIRECTION, ABLIT_LAYERS, ABLIT_ALPHA, ABLIT_INCLUDE_MTP
#   EXTRA_ARGS            extra vllm serve args, space separated
set -euo pipefail

RANK="${GW_NODE_RANK:?GW_NODE_RANK not set — set by the gateway multi-node custom mode}"
NNODES="${GW_NNODES:?}"
HEAD_IP="${GW_HEAD_IP:?}"
MASTER_PORT="${GW_MASTER_PORT:-29521}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-${GW_NODE_IP:-}}"

say() { echo "[glm53-exl3-rank${RANK}] $*"; }

MODEL_DIR="${MODEL_DIR:?MODEL_DIR not set}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
PORT="${PORT:-8888}"
TP="${TP:-$NNODES}"

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank "${RANK}"
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ "${RANK}" != "0" ] && ARGS+=(--headless)
[ "${ENFORCE_EAGER:-0}" = "1" ] && ARGS+=(--enforce-eager)
QUANTIZATION="${QUANTIZATION:-exl3}"
[ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
ARGS+=(--max-model-len "${MAX_MODEL_LEN:-1000000}")
ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL:-0.87}")
ARGS+=(--max-num-seqs "${MAX_NUM_SEQS:-4}")
ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}")
ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}")

SPEC_METHOD="${SPEC_METHOD:-mtp}"
if [ "${SPEC_METHOD}" = "dflash" ]; then
    : "${DFLASH_MODEL_DIR:?SPEC_METHOD=dflash needs DFLASH_MODEL_DIR}"
    DFLASH_SPEC="{\"method\":\"dflash\",\"model\":\"${DFLASH_MODEL_DIR}\",\"num_speculative_tokens\":${DFLASH_TOKENS:-7},\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\""
    if [ -n "${DFLASH_DRAFT_TP:-}" ]; then
        DFLASH_SPEC+=",\"draft_tensor_parallel_size\":${DFLASH_DRAFT_TP}"
    fi
    [ -n "${DFLASH_SPEC_EXTRA:-}" ] && DFLASH_SPEC+=",${DFLASH_SPEC_EXTRA}"
    DFLASH_SPEC+="}"
    ARGS+=(--speculative-config "${DFLASH_SPEC}")
elif [ "${SPEC_METHOD}" = "none" ]; then
    :
elif [ "${MTP_TOKENS:-2}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS:-2}}")
fi

CHAT_TEMPLATE="${CHAT_TEMPLATE:-/opt/glm53/chat_template.jinja}"
if [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
    say "language-model-only: no vision tower"
else
    LIMIT_MM_DEFAULT='{"image":4,"video":1}'
    ARGS+=(--limit-mm-per-prompt "${LIMIT_MM:-$LIMIT_MM_DEFAULT}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-default} chat-template=${CHAT_TEMPLATE}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }

# Runtime patches, in MiaAI-Lab's order. Each is a no-op if absent.
for p in patch_glm_video_placeholders patch_suppress_stops_in_reasoning patch_scheduler_decode_floor \
         patch_glm5_drafter_group patch_hybrid_prefix_hit patch_xgrammar_termination \
         patch_kpool_tail_slotmap patch_spinwait patch_exl3_grouped patch_ablit; do
    if [ -f "/opt/glm53/${p}.py" ]; then
        python3 "/opt/glm53/${p}.py"
    fi
done
if [ "${ABLIT:-0}" = "1" ]; then
    say "ablit: o_proj orthogonalization ON (method=${ABLIT_METHOD:-auto} direction=${ABLIT_DIRECTION:-dealign} layers=${ABLIT_LAYERS:-15-45} alpha=${ABLIT_ALPHA:-3.0} mtp=${ABLIT_INCLUDE_MTP:-1})"
else
    say "ablit: off — stock o_proj weights"
fi

say "rank ${RANK}/${NNODES} head=${HEAD_IP}:${MASTER_PORT} host_ip=${VLLM_HOST_IP}"
say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
