from __future__ import annotations

import os


if os.environ.get("MREC_PATCH_VLLM_MISTRAL_COMMON_TOKENIZER") == "1":
    try:
        from fact_checking.utils.vllm_mistral_common_patch import (
            apply_vllm_mistral_common_tokenizer_patch,
        )

        apply_vllm_mistral_common_tokenizer_patch()
    except Exception:
        pass
