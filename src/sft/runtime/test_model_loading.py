from __future__ import annotations

import unittest
from unittest.mock import patch

from sft.runtime.model_loading import load_causal_lm_compatible_model


class Mistral3ForConditionalGeneration:
    full_model = object()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        return cls.full_model


class ModelLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mistral3_loader = (Mistral3ForConditionalGeneration, object())

    def test_mistral3_text_only_remains_default_for_fullft(self) -> None:
        text_model = object()
        with (
            patch("sft.runtime.model_loading._finegrained_fp8_dequantize_config", return_value=None),
            patch("sft.runtime.model_loading.AutoModelForCausalLM.from_pretrained", side_effect=ValueError),
            patch(
                "sft.runtime.model_loading._conditional_generation_loader",
                return_value=self.mistral3_loader,
            ),
            patch("sft.runtime.model_loading._load_mistral3_text_only_causal_lm", return_value=text_model),
        ):
            loaded = load_causal_lm_compatible_model("model")

        self.assertIs(loaded, text_model)

    def test_mistral3_lora_can_force_legacy_full_model_path(self) -> None:
        with (
            patch("sft.runtime.model_loading._finegrained_fp8_dequantize_config", return_value=None),
            patch("sft.runtime.model_loading.AutoModelForCausalLM.from_pretrained", side_effect=ValueError),
            patch(
                "sft.runtime.model_loading._conditional_generation_loader",
                return_value=self.mistral3_loader,
            ),
            patch("sft.runtime.model_loading._load_mistral3_text_only_causal_lm") as text_loader,
        ):
            loaded = load_causal_lm_compatible_model(
                "model",
                use_mistral3_text_only=False,
            )

        self.assertIs(loaded, Mistral3ForConditionalGeneration.full_model)
        text_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
