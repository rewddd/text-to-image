import re
from typing import Any

DIFFUSERS_LORA_PATTERN = re.compile(
    r"(to_\w+_lora|lora_linear_layer)\.(up|down)\.weight"
)

KNOWN_LORA_PREFIXES = {
    "kohya-ss": lambda key: key.startswith(
        (
            "lora_te_",
            "lora_unet_",
            "lora_te2_",
            "lora_te1_",
        )
    ),
    "diffusers": DIFFUSERS_LORA_PATTERN.search,
}


def identify_lora_weights(state_dict: dict[str, Any]) -> set[str]:
    """Return a set of possible LoRA formats for the given state_dict."""
    formats = set()
    for key in state_dict.keys():
        for format, check_fn in KNOWN_LORA_PREFIXES.items():
            if check_fn(key):  # type: ignore
                formats.add(format)
                break
        else:
            formats.add("unknown")

    return formats


def determine_auxiliary_features(
    lora_formats: set[str], state_dict: dict[str, Any]
) -> set[str]:
    if len(lora_formats) != 1:
        return set()

    [lora_format] = lora_formats
    auxilary_features = set()
    state_keys = sorted(state_dict.keys())

    if lora_format == "kohya-ss":
        # LyCORIS (https://github.com/KohakuBlueleaf/LyCORIS) has multiple different
        # formats like LoCon, LoHa, LoKR, DyLoRA, etc.
        if any("_conv_" in key for key in state_keys):
            auxilary_features.add("LyCORIS")

        if any("lora_mid.weight" in key for key in state_keys):
            auxilary_features.add("LoCon")

    return auxilary_features


def stack_loras(
    state_dicts: list[dict[str, Any]],
    ratios: list[float],
) -> dict[str, Any]:
    import math

    import torch

    base_alphas, base_dims, merged_sd = {}, {}, {}  # type: ignore
    for lora_sd, ratio in zip(state_dicts, ratios):
        alphas, dims = {}, {}
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                base_alphas.setdefault(lora_module_name, alpha)
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                base_dims.setdefault(lora_module_name, dim)

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                base_alphas.setdefault(lora_module_name, alpha)

        for key in lora_sd.keys():
            if "alpha" in key:
                continue

            is_text_encoder = "te_" in key
            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            alpha_factor = math.sqrt(alpha / base_alpha)
            scale = alpha_factor * ratio
            if is_text_encoder:
                scale = alpha_factor

            if key in merged_sd:
                left_value = merged_sd[key]
                right_value = lora_sd[key]

                num_extra_dims = [1] * (len(left_value.size()) - 2)
                left_first_dim = left_value.size(dim=0)
                right_first_dim = right_value.size(dim=0)
                if left_first_dim > right_first_dim:
                    right_value = right_value.repeat(
                        left_first_dim // right_first_dim, 1, *num_extra_dims
                    )
                elif left_first_dim < right_first_dim:
                    left_value = left_value.repeat(
                        right_first_dim // left_first_dim, 1, *num_extra_dims
                    )

                left_second_dim = left_value.size(dim=1)
                right_second_dim = right_value.size(dim=1)
                if left_second_dim > right_second_dim:
                    right_value = right_value.repeat(
                        1, left_second_dim // right_second_dim, *num_extra_dims
                    )
                elif left_second_dim < right_second_dim:
                    left_value = left_value.repeat(
                        1, right_second_dim // left_second_dim, *num_extra_dims
                    )

                merged_sd[key] = left_value + right_value * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    return merged_sd
