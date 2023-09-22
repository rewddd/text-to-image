import copy
import re
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from fal import cached, function
from fal.toolkit import Image, ImageSizeInput, get_image_size
from pydantic import BaseModel, Field

CHECKPOINTS_DIR = Path("/data/checkpoints")
LORA_WEIGHTS_DIR = Path("/data/loras")
TEMP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"
)
ONE_MB = 1024**2
CHUNK_SIZE = 32 * ONE_MB
CACHE_PREFIX = ""

requirements = [
    "git+https://github.com/huggingface/diffusers.git@38a664a3d61e27ab18",
    "transformers",
    "xformers",
    "torch>=2.0",
    "torchvision",
    "safetensors",
    "pytorch-lightning",
    "accelerate",
    "omegaconf",
    "invisible-watermark",
    "pydantic==1.10.12",
]


_DIFFUSERS_LORA_PATTERN = re.compile(
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
    "diffusers": _DIFFUSERS_LORA_PATTERN.search,
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
            print(f"Unknown key: {key}")

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


SUPPORTED_SCHEDULERS = {
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {}),
    "DPM++ 2M Karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M SDE": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++"},
    ),
    "DPM++ 2M SDE Karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler A": ("EulerAncestralDiscreteScheduler", {}),
}


def _merge_stacked_loras(
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


@dataclass
class Model:
    pipeline: object
    last_cache_hit: float = 0

    def as_base(self) -> object:
        self.last_cache_hit = time.monotonic()

        pipe = self.pipeline
        return pipe


class LoraWeight(BaseModel):
    path: str = Field(
        description="URL or the path to the LoRA weights.",
        examples=[
            "https://civitai.com/api/download/models/135931",
            "https://filebin.net/3chfqasxpqu21y8n/my-custom-lora-v1.safetensors",
        ],
    )
    scale: float = Field(
        default=1.0,
        description="""
            The scale of the LoRA weight. This is used to scale the LoRA weight
            before merging it with the base model.
        """,
        ge=0.0,
        le=1.0,
    )


@dataclass
class GlobalRuntime:
    MAX_CAPACITY: ClassVar[int] = 5

    models: dict[tuple[str, ...], Model] = field(default_factory=dict)

    def download_model_if_needed(self, model_name: str) -> str:
        CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)
        if model_name.startswith("https://") or model_name.startswith("http://"):
            return str(
                self.download_to(model_name, CHECKPOINTS_DIR, extension="safetensors")
            )
        return model_name

    def download_lora_weight_if_needed(self, lora_weight: str) -> str:
        LORA_WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
        if lora_weight.startswith("https://") or lora_weight.startswith("http://"):
            download_path = self.download_to(
                lora_weight, LORA_WEIGHTS_DIR, extension="safetensors"
            )
            return str(download_path.relative_to(LORA_WEIGHTS_DIR))
        return lora_weight

    def download_to(
        self,
        url: str,
        directory: Path,
        extension: str | None = None,
    ) -> Path:
        import os
        import shutil
        import tempfile
        from hashlib import md5
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        if extension is not None and url.endswith(".ckpt"):
            raise ValueError("Can't load non-safetensor model files.")

        url_file = CACHE_PREFIX + urlparse(url).path.split("/")[-1].strip(".")
        url_hash = md5(url.encode()).hexdigest()
        download_path = directory / f"{url_file}-{url_hash}"
        if extension:
            download_path = download_path.with_suffix("." + extension)

        if not download_path.exists():
            request = Request(url, headers={"User-Agent": TEMP_USER_AGENT})
            fd, tmp_file = tempfile.mkstemp()
            try:
                with urlopen(request) as response, open(fd, "wb") as f_stream:
                    total_size = int(response.headers.get("content-length", 0))
                    while data := response.read(CHUNK_SIZE):
                        f_stream.write(data)
                        if total_size:
                            progress_msg = f"Downloading {url}... {f_stream.tell() / total_size:.2%}"
                        else:
                            progress_msg = f"Downloading {url}... {f_stream.tell() / ONE_MB:.2f} MB"
                        print(progress_msg)
            except Exception:
                os.remove(tmp_file)
                raise

            # Only move when the download is complete.
            shutil.move(tmp_file, download_path)

        return download_path

    def load_lora_weight(self, lora_weight_path: str) -> dict[str, Any]:
        lora_weight = self.download_lora_weight_if_needed(lora_weight_path)

        try:
            if lora_weight.endswith(".bin"):
                # This only happens for the LoRas that were trained on our platform
                # since HF's trainer script exports the weights in regular torch format.
                import torch

                state_dict = torch.load(LORA_WEIGHTS_DIR / lora_weight)
            else:
                from safetensors import torch

                state_dict = torch.load_file(LORA_WEIGHTS_DIR / lora_weight)
        except Exception as exc:
            raise ValueError(
                "Could not process the lora weights due to a safetensor serialization error."
            ) from exc

        # To avoid false positives, we'll start collecting information about LoRAs first
        # and then convert this check into a proper warning.
        self.check_lora_compatibility(lora_weight, state_dict)
        return state_dict

    def merge_and_apply_loras(
        self,
        pipe: object,
        loras: list[LoraWeight],
    ) -> float:
        print(f"LoRAs: {loras}")
        state_dicts = [self.load_lora_weight(lora_weight.path) for lora_weight in loras]
        lora_scales = [lora_weight.scale for lora_weight in loras]

        if len(loras) == 1:
            [state_dict] = state_dicts
            [global_scale] = lora_scales
        else:
            state_dict = _merge_stacked_loras(state_dicts, lora_scales)
            global_scale = 1.0

        pipe.load_lora_weights(state_dict)
        pipe.fuse_lora()
        return global_scale

    def check_lora_compatibility(
        self, lora_name: str, state_dict: dict[str, Any]
    ) -> None:
        lora_formats = identify_lora_weights(state_dict)
        auxiliary_features = determine_auxiliary_features(lora_formats, state_dict)
        print(
            f"LoRA {lora_name}: "
            f"formats={lora_formats} "
            f"| auxiliary={auxiliary_features}"
        )

    def get_model(self, model_name: str, arch: str) -> Model:
        import torch
        from diffusers import (
            DiffusionPipeline,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
        )

        model_key = (model_name, arch)
        if model_key not in self.models:
            # Maybe in the future we can offload the model to the disk.
            if len(self.models) >= self.MAX_CAPACITY:
                by_last_hit = lambda kv: kv[1].last_cache_hit
                oldest_model_key, _ = min(self.models.items(), key=by_last_hit)
                print("Unloading model:", oldest_model_key)
                del self.models[oldest_model_key]

            if model_name.endswith(".ckpt") or model_name.endswith(".safetensors"):
                if arch is None:
                    if "xl" in model_name.lower():
                        arch = "sdxl"
                    else:
                        arch = "sd"

                    print(f"Guessing {arch} architecture for {model_name}")

                if arch == "sdxl":
                    pipeline_cls = StableDiffusionXLPipeline
                else:
                    pipeline_cls = StableDiffusionPipeline

                pipe = pipeline_cls.from_single_file(
                    model_name,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                )
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                )

            if hasattr(pipe, "watermark"):
                pipe.watermark = None

            pipe.enable_xformers_memory_efficient_attention()
            pipe = pipe.to("cuda")
            self.models[model_key] = Model(pipe)

        return self.models[model_key]

    @contextmanager
    def load_model(
        self,
        model_name: str,
        loras: list[LoraWeight],
        clip_skip: int = 0,
        scheduler: str | None = None,
        model_architecture: str | None = None,
    ) -> Iterator[tuple[object, float | None]]:
        model_name = self.download_model_if_needed(model_name)

        if model_architecture is None:
            if "xl" in model_name.lower():
                arch = "sdxl"
            else:
                arch = "sd"
            print(f"Guessing {arch} architecture for {model_name}")
        else:
            arch = model_architecture

        model = self.get_model(model_name, arch=arch)
        pipe = model.as_base()

        with self.change_scheduler(pipe, scheduler):
            try:
                if loras:
                    global_scale = self.merge_and_apply_loras(pipe, loras)
                else:
                    global_scale = None

                if clip_skip > 0:
                    original_layers = copy.copy(
                        pipe.text_encoder.text_model.encoder.layers
                    )
                    pipe.text_encoder.text_model.encoder.layers = original_layers[
                        :-clip_skip
                    ]

                yield (pipe, global_scale)
            finally:
                if clip_skip > 0:
                    pipe.text_encoder.text_model.encoder.layers = original_layers

                if loras:
                    try:
                        pipe.unfuse_lora()
                    except Exception:
                        print(
                            "Failed to unfuse LoRAs from the pipe, clearing it out of memory."
                        )
                        traceback.print_exc()
                        self.models.pop((model_name, arch), None)
                    else:
                        pipe.unload_lora_weights()

    @contextmanager
    def change_scheduler(
        self, pipe: object, scheduler_name: str | None = None
    ) -> Iterator[None]:
        import diffusers

        if scheduler_name is None:
            yield
            return

        scheduler_cls_name, scheduler_kwargs = SUPPORTED_SCHEDULERS[scheduler_name]
        scheduler_cls = getattr(diffusers, scheduler_cls_name)
        if scheduler_cls not in pipe.scheduler.compatibles:
            compatibles = ", ".join(cls.__name__ for cls in pipe.scheduler.compatibles)
            raise ValueError(
                f"The scheduler {scheduler_name} is not compatible with this model.\n"
                f"Compatible schedulers: {compatibles}"
            )

        original_scheduler = pipe.scheduler
        try:
            pipe.scheduler = scheduler_cls.from_config(
                pipe.scheduler.config,
                **scheduler_kwargs,
            )
            yield
        finally:
            pipe.scheduler = original_scheduler


@cached
def load_session():
    return GlobalRuntime()


class TextToImageInput(BaseModel):
    model_name: str = Field(
        description="URL or HuggingFace ID of the base model to generate the image.",
        examples=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "SG161222/Realistic_Vision_V2.0",
        ],
    )
    prompt: str = Field(
        description="The prompt to use for generating the image. Be as descriptive as possible for best results.",
        examples=[
            "Photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
            "Photo of a classic red mustang car parked in las vegas strip at night",
        ],
    )
    negative_prompt: str = Field(
        default="",
        description="""
            The negative prompt to use.Use it to address details that you don't want
            in the image. This could be colors, objects, scenery and even the small details
            (e.g. moustache, blurry, low resolution).
        """,
        examples=[
            "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
            "nsfw, cartoon, (epicnegative:0.9)",
        ],
    )
    loras: list[LoraWeight] = Field(
        default_factory=list,
        description="""
            The LoRAs to use for the image generation. You can use any number of LoRAs
            and they will be merged together to generate the final image.
        """,
    )
    seed: int | None = Field(
        default=None,
        description="""
            The same seed and the same prompt given to the same version of Stable Diffusion
            will output the same image every time.
        """,
    )
    image_size: ImageSizeInput | None = Field(
        default="square_hd",
        description="""
            The size of the generated image. You can choose between some presets or custom height and width
            that **must be multiples of 8**.
        """,
    )
    num_inference_steps: int = Field(
        default=30,
        description="""
            Increasing the amount of steps tells Stable Diffusion that it should take more steps
            to generate your final result which can increase the amount of detail in your image.
        """,
        ge=0,
        le=150,
        title="Number of inference steps",
    )
    guidance_scale: float = Field(
        default=7.5,
        description="""
            The CFG (Classifier Free Guidance) scale is a measure of how close you want
            the model to stick to your prompt when looking for a related image to show you.
        """,
        ge=0.0,
        le=20.0,
        title="Guidance scale (CFG)",
    )
    clip_skip: int = Field(
        default=0,
        description="""
            Skips part of the image generation process, leading to slightly different results.
            This means the image renders faster, too.
        """,
        ge=0,
        le=2,
    )
    model_architecture: Literal["sd", "sdxl"] | None = Field(
        default=None,
        description=(
            "The architecture of the model to use. If an HF model is used, it will be automatically detected. Otherwise will assume depending on "
            "the model name (whether XL is in the name or not)."
        ),
    )
    scheduler: Literal._getitem(Literal, *SUPPORTED_SCHEDULERS) | None = Field(  # type: ignore
        default=None,
        description="Scheduler / sampler to use for the image denoising process.",
    )
    image_format: Literal["jpeg", "png"] = Field(
        default="png",
        description="The format of the generated image.",
        examples=["jpeg"],
    )
    num_images: int = Field(
        default=1,
        description="""
            Number of images to generate in one request. Note that the higher the batch size,
            the longer it will take to generate the images.
        """,
        ge=1,
        le=8,
        title="Number of images",
    )


class TextToImageOutput(BaseModel):
    images: list[Image] = Field(description="The generated image files info.")
    seed: int = Field(
        description="""
            Seed of the generated Image. It will be the same value of the one passed in the
            input or the randomly generated that was used in case none was passed.
        """
    )


@contextmanager
def wrap_excs():
    from fastapi import HTTPException

    try:
        yield
    except Exception:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=422, detail=traceback.format_exc())


@function(
    "virtualenv",
    requirements=requirements,
    machine_type="GPU",
    keep_alive=4000,
    serve=True,
    max_concurrency=4,
    _scheduler="distributed",
)
def generate_image(input: TextToImageInput) -> TextToImageOutput:
    """
    TODO: write in markdown format 1-2 paragraphs about this function implementation
    that might be relevant to the user (i.e. what makes it fast, lora support / formats, etc)
    """
    import torch

    session = load_session()

    image_size = None
    if input.image_size is not None:
        image_size = get_image_size(input.image_size)

    with wrap_excs():
        with session.load_model(
            input.model_name,
            loras=input.loras,
            clip_skip=input.clip_skip,
            scheduler=input.scheduler,
            model_architecture=input.model_architecture,
        ) as (pipe, global_lora_scale):
            seed = input.seed or torch.seed()

            kwargs = {
                "num_inference_steps": input.num_inference_steps,
                "guidance_scale": input.guidance_scale,
                "generator": torch.manual_seed(seed),
            }

            if image_size is not None:
                kwargs["width"] = image_size.width
                kwargs["height"] = image_size.height

            if global_lora_scale is not None:
                kwargs["cross_attention_kwargs"] = {"scale": global_lora_scale}

            print(f"Generating {input.num_images} images...")
            result = pipe(
                prompt=input.prompt,
                negative_prompt=input.negative_prompt,
                num_images_per_prompt=input.num_images,
                **kwargs,
            )
            images = [Image.from_pil(image) for image in result.images]
            return TextToImageOutput(images=images, seed=seed)
