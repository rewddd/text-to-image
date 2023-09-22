from registry.text_to_image import LoraWeight, TextToImageInput, generate_image

from .helpers import BaseModelTest, ModelTestParamType


class TestTextToImage(BaseModelTest):
    model_name = "text_to_image"

    def generate(self, input: ModelTestParamType):
        loras = input.get("loras", [])

        input["loras"] = []
        for lora in loras:
            input["loras"].append(LoraWeight(**lora))

        model_input = TextToImageInput(**input)

        local = generate_image.on(
            serve=False,
            keep_alive=30,
            machine_type="GPU",
            _scheduler=None,
        )

        return local(model_input)
