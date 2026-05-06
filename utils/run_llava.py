"""
LLaVA inference utility wrapper.

What this file does:
- loads a pretrained LLaVA model once via `run_proxy`
- normalizes image inputs (PIL / local path / URL)
- builds a conversation-style prompt with image tokens
- runs multimodal generation and returns `(prompt, response_text)`

Typical usage:
- instantiate `run_proxy(model_path, model_base)` one time
- call `run_model(args)` repeatedly with `args.query` + `args.image_file`
"""

import torch
import random

# Fix random seeds for more reproducible runs.
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re

def image_parser(args):
    """Normalize `args.image_file` into a list for batch-compatible handling."""
    if isinstance(args.image_file, list):
        return args.image_file
    return [args.image_file]


def load_image(image_file):
    """
    Load one image and convert to RGB.

    Supported inputs:
    - PIL Image object
    - HTTP/HTTPS URL
    - local file path
    """
    if isinstance(image_file, Image.Image):
        image = image_file.convert("RGB")
    elif image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    """Load multiple images with `load_image` while preserving order."""
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

class run_proxy():
    """Thin proxy around model/tokenizer/image processor for repeated inference."""
    def __init__(self, model_path, model_base) -> None:
        # Skip default parameter initialization to reduce startup overhead.
        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        # Load tokenizer, model, visual preprocessor, and context length together.
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name
        )

    def run_model(self, args, only_encode_images=False):
        """
        Execute one multimodal generation call.

        Expected `args` fields include:
        - query, image_file, conv_mode
        - temperature, top_p, num_beams, max_new_tokens
        """
        model_name = self.model_name
        model = self.model
        image_processor = self.image_processor
        tokenizer = self.tokenizer

        # Prepare query text with required image placeholder tokens.
        
        # Build the final text query `qs` expected by LLaVA.
        # LLaVA needs an explicit image token in the prompt so the model knows
        # where visual features should be fused with text.
        qs = args.query
        # Token form used by models that require explicit image start/end wrappers.
        # Equivalent structure: <im_start><image><im_end>
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # Case A: caller already provided IMAGE_PLACEHOLDER in the prompt template.
        if IMAGE_PLACEHOLDER in qs:
            # Replace placeholder with the token format required by current model config.
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        # Case B: no placeholder in user text -> prepend one image token line.
        else:
            # Prepending keeps raw user query unchanged while still injecting image context.
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # Auto-select conversation template by model naming convention.
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        # Build two-turn prompt: user query + empty assistant slot.
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Parse and preprocess image inputs for multimodal generation.
        image_files = image_parser(args)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        # Tokenize the text prompt with image token index inserted.
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # Decode with generation params from args.
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        # Return both the exact prompt used and final decoded model text.
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return prompt, outputs


