"""
RID pipeline for iterative rule induction with LLaVA.

Logic overview:
1) Load train data and precomputed SSR neighbors for a dataset.
2) For each test item, select top-N related train samples from SSR output.
3) Iterate through those samples and repeatedly prompt LLaVA to update rules.
4) Run induction in both forward and reversed sample orders for robustness.
5) Save one JSONL line per test item with forward/backward induced rules.

Output format (one line per test sample):
{
  "index": <test_index>,
  "forward": "<rules induced in forward order>",
  "backward": "<rules induced in reverse order>"
}
"""

# There is no configurable GPU inference batch size in RID.py, and inference is always batch=1 (called one by one).

import os
import numpy as np
import torch
from termcolor import colored
import json
import random

# Import LLaVA related modules from utils
from utils.run_llava import run_proxy
from llava.mm_utils import get_model_name_from_path

# Import prompts
from utils.prompts import RID_prompt

# Import data utility functions
from utils.data_utils import get_item_data, DATASET_CONFIGS # DATASET_CONFIGS is imported for consistency, though not directly used in this file's logic

# --- LLaVA Model Initialization (Global, done once) ---
model_path = "liuhaotian/llava-v1.5-13b"
model_name = get_model_name_from_path(model_path)
print(f"Loading LLaVA model: {model_name} from {model_path}...")
llava_proxy = run_proxy(model_path, None)
print("LLaVA model loaded.")

# Arguments object for LLaVA proxy
args = type('Args', (), {
    "query": None,
    "conv_mode": None,
    "image_file": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 1024,
})()

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

def get_model_res(prompt: str, image_path: str):
    """
    Run one inference call through the shared LLaVA proxy.

    Parameters:
    - prompt: The textual instruction/question passed to the model.
    - image_path: Local path of the image paired with the prompt.

    Notes:
    - This helper mutates the global `args` object so the proxy receives
      the current image and query.
    - The proxy returns a tuple; we keep only the text response.
    """
    # Bind the current image to the shared argument container.
    args.image_file = image_path
    # Bind the current text prompt to the same container.
    args.query = prompt
    # Execute model inference; ignore the first return value and keep response text.
    _, response = llava_proxy.run_model(args)
    # Return raw model output for downstream parsing/usage.
    return response

def process_rid_generation(dataset_name: str, num_ssr_examples: int = 3):
    """
    Processes Rule Induction and Derivation (RID) for a given dataset.
    Uses a ssr approach to generate and refine rules based on similar memes.
        Run Rule Induction and Derivation (RID) for one dataset.

    Process:
    - read train data + SSR retrieval results
    - choose a fixed number of related train examples per test item
    - iteratively refine rules with LLaVA prompts
    - store rules from forward and reverse induction orders
    """
    print(f"\n--- Starting RID generation for dataset: {dataset_name} ---")

    base_data_path = f"data/{dataset_name}"
    image_base_path = f"{base_data_path}/images"
    train_jsonl_path = f"{base_data_path}/train.jsonl"
    ssr_result_path = f"SSR/{dataset_name}_SSR.jsonl"
    rid_result_path = f"RID/{dataset_name}_RID.jsonl"

    os.makedirs(os.path.dirname(rid_result_path), exist_ok=True)

    # Load train/reference items used as analogical examples.
    try:
        train_data = [json.loads(line) for line in open(train_jsonl_path, 'r').readlines()]
        print(f"Loaded {len(train_data)} train items for {dataset_name}.")
    except FileNotFoundError:
        print(colored(f"Error: Train data file not found at {train_jsonl_path}. Skipping {dataset_name}.", 'red'))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Could not decode JSON from {train_jsonl_path}. Skipping {dataset_name}.", 'red'))
        return

    # Load SSR results mapping each test item to similar train indices.
    try:
        ssr_lines = [json.loads(line) for line in open(ssr_result_path, 'r').readlines()]
        print(f"Loaded {len(ssr_lines)} ssr results from {ssr_result_path}.")
    except FileNotFoundError:
        print(colored(f"Error: ssr result file not found at {ssr_result_path}. Skipping {dataset_name}.", 'red'))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Could not decode JSON from {ssr_result_path}. Skipping {dataset_name}.", 'red'))
        return

    def get_analog_rules(indices_to_use: list):
        """Iteratively generates and refines rules based on a sequence of related memes."""
        """
        Induce rules by traversing selected train indices in order.

        The model receives current rules + current sample and returns "Updated rules:".
        """
        rules = "No rules yet."
        for idx in indices_to_use:
            if 0 <= idx < len(train_data):
                item = train_data[idx]
                # Use get_item_data for unified access
                image_file_name, text_content, _ = get_item_data(item, dataset_name)
                if not image_file_name or not text_content:
                    print(colored(f"Warning: Missing image file name or text content for train item {idx}. Skipping rule generation for this item.", 'yellow'))
                    continue

                image_file_path = os.path.join(image_base_path, image_file_name)

                # Inject current text and accumulated rules into the RID prompt template.
                input_prompt = RID_prompt.format(
                    org_sent=text_content,
                    rules=rules
                )

                response = get_model_res(input_prompt, image_file_path)

                print(colored(f"\n--- Input for ssr index {idx} ---", 'green'))
                print(colored(input_prompt, 'green'))
                print(colored(f"\n--- LLaVA Output for ssr index {idx} ---", 'blue'))
                print(colored(response, 'blue'))

                # Parse only the rule section; keep previous rules if the marker is absent.
                if "Updated rules:" in response:
                    rules = response.split("Updated rules:", 1)[-1].strip()
                else:
                    print(colored("Warning: 'Updated rules:' tag not found. Keeping previous rules.", 'yellow'))
            else:
                print(colored(f"Warning: ssr index {idx} out of bounds. Skipping.", 'yellow'))
        return rules

    # Continuation logic: resume from existing output length if a run was interrupted.
    start_idx = 0
    if os.path.exists(rid_result_path):
        with open(rid_result_path, 'r') as f_read:
            file_lines = f_read.readlines()
            start_idx = len(file_lines)
            if start_idx > 0:
                ssr_lines = ssr_lines[start_idx:]
                print(colored(f'Continuing RID generation from index: {start_idx}', 'cyan'))

    # Main loop: build forward/backward rules for each test sample.
    with open(rid_result_path, 'a') as f_write:
        for idx_offset, item in enumerate(ssr_lines):
            current_test_index = item['index']
            absolute_idx = idx_offset + start_idx


            # Use 'samples' key from previous script's output, fallback to  'example'
            # Prefer current SSR schema ('samples'); keep backward compatibility ('example').
            # Each `item` is one SSR record for a test sample, typically shaped like:
            # {"index": <test_id>, "samples": [train_id1, train_id2, ...], "scores": [...]}
            #
            # We need exactly `num_ssr_examples` retrieved train indices to build RID prompts.
            # 1) Prefer `samples` (current SSR schema used by SSR.py).
            # 2) Fallback to `example` (legacy/older schema compatibility).
            # 3) If neither key has enough entries, skip this test item safely.
            if 'samples' in item and len(item['samples']) >= num_ssr_examples:
                # Keep only the first N nearest neighbors (already sorted by retrieval score).
                examples = item['samples'][:num_ssr_examples]
            elif 'example' in item and len(item['example']) >= num_ssr_examples:
                # Legacy field name, handled for backward compatibility with old result files.
                examples = item['example'][:num_ssr_examples]
            else:
                # No usable SSR neighbors -> cannot continue RID induction for this sample.
                print(colored(f"Warning: Not enough ssr samples for index {absolute_idx}. Skipping.", 'yellow'))
                continue

            # Reversed order provides a second rule set less sensitive to sample ordering.
            reversed_examples = examples[::-1]

            print(f"\nProcessing test item {current_test_index} (ssr list index {absolute_idx})...")
            print(f"Using ssr samples (forward): {examples}")
            print(f"Using ssr samples (backward): {reversed_examples}")

            analog_rules = get_analog_rules(examples)
            re_analog_rules = get_analog_rules(reversed_examples)

            result = {
                'index': current_test_index,
                'forward': analog_rules,
                'backward': re_analog_rules
            }
            json.dump(result, f_write)
            f_write.write('\n')
            f_write.flush()

    print(f"\n--- Finished RID generation for dataset: {dataset_name} ---")

# --- Main execution for all datasets ---
if __name__ == "__main__":
    datasets_to_process = ["HarM", "FHM", "MAMI"] # 

    for dataset in datasets_to_process:
        process_rid_generation(dataset)

    print("\nAll datasets processed for RID generation.")
