"""
IAI (Iterative Analog Inference) evaluation pipeline.

File structure overview:
1) Global setup:
   - imports, random seeds, and one-time LLaVA proxy initialization.
2) Utility helpers:
   - `get_model_res`: sends one image+prompt request to LLaVA.
   - `find_answer_position`: maps textual answers to binary labels.
3) Main dataset routine:
   - `process_iai_evaluation(dataset_name)`:
     * load test data + RID rules
     * resume from partially completed output if present
     * run two "debater" inferences (forward/backward rules)
     * optionally run a "judge" when debaters disagree
     * update running metrics and stream per-item JSONL output
     * append final summary (accuracy, macro-F1)
4) Script entry:
   - iterate configured datasets and run evaluation.
"""

import os
import numpy as np
import torch
from termcolor import colored
import json
import random
from sklearn.metrics import f1_score # Import f1_score

# Import LLaVA related modules from utils
from utils.run_llava import run_proxy
from llava.mm_utils import get_model_name_from_path

# Import prompts
from utils.prompts import IAI_debater_prompt, IAI_judge_prompt

# Import data utility functions
from utils.data_utils import get_item_data, DATASET_CONFIGS

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

# --- LLaVA Model Initialization (Global, done once) ---
# ----------------------------
# 1) Global model initialization
# ----------------------------
# Load a single global LLaVA proxy so repeated inference calls are efficient.
model_path = "liuhaotian/llava-v1.5-13b"
model_name = get_model_name_from_path(model_path)
print(f"Loading LLaVA model: {model_name} from {model_path}...")
llava_proxy = run_proxy(model_path, None)
print("LLaVA model loaded.")

# Arguments object for LLaVA proxy
# Mutable argument container reused across model calls.
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

def get_model_res(prompt: str, image_path: str):
    """
    Run one LLaVA inference call and return raw text output.

    This helper mutates the shared global `args` object before dispatching.
    """
    args.image_file = image_path
    args.query = prompt
    _, response = llava_proxy.run_model(args)
    return response

def find_answer_position(answer: str):
    """Determines if the answer indicates 'harmful' or 'harmless'."""
    """
    Convert free-form model answer text into binary label.

    Returns:
    - 1 if text contains "harmful"
    - 0 if text contains "harmless"
    - None if no clear keyword is found
    """
    answer = answer.lower().strip()
    if "harmful" in answer:
        return 1
    elif "harmless" in answer:
        return 0
    return None

def process_iai_evaluation(dataset_name: str):
    """
        Performs Iterative Analog Inference (IAI) evaluation for a given 
    dataset.
    It uses rules derived from similar memes to assess target memes.
    
    Execute end-to-end IAI evaluation for one dataset.

    High-level flow:
    - load test samples and precomputed RID rule pairs
    - optionally resume from existing output JSONL
    - generate two debater predictions (forward/backward rules)
    - invoke judge only when two debaters disagree
    - persist per-item outputs and running metrics
    - append final summary metrics
    """
    print(f"\n--- Starting IAI evaluation for dataset: {dataset_name} ---")

    # ----------------------------
    # 2) Resolve paths
    # ----------------------------
    base_data_path = f"data/{dataset_name}"
    image_base_path = f"{base_data_path}/images"
    test_jsonl_path = f"{base_data_path}/test.jsonl"
    rid_result_path = f"RID/{dataset_name}_RID.jsonl"
    iai_result_path = f"IAI/{dataset_name}_IAI.jsonl"

    os.makedirs(os.path.dirname(iai_result_path), exist_ok=True)

    # ----------------------------
    # 3) Load required inputs
    # ----------------------------
    try:
        test_data = [json.loads(line) for line in open(test_jsonl_path, 'r').readlines()]
        print(f"Loaded {len(test_data)} test items for {dataset_name}.")
    except FileNotFoundError:
        print(colored(f"Error: Test data file not found at {test_jsonl_path}. Skipping {dataset_name}.", 'red'))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Could not decode JSON from {test_jsonl_path}. Skipping {dataset_name}.", 'red'))
        return

    try:
        rid_lines = [json.loads(line) for line in open(rid_result_path, 'r').readlines()]
        print(f"Loaded {len(rid_lines)} RID results from {rid_result_path}.")
    except FileNotFoundError:
        print(colored(f"Error: RID result file not found at {rid_result_path}. Skipping {dataset_name}.", 'red'))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Could not decode JSON from {rid_result_path}. Skipping {dataset_name}.", 'red'))
        return

    ratio = [0, 0] # [total_processed, correct_predictions]
    start_idx = 0

    # Lists to store actual and predicted labels for F1-score calculation
    # Accumulators for final macro-F1 calculation.
    all_actual_labels = []
    all_predicted_labels = []

    # Continuation logic for interrupted runs (matching the provided reference)
    # ----------------------------
    # 4) Resume support for interrupted runs
    # ----------------------------
    # If output exists, recover progress and continue from the next sample.
    if os.path.exists(iai_result_path):
        with open(iai_result_path, 'r') as f_read:
            file_lines = f_read.readlines()
            start_idx = len(file_lines)
            if start_idx > 0:
                try:
                    last_result = json.loads(file_lines[-1])
                    if 'ratio' in last_result and isinstance(last_result['ratio'], list) and len(last_result['ratio']) == 2:
                        ratio = last_result['ratio']
                        if 'accuracy' in last_result:
                            print(colored(f"Dataset {dataset_name} seems to be fully processed. Delete {iai_result_path} to re-run.", 'green'))
                            return
                    else:
                        print(colored("Warning: 'ratio' key not found or malformed in last result, resetting ratio.", 'yellow'))
                        ratio = [0,0]
                except json.JSONDecodeError:
                    print(colored("Warning: Could not decode last line for ratio, resetting ratio.", 'yellow'))
                    ratio = [0,0]

                # Populate all_actual_labels and all_predicted_labels from previous runs
                for i in range(start_idx):
                    try:
                        prev_item = json.loads(file_lines[i])
                        if 'actual' in prev_item and 'predict' in prev_item:
                            all_actual_labels.append(prev_item['actual'])
                            # Handle None predictions for previous runs as well
                            all_predicted_labels.append(prev_item['predict'] if prev_item['predict'] is not None else 1 - prev_item['actual'])
                    except json.JSONDecodeError:
                        continue # Skip malformed lines

                test_data = test_data[start_idx:]
                rid_lines = rid_lines[start_idx:]
                print(colored(f'Continuing IAI evaluation from index: {start_idx}', 'cyan'))
                print(f"Current ratio: {ratio}")

    # ----------------------------
    # 5) Main evaluation loop
    # ----------------------------
    with open(iai_result_path, 'a') as f_write:
        for idx_offset, (item, rid_line) in enumerate(zip(test_data, rid_lines)):
            current_absolute_idx = start_idx + idx_offset

            # Extract unified item fields across datasets.
            image_file_name, text_content, label = get_item_data(item, dataset_name)
            if image_file_name is None or text_content is None or label is None:
                print(colored(f"Warning: Missing essential data for test item {current_absolute_idx}. Skipping.", 'yellow'))
                continue

            # Defensive check: test item and RID rules should refer to same index.
            if item.get('index') is not None and rid_line.get('index') is not None and item['index'] != rid_line['index']:
                print(colored(f"Error: Mismatch in test_data index ({item['index']}) and RID result index ({rid_line['index']}) at absolute index {current_absolute_idx}. Skipping.", 'red'))
                continue

            image_file_path = os.path.join(image_base_path, image_file_name)
            forward_rules = rid_line['forward']
            backward_rules = rid_line['backward']

            is_equal = False

            # Debater 1 uses forward RID rule order.
            input_debater1 = IAI_debater_prompt.format(text_content, forward_rules)
            output_debater1 = get_model_res(input_debater1, image_file_path)
            print(colored("\n--- Debater 1 Output ---", 'green'))
            print(colored(output_debater1, 'green'))
            predict_1 = output_debater1.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            thought_1 = output_debater1.split("Thought: ")[-1]

            # Debater 2 uses backward RID rule order.
            input_debater2 = IAI_debater_prompt.format(text_content, backward_rules)
            output_debater2 = get_model_res(input_debater2, image_file_path)
            print(colored("\n--- Debater 2 Output ---", 'yellow'))
            print(colored(output_debater2, 'yellow'))
            predict_2 = output_debater2.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            thought_2 = output_debater2.split("Thought: ")[-1]

            # If debaters disagree, ask a judge model prompt to arbitrate.
            if predict_1 != predict_2:
                input_judge = IAI_judge_prompt.format(text_content, predict_1, thought_1[:1200], predict_2, thought_2[:1200])
                output_judge = get_model_res(input_judge, image_file_path)
                print(colored("\n--- Judge Input ---", 'red'))
                print(colored(input_judge, 'red'))
                print(colored("\n--- Judge Output ---", 'blue'))
                print(colored(output_judge, 'blue'))
                predict = output_judge.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            else:
                is_equal = True
                predict = predict_1

            final_predict_val = find_answer_position(predict)
            
            # If final_predict_val is None, set it to the opposite of the actual label
            # This is a common practice to handle cases where the model fails to output a clear classification
            if final_predict_val is None:
                final_predict_val = 1 - label

            # Update ratio
            # Update running accuracy counters.
            if final_predict_val == label:
                ratio[1] += 1
            ratio[0] += 1

            # Append to lists for F1-score calculation
            # Collect labels for dataset-level macro-F1.
            all_actual_labels.append(label)
            all_predicted_labels.append(final_predict_val)

            result = {
                'index': current_absolute_idx,
                'ratio': list(ratio),
                'actual': label,
                'predict': final_predict_val,
                'text': text_content,
                'output': output_debater1 if is_equal else output_judge
            }

            result['debater1_predict'] = find_answer_position(predict_1)
            result['debater2_predict'] = find_answer_position(predict_2)

            print(f"Actual: {label}, Predict: {final_predict_val}, ratio: {ratio}")

            json.dump(result, f_write)
            f_write.write('\n')
            f_write.flush()

        # After loop, dump final ratio, accuracy, and macro-F1 score
        # ----------------------------
        # 6) Final summary metrics
        # ----------------------------
        # Write a trailing summary object after all per-item outputs.
        accuracy = ratio[1] / ratio[0] if ratio[0] > 0 else 0
        
        # Calculate Macro F1 score
        macro_f1 = 0
        if len(all_actual_labels) > 0:
            # We assume labels are 0 and 1, as per 'harmless' (0) and 'harmful' (1)
            macro_f1 = f1_score(all_actual_labels, all_predicted_labels, average='macro')

        final_summary = {'ratio': ratio, 'accuracy': accuracy, 'macro_f1': macro_f1}
        json.dump(final_summary, f_write)
        f_write.write('\n')

    print(f"\n--- Finished IAI evaluation for dataset: {dataset_name} ---")
    print(f"Final Accuracy for {dataset_name}: {accuracy:.4f} ({ratio[1]}/{ratio[0]})")
    print(f"Final Macro F1 Score for {dataset_name}: {macro_f1:.4f}")


if __name__ == "__main__":
    datasets_to_process = ["FHM", "HarM", "MAMI"]

    for dataset in datasets_to_process:
        process_iai_evaluation(dataset)

    print("\nAll datasets processed for IAI evaluation.")