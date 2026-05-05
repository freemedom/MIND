# --- Dataset Configuration ---
# Define key mappings for image filename, text content, and label for each dataset
DATASET_CONFIGS = {
    "FHM": {
        "image_key": "img",
        "text_key": "text",
        "label_key": "label",
        "label_mapping": None
    },
    "HarM": {
        "image_key": "image",
        "text_key": "text",
        "label_key": "labels",
        "label_mapping": {"not harmful": 0, "default_harmful": 1} # "default_harmful" is a placeholder for any other case
    },
    "MAMI": {
        "image_key": "image",
        "text_key": "text",
        "label_key": "label",
        "label_mapping": None
    }
}

def get_item_data(item: dict, dataset_name: str):
    """
    Retrieves image filename, text content, and label from a data item
    based on the dataset's configuration. Handles label mapping.

    Args:
        item (dict): A dictionary representing a single data entry from a JSONL file.
        dataset_name (str): The name of the dataset (e.g., "FHM", "HarM", "MAMI").

    Returns:
        tuple: (image_filename: str, text_content: str, label: int or None)
               Returns None for label if not found or not applicable.
    """
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        raise ValueError(f"Configuration not found for dataset: {dataset_name}")

    image_filename = item.get(config["image_key"])
    text_content = item.get(config["text_key"])
    raw_label = item.get(config["label_key"])

    processed_label = None
    if raw_label is not None:
        if dataset_name == "HarM":
            if isinstance(raw_label, list) and raw_label:
                # If the first label is "not harmful", map to 0
                if raw_label[0].lower() == "not harmful":
                    processed_label = 0
                else:
                    # Otherwise, for any other label (e.g., "very harmful", "somewhat harmful"), map to 1
                    processed_label = 1
            else:
                # If it's not a list or an empty list for HarM, default to 1 (harmful)
                processed_label = 1
        elif config["label_mapping"]: # For other datasets with explicit mappings (if any)
            if isinstance(raw_label, list) and raw_label:
                processed_label = config["label_mapping"].get(raw_label[0].lower())
            elif isinstance(raw_label, str):
                processed_label = config["label_mapping"].get(raw_label.lower())
            else:
                processed_label = raw_label
        else: # Default case: no specific mapping, assume label is already 0/1
            processed_label = raw_label

    return image_filename, text_content, processed_label
