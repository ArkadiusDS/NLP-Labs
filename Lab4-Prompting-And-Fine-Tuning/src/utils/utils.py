import os
import time

import numpy as np
import torch
import json
import yaml
import pandas as pd
import concurrent.futures
from transformers import EvalPrediction
from sklearn.metrics import f1_score
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")


class DisinformationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_config(file_path='config.yaml'):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def predict_disinformation(text, tokenizer, model):
    """
    Function that predicts the label for input text using argmax
    """

    tokenized_text = tokenizer([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokenized_text)

    logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    predicted_label = np.argmax(probabilities)
    return predicted_label


def load_prompts(prompts_file_path, method_type):
    """

    """
    try:
        with open(prompts_file_path + method_type + ".yaml", "r") as file:
            prompt = yaml.safe_load(file)
            system_prompt = prompt['system']
            user_prompt = prompt['user']
            return system_prompt, user_prompt
    except:
        raise Exception(
            f"Prompting method '{method_type}' not found in the YAML files in {prompts_file_path} directory."
        )


def client_instance(model):
    if "llama" in model.lower():
        return OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
    else:
        return OpenAI(api_key=API_KEY)


def process_text_with_model(index, text, model, prompt_template):
    try:
        user_prompt = prompt_template.format_user_prompt(text)
        client = client_instance(model)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_template.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )
        return {
            "index": index,
            "system_prompt": prompt_template.system_prompt,
            "user_prompt": user_prompt,
            "completion": completion.choices[0].message.content,
        }
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(2)  # Avoid rapid retries
        return {"index": index, "system_prompt": None, "user_prompt": None, "completion": None}


def parallel_text_processing(dataframe, col_with_content, column, filename, model, prompt_template):
    dataframe["system_prompt"] = None
    dataframe["user_prompt"] = None
    dataframe[column] = None
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_text_with_model, index, text, model, prompt_template)
            for index, text in enumerate(dataframe[col_with_content])
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Thread failed with error: {e}")

    for result in results:
        dataframe.at[result["index"], "system_prompt"] = result["system_prompt"]
        dataframe.at[result["index"], "user_prompt"] = result["user_prompt"]
        dataframe.at[result["index"], column] = result["completion"]

    dataframe.to_csv(filename, index=False)
    return dataframe


def extract_numbers(dataset, column_name, new_column_name):
    """
    Extracts numbers from a JSON string in the specified column and creates a new column with those numbers.

    Parameters:
        dataset (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing JSON strings.
        new_column_name (str): The name of the new column to store extracted numbers.

    Returns:
        pd.DataFrame: The DataFrame with the new column added.
    """
    dataset = dataset.copy()

    def safe_extract(json_str):
        if pd.notna(json_str):
            try:
                # Ensure proper escaping
                json_str = json_str.replace("\\", "\\\\")
                return float(json.loads(json_str)['answer'])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing JSON: {e} | Data: {json_str}")
                return None
        return None

    dataset[new_column_name] = dataset[column_name].apply(safe_extract)

    return dataset


def calculate_accuracy(dataset, true_col, pred_col):
    agreement = dataset[dataset[true_col] == dataset[pred_col]].shape[0]
    accuracy = agreement / dataset.shape[0]

    return accuracy


def compute_metrics(pred=None, y_true=None, y_pred=None):
    """
    Computes F1 scores (micro, macro, weighted) for both training and testing data.

    If `pred` is provided, it computes metrics for the trainer using `EvalPrediction`.
    If `y_true` and `y_pred` are provided, it computes metrics for test data predictions.

    Parameters:
        - pred (EvalPrediction, optional): The evaluation prediction object for Trainer.
        - y_true (list, optional): The ground truth labels for the test data.
        - y_pred (list, optional): The predicted labels for the test data.

    Returns:
        - dict: A dictionary containing F1 metrics.
    """
    if pred is not None:
        # When working with the Trainer, pred is an EvalPrediction object
        labels = pred.label_ids
        y_pred = pred.predictions.argmax(-1)
    elif y_true is not None and y_pred is not None:
        # If y_true and y_pred are provided, use them for test evaluation
        labels = y_true
    else:
        raise ValueError("Either `pred` or both `y_true` and `y_pred` must be provided.")

    # Compute F1 scores
    f1 = f1_score(y_true=labels, y_pred=y_pred)
    f1_micro = f1_score(y_true=labels, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=labels, y_pred=y_pred, average='weighted')

    return {
        'f1': f1,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_macro_weighted': f1_macro_weighted
    }


def compute_metrics_for_trainer(pred: EvalPrediction):
    return compute_metrics(pred=pred)
