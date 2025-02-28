import json
import os
import warnings
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, BatchEncoding
)
from utils.custom_callbacks import SaveMetricsCallback
from utils.utils import (
    load_config, compute_metrics, compute_metrics_for_trainer, predict_disinformation, DisinformationDataset
)

# Suppress specific warnings related to tensor dimensions
warnings.filterwarnings(
    "ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars"
)


def load_and_process_data(file_path: str, label_column: str = "label", text_column: str = "content") -> pd.DataFrame:
    """
    Loads the data from a CSV file and processes the labels.
    Args:
        file_path (str): Path to the CSV file.
        label_column (str): The column name containing the labels.
        text_column (str): The column name containing the text content.
    Returns:
        pd.DataFrame: Processed dataframe with labels and text content.
    """
    data = pd.read_csv(file_path, encoding='utf-8')
    data[label_column] = data[label_column].apply(lambda x: 1 if "fake" in x.lower() else 0)
    return data


def tokenize_data(tokenizer, data: pd.Series, config: dict) -> BatchEncoding:
    """
    Tokenizes the text data.
    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        data (pd.Series): The text data to tokenize.
        config (dict): Configuration dictionary with tokenizer settings.
    Returns:
        dict: The tokenized data.
    """
    return tokenizer(
        data.tolist(),
        truncation=config["tokenizer"]["truncation"],
        padding=config["tokenizer"]["padding"],
        max_length=config["tokenizer"]["max_length"]
    )


def setup_trainer(config: dict, train_dataset, val_dataset) -> Trainer:
    """
    Configures the Trainer for model training.
    Args:
        config (dict): Configuration dictionary.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
    Returns:
        Trainer: Configured Trainer instance.
    """
    # Load model before passing it to Trainer
    model = AutoModelForSequenceClassification.from_pretrained(config["model"]["model_name"])

    training_args = TrainingArguments(
        output_dir=config["model"]["output"],
        evaluation_strategy=config["model"]["hyperparameters"]["evaluation_strategy"],
        learning_rate=config["model"]["hyperparameters"]["learning_rate"],
        per_device_train_batch_size=config["model"]["hyperparameters"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["model"]["hyperparameters"]["per_device_eval_batch_size"],
        num_train_epochs=config["model"]["hyperparameters"]["num_train_epochs"],
        warmup_steps=config["model"]["hyperparameters"]["warmup_steps"],
        weight_decay=config["model"]["hyperparameters"]["weight_decay"],
        fp16=config["model"]["hyperparameters"]["fp16"],
        metric_for_best_model=config["model"]["hyperparameters"]["metric_for_best_model"],
        load_best_model_at_end=config["model"]["hyperparameters"]["load_best_model_at_end"],
        save_total_limit=config["model"]["hyperparameters"]["save_total_limit"],
        greater_is_better=config["model"]["hyperparameters"]["greater_is_better"],
        save_strategy=config["model"]["hyperparameters"]["save_strategy"],
        eval_steps=config["model"]["hyperparameters"]["eval_steps"],
        save_on_each_node=True
    )

    return Trainer(
        model=model,  # Pass the actual model instance
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[SaveMetricsCallback(
            csv_file_name=f"{config['model']['valid_metrics']}.csv",
            hyperparameters=config["model"]["hyperparameters"]
        )]
    )


def save_metrics_to_json(metrics: dict, output_file_path: str):
    """
    Saves the metrics to a JSON file.
    Args:
        metrics (dict): The evaluation metrics.
        output_file_path (str): The file path to save the metrics.
    """
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as output_file:
        json.dump(metrics, output_file, indent=4)


def main():
    config = load_config()

    # Load and preprocess the datasets
    train_data = load_and_process_data(config["data"]["train"])
    validation_data = load_and_process_data(config["data"]["validation"])

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(config["model"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])

    # Tokenize datasets
    train_encodings = tokenize_data(tokenizer, train_data['content'], config)
    val_encodings = tokenize_data(tokenizer, validation_data['content'], config)

    # Create custom datasets
    train_dataset = DisinformationDataset(train_encodings, train_data['label'].tolist())
    val_dataset = DisinformationDataset(val_encodings, validation_data['label'].tolist())

    # Setup the Trainer
    trainer = setup_trainer(config, train_dataset, val_dataset)

    # Train the model
    trainer.train()

    # Save the trained model
    model_saved_path = config["model"]["path_to_save_model"]
    trainer.save_model(model_saved_path)

    # Load the test data and preprocess
    test_data = load_and_process_data(config["data"]["test"])

    # Make predictions on the test data
    model = AutoModelForSequenceClassification.from_pretrained(model_saved_path)
    test_data["predictions"] = test_data["content"].apply(
        lambda x: predict_disinformation(x, tokenizer, model)
    )

    # Compute evaluation metrics on the test data
    evaluation_results = compute_metrics(y_true=test_data["label"], y_pred=test_data["predictions"])

    # Save the evaluation metrics to a JSON file
    output_file_path = f"{config['model']['test_metrics']}.json"
    save_metrics_to_json(evaluation_results, output_file_path)


if __name__ == '__main__':
    main()
