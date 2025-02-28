import pandas as pd
import logging
from sklearn.metrics import f1_score
from dotenv import load_dotenv
from utils.utils import (parallel_text_processing, load_prompts, PromptTemplate)
import os

# Configure logging
if not os.path.exists("logging"):
    os.makedirs("logging")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logging/prompting-classification.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Suppress logging from external libraries (like HTTP requests)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def process_dataset(method_type, dataset, model, result_file):
    """Processes the dataset with the given method type and returns accuracy."""
    logger.info(f"Processing dataset with method: {method_type}")
    logger.info(f"Processing dataset with model: {model}")
    system_prompt, user_prompt = load_prompts(
        prompts_file_path="prompts/",
        method_type=method_type
    )

    prompt_template = PromptTemplate(system_prompt, user_prompt)

    df = parallel_text_processing(
        dataframe=dataset.copy(),
        col_with_content="content",
        column="pred_solution",
        filename=result_file,
        model=model,
        prompt_template=prompt_template
    )

    y_pred = df.pred_solution.apply(lambda x: 1 if "fake" in x.lower() else 0)
    y_true = df.label.apply(lambda x: 1 if "fake" in x.lower() else 0)
    f1_micro = f1_score(y_true, y_pred, average="micro")
    logger.info(f"{method_type}, {model}, F1 Micro Score: {f1_micro:.4f}")


def main():
    logger.info("Loading environment variables")
    load_dotenv()
    df = pd.read_csv("../../data/ECTF/test.csv")
    logger.info(f"Dataset length for our experiment: {df.shape[0]}")

    methods = [
        ("disinformation-zero-shot", "gpt-4o-mini", "result/dis_classification_zero_shot_gpt4o_mini.csv"),
        ("disinformation-zero-shot", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
         "result/dis_classification_zero_shot_llama.csv"),
        ("disinformation-zero-shot", "gpt-3.5-turbo", "result/dis_classification_zero_shot_gpt3_5.csv"),
    ]

    for method, model, file in methods:
        process_dataset(method, df, model, file)

    logger.info("Processing completed")


if __name__ == "__main__":
    main()
