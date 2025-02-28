import pandas as pd
import logging
from datasets import load_dataset
from dotenv import load_dotenv
from utils.utils import (parallel_text_processing, extract_numbers,
                         calculate_accuracy, load_prompts)
from utils.prompt_template import PromptTemplate
import os

# Configure logging
if not os.path.exists("logging"):
    os.makedirs("logging")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logging/prompting.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Suppress logging from external libraries (like HTTP requests)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def process_dataset(method_type, dataset, model, result_file):
    """Processes the dataset with the given method type and returns accuracy."""
    logger.info(f"Processing dataset with method: {method_type}")
    system_prompt, user_prompt = load_prompts(
        prompts_file_path="prompts/",
        method_type=method_type
    )

    prompt_template = PromptTemplate(system_prompt, user_prompt)

    df = parallel_text_processing(
        dataframe=dataset.copy(),
        col_with_content="questions",
        column="pred_solution",
        filename=result_file,
        model=model,
        prompt_template=prompt_template
    )

    df = extract_numbers(dataset=df, column_name='pred_solution', new_column_name='extracted_number')
    accuracy = calculate_accuracy(df, "answer", "extracted_number")
    logger.info(f"{method_type} Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    logger.info("Loading environment variables")
    load_dotenv()
    dataset = load_dataset("gsm8k", "main")
    dataset_length = len(dataset['test'])
    logger.info(f"Dataset length: {dataset_length}")

    # Prepare dataset
    test_samples = dataset["test"][:100]
    questions = test_samples["question"]
    long_answers = test_samples["answer"]

    answers = [float(ans.split("#### ")[-1]) for ans in long_answers]

    df = pd.DataFrame({"questions": questions, "long_answers": long_answers, "answer": answers})
    logger.info(f"Dataset length for our experiment: {len(df)}")

    model = "gpt-4o-mini"
    methods = [
        ("zero-shot", "result/math_problem_zero_shot.csv"),
        ("few-shot", "result/math_problem_few_shot.csv"),
        ("zero-shot-CoT", "result/math_problem_zero_shot_cot.csv")
    ]

    for method, file in methods:
        process_dataset(method, df, model, file)

    logger.info("Processing completed")


if __name__ == "__main__":
    main()
