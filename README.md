# Natural Language Processing Labs

This project was developed for the NLP course at the University of Padova. It includes work on prompting with LLMs, fine-tuning BERT, and Retrieval-Augmented Generation (RAG) methods.


### Author
Prepared by **Arkadiusz Modzelewski**.

Website with contact details: [Arkadiusz Modzelewski](https://amodzelewski.com/)

## Project Structure

```markdown
- NLP-Labs/
    - LICENSE
    - uv.lock
    - pyproject.toml
    - README.md 
    - Prompting/
      - Prompting-Exercise-with-HuggingFace.ipynb
      - Prompting-with-LLMs.ipynb
      - result/
        - math_problem_cot.csv
        - math_problem_few_shot.csv
        - math_problem_zero_shot.csv
    - Lab-Fine-Tuning/
        - Assignment/
            - Assignment.ipynb
            - resulting_file/
                - experiments_Arkadiusz_Modzelewski_29580.json
                - experiments_name_surname_student_id.json
            - test/
                - test_structure_of_json_file.py
        - Fine-Tuning-BERT.ipynb 
    - data/
        - CoAID/
            - validation.csv
            - test.csv
            - train.csv
        - ECTF/
            - validation.csv
            - test.csv
            - train.csv
```

## NLP Labs 

In this repository you will find labs on three topics:
- Prompting with closed LLMs and open-weight LLMs
- Fine-Tuning with BERT model
- Retrieval Augmented Generation (in preparation)

## Description

### Prompting with LLMs
In this lab, we explore prompting techniques using **GPT-4o-mini** model from OpenAI (using their API) and chosen smaller open-weight LLM from Microsoft . The prompting approach is applied to the **GSM8K** dataset for math problems.

We experiment with:
- Zero-shot prompting
- Few-shot prompting
- Chain of Thought (CoT) with zer-shot prompting


### Fine-Tuning BERT for Fake News Detection
The next lab demonstrates fine-tuning the **BERT base model** for fake news detection using:
- **ECTF Twitter Dataset**
- **CoAID News Dataset**

#### Assignment
Students are required to fine-tune BERT on the **CoAID dataset** as part of the **Assignment.ipynb** notebook.

As a result of assignemnt students need to prepare a notebook and resulting json file.

Template for resulting json file provided in `Lab-Fine-Tuning/Assignment/resulting_file/experiments_name_surname_student_id.json`.

Example resulting json file provided in `Lab-Fine-Tuning/Assignment/resulting_file/experiments_Arkadiusz_Modzelewski_29580.json`.
In this example, I used zeros as values for both evaluation metrics and hyperparameters. This was done intentionally to avoid influencing you toward any specific results or parameter choices.
Before submitting the json file, please pass it through tests that are included in `Lab-Fine-Tuning/Assignment/test/test_structure_of_json_file.py`.

Sure! Here's a **simple, step-by-step guide** you can give to your students so they can successfully **run the test** from your GitHub repo.

---

##### ✅ How to Run the Test Script from GitHub

###### Prerequisites:

* Make sure Python is installed on your computer. You can check by running:

  ```bash
  python --version
  ```
---

###### Step-by-Step Instructions

1. ###### **Open Terminal or Command Prompt**

   * On Windows: Open **Command Prompt** or **PowerShell**.
   * On Mac/Linux: Open the **Terminal**.

2. ###### **Clone the GitHub Repository**

   Type the following command and press **Enter**:

   ```bash
   git clone https://github.com/ArkadiusDS/NLP-Labs.git
   ```

3. ###### **Navigate to the Test Directory**

   After cloning, change into the test directory:

   ```bash
   cd Lab-Fine-Tuning/Assignment/test
   ```

4. ###### **Make Sure Your JSON File is Ready**

   Make sure your result file (e.g. `your_resulting_json_file.json`) is in the same folder or note its full path.

5. ###### **Run the Test**

   Run the test using this command:

   ```bash
   python test_structure_of_json_file.py your_resulting_json_file.json
   ```

---

###### ❗Common Errors to Avoid

* **File not found?** Double check the path to your `.json` file.
* **Python not recognized?** You may need to use `python3` instead of `python` depending on your setup.

---

### Explanation of each directory
- `data/`: Datasets for fake news detection.
- `Prompting/`: Contains notebooks for prompting and directory with results from running Prompting-with-LLMs.ipynb notebook.
- `Lab-Fine-Tuning/`: Contains notebooks and files for fine-tuning BERT lab.


## Setup
To set up the environment, install **uv** with one of the following methods:

### Standalone Installer (Recommended)
#### macOS and Linux
Use curl to download and install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
If your system doesn't have curl, use wget:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
Request a specific version:
```bash
curl -LsSf https://astral.sh/uv/0.6.3/install.sh | sh
```

### PyPI Installation
Install uv into an isolated environment with pipx:
```bash
pipx install uv
```
Alternatively, use pip:
```bash
pip install uv
```

For more installation methods, visit [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

Once installed, sync dependencies with:
```bash
uv sync
```

## Usage
To install new libraries, use:
```bash
uv add <library_name>
```

Run the script:
```bash
uv run <script.py>
```

## License
This project is licensed under the terms of the **MIT License**.
