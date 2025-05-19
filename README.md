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

Students are required to fine-tune BERT on the **CoAID dataset** as part of the **Assignment.ipynb** notebook.


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
