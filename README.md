# Natural Language Processing Labs

This project contains **Lab 4**  and **Lab 5** prepared for the **NLP** class at the **University of Padova**. 

### Author
Prepared by **Arkadiusz Modzelewski**.

If you have any questions, please send them to this email address: arkadiusz.modzelewski@pja.edu.pl

## Project Structure

```markdown
- NLP-Labs/
    - LICENSE
    - uv.lock
    - pyproject.toml
    - README.md
    - Lab4-Prompting-And-Fine-Tuning/
        - Assignments.ipynb
        - Prompting-And-Fine-Tuning.ipynb
        - src/
            - fine_tuning.py
            - prompting.py
            - config.yaml
            - prompting-classification.py
            - utils/
                - utils.py
                - custom_callbacks.py
                - prompt_template.py
            - prompts/
                - zero-shot-CoT.yaml
                - disinformation-zero-shot.yaml
                - zero-shot.yaml
                - few-shot.yaml
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

## NLP Labs - Lab 4: Prompting and Fine-Tuning

The lab covers basic topics in **Prompting with LLMs** and **Fine-Tuning BERT models**.

## Description

### Prompting with LLMs
In this lab, we explore prompting techniques using **GPT-4o-mini** and **LLaMA 3.1** models. The prompting approach is applied to the **GSM8K** dataset for math problems.

We experiment with:
- Zero-shot prompting
- Zero-shot with Chain of Thought (CoT) prompting
- Few-shot prompting

Prompt templates are stored in the `Lab4-Prompting-And-Fine-Tuning/src/prompts/` directory.

### Fine-Tuning BERT for Fake News Detection
The second part of the lab demonstrates fine-tuning the **BERT base model** for fake news detection using:
- **ECTF Twitter Dataset**
- **CoAID News Dataset**

Students are required to fine-tune BERT on the **CoAID dataset** as part of the **Assignments.ipynb** notebook.

### Folder Explanation
- `Lab4-Prompting-And-Fine-Tuning/src/`: Contains Python scripts for prompting and fine-tuning.
- `Lab4-Prompting-And-Fine-Tuning/src/utils/`: Utility functions and classes.
- `Lab4-Prompting-And-Fine-Tuning/src/prompts/`: YAML files for different prompting strategies.
- `data/`: Datasets for fake news detection.


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
uv run script.py
```

## License
This project is licensed under the terms of the **MIT License**.
