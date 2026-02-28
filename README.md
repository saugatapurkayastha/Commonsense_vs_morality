# Common Sense vs. Morality: The Curious Case of Narrative Focus Bias in Instruction-Tuned LLMs
[![License](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)]()


>**Abstract**
>
>Large Language Models (LLMs) are  increasingly deployed across diverse real-world applications and user communities. As such, it is crucial that these models remain both morally grounded and knowledge-aware. In this work, we uncover a critical limitation of current LLMs—their tendency to prioritize moral reasoning over commonsense understanding. To investigate this phenomenon, we introduce **CoMoral**, a novel benchmark dataset containing commonsense contradictions embedded within moral dilemmas. Through extensive evaluation of ten LLMs across different model sizes, we find that existing models consistently struggle to identify such contradictions without prior signal. Furthermore, we observe a pervasive *narrative focus* bias, wherein LLMs more readily detect commonsense contradictions when they are attributed to a secondary character rather than the primary (narrator) character. Our comprehensive analysis underscores the need for enhanced reasoning-aware training to improve the commonsense robustness of large language models.

This repository contains scripts to generate LLM responses for common-sense scenarios and evaluate them using an LLM-as-a-Judge approach (via Groq).

Contact Person: [Saugata Purkayastha](mailto:sapu00001@stud.uni-saarland.de)
Universitat des Saarlandes, Germany

## Setup

1. **Install dependencies:**
   ```bash
   pip install transformers torch pandas tqdm groq datasets
   ```

2. **Set your Groq API key:**
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   ```

## Usage

### 1. Generate Responses

Use `commonsense.py` (implicit prompt) or `commonsense_with_contra.py` (explicit prompt) to generate model outputs.

**Arguments:**
- `--model_name`: Hugging Face model path (e.g., `meta-llama/Llama-3.2-1B-Instruct`).
- `--hf_dataset`: Hugging Face dataset ID (default: `spurkayastha/CoMoral`).
- `--dataset_path`: Path to local input dataset (optional, overrides `--hf_dataset`).
- `--chat_model`: Flag to indicate if the model uses a chat template.

**Examples:**

Using default Hugging Face dataset:
```bash
python commonsense.py --model_name meta-llama/Llama-3.2-1B-Instruct --chat_model
```
This produces a file named `Responses_<model_name>.tsv`.

### 2. Evaluate Responses

Use `llm_as_a_judge_groq.py` to grade the generated responses against the ground truth.

**Arguments:**
- `--input_path`: Path to the generated responses file (from step 1).
- `--hf_ground_truth`: Hugging Face dataset ID for ground truth (default: `spurkayastha/CoMoral`).
- `--evaluator_model`: Groq model to use as judge (default: `llama3-70b-8192`).

**Example:**
```bash
python llm_as_a_judge_groq.py --input_path Responses_Llama-3.2-1B-Instruct.tsv
```
### Citation
```bib
```