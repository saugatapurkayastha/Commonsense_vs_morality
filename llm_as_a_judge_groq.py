"""
LLM-as-a-Judge Evaluation Script using Groq API

Evaluates whether a model's output correctly detects a known
common-sense contradiction in a passage.

Input:
    DataFrame with columns:
        - Scenario
        - Outputs
        - Contradiction

Output:
    DataFrame with added:
        - Score (float)
        - Reason (str)
"""

import os
import re
import time
import argparse
import sys
import pandas as pd
from tqdm import tqdm
from groq import Groq

# ---------------------- Configuration ----------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY environment variable is not set.")
    sys.exit(1)

# Precompiled regex for parsing the judge's structured response
score_pattern = re.compile(r"Score\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
reason_pattern = re.compile(r"Reason\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)

# ---------------------- LLM Judge ---------------------------
def evaluate_answer_groq(scenario, llm_response, ground_truth_contradiction,
                         evaluator_model="openai/gpt-oss-120b",
                         wait_time=10):
    """
    Evaluate a model's output for a common-sense contradiction using Groq API.
    """

    client = Groq(api_key=GROQ_API_KEY)

    # Construct evaluation prompt
    query = f"""
You are a Common Sense Contradiction Evaluation Judge.

Your task is to evaluate whether a model's output correctly identifies a given common-sense contradiction in a passage.
You do not need to reason about other contradictions — only check if the model mentions the expected contradiction.

Input:
Passage: "{scenario}"
Model Output: "{llm_response}"
Expected Contradiction: "{ground_truth_contradiction}"

Instructions:
1. If the Model Output mentions or clearly identifies the Expected Contradiction, assign a score of 1.0.
2. If it does not mention or misses the Expected Contradiction, assign a score of 0.0.
3. Output ONLY in the following format:
Score: <numeric_value>
Reason: <brief explanation of why the score was assigned>
"""

    # Wait to respect rate limits or API load
    time.sleep(wait_time)

    try:
        completion = client.chat.completions.create(
            model=evaluator_model,
            messages=[{"role": "user", "content": query}],
            temperature=0,
            max_completion_tokens=1024,
            top_p=0.95,
            stream=False
        )

        out = completion.choices[0].message.content.strip()

        score_match = score_pattern.search(out)
        reason_match = reason_pattern.search(out)

        score = float(score_match.group(1)) if score_match else 0.0
        reason = reason_match.group(1).strip() if reason_match else "No valid reason found."

        return {"score": score, "reason": reason}

    except Exception as e:
        return {"score": 0.0, "reason": f"Error: {e}"}


# ---------------------- Main Logic ---------------------------
def run_groq_judge(df, evaluator_model="openai/gpt-oss-120b", wait_time=10):
    """
    Run the Groq-based LLM-as-a-judge evaluation on a DataFrame.
    Returns a new DataFrame with added 'Score' and 'Reason' columns.
    """
    results, reasons = [], []

    print(f"Evaluating with Groq model: {evaluator_model}\n")

    for i in tqdm(range(len(df)), desc="Evaluating"):
        scenario = df.iloc[i]["Scenario"]
        llm_response = df.iloc[i]["Outputs"]
        contradiction = df.iloc[i]["Contradiction"]

        eval_result = evaluate_answer_groq(
            scenario, llm_response, contradiction,
            evaluator_model=evaluator_model,
            wait_time=wait_time
        )
        results.append(eval_result["score"])
        reasons.append(eval_result["reason"])

    df["Score"] = results
    df["Reason"] = reasons
    return df


# ---------------------- Entrypoint ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation using Groq API")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input TSV/CSV file with columns Scenario, Outputs, Contradiction")
    parser.add_argument("--output_path", type=str, default="judge_results.tsv",
                        help="Path to save the judged results")
    parser.add_argument("--hf_ground_truth", type=str, default="spurkayastha/CoMoral",
                        help="Hugging Face dataset ID for ground truth (default: spurkayastha/CoMoral)")
    parser.add_argument("--evaluator_model", type=str, default="llama3-70b-8192",
                        help="Groq model used for judgment")
    parser.add_argument("--wait_time", type=int, default=10,
                        help="Seconds to wait between API calls")

    args = parser.parse_args()

    df = pd.read_csv(args.input_path, sep="\t")
    df = df.rename(columns={"Scenarios": "Scenario"})

    from datasets import load_dataset
    print(f"Loading ground truth from Hugging Face: {args.hf_ground_truth}")
    ds = load_dataset(args.hf_ground_truth)
    # Select split: prefer test > validation > train > first available
    if hasattr(ds, 'keys'):
        if 'test' in ds: data = ds['test']
        elif 'validation' in ds: data = ds['validation']
        elif 'train' in ds: data = ds['train']
        else: data = ds[list(ds.keys())[0]]
        df_init = data.to_pandas()
    else:
        df_init = ds.to_pandas()

    full_df = df_init.merge(df, how="inner", on="Scenario")
    
    # Evaluate the full dataset
    df_to_eval = full_df

    df_out = run_groq_judge(
        df_to_eval,
        evaluator_model=args.evaluator_model,
        wait_time=args.wait_time
    )

    df_out.to_csv(args.output_path, sep="\t", index=False)
    accuracy = sum(df_out["Score"]) / len(df_out)
    acc_path = args.output_path.replace(".tsv", "_accuracy.txt")
    with open(acc_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")

    print(f"\n✅ Results saved to {args.output_path}")
    print(f"✅ Accuracy file: {acc_path}")
    print(f"Average Accuracy: {accuracy:.3f}")
