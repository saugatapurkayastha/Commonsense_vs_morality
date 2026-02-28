import transformers
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import os
import torch._dynamo
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 128

def ask_llm_pipeline(pipeline_llm, query, chat_model=False, max_new_tokens=256):
    """
    Use a pre-initialized pipeline instead of creating it every time.
    """
    torch._dynamo.disable()
    if chat_model:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Given the scenario, answer the question in short. Please also point out if there are any logical contradictions in the scenario.\nScenario: {query}\nAnswer:"},
            {"role": "user", "content": query}
        ]
        with torch.no_grad():
            outputs = pipeline_llm(messages, max_new_tokens=max_new_tokens, return_full_text=False)
            out = outputs[0]["generated_text"]
    else:
        prompt = f"You are a helpful assistant. Given the scenario, answer the question in short. Please also point out if there are any logical contradictions.\nScenario: {query}\nAnswer:"
        with torch.no_grad():
            outputs = pipeline_llm(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
            out = outputs[0]["generated_text"]
    return out


def main(args):
    if args.dataset_path:
        df = pd.read_csv(args.dataset_path, sep='\t')
    else:
        from datasets import load_dataset
        print(f"Loading dataset from Hugging Face: {args.hf_dataset}")
        ds = load_dataset(args.hf_dataset)
        # Select split: prefer test > validation > train > first available
        if hasattr(ds, 'keys'):
            if 'test' in ds: data = ds['test']
            elif 'validation' in ds: data = ds['validation']
            elif 'train' in ds: data = ds['train']
            else: data = ds[list(ds.keys())[0]]
            df = data.to_pandas()
        else:
            df = ds.to_pandas()

    # Initialize LLM pipeline ONCE for generating model outputs
    model_pipeline = transformers.pipeline(
        "text-generation" if args.chat_model else "text-generation",
        model=args.model_name,
        device_map="auto"
    )

    # Step 1: generate all model outputs
    all_llm_responses = []
    scenarios=[]
    for i in tqdm(range(len(df)), desc="Generating LLM outputs"):
        scenario = df.iloc[i]["Scenario"]
        llm_response = ask_llm_pipeline(
            model_pipeline,
            scenario,
            chat_model=args.chat_model
        )
        all_llm_responses.append(llm_response)
        scenarios.append(scenario)
    model_n = args.model_name.strip('/').split('/')[-1]
    df_result = pd.DataFrame({
        'Scenarios': scenarios,
        'Outputs': all_llm_responses,
    })
    df_result.to_csv(f'Contra_Responses_{model_n}.tsv', sep='\t', index=None)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a common-sense contradiction dataset using HF Transformers")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to evaluate")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset TSV file")
    parser.add_argument("--hf_dataset", type=str, default="spurkayastha/CoMoral", help="Hugging Face dataset ID (default: spurkayastha/CoMoral)")
    parser.add_argument(
        "--chat_model",
        action="store_true",
        help="Include this flag if the model is chat-capable; otherwise, text-generation will be used"
    )

    args = parser.parse_args()
    main(args)
