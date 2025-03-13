import os
import openai
import litellm
import pandas as pd
import time
import logging
from agentless.util.utils import load_jsonl
import json
import argparse
from datasets import load_dataset


# Configure logging
logging.basicConfig(
    filename='classification.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Define the classification prompt template
classification_prompt_template = """
Classify the given problem statement into one of the following categories:
- Bug Report: Something doesn't work as intended.
- Feature Request: A suggestion for a new capability.
- Security Vulnerability: Addressing a security risk.
- Performance Issue: Improving speed or efficiency.

Guidelines:
- Provide only the category name (e.g., Bug Report) as the response.
- Ensure the classification is based solely on the content of the problem statement.
- If the statement could fit multiple categories, choose the one that best represents the primary issue.

Problem Statement:
{problem_statement}

Category:
"""

# def preprocess_statement(statement):
#     if not statement or not isinstance(statement, str):
#         return ""
#     max_length = 2000
#     if len(statement) > max_length:
#         return statement[:max_length] + "..."
#     return statement.strip()

def classify_problem_statement(problem_statement, model="openai/gpt-4", temperature=0):
    prompt = classification_prompt_template.format(problem_statement=problem_statement)
    
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that categorizes problem statements."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=10,
            n=1,
            stop=["Category:"]
        )
        category = response.choices[0].message["content"].strip()
        return category
    except Exception as e:
        logging.error(f"Error classifying problem statement: {e}")
        return None

def classify_with_retry(problem_statement, model, retries=3, delay=1):
    for attempt in range(retries):
        try:
            category = classify_problem_statement(problem_statement, model=model)
            valid_categories = ["Bug Report", "Feature Request", "Security Vulnerability", "Performance Issue"]
            if category not in valid_categories:
                # Attempt to extract valid category
                for valid in valid_categories:
                    if valid.lower() in category.lower():
                        category = valid
                        return category
                else:
                    category = "Uncategorized"
                    logging.warning(f"Uncategorized problem statement: {problem_statement}")
            else:
                logging.info(f"Classified as {category}: {problem_statement}")
                return category
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            
    logging.error("All retry attempts failed.")
    return "Uncategorized"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="classify_outputs.jsonl")
    parser.add_argument("--eval_n_limit", type=int, default=1)
    parser.add_argument(
        "--model", type=str,
        default="openai/gpt-4o-2024-05-13",
        choices=["gpt-4", "gpt-4o", "gpt-35-turbo",
                 "azure/gpt-4o",
                 "openai/gpt-4o-2024-05-13",
                 "litellm_proxy/claude-3-5-sonnet-20241022",
                 ]
    )
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--loc_bench', action='store_true')

    args = parser.parse_args()
    
    if args.loc_bench:                
        loc_bench_data = []
        with open(args.dataset, 'r') as dtf:
            for line in dtf:
                instance = json.loads(line)
                loc_bench_data.append(instance)
                    
        swe_bench_tests = loc_bench_data
        if args.eval_n_limit:
            eval_n_limit = min(args.eval_n_limit, len(swe_bench_tests))
            selected_swe_bench_data = swe_bench_tests[:eval_n_limit]
    else:
        swe_bench_data = load_dataset(args.dataset, split=args.split)
        # swe_bench_tests = filter_dataset(swe_bench_data, 'instance_id', args.used_list)
        if args.eval_n_limit:
            eval_n_limit = min(args.eval_n_limit, len(swe_bench_data))
            selected_swe_bench_data = swe_bench_data.select(range(0, eval_n_limit))
            logging.info(f'Limiting evaluation to first {eval_n_limit} instances.')

    args.output_file = os.path.join(args.output_folder, args.output_file)
    os.makedirs(args.output_folder, exist_ok=True)
    
    processed_instances = []
    if os.path.exists(args.output_file):
        results = load_jsonl(args.output_file)
        processed_instances = [res['instance_id'] for res in results]
        
    for instance in selected_swe_bench_data:
        if instance['instance_id'] in processed_instances:
            continue
        
        problem_statement = instance['problem_statement']
    
        # Classify the problem statements
        category = classify_with_retry(problem_statement, model=args.model, retries=5)
        res = {
            "instance_id": instance['instance_id'],
            "category": category,
            "problem_statement": problem_statement
            
        }
        with open(args.output_file, 'a') as f:
            f.write(json.dumps(res) + '\n')
    
    df_issues = pd.read_json(args.output_file, lines=True)
    category_counts = df_issues['category'].value_counts()
    # Display the results
    print("\nClassification Results:")
    print(category_counts)
