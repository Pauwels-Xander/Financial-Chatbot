from typing import List, Dict, Callable
import sys
import os
# get PipelineOrchestrator
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from backend.orchestrator import PipelineOrchestrator
import time
import json
import csv
import re

def evaluate_pipeline(querry:List[Dict]) -> List[Dict]:
    results = []
    pipeline = PipelineOrchestrator(database_path="data/db/trial_balance.duckdb")
    for q in querry:
        true_numeric = q['answer'] 

        #measure latency
        start_time = time.time()
        output = pipeline.process_query(q['question']) #get pipeline response
        latency = time.time() - start_time

        answer_text = output.answer # text response from pipeline

        #extract the number from the answer
        predicted_numeric = extract_number(answer_text)
        
        #try to turn number from test into a float
        try:
            true_numeric_val = float(true_numeric)
        except (ValueError, TypeError):
            true_numeric_val = None  # or handle error

        if predicted_numeric is not None and true_numeric_val is not None:
            absolute_numeric = predicted_numeric == true_numeric
            if(absolute_numeric):
                relative_error = 0
            else:

                try:
                    relative_error = abs(predicted_numeric - true_numeric_val) / abs(true_numeric_val)
                except Exception as e: 
                    print(e)

                    relative_error = None
            
        else:
            relative_error = None  # or some default/error value
            absolute_numeric = None

        validation_sql = output.validation_status

        # Store results per query
        results.append({
            "question": q['question'],
            "answer": answer_text,
            "ground_truth": true_numeric,
            "absolute_numeric": absolute_numeric,
            "relative_error": relative_error,
            "correct_sql": q['sql'],
            "generated_sql": output.generated_sql,
            "sql_validity": validation_sql,
            "latency_sec": latency
        })
    return results

def extract_number(text: str):
    """
    Extracts a monetary or numeric amount from text.
    Handles suffixes like K, M, B and ignores years (1900–2100).
    Example: 'For 2015, amount was $10.07M.' → 10070000
    """

    # Patterns like 10.07M, 5.3B, 120k, $3.5M, 15,200.42
    pattern = r"\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+(?:\.\d+)?))\s*([KkMmBb]?)"


    matches = re.findall(pattern, text)

    if not matches:
        return None

    best_value = None

    for num_str, suffix in matches:
        # Remove commas → "10,200.5" → "10200.5"
        clean = num_str.replace(",", "")

        try:
            value = float(clean)
        except ValueError:
            continue

        # Skip years like 2015, 2022, 1999 etc.
        if 1900 <= value <= 2100:
            continue

        # Apply suffix multiplier
        if suffix.lower() == "k":
            value *= 1_000
        elif suffix.lower() == "m":
            value *= 1_000_000
        elif suffix.lower() == "b":
            value *= 1_000_000_000

        # Keep the *largest* meaningful number (usually the amount)
        if best_value is None or value > best_value:
            best_value = value

    return best_value

def run_evaluation(json_file_path: str):
    # Load queries from JSON
    with open(json_file_path, "r") as f:
        qa_set: List[Dict] = json.load(f)

    # Evaluate all queries
    results = evaluate_pipeline(qa_set)

    # save results
    save_results_ordered_pretty(results, csv_path="tests/evaluation_results.csv")

    # Print results
    #for r in results:
    #    pretty_print_result(r)

def pretty_print_result(r):
    print("\n" + "="*80)
    print(f"QUESTION:        {r['question']}")
    print(f"ANSWER:           {r['answer']}")
    print(f"Ground Truth      {r['ground_truth']}")
    print(f"ABS MATCH:        {r['absolute_numeric']}")
    print(f"REL ERROR:        {r['relative_error']}")
    print(f"SQL VALIDITY:     {r['sql_validity']}")
    print(f"LATENCY:          {r['latency_sec']:.4f} sec")
    print("-"*80)
    print("CORRECT SQL:")
    print(r['correct_sql'])
    print("-"*80)
    print("GENERATED SQL:")
    print(r['generated_sql'])
    print("="*80 + "\n")

def save_results_ordered_pretty(results: List[Dict], csv_path: str = None):
    with open(csv_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write("\n" + "="*80 + "\n")
            f.write(f"QUESTION:        {r['question']}\n")
            f.write(f"ANSWER:          {r['answer']}\n")
            f.write(f"Ground truth:    {r['ground_truth']}\n")
            f.write(f"ABS MATCH:       {r['absolute_numeric']}\n")
            f.write(f"REL ERROR:       {r['relative_error']}\n")
            f.write(f"SQL VALIDITY:    {r['sql_validity']}\n")
            f.write(f"LATENCY:         {r['latency_sec']:.4f} sec\n")
            f.write("-"*80 + "\n")
            f.write("CORRECT SQL:\n")
            f.write(r['correct_sql'].replace("\n", " ") + "\n")
            f.write("-"*80 + "\n")
            f.write("GENERATED SQL:\n")
            f.write(r['generated_sql'].replace("\n", " ") + "\n")
            f.write("="*80 + "\n")
    print(f"Results saved in ordered, readable format: {csv_path}")


def save_results(results: List[Dict], csv_path: str = None):
    # Define header for both CSV and Markdown
    headers = [
        "question",
        "answer",
        "absolute_numeric",
        "relative_error",
        "correct_sql",
        "generated_sql",
        "sql_validity",
        "latency_sec"
    ]

    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            row = {}
            for h in headers:
                val = r.get(h, "")
                if val is None:
                    val = ""
                elif isinstance(val, float):
                    val = f"{val:.6f}"  # uniform 6 decimal places
                elif isinstance(val, str):
                    val = val.replace("\n", " ").replace("\r", " ").strip()  # clean newlines/spaces
                row[h] = val
            writer.writerow(row)
    print(f"Results saved to CSV: {csv_path}")

    


run_evaluation("tests/qa_set.json")
#text = "For 2018, amount was $-10.11M."
#print(extract_number(text))    
    


