"""
LTR Scheduler with Starvation Prevention
Based on ltr_from_fcfs_separate.py, adds starvation prevention mechanism:
- Tracks starvation_count for each request not executed
- Promotes priority when starvation_count reaches threshold
- Allocates quantum to promoted requests
- Calculates max_waiting_time fairness metric: max(TTFT, max(TPOT))

Reference: Algorithm 1 from paper
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from opt_ltr_model import OPTRankingPredictor
import glob
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Starvation prevention parameters
STARVATION_THRESHOLD = 5  # Number of scheduling steps before promotion
QUANTUM_TOKENS = 50       # Number of tokens to allocate when promoted

def generate_poisson_arrivals(num_requests, mean_rate=1.0, seed=42, min_gap=1.0):
    """
    Generate arrival times using Poisson Process
    Reference: Kleinrock, L. (1975). "Queueing Systems, Volume 1: Theory"
    """
    np.random.seed(seed)
    inter_arrival_times = np.random.exponential(1.0 / mean_rate, num_requests)
    inter_arrival_times = np.maximum(inter_arrival_times, min_gap)
    arrival_times = np.cumsum(inter_arrival_times)
    return arrival_times

def create_synthetic_prompts(prompt_tokens_list):
    """
    Create synthetic prompts based on prompt_tokens count
    Since main_llm CSV doesn't have actual prompts, we create proxy prompts
    """
    prompts = []
    for n_tokens in prompt_tokens_list:
        # Create a prompt that approximates the token count
        # This allows LTR model to at least use length as a signal
        prompt = "Please analyze: " + "information " * max(1, int(n_tokens) - 3)
        prompts.append(prompt)
    return prompts

def predict_ltr_scores(prompts, model, tokenizer, max_length=256):
    """
    Use trained LTR model to predict ranking scores for prompts
    
    Args:
        prompts: List of prompt strings
        model: Trained OPT-125M ranking predictor
        tokenizer: OPT tokenizer
        max_length: Max sequence length
    
    Returns:
        List of ranking scores (higher = should be scheduled earlier)
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize prompt
            encoding = tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            
            # Get ranking score
            score = model(input_ids, attention_mask)
            scores.append(score.item())
    
    return scores

def calculate_max_waiting_time(ttft, tpot_values):
    """
    Calculate max_waiting_time fairness metric
    
    max_waiting_time = max(TTFT, max(TPOT))
    
    Args:
        ttft: Time To First Token
        tpot_values: List of Time Per Output Token values
    
    Returns:
        max_waiting_time: Maximum time interval between receiving tokens
    """
    if len(tpot_values) == 0:
        return ttft
    
    max_tpot = max(tpot_values)
    return max(ttft, max_tpot)

def simulate_ltr_with_starvation_prevention(fcfs_file, model, tokenizer):
    """
    Simulate LTR scheduling with starvation prevention
    
    Starvation Prevention Algorithm:
    1. Track starvation_count for each waiting request
    2. When starvation_count >= STARVATION_THRESHOLD, promote to high priority
    3. Allocate QUANTUM_TOKENS to promoted request
    4. After quantum exhausted, demote back to original priority
    
    Args:
        fcfs_file: Path to FCFS results CSV
        model: Trained LTR model
        tokenizer: OPT tokenizer
    
    Returns:
        DataFrame with LTR schedule including starvation metrics
    """
    # Load FCFS results
    df = pd.read_csv(fcfs_file)
    
    ltr_results = []
    
    # Process each model separately
    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name].copy()
        
        # Use arrival times from CSV or generate Poisson arrivals
        model_data = model_data.sort_values('task_id')
        if 'arrival_time' not in model_data.columns:
            # Generate Poisson arrivals (λ=1.0 req/sec, min_gap=1.0s)
            arrival_times = generate_poisson_arrivals(len(model_data), mean_rate=1.0, min_gap=1.0)
            model_data['arrival_time'] = arrival_times
        
        # Create synthetic prompts based on prompt_tokens
        prompts = create_synthetic_prompts(model_data['prompt_tokens'].tolist())
        
        # Predict LTR scores for all tasks
        print(f"  Predicting LTR scores for {len(prompts)} tasks...")
        ltr_scores = predict_ltr_scores(prompts, model, tokenizer)
        model_data['ltr_score'] = ltr_scores
        
        # Create task objects with starvation tracking
        pending_tasks = []
        for idx, task in model_data.iterrows():
            task_dict = task.to_dict()
            task_dict['starvation_count'] = 0
            task_dict['is_promoted'] = False
            task_dict['quantum_remaining'] = 0
            task_dict['original_ltr_score'] = task_dict['ltr_score']
            task_dict['tokens_generated'] = 0  # Track tokens generated so far
            task_dict['first_start_time'] = None  # Track first execution start
            task_dict['first_token_time'] = None  # Track TTFT
            task_dict['tpot_values'] = []  # Track all TPOT values
            pending_tasks.append(task_dict)
        
        current_time = 0.0
        scheduling_step = 0
        
        while pending_tasks:
            # Get all tasks that have arrived
            available_tasks = [t for t in pending_tasks if t['arrival_time'] <= current_time]
            
            if not available_tasks:
                # No tasks available, fast forward to next arrival
                current_time = min(t['arrival_time'] for t in pending_tasks)
                available_tasks = [t for t in pending_tasks if t['arrival_time'] <= current_time]
            
            # Increment starvation count for all waiting tasks (except the one being scheduled)
            scheduling_step += 1
            
            # Apply starvation prevention
            for task in available_tasks:
                if not task['is_promoted']:
                    task['starvation_count'] += 1
                    
                    # Promote if starvation threshold reached
                    if task['starvation_count'] >= STARVATION_THRESHOLD:
                        task['is_promoted'] = True
                        task['quantum_remaining'] = QUANTUM_TOKENS
                        # Promote priority by setting very low score (will be scheduled first)
                        task['ltr_score'] = -1000.0  # High priority
            
            # Sort available tasks:
            # 1. Promoted tasks first (is_promoted=True, lowest ltr_score)
            # 2. Regular tasks by LTR score (ascending = shortest first)
            available_tasks.sort(key=lambda x: (
                not x['is_promoted'],  # Promoted tasks first
                x['ltr_score'],        # Then by LTR score
                x['task_id']           # Tie-breaker
            ))
            
            # Schedule the highest priority task
            task = available_tasks[0]
            
            # Determine how much work to do
            if task['is_promoted']:
                # Execute quantum or remaining work, whichever is less
                tokens_to_generate = min(task['quantum_remaining'], 
                                        task['output_tokens'] - task['tokens_generated'])
            else:
                # Execute entire task (non-preemptive within a task)
                tokens_to_generate = task['output_tokens'] - task['tokens_generated']
            
            # Calculate execution time for this quantum
            if task['tokens_per_sec'] > 0:
                quantum_exec_time = tokens_to_generate / task['tokens_per_sec']
            else:
                quantum_exec_time = tokens_to_generate * 0.1  # Fallback
            
            arrival_time = task['arrival_time']
            actual_start = max(current_time, arrival_time)
            actual_end = actual_start + quantum_exec_time
            
            # Track first start time (for waiting time calculation)
            if task['first_start_time'] is None:
                task['first_start_time'] = actual_start
            
            # Track TTFT (Time To First Token)
            if task['first_token_time'] is None:
                task['first_token_time'] = actual_start - arrival_time
            
            # Track TPOT (Time Per Output Token)
            if tokens_to_generate > 0:
                tpot = quantum_exec_time / tokens_to_generate
                task['tpot_values'].append(tpot)
            
            # Update task state
            task['tokens_generated'] += tokens_to_generate
            
            if task['is_promoted']:
                task['quantum_remaining'] -= tokens_to_generate
                
                # Check if quantum exhausted or task completed
                if task['quantum_remaining'] <= 0 or task['tokens_generated'] >= task['output_tokens']:
                    # Demote back to original priority
                    task['is_promoted'] = False
                    task['ltr_score'] = task['original_ltr_score']
                    task['starvation_count'] = 0  # Reset starvation count
            
            # Check if task is fully completed
            if task['tokens_generated'] >= task['output_tokens']:
                # Task completed, remove from pending
                pending_tasks.remove(task)
                
                # Calculate final metrics using first start and final end
                first_start = task['first_start_time']
                final_end = actual_end
                total_execution_time = task['execution_time_sec']
                
                turnaround_time = final_end - arrival_time  # Total time in system
                waiting_time = first_start - arrival_time    # Time before first execution
                
                # Use measured TTFT
                ttft = task.get('ttft', task['first_token_time'])
                response_time = waiting_time + ttft  # Time to receive first token
                
                # Average latency per token
                if task['output_tokens'] > 0:
                    average_latency = turnaround_time / task['output_tokens']
                else:
                    average_latency = turnaround_time
                
                ltr_results.append({
                    'model': task['model'],
                    'task_id': task['task_id'],
                    'prompt_tokens': task['prompt_tokens'],
                    'ltr_score': task['original_ltr_score'],
                    'arrival_time': arrival_time,
                    'start_sec': first_start,
                    'end_sec': final_end,
                    'execution_time_sec': total_execution_time,
                    'output_tokens': task['output_tokens'],
                    'ttft': ttft,
                    'tokens_per_sec': task.get('tokens_per_sec', 0.0),
                    'turnaround_time': turnaround_time,
                    'waiting_time': waiting_time,
                    'response_time': response_time,
                    'average_latency': average_latency
                })
            
            # Update current time
            current_time = actual_end
    
    return pd.DataFrame(ltr_results)

def main():
    print("="*80)
    print("LTR Scheduler - Using Pre-computed FCFS Results")
    print("="*80)
    
    # Load trained LTR model
    print("\nLoading trained LTR model...")
    model_path = "opt_ltr_model_best.pt"
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("  Please train the model first using: python3 train_opt_ltr.py")
        return
    
    model = OPTRankingPredictor().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"✓ Loaded model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    print("✓ Loaded OPT tokenizer")
    
    # Find all main_llm CSV files
    fcfs_files = glob.glob("main_llm_*.csv")
    
    if not fcfs_files:
        print("\n✗ No main_llm_*.csv files found!")
        print("  Please run fcfs_main_llm.py first to generate the data.")
        return
    
    print(f"\n✓ Found {len(fcfs_files)} FCFS result files")
    
    # Create output directory
    output_dir = "ltr_starv_prevention_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    all_results = []
    
    for fcfs_file in sorted(fcfs_files):
        print(f"\n{'='*80}")
        print(f"Processing: {fcfs_file}")
        print(f"{'='*80}")
        
        # Simulate LTR scheduling with starvation prevention
        ltr_df = simulate_ltr_with_starvation_prevention(fcfs_file, model, tokenizer)
        
        # Generate output filename
        base_name = os.path.basename(fcfs_file)
        output_file = os.path.join(output_dir, f"ltr_starv_prevention_{base_name}")
        
        # Save results
        ltr_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved LTR results to: {output_file}")
        
        # Print summary statistics
        print("\nLTR Scheduling Metrics:")
        print(f"  Average Turnaround Time: {ltr_df['turnaround_time'].mean():.2f}s")
        print(f"  Average Waiting Time: {ltr_df['waiting_time'].mean():.2f}s")
        print(f"  Average Response Time: {ltr_df['response_time'].mean():.2f}s")
        print(f"  Average Latency (per token): {ltr_df['average_latency'].mean():.4f}s")
        print(f"  Makespan: {ltr_df['end_sec'].max():.2f}s")
        
        all_results.append({
            'file': base_name,
            'avg_turnaround_time': ltr_df['turnaround_time'].mean(),
            'avg_waiting_time': ltr_df['waiting_time'].mean(),
            'avg_response_time': ltr_df['response_time'].mean(),
            'avg_latency': ltr_df['average_latency'].mean(),
            'makespan': ltr_df['end_sec'].max()
        })
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(output_dir, "ltr_starv_prevention_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n{'='*80}")
    print("LTR Scheduling Complete!")
    print(f"{'='*80}")
    print(f"\n✓ Results saved to: {output_dir}/")
    print(f"✓ Summary saved to: {summary_file}")
    
    print("\nOverall Statistics:")
    print(f"  Average Turnaround Time: {summary_df['avg_turnaround_time'].mean():.2f}s")
    print(f"  Average Waiting Time: {summary_df['avg_waiting_time'].mean():.2f}s")
    print(f"  Average Response Time: {summary_df['avg_response_time'].mean():.2f}s")

if __name__ == "__main__":
    main()
