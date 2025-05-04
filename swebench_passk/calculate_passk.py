import json
import os
import argparse
from collections import defaultdict

def calculate_pass_at_k(directory_path, chunking_strategy, retriever, generation_model, k=8):
    """
    Calculate pass@k metric for SWE-bench evaluations.

    Args:
        directory_path: Path to the directory containing evaluation results
        chunking_strategy: Chunking strategy used (e.g., 'fixed' or 'ast')
        retriever: Retriever model used (e.g., 'BGE-base')
        generation_model: Generation model used (e.g., 'gemini-2.5-pro')
        k: Number of attempts per instance

    Returns:
        float: pass@k score (percentage of instances resolved in at least 1 of k attempts)
    """
    # Dictionary to track resolution status for each instance
    instance_results = defaultdict(list)

    # Process each result file from 0 to k-1
    for i in range(k):
        file_name = f"swebench-lite_{chunking_strategy}-chunking_{retriever}_{generation_model}_generations_swebench-lite_{i}.jsonl"
        file_path = os.path.join(directory_path, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            continue

        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                instance_id = data.get("instance_id")
                resolved = data.get("report", {}).get("resolved", False)

                # Store resolution status for this instance in this attempt
                instance_results[instance_id].append(resolved)

    # Count instances resolved in at least one attempt
    resolved_count = 0
    total_instances = 0

    for instance_id, results in instance_results.items():
        total_instances += 1
        if any(results):
            resolved_count += 1

    # Calculate pass@k
    pass_at_k = (resolved_count / total_instances) * 100 if total_instances > 0 else 0

    return {
        "pass_at_k": pass_at_k,
        "resolved_count": resolved_count,
        "total_instances": total_instances,
        "instance_results": instance_results
    }

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Calculate pass@k metric for SWE-bench evaluations')
    parser.add_argument('--chunking_strategy', type=str, default='ast',
                        help='Chunking strategy (fixed or ast)')
    parser.add_argument('--retriever', type=str, default='BGE-base',
                        help='Retriever model (e.g., BGE-base)')
    parser.add_argument('--generation_model', type=str, default='gemini-2.5-pro',
                        help='Generation model (e.g., gemini-2.5-pro)')
    parser.add_argument('--k', type=int, default=8,
                        help='Number of attempts per instance')
    parser.add_argument('--detailed_report', action='store_true',
                        help='Generate detailed report of resolved instances')

    args = parser.parse_args()

    # Construct directory path based on chunking strategy
    directory = f"swebench_passk/{args.chunking_strategy}_chunking"

    # Calculate pass@k
    results = calculate_pass_at_k(
        directory_path=directory,
        chunking_strategy=args.chunking_strategy,
        retriever=args.retriever,
        generation_model=args.generation_model,
        k=args.k
    )

    print(f"Pass@{args.k} Results for:")
    print(f"  Chunking Strategy: {args.chunking_strategy}")
    print(f"  Retriever: {args.retriever}")
    print(f"  Generation Model: {args.generation_model}")
    print(f"Resolved instances: {results['resolved_count']}/{results['total_instances']} ({results['pass_at_k']:.2f}%)")

    # Print list of resolved instance IDs if detailed report is requested
    if args.detailed_report:
        print("\nGenerating detailed report...")

        # Create a mapping of instance_id to the attempts where it was resolved
        resolved_attempts = defaultdict(list)

        # Process each result file from 0 to k-1
        for i in range(args.k):
            file_name = f"swebench-lite_{args.chunking_strategy}-chunking_{args.retriever}_{args.generation_model}_generations_swebench-lite_{i}.jsonl"
            file_path = os.path.join(directory, file_name)

            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    instance_id = data.get("instance_id")
                    resolved = data.get("report", {}).get("resolved", False)

                    if resolved:
                        resolved_attempts[instance_id].append(str(i))

        if resolved_attempts:
            print("\nResolved instances:")
            for instance_id, file_indices in resolved_attempts.items():
                print(f"  {instance_id}: resolved in attempt(s) {', '.join(file_indices)}")
        else:
            print("\nNo instances were resolved in any attempt.")