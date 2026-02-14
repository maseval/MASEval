python examples/colbench_benchmark/colbench_benchmark.py \
    --agent_model meta-llama/Llama-3.1-8B-Instruct \
    --hostname localhost \
    --task_type code \
    --num_tasks 1000 \
    --input_path examples/colbench_benchmark/results/test.jsonl \
    --output_path examples/colbench_benchmark/results/temp_test.jsonl \
    --env_model meta-llama/Llama-3.1-8B-Instruct 
python examples/colbench_benchmark/evaluate_code.py examples/colbench_benchmark/results/temp_test.jsonl
