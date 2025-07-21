TASK_ID=grpo_test_now
MODEL="Qwen/Qwen3-1.7B-Base"
DATASET="test"
DATASET_TYPE='{"field_prompt": "prompt", "reward_functions": [{"reward_func": "def reward_long_completions(completions, **kwargs):\n    \"\"\"Reward function that gives higher scores to longer completions.\"\"\"\n    return [float(len(completion)) for completion in completions]\n", "reward_weight": 2.8525694672837068, "func_hash": "86ffe4cc83f1383ea35e5bacc67789b8aee2ef095df337b68f67bb7191c8b72e", "is_generic": true}, {"reward_func": "def reward_flesch_kincaid_grade(completions, **kwargs):\n    \"\"\"Rewards text matching target grade level via Flesch-Kincaid Grade Level.\"\"\"\n    import textstat\n    target_grade = 12\n    scores = [textstat.flesch_kincaid_grade(comp) for comp in completions]\n    return [1 - min(abs(s - target_grade)/10, 1) for s in scores]\n", "reward_weight": 8.024360087180563, "func_hash": "d2191dc2a768db4ab6d3412584db0bd0936f0e5ad5e7078b36d637a8bf7b718d", "is_generic": true}]}'
TASK_TYPE="GrpoTask"
FILE_FORMAT="s3"
EXPECTED_REPO_NAME="test-repo-name-grpo"
HOURS_TO_COMPLETE=0.4
# URL of the dataset

python -m text_trainer --task-id $TASK_ID --model $MODEL --dataset $DATASET --dataset-type "$DATASET_TYPE" --task-type $TASK_TYPE --file-format $FILE_FORMAT --expected-repo-name $EXPECTED_REPO_NAME --hours-to-complete $HOURS_TO_COMPLETE