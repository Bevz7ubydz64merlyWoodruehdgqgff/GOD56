TASK_ID=dpo_test_now
MODEL="Qwen/Qwen3-1.7B-Base"
DATASET="test"
DATASET_TYPE='{"field_prompt": "prompt", "field_system": null, "field_chosen": "chosen", "field_rejected": "rejected", "prompt_format": null, "chosen_format": null, "rejected_format": null}'
TASK_TYPE="DpoTask"
FILE_FORMAT="s3"
EXPECTED_REPO_NAME="test-repo-name-dpo"
HOURS_TO_COMPLETE=0.4
# URL of the dataset

python -m text_trainer --task-id $TASK_ID --model $MODEL --dataset $DATASET --dataset-type "$DATASET_TYPE" --task-type $TASK_TYPE --file-format $FILE_FORMAT --expected-repo-name $EXPECTED_REPO_NAME --hours-to-complete $HOURS_TO_COMPLETE