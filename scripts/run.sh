TASK_ID=4841bc5a-2ee7-46a2-9753-3bea798eabd3
MODEL="Qwen/Qwen3-1.7B-Base"
DATASET="test"
DATASET_TYPE='{"system_prompt": "", "system_format": "{system}", "field_system": null, "field_instruction": "instruct", "field_input": "input", "field_output": "output", "format": null, "no_input_format": null, "field": null}'
TASK_TYPE="InstructTextTask"
FILE_FORMAT="s3"
EXPECTED_REPO_NAME="test-repo-name"
HOURS_TO_COMPLETE=0.4
# URL of the dataset

python -m text_trainer --task-id $TASK_ID --model $MODEL --dataset $DATASET --dataset-type "$DATASET_TYPE" --task-type $TASK_TYPE --file-format $FILE_FORMAT --expected-repo-name $EXPECTED_REPO_NAME --hours-to-complete $HOURS_TO_COMPLETE