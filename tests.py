from datasets import load_dataset
import random

# Load the structured data extraction dataset
dataset = load_dataset("mrdbourke/FoodExtract-1k")

dataset = dataset["train"].train_test_split(test_size=0.1)


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
#
# Let's look at a random sample to see the exact input and output strings
random_idx = random.randint(0, len(dataset["train"])-1)
random_sample = dataset["train"][random_idx]

# This is the raw unstructured text (Input)
example_input = random_sample["sequence"]
def format_chat(example):
    # Using standard chat format allows SFTTrainer to automatically apply chat templates
    # and works perfectly with `completion_only_loss=True`
    return {
        "messages": [
            {"role": "user", "content": f"Extract food and drink information from the following text:\n\n{example['sequence']}"},
            {"role": "assistant", "content": example['gpt-oss-120b-label-condensed']}
        ]
    }

train = dataset["train"].map(format_chat)
test = dataset["test"].map(format_chat)
print(len(test))



# This is the condensed YAML-like format we want the model to generate (Output)
example_output_condensed = random_sample["gpt-oss-120b-label-condensed"]
print(random_sample.keys())

print(f"--- INPUT SHAPE (Raw Text) ---\n{example_input}\n")
print(f"--- OUTPUT SHAPE (Target Completion) ---\n{example_output_condensed}")

import torch
from trl import SFTConfig

# Note: Lowered batch size to prevent MPS Out-Of-Memory errors during training
BATCH_SIZE = 2
BASE_LEARNING_RATE = 5e-5
CHECKPOINT_DIR_NAME = "gemma-3-270m-food-extract-checkpoints"

# Setup Supervised Fine-Tuning parameters
sft_config = SFTConfig(
    output_dir=CHECKPOINT_DIR_NAME,
    max_length=512, # Inputs/Outputs longer than this in tokens will be truncated
    packing=False,
    num_train_epochs=1, # Passes through the whole dataset 3 times
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    
    # CRITICAL: We want our model to only learn how to *complete* the output tokens 
    # given the input tokens. The loss is calculated only on the generated labels.
    completion_only_loss=True, 
    
    gradient_checkpointing=False,
    optim="adamw_torch_fused", 
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=BASE_LEARNING_RATE,
    bf16=True, # Assuming you are using an Ampere GPU or newer (otherwise use fp16=True)
    load_best_model_at_end=True,
    metric_for_best_model="mean_token_accuracy",
    greater_is_better=True,
    lr_scheduler_type="constant",
)
from trl import SFTTrainer

# Note: This assumes you have already loaded your base model and tokenizer
# e.g., model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
# e.g., tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train,
    eval_dataset=test,
    processing_class=tokenizer    # The tokenizer handles formatting the input/output shapes
)

# Start fine-tuning!
training_output = trainer.train()
