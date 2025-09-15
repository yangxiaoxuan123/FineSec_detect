%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps bitsandbytes==0.41.1 peft==0.8.2 trl==0.7.4 accelerate==0.24.1 datasets==2.14.6

import torch
import os
import time
from datetime import datetime
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer
)
from peft import PeftModel, LoraConfig
from unsloth import FastLanguageModel
import wandb

# ------------------------------------------------------------------------------
# 1. Core Configuration
# ------------------------------------------------------------------------------

fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
]

selected_model = ""
print(f"Selected base model: {selected_model}")

train_data_path = "./train_data.csv"
val_data_path = "./val_data.csv"
data_process_dir = "./data_process"
output_root = "./iterative_training"
os.makedirs(data_process_dir, exist_ok=True)
os.makedirs(output_root, exist_ok=True)

// dynamic adjustment
Lh = 0.8
Ll = 0.3
max_iterations = 5
current_iteration = 1
best_loss = float("inf")
best_model_path = None

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def get_gpu_memory() -> float:
    return round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 3)

start_gpu_memory = get_gpu_memory()
max_gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)
print(f"Initial GPU memory usage: {start_gpu_memory} GB / Total memory: {max_gpu_memory} GB")

# ------------------------------------------------------------------------------
# 2. Data Loading with Basic Cleaning
# ------------------------------------------------------------------------------
def load_and_basic_clean(iteration: int, refined_train_path: str = None, refined_val_path: str = None) -> tuple[Dataset, Dataset]:
    if iteration == 1 or refined_train_path is None:
        train_data = load_dataset('csv', data_files=train_data_path, split="train")
    else:
        train_data = load_dataset('csv', data_files=refined_train_path, split="train")
    
    if iteration == 1 or refined_val_path is None:
        val_data = load_dataset('csv', data_files=val_data_path, split="train")
    else:
        val_data = load_dataset('csv', data_files=refined_val_path, split="train")
    
    train_data = train_data.filter(lambda x: x["text"] is not None and len(str(x["text"]).strip()) > 50)
    val_data = val_data.filter(lambda x: x["text"] is not None and len(str(x["text"]).strip()) > 50)
    
    train_data = train_data.unique("text")
    val_data = val_data.unique("text")
    
    print(f"Data cleaning completed (Iteration {iteration}):")
    print(f"   - Training set: {len(train_data)} samples")
    print(f"   - Validation set: {len(val_data)} samples")
    
    assert "text" in train_data.column_names and "text" in val_data.column_names, \
        "Training/validation set must contain 'text' column"
    return train_data, val_data

# ------------------------------------------------------------------------------
# 3. Model Loading (with QLoRA Configuration)
# ------------------------------------------------------------------------------
def load_qlora_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        bitsandbytes_config=bnb_config,
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_gradient_checkpointing="unsloth"
    )
    
    model = FastLanguageModel.get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer

# ------------------------------------------------------------------------------
# 4. Model Performance Testing
# ------------------------------------------------------------------------------
def test_model_performance(model, tokenizer, test_name: str = "Pre-training Test"):
    print(f"\n================================ {test_name} ==================================")
    FastLanguageModel.for_inference(model)
    
    eval_prompt = """List all the vulnerabilities in the following C/C++ code:
int process_data(int* input_array, size_t array_size) {
    size_t total_size = array_size * sizeof(int);
    if (total_size < array_size) {
        return -1;
    }
    
    int* buffer = (int*)malloc(total_size);
    if (!buffer) {
        return -1;
    }

    for (size_t i = 0; i <= array_size; i++) {
        if (i < array_size) {
            buffer[i] = input_array[i] * 2;
        }
    }
    
    int result = 0;
    for (size_t i = 0; i < array_size; i++) {
        result += input_array[i];
        if (result < 0) {
            break;
        }
    }
    
    free(buffer);
    return result;
}

### Response:"""
    
    inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    print("Model output:")
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=600, use_cache=True)
    
    test_gpu_memory = get_gpu_memory()
    print(f"GPU memory usage during {test_name}: {test_gpu_memory} GB")

# ------------------------------------------------------------------------------
# 5. Data Processing Transition
# ------------------------------------------------------------------------------
def prepare_next_iter_data(current_iteration: int, train_data: Dataset, val_data: Dataset) -> tuple[str, str]:
    """
    Prepares data for next iteration with explicit basic cleaning steps
    Includes duplicate removal and null filtering with detailed implementation
    """
    next_train_path = os.path.join(data_process_dir, f"train_processed_iter{current_iteration+1}.csv")
    next_val_path = os.path.join(data_process_dir, f"val_processed_iter{current_iteration+1}.csv")
    
    # Check if processed data already exists
    if os.path.exists(next_train_path) and os.path.exists(next_val_path):
        print(f"Detected processed data for Iteration {current_iteration+1}, loading directly")
        return next_train_path, next_val_path
    
    print(f"\nBasic data processing for Iteration {current_iteration} started...")
    
    # --------------------------
    # 1. Explicit basic cleaning
    # --------------------------
    print("Performing basic cleaning steps:")
    
    # 1.1 Null filtering implementation
    print("   - Removing null/empty text samples...")
    initial_train_count = len(train_data)
    initial_val_count = len(val_data)
    
    # Filter out samples where text is None or empty/whitespace
    train_data = train_data.filter(
        lambda x: x["text"] is not None and len(str(x["text"]).strip()) > 0
    )
    val_data = val_data.filter(
        lambda x: x["text"] is not None and len(str(x["text"]).strip()) > 0
    )
    
    # Calculate and report null filtering results
    train_null_removed = initial_train_count - len(train_data)
    val_null_removed = initial_val_count - len(val_data)
    print(f"   - Removed {train_null_removed} null/empty samples from training set")
    print(f"   - Removed {val_null_removed} null/empty samples from validation set")
    
    # 1.2 Duplicate removal implementation
    print("   - Removing duplicate samples...")
    initial_train_clean = len(train_data)
    initial_val_clean = len(val_data)
    
    # Remove duplicate entries based on text field
    train_data = train_data.unique("text")
    val_data = val_data.unique("text")
    
    # Calculate and report duplicate removal results
    train_dups_removed = initial_train_clean - len(train_data)
    val_dups_removed = initial_val_clean - len(val_data)
    print(f"   - Removed {train_dups_removed} duplicate samples from training set")
    print(f"   - Removed {val_dups_removed} duplicate samples from validation set")
    
    # Report final cleaning statistics
    print(f"   - Training set after cleaning: {len(train_data)} samples")
    print(f"   - Validation set after cleaning: {len(val_data)} samples")
    
    # Save intermediate cleaned data (before manual augmentation)
    temp_train_path = os.path.join(data_process_dir, f"train_cleaned_temp_iter{current_iteration}.csv")
    temp_val_path = os.path.join(data_process_dir, f"val_cleaned_temp_iter{current_iteration}.csv")
    train_data.to_csv(temp_train_path, index=False)
    val_data.to_csv(temp_val_path, index=False)
    print(f"   - Intermediate cleaned data saved to: {temp_train_path} and {temp_val_path}")
    
    # --------------------------
    # 2. Manual optimization steps
    # --------------------------
    print("\nData optimization tips:")
    print("   1. Basic cleaning completed (duplicate removal / null filtering)")
    print("   2. Supplementary steps needed:")
    print("      - Manual data augmentation (add diverse vulnerability scenarios)")
    print("      - Label correction (verify and fix vulnerability annotations)")
    
    # Wait for user confirmation
    while True:
        user_input = input("   Confirm data has been saved (enter 'y' to continue): ").strip().lower()
        if user_input == 'y':
            break
        print("   Please complete data processing and save to specified paths first, then confirm to continue")
    
    # Validate file existence
    while not (os.path.exists(next_train_path) and os.path.exists(next_val_path)):
        print(f"Data files not detected, please check paths: {next_train_path} and {next_val_path}")
        time.sleep(2)
    
    print(f"Data preparation for Iteration {current_iteration+1} completed")
    return next_train_path, next_val_path
    
# ------------------------------------------------------------------------------
# 6. Single Iteration Training and Evaluation
# ------------------------------------------------------------------------------
def train_single_iteration(iteration: int, model, tokenizer, train_data, val_data) -> tuple[float, SFTTrainer]:
    run_name = f"iter{iteration}_{selected_model.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    training_args = TrainingArguments(
        output_dir=os.path.join(output_root, run_name),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=500,
        learning_rate=5e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        report_to="wandb" if wandb.run else None,
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    print(f"\nStarting Iteration {iteration} training, current GPU memory: {get_gpu_memory()} GB")
    trainer_stats = trainer.train()
    
    eval_metrics = trainer.evaluate()
    current_loss = eval_metrics["eval_loss"]
    print(f"\n=== Iteration {iteration} Completed ===")
    print(f"Current loss Lj = {current_loss:.4f} | Thresholds: Ll={Ll}, Lh={Lh}")
    
    peak_train_memory = round(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 3)
    print(f"Peak GPU memory in Iteration {iteration}: {peak_train_memory} GB / Total memory: {max_gpu_memory} GB")
    return current_loss, trainer, trainer_stats

# ------------------------------------------------------------------------------
# 7. Main Logic for Iterative Refinement
# ------------------------------------------------------------------------------
wandb.init(project="iterative-qlora-finetune", name=f"main-run-{selected_model.split('/')[-1]}", reinit=True)

train_data, val_data = load_and_basic_clean(current_iteration)

print("\n==================== Pre-training Model Test ====================")
init_model, init_tokenizer = load_qlora_model(selected_model)
test_model_performance(init_model, init_tokenizer, test_name="Pre-training Test")
del init_model
torch.cuda.empty_cache()

while current_iteration <= max_iterations:
    print(f"\n==================== Iteration {current_iteration}/{max_iterations} ====================")
    
    model, tokenizer = load_qlora_model(selected_model)
    
    current_loss, trainer, trainer_stats = train_single_iteration(
        current_iteration, model, tokenizer, train_data, val_data
    )
    
    if current_loss < Ll:
        print(f"Iteration {current_iteration}: Lj < Ll, model meets standards!")
        best_model_path = os.path.join(output_root, f"best_model_iter{current_iteration}")
        trainer.model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"Best model saved to: {best_model_path}")
        
        print("\n==================== Post-training Best Model Test ====================")
        base_model = AutoModelForCausalLM.from_pretrained(
            selected_model,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            bitsandbytes_config=bnb_config
        )
        best_merged_model = PeftModel.from_pretrained(base_model, best_model_path)
        best_merged_model = best_merged_model.merge_and_unload()
        test_model_performance(best_merged_model, tokenizer, test_name="Post-training Best Model Test")
        break
    
    elif current_loss > Lh:
        print(f"Iteration {current_iteration}: Lj > Lh, model failed to learn effectively, discarding!")
        print("Please check: 1. Data quality 2. Learning rate 3. Model compatibility")
        next_train_path, next_val_path = prepare_next_iter_data(current_iteration)
        train_data, val_data = load_and_basic_clean(current_iteration + 1, next_train_path, next_val_path)
        current_iteration += 1
        del model
        torch.cuda.empty_cache()
    
    else:
        print(f"Iteration {current_iteration}: Medium loss, starting data optimization process...")
        temp_model_path = os.path.join(output_root, f"temp_model_iter{current_iteration}")
        trainer.model.save_pretrained(temp_model_path)
        tokenizer.save_pretrained(temp_model_path)
        print(f"Temporary model saved to: {temp_model_path}")
        
        refined_train, refined_val = prepare_next_iter_data(current_iteration)
        
        train_data, val_data = load_and_basic_clean(current_iteration + 1, refined_train, refined_val)
        current_iteration += 1
        del model
        torch.cuda.empty_cache()

# ------------------------------------------------------------------------------
# 8. Final Training Statistics
# ------------------------------------------------------------------------------
print("\n==================== Final Training Statistics ====================")
final_gpu_memory = get_gpu_memory()
peak_total_memory = round(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 3)
print(f"Initial GPU memory: {start_gpu_memory} GB")
print(f"Final GPU memory: {final_gpu_memory} GB")
print(f"Peak GPU memory: {peak_total_memory} GB / Total memory: {max_gpu_memory} GB")
print(f"Peak memory usage: {round(peak_total_memory/max_gpu_memory*100, 2)}%")

if 'trainer_stats' in locals():
    train_time = round(trainer_stats.metrics['train_runtime'], 2)
    train_minutes = round(train_time / 60, 2)
    print(f"Total training time: {train_time} seconds ({train_minutes} minutes)")
    print(f"Average training speed: {round(trainer_stats.metrics['train_samples_per_second'], 2)} samples/second")

wandb.finish()
