import torch
import json
import os
from datetime import datetime
from trl import SFTTrainer
from transformers import (
    TrainingArguments, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
from datasets import Dataset, load_dataset


class IterativeFineTuner:
    def __init__(self, config):
        self.D = {0: config["initial_dataset"]} 
        self.M = {0: config["domain_adapted_models"]}  
        self.L_b = config["L_b"] 
        self.L_l = config["L_l"]  
        self.L_h = config["L_h"] 
        
        self.k = 0  
        self.max_iterations = config.get("max_iterations", 10) 
        self.output_dir = config.get("output_dir", "./iterative_finetune")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.base_model_path = config["base_model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token  

    def _compute_combined_loss(self, model, dataset):
        y_true, y_pred = [], []  # 真实标签y_i、预测标签ŷ_i
        r_true, r_pred = [], []  # 真实解释r_i、预测解释âr_i
        
        for item in dataset:
            inputs = self.tokenizer(
                item["x"], 
                return_tensors="pt",
                truncation=True,
                max_length=1024  
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.pad_token_id
            )
            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            pred_y, pred_r = self._parse_model_output(pred_text)
            y_true.append(item["y"])
            y_pred.append(pred_y)
            r_true.append(item["r"])
            r_pred.append(pred_r)
        
        label_loss = self._compute_label_loss(y_true, y_pred) 
        function_loss = self._compute_function_loss(r_true, r_pred) 
        combined_loss = 0.5 * label_loss + 0.5 * function_loss  
        
        return {
            "L_j": combined_loss,      
            "label_loss": label_loss  
        }


    def _parse_model_output(self, pred_text):
        pred_y = "UNKNOWN" 
        pred_r = "No rationale parsed" 
        if "LABEL:" in pred_text:
            pred_y = pred_text.split("LABEL:")[-1].split("\n")[0].strip()
        if "RATIONALE:" in pred_text:
            pred_r = pred_text.split("RATIONALE:")[-1].strip()
        return pred_y, pred_r

    def _compute_label_loss(self, y_true, y_pred):
        return sum(1 for t, p in zip(y_true, y_pred) if t != p) / len(y_true)

    def _compute_function_loss(self, r_true, r_pred):
        return sum(len(t) != len(p) for t, p in zip(r_true, r_pred)) / len(r_true)

    def _fine_tune_models(self):
        current_models = self.M[self.k]  # M(k)_s
        current_dataset = self.D[self.k]  # D(k)
        fine_tuned_models = []
        
        for idx, model in enumerate(current_models):
            run_name = f"iter_{self.k}_model_{idx}"
            train_args = TrainingArguments(
                output_dir=os.path.join(self.output_dir, run_name),
                per_device_train_batch_size=2,  # 通用配置
                gradient_accumulation_steps=4,  # 通用配置
                max_steps=300,  # 通用配置（算法未定义，仅保证流程）
                learning_rate=2e-4,  # 通用配置
                optim="adamw_8bit",  # QLoRA高效微调（算法提及参数高效技术）
                logging_steps=10,
                save_steps=100,
                eval_steps=50,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                seed=3407,
                report_to="none"  
            )
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=self.tokenizer,
                train_dataset=current_dataset,
                eval_dataset=current_dataset,
                dataset_text_field="text",  
                max_seq_length=1024,
                packing=False
            )
      
            trainer.train()
            fine_tuned_models.append(model)
        
        return fine_tuned_models  # M(k)_ft


    def run(self):
        print(f"开始迭代微调流程，初始迭代k={self.k}")
        
        while self.k < self.max_iterations:
            print(f"\n=== 迭代k={self.k} ===")
            
            M_ft = self._fine_tune_models()
            current_models = self.M[self.k]
            need_refine_D = False 
            satisfy_model = None 
            keep_models = []   
            
            for idx, model in enumerate(current_models):
                loss_info = self._compute_combined_loss(model, self.D[self.k])
                L_j = loss_info["L_j"]
                label_loss = loss_info["label_loss"]
                print(f"模型{idx}：组合损失L_j={L_j:.4f}，标签损失={label_loss:.4f}")
                
                if label_loss > self.L_b:
                    print(f"模型{idx}：标签损失>{self.L_b}，移除（隐含逻辑，保证标签任务基础精度）")
                    continue
                  
                if L_j > self.L_h:
                    print(f"模型{idx}：L_j>{self.L_h}，从模型集合移除")
                    continue
                
                if L_j < self.L_l:
                    print(f"模型{idx}：L_j<{self.L_l}，满足要求，终止迭代")
                    satisfy_model = model
                    break
                
                if self.L_l <= L_j <= self.L_h:
                    print(f"模型{idx}：L_j在[{self.L_l},{self.L_h}]区间，保留模型并标记数据集优化")
                    keep_models.append(model)
                    need_refine_D = True
            
            if satisfy_model is not None:
                final_D = self.D[self.k]  # D_train = D(k)（算法Step 29）
                self._save_final_results(satisfy_model, final_D)
                print("\n算法流程终止：找到满足要求的模型和数据集")
                return {
                    "satisfactory_models": [satisfy_model],
                    "D_train": final_D,
                    "iteration": self.k
                }
            
            if need_refine_D:
                self.D[self.k+1] = self._refine_dataset(self.D[self.k])
                print(f"迭代k={self.k}：生成优化后数据集D({self.k+1})")
            else:
                self.D[self.k+1] = self.D[self.k]
                print(f"迭代k={self.k}：无需优化数据集，D({self.k+1})=D({self.k})")
            
            self.M[self.k+1] = M_ft if len(M_ft) > 0 else keep_models
            print(f"迭代k={self.k}：更新模型集合M({self.k+1})_s，共{len(self.M[self.k+1])}个模型")
            
            self.k += 1
            print(f"迭代k={self.k-1}完成，进入迭代k={self.k}")
        
        print(f"\n达到最大迭代次数{self.max_iterations}，算法流程终止")
        self._save_final_results(self.M[self.k], self.D[self.k])
        return {
            "satisfactory_models": self.M[self.k],
            "D_train": self.D[self.k],
            "iteration": self.k
        }


    def _save_final_results(self, models, dataset):
        # 保存达标模型
        model_dir = os.path.join(self.output_dir, "satisfactory_models")
        os.makedirs(model_dir, exist_ok=True)
        if isinstance(models, list):
            for idx, model in enumerate(models):
                model.save_pretrained(os.path.join(model_dir, f"model_{idx}"))
        else:
            models.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # 保存最终数据集
        dataset_path = os.path.join(self.output_dir, "D_train.json")
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)
        
        print(f"达标模型保存至：{model_dir}")
        print(f"最终数据集保存至：{dataset_path}")


