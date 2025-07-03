"""
Unsloth-based LLM Fine-tuning with GRPO for A02
Optimized for DeepSeek R1 with 4GB VRAM + Core i5 CPU
"""

import os
import torch
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import gc
import psutil
import GPUtil

# Unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth import UnslothTrainer, UnslothTrainingArguments

# Transformers and datasets
from transformers import TrainingArguments, TextStreamer
from datasets import Dataset, load_dataset
from trl import SFTTrainer, DPOTrainer

# GRPO implementation
try:
    from unsloth.grpo import GRPOTrainer, GRPOConfig
    GRPO_AVAILABLE = True
except ImportError:
    GRPO_AVAILABLE = False
    logging.warning("GRPO not available, falling back to standard fine-tuning")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnslothConfig:
    """Configuration for Unsloth fine-tuning"""
    model_name: str = "unsloth/DeepSeek-R1-Distill-Llama-8B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict] = None
    
    # Training parameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    fp16: bool = not is_bfloat16_supported()
    bf16: bool = is_bfloat16_supported()
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    
    # GRPO specific parameters
    use_grpo: bool = True
    grpo_beta: float = 0.1
    grpo_gamma: float = 0.99
    grpo_group_size: int = 4
    grpo_temperature: float = 1.0
    
    # Memory optimization
    max_memory_usage: float = 0.8  # 80% of available memory
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    
    # Output settings
    output_dir: str = "./results"
    save_steps: int = 20
    save_total_limit: int = 2

@dataclass
class FineTuningResult:
    """Result of fine-tuning operation"""
    success: bool
    model_path: Optional[str] = None
    training_time: float = 0.0
    final_loss: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

class UnslothFineTuner:
    """
    Unsloth-based fine-tuner for LLMs with GRPO support
    
    Features:
    - DeepSeek R1 fine-tuning with 4GB VRAM
    - GRPO (Group Relative Policy Optimization)
    - Memory-efficient training with LoRA
    - Automatic model export to GGUF
    - Performance monitoring
    """
    
    def __init__(self, config: UnslothConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Performance tracking
        self.training_start_time = 0.0
        self.memory_stats = {}
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
    def _check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements"""
        try:
            # Check GPU memory
            if torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                gpu_memory_gb = gpu.memoryTotal / 1024
                logger.info(f"GPU: {gpu.name}, Memory: {gpu_memory_gb:.1f}GB")
                
                if gpu_memory_gb < 3.5:
                    logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) is below recommended 4GB")
                    return False
            else:
                logger.warning("No GPU detected, using CPU (very slow)")
            
            # Check RAM
            ram_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"System RAM: {ram_gb:.1f}GB")
            
            if ram_gb < 8:
                logger.warning(f"System RAM ({ram_gb:.1f}GB) is below recommended 8GB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"System check failed: {e}")
            return False
    
    def _load_model_and_tokenizer(self) -> bool:
        """Load model and tokenizer with Unsloth optimizations"""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load model with Unsloth optimizations
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
            )
            
            # Apply LoRA adapters
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.bias,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=self.config.random_state,
                use_rslora=self.config.use_rslora,
                loftq_config=self.config.loftq_config,
            )
            
            # Set chat template
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="chatml",  # or "llama-3", "mistral", etc.
            )
            
            logger.info("Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _prepare_dataset(self, dataset_path: str, dataset_type: str = "json") -> Optional[Dataset]:
        """Prepare dataset for training"""
        try:
            logger.info(f"Loading dataset from: {dataset_path}")
            
            if dataset_type == "json":
                # Load JSON dataset
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to Hugging Face dataset
                dataset = Dataset.from_list(data)
                
            elif dataset_type == "huggingface":
                # Load from Hugging Face hub
                dataset = load_dataset(dataset_path, split="train")
                
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
            # Format dataset for chat template
            def formatting_prompts_func(examples):
                convos = examples["conversations"]
                texts = []
                for convo in convos:
                    text = self.tokenizer.apply_chat_template(
                        convo, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    texts.append(text)
                return {"text": texts}
            
            dataset = dataset.map(
                formatting_prompts_func, 
                batched=True,
                remove_columns=dataset.column_names
            )
            
            logger.info(f"Dataset prepared: {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            return None
    
    def _create_trainer(self, dataset: Dataset, use_grpo: bool = False) -> bool:
        """Create trainer (SFT or GRPO)"""
        try:
            # Training arguments
            training_args = UnslothTrainingArguments(
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                logging_steps=self.config.logging_steps,
                optim=self.config.optim,
                weight_decay=self.config.weight_decay,
                lr_scheduler_type=self.config.lr_scheduler_type,
                seed=self.config.seed,
                output_dir=self.config.output_dir,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                dataloader_num_workers=self.config.dataloader_num_workers,
                gradient_checkpointing=self.config.gradient_checkpointing,
            )
            
            if use_grpo and GRPO_AVAILABLE:
                logger.info("Creating GRPO trainer")
                
                # GRPO configuration
                grpo_config = GRPOConfig(
                    beta=self.config.grpo_beta,
                    gamma=self.config.grpo_gamma,
                    group_size=self.config.grpo_group_size,
                    temperature=self.config.grpo_temperature,
                )
                
                self.trainer = GRPOTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=dataset,
                    tokenizer=self.tokenizer,
                    grpo_config=grpo_config,
                    max_length=self.config.max_seq_length,
                )
                
            else:
                logger.info("Creating SFT trainer")
                
                self.trainer = UnslothTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    train_dataset=dataset,
                    args=training_args,
                    max_seq_length=self.config.max_seq_length,
                    dataset_text_field="text",
                    packing=False,  # Can make training 5x faster for short sequences
                )
            
            logger.info("Trainer created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create trainer: {e}")
            return False
    
    def _monitor_memory_usage(self):
        """Monitor memory usage during training"""
        try:
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
                self.memory_stats['gpu_memory_gb'] = gpu_memory
                self.memory_stats['gpu_memory_max_gb'] = gpu_memory_max
            
            # System memory
            memory = psutil.virtual_memory()
            self.memory_stats['ram_usage_gb'] = memory.used / 1024**3
            self.memory_stats['ram_percent'] = memory.percent
            
            # Check if memory usage is too high
            if memory.percent > self.config.max_memory_usage * 100:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")
    
    def fine_tune(self, 
                  dataset_path: str, 
                  dataset_type: str = "json",
                  use_grpo: bool = None) -> FineTuningResult:
        """
        Fine-tune the model
        
        Args:
            dataset_path: Path to training dataset
            dataset_type: Type of dataset ("json" or "huggingface")
            use_grpo: Whether to use GRPO (default: from config)
        
        Returns:
            FineTuningResult object
        """
        
        start_time = time.time()
        self.training_start_time = start_time
        
        try:
            # Check system requirements
            if not self._check_system_requirements():
                return FineTuningResult(
                    success=False,
                    error="System requirements not met"
                )
            
            # Load model and tokenizer
            if not self._load_model_and_tokenizer():
                return FineTuningResult(
                    success=False,
                    error="Failed to load model and tokenizer"
                )
            
            # Prepare dataset
            dataset = self._prepare_dataset(dataset_path, dataset_type)
            if dataset is None:
                return FineTuningResult(
                    success=False,
                    error="Failed to prepare dataset"
                )
            
            # Create trainer
            use_grpo = use_grpo if use_grpo is not None else self.config.use_grpo
            if not self._create_trainer(dataset, use_grpo):
                return FineTuningResult(
                    success=False,
                    error="Failed to create trainer"
                )
            
            # Start training
            logger.info("Starting fine-tuning...")
            self._monitor_memory_usage()
            
            # Train the model
            trainer_stats = self.trainer.train()
            
            # Monitor final memory usage
            self._monitor_memory_usage()
            
            # Save the model
            model_path = os.path.join(self.config.output_dir, "final_model")
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            training_time = time.time() - start_time
            final_loss = trainer_stats.training_loss if hasattr(trainer_stats, 'training_loss') else 0.0
            
            logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")
            logger.info(f"Final loss: {final_loss:.4f}")
            
            return FineTuningResult(
                success=True,
                model_path=model_path,
                training_time=training_time,
                final_loss=final_loss,
                memory_usage=self.memory_stats.copy(),
                metrics={
                    'trainer_stats': trainer_stats.metrics if hasattr(trainer_stats, 'metrics') else {},
                    'use_grpo': use_grpo,
                    'model_name': self.config.model_name,
                }
            )
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return FineTuningResult(
                success=False,
                training_time=time.time() - start_time,
                memory_usage=self.memory_stats.copy(),
                error=str(e)
            )
    
    def export_to_gguf(self, 
                       model_path: str, 
                       output_path: str,
                       quantization: str = "q4_k_m") -> bool:
        """
        Export model to GGUF format for llama.cpp
        
        Args:
            model_path: Path to the fine-tuned model
            output_path: Output path for GGUF file
            quantization: Quantization method (q4_k_m, q5_k_m, q8_0, etc.)
        
        Returns:
            Success status
        """
        try:
            logger.info(f"Exporting model to GGUF: {output_path}")
            
            # Load the fine-tuned model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=False,  # Don't quantize for export
            )
            
            # Export to GGUF
            model.save_pretrained_gguf(
                output_path,
                tokenizer,
                quantization_method=quantization
            )
            
            logger.info(f"Model exported to GGUF: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            return False
    
    def test_inference(self, 
                      model_path: str, 
                      test_prompt: str,
                      max_new_tokens: int = 128) -> Optional[str]:
        """
        Test inference with the fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned model
            test_prompt: Test prompt
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated text or None if failed
        """
        try:
            logger.info("Testing inference...")
            
            # Load the fine-tuned model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable fast inference
            FastLanguageModel.for_inference(model)
            
            # Format prompt
            messages = [{"role": "user", "content": test_prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            text_streamer = TextStreamer(tokenizer, skip_prompt=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    streamer=text_streamer,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=0.7,
                    do_sample=True,
                )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][len(inputs[0]):], 
                skip_special_tokens=True
            )            
            logger.info(f"Generated response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            return None

# Example usage and testing
def create_sample_dataset(output_path: str, tokenizer):
    """Create a sample dataset for testing"""
    sample_conversations = [
        [
            {"role": "user", "content": "Trí tuệ nhân tạo là gì?"},
            {"role": "assistant", "content": "Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ đòi hỏi trí thông minh của con người."}
        ],
        [
            {"role": "user", "content": "Giải thích học sâu."},
            {"role": "assistant", "content": "Học sâu là một nhánh của học máy sử dụng mạng nơ-ron nhân tạo với nhiều lớp để mô hình hóa và hiểu các mẫu phức tạp trong dữ liệu."}
        ],
        [
            {"role": "user", "content": "Sự khác biệt giữa AI và ML là gì?"},
            {"role": "assistant", "content": "AI là khái niệm rộng hơn về khả năng của máy móc thực hiện các nhiệm vụ một cách thông minh, trong khi ML là một ứng dụng cụ thể của AI tập trung vào việc máy móc có thể học từ dữ liệu."}
        ]
    ]

    formatted_data = []
    for convo in sample_conversations:
        formatted_data.append({"conversations": convo})
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample dataset created: {output_path}")

def main():
    """Example usage of UnslothFineTuner"""
    
    # Create sample dataset
    dataset_path = "./sample_dataset.json"
    create_sample_dataset(dataset_path, fine_tuner.tokenizer)    
    # Configuration for 4GB VRAM + Core i5
    config = UnslothConfig(
        model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length=1024,  # Reduced for 4GB VRAM
        load_in_4bit=True,
        per_device_train_batch_size=1,  # Small batch size
        gradient_accumulation_steps=8,  # Compensate with accumulation
        max_steps=30,  # Quick test
        learning_rate=2e-4,
        use_grpo=True,  # Enable GRPO
        output_dir="./fine_tuned_model"
    )
    
    # Create fine-tuner
    fine_tuner = UnslothFineTuner(config)
    
    # Fine-tune the model
    result = fine_tuner.fine_tune(
        dataset_path=dataset_path,
        dataset_type="json",
        use_grpo=True
    )
    
    if result.success:
        print(f"Fine-tuning successful!")
        print(f"Model saved to: {result.model_path}")
        print(f"Training time: {result.training_time:.2f} seconds")
        print(f"Final loss: {result.final_loss:.4f}")
        print(f"Memory usage: {result.memory_usage}")
        
        # Test inference
        response = fine_tuner.test_inference(
            model_path=result.model_path,
            test_prompt="What is artificial intelligence?",
            max_new_tokens=64
        )
        
        if response:
            print(f"Test response: {response}")
        
        # Export to GGUF
        gguf_path = "./fine_tuned_model.gguf"
        if fine_tuner.export_to_gguf(result.model_path, gguf_path):
            print(f"Model exported to GGUF: {gguf_path}")
    
    else:
        print(f"Fine-tuning failed: {result.error}")

if __name__ == "__main__":
    main()

