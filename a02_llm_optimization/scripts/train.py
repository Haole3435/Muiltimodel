#!/usr/bin/env python3
"""
A02 LLM Optimization Training Script
Fine-tune DeepSeek R1 with Unsloth and GRPO
Optimized for 4GB VRAM + Core i5 CPU
"""

import os
import sys
import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from unsloth_fine_tuner import UnslothFineTuner, UnslothConfig, create_sample_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def create_unsloth_config(yaml_config: Dict[str, Any]) -> UnslothConfig:
    """Convert YAML config to UnslothConfig"""
    
    model_config = yaml_config.get('model', {})
    lora_config = yaml_config.get('lora', {})
    training_config = yaml_config.get('training', {})
    grpo_config = yaml_config.get('grpo', {})
    output_config = yaml_config.get('output', {})
    
    return UnslothConfig(
        # Model settings
        model_name=model_config.get('name', 'unsloth/DeepSeek-R1-Distill-Llama-8B'),
        max_seq_length=model_config.get('max_seq_length', 2048),
        load_in_4bit=model_config.get('load_in_4bit', True),
        load_in_8bit=model_config.get('load_in_8bit', False),
        
        # LoRA settings
        lora_r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 16),
        lora_dropout=lora_config.get('dropout', 0.0),
        bias=lora_config.get('bias', 'none'),
        use_gradient_checkpointing=lora_config.get('use_gradient_checkpointing', 'unsloth'),
        random_state=lora_config.get('random_state', 3407),
        use_rslora=lora_config.get('use_rslora', False),
        
        # Training settings
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        warmup_steps=training_config.get('warmup_steps', 5),
        max_steps=training_config.get('max_steps', 60),
        learning_rate=training_config.get('learning_rate', 2e-4),
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', True),
        logging_steps=training_config.get('logging_steps', 1),
        optim=training_config.get('optim', 'adamw_8bit'),
        weight_decay=training_config.get('weight_decay', 0.01),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
        seed=training_config.get('seed', 3407),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        
        # GRPO settings
        use_grpo=grpo_config.get('enabled', True),
        grpo_beta=grpo_config.get('beta', 0.1),
        grpo_gamma=grpo_config.get('gamma', 0.99),
        grpo_group_size=grpo_config.get('group_size', 4),
        grpo_temperature=grpo_config.get('temperature', 1.0),
        
        # Output settings
        output_dir=output_config.get('dir', './results'),
        save_steps=output_config.get('save_steps', 20),
        save_total_limit=output_config.get('save_total_limit', 2),
    )

def prepare_dataset(dataset_path: str, dataset_config: Dict[str, Any], tokenizer) -> str:
    """Prepare dataset for training"""
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found: {dataset_path}")
        logger.info("Creating sample dataset for demonstration...")
        
        # Create sample dataset
        sample_path = os.path.join(os.path.dirname(dataset_path), "sample_dataset.json")
        create_sample_dataset(sample_path, tokenizer)
        return sample_path
    
    return dataset_path

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="A02 LLM Fine-tuning with Unsloth")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/unsloth_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/training_dataset.json",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        help="Maximum training steps (overrides config)"
    )
    parser.add_argument(
        "--use-grpo", 
        action="store_true", 
        help="Enable GRPO training"
    )
    parser.add_argument(
        "--no-grpo", 
        action="store_true", 
        help="Disable GRPO training"
    )
    parser.add_argument(
        "--export-gguf", 
        action="store_true", 
        help="Export model to GGUF format"
    )
    parser.add_argument(
        "--test-inference", 
        action="store_true", 
        help="Test inference after training"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_path = args.config
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return 1
        
        yaml_config = load_config(config_path)
        unsloth_config = create_unsloth_config(yaml_config)
        
        # Override config with command line arguments
        if args.output_dir:
            unsloth_config.output_dir = args.output_dir
        if args.max_steps:
            unsloth_config.max_steps = args.max_steps
        if args.use_grpo:
            unsloth_config.use_grpo = True
        if args.no_grpo:
            unsloth_config.use_grpo = False
        
        # Prepare dataset
        dataset_config = yaml_config.get("dataset", {})
        dataset_path = prepare_dataset(args.dataset, dataset_config, fine_tuner.tokenizer)        
        logger.info("Starting A02 LLM Fine-tuning...")
        logger.info(f"Model: {unsloth_config.model_name}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Output: {unsloth_config.output_dir}")
        logger.info(f"GRPO: {unsloth_config.use_grpo}")
        logger.info(f"Max steps: {unsloth_config.max_steps}")
        
        # Create fine-tuner
        fine_tuner = UnslothFineTuner(unsloth_config)
        
        # Start fine-tuning
        result = fine_tuner.fine_tune(
            dataset_path=dataset_path,
            dataset_type=dataset_config.get('format', 'json'),
            use_grpo=unsloth_config.use_grpo
        )
        
        if result.success:
            logger.info("✅ Fine-tuning completed successfully!")
            logger.info(f"📁 Model saved to: {result.model_path}")
            logger.info(f"⏱️  Training time: {result.training_time:.2f} seconds")
            logger.info(f"📉 Final loss: {result.final_loss:.4f}")
            logger.info(f"💾 Memory usage: {result.memory_usage}")
            
            # Save training results
            results_file = os.path.join(unsloth_config.output_dir, "training_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    'success': result.success,
                    'model_path': result.model_path,
                    'training_time': result.training_time,
                    'final_loss': result.final_loss,
                    'memory_usage': result.memory_usage,
                    'metrics': result.metrics,
                    'config': unsloth_config.__dict__
                }, f, indent=2)
            
            logger.info(f"📊 Results saved to: {results_file}")
            
            # Test inference
            if args.test_inference:
                logger.info("🧪 Testing inference...")
                inference_config = yaml_config.get('inference', {})
                test_prompt = inference_config.get('test_prompt', 'What is artificial intelligence?')
                max_new_tokens = inference_config.get('max_new_tokens', 128)
                
                response = fine_tuner.test_inference(
                    model_path=result.model_path,
                    test_prompt=test_prompt,
                    max_new_tokens=max_new_tokens
                )
                
                if response:
                    logger.info(f"🤖 Test response: {response}")
                else:
                    logger.warning("❌ Inference test failed")
            
            # Export to GGUF
            if args.export_gguf:
                logger.info("📦 Exporting to GGUF format...")
                export_config = yaml_config.get('export', {})
                quantization = export_config.get('gguf_quantization', 'q4_k_m')
                
                gguf_path = os.path.join(unsloth_config.output_dir, f"model_{quantization}.gguf")
                
                if fine_tuner.export_to_gguf(result.model_path, gguf_path, quantization):
                    logger.info(f"✅ Model exported to GGUF: {gguf_path}")
                else:
                    logger.warning("❌ GGUF export failed")
            
            return 0
            
        else:
            logger.error(f"❌ Fine-tuning failed: {result.error}")
            logger.error(f"⏱️  Time elapsed: {result.training_time:.2f} seconds")
            logger.error(f"💾 Memory usage: {result.memory_usage}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

