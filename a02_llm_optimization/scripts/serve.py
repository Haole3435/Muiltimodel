#!/usr/bin/env python3
"""
A02 LLM Optimization Serving Script
Serve fine-tuned models with llama.cpp
Optimized for 4GB VRAM + Core i5 CPU
"""

import os
import sys
import argparse
import yaml
import logging
import signal
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llama_cpp_server import LlamaCppServer, LlamaCppConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def create_server_config(yaml_config: dict, args: argparse.Namespace) -> LlamaCppConfig:
    """Create server configuration"""
    
    # Get hardware config
    hardware_config = yaml_config.get('hardware', {})
    
    # Determine optimal settings based on hardware
    gpu_memory_gb = hardware_config.get('gpu_memory_gb', 4)
    cpu_cores = hardware_config.get('cpu_cores', 4)
    
    # Calculate optimal GPU layers based on VRAM
    if gpu_memory_gb >= 8:
        n_gpu_layers = 35  # Full model on GPU
    elif gpu_memory_gb >= 6:
        n_gpu_layers = 28  # Most layers on GPU
    elif gpu_memory_gb >= 4:
        n_gpu_layers = 20  # Partial GPU acceleration
    else:
        n_gpu_layers = 0   # CPU only
    
    # Calculate optimal batch size
    if gpu_memory_gb >= 6:
        n_batch = 1024
    elif gpu_memory_gb >= 4:
        n_batch = 512
    else:
        n_batch = 256
    
    # Calculate context size
    if gpu_memory_gb >= 8:
        n_ctx = 4096
    elif gpu_memory_gb >= 6:
        n_ctx = 2048
    else:
        n_ctx = 1024
    
    config = LlamaCppConfig(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=cpu_cores,
        n_gpu_layers=n_gpu_layers,
        use_mmap=True,
        use_mlock=False,
        numa=False,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=512,
        timeout=300.0,
        max_concurrent_requests=2 if gpu_memory_gb < 6 else 4,
        enable_metrics=True,
        log_requests=True
    )
    
    # Override with command line arguments
    if args.ctx_size:
        config.n_ctx = args.ctx_size
    if args.batch_size:
        config.n_batch = args.batch_size
    if args.gpu_layers is not None:
        config.n_gpu_layers = args.gpu_layers
    if args.threads:
        config.n_threads = args.threads
    
    return config

def find_model_file(model_dir: str) -> str:
    """Find GGUF model file in directory"""
    model_path = Path(model_dir)
    
    # Look for GGUF files
    gguf_files = list(model_path.glob("*.gguf"))
    
    if gguf_files:
        # Prefer q4_k_m quantization
        for gguf_file in gguf_files:
            if "q4_k_m" in gguf_file.name.lower():
                return str(gguf_file)
        
        # Return first GGUF file found
        return str(gguf_files[0])
    
    # Look for other model formats
    model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
    
    if model_files:
        logger.warning("Found non-GGUF model files. Consider converting to GGUF for better performance.")
        return str(model_files[0])
    
    raise FileNotFoundError(f"No model files found in {model_dir}")

def main():
    """Main serving function"""
    parser = argparse.ArgumentParser(description="A02 LLM Serving with llama.cpp")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/unsloth_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        help="Path to GGUF model file or directory"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind server"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind server"
    )
    parser.add_argument(
        "--ctx-size", 
        type=int, 
        help="Context size (default: auto-detect based on VRAM)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Batch size (default: auto-detect based on VRAM)"
    )
    parser.add_argument(
        "--gpu-layers", 
        type=int, 
        help="Number of GPU layers (default: auto-detect based on VRAM)"
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        help="Number of CPU threads (default: auto-detect)"
    )
    parser.add_argument(
        "--cpu-only", 
        action="store_true", 
        help="Force CPU-only inference"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config_path = args.config
        if os.path.exists(config_path):
            yaml_config = load_config(config_path)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            yaml_config = {}
        
        # Determine model path
        if args.model_path:
            model_path = args.model_path
        else:
            # Try to find model in default locations
            default_paths = [
                "./results/final_model",
                "./models",
                "../models",
                "../../models"
            ]
            
            model_path = None
            for path in default_paths:
                if os.path.exists(path):
                    try:
                        model_path = find_model_file(path)
                        break
                    except FileNotFoundError:
                        continue
            
            if not model_path:
                logger.error("No model found. Please specify --model-path")
                return 1
        
        # Handle directory vs file path
        if os.path.isdir(model_path):
            model_path = find_model_file(model_path)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return 1
        
        logger.info(f"Using model: {model_path}")
        
        # Force CPU-only if requested
        if args.cpu_only:
            args.gpu_layers = 0
        
        # Create server configuration
        server_config = create_server_config(yaml_config, args)
        
        logger.info("Starting A02 LLM Optimization Server...")
        logger.info(f"🤖 Model: {server_config.model_path}")
        logger.info(f"🌐 Host: {server_config.host}:{server_config.port}")
        logger.info(f"🧠 Context size: {server_config.n_ctx}")
        logger.info(f"📦 Batch size: {server_config.n_batch}")
        logger.info(f"🔧 CPU threads: {server_config.n_threads}")
        logger.info(f"🎮 GPU layers: {server_config.n_gpu_layers}")
        logger.info(f"🚀 Max concurrent requests: {server_config.max_concurrent_requests}")
        
        # Create and start server
        server = LlamaCppServer(server_config)
        
        # Handle graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            server.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        server.start()
        
    except KeyboardInterrupt:
        logger.info("⚠️  Server interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Server failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

