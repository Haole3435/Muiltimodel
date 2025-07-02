# Báo cáo A02: Hướng dẫn tinh chỉnh LLM nâng cao với Unsloth, llama.cpp và triển khai trên GCP

## 1. Giới thiệu

Báo cáo này cung cấp hướng dẫn toàn diện về tinh chỉnh mô hình ngôn ngữ lớn (LLM) với trọng tâm vào việc tối ưu hóa cho máy cục bộ có tài nguyên hạn chế (4GB VRAM, CPU Core i5) và triển khai trên Google Cloud Platform (GCP). Chúng ta sẽ sử dụng các công cụ tiên tiến như Unsloth và llama.cpp để tinh chỉnh mô hình DeepSeek R1 với định dạng GGUF, áp dụng phương pháp Reinforcement Learning GRPO (Group Relative Policy Optimization), và xây dựng hệ thống triển khai hoàn chỉnh với Docker, Kubernetes, và giám sát bằng Grafana và Prometheus.

## 2. Tổng quan về các công nghệ sử dụng

### 2.1. Unsloth

Unsloth là một framework mã nguồn mở được thiết kế để tăng tốc quá trình tinh chỉnh LLM lên đến 30 lần so với Flash Attention 2 (FA2) và giảm 70% sử dụng bộ nhớ. Các đặc điểm chính:

- **Tối ưu hóa bộ nhớ:** Giảm đáng kể yêu cầu VRAM, cho phép tinh chỉnh trên phần cứng khiêm tốn
- **Tăng tốc độ:** Sử dụng các kernel tối ưu hóa để tăng tốc độ đào tạo
- **Hỗ trợ đa mô hình:** Tương thích với Llama, Mistral, Qwen, DeepSeek và nhiều mô hình khác
- **Tích hợp PEFT:** Hỗ trợ LoRA, QLoRA và các kỹ thuật Parameter-Efficient Fine-tuning khác

### 2.2. llama.cpp và GGUF

llama.cpp là một thư viện C++ để chạy suy luận LLM trên CPU và GPU với hiệu suất cao. GGUF (General GGML Universal Format) là định dạng tệp được tối ưu hóa cho việc lưu trữ và tải mô hình:

- **Hiệu quả bộ nhớ:** Hỗ trợ lượng tử hóa để giảm kích thước mô hình
- **Tương thích đa nền tảng:** Chạy trên CPU, GPU, và các thiết bị biên
- **Tối ưu hóa suy luận:** Tăng tốc độ suy luận đáng kể so với các framework khác

### 2.3. DeepSeek R1

DeepSeek R1 là một mô hình ngôn ngữ lớn được thiết kế đặc biệt cho khả năng lý luận (reasoning). Mô hình này sử dụng các kỹ thuật tiên tiến để cải thiện khả năng giải quyết vấn đề phức tạp và lý luận logic.

### 2.4. GRPO (Group Relative Policy Optimization)

GRPO là một thuật toán Reinforcement Learning được phát triển bởi DeepSeek, cải tiến từ PPO (Proximal Policy Optimization) với những ưu điểm:

- **Đơn giản hóa:** Loại bỏ nhu cầu về mô hình critic riêng biệt
- **Hiệu quả tính toán:** Giảm chi phí tính toán so với RLHF truyền thống
- **Cải thiện lý luận:** Đặc biệt hiệu quả cho các tác vụ đòi hỏi khả năng lý luận

## 3. Thiết lập môi trường và cài đặt

### 3.1. Yêu cầu hệ thống

Để tinh chỉnh DeepSeek R1 trên máy cục bộ với 4GB VRAM và CPU Core i5, chúng ta cần:

- **GPU:** NVIDIA GPU với ít nhất 4GB VRAM (hỗ trợ CUDA)
- **CPU:** Intel Core i5 hoặc tương đương
- **RAM:** Ít nhất 16GB RAM hệ thống
- **Lưu trữ:** 50GB dung lượng trống cho mô hình và dữ liệu
- **Hệ điều hành:** Ubuntu 20.04+ hoặc Windows 10/11 với WSL2

### 3.2. Cài đặt dependencies

```bash
# Cài đặt Python và pip
sudo apt update
sudo apt install python3.10 python3-pip git

# Cài đặt CUDA toolkit (nếu chưa có)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Cài đặt Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes

# Cài đặt llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUBLAS=1  # Để hỗ trợ CUDA
```

## 4. Tinh chỉnh DeepSeek R1 với Unsloth

### 4.1. Tải và chuẩn bị mô hình

```python
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import os

# Cấu hình cho máy có 4GB VRAM
max_seq_length = 2048  # Giảm độ dài sequence để tiết kiệm bộ nhớ
dtype = None  # Auto detection
load_in_4bit = True  # Sử dụng 4-bit quantization

# Tải mô hình DeepSeek R1
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="hf_...", # Sử dụng nếu cần token Hugging Face
)

# Cấu hình LoRA cho tinh chỉnh hiệu quả
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank của LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

### 4.2. Chuẩn bị dữ liệu đào tạo

```python
# Tải dataset cho reasoning tasks
dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train[:5000]")

# Template cho DeepSeek R1
chat_template = get_chat_template(
    tokenizer,
    chat_template="deepseek",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        # Chuyển đổi conversation thành format phù hợp
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

# Áp dụng formatting
dataset = dataset.map(formatting_prompts_func, batched=True)
```

### 4.3. Cấu hình đào tạo với GRPO

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Cấu hình training arguments tối ưu cho 4GB VRAM
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Batch size nhỏ để tiết kiệm bộ nhớ
    gradient_accumulation_steps=4,  # Tích lũy gradient để mô phỏng batch size lớn hơn
    warmup_steps=5,
    max_steps=100,  # Số bước đào tạo (điều chỉnh theo nhu cầu)
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",  # Optimizer 8-bit để tiết kiệm bộ nhớ
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    dataloader_num_workers=0,  # Giảm số worker để tiết kiệm RAM
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

# Khởi tạo trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=training_args,
)

# Bắt đầu đào tạo
trainer_stats = trainer.train()
```

### 4.4. Áp dụng GRPO Reinforcement Learning

```python
from unsloth import PatchDPOTrainer
from trl import DPOTrainer

# Chuẩn bị dataset cho GRPO
def create_grpo_dataset(base_dataset, model, tokenizer):
    """
    Tạo dataset cho GRPO bằng cách sinh multiple responses
    và đánh giá chúng theo nhóm
    """
    grpo_data = []
    
    for example in base_dataset:
        prompt = example["prompt"]
        
        # Sinh multiple responses
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=4,  # Sinh 4 responses
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        
        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Đánh giá responses (có thể sử dụng reward model hoặc heuristics)
        scores = evaluate_responses(responses, prompt)
        
        # Tạo preference pairs
        best_idx = scores.index(max(scores))
        worst_idx = scores.index(min(scores))
        
        grpo_data.append({
            "prompt": prompt,
            "chosen": responses[best_idx],
            "rejected": responses[worst_idx]
        })
    
    return grpo_data

def evaluate_responses(responses, prompt):
    """
    Đánh giá chất lượng responses
    Có thể sử dụng reward model hoặc heuristics đơn giản
    """
    scores = []
    for response in responses:
        # Ví dụ heuristic đơn giản: độ dài và từ khóa
        score = len(response.split()) * 0.1
        if "reasoning" in response.lower():
            score += 1.0
        if "step" in response.lower():
            score += 0.5
        scores.append(score)
    return scores

# Tạo GRPO dataset
grpo_dataset = create_grpo_dataset(dataset, model, tokenizer)

# Cấu hình DPO trainer cho GRPO
PatchDPOTrainer()
dpo_trainer = DPOTrainer(
    model,
    model_ref=None,  # Sử dụng model hiện tại làm reference
    args=training_args,
    beta=0.1,
    train_dataset=grpo_dataset,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=max_seq_length // 2,
)

# Đào tạo với GRPO
dpo_trainer.train()
```

## 5. Chuyển đổi sang định dạng GGUF

### 5.1. Lưu mô hình đã tinh chỉnh

```python
# Lưu mô hình và tokenizer
model.save_pretrained("deepseek_r1_finetuned")
tokenizer.save_pretrained("deepseek_r1_finetuned")

# Merge LoRA weights với base model
model.save_pretrained_merged("deepseek_r1_merged", tokenizer, save_method="merged_16bit")
```

### 5.2. Chuyển đổi sang GGUF

```bash
# Chuyển đổi sang GGUF format
cd llama.cpp
python convert.py ../deepseek_r1_merged --outdir ../deepseek_r1_gguf --outtype f16

# Lượng tử hóa mô hình để giảm kích thước
./quantize ../deepseek_r1_gguf/ggml-model-f16.gguf ../deepseek_r1_gguf/ggml-model-q4_0.gguf q4_0
./quantize ../deepseek_r1_gguf/ggml-model-f16.gguf ../deepseek_r1_gguf/ggml-model-q8_0.gguf q8_0
```

### 5.3. Kiểm tra mô hình GGUF

```bash
# Kiểm tra mô hình
./main -m ../deepseek_r1_gguf/ggml-model-q4_0.gguf -p "Solve this math problem: What is 15 * 23?" -n 256
```


## 6. Triển khai với Docker

### 6.1. Dockerfile cho mô hình đã tinh chỉnh

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu20.04

# Cài đặt dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Cài đặt llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
WORKDIR /opt/llama.cpp
RUN make LLAMA_CUBLAS=1

# Copy mô hình và code
COPY deepseek_r1_gguf/ /opt/models/
COPY src/ /opt/src/
COPY config/ /opt/config/

# Thiết lập working directory
WORKDIR /opt/src

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python3", "api_server.py"]
```

### 6.2. Requirements.txt

```txt
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
torch==2.1.0
transformers==4.36.0
accelerate==0.24.1
bitsandbytes==0.41.3
datasets==2.14.6
trl==0.7.4
peft==0.6.2
unsloth @ git+https://github.com/unslothai/unsloth.git
numpy==1.24.3
requests==2.31.0
prometheus-client==0.19.0
psutil==5.9.6
```

### 6.3. API Server

```python
# src/api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import time
import psutil
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('llm_requests_total', 'Total LLM requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('llm_request_duration_seconds', 'LLM request duration')
INFERENCE_DURATION = Histogram('llm_inference_duration_seconds', 'LLM inference duration')

app = FastAPI(title="DeepSeek R1 Fine-tuned API", version="1.0.0")

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

class InferenceResponse(BaseModel):
    response: str
    inference_time: float
    tokens_generated: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_memory_used: float
    cpu_usage: float

# Cấu hình mô hình
MODEL_PATH = "/opt/models/ggml-model-q4_0.gguf"
LLAMA_CPP_PATH = "/opt/llama.cpp/main"

def run_inference(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> tuple:
    """Chạy inference với llama.cpp"""
    start_time = time.time()
    
    cmd = [
        LLAMA_CPP_PATH,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--no-display-prompt"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise Exception(f"Inference failed: {result.stderr}")
        
        inference_time = time.time() - start_time
        response = result.stdout.strip()
        tokens_generated = len(response.split())
        
        return response, inference_time, tokens_generated
    
    except subprocess.TimeoutExpired:
        raise Exception("Inference timeout")
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Endpoint cho inference"""
    REQUEST_COUNT.labels(method="POST", endpoint="/inference").inc()
    
    with REQUEST_DURATION.time():
        try:
            with INFERENCE_DURATION.time():
                response, inference_time, tokens_generated = run_inference(
                    request.prompt, 
                    request.max_tokens, 
                    request.temperature
                )
            
            return InferenceResponse(
                response=response,
                inference_time=inference_time,
                tokens_generated=tokens_generated
            )
        
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Kiểm tra mô hình có tồn tại không
        model_loaded = os.path.exists(MODEL_PATH)
        
        # Lấy thông tin hệ thống
        cpu_usage = psutil.cpu_percent()
        
        # GPU memory (nếu có)
        gpu_memory_used = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used = info.used / info.total * 100
        except:
            pass
        
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            gpu_memory_used=gpu_memory_used,
            cpu_usage=cpu_usage
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DeepSeek R1 Fine-tuned API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.4. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  deepseek-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./logs:/opt/logs
      - ./models:/opt/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

## 7. Triển khai với Kubernetes

### 7.1. Namespace và ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: deepseek-llm
  labels:
    name: deepseek-llm

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: deepseek-config
  namespace: deepseek-llm
data:
  model_path: "/opt/models/ggml-model-q4_0.gguf"
  max_tokens: "256"
  temperature: "0.7"
  log_level: "INFO"
```

### 7.2. Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek-api
  namespace: deepseek-llm
  labels:
    app: deepseek-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deepseek-api
  template:
    metadata:
      labels:
        app: deepseek-api
    spec:
      containers:
      - name: deepseek-api
        image: deepseek-r1:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          valueFrom:
            configMapKeyRef:
              name: deepseek-config
              key: model_path
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: deepseek-config
              key: log_level
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: model-storage
          mountPath: /opt/models
        - name: logs
          mountPath: /opt/logs
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: logs
        emptyDir: {}
      nodeSelector:
        accelerator: nvidia-tesla-t4
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### 7.3. Service và Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: deepseek-service
  namespace: deepseek-llm
  labels:
    app: deepseek-api
spec:
  selector:
    app: deepseek-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: deepseek-ingress
  namespace: deepseek-llm
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - deepseek-api.yourdomain.com
    secretName: deepseek-tls
  rules:
  - host: deepseek-api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: deepseek-service
            port:
              number: 80
```

### 7.4. Persistent Volume

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: deepseek-llm
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: ssd-retain
```

### 7.5. HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: deepseek-hpa
  namespace: deepseek-llm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deepseek-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```


## 8. Giám sát với Prometheus và Grafana

### 8.1. Cấu hình Prometheus

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'deepseek-api'
    static_configs:
      - targets: ['deepseek-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
      action: keep
      regex: default;kubernetes;https

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
    - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - action: labelmap
      regex: __meta_kubernetes_node_label_(.+)

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__
    - action: labelmap
      regex: __meta_kubernetes_pod_label_(.+)
    - source_labels: [__meta_kubernetes_namespace]
      action: replace
      target_label: kubernetes_namespace
    - source_labels: [__meta_kubernetes_pod_name]
      action: replace
      target_label: kubernetes_pod_name
```

### 8.2. Alert Rules

```yaml
# config/alert_rules.yml
groups:
- name: deepseek_alerts
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is above 80% for more than 5 minutes"

  - alert: HighMemoryUsage
    expr: memory_usage > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is above 85% for more than 5 minutes"

  - alert: HighInferenceLatency
    expr: histogram_quantile(0.95, llm_inference_duration_seconds) > 10
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High inference latency"
      description: "95th percentile inference latency is above 10 seconds"

  - alert: APIDown
    expr: up{job="deepseek-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "DeepSeek API is down"
      description: "DeepSeek API has been down for more than 1 minute"

  - alert: HighErrorRate
    expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.1
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "High error rate"
      description: "Error rate is above 10% for more than 3 minutes"
```

### 8.3. Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "DeepSeek R1 LLM Monitoring",
    "tags": ["llm", "deepseek", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, llm_inference_duration_seconds)",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, llm_inference_duration_seconds)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, llm_inference_duration_seconds)",
            "legendFormat": "99th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "memory_usage",
            "legendFormat": "Memory Usage %"
          },
          {
            "expr": "gpu_memory_usage",
            "legendFormat": "GPU Memory Usage %"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(llm_requests_total{status=\"error\"}[5m]) / rate(llm_requests_total[5m]) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "valueName": "current",
        "format": "percent",
        "thresholds": "5,10",
        "colorBackground": true,
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 0,
          "y": 16
        }
      },
      {
        "id": 5,
        "title": "Active Connections",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_connections",
            "legendFormat": "Connections"
          }
        ],
        "valueName": "current",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 6,
          "y": 16
        }
      },
      {
        "id": 6,
        "title": "Tokens Generated/sec",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(tokens_generated_total[5m])",
            "legendFormat": "Tokens/sec"
          }
        ],
        "valueName": "current",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 12,
          "y": 16
        }
      },
      {
        "id": 7,
        "title": "Model Load Status",
        "type": "singlestat",
        "targets": [
          {
            "expr": "model_loaded",
            "legendFormat": "Model Loaded"
          }
        ],
        "valueName": "current",
        "valueMaps": [
          {
            "value": "1",
            "text": "Loaded"
          },
          {
            "value": "0",
            "text": "Not Loaded"
          }
        ],
        "colorBackground": true,
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 18,
          "y": 16
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

### 8.4. Kubernetes Monitoring với Prometheus Operator

```yaml
# k8s/prometheus-operator.yaml
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: prometheus
  namespace: deepseek-llm
spec:
  serviceAccountName: prometheus
  serviceMonitorSelector:
    matchLabels:
      team: deepseek
  ruleSelector:
    matchLabels:
      team: deepseek
      prometheus: deepseek
  resources:
    requests:
      memory: 400Mi
  storage:
    volumeClaimTemplate:
      spec:
        storageClassName: ssd-retain
        resources:
          requests:
            storage: 50Gi

---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: deepseek-api-monitor
  namespace: deepseek-llm
  labels:
    team: deepseek
spec:
  selector:
    matchLabels:
      app: deepseek-api
  endpoints:
  - port: http
    path: /metrics
    interval: 10s

---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: deepseek-rules
  namespace: deepseek-llm
  labels:
    team: deepseek
    prometheus: deepseek
spec:
  groups:
  - name: deepseek.rules
    rules:
    - alert: DeepSeekHighLatency
      expr: histogram_quantile(0.95, rate(llm_inference_duration_seconds_bucket[5m])) > 10
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "DeepSeek API high latency"
        description: "95th percentile latency is {{ $value }}s"
```

## 9. Triển khai trên Google Cloud Platform (GCP)

### 9.1. Chuẩn bị GCP Environment

```bash
# Cài đặt Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Tạo project mới
gcloud projects create deepseek-llm-project --name="DeepSeek LLM Project"
gcloud config set project deepseek-llm-project

# Enable APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
```

### 9.2. Tạo GKE Cluster với GPU

```bash
# Tạo GKE cluster với GPU nodes
gcloud container clusters create deepseek-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=5 \
    --enable-autorepair \
    --enable-autoupgrade

# Tạo GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster=deepseek-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --num-nodes=1 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=3

# Cài đặt NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### 9.3. Terraform Configuration cho GCP

```hcl
# terraform/main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

# GKE Cluster
resource "google_container_cluster" "deepseek_cluster" {
  name     = "deepseek-cluster"
  location = var.zone

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
  }
}

# CPU Node Pool
resource "google_container_node_pool" "cpu_nodes" {
  name       = "cpu-pool"
  location   = var.zone
  cluster    = google_container_cluster.deepseek_cluster.name
  node_count = 2

  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"

    service_account = google_service_account.gke_service_account.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      env = "production"
    }

    tags = ["gke-node", "deepseek-cluster"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# GPU Node Pool
resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-pool"
  location   = var.zone
  cluster    = google_container_cluster.deepseek_cluster.name
  node_count = 1

  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }

    service_account = google_service_account.gke_service_account.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      env = "production"
      accelerator = "nvidia-tesla-t4"
    }

    tags = ["gke-node", "deepseek-cluster", "gpu"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  autoscaling {
    min_node_count = 0
    max_node_count = 3
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "deepseek-vpc"
  auto_create_subnetworks = "false"
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "deepseek-subnet"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = "10.10.0.0/24"
}

# Service Account
resource "google_service_account" "gke_service_account" {
  account_id   = "gke-service-account"
  display_name = "GKE Service Account"
}

# Cloud Storage for model storage
resource "google_storage_bucket" "model_bucket" {
  name     = "${var.project_id}-deepseek-models"
  location = var.region

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud SQL for metadata storage
resource "google_sql_database_instance" "metadata_db" {
  name             = "deepseek-metadata"
  database_version = "POSTGRES_13"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    backup_configuration {
      enabled = true
    }

    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        value = "0.0.0.0/0"
        name  = "all"
      }
    }
  }

  deletion_protection = false
}

resource "google_sql_database" "metadata" {
  name     = "metadata"
  instance = google_sql_database_instance.metadata_db.name
}

# Outputs
output "kubernetes_cluster_name" {
  value = google_container_cluster.deepseek_cluster.name
}

output "kubernetes_cluster_host" {
  value = google_container_cluster.deepseek_cluster.endpoint
}

output "model_bucket_name" {
  value = google_storage_bucket.model_bucket.name
}
```

### 9.4. CI/CD Pipeline với Cloud Build

```yaml
# cloudbuild.yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/deepseek-api:$COMMIT_SHA', '.']

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/deepseek-api:$COMMIT_SHA']

  # Deploy to GKE
  - name: 'gcr.io/cloud-builders/gke-deploy'
    args:
    - run
    - --filename=k8s/
    - --image=gcr.io/$PROJECT_ID/deepseek-api:$COMMIT_SHA
    - --location=us-central1-a
    - --cluster=deepseek-cluster
    - --namespace=deepseek-llm

  # Run tests
  - name: 'gcr.io/cloud-builders/kubectl'
    args: ['apply', '-f', 'k8s/test-job.yaml']
    env:
    - 'CLOUDSDK_COMPUTE_ZONE=us-central1-a'
    - 'CLOUDSDK_CONTAINER_CLUSTER=deepseek-cluster'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'N1_HIGHCPU_8'

timeout: '1200s'
```

## 10. Kết luận và Best Practices

### 10.1. Tóm tắt kiến trúc

Hệ thống tinh chỉnh và triển khai DeepSeek R1 đã được thiết kế với các thành phần chính:

1. **Tinh chỉnh cục bộ:** Sử dụng Unsloth và GRPO để tối ưu hóa cho máy có 4GB VRAM
2. **Containerization:** Docker để đóng gói ứng dụng và dependencies
3. **Orchestration:** Kubernetes cho quản lý container và auto-scaling
4. **Monitoring:** Prometheus và Grafana cho giám sát real-time
5. **Cloud Deployment:** GCP với GKE cho khả năng mở rộng và độ tin cậy

### 10.2. Best Practices

- **Tối ưu hóa bộ nhớ:** Sử dụng quantization và PEFT để giảm yêu cầu VRAM
- **Monitoring toàn diện:** Theo dõi metrics về hiệu suất, tài nguyên và lỗi
- **Auto-scaling:** Cấu hình HPA để tự động điều chỉnh số lượng pod
- **Security:** Sử dụng RBAC, network policies và image scanning
- **Cost optimization:** Sử dụng preemptible instances và auto-scaling để giảm chi phí

### 10.3. Roadmap phát triển

1. **Phase 1:** Triển khai cơ bản với monitoring
2. **Phase 2:** Tích hợp A/B testing và model versioning
3. **Phase 3:** Thêm caching layer và load balancing nâng cao
4. **Phase 4:** Multi-region deployment và disaster recovery

