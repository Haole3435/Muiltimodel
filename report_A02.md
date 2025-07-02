# Báo cáo A02: Hướng dẫn tinh chỉnh LLM

## 1. Giới thiệu

Tinh chỉnh (fine-tuning) các mô hình ngôn ngữ lớn (LLM) là một kỹ thuật quan trọng cho phép tùy chỉnh các mô hình đã được đào tạo trước (pre-trained models) cho các tác vụ hoặc miền cụ thể. Điều này giúp cải thiện đáng kể hiệu suất của LLM trên các tập dữ liệu chuyên biệt mà không cần phải đào tạo một mô hình từ đầu. Hướng dẫn này sẽ bao gồm các phương pháp tinh chỉnh khác nhau, các cân nhắc kỹ thuật, chiến lược triển khai thực tế, tối ưu hóa hiệu suất, khắc phục sự cố và phân tích chi phí.

## 2. Các phương pháp tinh chỉnh LLM

Có nhiều phương pháp tinh chỉnh LLM, mỗi phương pháp có những ưu và nhược điểm riêng, phù hợp với các trường hợp sử dụng và tài nguyên khác nhau:

### 2.1. Tinh chỉnh toàn bộ (Full Fine-tuning)

**Mô tả:** Đây là phương pháp truyền thống, trong đó tất cả các tham số của mô hình đã được đào tạo trước đều được cập nhật trong quá trình tinh chỉnh trên tập dữ liệu mới. Điều này đòi hỏi tài nguyên tính toán và bộ nhớ đáng kể.

**Khi nào sử dụng:**
- Khi có một lượng lớn dữ liệu miền cụ thể chất lượng cao.
- Khi cần đạt được hiệu suất tối đa và mô hình cần học các mẫu mới hoàn toàn.
- Khi có đủ tài nguyên GPU và thời gian đào tạo.

**Ưu điểm:**
- Tiềm năng đạt hiệu suất cao nhất trên tác vụ mục tiêu.
- Mô hình có thể thích nghi sâu sắc với miền dữ liệu mới.

**Nhược điểm:**
- Yêu cầu tài nguyên tính toán (GPU VRAM, thời gian đào tạo) rất lớn.
- Dễ bị quá khớp (overfitting) nếu tập dữ liệu tinh chỉnh nhỏ.
- Tạo ra các mô hình lớn, khó triển khai và lưu trữ.

### 2.2. Tinh chỉnh hiệu quả tham số (Parameter-Efficient Fine-tuning - PEFT)

PEFT là một nhóm các kỹ thuật cho phép tinh chỉnh LLM mà chỉ cập nhật một phần nhỏ các tham số của mô hình, hoặc thêm các lớp nhỏ mới vào mô hình. Điều này giúp giảm đáng kể tài nguyên tính toán và bộ nhớ cần thiết.

#### 2.2.1. LoRA (Low-Rank Adaptation)

**Mô tả:** LoRA đóng băng các trọng số của mô hình đã được đào tạo trước và tiêm các ma trận hạng thấp (low-rank matrices) vào các lớp biến đổi (transformer layers). Chỉ các ma trận hạng thấp này được đào tạo trong quá trình tinh chỉnh.

**Khi nào sử dụng:**
- Khi tài nguyên tính toán hạn chế.
- Khi cần tinh chỉnh nhiều mô hình cho các tác vụ khác nhau từ một mô hình cơ sở.
- Khi cần triển khai nhanh chóng các mô hình tinh chỉnh.

**Ưu điểm:**
- Giảm đáng kể số lượng tham số có thể đào tạo, tiết kiệm bộ nhớ và thời gian đào tạo.
- Hiệu suất gần bằng tinh chỉnh toàn bộ trong nhiều trường hợp.
- Dễ dàng chuyển đổi giữa các mô hình tinh chỉnh bằng cách hoán đổi các ma trận LoRA.

**Nhược điểm:**
- Có thể không đạt được hiệu suất tối đa như tinh chỉnh toàn bộ trong một số trường hợp phức tạp.

#### 2.2.2. QLoRA (Quantized Low-Rank Adaptation)

**Mô tả:** QLoRA là một phần mở rộng của LoRA, trong đó mô hình đã được đào tạo trước được lượng tử hóa (quantized) thành 4-bit để giảm hơn nữa yêu cầu bộ nhớ. Sau đó, LoRA được áp dụng trên mô hình lượng tử hóa này.

**Khi nào sử dụng:**
- Khi tài nguyên GPU cực kỳ hạn chế (ví dụ: GPU tiêu dùng).
- Khi cần tinh chỉnh các mô hình rất lớn mà không thể vừa với bộ nhớ GPU.

**Ưu điểm:**
- Giảm đáng kể yêu cầu bộ nhớ, cho phép tinh chỉnh các mô hình lớn trên phần cứng khiêm tốn.
- Vẫn giữ được hiệu suất tốt.

**Nhược điểm:**
- Có thể có một chút suy giảm hiệu suất so với LoRA thông thường do lượng tử hóa.

#### 2.2.3. Adapters

**Mô tả:** Kỹ thuật Adapters thêm các module nhỏ (adapter layers) vào giữa các lớp của mô hình đã được đào tạo trước. Chỉ các adapter layers này được đào tạo trong quá trình tinh chỉnh.

**Khi nào sử dụng:**
- Tương tự như LoRA, khi cần tinh chỉnh hiệu quả tham số.
- Khi cần khả năng mở rộng và mô-đun hóa cao.

**Ưu điểm:**
- Giảm số lượng tham số có thể đào tạo.
- Có thể dễ dàng kết hợp nhiều adapter cho các tác vụ khác nhau.

**Nhược điểm:**
- Có thể làm tăng độ trễ suy luận (inference latency) do thêm các lớp mới.

### 2.3. Tinh chỉnh hướng dẫn (Instruction Tuning)

**Mô tả:** Tinh chỉnh hướng dẫn tập trung vào việc đào tạo LLM để tuân theo các hướng dẫn bằng văn bản một cách hiệu quả. Điều này thường được thực hiện bằng cách tinh chỉnh mô hình trên một tập dữ liệu gồm các cặp (hướng dẫn, phản hồi) đa dạng.

**Khi nào sử dụng:**
- Khi muốn mô hình trở nên hữu ích hơn và dễ điều khiển hơn thông qua các lời nhắc (prompts).
- Khi cần cải thiện khả năng tổng quát hóa của mô hình trên các tác vụ mới.

**Ưu điểm:**
- Cải thiện đáng kể khả năng tuân thủ hướng dẫn và tính hữu ích của mô hình.
- Giúp mô hình trở nên linh hoạt hơn cho nhiều tác vụ.

**Nhược điểm:**
- Yêu cầu tập dữ liệu hướng dẫn chất lượng cao và đa dạng.

### 2.4. Học tăng cường từ phản hồi của con người (Reinforcement Learning from Human Feedback - RLHF)

**Mô tả:** RLHF là một kỹ thuật tinh chỉnh sau đào tạo (post-training fine-tuning) nhằm căn chỉnh hành vi của LLM với sở thích và giá trị của con người. Nó bao gồm việc thu thập dữ liệu so sánh từ con người (xếp hạng các phản hồi của mô hình), đào tạo một mô hình phần thưởng (reward model) dựa trên dữ liệu này, và sau đó sử dụng học tăng cường để tinh chỉnh LLM dựa trên mô hình phần thưởng.

**Khi nào sử dụng:**
- Khi cần căn chỉnh mô hình với các giá trị đạo đức, an toàn hoặc sở thích cụ thể của con người.
- Khi muốn giảm thiểu các phản hồi độc hại, thiên vị hoặc không mong muốn.

**Ưu điểm:**
- Cải thiện đáng kể tính an toàn, hữu ích và phù hợp của mô hình.
- Giúp mô hình tạo ra các phản hồi tự nhiên và giống con người hơn.

**Nhược điểm:**
- Quy trình phức tạp và tốn kém, đòi hỏi thu thập dữ liệu phản hồi của con người.
- Khó khăn trong việc mở rộng quy mô.

### 2.5. Lượng tử hóa (Quantization)

**Mô tả:** Lượng tử hóa là quá trình giảm độ chính xác số học của các trọng số và kích hoạt của mô hình (ví dụ: từ 32-bit float xuống 8-bit int hoặc 4-bit int). Mặc dù không phải là một phương pháp tinh chỉnh theo nghĩa truyền thống, nó thường được sử dụng kết hợp với tinh chỉnh (đặc biệt là QLoRA) để giảm kích thước mô hình và yêu cầu bộ nhớ.

**Khi nào sử dụng:**
- Khi cần triển khai mô hình trên các thiết bị có tài nguyên hạn chế (ví dụ: thiết bị biên).
- Khi cần giảm chi phí suy luận và tăng tốc độ.

**Ưu điểm:**
- Giảm đáng kể kích thước mô hình và yêu cầu bộ nhớ.
- Tăng tốc độ suy luận.

**Nhược điểm:**
- Có thể dẫn đến suy giảm hiệu suất nhỏ nếu không được thực hiện cẩn thận.
- Yêu cầu các công cụ và thư viện chuyên biệt.



## 3. Thông số kỹ thuật và các bước triển khai

### 3.1. Thông số kỹ thuật

Khi tinh chỉnh LLM, cần xem xét các thông số kỹ thuật sau:

- **Yêu cầu dữ liệu:**
  - **Kích thước tập dữ liệu:** Kích thước tập dữ liệu tinh chỉnh ảnh hưởng trực tiếp đến hiệu suất. Tập dữ liệu lớn hơn thường dẫn đến kết quả tốt hơn, nhưng cũng đòi hỏi nhiều tài nguyên hơn.
  - **Chất lượng dữ liệu:** Dữ liệu sạch, liên quan và được định dạng tốt là rất quan trọng. Dữ liệu nhiễu hoặc không liên quan có thể làm suy giảm hiệu suất của mô hình.
  - **Định dạng dữ liệu:** Dữ liệu thường cần được định dạng thành các cặp (đầu vào, đầu ra) hoặc (lời nhắc, phản hồi) phù hợp với tác vụ tinh chỉnh.
- **Yêu cầu phần cứng:**
  - **GPU VRAM:** Yêu cầu bộ nhớ GPU (VRAM) là yếu tố hạn chế chính. Tinh chỉnh toàn bộ yêu cầu VRAM rất lớn (ví dụ: 80GB cho các mô hình lớn), trong khi PEFT (đặc biệt là QLoRA) có thể giảm xuống còn vài GB.
  - **Số lượng GPU:** Tinh chỉnh phân tán trên nhiều GPU có thể tăng tốc quá trình đào tạo.
  - **CPU và RAM:** Mặc dù GPU là quan trọng nhất, CPU và RAM cũng cần đủ để xử lý dữ liệu và quản lý quy trình đào tạo.
- **Thư viện và Framework:**
  - **Hugging Face Transformers:** Thư viện phổ biến nhất để làm việc với LLM, cung cấp các công cụ để tải, tinh chỉnh và đánh giá mô hình.
  - **PEFT (Parameter-Efficient Fine-tuning):** Thư viện của Hugging Face cung cấp các triển khai của LoRA, QLoRA và các kỹ thuật PEFT khác.
  - **PyTorch/TensorFlow:** Các framework học sâu cơ bản.

### 3.2. Các bước triển khai tinh chỉnh (ví dụ với LoRA)

Đây là các bước chung để tinh chỉnh một LLM bằng kỹ thuật LoRA:

1.  **Chuẩn bị môi trường:**
    - Cài đặt các thư viện cần thiết: `pip install transformers peft accelerate bitsandbytes`
    - Đảm bảo có GPU và driver tương thích.

2.  **Tải mô hình và tokenizer:**
    - Chọn một mô hình ngôn ngữ lớn đã được đào tạo trước (ví dụ: `meta-llama/Llama-2-7b-hf`).
    - Tải tokenizer tương ứng.

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ```

3.  **Chuẩn bị tập dữ liệu:**
    - Tải hoặc tạo tập dữ liệu tinh chỉnh của bạn. Định dạng nó thành các cặp đầu vào/đầu ra.
    - Mã hóa (tokenize) tập dữ liệu bằng tokenizer đã tải.

    ```python
    from datasets import Dataset

    # Ví dụ tập dữ liệu (thay thế bằng dữ liệu thực của bạn)
    data = {
        "prompt": ["What is the capital of France?", "Who painted the Mona Lisa?"],
        "completion": ["Paris.", "Leonardo da Vinci."]
    }
    dataset = Dataset.from_dict(data)

    def tokenize_function(examples):
        return tokenizer(examples["prompt"], text_target=examples["completion"], truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    ```

4.  **Cấu hình LoRA:**
    - Định nghĩa cấu hình LoRA, bao gồm `r` (hạng của ma trận LoRA) và `lora_alpha`.

    ```python
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output sẽ hiển thị số lượng tham số có thể đào tạo rất nhỏ so với tổng số tham số
    ```

5.  **Cấu hình và bắt đầu đào tạo:**
    - Sử dụng `TrainingArguments` và `Trainer` từ Hugging Face để cấu hình và quản lý quá trình đào tạo.

    ```python
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True, # Sử dụng mixed precision training nếu GPU hỗ trợ
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    ```

6.  **Lưu và tải mô hình tinh chỉnh:**
    - Sau khi đào tạo, bạn có thể lưu các adapter LoRA.
    - Để sử dụng mô hình, bạn có thể tải mô hình cơ sở và sau đó tải các adapter LoRA lên trên đó.

    ```python
    # Lưu adapter
    trainer.save_model("./final_lora_model")

    # Tải mô hình để suy luận
    from peft import PeftModel, PeftConfig

    # Tải cấu hình PEFT
    config = PeftConfig.from_pretrained("./final_lora_model")
    # Tải mô hình cơ sở
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    # Tải adapter lên mô hình cơ sở
    peft_model = PeftModel.from_pretrained(base_model, "./final_lora_model")

    # Sử dụng peft_model để suy luận
    ```



## 4. Tối ưu hóa hiệu suất

Để đạt được hiệu suất tốt nhất trong quá trình tinh chỉnh LLM, cần xem xét các kỹ thuật tối ưu hóa sau:

- **Kích thước batch lớn hơn:** Sử dụng kích thước batch lớn hơn có thể giúp tận dụng hiệu quả hơn tài nguyên GPU và tăng tốc độ đào tạo. Tuy nhiên, cần điều chỉnh tốc độ học (learning rate) phù hợp.
- **Mixed Precision Training (FP16/BF16):** Sử dụng kiểu dữ liệu dấu phẩy động 16-bit thay vì 32-bit giúp giảm yêu cầu bộ nhớ và tăng tốc độ tính toán trên các GPU hỗ trợ. Thư viện `accelerate` của Hugging Face hỗ trợ điều này dễ dàng.
- **Gradient Accumulation:** Khi không thể sử dụng kích thước batch lớn do hạn chế bộ nhớ, có thể sử dụng tích lũy gradient để mô phỏng kích thước batch lớn hơn bằng cách tính toán gradient trên nhiều batch nhỏ và tích lũy chúng trước khi cập nhật trọng số.
- **Distributed Training:** Phân tán quá trình đào tạo trên nhiều GPU hoặc nhiều máy chủ để tăng tốc độ đáng kể, đặc biệt với các mô hình và tập dữ liệu lớn.
- **Tối ưu hóa tốc độ học (Learning Rate Scheduling):** Sử dụng các lịch trình tốc độ học như Warmup, Cosine Annealing để điều chỉnh tốc độ học trong suốt quá trình đào tạo, giúp mô hình hội tụ tốt hơn và nhanh hơn.
- **Giảm số lượng tham số có thể đào tạo:** Các kỹ thuật PEFT như LoRA, QLoRA là cách hiệu quả nhất để giảm tài nguyên cần thiết và tăng tốc độ tinh chỉnh.

## 5. Hướng dẫn khắc phục sự cố

Trong quá trình tinh chỉnh LLM, có thể gặp phải một số vấn đề phổ biến. Dưới đây là hướng dẫn khắc phục:

- **Lỗi Out-of-Memory (OOM):**
  - **Giải pháp:** Giảm kích thước batch, sử dụng Mixed Precision Training (FP16/BF16), sử dụng Gradient Accumulation, hoặc chuyển sang các kỹ thuật PEFT như QLoRA.
- **Mô hình không hội tụ hoặc hiệu suất kém:**
  - **Giải pháp:** Kiểm tra chất lượng và kích thước tập dữ liệu tinh chỉnh. Điều chỉnh tốc độ học (learning rate) và lịch trình tốc độ học. Tăng số lượng epoch đào tạo. Kiểm tra xem có lỗi trong quá trình tiền xử lý dữ liệu không.
- **Quá khớp (Overfitting):**
  - **Giải pháp:** Giảm số lượng epoch đào tạo, tăng kích thước tập dữ liệu tinh chỉnh, sử dụng các kỹ thuật điều hòa (regularization) như dropout (nếu mô hình hỗ trợ), hoặc giảm kích thước mô hình (nếu tinh chỉnh toàn bộ).
- **Tốc độ đào tạo chậm:**
  - **Giải pháp:** Đảm bảo sử dụng GPU hiệu quả (kiểm tra `nvidia-smi`). Sử dụng Mixed Precision Training. Tối ưu hóa việc tải và tiền xử lý dữ liệu. Cân nhắc Distributed Training.
- **Lỗi cài đặt thư viện/phần mềm:**
  - **Giải pháp:** Đảm bảo tất cả các thư viện được cài đặt đúng phiên bản và tương thích với nhau. Kiểm tra driver GPU. Sử dụng môi trường ảo (virtual environment) để tránh xung đột.

## 6. Phân tích chi phí

Chi phí tinh chỉnh LLM chủ yếu đến từ tài nguyên tính toán (GPU) và chi phí thu thập/chuẩn bị dữ liệu. Dưới đây là các yếu tố cần xem xét:

- **Chi phí GPU:**
  - **Thuê GPU trên đám mây:** Các nhà cung cấp như AWS, Google Cloud, Azure, RunPod, Vast.ai cung cấp GPU theo giờ. Chi phí thay đổi tùy thuộc vào loại GPU (ví dụ: A100, H100) và khu vực.
  - **Mua GPU:** Đầu tư ban đầu lớn nhưng không có chi phí thuê theo giờ. Phù hợp cho việc sử dụng liên tục và lâu dài.
- **Chi phí dữ liệu:**
  - **Thu thập dữ liệu:** Chi phí để thu thập hoặc tạo tập dữ liệu tinh chỉnh. Có thể bao gồm chi phí nhân công để gắn nhãn hoặc làm sạch dữ liệu.
  - **Lưu trữ dữ liệu:** Chi phí lưu trữ tập dữ liệu lớn.
- **Chi phí nhân sự:** Thời gian của kỹ sư để thiết lập, chạy và giám sát quá trình tinh chỉnh.
- **Chi phí suy luận (sau tinh chỉnh):** Mặc dù không phải là chi phí tinh chỉnh trực tiếp, nhưng cần xem xét chi phí chạy mô hình đã tinh chỉnh trong môi trường sản xuất. Các mô hình nhỏ hơn hoặc lượng tử hóa có thể giảm đáng kể chi phí này.

**So sánh chi phí giữa các phương pháp:**

- **Tinh chỉnh toàn bộ:** Chi phí cao nhất về GPU và thời gian, do yêu cầu tài nguyên lớn.
- **PEFT (LoRA, QLoRA):** Chi phí thấp hơn đáng kể. QLoRA đặc biệt tiết kiệm chi phí vì nó cho phép sử dụng GPU ít mạnh hơn hoặc thuê GPU với chi phí thấp hơn.
- **Instruction Tuning/RLHF:** Ngoài chi phí GPU, còn có chi phí đáng kể cho việc thu thập dữ liệu phản hồi của con người và đào tạo mô hình phần thưởng.

Việc lựa chọn phương pháp tinh chỉnh cần cân bằng giữa hiệu suất mong muốn, tài nguyên sẵn có và ngân sách. PEFT thường là lựa chọn tối ưu cho hầu hết các trường hợp, đặc biệt là khi tài nguyên hạn chế.

