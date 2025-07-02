# Báo cáo B01: Hướng dẫn về Cơ sở dữ liệu Vector

## 1. Giới thiệu về Cơ sở dữ liệu Vector

Trong kỷ nguyên của Trí tuệ Nhân tạo (AI) và Học máy (ML), đặc biệt là với sự phát triển của các mô hình ngôn ngữ lớn (LLM) và các mô hình nhúng (embedding models), nhu cầu lưu trữ và truy vấn dữ liệu phi cấu trúc một cách hiệu quả đã trở nên cấp thiết. Cơ sở dữ liệu vector (Vector Database) ra đời để giải quyết thách thức này.

**Cơ sở dữ liệu vector là gì?**

Cơ sở dữ liệu vector là một loại cơ sở dữ liệu được thiết kế đặc biệt để lưu trữ, quản lý và tìm kiếm các vector nhúng (vector embeddings) một cách hiệu quả. Các vector nhúng là các biểu diễn số học của dữ liệu (văn bản, hình ảnh, âm thanh, video, v.v.) trong một không gian đa chiều, nơi mà các mục có ý nghĩa tương tự sẽ có các vector gần nhau hơn về mặt khoảng cách.

**Tại sao cần Cơ sở dữ liệu Vector?**

Các cơ sở dữ liệu truyền thống (quan hệ, NoSQL) được tối ưu hóa cho việc tìm kiếm chính xác (exact matching) dựa trên các giá trị cụ thể hoặc các trường có cấu trúc. Tuy nhiên, chúng không hiệu quả khi cần tìm kiếm dựa trên sự tương đồng ngữ nghĩa hoặc khái niệm. Ví dụ, làm thế nào để tìm tất cả các hình ảnh 


có nội dung tương tự một hình ảnh cho trước, hoặc tìm các đoạn văn bản có ý nghĩa tương tự một câu hỏi, ngay cả khi không có từ khóa trùng khớp?

Đây chính là lúc cơ sở dữ liệu vector phát huy tác dụng. Chúng cho phép thực hiện tìm kiếm tương tự (similarity search) hoặc tìm kiếm ngữ nghĩa (semantic search) bằng cách tính toán khoảng cách (ví dụ: khoảng cách cosine, khoảng cách Euclidean) giữa các vector. Các ứng dụng phổ biến bao gồm:

-   **Hệ thống khuyến nghị:** Tìm các sản phẩm hoặc nội dung tương tự dựa trên sở thích của người dùng.
-   **Tìm kiếm ngữ nghĩa:** Tìm kiếm tài liệu hoặc thông tin dựa trên ý nghĩa thay vì từ khóa chính xác.
-   **Hệ thống hỏi đáp (Q&A) và RAG (Retrieval Augmented Generation):** Truy xuất các đoạn văn bản liên quan từ một kho tri thức để trả lời câu hỏi hoặc bổ sung cho các mô hình ngôn ngữ lớn.
-   **Phát hiện dị thường:** Tìm các điểm dữ liệu bất thường trong các tập dữ liệu lớn.
-   **Nhận dạng hình ảnh/âm thanh:** Tìm kiếm các hình ảnh hoặc đoạn âm thanh tương tự.

## 2. So sánh các tùy chọn Cơ sở dữ liệu Vector phổ biến

Thị trường cơ sở dữ liệu vector đang phát triển nhanh chóng với nhiều lựa chọn khác nhau, mỗi lựa chọn có những đặc điểm và ưu điểm riêng. Dưới đây là so sánh một số công cụ phổ biến:

### 2.1. Pinecone

-   **Loại:** Dịch vụ đám mây được quản lý hoàn toàn (fully managed cloud service).
-   **Ưu điểm:**
    -   Dễ sử dụng và triển khai, không cần quản lý cơ sở hạ tầng.
    -   Khả năng mở rộng cao, hỗ trợ hàng tỷ vector.
    -   Hiệu suất tìm kiếm tương tự nhanh chóng.
    -   Hỗ trợ lọc siêu dữ liệu (metadata filtering) mạnh mẽ.
-   **Nhược điểm:**
    -   Là dịch vụ trả phí, có thể tốn kém với quy mô lớn.
    -   Ít linh hoạt hơn trong việc tùy chỉnh so với các giải pháp tự lưu trữ.
-   **Trường hợp sử dụng:** Các ứng dụng yêu cầu khả năng mở rộng nhanh chóng, hiệu suất cao và không muốn quản lý cơ sở hạ tầng.

### 2.2. Weaviate

-   **Loại:** Mã nguồn mở, có thể tự lưu trữ (self-hosted) hoặc dịch vụ đám mây được quản lý.
-   **Ưu điểm:**
    -   Hỗ trợ tìm kiếm ngữ nghĩa và tìm kiếm đồ thị.
    -   Tích hợp sẵn các mô hình nhúng (embedding models).
    -   Khả năng mở rộng tốt.
    -   Cộng đồng lớn và tài liệu phong phú.
-   **Nhược điểm:**
    -   Yêu cầu kiến thức về quản lý cơ sở dữ liệu nếu tự lưu trữ.
-   **Trường hợp sử dụng:** Các ứng dụng cần sự linh hoạt trong triển khai, tích hợp tìm kiếm ngữ nghĩa và đồ thị, hoặc muốn kiểm soát hoàn toàn dữ liệu.

### 2.3. Chroma

-   **Loại:** Mã nguồn mở, nhẹ, dễ nhúng (embeddable).
-   **Ưu điểm:**
    -   Rất dễ cài đặt và sử dụng, có thể chạy trong bộ nhớ hoặc trên đĩa.
    -   Lý tưởng cho các dự án nhỏ và phát triển cục bộ.
    -   Tích hợp tốt với LangChain và LlamaIndex.
-   **Nhược điểm:**
    -   Khả năng mở rộng hạn chế so với Pinecone hoặc Weaviate cho các ứng dụng quy mô lớn.
    -   Chưa có các tính năng nâng cao như phân tán.
-   **Trường hợp sử dụng:** Phát triển nhanh, thử nghiệm, các ứng dụng nhỏ hoặc các trường hợp sử dụng yêu cầu cơ sở dữ liệu vector cục bộ.

### 2.4. Milvus / Zilliz

-   **Loại:** Mã nguồn mở (Milvus) và dịch vụ đám mây được quản lý (Zilliz).
-   **Ưu điểm:**
    -   Được thiết kế cho quy mô lớn, hỗ trợ hàng tỷ vector.
    -   Hiệu suất cao và khả năng chịu lỗi tốt.
    -   Hỗ trợ nhiều chỉ mục tìm kiếm (index types).
-   **Nhược điểm:**
    -   Phức tạp hơn trong việc triển khai và quản lý so với các giải pháp khác.
-   **Trường hợp sử dụng:** Các ứng dụng quy mô lớn, yêu cầu hiệu suất cao và khả năng mở rộng vượt trội.

### 2.5. Faiss (Facebook AI Similarity Search)

-   **Loại:** Thư viện mã nguồn mở, không phải là một cơ sở dữ liệu hoàn chỉnh.
-   **Ưu điểm:**
    -   Cực kỳ nhanh và hiệu quả cho tìm kiếm tương tự trên CPU và GPU.
    -   Cung cấp nhiều thuật toán chỉ mục khác nhau.
-   **Nhược điểm:**
    -   Chỉ là một thư viện, không có các tính năng của cơ sở dữ liệu như lưu trữ bền vững, quản lý siêu dữ liệu, phân tán.
    -   Yêu cầu tích hợp với một hệ thống lưu trữ khác.
-   **Trường hợp sử dụng:** Khi cần tìm kiếm tương tự tốc độ cao trong bộ nhớ, thường được sử dụng làm thành phần bên dưới của các hệ thống lớn hơn.

## 3. Phân tích chuyên sâu: Chroma

Trong phần này, chúng ta sẽ đi sâu vào Chroma, một cơ sở dữ liệu vector mã nguồn mở, nhẹ và dễ sử dụng, đặc biệt phù hợp cho các dự án phát triển nhanh và các ứng dụng cục bộ.

### 3.1. Giới thiệu về Chroma

Chroma là một cơ sở dữ liệu vector được thiết kế để đơn giản hóa việc xây dựng các ứng dụng AI với nhúng. Nó cung cấp một API dễ sử dụng để thêm, truy vấn và quản lý các nhúng. Chroma có thể chạy ở nhiều chế độ khác nhau: trong bộ nhớ (in-memory), trên đĩa (on-disk) hoặc dưới dạng máy chủ client/server.

### 3.2. Cài đặt và Khởi tạo

Cài đặt Chroma rất đơn giản thông qua pip:

```bash
pip install chromadb
```

Khởi tạo một client Chroma:

```python
import chromadb

# Chế độ trong bộ nhớ (dữ liệu sẽ mất khi chương trình kết thúc)
client = chromadb.Client()

# Chế độ trên đĩa (dữ liệu được lưu trữ bền vững)
# client = chromadb.PersistentClient(path="./chroma_db")

# Chế độ client/server (kết nối đến một máy chủ Chroma đang chạy)
# client = chromadb.HttpClient(host="localhost", port=8000)
```

### 3.3. Quản lý Collection

Trong Chroma, dữ liệu được tổ chức thành các `Collection`. Mỗi `Collection` là một tập hợp các nhúng, siêu dữ liệu và ID.

```python
# Tạo một collection mới
collection = client.create_collection(name="my_documents")

# Lấy một collection hiện có
# collection = client.get_collection(name="my_documents")

# Xóa một collection
# client.delete_collection(name="my_documents")
```

### 3.4. Thêm dữ liệu

Bạn có thể thêm dữ liệu vào collection bằng cách cung cấp ID, nhúng (tùy chọn), tài liệu (tùy chọn) và siêu dữ liệu (tùy chọn). Nếu bạn không cung cấp nhúng, Chroma sẽ tự động tạo chúng bằng một mô hình nhúng mặc định (hoặc mô hình bạn cấu hình).

```python
# Thêm tài liệu và để Chroma tự tạo nhúng
collection.add(
    documents=["This is a document about cats.", "Dogs are great pets."],
    metadatas=[
        {"source": "wikipedia", "category": "animals"},
        {"source": "blog", "category": "pets"}
    ],
    ids=["doc1", "doc2"]
)

# Thêm tài liệu với nhúng đã có
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(["This is another document."]).tolist()
# collection.add(
#     embeddings=embeddings,
#     documents=["This is another document."],
#     ids=["doc3"]
# )
```

### 3.5. Truy vấn dữ liệu (Tìm kiếm tương tự)

Để tìm kiếm các tài liệu tương tự, bạn sử dụng phương thức `query`. Bạn có thể cung cấp văn bản truy vấn (Chroma sẽ tạo nhúng cho nó) hoặc trực tiếp cung cấp vector nhúng truy vấn.

```python
results = collection.query(
    query_texts=["Tell me about animals."],
    n_results=2, # Số lượng kết quả mong muốn
    # where={"$and": [{"category": "animals"}, {"source": "wikipedia"}]} # Lọc siêu dữ liệu
)

print(results)
```

Kết quả sẽ bao gồm các tài liệu, siêu dữ liệu và khoảng cách tương tự.

### 3.6. Cập nhật và Xóa dữ liệu

Chroma cũng hỗ trợ cập nhật và xóa dữ liệu:

```python
# Cập nhật tài liệu
collection.update(
    ids=["doc1"],
    documents=["This is an updated document about cats and their habits."],
    metadatas=[
        {"source": "wikipedia", "category": "feline"}
    ]
)

# Xóa tài liệu theo ID
collection.delete(ids=["doc2"])

# Xóa tài liệu theo điều kiện siêu dữ liệu
# collection.delete(where={"category": "pets"})
```

## 4. Hướng dẫn triển khai thực tế

Để triển khai Chroma trong một ứng dụng thực tế, bạn có thể làm theo các bước sau:

1.  **Chọn chế độ triển khai:**
    -   **Phát triển/Thử nghiệm:** Sử dụng chế độ trong bộ nhớ hoặc trên đĩa để nhanh chóng khởi tạo và kiểm tra.
    -   **Sản xuất (quy mô nhỏ đến trung bình):** Sử dụng chế độ trên đĩa với một thư mục bền vững để đảm bảo dữ liệu không bị mất. Đối với quy mô lớn hơn, cân nhắc chạy Chroma dưới dạng máy chủ riêng biệt hoặc chuyển sang các giải pháp như Pinecone/Weaviate.

2.  **Tạo nhúng:**
    -   Sử dụng một mô hình nhúng phù hợp với miền dữ liệu của bạn (ví dụ: `SentenceTransformer`, OpenAI Embeddings, Cohere Embeddings). Đảm bảo rằng mô hình này được sử dụng nhất quán cho cả việc thêm dữ liệu và truy vấn.

3.  **Xử lý dữ liệu:**
    -   **Phân đoạn (Chunking):** Đối với các tài liệu lớn (ví dụ: sách, bài báo), hãy chia chúng thành các đoạn nhỏ hơn (chunks) để cải thiện độ chính xác của tìm kiếm tương tự. Mỗi đoạn nên có đủ ngữ cảnh nhưng không quá dài.
    -   **Siêu dữ liệu:** Gắn siêu dữ liệu có ý nghĩa cho mỗi đoạn (ví dụ: tiêu đề, tác giả, ngày, nguồn, loại tài liệu). Siêu dữ liệu này rất quan trọng cho việc lọc kết quả truy vấn.

4.  **Tích hợp với LLM và RAG:**
    -   Chroma thường được sử dụng như một thành phần trong kiến trúc RAG. Khi người dùng đặt câu hỏi, câu hỏi đó được chuyển đổi thành vector nhúng, sau đó được sử dụng để truy vấn Chroma. Các đoạn văn bản liên quan được truy xuất và đưa vào lời nhắc (prompt) của LLM để tạo ra câu trả lời chính xác và có ngữ cảnh.

    ```python
    # Ví dụ tích hợp RAG cơ bản với Chroma và OpenAI
    from openai import OpenAI
    # from sentence_transformers import SentenceTransformer # Nếu tự tạo nhúng

    # Khởi tạo Chroma client (ví dụ: PersistentClient)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="my_documents")

    # Giả sử collection đã có dữ liệu

    openai_client = OpenAI()

    def ask_rag(question):
        # 1. Tạo nhúng cho câu hỏi
        # query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode(question).tolist() # Nếu tự tạo nhúng

        # 2. Truy vấn Chroma để tìm các đoạn liên quan
        results = collection.query(
            query_texts=[question], # Để Chroma tự tạo nhúng cho câu hỏi
            n_results=3
        )

        # 3. Xây dựng ngữ cảnh từ các đoạn truy xuất
        context = "\n".join(results['documents'][0])

        # 4. Gửi ngữ cảnh và câu hỏi đến LLM
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content

    # Sử dụng
    # print(ask_rag("Tell me about cats."))
    ```

## 5. Các phương pháp hay nhất và cân nhắc về hiệu suất

Để tối ưu hóa hiệu suất và độ tin cậy khi sử dụng cơ sở dữ liệu vector:

-   **Chọn mô hình nhúng phù hợp:** Chất lượng của vector nhúng ảnh hưởng trực tiếp đến độ chính xác của tìm kiếm tương tự. Chọn mô hình nhúng được đào tạo trên dữ liệu tương tự với miền dữ liệu của bạn.
-   **Tối ưu hóa phân đoạn (Chunking Strategy):** Kích thước và chiến lược phân đoạn tài liệu có tác động lớn đến hiệu suất RAG. Thử nghiệm với các kích thước đoạn khác nhau để tìm ra tối ưu.
-   **Sử dụng siêu dữ liệu hiệu quả:** Tận dụng siêu dữ liệu để lọc kết quả tìm kiếm, giúp thu hẹp phạm vi và cải thiện độ chính xác.
-   **Quản lý chỉ mục (Indexing):** Đối với các cơ sở dữ liệu vector lớn, việc chọn loại chỉ mục phù hợp (ví dụ: HNSW, IVF_FLAT) là rất quan trọng để cân bằng giữa tốc độ tìm kiếm và độ chính xác.
-   **Cập nhật và đồng bộ hóa:** Đảm bảo rằng cơ sở dữ liệu vector được cập nhật thường xuyên khi dữ liệu nguồn thay đổi. Xây dựng các quy trình tự động để đồng bộ hóa.
-   **Giám sát hiệu suất:** Theo dõi các chỉ số như độ trễ truy vấn, thông lượng và việc sử dụng tài nguyên để xác định các điểm nghẽn và tối ưu hóa.
-   **Xử lý lỗi và khả năng chịu lỗi:** Triển khai cơ chế xử lý lỗi mạnh mẽ và cân nhắc các giải pháp có khả năng chịu lỗi cho môi trường sản xuất.
-   **Bảo mật:** Đảm bảo dữ liệu trong cơ sở dữ liệu vector được bảo vệ bằng các biện pháp bảo mật phù hợp (mã hóa, kiểm soát truy cập).

Bằng cách tuân thủ các phương pháp hay nhất này, bạn có thể xây dựng các ứng dụng AI mạnh mẽ và hiệu quả dựa trên cơ sở dữ liệu vector.

