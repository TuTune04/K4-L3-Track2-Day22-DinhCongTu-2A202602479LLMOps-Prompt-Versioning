> **📌 Hình thức: BÀI CÁ NHÂN** — mỗi học viên tự làm và tự nộp 1 repo theo quy ước đặt tên.
> **⏰ Thời lượng:** ~3–4 giờ · **Deadline:** 23:59 ngày học lab (GMT+7)
>
> | Tài liệu | Nội dung |
> |---|---|
> | [CHECKPOINTS.md](CHECKPOINTS.md) | **Hướng dẫn làm bài từng bước**: cần làm gì, sản phẩm, cách tự kiểm tra |
> | [RUBRIC.md](RUBRIC.md) | Tiêu chí chấm điểm, điểm thưởng (tối đa +10) |
> | [SUBMISSION.md](SUBMISSION.md) | Tên repo, cấu trúc nộp bài, nơi nộp, deadline |
> | [RULES.md](RULES.md) | Quy định sử dụng AI, sao chép, nộp muộn, bảo mật API key |

# Day 22: LangSmith + Prompt Versioning

## Thông tin bài làm

- **Học viên:** Đinh Công Tú
- **MSSV:** 2A202602479
- **GitHub:** [TuTune04](https://github.com/TuTune04)
- **Repository:** [K4-L3-Track2-Day22-DinhCongTu-2A202602479LLMOps-Prompt-Versioning](https://github.com/TuTune04/K4-L3-Track2-Day22-DinhCongTu-2A202602479LLMOps-Prompt-Versioning)
- **LangSmith project:** [tutune04-day22-lab](https://smith.langchain.com/o/96b61f89-7955-48f6-97c2-694719792b85/projects/p/41895b80-0ae0-4d38-8ceb-ae4a13f3bea2)

Lần chạy thật đã được đối chiếu trên LangSmith server với 50 `rag-query` và
50 `ab-rag-query` thành công. Các A/B traces gồm 19 lượt V1 và 31 lượt V2;
retrieved contexts có mặt trong output/child run. Hai prompt thật trên Hub có
commit V1 `50acd796` và V2 `76267939`. Ảnh trong `evidence/` được chụp từ UI
LangSmith và terminal thật.

Tên repository hiện tại là tên đã được cung cấp cho bài làm, nhưng khác mẫu
trong `SUBMISSION.md` (`K4-L3-DAY22-DinhCongTu-2A202602479-LLMOpsPromptVersioning`).
Cần coach xác nhận tên này trước khi nộp; repository không được tự đổi tên trong
quá trình hoàn thiện code.

## Tổng quan

Trong lab này, bạn sẽ xây dựng một hệ thống hỏi đáp hoàn chỉnh tích hợp nhiều công nghệ AI hiện đại:

- **RAG Pipeline**: Xây dựng pipeline Retrieval-Augmented Generation sử dụng FAISS làm vector store và LangChain để kết nối các thành phần.
- **LangSmith Tracing**: Theo dõi và quan sát toàn bộ luồng xử lý của ứng dụng LLM thông qua LangSmith dashboard.
- **Prompt Hub & A/B Testing**: Quản lý phiên bản prompt trên LangSmith Prompt Hub và thực hiện A/B routing để so sánh hiệu quả giữa các phiên bản.
- **RAGAS Evaluation**: Đánh giá chất lượng hệ thống RAG theo 4 chỉ số định lượng: faithfulness, answer relevancy, context recall, context precision.
- **Guardrails AI**: Triển khai các bộ kiểm duyệt tự động để phát hiện thông tin cá nhân (PII) và sửa lỗi định dạng JSON trong đầu ra của LLM.

---

## Mục tiêu học tập

Sau khi hoàn thành lab này, bạn sẽ có thể:

- Xây dựng và triển khai RAG pipeline hoàn chỉnh với LangChain LCEL và FAISS vector store.
- Tích hợp LangSmith để theo dõi, gỡ lỗi và phân tích hiệu suất của ứng dụng LLM trong thực tế.
- Quản lý vòng đời prompt bằng LangSmith Prompt Hub và thực hiện A/B testing có kiểm soát.
- Đánh giá hệ thống RAG một cách định lượng bằng framework RAGAS với các chỉ số chuẩn công nghiệp.
- Áp dụng Guardrails AI để xây dựng validator tùy chỉnh nhằm bảo vệ đầu ra của LLM khỏi dữ liệu nhạy cảm và lỗi định dạng.

---

## Yêu cầu trước

Trước khi bắt đầu, hãy đảm bảo bạn đã có:

- **Python 3.10 trở lên** — kiểm tra bằng lệnh `python --version`
- **API key** của ít nhất một trong các nhà cung cấp LLM sau:
  - OpenAI (`OPENAI_API_KEY`)
  - Google Gemini (`GOOGLE_API_KEY`)
  - Anthropic Claude (`ANTHROPIC_API_KEY`)
  - OpenRouter (`OPENROUTER_API_KEY`)
  - Ollama (chạy local, không cần API key)
- **Tài khoản LangSmith** — đăng ký miễn phí tại [smith.langchain.com](https://smith.langchain.com) và lấy API key

---

## Cài đặt nhanh

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
cp .env.example .env
# Điền LANGCHAIN_API_KEY, OPENAI_API_KEY và giữ PROVIDER=openai.
.venv/bin/python src/config.py
```

Hướng dẫn chi tiết (tạo venv, lấy API key LangSmith, chọn provider, lưu ý cho Windows) ở **Checkpoint 0** trong [CHECKPOINTS.md](CHECKPOINTS.md).

## Chạy và kiểm tra

```bash
# Chạy riêng từng bước để dễ quan sát và chụp evidence thật.
.venv/bin/python src/run_all.py --step 1
.venv/bin/python src/run_all.py --step 2
.venv/bin/python src/run_all.py --step 3
.venv/bin/python src/run_all.py --step 4

# Hoặc chạy tuần tự cả bốn bước; runner dừng và trả exit code khác 0 khi có lỗi.
.venv/bin/python src/run_all.py

# Kiểm tra đủ evidence, report, logs, PNG và secret trước khi nộp.
.venv/bin/python src/check_submission.py

# Bộ kiểm thử offline không dùng credentials hoặc gọi API.
.venv/bin/python -m unittest discover -s tests -v
```

Bốn metric trong báo cáo RAGAS có ý nghĩa như sau:

- `faithfulness`: mức độ các khẳng định trong câu trả lời được context hỗ trợ.
- `answer_relevancy`: mức độ câu trả lời liên quan trực tiếp đến câu hỏi.
- `context_recall`: mức độ context truy xuất bao phủ thông tin trong đáp án tham chiếu.
- `context_precision`: mức độ các context liên quan được xếp trước context ít liên quan.

`answer_relevancy` dựa trên cosine similarity nên điểm từng mẫu có thể âm. Ba
metric còn lại nằm trong miền 0–1. Bài làm chỉ đạt ngưỡng khi ít nhất một phiên
bản prompt có faithfulness từ 0.8 trở lên.

---

## Cấu trúc dự án

```
Lab/
├── src/
│   ├── config.py                      # Tải .env, cấu hình providers
│   ├── utils/
│   │   ├── llm_factory.py             # Factory tạo LLM và Embeddings (5 providers)
│   │   └── data_loader.py             # Load knowledge base, chunk, build FAISS
│   ├── qa_pairs.py                    # 50 cặp câu hỏi + đáp án chuẩn
│   ├── 01_langsmith_rag_pipeline.py   # Bước 1: RAG + LangSmith tracing
│   ├── 02_prompt_hub_ab_routing.py    # Bước 2: Prompt Hub + A/B routing
│   ├── 03_ragas_evaluation.py         # Bước 3: RAGAS evaluation (~15-30 phút)
│   ├── 04_guardrails_validator.py     # Bước 4: Guardrails AI validators
│   └── run_all.py                     # Chạy tất cả các bước
├── data/
│   ├── knowledge_base.txt             # Tài liệu nguồn cho RAG
│   └── ragas_report.json              # Được tạo ra ở Bước 3
├── evidence/                          # Nộp thư mục này lên GitHub
│   ├── 01_langsmith_traces.png
│   ├── 02_prompt_hub.png
│   ├── 02_ab_routing_log.txt
│   ├── 03_ragas_scores.png
│   ├── 03_ragas_report.json
│   ├── 04_pii_demo_log.txt
│   └── 04_json_demo_log.txt
├── .env.example                        # Template biến môi trường
├── requirements.txt
├── README.md                       # Tổng quan (file này)
├── CHECKPOINTS.md                  # Hướng dẫn làm bài từng bước
├── RUBRIC.md                       # Tiêu chí chấm điểm
├── SUBMISSION.md                   # Cách nộp bài
└── RULES.md                        # Quy định làm bài
```

---

## Các nhiệm vụ

Lab được chia thành 4 nhiệm vụ, mỗi nhiệm vụ 25 điểm (tổng 100 điểm):

| Nhiệm vụ | Tên                              | Điểm | Thời gian ước tính   |
|----------|----------------------------------|------|----------------------|
| 1        | RAG Pipeline với LangSmith       | 25đ  | 25–45 phút           |
| 2        | Prompt Hub & A/B Routing         | 25đ  | 20–30 phút           |
| 3        | RAGAS Evaluation                 | 25đ  | 45–75 phút           |
| 4        | Guardrails AI Validators         | 25đ  | 20–30 phút           |

**Nhiệm vụ 1 — RAG Pipeline với LangSmith (25đ):** Xây dựng vector store từ knowledge base, tạo RAG chain, và tích hợp `@traceable` để ghi lại ít nhất 50 traces trên LangSmith dashboard.

**Nhiệm vụ 2 — Prompt Hub & A/B Routing (25đ):** Soạn 2 system prompt có ngữ nghĩa khác biệt, đẩy lên LangSmith Prompt Hub, pull về khi chạy, và định tuyến câu hỏi theo hash của `request_id`.

**Nhiệm vụ 3 — RAGAS Evaluation (25đ):** Chạy 50 cặp QA qua cả 2 phiên bản prompt, xây dựng `EvaluationDataset`, tính 4 chỉ số RAGAS, và đạt faithfulness ≥ 0.8 với ít nhất 1 phiên bản.

**Nhiệm vụ 4 — Guardrails AI Validators (25đ):** Triển khai `PIIDetector` tự động che thông tin cá nhân và `JSONFormatter` tự động sửa JSON lỗi từ đầu ra của LLM.

---

Cách làm từng nhiệm vụ: xem [CHECKPOINTS.md](CHECKPOINTS.md). Cách nộp bài: xem [SUBMISSION.md](SUBMISSION.md).

---

## Tips và lưu ý

**LangSmith tracing — đặt biến môi trường đúng thứ tự:**
Các biến `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, và `LANGCHAIN_PROJECT` phải được đặt **trước khi import bất kỳ thứ gì từ LangChain**. Nếu import trước khi đặt biến, tracing sẽ không hoạt động.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"   # Phải đặt trước
os.environ["LANGCHAIN_API_KEY"]    = "..."     # Phải đặt trước
from langchain_core.prompts import ChatPromptTemplate  # Sau đó mới import
```

**RAGAS chậm — bắt đầu sớm:**
Bước 3 sẽ mất từ 15 đến 30 phút để hoàn thành do phải gọi LLM cho mỗi sample trong bộ đánh giá. Hãy bắt đầu bước này ngay khi bước 2 xong, đặc biệt nếu bạn đang dùng model có rate limit thấp.

**Guardrails AI — `on_fail` phải truyền đúng chỗ:**
Tham số `on_fail` phải được truyền vào **constructor của validator**, không phải vào `Guard.use()`:

```python
# ĐÚNG
Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

# SAI — sẽ không hoạt động đúng
Guard().use(PIIDetector(), on_fail=OnFailAction.FIX)
```

**Lưu ý phiên bản thư viện:**
- `langchain-community` phải `< 0.4` (chạy `pip install "langchain-community<0.4"` sau khi cài `requirements.txt`): bản 0.4 làm `import ragas` lỗi `No module named 'langchain_community.chat_models.vertexai'`.
- RAGAS 0.4: `result[metric_name]` trả về **list** điểm theo từng sample → dùng `numpy.mean()`; truyền `llm=` và `embeddings=` vào `evaluate()`. Cảnh báo deprecated khi import `ragas.metrics` có thể bỏ qua.
- Guardrails 0.6.8: với `OnFailAction.FIX`, `FailResult(fix_value=...)` thay output bằng giá trị đã sửa; case hợp lệ trả `PassResult()`.

**Bảo mật — không bao giờ commit `.env`:**
Tệp `.env` chứa API key nhạy cảm. Đảm bảo `.gitignore` đã có dòng `.env` trước khi push lên GitHub. Chỉ commit tệp `.env.example` (không chứa giá trị thật). Vi phạm quy tắc này sẽ bị trừ 10 điểm tự động.

---

## Tài liệu tham khảo

| Tài liệu                    | Đường dẫn                                                          |
|-----------------------------|--------------------------------------------------------------------|
| LangSmith Docs              | https://docs.smith.langchain.com                                   |
| LangChain LCEL              | https://python.langchain.com/docs/concepts/lcel                    |
| LangSmith Prompt Hub        | https://docs.smith.langchain.com/prompt-hub                        |
| RAGAS Documentation         | https://docs.ragas.io                                              |
| Guardrails AI               | https://www.guardrailsai.com/docs                                  |
| FAISS (Facebook AI)         | https://faiss.ai                                                   |
| LangChain FAISS Integration | https://python.langchain.com/docs/integrations/vectorstores/faiss  |
