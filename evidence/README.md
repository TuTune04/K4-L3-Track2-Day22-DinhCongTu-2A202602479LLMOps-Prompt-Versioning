# Evidence — Đinh Công Tú / 2A202602479

Evidence trong thư mục này được tạo từ lần chạy thật bằng OpenAI
`gpt-4o-mini`, embeddings `text-embedding-3-small`, LangSmith project
`tutune04-day22-lab` và RAGAS 0.4.3. Không có ảnh dashboard hoặc điểm số tự
dựng.

## LangSmith và Prompt Hub

- Project: [tutune04-day22-lab](https://smith.langchain.com/o/96b61f89-7955-48f6-97c2-694719792b85/projects/p/41895b80-0ae0-4d38-8ceb-ae4a13f3bea2)
- Server đã xác minh 50 `rag-query` và 50 `ab-rag-query` thành công. Các traces
  có question, answer và retrieved contexts trong output hoặc child retriever run.
- A/B routing thật phân phối V1=19 và V2=31 cho 50 request IDs.
- Prompt V1 `tutune04-dinhcongtu-rag-prompt-v1`, commit `50acd796`.
- Prompt V2 `tutune04-dinhcongtu-rag-prompt-v2`, commit `76267939`.
- `01_langsmith_traces.png` và `02_prompt_hub.png` được chụp từ phiên Chromium
  đã đăng nhập; log A/B nằm trong `02_ab_routing_log.txt`.

## Kết quả RAGAS thật

Mỗi prompt version chạy đủ cùng 50 cặp QA. Report chứa 100 outputs với
question, answer, reference và contexts, cùng 100 dòng điểm từng mẫu. Hai bản
`data/ragas_report.json` và `evidence/03_ragas_report.json` giống nhau từng byte.

| Metric | V1 | V2 | Nhận xét |
|---|---:|---:|---|
| Faithfulness | 0.984364 | 0.927142 | V1 cao hơn; cả hai vượt 0.9 |
| Answer relevancy | 0.914042 | 0.896442 | V1 cao hơn |
| Context recall | 1.000000 | 1.000000 | Hai version cùng bao phủ reference |
| Context precision | 0.941667 | 0.945000 | V2 cao hơn nhẹ |

Cả hai prompt đều vượt ngưỡng faithfulness 0.8. V1 đạt faithfulness và answer
relevancy cao hơn trong lần đo này. Một cách giải thích có thể là yêu cầu trả lời
ngắn của V1 tạo ít khẳng định cần kiểm chứng hơn; đây là suy luận, không phải kết
luận nhân quả từ phép đo. Context precision của V2 chỉ cao hơn khoảng 0.0033. Vì
hai version dùng cùng retriever và contexts, chênh lệch nhỏ này có thể đến từ độ
biến thiên của evaluator và không chứng minh retrieval của V2 tốt hơn. Kết quả
chỉ mô tả đúng lần chạy trong report, không khái quát cho mọi dataset.

`03_ragas_scores.png` là ảnh terminal thật của bảng bốn metric sau khi đủ 50 mẫu
cho mỗi version. `03_ragas_report.json` giữ điểm từng mẫu để đối chiếu lại các
giá trị trung bình trong bảng.

## Guardrails

- `04_pii_demo_log.txt`: Guard thật che email, phone, SSN, credit card, nhiều PII
  trong một input và overlap; clean input được giữ nguyên.
- `04_json_demo_log.txt`: Guard thật giữ JSON hợp lệ, sửa fences, nháy đơn và
  trailing comma, bảo toàn apostrophe/comma trong string, và trả strict JSON có
  `error` cho input không sửa được.

## Giới hạn và nộp bài

Repository đã được xác minh public tại
[TuTune04/K4-L3-Track2-Day22-DinhCongTu-2A202602479LLMOps-Prompt-Versioning](https://github.com/TuTune04/K4-L3-Track2-Day22-DinhCongTu-2A202602479LLMOps-Prompt-Versioning).
Tên này khác mẫu trong `SUBMISSION.md`; cần coach xác nhận hoặc đổi tên trước khi
nộp. Prompt Hub đang hiển thị visibility `Private`, và project sharing public
chưa được xác minh nên không yêu cầu điểm bonus URL public. Push GitHub không
đồng nghĩa hai URL đã được nộp trên LMS.
