TEST_CMD: .venv/bin/python -m unittest discover -s tests -v

## Tổng quan

Hoàn thiện Day 22 theo RUBRIC.md/CHECKPOINTS.md. Áp dụng workflow autowf từ TuTune04/AutoWorkFlow (37c257c): Codex đảm nhận plan, implementation và review theo yêu cầu người dùng; kiểm thử thực tế trước từng commit. Commit từng task trực tiếp main, author/committer TuTune04, không co-author. Giữ lịch sử template. File yêu cầu riêng LAB_REQUIREMENTS_LOCAL.md và .env chỉ local. Không tạo evidence giả hoặc báo điểm chưa đo. Kiểm thử offline dùng doubles và không tạo bằng chứng online. Chạy online khi có API key hợp lệ; ảnh dashboard lấy từ giao diện thật.

## Task 1: Môi trường và cấu hình

Giới hạn phiên bản thư viện tương thích code mẫu, kiểm tra provider/key/placeholder/tracing và exit code. Cài Python 3.12 venv, pip check. Nghiệm thu: config thiếu key/placeholder/unsupported provider trả thất bại; Anthropic/OpenRouter đòi key embeddings; không lộ secrets.

## Task 2: RAG, Prompt Hub và A/B

Hoàn thành FAISS/LCEL/tracing, prompt riêng TuTune04, dùng module prompt chung cho bước 2 và 3. Hub thật push/pull, không âm thầm dùng local fallback khi nộp. Routing MD5 tất định. Nghiệm thu: fake model/embeddings xác nhận retrieved context đến model, hai nhãn đều có trong 50 ID, lỗi Hub không được coi thành công.

## Task 3: RAGAS và báo cáo

100 output (50 QA/version), SingleTurnSample đúng trường, bốn metric với evaluator cấu hình. Không bỏ điểm lỗi/nonfinite; JSON strict và tự sao report vào evidence. Nghiệm thu: dataset giữ list contexts, từ chối mẫu/metric lỗi, report đúng dữ liệu; online faithfulness >=0.8 phải đo thật.

## Task 4: Guardrails và bằng chứng cục bộ

Custom validators, regex đủ 4 PII, FIX thực sự thay output. JSON repair fences/quotes/trailing commas và fallback, không làm hỏng apostrophe hoặc chuỗi có dấu phẩy. Demo ghi log thật. Nghiệm thu: assertions trên Guard.validate output cho toàn bộ demo và các edge case.

## Task 5: Kiểm tra nộp bài và tích hợp

run_all propagates exit code, checker kiểm tra 7 file, JSON/50 mẫu/metric/threshold, không track secrets. Ghi hướng dẫn chạy và tình trạng evidence thật. Nghiệm thu: toàn bộ offline tests và pip check pass; checker báo thiếu evidence đúng; push origin main và xác nhận commit identity.
