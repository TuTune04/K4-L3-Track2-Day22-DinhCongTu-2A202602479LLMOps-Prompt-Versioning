TEST_CMD: .venv/bin/python -m unittest discover -s tests -v

# Day 22 — DinhCongTu / TuTune04

Nguồn nghiệm thu: README.md, CHECKPOINTS.md, RUBRIC.md, SUBMISSION.md và RULES.md trong repo. Hoàn thành bốn phần lab bằng OpenAI, kiểm thử, thu evidence thật và push origin main. Deadline tài liệu: 23:59 ngày 08/10/2026 GMT+7 (16:59 UTC). Push GitHub không tương đương nộp hai URL trên LMS.

## Workflow và quyết định đã chốt

- Planner GPT-6.1 Sol viết plan, coder GPT-5.6 Sol thực hiện từng task, reviewer GPT-6 Luna đọc diff và kết quả kiểm thử, trả PASS/FAIL. Đây là override của người dùng đối với vai Claude trong autowf.
- Điều phối dùng model agents native theo thứ tự plan → code → test → review. Coder sửa lại khi FAIL; sau PASS, script gốc `/home/codespace/.codex/skills/autowf/scripts/auto.sh --adopt N` chạy TEST_CMD và commit trên main. Không sửa script, không tạo CLI giả. Tắt thông báo bên ngoài nếu có cấu hình. Không để adoption amend commit cũ.
- Mọi commit mới có author và committer `TuTune04 <125136358+TuTune04@users.noreply.github.com>`, không Co-authored-by. Giữ lịch sử template, không force-push. Commit từng task `Task N: <tiêu đề>`; commit plan riêng do điều phối ghi.
- Vì script dùng git add -A, worktree tại adoption chỉ chứa diff task đang xét. Bảo toàn draft task sau ở thư mục tạm ngoài repo rồi phục hồi đúng lượt. Không stage .env, .venv, logs điều phối hoặc LAB_REQUIREMENTS_LOCAL.md (đã local-exclude).
- Đã có plan commit 6f759d6 và Task 1 b7680af: audit theo plan này; có commit chưa chứng minh đạt nghiệm thu. Chỉnh sửa cần thiết dùng commit bổ sung, không rewrite lịch sử. Draft bước 1–4 là input để coder audit/sửa và nhận trách nhiệm đúng vai.
- Provider đã chọn openai. Giữ model local được cấu hình, mặc định lab gpt-4o-mini / text-embedding-3-small. Không tự đổi provider khi lỗi. .env.example chỉ có placeholders; không xuất credentials vào hội thoại/log/Git, chỉ kiểm tra tên biến và trạng thái hợp lệ.

## Cách nghiệm thu

TEST_CMD là offline test mỗi task: không credentials thật, không gọi API/tracing/telemetry mạng, không tạo evidence online. Fake model/embedding/Hub/evaluator chỉ dùng để kiểm tra logic; cô lập biến môi trường giữa tests.

Task code có thể review/commit khi offline đạt, nhưng phải ghi online chưa xác minh. Config OK không chứng minh quyền API, quota, Hub hay traces đến server. Hoàn thành lab đòi chạy API thật và đầy đủ evidence ở Task 5. Thiếu key, lỗi auth/quota hoặc không có phiên UI để chụp ảnh là blocker cụ thể; tiếp tục phần độc lập. Không dùng log mẫu, điểm tự điền, fake tests hoặc dashboard tự dựng làm evidence.

## Task 1: Audit môi trường và cấu hình hiện có

Phạm vi: requirements.txt, requirements.lock.txt, src/config.py, tests/test_config.py và template môi trường nếu cần. Coder audit b7680af, reviewer xét diff gốc và kết quả kiểm mới. Giữ Python 3.12 .venv và dependencies tương thích LangChain 0.3, langchain-community<0.4, RAGAS 0.4, Guardrails đang cài; chỉ đổi khi có lỗi được kiểm chứng.

Nghiệm thu:

- `.venv/bin/python -m pip check` sạch; import được LangChain/FAISS, LangSmith, RAGAS và Guardrails; lock phản ánh bộ phiên bản đã kiểm chứng.
- Config nạp tracing trước import LangChain; config CLI exit 0 khi đủ cấu hình, nonzero khi provider sai, key thiếu/placeholder hoặc tracing tắt. OpenAI yêu cầu OpenAI + LangSmith keys; Anthropic/OpenRouter thêm OpenAI key embeddings. Giữ providers tùy chọn của template.
- Tests dùng keys giả, không in key thật; .env và hướng dẫn local không track. Kiểm tra placeholder có whitespace/case và code lỗi CLI khi phù hợp.
- Nếu audit PASS, ghi Task 1 đạt offline; không gọi adopt 1 để amend/đổi tên commit đã có. Nếu FAIL, coder sửa, reviewer kiểm lại rồi commit bổ sung về phía trước.

## Task 2: RAG, Prompt Hub và A/B routing

Phạm vi: src/01_langsmith_rag_pipeline.py, src/02_prompt_hub_ab_routing.py, src/prompts.py, helpers cần thiết và tests/test_rag.py. Giữ draft task 3/4 ngoài worktree lượt này.

Triển khai: knowledge base → chunks 500/overlap 50 → embeddings → FAISS; retriever k=3 → LCEL context/question → prompt → LLM → StrOutputParser. build_rag_chain trả chain/retriever; traceable rag-query cho đủ 50 SAMPLE_QUESTIONS, lỗi làm bước thất bại.

Prompt chung có tên tutune04-dinhcongtu-rag-prompt-v1/v2: V1 ngắn 2–4 câu; V2 phân tích có cấu trúc 3–5 câu; cả hai chỉ dựa context, nói thiếu thông tin khi cần và giữ context/question variables. Push hai prompt, pull từ Hub khi chạy, không fallback local để báo thành công. Chỉ chấp nhận conflict do prompt không đổi; lỗi quyền/mạng khác fail. Prompt pull thiếu biến bắt buộc fail.

Routing MD5(request_id)%2 trả tên prompt tất định. Chạy 50 câu với request ID/version/question/answer, traceable ab-rag-query giữ retrieved contexts trong output hoặc run con. Sinh evidence/02_ab_routing_log.txt từ lần chạy thật thành công, flush tracing nhưng vẫn xác minh server tại Task 5.

Nghiệm thu offline: FAISS/LCEL dùng doubles chứng minh ba docs và câu hỏi tới model; hai prompt khác ngữ nghĩa và đủ biến; cùng ID luôn cùng version, 50 ID nhận cả hai version; mock Hub xác minh hai push/pull, template sai và lỗi Hub không bị bỏ qua. Không sinh evidence từ tests.

Nghiệm thu online tại Task 5: >=50 rag-query có question/context/answer, hai prompt thật trên Hub, >=50 ab-rag-query và log thực có hai nhãn; tổng ít nhất 100 traces bước 1/2 thuộc project/lần chạy của mình.

## Task 3: RAGAS đủ 100 outputs và report kiểm chứng được

Phạm vi: src/03_ragas_evaluation.py, tests evaluation và helpers chung cần thiết.

Chạy đủ 50 QA_PAIRS qua mỗi V1/V2 với cùng prompt đã publish tại Task 2; không đổi reference, giảm mẫu hay bỏ câu lỗi. Giữ contexts list[str]; SingleTurnSample gồm user_input, response, retrieved_contexts, reference; tạo EvaluationDataset đúng API. Tính đủ faithfulness, answer_relevancy, context_recall, context_precision bằng evaluator LLM/embeddings tương thích bản đã cài; coder kiểm chứng adapter/signature thực.

Mỗi metric phải có 50 điểm/version. Từ chối thiếu/None/NaN/Infinity thay vì lọc rồi tính trung bình. Kiểm tra miền metric phù hợp: answer_relevancy dựa cosine có thể âm. Report strict JSON allow_nan=False ghi cả V1/V2, số mẫu, đủ 100 output/reference/contexts, timestamp, provider/model, project và prompt snapshot/tên. Giữ điểm từng mẫu hoặc evaluator output đủ đối chiếu việc tổng hợp. data/ragas_report.json và evidence/03_ragas_report.json identical; evidence report chỉ sinh từ đánh giá thật.

In bảng bốn metric. Nếu cả hai faithfulness<0.8, giữ kết quả thật, exit nonzero và sửa retrieval/prompt rồi rerun đủ mẫu. Nếu đổi prompt, publish lại để Hub/report khớp. Không tự sửa điểm hoặc hứa bonus trước khi đo.

Nghiệm thu offline: mapping bốn trường/list contexts; đủ 50 outputs/version bằng doubles; evaluator nhận bốn metric và model/embeddings; aggregation rejects thiếu/nonfinite; strict JSON và bản sao identical trong temp directory; thấp threshold làm bước thất bại.

Nghiệm thu online tại Task 5: 50 QA/version, bốn metric hữu hạn, ít nhất một faithfulness>=0.8, report thật và ảnh terminal bảng điểm thật.

## Task 4: Guardrails tùy chỉnh và demo thật

Phạm vi: src/04_guardrails_validator.py, tests validators, hai demo logs sinh qua Guardrails thật; task này độc lập keys OpenAI/LangSmith.

Tự viết @register_validator cho PIIDetector và JSONFormatter; không dùng validator Hub. PII regex đủ email/phone/SSN/card, xử lý nhiều PII và overlap. OnFailAction.FIX ở constructor; có sửa trả FailResult(fix_value=...), sạch trả PassResult. JSON sửa fences/nháy đơn/trailing comma, bảo toàn apostrophe/commas trong string; không sửa được trả JSON có error, JSON hợp lệ giữ nguyên.

Demo Guard.validate thật với >=6 PII cases (bốn loại, nhiều PII, sạch) và >=5 JSON cases (valid, fences, single quotes, trailing comma, invalid); dữ liệu PII giả. Lưu stdout thật thành evidence/04_pii_demo_log.txt và evidence/04_json_demo_log.txt.

Nghiệm thu: assertions trên validated_output Guard thật chứng minh PII bị che hết, sạch không đổi, JSON parse được/giữ ý nghĩa string, invalid có error. Kiểm tra overlap, nhiều PII, apostrophe và commas trong strings. Log đủ case, tái sinh bằng code được review; không viết tay output giả.

## Task 5: Tích hợp, evidence online, kiểm tra nộp và push main

Phạm vi: src/run_all.py, src/check_submission.py, tests runner/checker, README hướng dẫn cá nhân, evidence/README.md và evidence thật. Không sửa quy định để làm nhẹ tiêu chí.

Nghiệm thu offline:

- Runner giữ --step 1..4 và mặc định bốn bước; propagate nonzero khi config/import/API/evaluation/validator lỗi, dừng tại bước thất bại, không báo tất cả hoàn thành. Kiểm thử exit code của runner bằng module doubles.
- Checker default kiểm đủ bảy evidence files không rỗng/PNG đúng định dạng, strict JSON có V1/V2/bốn metric/50 mẫu/threshold khớp dữ liệu; A/B log 50 request/cả hai nhãn, Guardrails logs đủ case. PNG hợp lệ chưa chứng minh dashboard thật: reviewer phải đối chiếu trực tiếp.
- .env/hướng dẫn local không track; secret scan chỉ báo file/loại lỗi, không in giá trị match. Checker fail nếu evidence thiếu; fixtures ở temp directory, không tạo artifacts giả trong evidence/.
- TEST_CMD, pip check và import smoke pass. Hướng dẫn chạy và ý nghĩa metric rõ; ghi blockers online nếu còn.

Nghiệm thu hoàn thành lab:

1. Chạy config và bước 1/2 bằng OpenAI/LangSmith thật. Xác minh trên server >=50 rag-query và >=50 ab-rag-query thành công thuộc lần chạy của mình; mở trace thấy question, retrieved contexts, answer. Chụp UI LangSmith thật vào 01_langsmith_traces.png; Hub thật thấy hai prompt vào 02_prompt_hub.png. Không dựng dashboard thay UI.
2. Chạy bước 3 đủ 100 outputs/evaluator thật, xác minh report/threshold/provenance; chụp terminal thật bảng bốn metric vào 03_ragas_scores.png. Không có phiên chụp ảnh thực thì báo thiếu, không thay bằng ảnh tự vẽ.
3. Sinh logs bước 4 thật; đối chiếu đủ bảy file SUBMISSION và report hai bản identical. evidence/README.md phân tích điểm đo được, traces đã xác minh, URL project/Hub và limitations cụ thể.
4. Reviewer Luna review diff + kết quả thật PASS; adopt Task 5, kiểm author/committer/không coauthor của commits mới, push origin main không force; xác minh remote main đúng local HEAD và worktree sạch.
5. Báo URL repo và LangSmith để user nộp LMS. Tên repo hiện tại khác chuẩn SUBMISSION `K4-L3-DAY22-DinhCongTu-2A202602479-LLMOpsPromptVersioning`: báo cần coach xác nhận hoặc đổi tên trước nộp, không tự rename ngoài phạm vi push main. Xác minh public nếu có công cụ; không khẳng định đã nộp LMS hoặc project public khi chưa xác minh.

Nếu online bị chặn, vẫn review/push phần code được ủy quyền và ghi trạng thái trung thực bằng commit riêng khi cần. Task 5 giữ trạng thái chưa đủ bằng chứng online; không dùng adoption/commit Task 5 để ngụ ý lab đã đạt các tiêu chí còn thiếu.
