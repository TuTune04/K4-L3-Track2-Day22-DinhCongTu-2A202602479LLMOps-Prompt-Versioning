"""Single source of truth for the two versioned, context grounded prompts."""
import config  # Initialize tracing before LangChain.
from langchain_core.prompts import ChatPromptTemplate

PROMPT_V1_NAME = "tutune04-dinhcongtu-rag-prompt-v1"
PROMPT_V2_NAME = "tutune04-dinhcongtu-rag-prompt-v2"
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn trong 2–4 câu, "
    "chỉ sử dụng thông tin có trong context. Không bổ sung kiến thức bên ngoài. "
    "Nếu context không đủ, nói rõ không có đủ thông tin để trả lời.\n\n"
    "Context:\n{context}"
)
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích tài liệu. Xác định các facts liên quan trong context, "
    "rồi trả lời có cấu trúc trong 3–5 câu: kết luận, giải thích và chi tiết hỗ trợ. "
    "Mọi khẳng định phải dựa trên context; không suy đoán, không thêm ví dụ ngoài tài liệu. "
    "Nếu context không đủ, chỉ rõ phần thông tin còn thiếu.\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([('system', SYSTEM_V1), ('human', '{question}')])
PROMPT_V2 = ChatPromptTemplate.from_messages([('system', SYSTEM_V2), ('human', '{question}')])
PROMPTS = {'v1': PROMPT_V1, 'v2': PROMPT_V2}
