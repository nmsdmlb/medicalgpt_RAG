import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# --- 配置区 ---
DEEPSEEK_API_KEY = "sk-3d3a867afba84f16b282decc55d1c7a3"

# 1. 自动处理 PDF (确保你有 data 文件夹和 PDF 文件)
print("--- 步骤 1: 正在读取 PDF ---")
loader = DirectoryLoader('./data', glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# 2. 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. 向量化 (使用较小的模型，下载比 Ollama 快得多，仅几百MB)
print("--- 步骤 3: 正在从国内源生成索引 ---")
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from langchain_community.embeddings import HuggingFaceEmbeddings

# 使用魔搭社区的路径
model_id = 'AI-ModelScope/bge-small-zh-v1.5'
# 这行代码会自动处理下载，通常只有几百兆
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download(model_id)

embeddings = HuggingFaceEmbeddings(
    model_name=model_dir, # 直接引用下载好的本地路径
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.from_documents(texts, embeddings)
print("--- 索引创建完成！ ---")

# --- 定义问答逻辑 (使用云端 API) ---
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def ask_rag_question(query):
    # 检索本地 PDF 片段
    docs = vectorstore.similarity_search(query, k=8)   #Top-K=10
    context = "\n".join([doc.page_content for doc in docs])

    # 优化后的提示词内容
    system_instruction = """你是一个严谨的专业中医助手。请遵循以下回答准则：
    1. **直接回答**：请直接回答用户的问题，不要说“根据提供的资料”或“资料显示”等废话。
    2. **精炼准确**：回答要简洁、专业，去除所有与问题无关的信息，严禁发散式回复。
    3. **结构清晰**：使用 Markdown 列表或加粗关键术语，让答案一目了然。
    4. **诚实原则**：如果参考内容中完全没有提到相关知识，请直接告知用户“参考资料中未记载相关内容”，不要利用背景知识瞎编。"""

    user_content = f"【参考内容】：\n{context}\n\n【用户问题】：\n{query}\n\n请根据参考内容给出最精准的回答："

    # 调用 DeepSeek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1,  # 调低温度，让回答更稳定、不乱跑
        stream=False
    )
    return response.choices[0].message.content


# --- 运行循环 ---
print("\n✅ RAG 系统已启动（API 版）！请输入问题：")
while True:
    q = input("用户: ")
    if q.lower() == 'exit': break
    print("AI 正在思考...")
    print(f"AI: {ask_rag_question(q)}\n")