import streamlit as st
import os
from modelscope import snapshot_download
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader

# --- 1. 配置页面 ---
st.set_page_config(page_title="中医智能助手 - RAG版", layout="wide")
st.title("🌿 中医医疗知识库助手 (RAG)")

# --- 2. 侧边栏配置 (API Key 和 文件管理) ---
with st.sidebar:
    st.header("系统设置")
    api_key = st.text_input("输入你的 API Key", type="password")
    base_url = "https://api.deepseek.com"  # 或者你使用的其它地址

    st.divider()
    st.markdown("### 项目状态")
    if api_key:
        st.success("API 已就绪")
    else:
        st.warning("请在上方填入 API Key 以开始运行")


# --- 3. 初始化 Embedding 和 向量库 (缓存处理) ---
@st.cache_resource
def init_rag_system():
    # 1. 下载并加载本地 Embedding 模型
    model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", revision="master")
    embeddings = HuggingFaceEmbeddings(model_name=model_dir)

    # 2. 加载 PDF (假设你的文档在 data 文件夹下)
    pdf_path = "F:/TCM_RAG_Project/RAGproject/data"   # 👈 这里改成你实际的PDF路径
    if not os.path.exists(pdf_path):
        st.error(f"找不到文件: {pdf_path}")
        return None

    loader = PyPDFDirectoryLoader(pdf_path)
    docs = loader.load()

    # 3. 文档切片
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # 4. 创建向量库
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store


# 初始化系统
if api_key:
    with st.spinner("系统正在加载医疗文库，请稍候..."):
        vector_store = init_rag_system()
else:
    vector_store = None

# --- 4. 聊天界面逻辑 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            st.caption(f"📍 来源参考: {message['source']}")

# 用户输入
if prompt := st.chat_input("请输入你的问诊问题或药物查询..."):
    if not api_key:
        st.error("请先在左侧输入 API Key")
    else:
        # 展示用户问题
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 检索与生成
        with st.chat_message("assistant"):
            # A. 检索阶段
            related_docs = vector_store.similarity_search(prompt, k=3)
            context = "\n".join([doc.page_content for doc in related_docs])

            # 提取来源信息 (文件名 + 页码)
            sources = list(
                set([f"《{os.path.basename(doc.metadata['source'])}》第 {doc.metadata['page'] + 1} 页" for doc in
                     related_docs]))
            source_text = " | ".join(sources)

            # B. 生成阶段
            client = OpenAI(api_key=api_key, base_url=base_url)

            # 构建 Prompt
            system_prompt = f"你是一个专业的中医助手。请根据以下参考资料回答问题：\n\n{context}\n\n如果不确定，请说明。回答要专业准确。"

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            answer = response.choices[0].message.content
            st.markdown(answer)
            st.caption(f"📍 来源参考: {source_text}")

            # 保存记录
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source": source_text
            })