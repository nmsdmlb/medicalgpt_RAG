import os
import json
import pandas as pd
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness
from ragas.metrics import AnswerRelevancy
from ragas.metrics import ContextRecall
from ragas.metrics import ContextPrecision
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from modelscope.hub.snapshot_download import snapshot_download


faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy()
context_recall = ContextRecall()
context_precision = ContextPrecision()


# ==================== 1. 配置区 (保持和你 start_rag.py 一致) ====================
DEEPSEEK_API_KEY = "sk-3d3a867afba84f16b282decc55d1c7a3"

print("--- 正在初始化 RAG 环境 (加载 PDF 和 向量库) ---")
# 加载 PDF
data_path = r'F:/TCM_RAG_Project/RAGproject/data'

loader = DirectoryLoader(
    data_path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# 切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 下载并加载 Embedding 模型
model_id = 'AI-ModelScope/bge-small-zh-v1.5'
model_dir = snapshot_download(model_id)
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={'device': 'cpu'}
)

# 创建向量库
vectorstore = FAISS.from_documents(texts, embeddings)
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

print("--- RAG 环境就绪 ---")


# ==================== 2. 定义评估专用的问答函数 ====================
def get_answer_and_docs_for_eval(query):
    # 1. 检索原文内容
    docs = vectorstore.similarity_search(query, k=8)
    # Ragas 需要列表格式的原文
    contexts = [doc.page_content for doc in docs]
    context_str = "\n".join(contexts)

    # 2. 调用 DeepSeek 获取答案 (优化了 Prompt 和 Temperature)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个精准的中医问答机器人。你的任务是：\n"
                    "1. 仅根据参考内容回答问题。\n"
                    "2. **直接给出答案**，严禁任何前缀（如：根据资料显示、好的、关于您的问题...）。\n"
                    "3. 如果问题是问‘是什么’，你就只回答‘是什么’，不要扩展‘为什么要这样’或其背景知识。\n"
                    "4. 不要提及参考资料中的编号或无关术语。\n"
                    "5. 你的回答必须短小精悍，直击痛点。"
                )
            },
            {
                "role": "user",
                "content": f"参考内容：\n{context_str}\n\n问题：{query}\n\n请直接回答："
            }
        ],
        temperature=0.0,  # 降到极限，保证最稳定的输出
        stream=False
    )

    answer = response.choices[0].message.content
    return answer, contexts
# 这是没有 RAG”的情况：直接问 AI，不给资料
def get_answer_no_rag(query):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个中医助手。请根据你的知识点回答问题。"},
            {"role": "user", "content": query}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content



# ==================== 3. 运行评估逻辑 ====================
def run_eval():
    # A. 加载18 个手动编写的测试题
    json_path = r'F:/TCM_RAG_Project/RAGproject/data/test_question.json'
    with open(json_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    questions, ground_truths = [], []
    answers_rag, answers_no_rag, contexts_list = [], [], []

    print(f"🚀 开始测试对比评估 (共 {len(test_cases)} 道题)...")
    for i, case in enumerate(test_cases):
        q = case['question']
        gt = case['ground_truth']
        print(f"[{i + 1}/{len(test_cases)}] 正在获取回答: {q[:15]}...")

        ans_rag, ctxs = get_answer_and_docs_for_eval(q)
        ans_no_rag = get_answer_no_rag(q)

        questions.append(q)
        ground_truths.append(gt)
        answers_rag.append(ans_rag)
        answers_no_rag.append(ans_no_rag)
        contexts_list.append(ctxs)

    # B. 准备数据集
    ds_rag = Dataset.from_dict(
        {"question": questions, "answer": answers_rag, "contexts": contexts_list, "ground_truth": ground_truths})
    ds_no_rag = Dataset.from_dict(
        {"question": questions, "answer": answers_no_rag, "contexts": contexts_list, "ground_truth": ground_truths})

    # C. 配置裁判模型
    evaluator_llm = ChatOpenAI(model='deepseek-chat', openai_api_key=DEEPSEEK_API_KEY,
                               openai_api_base='https://api.deepseek.com')

    # D. 开始算分
    print("\n⚖️ 正在为 [中医 RAG 系统] 打分...")
    res_rag = evaluate(ds_rag, metrics=[faithfulness, answer_relevancy, context_recall], llm=evaluator_llm,
                       embeddings=embeddings)

    print("⚖️ 正在为 [纯大模型 No-RAG] 打分...")
    res_no_rag = evaluate(ds_no_rag, metrics=[faithfulness, answer_relevancy], llm=evaluator_llm, embeddings=embeddings)

    # --- 修复报错的关键：强制转换为平均分 ---
    def to_float(val):
        """处理 Ragas 可能返回 list 的情况，确保返回的是单个平均分"""
        if isinstance(val, (list, tuple)):
            import numpy as np
            # 如果列表全是空的，返回 0.0 而不是报错
            if not val or np.all(np.isnan(val)):
                return 0.0
            return float(np.nanmean(val))
        return float(val) if val is not None else 0.0

    # E. 打印对比报告
    print("\n" + "=" * 50)
    print("🚀 中医 RAG 性能提升对比报告")
    print("=" * 50)

    rag_f = to_float(res_rag['faithfulness'])
    rag_r = to_float(res_rag['answer_relevancy'])
    no_rag_f = to_float(res_no_rag['faithfulness'])
    no_rag_r = to_float(res_no_rag['answer_relevancy'])

    comparison_data = {
        "评估维度": ["忠实度 (Faithfulness)", "回答相关性 (Relevancy)"],
        "纯大模型 (Baseline)": [no_rag_f, no_rag_r],
        "中医 RAG 系统": [rag_f, rag_r]
    }

    df_compare = pd.DataFrame(comparison_data)

    # 安全计算提升率，防止除以零
    def calc_lift(row):
        base = row["纯大模型 (Baseline)"]
        if base == 0: return "+100.00%"
        lift = (row["中医 RAG 系统"] - base) / base
        return f"{lift:+.2%}"

    df_compare["提升效果"] = df_compare.apply(calc_lift, axis=1)

    print(df_compare.to_markdown(index=False))
    print("-" * 50)
    print(f"RAG 系统特有指标 - 召回率 (Context Recall): {to_float(res_rag['context_recall']):.4f}")
    print("=" * 50)

    # 导出
    df_rag = res_rag.to_pandas()
    df_rag.to_csv("中医RAG详细评估报告.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    run_eval()