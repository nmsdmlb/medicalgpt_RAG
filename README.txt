#MedicalGPT-RAG 医疗问答系统
基于大模型微调 + RAG 检索增强的可交互医疗问答助手
适配鼻炎、感冒、鼻窦炎等呼吸道疾病，支持权威回答、可溯源、低幻觉。

##项目介绍
本项目基于大模型 + LoRA 微调 + RAG 检索增强 构建垂直医疗问答系统。
针对变应性鼻炎、急性上呼吸道感染、慢性鼻窦炎、风寒 / 风热感冒等常见病症，接入权威诊疗指南，实现准确、专业、可溯源的医学问答。

# 项目架构流程（RAG + 大模型）
1. 文档加载 → PDF 读取
2. 文本切片 → Chunk 切分
3. 向量生成 → Embedding 模型
4. 向量存储 → FAISS 向量库
5. 用户提问 → 语义检索 Top-K
6. 提示词构造 → 上下文 + 问题
7. 大模型生成 → 准确、可溯源回答
8. 结果展示 → 带来源参考

##核心功能
大模型微调：基于医疗对话数据 LoRA 微调
检索增强：接入权威诊疗指南，降低模型幻觉
可视化界面：支持问答交互、参考资料展示
自动评估：对比有无 RAG 的效果差异

##技术栈
模型：LLaMA-Factory / Qwen / DeepSeek-V3 / 医疗领域小模型
检索：LangChain + 向量库
界面：Streamlit
评估：准确率、幻觉率、专业度评估

##项目亮点
完整工程化：训练 → 检索 → 界面 → 评估
可演示、可复现、低资源运行
医疗领域垂直优化，专业度高
支持溯源回答，安全性强

##快速启动
1.环境配置：
conda activate medicalgpt
pip install -r requirements.txt

2.配置API Key：
DEEPSEEK_API_KEY = "sk-3d3a867afba84f16b282decc55d1c7a3"

3.启动web:
streamlit run webapp.py

4.运行对话系统：
python start_rag.py

5.运行评估系统：
python rag_evaluator.py

