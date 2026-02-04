import os
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import gradio as gr

# --- 配置参数 ---
# vLLM 服务地址 (OpenAI 兼容 API)
VLLM_API_BASE = "http://localhost:8090/v1"
VLLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Embedding 模型配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

VECTOR_DB_PATH = "./chroma_db_requirements"
COLLECTION_NAME = "hubei_nongxin_requirements"

# --- 模拟历史需求数据 ---
HISTORICAL_REQUIREMENTS: List[Dict[str, str]] = [
    {
        "id": "XQ-20230101-001",
        "title": "网银渠道新增转账汇款功能，支持大额和定时交易。",
        "content": "业务部要求在个人网银中增加每日超过50万元的转账交易功能，并提供预设时间转账。",
        "solution": "已开发完成，使用了第三方安全模块进行加密。",
        "status": "已上线"
    },
    {
        "id": "XQ-20230315-002",
        "title": "柜面系统优化，提高存取款效率。",
        "content": "柜面操作人员反馈，存取款流程步骤过多，希望整合到单页面，减少点击。",
        "solution": "优化了前端界面，将多个步骤合并，减少了响应时间。",
        "status": "已上线"
    },
    {
        "id": "XQ-20240520-003",
        "title": "移动App支持生物识别登录和快捷支付。",
        "content": "科技部建议在App中引入指纹和人脸识别，并支持小额免密支付，提高用户体验。",
        "solution": "正在开发中，预计2025年Q1投产。",
        "status": "开发中"
    },
    {
        "id": "XQ-20230102-004",
        "title": "网银渠道优化，支持大额汇款和预约功能。",
        "content": "零售业务部提出，客户需要预约特定日期和金额的汇款，但现有网银系统不支持。",
        "solution": "已在2023年Q2实现，功能与001号需求类似，但侧重预约。",
        "status": "已上线"
    }
]

# 全局变量
vectorstore = None
qa_chain = None

def build_knowledge_base():
    """构建知识库"""
    global vectorstore, qa_chain

    print("🚀 初始化 Embeddings 模型...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    documents = []
    for req in HISTORICAL_REQUIREMENTS:
        content = f"需求标题: {req['title']}\n需求内容: {req['content']}"
        doc = Document(page_content=content, metadata=req)
        documents.append(doc)

    print(f"📖 创建向量数据库...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    print("✅ 知识库构建完成！")

    # 初始化 LLM
    print("🚀 初始化 LLM...")
    llm = ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_API_BASE,
        api_key="EMPTY",
        temperature=0.1
    )

    # 创建 QA 链
    prompt_template = """
    你是一位资深的银行科技部门项目经理。你的任务是审核新的业务需求，并判断它是否与历史需求重复或高度相似。

    【历史相似需求参考】：
    {context}

    【当前提交的新需求】：
    {question}

    请根据历史参考，给出你的专业建议，判断是否为重复需求。
    请以清晰的分点格式输出：
    1. 查重结论（是/否重复）：
    2. 相似度最高的历史需求ID和标题：
    3. 详细分析和建议：
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("✅ QA 链初始化完成！")
    return vectorstore

def analyze_requirement(requirement_title: str, requirement_content: str) -> str:
    """分析需求是否重复"""
    global qa_chain

    if qa_chain is None:
        return "❌ 知识库未初始化，请先点击上方按钮构建知识库"

    if not requirement_title.strip() or not requirement_content.strip():
        return "⚠️ 请输入需求标题和内容"

    new_requirement = f"标题: {requirement_title}\n内容: {requirement_content}"

    try:
        result = qa_chain.invoke(new_requirement)
        return result['result']
    except Exception as e:
        return f"❌ 错误: {str(e)}"

# 启动时构建知识库
print("\n" + "="*50)
print("🚀 启动智能需求查重系统...")
print("="*50 + "\n")
build_knowledge_base()
print("\n" + "="*50)
print("✅ 系统就绪！请访问下方 Gradio 界面")
print("="*50 + "\n")

# --- Gradio 界面 ---
with gr.Blocks(title="智能需求查重系统", css="""
    .gradio-container {max-width: 900px !important;}
    .primary-btn {background-color: #4CAF50 !important;}
""") as demo:
    gr.Markdown("# 🏦 智能需求查重系统")
    gr.Markdown("基于 AI 的银行需求重复检测助手 - 向量检索 + 大语言模型分析")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 录入新需求")
            title_input = gr.Textbox(
                label="需求标题",
                placeholder="例如：手机银行增加大额转账功能",
                lines=2
            )
            content_input = gr.Textbox(
                label="需求内容",
                placeholder="详细描述需求的具体内容和业务场景...",
                lines=5
            )
            submit_btn = gr.Button("🔍 开始查重分析", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 查重分析结果")
            output = gr.Textbox(
                label="分析结果",
                lines=15,
                show_label=True
            )

    # 示例需求
    gr.Markdown("### 💡 示例需求")
    gr.Examples(
        examples=[
            ["业务部门要求在手机银行App上增加大额转账功能，并提供预约转账的选项。", "业务部门要求在手机银行App上增加大额转账功能，并提供预约转账的选项。"],
            ["需要调整内部人力资源系统的权限配置，增加一个临时管理员角色。", "需要调整内部人力资源系统的权限配置，增加一个临时管理员角色，用于临时授权。"],
        ],
        inputs=[title_input, content_input]
    )

    submit_btn.click(
        fn=analyze_requirement,
        inputs=[title_input, content_input],
        outputs=output
    )

    # 快捷输入
    title_input.submit(
        fn=analyze_requirement,
        inputs=[title_input, content_input],
        outputs=output
    )
    content_input.submit(
        fn=analyze_requirement,
        inputs=[title_input, content_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
