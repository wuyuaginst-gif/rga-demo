import os
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

# --- 配置参数 ---
# 确保 Ollama 服务正在运行，并且 Qwen2:7b 模型已拉取
OLLAMA_MODEL = "qwen2:7b" 
VECTOR_DB_PATH = "./chroma_db_requirements"
COLLECTION_NAME = "hubei_nongxin_requirements"
BASE_URL = "http://localhost:11434" # Ollama 默认地址

# --- 1. 模拟历史需求数据 ---
# 模拟您从历史 Excel 或系统导出的数据
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

# --- 2. 构建知识库 (Embedding & Store) ---

def build_knowledge_base():
    """将历史需求数据向量化并存储到 ChromaDB"""
    
    print("🚀 步骤 1: 初始化 Ollama Embeddings 模型...")
    # 使用 Ollama 的 API 作为嵌入模型 (默认使用 'llama2'，但性能取决于模型)
    # 在实际应用中，推荐使用专门的本地中文 Embedding 模型
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=BASE_URL)

    # 将 Python Dict 转换为 LangChain Document 格式
    documents = []
    for req in HISTORICAL_REQUIREMENTS:
        # 使用标题+内容作为 Document 的 page_content，元数据存储详细信息
        content = f"需求标题: {req['title']}\n需求内容: {req['content']}"
        doc = Document(page_content=content, metadata=req)
        documents.append(doc)

    print(f"📖 步骤 2: 正在创建/加载向量数据库到路径: {VECTOR_DB_PATH}")
    # 创建 ChromaDB 向量存储，并导入 Document
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    vectorstore.persist()
    print("✅ 知识库构建完成，共计 %d 条需求。" % len(documents))
    return vectorstore

# --- 3. 智能查重与分析 ---

def smart_deduplication_analysis(vectorstore: Chroma, new_requirement: str):
    """
    接收新需求，进行向量搜索和 LLM 分析。
    :param new_requirement: 新需求的标题和内容
    """
    print("\n🔍 步骤 3: 正在对新需求进行智能查重分析...")
    
    # 初始化本地 LLM
    llm = Ollama(model=OLLAMA_MODEL, base_url=BASE_URL)

    # LangChain Prompt Template - 实现查重分析
    # 使用中文提示词，引导 LLM 进行角色扮演和结构化输出
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
    3. 详细分析和建议（例如：相似度85%，建议合并至XQ-20230101-001项目的二期进行开发）：
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 创建 RetrievalQA 链
    # retriver 会自动根据新需求在向量数据库中搜索最相似的 K 个文档
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 将所有检索到的文档塞入上下文
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), # 搜索最相似的2条
        chain_type_kwargs={"prompt": PROMPT}
    )

    # 运行查重分析链
    result = qa_chain.invoke(new_requirement)
    
    print("\n--- AI 查重分析结果 ---")
    print(result['result'])
    print("------------------------")


# --- 主程序执行逻辑 ---

if __name__ == "__main__":
    # 1. 确保知识库已构建
    vector_db = build_knowledge_base()
    
    # 2. 模拟一个新的需求提交（与 XQ-20230101-001/XQ-20230102-004 高度相似）
    NEW_REQUEST_1 = "业务部门要求在手机银行App上增加大额转账功能，并提供预约转账的选项。"
    
    smart_deduplication_analysis(vector_db, NEW_REQUEST_1)
    
    print("\n\n" + "="*50 + "\n")
    
    # 3. 模拟另一个新的需求提交（不相似）
    NEW_REQUEST_2 = "需要调整内部人力资源系统的权限配置，增加一个“临时管理员”角色。"
    
    smart_deduplication_analysis(vector_db, NEW_REQUEST_2)
