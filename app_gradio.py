"""
智能需求查重系统 - 客户演示版 V2.0
基于向量检索 + 大语言模型的银行需求智能查重助手
新增功能：需求分类、实施建议生成
"""

import os
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
import gradio as gr

# ==================== 配置参数 ====================
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8090/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
VECTOR_DB_PATH = "./chroma_db_requirements"
COLLECTION_NAME = "hubei_nongxin_requirements"

# ==================== 分类体系定义 ====================
CATEGORY_SYSTEM = {
    "开发类": {
        "description": "涉及系统功能开发、代码实现的需求",
        "subcategories": {
            "需求开发": "新业务功能的分析和设计工作",
            "功能开发": "新增或修改系统功能模块",
            "接口开发": "内外部系统接口开发",
            "报表开发": "报表设计、开发、优化",
            "界面开发": "前端页面、交互界面开发"
        }
    },
    "服务类": {
        "description": "涉及业务服务、客户服务、系统服务相关需求",
        "subcategories": {
            "业务服务": "业务流程、服务功能相关",
            "客户服务": "客户咨询、投诉、服务体验",
            "系统服务": "系统级服务、后台服务",
            "接口服务": "API 服务、接口调用"
        }
    },
    "运维类": {
        "description": "涉及系统运维、保障类需求",
        "subcategories": {
            "系统运维": "系统日常运维、监控",
            "数据运维": "数据清洗、迁移、治理",
            "安全运维": "安全加固、漏洞修复",
            "性能运维": "性能优化、调优"
        }
    },
    "提数类": {
        "description": "涉及数据提取、分析类需求",
        "subcategories": {
            "数据提取": "按需求提取特定数据",
            "报表生成": "定期/临时报表生成",
            "数据分析": "业务数据分析、挖掘",
            "数据核对": "数据核对、校验"
        }
    }
}

# ==================== 历史需求数据 ====================
HISTORICAL_REQUIREMENTS: List[Dict] = [
    {
        "id": "XQ-20230101-001",
        "title": "网银渠道新增转账汇款功能，支持大额和定时交易",
        "content": "业务部要求在个人网银中增加每日超过50万元的转账交易功能，并提供预设时间转账。需支持跨行转账、定时转账、批量转账等功能。",
        "solution": "已开发完成，使用了第三方安全模块进行加密。与核心系统对接，实现实时到账。",
        "status": "已上线",
        "dept": "零售银行部",
        "priority": "高"
    },
    {
        "id": "XQ-20230315-002",
        "title": "柜面系统优化，提高存取款效率",
        "content": "柜面操作人员反馈，存取款流程步骤过多，希望整合到单页面，减少点击次数，提高办理效率。",
        "solution": "优化了前端界面，将多个步骤合并，减少了响应时间。引入OCR识别自动填充客户信息。",
        "status": "已上线",
        "dept": "运营管理部",
        "priority": "中"
    },
    {
        "id": "XQ-20240520-003",
        "title": "移动App支持生物识别登录和快捷支付",
        "content": "科技部建议在App中引入指纹和人脸识别，并支持小额免密支付，提高用户体验。需支持Face ID、指纹支付，单笔1000元以下免密。",
        "solution": "正在开发中，预计2025年Q1投产。已对接生物识别SDK，通过安全评估。",
        "status": "开发中",
        "dept": "数字银行部",
        "priority": "高"
    },
    {
        "id": "XQ-20230102-004",
        "title": "网银渠道优化，支持大额汇款和预约功能",
        "content": "零售业务部提出，客户需要预约特定日期和金额的汇款，但现有网银系统不支持预约转账功能。",
        "solution": "已在2023年Q2实现，功能与001号需求类似，但侧重预约功能。支持设置未来30天内的转账预约。",
        "status": "已上线",
        "dept": "零售银行部",
        "priority": "中"
    },
    {
        "id": "XQ-20240601-005",
        "title": "企业网银增加银企直连接口",
        "content": "大型企业客户要求提供银企直连功能，实现ERP系统与银行系统的直接对接，支持批量代发、自动对账等功能。",
        "solution": "已完成技术方案设计，预计2025年Q3投产。需改造核心系统接口，增加报文加密和签名验证。",
        "status": "设计阶段",
        "dept": "公司银行部",
        "priority": "高"
    },
    {
        "id": "XQ-20240615-006",
        "title": "手机银行增加养老金融模块",
        "content": "响应监管要求，个人手机银行需增加个人养老金账户开户、缴费、投资功能，支持税收优惠查询。",
        "solution": "已完成需求分析，正在进行系统设计。需对接养老金中央平台系统。",
        "status": "需求分析",
        "dept": "零售银行部",
        "priority": "高"
    }
]

# ==================== 数据结构 ====================
@dataclass
class RequirementCategory:
    """需求分类结果"""
    primary_category: str
    secondary_category: str
    confidence: int
    reason: str

@dataclass
class EffortEstimate:
    """工时估算"""
    analysis_days: int
    design_days: int
    develop_days: int
    test_days: int
    deploy_days: int
    total_days: int

@dataclass
class TeamSuggestion:
    """团队建议"""
    pm: int
    frontend: int
    backend: int
    tester: int

@dataclass
class SuggestionResult:
    """完整建议结果"""
    category: RequirementCategory
    effort: EffortEstimate
    weeks: int
    team: TeamSuggestion
    tech_notes: str
    risk_tech: str
    risk_business: str
    risk_schedule: str

# ==================== 全局状态 ====================
vectorstore = None
qa_chain = None
llm = None  # 共享 LLM 实例
system_stats = {
    "total_requirements": len(HISTORICAL_REQUIREMENTS),
    "online_count": 4,
    "developing_count": 2
}

# ==================== 核心功能 ====================
def init_system() -> Tuple:
    """初始化系统（知识库 + LLM）"""
    global vectorstore, qa_chain, llm

    start_time = time.time()

    print("🚀 正在初始化向量检索模型...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 转换为 Document
    documents = []
    for req in HISTORICAL_REQUIREMENTS:
        content = f"【需求ID】{req['id']}\n【需求标题】{req['title']}\n【需求内容】{req['content']}\n【解决方案】{req['solution']}\n【当前状态】{req['status']}"
        doc = Document(page_content=content, metadata={k: v for k, v in req.items()})
        documents.append(doc)

    print(f"📖 正在加载/创建向量数据库...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    # 初始化 LLM（共享实例）
    print("🤖 正在初始化大语言模型...")
    llm = ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_API_BASE,
        api_key="EMPTY",
        temperature=0.1
    )

    # 创建 QA 链
    prompt_template = """你是一位资深的银行科技项目经理，专注于需求查重工作。

## 背景知识（历史相似需求）：
{context}

## 当前新需求：
{question}

## 任务要求：
请对当前新需求进行全面分析，判断是否与历史需求重复或高度相似，给出专业建议。

### 分析维度：
1. **查重结论**：明确判断是否重复（完全重复/高度相似/部分相似/全新需求）
2. **相似度评分**：0-100分
3. **匹配的历史需求**：列出最相似的1-2条历史需求
4. **详细分析**：从功能模块、业务场景、技术实现等角度分析相似性
5. **建议措施**：如建议合并、复用、优化或全新开发

### 输出格式：
**【查重结论】**：🎯 [结论]
**【相似度】**：📊 [分数]%
**【匹配需求】**：📁 [历史需求ID] - [标题]
**【分析说明】**：💡 [详细分析]
**【处理建议】**：✅ [具体建议]

请用专业的银行IT视角进行分析，输出简洁明了。"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    elapsed = time.time() - start_time
    print(f"✅ 系统初始化完成！耗时: {elapsed:.2f}秒")

    return vectorstore, qa_chain, llm, elapsed

def classify_requirement(title: str, content: str, dept: str) -> RequirementCategory:
    """
    需求分类

    基于需求内容判断所属类别，返回一级分类、二级分类和置信度
    """
    # 构建分类选项描述
    options_text = ""
    for primary, info in CATEGORY_SYSTEM.items():
        options_text += f"\n【{primary}】{info['description']}\n  子类："
        for sub, desc in info['subcategories'].items():
            options_text += f"{sub}（{desc}）、"
        options_text = options_text.rstrip("、")

    classify_prompt = f"""你是一位银行需求分析师，需要对以下需求进行专业分类。

**需求信息**：
- 标题：{title}
- 内容：{content}
- 部门：{dept}

**分类体系**：
{options_text}

**分析要求**：
1. 仔细阅读需求内容，判断最合适的一级分类（大类）
2. 选择最合适的二级分类（子类）
3. 给出分类置信度（0-100）
4. 说明分类依据

**输出格式**（请严格按照此格式输出）：
```
一级分类：开发类
二级分类：功能开发
置信度：85
分类依据：需求涉及手机银行App新功能开发，属于新增业务功能范畴，与现有网银转账功能不同终端的实现
```"""

    try:
        response = llm.invoke(classify_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # 解析响应
        result = parse_category_response(response_text)

        return RequirementCategory(
            primary_category=result.get("primary", "待分类"),
            secondary_category=result.get("secondary", "待分类"),
            confidence=result.get("confidence", 50),
            reason=result.get("reason", "AI自动分析")
        )
    except Exception as e:
        print(f"分类失败: {e}")
        return RequirementCategory(
            primary_category="待分类",
            secondary_category="待分类",
            confidence=0,
            reason=f"分类失败: {str(e)}"
        )

def parse_category_response(text: str) -> Dict:
    """解析分类响应"""
    result = {}
    lines = text.strip().split('\n')
    for line in lines:
        if '：' in line or ':' in line:
            key, value = line.split('：') if '：' in line else line.split(':')
            key = key.strip()
            value = value.strip()
            if '一级分类' in key:
                result['primary'] = value
            elif '二级分类' in key:
                result['secondary'] = value
            elif '置信度' in key:
                try:
                    result['confidence'] = int(re.search(r'\d+', value).group())
                except:
                    result['confidence'] = 50
            elif '分类依据' in key or '依据' in key:
                result['reason'] = value
    return result

def generate_suggestion(
    title: str,
    content: str,
    dept: str,
    priority: str,
    category: RequirementCategory,
    context: str = ""
) -> SuggestionResult:
    """
    生成实施建议

    基于需求分析和分类结果，生成工时估算、团队建议、实施计划等
    """
    # 构建 Prompt
    suggestion_prompt = f"""你是一位资深银行科技项目经理，需要为以下新需求生成专业实施建议。

**需求信息**：
- 标题：{title}
- 内容：{content}
- 部门：{dept}
- 优先级：{priority}
- 需求分类：{category.primary_category} > {category.secondary_category}

**参考历史需求**：
{context if context else "无参考历史需求"}

**分析任务**：
请从以下维度给出专业建议（必须严格按照格式输出）：

```
工时估算-需求分析：3
工时估算-系统设计：5
工时估算-开发实现：15
工时估算-测试验收：5
工时估算-上线部署：2
建议周期：6周
团队建议-产品经理：1
团队建议-前端开发：1
团队建议-后端开发：2
团队建议-测试工程师：1
技术建议：1. 可复用现有模块；2. 建议采用微服务架构；3. 需要对接核心系统
风险提示-技术：涉及核心系统改造，建议提前沟通
风险提示-业务：需要进行安全评估
风险提示-进度：建议分批上线降低风险
```"""

    try:
        response = llm.invoke(suggestion_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        return parse_suggestion_response(response_text, category)
    except Exception as e:
        print(f"建议生成失败: {e}")
        # 返回默认建议
        return SuggestionResult(
            category=category,
            effort=EffortEstimate(3, 5, 15, 5, 2, 30),
            weeks=6,
            team=TeamSuggestion(1, 1, 2, 1),
            tech_notes="请根据实际情况制定技术方案",
            risk_tech="请评估技术风险",
            risk_business="请评估业务风险",
            risk_schedule="请评估进度风险"
        )

def parse_suggestion_response(text: str, category: RequirementCategory) -> SuggestionResult:
    """解析建议响应"""
    effort = EffortEstimate(3, 5, 15, 5, 2, 30)
    weeks = 6
    team = TeamSuggestion(1, 1, 2, 1)
    tech_notes = "请参考同类项目制定技术方案"
    risk_tech = "请评估技术风险"
    risk_business = "请评估业务风险"
    risk_schedule = "请评估进度风险"

    lines = text.strip().split('\n')
    for line in lines:
        if '工时估算' in line or '需求分析' in line:
            try:
                match = re.search(r'需求分析[：:]\s*(\d+)', line)
                if match: effort.analysis_days = int(match.group(1))
            except: pass
        if '系统设计' in line:
            try:
                match = re.search(r'系统设计[：:]\s*(\d+)', line)
                if match: effort.design_days = int(match.group(1))
            except: pass
        if '开发实现' in line:
            try:
                match = re.search(r'开发实现[：:]\s*(\d+)', line)
                if match: effort.develop_days = int(match.group(1))
            except: pass
        if '测试验收' in line:
            try:
                match = re.search(r'测试验收[：:]\s*(\d+)', line)
                if match: effort.test_days = int(match.group(1))
            except: pass
        if '上线部署' in line:
            try:
                match = re.search(r'上线部署[：:]\s*(\d+)', line)
                if match: effort.deploy_days = int(match.group(1))
            except: pass
        if '建议周期' in line:
            try:
                match = re.search(r'(\d+)周', line)
                if match: weeks = int(match.group(1))
            except: pass
        if '产品经理' in line:
            try:
                match = re.search(r'产品经理[：:]\s*(\d+)', line)
                if match: team.pm = int(match.group(1))
            except: pass
        if '前端开发' in line:
            try:
                match = re.search(r'前端开发[：:]\s*(\d+)', line)
                if match: team.frontend = int(match.group(1))
            except: pass
        if '后端开发' in line:
            try:
                match = re.search(r'后端开发[：:]\s*(\d+)', line)
                if match: team.backend = int(match.group(1))
            except: pass
        if '测试工程师' in line:
            try:
                match = re.search(r'测试工程师[：:]\s*(\d+)', line)
                if match: team.tester = int(match.group(1))
            except: pass
        if '技术建议' in line or '技术方案' in line:
            tech_notes = line.split('：')[-1].split(':')[-1].strip() if '：' in line or ':' in line else line
        if '风险提示' in line or '风险' in line:
            if '技术' in line:
                risk_tech = line.split('：')[-1].split(':')[-1].strip()
            elif '业务' in line:
                risk_business = line.split('：')[-1].split(':')[-1].strip()
            elif '进度' in line:
                risk_schedule = line.split('：')[-1].split(':')[-1].strip()

    # 计算总工时
    effort.total_days = (effort.analysis_days + effort.design_days +
                        effort.develop_days + effort.test_days + effort.deploy_days)

    return SuggestionResult(
        category=category,
        effort=effort,
        weeks=weeks,
        team=team,
        tech_notes=tech_notes,
        risk_tech=risk_tech,
        risk_business=risk_business,
        risk_schedule=risk_schedule
    )

def format_category_output(category: RequirementCategory) -> str:
    """格式化分类结果输出"""
    emoji_map = {
        "开发类": "🛠️",
        "服务类": "🔧",
        "运维类": "⚙️",
        "提数类": "📊",
        "待分类": "❓"
    }
    primary_emoji = emoji_map.get(category.primary_category, "📁")

    return f"""
### 📝 需求分类

| 属性 | 值 |
|------|-----|
| **大类** | {primary_emoji} {category.primary_category} |
| **子类** | 📂 {category.secondary_category} |
| **置信度** | 🎯 {category.confidence}% |
| **分类依据** | 💡 {category.reason} |
"""

def format_suggestion_output(suggestion: SuggestionResult) -> str:
    """格式化建议输出"""
    effort = suggestion.effort
    team = suggestion.team

    return f"""
### 💡 实施建议

#### 📊 工时估算

| 阶段 | 工作量 |
|------|--------|
| 需求分析 | {effort.analysis_days} 人天 |
| 系统设计 | {effort.design_days} 人天 |
| 开发实现 | {effort.develop_days} 人天 |
| 测试验收 | {effort.test_days} 人天 |
| 上线部署 | {effort.deploy_days} 人天 |
| **合计** | **{effort.total_days} 人天** |

#### 📅 实施计划

- **建议总周期**：⏱️ {suggestion.weeks} 周

#### 👥 团队建议

| 角色 | 人数 |
|------|------|
| 产品经理 | {team.pm} 名 |
| 前端开发 | {team.frontend} 名 |
| 后端开发 | {team.backend} 名 |
| 测试工程师 | {team.tester} 名 |
| **合计** | **{team.pm + team.frontend + team.backend + team.tester} 人** |

#### 🔧 技术建议

{suggestion.tech_notes}

#### ⚠️ 风险提示

| 风险类型 | 风险说明 |
|---------|---------|
| 🔴 技术风险 | {suggestion.risk_tech} |
| 🟡 业务风险 | {suggestion.risk_business} |
| 🟠 进度风险 | {suggestion.risk_schedule} |
"""

def analyze_requirement(
    requirement_title: str,
    requirement_content: str,
    requirement_dept: str = "未知",
    requirement_priority: str = "中"
) -> Tuple[str, str]:
    """
    增强版分析函数：查重 + 分类 + 建议
    """
    global qa_chain, llm

    start_time = time.time()

    if qa_chain is None:
        return "❌ 系统未初始化，请联系管理员", ""

    if not requirement_title.strip() or not requirement_content.strip():
        return "⚠️ 请输入需求标题和内容", ""

    # 构建完整需求描述
    full_requirement = f"""**需求基本信息**
- 需求标题：{requirement_title}
- 所属部门：{requirement_dept}
- 优先级：{requirement_priority}

**需求详细描述**：
{requirement_content}"""

    try:
        # 1. 查重分析
        deduplication_result = qa_chain.invoke(full_requirement)
        deduplication_text = deduplication_result['result']

        # 2. 需求分类
        category = classify_requirement(
            requirement_title,
            requirement_content,
            requirement_dept
        )

        # 3. 生成实施建议
        context = extract_context(deduplication_text)
        suggestion = generate_suggestion(
            requirement_title,
            requirement_content,
            requirement_dept,
            requirement_priority,
            category,
            context
        )

        # 4. 格式化输出
        elapsed = time.time() - start_time

        category_output = format_category_output(category)
        suggestion_output = format_suggestion_output(suggestion)

        full_result = f"""
---

## 📊 智能分析结果

{deduplication_text}

---

{category_output}

---

{suggestion_output}

---

*⏱️ 分析耗时：{elapsed:.2f}秒*
"""

        return full_result, f"✅ 分析完成 | 耗时: {elapsed:.2f}秒"

    except Exception as e:
        return f"❌ 分析失败：{str(e)}", ""

def extract_context(deduplication_text: str) -> str:
    """从查重结果中提取上下文用于建议生成"""
    # 提取匹配的需求信息
    lines = deduplication_text.split('\n')
    context_lines = []
    for line in lines:
        if 'XQ-' in line or '匹配需求' in line or '历史需求' in line:
            context_lines.append(line)
    return '\n'.join(context_lines) if context_lines else ""

# ==================== 初始化 ====================
print("\n" + "="*60)
print("🚀 智能需求查重系统 V2.0 启动中...")
print("  新增：需求分类 | 实施建议生成 | 风险提示")
print("="*60)

try:
    vectorstore, qa_chain, llm, init_time = init_system()
    print(f"\n✅ 系统初始化完成！耗时: {init_time:.2f}秒")
    print(f"📊 已加载 {len(HISTORICAL_REQUIREMENTS)} 条历史需求")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    print("⚠️ 请检查 vLLM 服务是否启动")

# ==================== Gradio 界面 ====================
CSS = """
/* 全局样式 */
.gradio-container {max-width: 1300px !important; margin: 0 auto !important;}
.main-header {text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 12px; margin-bottom: 20px;}
.main-header h1 {color: white !important; margin: 0 !important; font-size: 28px !important;}
.main-header p {color: rgba(255,255,255,0.9) !important; margin: 10px 0 0 0 !important;}

/* 按钮样式 */
.primary-btn {background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important; border: none !important; border-radius: 8px !important;}
.primary-btn:hover {transform: translateY(-2px); box-shadow: 0 4px 12px rgba(76,175,80,0.4) !important;}

/* 状态指示器 */
.status-indicator {display: flex; align-items: center; gap: 8px; padding: 12px 16px; background: #e8f5e9; border-radius: 8px; border-left: 4px solid #4CAF50;}
.status-dot {width: 12px; height: 12px; background: #4CAF50; border-radius: 50%; animation: pulse 2s infinite;}
@keyframes pulse {0% {opacity: 1;} 50% {opacity: 0.5;} 100% {opacity: 1;}}

/* 结果框样式 */
.result-box {background: #f8f9fa; border-radius: 12px; padding: 20px; border: 1px solid #e0e0e0;}
.result-box h3 {color: #1e3a5f; margin-top: 0 !important;}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    # 头部
    gr.HTML("""
    <div class="main-header">
        <h1>🏦 智能需求查重系统 V2.0</h1>
        <p>基于 AI 大模型 + 向量检索的银行需求智能分析助手</p>
        <p style="font-size: 14px; margin-top: 8px;">✨ 新增：需求分类 | 📊 实施建议 | ⚠️ 风险提示</p>
    </div>
    """)

    # 系统状态
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(f"""
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>🤖 系统运行正常 | 已加载 {len(HISTORICAL_REQUIREMENTS)} 条历史需求</span>
            </div>
            """)
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 刷新状态", size="sm", variant="secondary")

    # 主工作区
    with gr.Tabs():
        # Tab 1: 智能查重
        with gr.TabItem("🔍 智能查重", id="analyze"):
            with gr.Row():
                # 左侧 - 输入区
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 录入新需求")

                    with gr.Group():
                        title_input = gr.Textbox(
                            label="需求标题",
                            placeholder="请输入需求标题，例如：手机银行增加大额转账功能",
                            lines=2
                        )

                        content_input = gr.Textbox(
                            label="需求详细描述",
                            placeholder="请详细描述需求的业务场景、功能要求、技术约束等...",
                            lines=6
                        )

                    with gr.Row():
                        dept_input = gr.Dropdown(
                            label="所属部门",
                            choices=["零售银行部", "数字银行部", "公司银行部", "运营管理部", "风险管理部", "科技部", "其他"],
                            value="零售银行部"
                        )
                        priority_input = gr.Dropdown(
                            label="优先级",
                            choices=["高", "中", "低"],
                            value="中"
                        )

                    analyze_btn = gr.Button("🚀 开始智能分析（查重+分类+建议）", variant="primary", size="lg")

                    gr.Markdown("#### 💡 快速示例")
                    gr.Examples(
                        examples=[
                            ["手机银行增加大额转账预约功能", "业务部门希望在手机银行App上增加大额转账功能，支持单笔50万以上的转账，并提供预约转账选项，客户可以预设转账时间和金额。", "零售银行部", "高"],
                            ["柜面系统增加人脸识别登录", "为了提高柜面操作安全性，要求在柜面系统中增加人脸识别功能，柜员登录时需要进行人脸比对验证。", "运营管理部", "中"],
                            ["网银增加批量代发功能", "企业客户要求在企业网银上增加批量代发工资功能，支持一次上传最多500条转账记录，自动处理。", "公司银行部", "高"],
                            ["手机银行增加外汇买卖功能", "零售客户希望在外汇App上直接进行外汇买卖，支持实时行情查看和外汇买卖交易。", "零售银行部", "中"],
                        ],
                        inputs=[title_input, content_input, dept_input, priority_input],
                        label="点击填充示例需求"
                    )

                # 右侧 - 结果区
                with gr.Column(scale=1.2):
                    gr.Markdown("### 📊 智能分析结果")
                    result_output = gr.Textbox(
                        label="完整分析报告",
                        lines=28,
                        show_label=True,
                        interactive=False,
                        elem_classes=["result-box"]
                    )

                    time_display = gr.Markdown("", visible=True)

                    analyze_btn.click(
                        fn=analyze_requirement,
                        inputs=[title_input, content_input, dept_input, priority_input],
                        outputs=[result_output, time_display]
                    )

                    title_input.submit(
                        fn=analyze_requirement,
                        inputs=[title_input, content_input, dept_input, priority_input],
                        outputs=[result_output, time_display]
                    )

        # Tab 2: 历史需求库
        with gr.TabItem("📚 历史需求库", id="history"):
            with gr.Accordion("🔍 搜索历史需求", open=True):
                search_input = gr.Textbox(
                    label="关键词搜索",
                    placeholder="输入关键词搜索历史需求...",
                    lines=1
                )
            with gr.Row():
                status_filter = gr.Dropdown(
                    label="状态筛选",
                    choices=["全部", "已上线", "开发中", "需求分析", "设计阶段"],
                    value="全部"
                )
                dept_filter = gr.Dropdown(
                    label="部门筛选",
                    choices=["全部"] + list(set(req['dept'] for req in HISTORICAL_REQUIREMENTS)),
                    value="全部"
                )
            history_output = gr.Markdown()
            gr.RefreshButton(value=get_history_requirements, outputs=history_output)

            def filter_history(keyword: str, status: str, dept: str) -> str:
                result = "## 📋 历史需求清单\n\n"
                for req in HISTORICAL_REQUIREMENTS:
                    if status != "全部" and req["status"] != status:
                        continue
                    if dept != "全部" and req["dept"] != dept:
                        continue
                    if keyword:
                        kw = keyword.lower()
                        if kw not in req['title'].lower() and kw not in req['content'].lower():
                            continue

                    status_emoji = "✅" if req["status"] == "已上线" else "🔄" if "开发" in req["status"] else "📋"
                    priority_🔴" if req["priority"] ==emoji = " "高" else "🟡" if req["priority"] == "中" else "🟢"

                    result += f"""### {status_emoji} {req['id']} - {req['title']}

| 属性 | 值 |
|------|-----|
| **部门** | {req['dept']} |
| **优先级** | {priority_emoji} {req['priority']} |
| **状态** | {req['status']} |

**需求内容**：{req['content']}

**解决方案**：{req['solution']}

---

"""
                return result

            search_input.submit(filter_history, inputs=[search_input, status_filter, dept_filter], outputs=history_output)
            status_filter.change(filter_history, inputs=[search_input, status_filter, dept_filter], outputs=history_output)
            dept_filter.change(filter_history, inputs=[search_input, status_filter, dept_filter], outputs=history_output)

        # Tab 3: 系统状态
        with gr.TabItem("⚙️ 系统状态", id="status"):
            gr.Markdown("## 🏦 智能需求查重系统 V2.0 - 系统信息")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 🤖 模型配置
                    | 配置项 | 值 |
                    |-------|-----|
                    | 大语言模型 | Qwen2.5-1.5B-Instruct |
                    | 向量模型 | BAAI/bge-small-zh-v1.5 |
                    | 向量数据库 | ChromaDB |
                    | API 接口 | OpenAI Compatible |
                    """)
                with gr.Column():
                    gr.Markdown("""
                    ### 📊 知识库统计
                    | 指标 | 值 |
                    |-----|-----|
                    | 历史需求总数 | {} 条 |
                    | 已上线需求 | {} 条 |
                    | 开发中需求 | {} 条 |
                    | 向量维度 | 512 |
                    """.format(
                        system_stats['total_requirements'],
                        system_stats['online_count'],
                        system_stats['developing_count']
                    ))

            gr.Markdown("""
            ### 💡 V2.0 新增功能

            #### 📝 需求分类
            自动将需求分类到四大类及其子类别：
            | 大类 | 子类别 |
            |------|--------|
            | 🛠️ 开发类 | 需求开发、功能开发、接口开发、报表开发、界面开发 |
            | 🔧 服务类 | 业务服务、客户服务、系统服务、接口服务 |
            | ⚙️ 运维类 | 系统运维、数据运维、安全运维、性能运维 |
            | 📊 提数类 | 数据提取、报表生成、数据分析、数据核对 |

            #### 💡 实施建议
            - 📊 工时估算：按阶段拆分，预估总工作量
            - 📅 实施计划：建议周期和分阶段安排
            - 👥 团队建议：推荐各角色人数配置
            - 🔧 技术建议：技术选型和架构方案
            - ⚠️ 风险提示：识别技术、业务、进度风险

            ### 🔧 技术架构
            - **前端**：Gradio
            - **后端**：LangChain
            - **LLM**：vLLM (Qwen2.5)
            - **向量库**：ChromaDB
            - **Embedding**：Sentence-BERT (bge-small-zh)
            """)

    # 底部
    gr.HTML("""
    <div style="text-align: center; padding: 20px; color: #666; font-size: 12px;">
        <p>🏦 智能需求查重系统 V2.0 | 基于 LangChain + vLLM</p>
        <p>© 2024 银行科技部</p>
    </div>
    """)

# ==================== 启动入口 ====================
def get_history_requirements():
    """获取历史需求列表"""
    return "## 📋 历史需求清单\n\n" + "暂无数据"

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
