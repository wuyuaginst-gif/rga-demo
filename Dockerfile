# 使用官方的 Python 3.12 镜像作为基础。
# slim 版本更小，适合生产环境。
# 也可以考虑使用 'alpine' 版本进一步缩小体积，但可能需要安装额外的依赖。
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 到工作目录
COPY requirements.txt .

# 优化 pip 安装速度，并安装依赖。
# 使用 --no-cache-dir 减少最终镜像体积。
RUN pip install --no-cache-dir -r requirements.txt

# 复制你的 Python 代码到容器
# 假设你的代码文件名为 smart_deduplication.py
COPY smart_deduplication.py .

# --- 容器启动配置 ---

# 暴露端口 (如果未来要封装成 FastAPI 服务，此处需要暴露端口，例如 8000)
# EXPOSE 8000

# 启动命令。
# 在生产环境中，我们通常会封装一个入口脚本 (start.sh) 或使用 Web 框架启动。
# 在这个 Demo 中，我们直接运行 Python 脚本进行知识库构建和查重测试。

# 注意：知识库构建 (build_knowledge_base) 会在第一次运行时创建向量数据文件。
# 在实际生产中，知识库应该事先构建好，或者存储在持久化卷中。
# 此处为了演示，我们假设容器启动时即运行查重功能。
CMD ["python", "smart_deduplication.py"]
