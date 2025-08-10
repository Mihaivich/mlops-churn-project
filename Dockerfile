# Dockerfile (新版本)

# 我们继续使用与本地兼容的Python版本
FROM python:3.11-slim

# 更新包管理器并安装基础工具
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 设置标准工作目录
WORKDIR /opt/program

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型文件到SageMaker的标准路径
# 注意：我们现在复制整个 'model' 文件夹
RUN mkdir -p /opt/ml/model
COPY model/ /opt/ml/model/

# 复制推理代码
COPY inference/predict.py .

# 创建一个简单的wsgi.py文件来作为Gunicorn的入口
RUN echo 'from predict import app\n\nif __name__ == "__main__":\n    app.run()' > /opt/program/wsgi.py

# 创建一个健壮的启动脚本 'serve'
RUN echo '#!/bin/bash\n\
if [[ "$1" == "serve" ]]; then\n\
    cd /opt/program\n\
    echo "Starting Gunicorn with Flask app..."\n\
    exec gunicorn --bind 0.0.0.0:8080 --workers 1 wsgi:app\n\
else\n\
    exec "$@"\n\
fi' > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

# 将我们的程序目录添加到Python和系统的PATH中
ENV PYTHONPATH="/opt/program:${PYTHONPATH}"
ENV PATH="/opt/program:${PATH}"

# 暴露端口
EXPOSE 8080

# 设置启动入口点
ENTRYPOINT ["/usr/local/bin/serve"]
CMD ["serve"]