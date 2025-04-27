FROM python:3.10-slim

WORKDIR /app

# 安装依赖
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    tesseract-ocr-chi-tra \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装Python包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY ./app .

# 创建临时目录
RUN mkdir -p /app/temp && chmod 777 /app/temp
RUN mkdir -p /app/shared && chmod 777 /app/shared

# 暴露端口
EXPOSE 5000

# 运行应用
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"] 