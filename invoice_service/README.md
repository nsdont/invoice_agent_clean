# 发票处理微服务

这个微服务集成了发票处理的所有功能，包括OCR文本提取、LLM结构化信息提取和产品匹配，提供一站式发票处理解决方案。

## 功能特点

- **集成多种功能**：OCR、LLM提取、产品匹配
- **多引擎OCR**：支持Tesseract和EasyOCR，自动选择最佳引擎
- **图像预处理**：提高OCR准确性
- **灵活的API**：支持单步或端到端处理
- **多种数据格式**：支持文件上传和base64编码
- **中英文支持**：支持识别中英文发票
- **容器化部署**：使用Docker简化部署

## API接口

### 1. 健康检查

```
GET /health
```

响应示例:
```json
{
  "status": "ok",
  "easyocr_available": true,
  "tesseract_version": "4.1.1",
  "openai_api_key_configured": true,
  "products_loaded": 150,
  "temp_dir": true,
  "shared_dir": true
}
```

### 2. OCR处理

**使用文件上传:**

```
POST /ocr
Content-Type: multipart/form-data
```

参数:
- `file`: 发票图像文件
- `engine`: (可选) OCR引擎(auto/tesseract/easyocr)

**使用base64编码:**

```
POST /ocr
Content-Type: application/json
```

请求体:
```json
{
  "base64": "图像的base64编码字符串"
}
```

响应示例:
```json
{
  "text": "提取的OCR文本",
  "engine": "使用的OCR引擎",
  "processing_time": 1.234,
  "output_file": "/app/shared/ocr_result_uuid.txt",
  "success": true
}
```

### 3. 结构化信息提取

```
POST /extract
Content-Type: application/json
```

请求体:
```json
{
  "text": "OCR提取的文本",
  "customer_name": "可选的客户名称"
}
```

响应示例:
```json
{
  "order_data": {
    "customer_name": "客户名称",
    "order_id": "订单ID",
    "order_date": "2023-06-01",
    "items": [
      {
        "original_input": "产品A x2",
        "quantity": 2,
        "matched_name": null,
        "match_score": 0
      }
    ],
    "status": "pending"
  },
  "success": true
}
```

### 4. 产品匹配

```
POST /match
Content-Type: application/json
```

请求体:
```json
{
  "order_data": {
    // 从/extract接口返回的order_data
  },
  "threshold": 0.6
}
```

响应示例:
```json
{
  "order_data": {
    "customer_name": "客户名称",
    "order_id": "ORD-A1B2C3D4",
    "order_date": "2023-06-01",
    "items": [
      {
        "original_input": "产品A x2",
        "quantity": 2,
        "matched_name": "产品A",
        "product_id": "PROD001",
        "unit_price": 99.99,
        "match_score": 0.95,
        "needs_review": false
      }
    ],
    "status": "completed",
    "total": 199.98
  },
  "output_file": "/app/shared/order_ORD-A1B2C3D4_2023-06-01.json",
  "success": true
}
```

### 5. 端到端处理

```
POST /process
Content-Type: multipart/form-data
```

参数:
- `file`: 发票图像文件
- `customer_name`: (可选) 客户名称
- `threshold`: (可选) 匹配阈值，默认0.6

或使用JSON:

```
POST /process
Content-Type: application/json
```

请求体:
```json
{
  "base64": "图像的base64编码字符串",
  "customer_name": "客户名称"
}
```

响应示例:
```json
{
  "order_data": {
    // 完整的订单数据，同/match接口
  },
  "ocr_file": "/app/shared/ocr_result_uuid.txt",
  "output_file": "/app/shared/order_ORD-A1B2C3D4_2023-06-01.json",
  "ocr_engine": "easyocr",
  "processing_time": 3.456,
  "success": true
}
```

## 安装与运行

### 使用Docker Compose

1. 确保已安装Docker和Docker Compose
2. 创建.env文件设置OpenAI API密钥
3. 准备products.json产品数据文件
4. 运行服务:

```bash
cd app/ocr_service
docker-compose up -d
```

### 非Docker环境运行

1. 确保已安装Python 3.8或更高版本
2. 安装依赖:

```bash
cd invoice_service
pip install -r requirements.txt
```

3. 创建必要的目录结构:

```bash
mkdir -p invoice_service/temp invoice_service/shared
```

4. 在`invoice_service/shared`目录下放置products.json产品数据文件

5. 设置环境变量(可选):
```bash
# Linux/Mac
export OPENAI_API_KEY=your_openai_api_key
export APP_BASE_DIR=/path/to/your/app  # 可选，默认为项目根目录
export APP_TEMP_DIR=/path/to/temp      # 可选，默认为APP_BASE_DIR/temp
export APP_SHARED_DIR=/path/to/shared  # 可选，默认为APP_BASE_DIR/shared

# Windows
set OPENAI_API_KEY=your_openai_api_key
set APP_BASE_DIR=C:\path\to\your\app  # 可选
set APP_TEMP_DIR=C:\path\to\temp      # 可选
set APP_SHARED_DIR=C:\path\to\shared  # 可选
```

6. 运行服务:
```bash
cd invoice_service
python -m app.app
```

服务将在http://localhost:5000运行，可通过API文档中描述的接口访问。

注意：在非Docker环境下，你需要确保已安装所有OCR依赖（如Tesseract和EasyOCR）。

### 环境变量

创建.env文件，包含以下内容:

```
OPENAI_API_KEY=your_openai_api_key
PORT=5000
LOG_LEVEL=INFO
```

### 产品数据

在Docker环境下，请在`/app/shared`目录下创建products.json文件；在非Docker环境下，请在`invoice_service/shared`目录下创建。格式如下:

```json
[
  {
    "id": "PROD001",
    "name": "产品A",
    "name_en": "Product A",
    "price": 99.99,
    "unit": "个",
    "code": "PA001",
    "category": "电子"
  },
  ...
]
```

## 与n8n集成

在n8n工作流中，使用HTTP Request节点调用服务:

1. 配置HTTP Request节点:
   - 方法: POST
   - URL: `http://invoice-service:5000/process`
   - 请求格式: JSON
   - 请求体: `{"base64": "{{$binary.toBase64($node['上传节点'].binary.file)}}", "customer_name": "{{$node['客户信息'].json.customer_name}}"}`

2. 处理返回的结果:
   - 使用IF节点检查order_data.status
   - 根据不同状态执行不同流程

## 许可证

MIT 