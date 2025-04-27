# LLM服务SDK

LLM服务SDK提供了统一的接口来访问不同的大语言模型服务，包括OpenAI、Google Gemini和Deepseek等。

## 模块结构

```
llm/
├── __init__.py        # 包入口
├── base.py            # 基础定义
├── factory.py         # 客户端工厂
├── service.py         # 服务封装
├── demo.py            # 示例程序
└── clients/           # 客户端实现
    ├── __init__.py    # 客户端包入口
    ├── openai_client.py    # OpenAI客户端
    ├── gemini_client.py    # Gemini客户端
    └── deepseek_client.py  # Deepseek客户端
```

## 快速开始

### 1. 创建并使用LLM客户端

```python
import asyncio
from app.sdk.llm import create_client, LLMProvider

async def main():
    # 创建OpenAI客户端
    client = create_client(
        provider=LLMProvider.OPENAI,
        api_key="your-api-key",
        model_name="gpt-4"
    )
    
    # 生成文本
    response = await client.generate_text("解释什么是发票处理系统")
    print(response)

asyncio.run(main())
```

### 2. 使用LLM服务

```python
import asyncio
from app.sdk.llm import LLMService, LLMProvider

async def main():
    # 创建LLM服务
    service = LLMService(
        provider=LLMProvider.GEMINI,
        api_key="your-api-key",
        model_name="gemini-2.0-flash"
    )
    
    # 生成文本
    response = await service.generate_text(
        prompt="写一个关于人工智能的短文",
        system_prompt="你是一位AI专家"
    )
    print(response)

asyncio.run(main())
```

### 3. 使用默认客户端

```python
import os
import asyncio
from app.sdk.llm import setup_default_client, get_default_client

# 设置环境变量
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_API_KEY"] = "your-api-key"
os.environ["LLM_MODEL_NAME"] = "gpt-4"

async def main():
    # 设置默认客户端
    setup_default_client()
    
    # 使用默认客户端
    client = get_default_client()
    response = await client.generate_text("什么是发票OCR?")
    print(response)

asyncio.run(main())
```

## 支持的提供商

- OpenAI (`LLMProvider.OPENAI`)
- Google Gemini (`LLMProvider.GEMINI`)
- Deepseek (`LLMProvider.DEEPSEEK`) 