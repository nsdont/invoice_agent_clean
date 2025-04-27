"""
LLM客户端演示程序
"""
import os
import sys
import argparse
import asyncio
from typing import Dict, Any

from .base import LLMProvider
from .factory import create_client, register_client, get_client, get_default_client, setup_default_client


async def run_text_demo(client_name: str = None, prompt: str = None):
    """
    运行文本生成演示
    
    Args:
        client_name: 客户端名称，如果为None则使用默认客户端
        prompt: 提示词，如果为None则使用默认提示词
    """
    try:
        client = get_client(client_name) if client_name else get_default_client()
    except (KeyError, RuntimeError):
        print(f"未找到LLM客户端: {client_name or '默认'}")
        return
    
    if prompt is None:
        prompt = "解释一下什么是发票处理系统，并给出三个可能的功能点。"
    
    print(f"使用客户端: {client.__class__.__name__}")
    print(f"提示词: {prompt}")
    print("-" * 50)
    
    try:
        response = await client.generate_text(prompt)
        print(response)
    except Exception as e:
        print(f"生成文本时发生错误: {e}")


async def setup_clients():
    """
    设置演示用的LLM客户端
    """
    # 尝试从环境变量设置默认客户端
    try:
        default_client = setup_default_client()
        print(f"已设置默认客户端: {default_client.__class__.__name__}")
    except Exception as e:
        print(f"设置默认客户端失败: {e}")
    
    # 如果没有设置默认客户端，尝试使用环境变量API密钥创建客户端
    if "OPENAI_API_KEY" in os.environ:
        openai_client = create_client(
            provider=LLMProvider.OPENAI,
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4",
        )
        register_client("openai", openai_client, set_as_default=True)
    
    if "GEMINI_API_KEY" in os.environ:
        gemini_client = create_client(
            provider=LLMProvider.GEMINI,
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.0-flash",
        )
        register_client("gemini", gemini_client)
    
    if "DEEPSEEK_API_KEY" in os.environ:
        deepseek_client = create_client(
            provider=LLMProvider.DEEPSEEK,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model_name="deepseek-chat",
        )
        register_client("deepseek", deepseek_client)


async def main_async():
    """
    异步主程序入口
    """
    parser = argparse.ArgumentParser(description="LLM客户端演示程序")
    parser.add_argument("--client", type=str, help="要使用的客户端名称")
    parser.add_argument("--prompt", type=str, help="提示词")
    
    args = parser.parse_args()
    
    # 设置客户端
    await setup_clients()
    
    # 运行文本演示
    await run_text_demo(args.client, args.prompt)


def main():
    """
    主程序入口
    """
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 