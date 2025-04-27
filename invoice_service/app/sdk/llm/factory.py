"""
LLM客户端工厂模块

提供LLM客户端的创建、获取和管理功能
"""
import os
import logging
from typing import Optional, Dict, Any

from .base import LLMClient, LLMConfig, LLMProvider
from .clients import OpenAIClient, GeminiClient, DeepseekClient

logger = logging.getLogger(__name__)

# 全局客户端实例缓存
_clients: Dict[str, LLMClient] = {}
_default_client: Optional[LLMClient] = None


def create_client(
    provider: LLMProvider,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: int = 600,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    additional_params: Optional[Dict[str, Any]] = None
) -> LLMClient:
    """
    创建LLM客户端
    
    Args:
        provider: LLM提供商
        api_key: API密钥，如果为None则尝试从环境变量获取
        model_name: 模型名称
        api_base: API基础URL
        timeout: 超时时间(秒)
        max_tokens: 最大生成令牌数
        temperature: 温度参数
        additional_params: 额外参数
        
    Returns:
        LLMClient: LLM客户端实例
        
    Raises:
        ValueError: 当提供商不支持或API密钥未提供时
    """
    # 处理API密钥
    if api_key is None:
        if provider == LLMProvider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == LLMProvider.GEMINI:
            api_key = os.environ.get("GEMINI_API_KEY")
        elif provider == LLMProvider.DEEPSEEK:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        raise ValueError(f"未提供{provider}的API密钥，请在参数中指定或通过环境变量设置")
    
    # 创建配置
    config = LLMConfig(
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        additional_params=additional_params
    )
    
    # 根据提供商创建对应的客户端
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(config)
    elif provider == LLMProvider.GEMINI:
        return GeminiClient(config)
    elif provider == LLMProvider.DEEPSEEK:
        return DeepseekClient(config)
    else:
        raise ValueError(f"不支持的LLM提供商: {provider}")


def get_client(client_name: str) -> LLMClient:
    """
    获取一个命名的LLM客户端实例
    
    Args:
        client_name: 客户端名称
        
    Returns:
        LLMClient: LLM客户端实例
        
    Raises:
        KeyError: 如果客户端名称不存在
    """
    if client_name not in _clients:
        raise KeyError(f"未找到名为 '{client_name}' 的LLM客户端")
    
    return _clients[client_name]


def register_client(name: str, client: LLMClient, set_as_default: bool = False) -> None:
    """
    注册一个LLM客户端实例
    
    Args:
        name: 客户端名称
        client: LLM客户端实例
        set_as_default: 是否设置为默认客户端
    """
    _clients[name] = client
    
    global _default_client
    if set_as_default or _default_client is None:
        _default_client = client


def get_default_client() -> LLMClient:
    """
    获取默认的LLM客户端实例
    
    Returns:
        LLMClient: 默认的LLM客户端实例
        
    Raises:
        RuntimeError: 如果没有设置默认客户端
    """
    if _default_client is None:
        raise RuntimeError("未设置默认LLM客户端")
    
    return _default_client


def setup_default_client(env_prefix: str = "LLM") -> LLMClient:
    """
    从环境变量设置默认的LLM客户端
    
    Args:
        env_prefix: 环境变量前缀，默认为"LLM"
        
    Returns:
        LLMClient: 设置的默认LLM客户端
        
    环境变量示例:
        LLM_PROVIDER=openai
        LLM_API_KEY=sk-xxxx
        LLM_MODEL_NAME=gpt-4
        LLM_TEMPERATURE=0.7
        LLM_MAX_TOKENS=1000
    """
    provider_name = os.environ.get(f"{env_prefix}_PROVIDER", "openai").upper()
    
    try:
        provider = LLMProvider[provider_name]
    except KeyError:
        raise ValueError(f"不支持的LLM提供商: {provider_name}")
    
    client = create_client(
        provider=provider,
        api_key=os.environ.get(f"{env_prefix}_API_KEY"),
        model_name=os.environ.get(f"{env_prefix}_MODEL_NAME"),
        api_base=os.environ.get(f"{env_prefix}_API_BASE"),
        temperature=float(os.environ.get(f"{env_prefix}_TEMPERATURE", "0.7")),
        max_tokens=int(os.environ.get(f"{env_prefix}_MAX_TOKENS", "1000")),
    )
    
    register_client("default", client, set_as_default=True)
    
    return client 