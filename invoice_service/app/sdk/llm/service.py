"""
LLM服务模块

提供LLM服务接口，统一处理不同LLM客户端
"""
import logging
from typing import List, Dict, Any, Optional

from .base import LLMProvider
from .factory import create_client


logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM服务类
    
    提供统一的LLM服务接口，封装不同LLM客户端的调用
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 600,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        additional_params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化LLM服务
        
        Args:
            provider: LLM提供商
            api_key: API密钥
            model_name: 模型名称
            api_base: API基础URL
            timeout: 超时时间(秒)
            max_tokens: 最大生成令牌数
            temperature: 温度参数
            additional_params: 额外参数
        """
        self.client = create_client(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            api_base=api_base,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            additional_params=additional_params
        )
        logger.info(f"已初始化LLM服务，提供商: {provider}")
    
    async def generate_text(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示文本
            system_prompt: 系统提示文本
            **kwargs: 额外参数
            
        Returns:
            str: 生成的文本内容
            
        Raises:
            Exception: 当生成文本失败时
        """
        try:
            return await self.client.generate_text(
                prompt=prompt, 
                system_prompt=system_prompt,
                **kwargs
            )
        except Exception as e:
            logger.error(f"生成文本失败: {str(e)}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        获取文本嵌入向量
        
        Args:
            text: 需要嵌入的文本
            
        Returns:
            List[float]: 嵌入向量
            
        Raises:
            Exception: 当获取嵌入向量失败时
        """
        try:
            return await self.client.get_embedding(text)
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {str(e)}")
            raise
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取多个文本的嵌入向量
        
        Args:
            texts: 需要嵌入的文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            Exception: 当获取嵌入向量失败时
        """
        try:
            embeddings = []
            for text in texts:
                embedding = await self.client.get_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"获取多个嵌入向量失败: {str(e)}")
            raise 