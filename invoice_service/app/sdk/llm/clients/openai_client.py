"""
OpenAI LLM客户端实现

提供OpenAI API的客户端实现
"""
import logging
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from ..base import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """
    OpenAI LLM客户端
    
    使用OpenAI的API实现LLM客户端接口
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化OpenAI客户端
        
        Args:
            config: LLM配置
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout
        )
        self.model = config.model_name or "gpt-3.5-turbo"
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.additional_params = config.additional_params or {}
        
        logger.info(f"OpenAI客户端初始化: model={self.model}")
    
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
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            messages.append({"role": "user", "content": prompt})
            
            # 合并默认参数和额外参数
            params = {
                "model": self.model,
                "temperature": self.temperature,
                **self.additional_params,
                **kwargs
            }
            
            if self.max_tokens:
                params["max_tokens"] = self.max_tokens
            
            response: ChatCompletion = await self.client.chat.completions.create(
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI生成文本失败: {str(e)}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        获取文本嵌入向量
        
        Args:
            text: 需要嵌入的文本
            
        Returns:
            List[float]: 嵌入向量
        """
        try:
            embedding_model = self.additional_params.get("embedding_model", "text-embedding-3-small")
            
            response = await self.client.embeddings.create(
                input=text,
                model=embedding_model
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI获取嵌入向量失败: {str(e)}")
            raise 