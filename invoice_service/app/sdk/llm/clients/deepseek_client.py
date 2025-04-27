"""
Deepseek LLM客户端实现

提供Deepseek API的客户端实现
"""
import logging
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from ..base import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class DeepseekClient(LLMClient):
    """
    Deepseek LLM客户端
    
    使用Deepseek的API实现LLM客户端接口
    """

    async def generate_content(self, model, contents, generation_config, system_instruction) -> str:
        """
        生成内容（支持多模态输入）

        Args:
            model: 模型名称
            contents: 输入内容
            generation_config: 生成配置
            system_instruction: 系统指令

        Returns:
            str: 生成的内容
        """
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            
            # 处理contents内容
            if isinstance(contents, str):
                messages.append({"role": "user", "content": contents})
            elif isinstance(contents, list):
                for item in contents:
                    if isinstance(item, dict) and "role" in item and "content" in item:
                        messages.append(item)
                    else:
                        # 这里可以处理更复杂的消息格式，如果需要的话
                        messages.append({"role": "user", "content": str(item)})
            
            # 合并生成配置
            params = {
                "model": model or self.model,
                "temperature": generation_config.get("temperature", self.temperature),
                **({"max_tokens": generation_config.get("max_tokens", self.max_tokens)} if self.max_tokens or "max_tokens" in generation_config else {}),
                **{k: v for k, v in generation_config.items() if k not in ["temperature", "max_tokens"]}
            }
            
            response = await self.client.chat.completions.create(
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Deepseek生成内容失败: {str(e)}")
            raise

    def _initialize(self) -> None:
        """
        初始化Deepseek客户端
        """
        self.api_key = self.config.api_key
        self.api_base = self.config.api_base or "https://api.deepseek.com"
        self.model = self.config.model_name or "deepseek-chat"
        self.max_tokens = self.config.max_tokens
        self.temperature = self.config.temperature
        self.additional_params = self.config.additional_params or {}
        self.timeout = self.config.timeout
        
        # 初始化 OpenAI 客户端，但配置为使用 Deepseek 的 API
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout
        )
        
        logger.info(f"Deepseek客户端初始化: model={self.model}")
    
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
            
            params = {
                "model": kwargs.get("model", self.model),
                "temperature": kwargs.get("temperature", self.temperature),
                **({"max_tokens": kwargs.get("max_tokens", self.max_tokens)} if self.max_tokens or "max_tokens" in kwargs else {}),
                **{k: v for k, v in self.additional_params.items() if k not in ["model", "temperature", "max_tokens"]},
                **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
            }
            
            response = await self.client.chat.completions.create(
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Deepseek生成文本失败: {str(e)}")
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
            embedding_model = self.additional_params.get("embedding_model", "embedding-2")
            
            response = await self.client.embeddings.create(
                model=embedding_model,
                input=text
            )
            
            return response.data[0].embedding
        
        except Exception as e:
            logger.error(f"Deepseek获取嵌入向量失败: {str(e)}")
            raise 