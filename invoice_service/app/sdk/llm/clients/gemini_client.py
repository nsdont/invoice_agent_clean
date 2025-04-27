"""
Gemini LLM客户端实现

提供Google Gemini API的客户端实现
"""
import logging
from typing import List, Dict, Any, Optional

from google import genai
from google.genai.types import GenerateContentConfig

from ..base import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """
    Gemini LLM客户端
    
    使用Google Gemini API实现LLM客户端接口
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化Gemini客户端
        
        Args:
            config: LLM配置
        """
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        
        if config.api_base:
            # 如果提供了自定义API基础URL
            # 注意：Gemini Python SDK可能不直接支持自定义URL
            # 这里仅为示例，实际使用需要查阅最新的SDK文档
            logger.warning("Gemini SDK可能不支持自定义API基础URL")
        
        self.model = config.model_name or "gemini-2.0-flash"
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.additional_params = config.additional_params or {}

        logger.info(f"Gemini客户端初始化: default model={self.model}")
    
    @property
    def models(self):
        """
        提供对GenAI模型的访问，支持直接使用models.generate_content方法
        
        Returns:
            对象本身，模拟SDK中的models属性
        """
        return self
    
    async def generate_content(
        self,
        model: str = None,
        contents: Any = None,
        generation_config: GenerateContentConfig = None,
        **kwargs
    ) -> str:
        """
        生成内容（支持多模态输入）
        
        Args:
            model: 模型名称（如果不提供，使用默认配置的模型）
            contents: 输入内容（文本、图像或组合）
            generation_config: 生成配置
            **kwargs: 额外参数
            
        Returns:
            str: 生成的内容响应
        """
        try:
            # 使用模型生成内容
            response = self.client.models.generate_content(model=model, contents=contents, config=generation_config)
            return response.text

        except Exception as e:
            logger.error(f"Gemini生成内容失败: {str(e)}")
            raise
    
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
            # 创建生成配置
            generation_config = GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=[system_prompt],
                max_output_tokens=self.max_tokens
            )

            # 使用generate_content方法处理请求
            response_text = await self.generate_content(
                model=self.model,
                contents=prompt,
                generation_config=generation_config,
                system_instruction=system_prompt
            )
            
            # 返回生成的文本
            return response_text
                
        except Exception as e:
            logger.error(f"Gemini生成文本失败: {str(e)}")
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
            embedding_model = self.additional_params.get("embedding_model", "embedding-001")
            
            # 使用Gemini的嵌入模型
            result = genai.embed_content(
                model=embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini获取嵌入向量失败: {str(e)}")
            raise 