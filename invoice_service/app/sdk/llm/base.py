"""
LLM基础模块

定义LLM基类和枚举
"""
import abc
from enum import Enum, auto
from typing import List, Dict, Any, Optional


class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    
    def __str__(self):
        return self.value


class LLMClient(abc.ABC):
    """
    LLM客户端基类
    
    定义所有LLM客户端必须实现的接口
    """
    
    def __init__(self, config: 'LLMConfig'):
        """
        初始化LLM客户端
        
        Args:
            config: LLM配置
        """
        self.config = config
        self._initialize()
    
    def _initialize(self) -> None:
        """
        初始化客户端资源
        
        子类可以重写此方法以实现特定的初始化逻辑
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """
        获取文本嵌入向量
        
        Args:
            text: 需要嵌入的文本
            
        Returns:
            List[float]: 嵌入向量
        """
        pass

    @abc.abstractmethod
    async def generate_content(self, model, contents, generation_config, system_instruction) -> str:
        """
        生成内容（支持多模态输入）

        Args:
            model: 模型名称（如果不提供，使用默认配置的模型）
            contents: 输入内容（文本、图像或组合）
            generation_config: 生成配置
            system_instruction: 系统指令
            **kwargs: 额外参数

        Returns:
            GenerateContentResponse: 生成的内容响应
        """
        pass


class LLMConfig:
    """
    LLM配置类
    
    存储LLM客户端的配置参数
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 600,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        additional_params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化LLM配置
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            api_base: API基础URL
            timeout: 超时时间(秒)
            max_tokens: 最大生成令牌数
            temperature: 温度参数
            additional_params: 额外参数
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_params = additional_params or {} 