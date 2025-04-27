import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Literal
from flask import current_app

from ..models import ExtractedOrderData, ExtractedOrderItem
from ..sdk import json_util
from ..shared.logger_config import configure_logger
from ..sdk.llm import LLMProvider, create_client

# 获取模块专属logger
logger = configure_logger()

# 加载环境变量和API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or OPENAI_API_KEY  # 如果未设置，默认使用OpenAI API密钥

# 配置选择的模型
ORDER_SERVICE_MODEL = os.getenv("ORDER_SERVICE_MODEL", "deepseek").lower()  # 默认使用DeepSeek

# 存储客户端实例
_llm_clients = {}

# 设置系统提示和示例 - 文字版PDF提示词
PDF_PROMPT_TEMPLATE = """You are an order information extraction assistant. Extract key information from order text into valid JSON format.

FORMAT REQUIREMENTS:
- ONLY return a valid JSON object, nothing else
- DO NOT include ANY explanations or notes
- DO NOT use markdown code blocks (```json ... ```)
- DO NOT add any formatting symbols around the JSON
- Start your response with {{ and end with }}
- Use null for missing values

CRITICAL: Return ONLY the raw JSON object without any markdown formatting or code blocks.

Extract order information from this text into JSON format:

{custom_name}
{text}

JSON SCHEMA:
{{
  "customer_name": "string",
  "order_id": "string or null",
  "order_date": "YYYY-MM-DD",
  "items": [
    {{
      "original_input": "string",
      "product_name": "string",
      "quantity": number,
      "unit_price": number or null,
      "notes": "string or null"
    }}
  ],
  "total": number or null,
  "currency": "string",
  "notes": "string or null",
  "meta": {{}}
}}"""

# 设置系统提示和示例 - 手写订单照片提示词
HANDWRITTEN_PROMPT_TEMPLATE = """You are an order information extraction assistant specializing in Chinese handwritten orders. Extract key information from order text into valid JSON format.

FORMAT REQUIREMENTS:
- ONLY return a valid JSON object, nothing else
- DO NOT include ANY explanations or notes
- DO NOT use markdown code blocks (```json ... ```)
- DO NOT add any formatting symbols around the JSON
- Start your response with {{ and end with }}
- Use null for missing values

CRITICAL: Return ONLY the raw JSON object without any markdown formatting or code blocks.

EXTRACTION RULES:
1. For "quantity", extract ONLY the numeric value (e.g. "8斤" should be quantity=8.0, notes="斤", "10斤 3x3" should be quantity=10, notes="斤 3x3")
2. For "notes", include any size specifications or additional descriptions (e.g. "3×3" or special instructions)
3. DO NOT multiply values in "quantity" (e.g. "12×1242塊" should be quantity=42, notes="12x12")
4. Chinese units like "斤", "支", "塊" should NOT be included in quantity
5. If the text contains multiple numbers, the primary quantity is usually the number followed by a unit (斤, 支, etc.)

Extract order information from this text into JSON format:

{custom_name}
{text}

JSON SCHEMA:
{{
  "customer_name": "string",
  "order_id": "string or null",
  "order_date": "YYYY-MM-DD",
  "items": [
    {{
      "original_input": "string",
      "product_name": "string",
      "quantity": number,
      "unit_price": number or null,
      "notes": "string or null"
    }}
  ],
  "total": number or null,
  "currency": "string",
  "notes": "string or null",
  "meta": {{}}
}}"""

class OrderService:
    """订单处理服务，处理OCR文本提取和结构化信息提取"""
    
    def __init__(self):
        """初始化订单服务"""
        # 尝试从Flask应用配置获取目录信息
        try:
            shared_dir = current_app.config.get('SHARED_DIR')
        except RuntimeError:
            # 如果不在Flask应用上下文中
            base_dir = os.environ.get('APP_BASE_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            shared_dir = os.environ.get('APP_SHARED_DIR', os.path.join(base_dir, 'shared'))

        # 设置存储目录
        self.shared_dir = shared_dir
        self.orders_dir = os.path.join(self.shared_dir, 'orders')
        
        # 确保目录存在
        os.makedirs(self.shared_dir, exist_ok=True)
        os.makedirs(self.orders_dir, exist_ok=True)
    
    def get_llm_client(self, provider=None):
        """
        获取指定提供商的LLM客户端实例
        
        Args:
            provider: LLM提供商名称，如果为None则使用环境变量配置
            
        Returns:
            LLMClient: LLM客户端实例
        """
        global _llm_clients
        
        # 如果未指定provider，使用环境变量配置
        if provider is None:
            provider = ORDER_SERVICE_MODEL
        
        # 如果客户端已存在，直接返回
        if provider in _llm_clients:
            return _llm_clients[provider]
        
        # 根据提供商创建对应的客户端
        try:
            if provider == "gemini" and GEMINI_API_KEY:
                client = create_client(
                    provider=LLMProvider.GEMINI,
                    api_key=GEMINI_API_KEY,
                    model_name="gemini-2.0-flash",
                    temperature=0.2,
                    max_tokens=50000
                )
                logger.info("Gemini client created successfully")
            elif provider == "openai" and OPENAI_API_KEY:
                client = create_client(
                    provider=LLMProvider.OPENAI,
                    api_key=OPENAI_API_KEY,
                    model_name="gpt-4",
                    temperature=0.2,
                    max_tokens=1000,
                    additional_params={
                        "response_format": {"type": "json_object"}
                    }
                )
                logger.info("OpenAI client created successfully")
            else:  # 默认使用DeepSeek
                client = create_client(
                    provider=LLMProvider.DEEPSEEK,
                    api_key=DEEPSEEK_API_KEY,
                    model_name="deepseek-chat",
                    temperature=0.2,
                    max_tokens=8192
                )
                logger.info("DeepSeek client created successfully")
            
            # 缓存客户端实例
            _llm_clients[provider] = client
            return client
            
        except Exception as e:
            logger.error(f"Failed to create LLM client: {str(e)}", exc_info=1)
            raise
    
    async def extract_text_with_llm(self, text: str, customer_name: Optional[str] = None, provider: Optional[str] = None, source_type: Literal["pdf", "handwritten"] = "pdf") -> Dict:
        """
        使用LLM提取订单信息
        
        Args:
            text: 订单文本
            customer_name: 可选的客户名称
            provider: 可选的LLM提供商，如果为None则使用环境变量配置
            source_type: 数据来源类型，"pdf"表示文字版PDF，"handwritten"表示手写订单照片
            
        Returns:
            Dict: 提取的订单信息
        """
        # 如果未指定provider，使用环境变量配置
        provider = provider or ORDER_SERVICE_MODEL
        logger.info(f"Using {provider} model to extract order information from {source_type}")
        
        try:
            # 获取对应的LLM客户端
            client = self.get_llm_client(provider)
            
            # 根据来源类型选择不同的提示词模板
            if source_type == "handwritten":
                prompt_template = HANDWRITTEN_PROMPT_TEMPLATE
            else:  # 默认使用PDF提示词
                prompt_template = PDF_PROMPT_TEMPLATE
            
            # 构建提示
            prompt = prompt_template.format(text=text, custom_name=f"Known customer name: {customer_name}" if customer_name else "")

            # logger.debug(f"extract_text_with_llm Prompt: {prompt}")
            logger.info(f"ready run extract_text_with_llm with {source_type} Prompt")
            # 调用LLM客户端
            response_text = await client.generate_text(
                prompt
            )
            
            logger.info(f"LLM response: {response_text}")
            
            # 解析响应
            return json_util.parse_json(response_text, logger=logger)
        except Exception as e:
            logger.error(f"Error calling LLM service: {str(e)}", exc_info=1)
            return {}
    
    def extract_order_from_text(self, text: str, customer_name: Optional[str] = None, provider: Optional[str] = None, source_type: Literal["pdf", "handwritten"] = "pdf") -> ExtractedOrderData:
        """
        从文本中提取订单信息，根据环境变量选择不同的LLM
        
        Args:
            text: OCR提取的文本
            customer_name: 可选的已知客户名称
            provider: 可选的LLM提供商，如果为None则使用环境变量配置
            source_type: 数据来源类型，"pdf"表示文字版PDF，"handwritten"表示手写订单照片
            
        Returns:
            ExtractedOrderData: 提取的订单数据对象
        """
        try:
            logger.info(f"Starting to extract order information from {source_type} text")
            
            # 异步调用LLM服务
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.extract_text_with_llm(text, customer_name, provider, source_type))
            loop.close()
            
            # 使用客户提供的名称（如果有）
            if customer_name and not result.get('customer_name'):
                result['customer_name'] = customer_name
                
            # 确保日期格式正确
            if not result.get('order_date'):
                result['order_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # 创建ExtractedOrderItem对象
            items = [ExtractedOrderItem(
                original_input=item['original_input'],
                product_name=item.get('product_name', ''),
                quantity=item.get('quantity', 1),
                unit_price=item.get('unit_price'),
                notes=item.get('notes')
            ) for item in result.get('items', [])]
            
            # 生成订单ID（如果没有提供）
            if not result.get('order_id'):
                result['order_id'] = str(uuid.uuid4())
            
            # 创建ExtractedOrderData对象
            order_data = ExtractedOrderData(
                customer_name=result.get('customer_name', 'Unknown customer'),
                order_id=result.get('order_id'),
                order_date=result.get('order_date'),
                items=items,
                total=result.get('total'),
                currency=result.get('currency', 'CNY'),
                notes=result.get('notes'),
                meta=result.get('meta', {})
            )
            
            logger.info(f"Order data extracted successfully, containing {len(items)} items")
            return order_data
            
        except Exception as e:
            logger.error(f"Order information extraction failed: {str(e)}", exc_info=1)
            raise
    
    def save_extraction_result(self, extraction_result: ExtractedOrderData, ocr_text: str, file_id: Optional[str] = None) -> Dict[str, str]:
        """
        保存提取结果到按UUID组织的目录
        
        Args:
            extraction_result: 提取的订单数据
            ocr_text: OCR提取的原始文本
            file_id: 文件ID，如果未提供则使用订单ID或生成新的UUID
            
        Returns:
            Dict[str, str]: 包含保存文件路径的字典
        """
        # 确保有file_id
        if not file_id:
            file_id = extraction_result.order_id or str(uuid.uuid4())
        
        # 创建订单专属目录
        order_dir = os.path.join(self.orders_dir, file_id)
        os.makedirs(order_dir, exist_ok=True)
        
        # 保存OCR文本
        ocr_file = os.path.join(order_dir, "ocr_text.txt")
        with open(ocr_file, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        
        # 保存提取的订单数据
        extraction_file = os.path.join(order_dir, "extracted_order.json")
        with open(extraction_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_result.dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存提取结果到目录: {order_dir}")
        
        return {
            "order_id": file_id,
            "order_dir": order_dir,
            "ocr_file": ocr_file,
            "extraction_file": extraction_file
        }

# 创建全局服务实例
order_service = OrderService()

# 兼容现有API的辅助函数
def get_llm_client(provider=None):
    """获取LLM客户端 - 兼容API"""
    return order_service.get_llm_client(provider)

async def extract_text_with_llm(text: str, customer_name: Optional[str] = None, provider: Optional[str] = None, source_type: Literal["pdf", "handwritten"] = "pdf") -> Dict:
    """使用LLM提取文本 - 兼容API"""
    return await order_service.extract_text_with_llm(text, customer_name, provider, source_type)

def extract_order_from_text(text: str, customer_name: Optional[str] = None, provider: Optional[str] = None, source_type: Literal["pdf", "handwritten"] = "pdf") -> ExtractedOrderData:
    """从文本提取订单 - 兼容API"""
    return order_service.extract_order_from_text(text, customer_name, provider, source_type)

def save_extraction_result(extraction_result: ExtractedOrderData, ocr_text: str, file_id: Optional[str] = None) -> Dict[str, str]:
    """保存提取结果 - 新API"""
    return order_service.save_extraction_result(extraction_result, ocr_text, file_id) 