from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class OrderItem(BaseModel):
    """单个订单项目的数据模型"""
    original_input: str
    product_id: Optional[str] = None
    quantity: float = 1.0
    unit_price: Optional[float] = None
    matched_name: Optional[str] = None
    match_score: float = 0.0
    needs_review: bool = True
    notes: Optional[str] = None
    unit: Optional[str] = None
    category: Optional[str] = None
    currency: Optional[str] = None
    
    
class ExtractedOrderItem(BaseModel):
    """从文档中提取的原始订单项目数据"""
    original_input: str
    product_name: str
    quantity: float = 1.0
    unit_price: Optional[float] = None
    notes: Optional[str] = None


class ExtractedOrderData(BaseModel):
    """从文档中提取的原始订单数据"""
    customer_name: str
    order_id: Optional[str] = None
    order_date: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    items: List[ExtractedOrderItem] = []
    total: Optional[float] = None
    currency: Optional[str] = "CNY"
    notes: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    
    
class OrderData(BaseModel):
    """订单数据的数据模型，包含产品匹配后的信息"""
    customer_name: str
    order_id: Optional[str] = None
    order_date: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    status: str = "pending"  # pending, completed, rejected
    items: List[OrderItem] = []
    total: Optional[float] = None
    currency: Optional[str] = "CNY"
    notes: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    
    
class ExtractRequest(BaseModel):
    """从文本提取订单信息的请求"""
    text: str
    customer_name: Optional[str] = None
    order_dir: Optional[str] = None
    ocr_file_path: Optional[str] = None
    task_id: str
    source_type: Optional[str] = "pdf"  # 数据来源类型: pdf 或 handwritten
    

class ProductData(BaseModel):
    """产品数据的数据模型"""
    product_code: str
    name: str
    name_en: Optional[str] = None
    price: float = 0.0
    unit: str
    currency: str = "NTD"
    category: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    
    
class ProductMatchRequest(BaseModel):
    """产品匹配请求的数据模型"""
    order_data: ExtractedOrderData
    threshold: float = 0.6  # 匹配阈值
    order_dir: Optional[str] = None
    task_id: str

class OutputOrderItem(BaseModel):
    """单个订单项目的数据模型"""
    product_id: Optional[str] = None
    matched_name: Optional[str] = None
    original_input: str
    quantity: float = 1.0
    match_score: float = 0.0

class OutputOrderData(BaseModel):
    """订单数据的数据模型，包含产品匹配后的信息"""
    customer_name: str
    order_date: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    status: str = "pending"  # pending, completed, rejected
    items: List[OutputOrderItem] = []