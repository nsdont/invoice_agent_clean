# 引入日志配置
from ..shared.logger_config import configure_logger

# 配置services模块日志记录器
logger = configure_logger(__name__)
logger.info("服务模块初始化")

# 导出服务函数和变量
from .ocr_service import process_image_ocr, GEMINI_AVAILABLE
from .order_service import extract_order_from_text
from .product_service import ProductMatcher 