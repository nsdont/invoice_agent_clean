# 初始化OCR服务包
# 首先引入和初始化日志配置
from .shared.logger_config import configure_logger

# 配置根日志记录器
root_logger = configure_logger()

# 导入应用程序
from .app import app 