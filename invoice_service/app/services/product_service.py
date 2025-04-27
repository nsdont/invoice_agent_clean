import os
import json
import asyncio
import re
import unicodedata
from typing import List, Dict, Any
from thefuzz import process, fuzz  # 使用thefuzz代替fuzzywuzzy，提供更高效的模糊字符串匹配
from dotenv import load_dotenv
import uuid
from flask import current_app

from ..models import OrderData, OrderItem, ProductData, ExtractedOrderData, ExtractedOrderItem
from ..sdk import json_util
from ..shared.logger_config import configure_logger
from ..sdk.llm import LLMProvider, LLMService

# 加载环境变量
load_dotenv()

# 获取模块专属logger
logger = configure_logger()

# 加载API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or OPENAI_API_KEY  # 如果未设置，默认使用OpenAI API密钥

# 配置选择的模型
PRODUCT_SERVICE_MODEL = os.getenv("PRODUCT_SERVICE_MODEL", "fuzzy").lower()  # 默认使用fuzzy匹配
# 配置预筛选的最大产品数量
MAX_PRODUCT_CANDIDATES = int(os.getenv("MAX_PRODUCT_CANDIDATES", "20"))
# 配置预筛选的最低匹配阈值
PRODUCT_MATCH_THRESHOLD = float(os.getenv("PRODUCT_MATCH_THRESHOLD", "0.1"))
# 配置产品匹配的最大并行度
MAX_PRODUCT_CONCURRENCY = int(os.getenv("MAX_PRODUCT_CONCURRENCY", "10"))

# 初始化LLM服务
llm_service = None

def get_llm_service():
    """获取LLM服务实例"""
    global llm_service
    
    if llm_service is not None:
        return llm_service
    
    # 根据配置选择模型
    if PRODUCT_SERVICE_MODEL == "gemini" and GEMINI_API_KEY:
        provider = LLMProvider.GEMINI
        api_key = GEMINI_API_KEY
        model_name = "gemini-2.0-flash"
        logger.info("使用Gemini模型进行产品匹配")
    elif PRODUCT_SERVICE_MODEL == "deepseek" and DEEPSEEK_API_KEY:
        provider = LLMProvider.DEEPSEEK
        api_key = DEEPSEEK_API_KEY
        model_name = "deepseek-chat"
        logger.info("使用DeepSeek模型进行产品匹配")
    else:
        # 如果不是LLM模型，返回None
        if PRODUCT_SERVICE_MODEL == "fuzzy":
            return None
        
        # 默认使用OpenAI
        provider = LLMProvider.OPENAI
        api_key = OPENAI_API_KEY
        model_name = "gpt-4"
        logger.info("使用OpenAI模型进行产品匹配")
    
    # 创建LLM服务实例
    llm_service = LLMService(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=1000,
        additional_params={
            "response_format": {"type": "json_object"} if provider == LLMProvider.OPENAI else None
        }
    )
    
    return llm_service

# 记录当前使用的模型
logger.info(f"产品匹配服务使用模型: {PRODUCT_SERVICE_MODEL}")

class ProductMatcher:
    def __init__(self, products_file=None):
        """
        初始化产品匹配器
        
        Args:
            products_file: 产品数据文件路径，默认在shared目录下
        """
        # 尝试从Flask应用配置获取SHARED_DIR，如果不可用则使用环境变量或默认路径
        try:
            data_dir = current_app.config.get('DATA_DIR')
        except RuntimeError:
            # 如果不在Flask应用上下文中
            base_dir = os.environ.get('APP_BASE_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            data_dir = os.environ.get('APP_DATA_DIR', os.path.join(base_dir, 'data'))
            
        self.data_dir = data_dir
        self.products_file = products_file or os.path.join(self.data_dir, 'products.json')
        self.products = self._load_products()
        
    def _load_products(self) -> List[ProductData]:
        """加载产品数据"""
        try:
            if not os.path.exists(self.products_file):
                logger.warning(f"产品数据文件不存在: {self.products_file}，使用空列表")
                return []
                
            with open(self.products_file, 'r', encoding='utf-8') as f:
                products_data = json.load(f)
                
            # 转换为ProductData对象
            products = [ProductData(**p) for p in products_data]
            logger.info(f"成功加载{len(products)}个产品数据")
            return products
        except Exception as e:
            logger.error(f"加载产品数据失败: {str(e)}", exc_info=1)
            return []
    
    def _get_product_by_code(self, product_code: str) -> ProductData:
        """
        根据产品代码获取产品信息
        
        Args:
            product_code: 产品代码
            
        Returns:
            ProductData: 产品数据，如果未找到则返回None
        """
        if not product_code:
            return None
            
        return next((p for p in self.products if p.product_code == product_code), None)
    
    def _update_order_item_from_product(self, item: OrderItem, product: ProductData) -> OrderItem:
        """
        使用产品信息更新订单项
        
        Args:
            item: 订单项
            product: 产品数据
            
        Returns:
            OrderItem: 更新后的订单项
        """
        if not product:
            return item
            
        # 更新订单项信息
        item.matched_name = product.name
        item.product_id = product.product_code
        
        # 如果订单项没有单价，使用产品单价
        if not item.unit_price:
            item.unit_price = product.price
            
        # 补充订单项的其他属性
        item.unit = product.unit
        item.category = product.category
        item.currency = product.currency
        
        return item

    def _complete_order_with_product_info(self, order_data: OrderData) -> OrderData:
        """
        使用产品信息补全订单数据
        
        Args:
            order_data: 订单数据
            
        Returns:
            OrderData: 补全后的订单数据
        """
        # 如果没有订单项，直接返回
        if not order_data.items:
            return order_data
            
        updated_items = []
        total_price = 0.0
        
        # 更新每个订单项
        for item in order_data.items:
            # 如果有匹配的产品ID，获取产品信息并更新
            if item.product_id:
                product = self._get_product_by_code(item.product_id)
                if product:
                    item = self._update_order_item_from_product(item, product)
            
            # 计算项目总价
            if item.unit_price and item.quantity:
                item_total = item.quantity * item.unit_price
                total_price += item_total
                
            updated_items.append(item)
            
        # 更新订单项和总价
        order_data.items = updated_items
        
        # 如果订单总价为0或未设置，使用计算得到的总价
        if not order_data.total or order_data.total == 0:
            order_data.total = total_price
            
        # 如果未设置货币，使用第一个有货币的订单项的货币
        if not order_data.currency:
            for item in updated_items:
                if hasattr(item, 'currency') and item.currency:
                    order_data.currency = item.currency
                    break
                    
        return order_data

    def _normalize_text(self, text: str) -> str:
        """
        规范化文本以提高匹配准确度
        
        Args:
            text: 输入文本
            
        Returns:
            str: 规范化后的文本
        """
        if not text:
            return ""
            
        # 转为小写
        text = text.lower()
        
        # 规范化Unicode字符
        text = unicodedata.normalize('NFKC', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _check_fuzzy_perfect_match(self, query: str) -> tuple[ProductData, float]:
        """
        检查是否有完美匹配(100%匹配)的产品
        
        Args:
            query: 查询文本
            
        Returns:
            tuple: (完美匹配的产品, 匹配分数)，如果没有完美匹配则返回(None, 0)
        """
        # 规范化查询文本
        normalized_query = self._normalize_text(query)
        
        # 检查是否有完全匹配
        for product in self.products:
            # 检查主名称
            if self._normalize_text(product.name) == normalized_query:
                logger.info(f"找到完美匹配(主名称): '{query}' -> '{product.name}'")
                return product, 1.0
                
            # 检查英文名称(如果有)
            if hasattr(product, 'name_en') and product.name_en and self._normalize_text(product.name_en) == normalized_query:
                logger.info(f"找到完美匹配(英文名称): '{query}' -> '{product.name_en}'")
                return product, 1.0
                
            # 检查所有别名
            for alias in product.aliases:
                if self._normalize_text(alias) == normalized_query:
                    logger.info(f"找到完美匹配(别名): '{query}' -> '{alias}'")
                    return product, 1.0
        
        # 如果没有完全匹配，尝试使用高分匹配(token_sort_ratio)
        best_score = 0
        best_product = None
        
        for product in self.products:
            # 检查主名称
            score = fuzz.token_sort_ratio(normalized_query, self._normalize_text(product.name))
            if score == 100:
                logger.info(f"找到高精度匹配(token_sort_ratio): '{query}' -> '{product.name}'")
                return product, 1.0
            elif score > best_score:
                best_score = score
                best_product = product
        
        # 如果最佳分数超过95，也认为是高精度匹配
        if best_score >= 95:
            logger.info(f"找到高精度匹配(token_sort_ratio={best_score}): '{query}' -> '{best_product.name}'")
            return best_product, best_score / 100.0
            
        # 没有找到完美匹配
        return None, 0.0
        
    def _pre_filter_products(self, query: str, max_candidates: int = None, threshold: float = None) -> List[ProductData]:
        """
        使用模糊匹配预筛选产品
        
        Args:
            query: 查询文本
            max_candidates: 最大候选数量，默认使用环境变量配置
            threshold: 最低匹配阈值，低于此阈值的匹配将被排除，默认使用环境变量配置
            
        Returns:
            List[ProductData]: 筛选后的产品列表
        """
        # 使用参数值或默认全局配置
        max_candidates = max_candidates or MAX_PRODUCT_CANDIDATES
        threshold = threshold or PRODUCT_MATCH_THRESHOLD
        
        # 规范化查询文本
        normalized_query = self._normalize_text(query)
        
        # 创建产品名称和别名列表以及对应的产品索引
        product_names_with_idx = []
        
        for idx, p in enumerate(self.products):
            # 添加主名称
            product_names_with_idx.append((self._normalize_text(p.name), idx))
            # 添加英文名称(如果有)
            if hasattr(p, 'name_en') and p.name_en:
                product_names_with_idx.append((self._normalize_text(p.name_en), idx))
            # 添加所有别名
            for alias in p.aliases:
                normalized_alias = self._normalize_text(alias)
                if normalized_alias and normalized_alias != self._normalize_text(p.name) and (not hasattr(p, 'name_en') or normalized_alias != self._normalize_text(p.name_en)):
                    product_names_with_idx.append((normalized_alias, idx))
        
        # 提取名称列表用于匹配
        product_names = [name for name, _ in product_names_with_idx]
        
        # 尝试不同的匹配方法以提高多语言匹配效果
        try:
            # 首先使用token_sort_ratio，对多语言更友好
            matches = process.extractBests(
                normalized_query,
                product_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold * 100,  # 转换为百分比分数
                limit=max_candidates
            )
            
            # 如果匹配结果太少，尝试使用部分比率匹配
            if len(matches) < 5:
                partial_matches = process.extractBests(
                    normalized_query,
                    product_names,
                    scorer=fuzz.partial_ratio,
                    score_cutoff=threshold * 100,
                    limit=max_candidates
                )
                
                # 合并结果并去重
                all_matches = matches.copy()
                for match in partial_matches:
                    if match[0] not in [m[0] for m in all_matches]:
                        all_matches.append(match)
                
                # 如果合并后的结果比原来多，使用合并结果
                if len(all_matches) > len(matches):
                    matches = all_matches[:max_candidates]
        except Exception as e:
            logger.warning(f"高级匹配方法失败，回退到基本匹配: {str(e)}")
            # 回退到基本匹配
            matches = process.extractBests(
                normalized_query,
                product_names,
                score_cutoff=threshold * 100,
                limit=max_candidates
            )
        
        # 获取匹配的产品索引，去重
        matched_indices = set()
        for name, score in matches:
            idx = next(idx for pname, idx in product_names_with_idx if pname == name)
            matched_indices.add(idx)
        
        # 返回筛选后的产品
        filtered_products = [self.products[idx] for idx in matched_indices]
        logger.info(f"预筛选产品: '{query}' - 从{len(self.products)}个产品中选出{len(filtered_products)}个候选产品")
        return filtered_products

    def match_with_fuzzy(self, order_data: OrderData, threshold: float = 0.6) -> OrderData:
        """
        使用fuzzy匹配算法匹配产品数据
        
        Args:
            order_data: 订单数据
            threshold: 匹配阈值，低于此阈值的匹配将被标记为需要审核
            
        Returns:
            OrderData: 更新后的订单数据
        """
        logger.info("使用Fuzzy匹配算法进行产品匹配")
        
        if not self.products:
            logger.warning("产品列表为空，无法进行匹配")
            # 标记所有项目为需要审查
            for item in order_data.items:
                item.needs_review = True
            return order_data
        
        # 匹配每个订单项
        matched_items = []
        total_price = 0.0
        all_matched = True
        
        for idx, item in enumerate(order_data.items):
            logger.info(f"正在匹配第 {idx+1}/{len(order_data.items)} 个产品: '{item.original_input}'")
            
            # 先检查是否有完美匹配
            perfect_product, perfect_score = self._check_fuzzy_perfect_match(item.original_input)
            
            if perfect_product and perfect_score >= threshold:
                # 使用完美匹配结果
                logger.info(f"使用完美匹配结果: 匹配度 {perfect_score:.2f}")
                
                # 更新订单项
                item.matched_name = perfect_product.name
                item.product_id = perfect_product.product_code
                item.unit_price = perfect_product.price
                item.match_score = perfect_score
                item.needs_review = perfect_score < 0.8  # 低置信度匹配仍需审核
                
                # 补充产品信息
                item = self._update_order_item_from_product(item, perfect_product)
                
                # 计算项目总价
                item_total = item.quantity * perfect_product.price
                total_price += item_total
            else:
                # 使用相同的预筛选方法来获取最可能的匹配
                filtered_products = self._pre_filter_products(item.original_input, max_candidates=1)
                
                if filtered_products:
                    # 获取最佳匹配产品
                    matched_product = filtered_products[0]
                    
                    # 对于fuzzy匹配，再次计算精确的匹配分数
                    normalized_query = self._normalize_text(item.original_input)
                    normalized_name = self._normalize_text(matched_product.name)
                    
                    # 计算token_sort_ratio得分，对多语言更友好
                    score = fuzz.token_sort_ratio(normalized_query, normalized_name)
                    normalized_score = score / 100.0
                    
                    if normalized_score >= threshold:
                        # 更新订单项
                        item.matched_name = matched_product.name
                        item.product_id = matched_product.product_code
                        item.unit_price = matched_product.price
                        item.match_score = normalized_score
                        item.needs_review = normalized_score < 0.8  # 低置信度匹配仍需审核
                        
                        # 补充产品信息
                        item = self._update_order_item_from_product(item, matched_product)
                        
                        # 计算项目总价
                        item_total = item.quantity * matched_product.price
                        total_price += item_total
                    else:
                        # 标记为需要审核
                        item.match_score = normalized_score
                        item.needs_review = True
                        item.matched_name = matched_product.name
                        all_matched = False
                else:
                    # 没有找到任何匹配
                    item.match_score = 0
                    item.needs_review = True
                    all_matched = False
                
            matched_items.append(item)
            
        # 更新订单状态和总价
        order_data.items = matched_items
        order_data.total = total_price
        order_data.status = "completed" if all_matched else "pending"
        
        # 补充整个订单的信息
        order_data = self._complete_order_with_product_info(order_data)
        
        logger.info(f"Fuzzy产品匹配完成: {len(matched_items)}个项目, 状态:{order_data.status}")
        return order_data

    async def _match_item_with_llm(self, item: OrderItem, filtered_products: List[ProductData], system_prompt: str, idx: int = 0) -> Dict:
        """
        使用LLM匹配单个订单项
        
        Args:
            item: 订单项
            filtered_products: 预筛选的产品列表
            system_prompt: 系统提示词
            idx: 订单项索引，用于日志跟踪
            
        Returns:
            Dict: 匹配结果
        """
        service = get_llm_service()
        if not service:
            logger.error("LLM服务未初始化", exc_info=1)
            return {}
        
        # 转换筛选后的产品为JSON，只包含必要的字段：name、aliases和unit
        products_json = json.dumps([{
            "product_code": p.product_code,
            "name": p.name,
            "unit": p.unit,
            "aliases": p.aliases
        } for p in filtered_products], ensure_ascii=False)
            
        user_prompt = f"""
Order item name: "{item.original_input}"

Product catalog: 
{products_json}

Please return the result in JSON format:
{{
  "product_code": "code of the best matching product",
  "matched_name": "name of the best matching product",
  "confidence": confidence score (between 0 and 1),
  "unit_price": unit price,
  "needs_review": whether manual review is needed (true/false)
}}

Matches with confidence lower than 0.7 should be marked for review (needs_review=true).
If no match is found, return empty code and name with confidence 0, and mark for review.
Only return JSON format, without MARKDOWN code block..
"""
        
        try:
            # 调用LLM服务
            start_time = asyncio.get_event_loop().time()
            response_text = await service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            elapsed_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"LLM调用完成，项目[{idx}]耗时: {elapsed_time:.2f}秒")
            
            # 生成唯一跟踪ID
            task_id = f"item_{idx}_{int(start_time)}"
            return json_util.parse_json(response_text, logger=logger, task_id=task_id)
        except Exception as e:
            logger.error(f"调用LLM服务出错: {str(e)}", exc_info=1)
            return {}

    def match_with_llm(self, order_data: OrderData, threshold: float = 0.6) -> OrderData:
        """使用LLM服务匹配产品数据"""
        logger.info(f"使用{PRODUCT_SERVICE_MODEL}模型进行产品匹配")
        
        if not self.products:
            logger.warning("产品列表为空，无法进行匹配")
            # 标记所有项目为需要审查
            for item in order_data.items:
                item.needs_review = True
            return order_data
        
        system_prompt = """You are a product matching expert. Your task is to match order items with products from the product catalog.

Important: You need to handle multilingual scenarios - items may be in English, Simplified Chinese, Traditional Chinese, or a mix of these languages.
Consider product aliases, which are alternative names for the same product in different languages or formats.

Analyze thoughtfully, but only return your answer in JSON format, without markdown code block.
"""
        
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 创建信号量控制并发度
            semaphore = asyncio.Semaphore(MAX_PRODUCT_CONCURRENCY)
            
            # 定义匹配单个订单项的异步函数
            async def match_single_item(order_item: OrderItem, idx: int):
                logger.info(f"开始处理第 {idx+1}/{len(order_data.items)} 个产品: '{order_item.original_input}'")
                
                # 先检查是否有完美匹配，如果有则无需调用LLM
                perfect_product, perfect_score = self._check_fuzzy_perfect_match(order_item.original_input)
                
                if perfect_product and perfect_score > 0.95:
                    # 使用完美匹配结果
                    logger.info(f"使用完美匹配结果，无需调用LLM: 匹配度 {perfect_score:.2f}")
                    
                    # 更新订单项
                    order_item.product_id = perfect_product.product_code
                    order_item.matched_name = perfect_product.name
                    order_item.match_score = perfect_score
                    order_item.unit_price = perfect_product.price
                    order_item.needs_review = False  # 完美匹配无需审核
                    
                    # 补充产品信息
                    order_item = self._update_order_item_from_product(order_item, perfect_product)
                    
                    return order_item, True
                
                # 没有完美匹配，准备使用LLM匹配
                # 预筛选产品(这一步耗时较少，可以在semaphore外执行)
                filtered_products = self._pre_filter_products(order_item.original_input)
                
                # 使用信号量限制LLM调用的并发
                async with semaphore:
                    logger.info(f"开始LLM匹配第 {idx+1}/{len(order_data.items)} 个产品")
                    # 调用LLM匹配
                    result = await self._match_item_with_llm(order_item, filtered_products, system_prompt, idx)
                
                # 解析并更新结果(这一步也可以在semaphore外执行)
                if result:
                    # 更新订单项
                    order_item.product_id = result.get("product_code")
                    order_item.matched_name = result.get("matched_name")
                    order_item.match_score = result.get("confidence", 0)
                    order_item.unit_price = result.get("unit_price")
                    order_item.needs_review = result.get("needs_review", True)
                    
                    # 补充产品信息
                    product = self._get_product_by_code(order_item.product_id)
                    if product:
                        order_item = self._update_order_item_from_product(order_item, product)
                else:
                    order_item.needs_review = True
                
                return order_item, bool(result and order_item.match_score > threshold)
            
            # 创建任务列表
            tasks = [match_single_item(item, idx) for idx, item in enumerate(order_data.items)]
            
            # 并行执行所有任务
            logger.info(f"开始并行处理 {len(tasks)} 个产品匹配任务，最大并发度: {MAX_PRODUCT_CONCURRENCY}")
            results = loop.run_until_complete(asyncio.gather(*tasks))
            
            # 处理结果
            matched_items = []
            total_price = 0.0
            all_matched = True
            
            for order_item, is_matched in results:
                matched_items.append(order_item)
                
                # 计算项目总价
                if order_item.unit_price and order_item.quantity:
                    item_total = order_item.quantity * order_item.unit_price
                    total_price += item_total
                
                # 检查是否全部匹配
                if not is_matched:
                    all_matched = False
            
            # 更新订单状态和总价
            order_data.items = matched_items
            order_data.total = total_price
            order_data.status = "completed" if all_matched else "pending"
            
            # 补充整个订单的信息
            order_data = self._complete_order_with_product_info(order_data)
            
            logger.info(f"LLM产品匹配完成: {len(matched_items)}个项目, 状态:{order_data.status}")
            return order_data
        finally:
            # 确保关闭事件循环
            loop.close()

    def match_products(self, order_data: ExtractedOrderData, threshold: float = 0.6) -> OrderData:
        """
        根据配置选择匹配方法
        
        Args:
            order_data: 提取的订单数据
            threshold: 匹配阈值
            
        Returns:
            OrderData: 匹配后的订单数据
        """
        logger.info(f"使用{PRODUCT_SERVICE_MODEL}进行产品匹配")
        
        # 先将ExtractedOrderData转换为OrderData
        matched_order = OrderData(
            customer_name=order_data.customer_name,
            order_id=order_data.order_id,
            order_date=order_data.order_date,
            items=[],
            total=order_data.total,
            currency=order_data.currency,
            notes=order_data.notes,
            meta=order_data.meta
        )
        
        # 转换订单项，现在将product_name保存到original_input中用于匹配
        for extracted_item in order_data.items:
            order_item = OrderItem(
                original_input=extracted_item.product_name,  # 使用product_name替代original_input用于匹配
                quantity=extracted_item.quantity,
                unit_price=extracted_item.unit_price,
                notes=extracted_item.notes,
                product_id=None,
                matched_name=None,
                match_score=0.0,
                needs_review=True
            )
            matched_order.items.append(order_item)
        
        # 根据配置选择匹配方法
        if PRODUCT_SERVICE_MODEL in ["openai", "gemini", "deepseek"]:
            return self.match_with_llm(matched_order, threshold)
        else:
            return self.match_with_fuzzy(matched_order, threshold)

    def save_order(self, order_data: OrderData) -> str:
        """
        保存订单数据到本地文件
        
        Args:
            order_data: 订单数据对象
            
        Returns:
            str: 保存的文件路径
        """
        # 确保订单有ID
        if not order_data.order_id:
            order_data.order_id = str(uuid.uuid4())
            
        # 构建文件名和路径
        orders_dir = os.path.join(self.data_dir, 'orders')
        os.makedirs(orders_dir, exist_ok=True)
        
        file_path = os.path.join(orders_dir, f"order_{order_data.order_id}.json")
        
        # 转换为字典并保存
        order_dict = order_data.model_dump()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(order_dict, f, ensure_ascii=False, indent=2)
            
        logger.info(f"订单已保存: {file_path}")
        return file_path 