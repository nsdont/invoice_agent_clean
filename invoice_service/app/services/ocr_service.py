import os
import uuid
from typing import Any, List, Dict, Optional, Union
import shutil
from pathlib import Path

import cv2
import pytesseract
import base64
import json
import time
import requests
from PIL import Image
import asyncio
import numpy as np
import tempfile
from flask import current_app

# 尝试导入pdf2image库
try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from invoice_service.app.sdk.llm import LLMClient
# 从新的LLM SDK引入
from ..sdk.llm import LLMProvider, create_client, LLMClient

# 引入统一日志配置
from ..shared.logger_config import configure_logger

# 获取模块专属logger
logger = configure_logger(__name__)

# 获取Google Gemini API密钥
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)

# 存储客户端实例
_llm_clients : dict[str, LLMClient] = {}

class OCRProcessor:
    """OCR处理器类，用于处理图像和PDF文件并提取文本"""
    
    def __init__(self):
        """初始化OCR处理器"""
        # 尝试从Flask应用配置获取目录信息
        try:
            shared_dir = current_app.config.get('SHARED_DIR')
            temp_dir = current_app.config.get('TEMP_DIR')
        except RuntimeError:
            # 如果不在Flask应用上下文中
            base_dir = os.environ.get('APP_BASE_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            shared_dir = os.environ.get('APP_SHARED_DIR', os.path.join(base_dir, 'shared'))
            temp_dir = os.environ.get('APP_TEMP_DIR', os.path.join(base_dir, 'temp'))
        
        # 设置图像存储目录
        self.image_storage_dir = os.path.join(shared_dir, "image")
        self.shared_dir = shared_dir
        self.temp_dir = temp_dir
        os.makedirs(self.image_storage_dir, exist_ok=True)
        
        # 初始化可用的OCR引擎
        self.gemini_available = GEMINI_AVAILABLE
        self.easyocr_available = False
        
        # 只有当Gemini不可用时，尝试初始化EasyOCR
        if not self.gemini_available:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en', 'ch_tra'])
                self.easyocr_available = True
                logger.info("EasyOCR successfully loaded")
            except ImportError as e:
                logger.error(f"EasyOCR import error: {str(e)}", exc_info=1)
        else:
            logger.info("Skipping EasyOCR import since Gemini is available")
            
        # 检查PDF支持
        if not PDF_SUPPORT:
            logger.warning("pdf2image 库未安装，PDF处理功能不可用。请安装：pip install pdf2image")
        
        # 如果Gemini可用，初始化客户端
        if self.gemini_available:
            self.gemini_client = self._get_llm_client("gemini")
            if self.gemini_client:
                logger.info("Gemini客户端初始化成功")
            else:
                logger.warning("Gemini客户端初始化失败")
                self.gemini_available = False
    
    def _get_llm_client(self, provider) -> Optional[LLMClient]:
        """
        获取指定提供商的LLM客户端实例
        
        Args:
            provider: LLM提供商名称
            
        Returns:
            LLMClient: LLM客户端实例
        """
        global _llm_clients
        
        # 如果客户端已存在，直接返回
        if provider in _llm_clients:
            return _llm_clients[provider]
        
        # 根据提供商创建对应的客户端
        try:
            if provider == "gemini" and GEMINI_API_KEY:
                client = create_client(
                    provider=LLMProvider.GEMINI,
                    api_key=GEMINI_API_KEY,
                    model_name="gemini-2.0-flash"
                )
                logger.info("创建Gemini客户端成功")
                # 缓存客户端实例
                _llm_clients[provider] = client
                return client
            else:
                logger.error(f"不支持的LLM提供商或API密钥未设置: {provider}", exc_info=1)
                return None
        except Exception as e:
            logger.error(f"创建LLM客户端失败: {str(e)}", exc_info=1)
            return None
    
    def is_pdf_file(self, file_path: str) -> bool:
        """
        检查文件是否为PDF文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否为PDF文件
        """
        if not file_path:
            return False
        return file_path.lower().endswith('.pdf')
    
    def save_image_to_storage(self, image_path: str, suffix: str = "", target_dir: str = None) -> str:
        """
        将图片保存到存储目录
        
        Args:
            image_path: 原始图片路径
            suffix: 文件名后缀（用于区分原图和处理后的图）
            target_dir: 目标保存目录，如果为None则使用默认的image_storage_dir
            
        Returns:
            str: 保存后的图片路径
        """
        try:
            # 生成唯一文件名
            original_filename = os.path.basename(image_path)
            filename, ext = os.path.splitext(original_filename)
            unique_id = str(uuid.uuid4())[:8]
            
            # 构建新文件名和路径
            if suffix:
                new_filename = f"{filename}_{suffix}_{unique_id}{ext}"
            else:
                new_filename = f"{filename}_{unique_id}{ext}"
            
            # 确定目标保存目录
            save_dir = target_dir if target_dir else self.image_storage_dir
            os.makedirs(save_dir, exist_ok=True)
            
            storage_path = os.path.join(save_dir, new_filename)
            
            # 复制文件到存储目录
            shutil.copy2(image_path, storage_path)
            logger.info(f"图片已保存至: {storage_path}")
            
            return storage_path
        except Exception as e:
            logger.error(f"保存图片到存储目录失败: {str(e)}", exc_info=True)
            return image_path
    
    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 300, target_dir: str = None) -> List[str]:
        """
        将PDF文件转换为图像文件并保存到存储目录
        
        Args:
            pdf_path: PDF文件路径
            dpi: PDF转换为图像的DPI
            target_dir: 目标保存目录，如果为None则使用默认的image_storage_dir
            
        Returns:
            List[str]: 保存的图像文件路径列表
        """
        if not PDF_SUPPORT:
            logger.error("pdf2image 库未安装，无法处理PDF文件", exc_info=1)
            raise ImportError("pdf2image 库未安装，无法处理PDF文件。请安装：pip install pdf2image")
            
        try:
            logger.info(f"开始将PDF转换为图像: {pdf_path}, DPI: {dpi}")
            
            # 保存原始PDF到存储目录
            stored_pdf_path = self.save_image_to_storage(pdf_path, suffix="original", target_dir=target_dir)
            logger.info(f"原始PDF已保存至: {stored_pdf_path}")
            
            # 转换PDF为图像
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            logger.info(f"PDF转换完成，共 {len(images)} 页")
            
            # 保存转换后的图像到临时目录
            image_paths = []
            for i, image in enumerate(images):
                # 创建临时文件保存图像
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                    temp_path = temp.name
                    
                # 保存图像到临时文件
                image.save(temp_path, 'PNG')
                
                # 保存到存储目录
                page_suffix = f"page_{i+1}"
                stored_image_path = self.save_image_to_storage(temp_path, suffix=page_suffix, target_dir=target_dir)
                image_paths.append(stored_image_path)
                
                # 删除临时文件
                os.unlink(temp_path)
                
            logger.info(f"PDF转换为图像完成，保存了 {len(image_paths)} 个图像文件")
            return image_paths
            
        except Exception as e:
            logger.error(f"PDF转换为图像失败: {str(e)}", exc_info=True)
            raise
    
    def preprocess_image(self, image_path: str, target_dir: str = None) -> Optional[str]:
        """
        预处理图像以提高OCR质量
        
        Args:
            image_path: 图像文件路径
            target_dir: 目标保存目录，如果为None则使用默认的image_storage_dir
            
        Returns:
            str: 处理后的图像路径，如果处理失败则返回None
        """
        try:
            # 首先保存原始图片到存储目录
            stored_original_path = self.save_image_to_storage(image_path, suffix="original", target_dir=target_dir)
            logger.info(f"原始图片已保存至: {stored_original_path}")
            
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to load image with OpenCV: {image_path}")
                return None
                
            # 灰度化
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 自适应二值化
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 降噪
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # 生成处理后图片的临时路径
            temp_processed_path = f"{os.path.splitext(image_path)[0]}_processed_temp.png"
            cv2.imwrite(temp_processed_path, denoised)
            
            # 保存处理后的图片到存储目录
            processed_storage_path = self.save_image_to_storage(temp_processed_path, suffix="processed", target_dir=target_dir)
            logger.info(f"处理后图片已保存至: {processed_storage_path}")
            
            # 删除临时文件
            if os.path.exists(temp_processed_path):
                os.remove(temp_processed_path)
                
            return processed_storage_path
        except Exception as e:
            logger.error(f"图像预处理错误: {str(e)}", exc_info=True)
            return None
    
    async def extract_text_with_gemini_async(self, image_path: str) -> Optional[str]:
        """
        使用Google Gemini 2.0 Flash模型提取图像中的文本 (异步版本)
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: 提取的文本，如果提取失败则返回None
        """
        if not self.gemini_available:
            logger.error("Gemini API key not set or client initialization failed", exc_info=1)
            return None
            
        try:
            logger.info(f"开始使用Gemini处理图像: {image_path}")
            
            # 获取Gemini客户端
            client = self._get_llm_client("gemini")
            if client is None:
                logger.error("无法获取Gemini客户端", exc_info=1)
                return None
            
            # 读取图像
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                
            # 准备OCR提示词
            prompt_text = """
            You are a specialized OCR (Optical Character Recognition) system. Extract all text content from this image with high precision. Follow these guidelines:
            
            1. Return ONLY the extracted text, without any analysis, comments, or explanations
            2. Preserve the original structure and formatting (paragraphs, bullets, tables, etc.)
            3. Maintain all numbers, dates, and special characters exactly as they appear
            4. If the content is in Chinese or any other language, output it in its original language
            5. For tables, preserve the tabular structure using spaces or tabs
            6. Include all header and footer text
            7. Maintain the reading order (left-to-right, top-to-bottom for most languages)
            8. Do not attempt to interpret or summarize the content
            
            CRITICAL: Return ONLY the raw text without any markdown formatting or code blocks.
            
            Act as a pure text extraction tool, providing only the raw text content from the image.
            """
            
            # 使用Gemini客户端处理图像 - 使用正确的API调用方式
            from google.genai import types
            
            # 构建请求内容
            contents = [
                prompt_text,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )
            ]
            
            # 调用模型生成内容
            extracted_text = await client.generate_content(
                model="gemini-2.0-flash",
                contents=contents
            )

            logger.info(f"Gemini成功提取文本，文本长度: {len(extracted_text)}")
            text_preview = extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text
            logger.debug(f"文本预览: {text_preview}")
            return extracted_text

        except Exception as e:
            logger.error(f"Gemini处理异常: {str(e)}", exc_info=True)
            return None
    
    def extract_text_with_gemini(self, image_path: str) -> Optional[str]:
        """
        使用Google Gemini 2.0 Flash模型提取图像中的文本 (同步版本)
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: 提取的文本，如果提取失败则返回None
        """
        # 创建事件循环并运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.extract_text_with_gemini_async(image_path))
            return result
        finally:
            loop.close()
    
    def process_image_ocr(self, image_path: str, engine: str = 'auto', lang: str = 'chi_sim+eng', images_dir: str = None, ocr_dir: str = None, already_saved: bool = False) -> Dict[str, Any]:
        """
        处理图像并提取文本
        
        Args:
            image_path: 图像文件路径
            engine: OCR引擎 ('auto', 'tesseract', 'easyocr', 'gemini')
            lang: 语言设置 (tesseract格式)
            images_dir: 图像存储目录，如果提供则将图像保存到此目录
            ocr_dir: OCR结果存储目录，如果提供则将OCR结果保存到此目录
            already_saved: 标记图像是否已经保存到指定目录，避免重复保存
            
        Returns:
            dict: 包含提取文本和处理信息的字典
        """
        try:
            # 记录处理开始信息
            logger.info(f"开始OCR处理，图像路径: {image_path}, 引擎: {engine}, 语言: {lang}")
            start_time = time.time()
            
            # 保存原始图片到存储目录（如果未保存）
            if not already_saved and images_dir:
                original_stored_path = self.save_image_to_storage(image_path, suffix="original", target_dir=images_dir)
                logger.info(f"原始图片已保存至: {original_stored_path}")
            else:
                # 图像已保存，直接使用现有路径
                original_stored_path = image_path
                logger.info(f"使用已保存的图像: {original_stored_path}")
            
            # 图像预处理
            logger.info("执行图像预处理...")
            processed_path = self.preprocess_image(image_path, images_dir)
            
            # 如果预处理失败，使用原始图片路径
            if not processed_path:
                processed_path = original_stored_path
                logger.info("预处理失败，使用原始图像路径")
            else:
                logger.info(f"图像预处理完成: {processed_path}")
            
            # 选择OCR引擎
            original_engine = engine
            if engine == 'auto':
                logger.info("自动选择OCR引擎...")
                if self.gemini_available:
                    engine = 'gemini'
                    logger.info("自动选择: Gemini 2.0 Flash")
                elif self.easyocr_available:
                    engine = 'easyocr'
                    logger.info("自动选择: EasyOCR")
                else:
                    engine = 'tesseract'
                    logger.info("自动选择: Tesseract")
            
            # 执行OCR
            text = ""
            
            # Gemini提取文本
            if engine == 'gemini':
                logger.info(f"使用Gemini 2.0 Flash处理图像: {processed_path}")
                text = self.extract_text_with_gemini(processed_path)
                
                # 如果Gemini失败，回退到其他引擎
                if not text:
                    logger.warning("Gemini处理失败，尝试回退...")
                    if self.easyocr_available:
                        engine = 'easyocr'
                        logger.info("回退到: EasyOCR")
                    else:
                        engine = 'tesseract'
                        logger.info("回退到: Tesseract")
            
            # EasyOCR提取文本
            if engine == 'easyocr' and self.easyocr_available:
                logger.info(f"使用EasyOCR处理图像: {processed_path}")
                try:
                    result = self.easyocr_reader.readtext(processed_path, detail=0)
                    text = '\n'.join(result)
                    logger.info(f"EasyOCR处理完成，提取了{len(result)}个文本块")
                except Exception as e:
                    logger.error(f"EasyOCR处理错误: {str(e)}", exc_info=1)
                    # 如果EasyOCR失败，回退到Tesseract
                    engine = 'tesseract'
                    logger.info("回退到: Tesseract")
            
            # Tesseract提取文本
            if engine == 'tesseract' or not text:
                logger.info(f"使用Tesseract处理图像: {processed_path}, 语言: {lang}")
                try:
                    image = Image.open(processed_path)
                    text = pytesseract.image_to_string(image, lang=lang)
                    logger.info(f"Tesseract处理完成，提取文本长度: {len(text)}")
                except Exception as e:
                    logger.error(f"Tesseract处理错误: {str(e)}", exc_info=1)
                    raise Exception(f"OCR处理失败: {str(e)}")
            
            # 计算处理时间
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"OCR处理完成，耗时: {processing_time:.2f}秒，初始引擎: {original_engine}，实际使用引擎: {engine}")
            
            # 预览提取的文本
            if text:
                text_preview = text[:100] + "..." if len(text) > 100 else text
                logger.info(f"提取文本预览: {text_preview}")
            else:
                logger.warning("未能提取任何文本")
                    
            return {
                "text": text,
                "engine": engine,
                "original_image": original_stored_path,
                "processed_image": processed_path,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"OCR处理异常: {str(e)}", exc_info=True)
            raise
    
    def process_pdf_ocr(self, pdf_path: str, engine: str = 'auto', lang: str = 'chi_sim+eng', dpi: int = 300, pdf_dir: str = None, images_dir: str = None, ocr_dir: str = None, already_saved: bool = False) -> Dict[str, Any]:
        """
        处理PDF文件并提取文本
        
        Args:
            pdf_path: PDF文件路径
            engine: OCR引擎 ('auto', 'tesseract', 'easyocr', 'gemini')
            lang: 语言设置 (tesseract格式)
            dpi: PDF转换为图像的DPI
            pdf_dir: PDF文件存储目录
            images_dir: 图像存储目录
            ocr_dir: OCR结果存储目录
            already_saved: 标记PDF是否已经保存到指定目录，避免重复保存
            
        Returns:
            dict: 包含提取文本和处理信息的字典
        """
        try:
            if not PDF_SUPPORT:
                logger.error("pdf2image 库未安装，无法处理PDF文件", exc_info=1)
                raise ImportError("pdf2image 库未安装，无法处理PDF文件。请安装：pip install pdf2image")
                
            logger.info(f"开始处理PDF文件: {pdf_path}, 引擎: {engine}, 语言: {lang}, DPI: {dpi}")
            start_time = time.time()
            
            # 保存原始PDF到存储目录（如果未保存）
            if not already_saved and pdf_dir:
                stored_pdf_path = self.save_image_to_storage(pdf_path, suffix="original", target_dir=pdf_dir)
                logger.info(f"原始PDF已保存至: {stored_pdf_path}")
            else:
                # PDF已保存，直接使用现有路径
                stored_pdf_path = pdf_path
                logger.info(f"使用已保存的PDF: {stored_pdf_path}")
            
            # 将PDF转换为图像
            image_paths = self.convert_pdf_to_images(pdf_path, dpi, images_dir)
            logger.info(f"PDF已转换为 {len(image_paths)} 个图像文件")
            
            # 对每个图像进行OCR处理
            all_text = []
            page_results = []
            
            for i, image_path in enumerate(image_paths):
                logger.info(f"处理第 {i+1}/{len(image_paths)} 页...")
                
                # 处理单个图像 - 因为是从PDF提取的图像，已经保存在images_dir中，所以设置already_saved=True
                result = self.process_image_ocr(image_path, engine, lang, images_dir, ocr_dir, already_saved=True)
                
                # 添加页码信息
                page_text = result["text"]
                all_text.append(f"=== 第 {i+1} 页 ===\n{page_text}")
                
                page_results.append({
                    "page": i+1,
                    "text": page_text,
                    "image_path": image_path,
                    "processed_image": result.get("processed_image", ""),
                    "engine": result["engine"]
                })
            
            # 合并所有文本
            combined_text = "\n\n".join(all_text)
            
            # 计算处理时间
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"PDF处理完成，耗时: {processing_time:.2f}秒，共 {len(image_paths)} 页")
            
            # 返回结果
            return {
                "text": combined_text,
                "engine": engine,
                "original_pdf": stored_pdf_path,
                "pages": page_results,
                "processing_time": processing_time,
                "page_count": len(image_paths)
            }
        
        except Exception as e:
            logger.error(f"PDF处理异常: {str(e)}", exc_info=True)
            raise
    
    def process_document(self, file_path: str, engine: str = 'auto', lang: str = 'chi_sim+eng', dpi: int = 300, order_dir: str = None, already_saved: bool = False) -> Dict[str, Any]:
        """
        处理文档（图像或PDF）并提取文本
        
        Args:
            file_path: 文件路径（图像或PDF）
            engine: OCR引擎 ('auto', 'tesseract', 'easyocr', 'gemini')
            lang: 语言设置 (tesseract格式)
            dpi: PDF转换为图像的DPI（仅对PDF文件有效）
            order_dir: 订单处理目录，如果提供则将处理结果保存到此目录下
            already_saved: 标记文件是否已保存到指定目录，避免重复保存
            
        Returns:
            dict: 包含提取文本和处理信息的字典
        """
        try:
            # 如果提供了订单目录，确保子目录结构存在
            images_dir = None
            pdf_dir = None
            ocr_dir = None
            
            if order_dir:
                images_dir = os.path.join(order_dir, 'images')
                pdf_dir = os.path.join(order_dir, 'pdf')
                ocr_dir = os.path.join(order_dir, 'ocr')
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(pdf_dir, exist_ok=True)
                os.makedirs(ocr_dir, exist_ok=True)
                logger.info(f"使用订单处理目录: {order_dir}")
            
            # 判断文件类型
            if self.is_pdf_file(file_path):
                logger.info(f"检测到PDF文件: {file_path}")
                result = self.process_pdf_ocr(file_path, engine, lang, dpi, pdf_dir, images_dir, ocr_dir, already_saved)
                return result
            else:
                logger.info(f"检测到图像文件: {file_path}")
                result = self.process_image_ocr(file_path, engine, lang, images_dir, ocr_dir, already_saved)
                return result
                
        except Exception as e:
            logger.error(f"文档处理异常: {str(e)}", exc_info=True)
            raise

# 兼容原有API的辅助函数
def process_document_ocr(file_path: str, engine: str = 'auto', lang: str = 'chi_sim+eng', dpi: int = 300, order_dir: str = None, already_saved: bool = False) -> Dict[str, Any]:
    """
    处理文档（图像或PDF）并提取文本
    
    Args:
        file_path: 文件路径（图像或PDF）
        engine: OCR引擎 ('auto', 'tesseract', 'easyocr', 'gemini')
        lang: 语言设置 (tesseract格式)
        dpi: PDF转换为图像的DPI（仅对PDF文件有效）
        order_dir: 订单处理目录，如果提供则将处理结果保存到此目录下
        already_saved: 标记文件是否已保存到指定目录，避免重复保存
        
    Returns:
        dict: 包含提取文本和处理信息的字典
    """
    processor = OCRProcessor()
    return processor.process_document(file_path, engine, lang, dpi, order_dir, already_saved)

def process_image_ocr(image_path: str, engine: str = 'auto', lang: str = 'chi_sim+eng', images_dir: str = None, ocr_dir: str = None, already_saved: bool = False) -> Dict[str, Any]:
    """图像OCR处理（模块级函数）"""
    processor = OCRProcessor()
    return processor.process_image_ocr(image_path, engine, lang, images_dir, ocr_dir, already_saved)

def process_pdf_ocr(pdf_path: str, engine: str = 'auto', lang: str = 'chi_sim+eng', dpi: int = 300, pdf_dir: str = None, images_dir: str = None, ocr_dir: str = None, already_saved: bool = False) -> Dict[str, Any]:
    """PDF OCR处理（模块级函数）"""
    processor = OCRProcessor()
    return processor.process_pdf_ocr(pdf_path, engine, lang, dpi, pdf_dir, images_dir, ocr_dir, already_saved)

def is_pdf_file(file_path: str) -> bool:
    """检查文件是否为PDF文件（模块级函数）"""
    processor = OCRProcessor()
    return processor.is_pdf_file(file_path) 