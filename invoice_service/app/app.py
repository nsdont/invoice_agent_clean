from flask import Flask, request, jsonify
import os
import uuid
import base64
import time
import json
from datetime import datetime

from invoice_service.app.services.ocr_service import process_pdf_ocr
# 引入统一日志配置
from .shared.logger_config import configure_logger

# 导入服务模块
from .services import process_image_ocr, extract_order_from_text, ProductMatcher
from .models import (
    ExtractRequest,
    ProductMatchRequest,
    OrderData,
    ExtractedOrderData,
    OutputOrderItem,
    OutputOrderData,
)

# 获取应用程序logger
logger = configure_logger(__name__)

# 获取基础目录
BASE_DIR = os.environ.get('APP_BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMP_DIR = os.environ.get('APP_TEMP_DIR', os.path.join(BASE_DIR, 'temp'))
SHARED_DIR = os.environ.get('APP_SHARED_DIR', os.path.join(BASE_DIR, 'shared'))
DATA_DIR = os.environ.get('APP_SHARED_DIR', os.path.join(BASE_DIR, 'data'))

# 日志目录
log_dir = os.environ.get('APP_LOG_DIR', os.path.join(BASE_DIR, 'logs'))
os.makedirs(log_dir, exist_ok=True)

app = Flask(__name__)

# 确保临时目录存在
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SHARED_DIR, exist_ok=True)

# 将路径存储在应用配置中，以便在其他地方使用
app.config['TEMP_DIR'] = TEMP_DIR
app.config['SHARED_DIR'] = SHARED_DIR
app.config['LOG_DIR'] = log_dir
app.config['DATA_DIR'] = DATA_DIR

logger.info("应用程序初始化完成，目录配置: TEMP_DIR=%s, SHARED_DIR=%s, LOG_DIR=%s",
           TEMP_DIR, SHARED_DIR, log_dir)

# 辅助函数：创建处理目录结构
def create_processing_dirs(file_id):
    """创建处理所需的目录结构"""
    order_dir = os.path.join(app.config['SHARED_DIR'], str(file_id))
    images_dir = os.path.join(order_dir, 'images')
    pdf_dir = os.path.join(order_dir, 'pdf')
    ocr_dir = os.path.join(order_dir, 'ocr')
    order_output_dir = os.path.join(order_dir, 'order')
    
    # 确保目录存在
    os.makedirs(order_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(order_output_dir, exist_ok=True)
    
    logger.info(f"创建处理目录: {order_dir} [ID: {file_id}]")
    
    return {
        'order_dir': order_dir,
        'images_dir': images_dir,
        'pdf_dir': pdf_dir,
        'ocr_dir': ocr_dir,
        'order_output_dir': order_output_dir
    }

# 辅助函数：处理上传的文件
def process_uploaded_file(file, dirs, file_id, ocr_engine='auto'):
    """处理上传的文件并执行OCR"""
    file_name = file.filename or f"upload_{file_id}.png"
    
    # 根据文件类型确定保存目录
    if file_name.lower().endswith('.pdf'):
        save_dir = dirs['pdf_dir']
        file_path = os.path.join(save_dir, file_name)
        file.save(file_path)
        logger.info(f"保存上传的PDF文件到 {file_path} [ID: {file_id}]")
        
        # 执行PDF OCR处理，传入already_saved=True标记文件已保存
        ocr_result = process_pdf_ocr(file_path, engine=ocr_engine, 
                                    pdf_dir=dirs['pdf_dir'], 
                                    images_dir=dirs['images_dir'], 
                                    ocr_dir=dirs['ocr_dir'],
                                    already_saved=True)
    else:
        save_dir = dirs['images_dir']
        file_path = os.path.join(save_dir, file_name)
        file.save(file_path)
        logger.info(f"保存上传的图片文件到 {file_path} [ID: {file_id}]")
        
        # 执行图片OCR处理，传入already_saved=True标记文件已保存
        ocr_result = process_image_ocr(file_path, engine=ocr_engine, 
                                     images_dir=dirs['images_dir'], 
                                     ocr_dir=dirs['ocr_dir'],
                                     already_saved=True)
    
    return process_ocr_result(ocr_result, file_name, dirs, file_id)

# 辅助函数：处理base64编码文件
def process_base64_file(base64_data, file_name, mime_type, dirs, file_id, ocr_engine='auto'):
    """处理base64编码文件并执行OCR"""
    # 根据MIME类型确定保存目录和文件扩展名
    if mime_type == 'application/pdf':
        save_dir = dirs['pdf_dir']
        file_ext = '.pdf'
    else:
        save_dir = dirs['images_dir']
        file_ext = os.path.splitext(file_name)[1] or '.png'
    
    # 确保文件名有扩展名
    if not os.path.splitext(file_name)[1]:
        file_name = f"{file_name}{file_ext}"
    
    try:
        # 解码base64数据
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]
        file_data_binary = base64.b64decode(base64_data)
        
        # 保存文件
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(file_data_binary)
        logger.info(f"保存base64数据到 {file_path} [ID: {file_id}]")
        
        # 根据文件类型执行OCR，传入already_saved=True标记文件已保存
        if mime_type == 'application/pdf':
            ocr_result = process_pdf_ocr(file_path, engine=ocr_engine, 
                                        pdf_dir=dirs['pdf_dir'], 
                                        images_dir=dirs['images_dir'], 
                                        ocr_dir=dirs['ocr_dir'],
                                        already_saved=True)
        else:
            ocr_result = process_image_ocr(file_path, engine=ocr_engine, 
                                         images_dir=dirs['images_dir'], 
                                         ocr_dir=dirs['ocr_dir'],
                                         already_saved=True)
        
        return process_ocr_result(ocr_result, file_name, dirs, file_id)
        
    except Exception as e:
        logger.error(f"处理base64数据失败: {str(e)} [ID: {file_id}]")
        return None, None, None

# 辅助函数：处理OCR结果
def process_ocr_result(ocr_result, file_name, dirs, file_id):
    """处理OCR结果，保存到文件并返回"""
    ocr_text = ocr_result['text']
    used_ocr_engine = ocr_result['engine']
    
    # 保存OCR结果
    file_base_name = os.path.splitext(file_name)[0]
    ocr_file_path = os.path.join(dirs['ocr_dir'], f"{file_base_name}_ocr.txt")
    with open(ocr_file_path, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
    
    logger.info(f"OCR处理完成，结果保存到 {ocr_file_path} [ID: {file_id}]")
    return ocr_text, file_name, used_ocr_engine

# 辅助函数：合并OCR结果
def combine_ocr_texts(all_ocr_texts, dirs, file_id):
    """合并OCR文本并保存"""
    if not all_ocr_texts:
        logger.error(f"没有从任何文件中提取到OCR文本 [ID: {file_id}]")
        return None
            
    ocr_text = "\n\n".join(all_ocr_texts)
    
    # 保存合并后的OCR结果
    combined_ocr_path = os.path.join(dirs['ocr_dir'], "combined_ocr.txt")
    with open(combined_ocr_path, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
        
    logger.info(f"合并了{len(all_ocr_texts)}个文件的OCR结果，保存到 {combined_ocr_path} [ID: {file_id}]")
    return ocr_text, combined_ocr_path

# 辅助函数：保存提取结果
def save_extraction_result(order_data, dirs, file_id):
    """保存提取的结构化订单数据"""
    extraction_file = os.path.join(dirs['order_output_dir'], "extraction.json")
    with open(extraction_file, 'w', encoding='utf-8') as f:
        f.write(order_data.model_dump_json(indent=2))
        
    logger.info(f"数据提取完成，保存到 {extraction_file}，订单用户名：{order_data.customer_name}, "
               f"订单编号: {order_data.order_id}, 共{len(order_data.items)}个商品 [ID: {file_id}]")
    return extraction_file

# 辅助函数：执行产品匹配
def perform_product_matching(order_data, threshold, dirs, file_id):
    """执行产品匹配并保存结果"""
    logger.info(f"开始产品匹配，阈值: {threshold} [ID: {file_id}]")
    product_matcher = ProductMatcher()
    matched_order = product_matcher.match_products(order_data, threshold)
    matched_count = sum(1 for item in matched_order.items if item.product_id)
    
    # 保存匹配结果
    output_file = os.path.join(dirs['order_output_dir'], "matched_order.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(matched_order.model_dump_json(indent=2))
        
    logger.info(f"产品匹配完成，结果保存到 {output_file}，成功匹配: {matched_count}/{len(matched_order.items)} [ID: {file_id}]")

    # 将 OrderData 转换为 OutputOrderData
    output_order_data_item = []
    for item in matched_order.items:
        output_order_data_item.append(OutputOrderItem(
            original_input=item.original_input,
            product_id=item.product_id,
            quantity=item.quantity,
            matched_name=item.matched_name,
            match_score=item.match_score
        ))
    output_order_data = OutputOrderData(
        customer_name=matched_order.customer_name,
        order_date=matched_order.order_date,
        items=output_order_data_item,
        status=matched_order.status
    )
    return output_order_data, matched_count, output_file

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Invoice Processing Service",
        "status": "running",
        "endpoints": {
            "/": "Service information",
            "/health": "Health check",
            "/ocr": "OCR processing",
            "/extract": "Extract structured data from text",
            "/match": "Match products from order",
            "/process": "End-to-end invoice processing"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    import pytesseract
    
    # 检查API密钥
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # 检查easyocr是否已加载（根据Gemini API可用性）
    from .services import GEMINI_AVAILABLE
    
    # 检查产品数据
    product_matcher = ProductMatcher()
    products_count = len(product_matcher.products)
    
    return jsonify({
        "status": "ok", 
        "easyocr_available": False,
        "easyocr_loaded": False,
        "tesseract_version": pytesseract.get_tesseract_version(),
        "gemini_api_key_configured": bool(gemini_api_key),
        "openai_api_key_configured": bool(openai_api_key),
        "products_loaded": products_count,
        "temp_dir": os.path.exists(TEMP_DIR),
        "shared_dir": os.path.exists(SHARED_DIR)
    })

@app.route('/ocr', methods=['POST'])
def process_ocr():
    """OCR处理端点"""
    start_time = time.time()
    
    # 使用时间戳作为文件ID，保持与/process一致
    file_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    logger.info(f"开始OCR处理 [ID: {file_id}]")
    source_type = "pdf"
    
    try:
        # 1. 解析请求
        if 'files' not in request.json:
            logger.warning(f"请求缺少文件 [ID: {file_id}]")
            return jsonify({"error": "No file or base64 data provided"}), 400
            
        # 选择OCR引擎
        ocr_engine = request.args.get('engine', 'auto')
        
        # 创建处理目录结构
        dirs = create_processing_dirs(file_id)
        
        # 2. 执行OCR
        all_ocr_texts = []
        processed_files = []
        used_ocr_engine = "unknown"

        # 处理多个文件 [{"fileName": "xx.png", "mimeType": "image/png", "base64Data": "..."}]
        files = request.json.get('files')
        if not files or not isinstance(files, list) or len(files) == 0:
            logger.warning(f"无效的文件数据 [ID: {file_id}]")
            return jsonify({"error": "Invalid files data"}), 400
        
        # 处理每个文件
        for index, file_data in enumerate(files):
            base64_data = file_data.get('base64Data', '')
            file_name = file_data.get('fileName', f"file_{index}.png")
            mime_type = file_data.get('mimeType', 'image/png')
            source_type = 'pdf' if mime_type == 'application/pdf' else 'handwritten'
            
            if not base64_data:
                logger.warning(f"文件 {file_name} 没有base64Data [ID: {file_id}]")
                continue
            
            ocr_text, file_name, engine = process_base64_file(
                base64_data, file_name, mime_type, dirs, file_id, ocr_engine)
                
            if ocr_text:
                all_ocr_texts.append(ocr_text)
                processed_files.append(file_name)
                used_ocr_engine = engine
        
        # 合并OCR文本
        combined_result = combine_ocr_texts(all_ocr_texts, dirs, file_id)
        if not combined_result:
            return jsonify({"error": "No text could be extracted from the provided files"}), 400
            
        ocr_text, combined_ocr_path = combined_result
        
        processing_time = time.time() - start_time
        
        # 返回处理结果
        result = {
            "text": ocr_text,
            "engine": used_ocr_engine,
            "processing_time": processing_time,
            "processed_files": processed_files,
            "files_count": len(processed_files),
            "order_dir": dirs['order_dir'],
            "ocr_dir": dirs['ocr_dir'],
            "combined_ocr_file": combined_ocr_path,
            "success": True,
            "task_id": file_id,
            "source_type": source_type
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"OCR处理失败: {str(e)} [ID: {file_id}]", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/extract', methods=['POST'])
def extract_data():
    """从OCR文本中提取结构化信息"""
    start_time = time.time()
    
    try:
        # 验证请求
        if not request.is_json:
            logger.warning(f"请求必须是JSON格式")
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        if 'text' not in data:
            logger.warning(f"请求缺少'text'字段")
            return jsonify({"error": "Missing 'text' field"}), 400
        
        # 从请求中获取数据
        extract_request = ExtractRequest(**data)
        file_id = extract_request.task_id
        ocr_text = extract_request.text
        customer_name = extract_request.customer_name
        ocr_file_path = extract_request.ocr_file_path if hasattr(extract_request, 'ocr_file_path') else None
        source_type = extract_request.source_type if hasattr(extract_request, 'source_type') else "pdf"
        
        # 创建或使用处理目录结构
        if extract_request.order_dir:
            order_dir = extract_request.order_dir
            # 确保子目录存在
            order_output_dir = os.path.join(order_dir, 'order')
            ocr_dir = os.path.join(order_dir, 'ocr')
            os.makedirs(order_output_dir, exist_ok=True)
            os.makedirs(ocr_dir, exist_ok=True)
            
            dirs = {
                'order_dir': order_dir,
                'ocr_dir': ocr_dir,
                'order_output_dir': order_output_dir
            }
        else:
            dirs = create_processing_dirs(file_id)
        
        logger.info(f"使用处理目录: {dirs['order_dir']}, 客户名称: {customer_name}, 源类型: {source_type} [ID: {file_id}]")
        
        # 如果提供了OCR文本但没有OCR文件路径，保存OCR文本到文件
        if ocr_text and not ocr_file_path:
            ocr_file_path = os.path.join(dirs['ocr_dir'], f"input_ocr_{file_id}.txt")
            with open(ocr_file_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            logger.info(f"保存输入的OCR文本到文件: {ocr_file_path} [ID: {file_id}]")
            
        # 提取结构化信息
        logger.info(f"开始从 {source_type} OCR 文本提取结构化数据 [ID: {file_id}]")
        order_data = extract_order_from_text(ocr_text, customer_name, source_type=source_type)
        
        # 保存提取结果
        extraction_file = save_extraction_result(order_data, dirs, file_id)
        
        processing_time = time.time() - start_time
        
        # 返回处理结果
        result = {
            "order_data": json.loads(order_data.model_dump_json()),
            "processing_time": processing_time,
            "order_dir": dirs['order_dir'],
            "extraction_file": extraction_file,
            "ocr_file": ocr_file_path,
            "success": True,
            "task_id": file_id
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"提取数据失败: {str(e)} [ID: {file_id}]", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/match', methods=['POST'])
def match_products():
    """匹配产品信息"""
    start_time = time.time()

    try:
        # 验证请求
        if not request.is_json:
            logger.warning(f"请求必须是JSON格式")
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        if 'order_data' not in data:
            logger.warning(f"请求缺少'order_data'字段")
            return jsonify({"error": "Missing 'order_data' field"}), 400
        
        # 转换为请求对象
        match_request = ProductMatchRequest(**data)
        file_id = match_request.task_id
        order_data = match_request.order_data
        threshold = match_request.threshold

        logger.info(f"开始产品匹配 [ID: {file_id}]")

        # 创建或使用处理目录结构
        if match_request.order_dir:
            order_dir = match_request.order_dir
            # 确保子目录存在
            order_output_dir = os.path.join(order_dir, 'order')
            os.makedirs(order_output_dir, exist_ok=True)
            
            dirs = {
                'order_dir': order_dir,
                'order_output_dir': order_output_dir
            }
        else:
            dirs = create_processing_dirs(file_id)
        
        # 执行产品匹配
        matched_order, matched_count, output_file = perform_product_matching(
            order_data, threshold, dirs, file_id)
        
        processing_time = time.time() - start_time
        
        # 返回处理结果
        # result = {
        #     "order_data": json.loads(matched_order.model_dump_json()),
        #     "matched_items_count": matched_count,
        #     "total_items_count": len(matched_order.items),
        #     "match_ratio": matched_count / len(matched_order.items) if matched_order.items else 0,
        #     "processing_time": processing_time,
        #     "order_dir": dirs['order_dir'],
        #     "matched_order_file": output_file,
        #     "success": True
        # }
        result = {
            "order_data": json.loads(matched_order.model_dump_json()),
            "matched_items_count": matched_count,
            "total_items_count": len(matched_order.items),
            "match_ratio": matched_count / len(matched_order.items) if matched_order.items else 0,
            "processing_time": processing_time,
            "order_dir": dirs['order_dir'],
            "matched_order_file": output_file,
            "success": True
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"产品匹配失败: {str(e)} [ID: {file_id}]", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/process', methods=['POST'])
def process_invoice():
    """端到端处理发票"""
    start_time = time.time()

    # 使用时间戳作为文件ID
    file_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    logger.info(f"开始处理发票 [ID: {file_id}]")
    
    try:
        # 1. 解析请求
        if 'file' not in request.files and 'files' not in request.json:
            logger.warning(f"请求缺少文件或base64数据 [ID: {file_id}]")
            return jsonify({"error": "No file or base64 data provided"}), 400
            
        customer_name = request.form.get('customer_name') if request.form else request.json.get('customer_name')
        threshold = float(request.args.get('threshold', 0.6))
        source_type = request.args.get('source_type', 'pdf')  # 默认为pdf
        logger.info(f"请求参数: customer_name={customer_name}, threshold={threshold}, source_type={source_type} [ID: {file_id}]")
        
        # 2. 创建处理目录结构
        dirs = create_processing_dirs(file_id)
        
        # 3. 执行OCR
        all_ocr_texts = []
        ocr_engine = "unknown"
        processed_files = []
        
        if 'file' in request.files:
            ocr_text, file_name, engine = process_uploaded_file(
                request.files['file'], dirs, file_id)
                
            if ocr_text:
                all_ocr_texts.append(ocr_text)
                processed_files.append(file_name)
                ocr_engine = engine
        else:
            # 处理多个文件 [{"fileName": "xx.png", "mimeType": "image/png", "base64Data": "..."}]
            files = request.json.get('files')
            if not files or not isinstance(files, list) or len(files) == 0:
                logger.warning(f"无效的文件数据 [ID: {file_id}]")
                return jsonify({"error": "Invalid files data"}), 400
            
            # 处理每个文件
            for index, file_data in enumerate(files):
                base64_data = file_data.get('base64Data', '')
                file_name = file_data.get('fileName', f"file_{index}.png")
                mime_type = file_data.get('mimeType', 'image/png')
                
                if not base64_data:
                    logger.warning(f"文件 {file_name} 没有base64Data [ID: {file_id}]")
                    continue
                
                ocr_text, file_name, engine = process_base64_file(
                    base64_data, file_name, mime_type, dirs, file_id)
                    
                if ocr_text:
                    all_ocr_texts.append(ocr_text)
                    processed_files.append(file_name)
                    ocr_engine = engine
            
        # 合并OCR文本
        combined_result = combine_ocr_texts(all_ocr_texts, dirs, file_id)
        if not combined_result:
            return jsonify({"error": "No text could be extracted from the provided files"}), 400
            
        ocr_text, combined_ocr_path = combined_result
        
        # 4. 提取结构化信息
        logger.info(f"开始从{source_type}文本提取结构化数据 [ID: {file_id}]")
        order_data = extract_order_from_text(ocr_text, customer_name, source_type=source_type)
        
        # 保存提取结果
        extraction_file = save_extraction_result(order_data, dirs, file_id)
        
        # 5. 产品匹配
        matched_order, matched_count, output_file = perform_product_matching(
            order_data, threshold, dirs, file_id)
            
        processing_time = time.time() - start_time
        logger.info(f"发票处理完成，耗时: {processing_time:.2f}秒 [ID: {file_id}]")
        
        # 返回处理结果
        result = {
            "order_data": json.loads(matched_order.json()),
            "processed_files": processed_files,
            "files_count": len(processed_files),
            "ocr_engine": ocr_engine,
            "source_type": source_type,
            "processing_time": processing_time,
            "order_dir": dirs['order_dir'],
            "combined_ocr_file": combined_ocr_path,
            "extraction_file": extraction_file,
            "matched_order_file": output_file,
            "success": True
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"发票处理失败: {str(e)} [ID: {file_id}]", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 