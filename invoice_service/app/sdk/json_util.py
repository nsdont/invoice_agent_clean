# 尝试从markdown代码块中提取JSON
import json
import re
import uuid
import time

def parse_json(text, logger, task_id=None) -> dict:
    """
    从文本中解析JSON
    
    Args:
        text: 包含JSON的文本
        logger: 日志记录器
        task_id: 可选的任务ID，用于日志跟踪
        
    Returns:
        dict: 解析出的JSON对象
    """
    # 如果没有提供task_id，生成一个唯一标识符
    if not task_id:
        task_id = str(uuid.uuid4())[:8]  # 使用前8位即可
        
    start_time = time.time()
    
    try:
        # 首先尝试直接解析整个响应
        result = json.loads(text)
        elapsed = time.time() - start_time
        logger.info(f"JSON解析成功 [任务:{task_id}] 耗时:{elapsed:.3f}秒")
        return result
    except json.JSONDecodeError as e:
        # 尝试从markdown代码块中提取
        markdown_json = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if markdown_json:
            try:
                json_text = markdown_json.group(1).strip()
                result = json.loads(json_text)
                elapsed = time.time() - start_time
                logger.info(f"从markdown代码块提取JSON成功 [任务:{task_id}] 耗时:{elapsed:.3f}秒")
                return result
            except json.JSONDecodeError:
                pass

        # 尝试提取任何看起来像JSON的部分
        json_match = re.search(r'({[\s\S]*?})', text)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                elapsed = time.time() - start_time
                logger.info(f"从文本中提取JSON成功 [任务:{task_id}] 耗时:{elapsed:.3f}秒")
                return result
            except Exception as json_e:
                logger.error(f"JSON解析失败 [任务:{task_id}] 原因:{str(json_e)}")
                
        elapsed = time.time() - start_time
        logger.error(f"JSON解析失败 [任务:{task_id}] 耗时:{elapsed:.3f}秒 错误:{str(e)}", exc_info=1)
    return {}