import json
import os
import re
import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import requests
import time
import functools
import traceback
import sys

# ==================== 导入 Flask Logger ====================
try:
    from models.logger import get_logger, RequestIdFilter

    logger = get_logger(
        name="knowledge_graph",
        log_file="logs/knowledge_graph.log"
    )
    print(f"✅ 使用Flask Logger，日志文件: logs/knowledge_graph.log")

except ImportError:
    import logging
    import os
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger("knowledge_graph")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "knowledge_graph.log")

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [Standalone] - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    abs_log_path = os.path.abspath(log_file)
    print(f"✅ 使用标准Logger，日志文件: {abs_log_path}")
    logger.info(f"日志系统初始化完成,文件路径: {abs_log_path}")


# ==================== 日志装饰器 ====================

def log_execution_time(logger_instance=None):
    """记录函数执行时间的装饰器"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger_instance or logger
            start_time = time.time()
            _logger.info(f"开始执行: {func.__name__}")

            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                _logger.info(f"执行完成: {func.__name__} | 耗时: {elapsed_time:.2f}秒")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                _logger.error(f"执行失败: {func.__name__} | 耗时: {elapsed_time:.2f}秒 | 错误: {str(e)}", exc_info=True)
                raise

        return wrapper

    return decorator


def log_function_call(logger_instance=None):
    """记录函数调用参数和返回值的装饰器"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger_instance or logger
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            _logger.debug(f"调用函数: {func.__name__}({signature})")

            try:
                result = func(*args, **kwargs)
                _logger.debug(f"函数返回: {func.__name__} -> {result!r}")
                return result
            except Exception as e:
                _logger.error(f"函数异常: {func.__name__} | 错误: {str(e)}", exc_info=True)
                raise

        return wrapper

    return decorator


# ==================== 数据类 ====================

@dataclass
class ExtractedTriple:
    entities: List[Dict]
    relationships: List[Dict]
    attributes: List[Dict]


# ==================== 元数据加载工具 ====================

def load_metadata(meta_path: str) -> Dict:
    """从 meta_path 读取 JSON，返回课程/章节元数据"""
    logger.info(f"开始加载元数据: {meta_path}")

    if not os.path.exists(meta_path):
        logger.warning(f"元数据文件不存在: {meta_path}，使用默认值")
        return {"course_name": "专业课程", "chapter_name": "第1章"}

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        required_fields = ["course_name", "chapter_name"]
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"元数据缺少字段 {field}，使用默认值")
                metadata[field] = f"默认{field}"

        logger.info(f"元数据加载成功: 课程={metadata['course_name']}, 章节={metadata['chapter_name']}")
        return metadata

    except Exception as e:
        logger.error(f"加载元数据失败: {str(e)}", exc_info=True)
        return {"course_name": "专业课程", "chapter_name": "第1章"}


def load_chapter_summaries(meta_path: str) -> List[Dict]:
    """从 JSON 读取章节摘要"""
    logger.info(f"开始加载章节摘要: {meta_path}")

    if not os.path.exists(meta_path):
        logger.warning(f"章节摘要文件不存在: {meta_path}")
        return []

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)

        valid_summaries = []
        for item in summaries:
            if 'chapter_name' in item and 'summary' in item:
                valid_summaries.append(item)
            else:
                logger.warning(f"章节摘要项缺少必要字段: {item}")

        logger.info(f"章节摘要加载成功: {len(valid_summaries)}个章节")
        return valid_summaries

    except Exception as e:
        logger.error(f"加载章节摘要失败: {str(e)}", exc_info=True)
        return []


# ==================== DeepSeek API 客户端 ====================

class DeepSeekClient:
    """DeepSeek API客户端"""

    def __init__(self, api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"DeepSeek客户端初始化: model={model}, base_url={base_url}")

    @log_execution_time(logger)
    def chat_completions_create(self, messages: List[Dict], temperature: float = 0.1,
                                max_tokens: int = 8192, max_retries: int = 3) -> Dict:
        """调用DeepSeek API - 带重试机制"""
        for attempt in range(max_retries):
            try:
                logger.info(f"API调用尝试 {attempt + 1}/{max_retries}")
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=60,
                    proxies=None
                )
                response.raise_for_status()
                logger.info(f"API调用成功: status_code={response.status_code}")
                return response.json()

            except requests.exceptions.Timeout:
                wait_time = (attempt + 1) * 5
                logger.warning(f"API调用超时 (尝试 {attempt + 1}/{max_retries}), 等待 {wait_time}秒后重试")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue

            except requests.exceptions.ConnectionError as e:
                wait_time = (attempt + 1) * 3
                logger.warning(f"API连接错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}, 等待 {wait_time}秒后重试")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP错误: status_code={e.response.status_code}, message={str(e)}")
                if e.response.status_code == 429:
                    logger.warning("触发API限流，等待30秒后重试")
                    time.sleep(30)
                    continue
                else:
                    break

            except Exception as e:
                logger.error(f"未知错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue

        logger.error("API调用完全失败，返回空结果")
        return {"choices": [{"message": {"content": "{\"entities\": [], \"relationships\": []}"}}]}


# ==================== 实体抽取器 ====================

class EntityExtractor:
    """实体关系抽取器 - 强化知识点间关系"""

    def __init__(self, deepseek_api_key: str, course_name: str = "专业课程"):
        self.deepseek = DeepSeekClient(deepseek_api_key)
        self.course_name = course_name
        self.arduino_keywords = [
            'Arduino', 'LED', 'sensor', '传感器', 'pin', '引脚', 'GPIO',
            'voltage', '电压', 'current', '电流', 'resistor', '电阻', 'PWM',
            'digital', '数字', 'analog', '模拟', 'serial', '串口', 'I2C', 'SPI',
            'breadboard', '面包板', 'wire', '导线', 'ground', '接地', 'VCC', '5V', '3.3V'
        ]
        logger.info(f"实体抽取器初始化: course_name={course_name}")

    def _extract_page_number(self, filename: str) -> int:
        """从文件名中提取页面/幻灯片号码"""
        patterns = [r'slide_?(\d+)', r'p(\d+)', r'page_?(\d+)', r'_(\d+)', r'(\d+)']
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                page_num = int(match.group(1))
                logger.debug(f"提取页码: {filename} -> {page_num}")
                return page_num
        logger.debug(f"未找到页码: {filename}")
        return 0

    @log_execution_time(logger)
    def load_multimodal_data(self, output_dir: str = "output") -> Dict:
        """加载多模态预处理的输出数据"""
        logger.info(f"开始加载多模态数据: output_dir={output_dir}")
        result = {'slides': [], 'images': []}

        try:
            text_dir = os.path.join(output_dir, "text")
            image_dir = os.path.join(output_dir, "images")

            if os.path.exists(text_dir):
                text_files = os.listdir(text_dir)
                slide_files = [f for f in text_files if f.endswith('.json') and not f.endswith('_desc.json')]
                logger.info(f"找到文本文件: {len(slide_files)}个")

                for slide_file in slide_files:
                    slide_path = os.path.join(text_dir, slide_file)
                    try:
                        with open(slide_path, 'r', encoding='utf-8') as f:
                            slide_data = json.load(f)
                        slide_num = self._extract_page_number(slide_file)
                        result['slides'].append({
                            "slide_number": slide_num,
                            "content": slide_data,
                            "source_file": slide_file
                        })
                        logger.debug(f"加载文本文件: {slide_file} (页面 {slide_num})")
                    except Exception as e:
                        logger.warning(f"加载文本文件失败 {slide_file}: {str(e)}")

            if os.path.exists(image_dir):
                image_files_list = os.listdir(image_dir)
                image_files = [f for f in image_files_list if
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                logger.info(f"找到图片文件: {len(image_files)}个")

                for image_file in image_files:
                    base_name = os.path.splitext(image_file)[0]
                    desc_file = base_name + '_desc.json'
                    desc_path = os.path.join(image_dir, desc_file)
                    slide_num = self._extract_page_number(image_file)

                    image_data = {
                        "image_path": os.path.join(image_dir, image_file),
                        "slide_number": slide_num,
                        "filename": image_file,
                        "descriptions": [],
                        "ocr_text": ""
                    }

                    if os.path.exists(desc_path):
                        try:
                            with open(desc_path, 'r', encoding='utf-8') as f:
                                desc_data = json.load(f)
                                image_data["descriptions"] = desc_data.get("clip_descriptions", [])
                            logger.debug(f"加载图片描述: {desc_file}")
                        except Exception as e:
                            logger.warning(f"加载图片描述失败 {desc_file}: {str(e)}")

                    result['images'].append(image_data)

            result['slides'].sort(key=lambda x: x['slide_number'])
            result['images'].sort(key=lambda x: x['slide_number'])

            logger.info(f"数据加载完成: 文本={len(result['slides'])}个, 图片={len(result['images'])}个")

        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}", exc_info=True)
            raise

        return result

    @log_execution_time(logger)
    def generate_chapter_summary(self, slides: List[Dict], document_name: str, chapter_name: str) -> Dict:
        """生成章节摘要"""
        logger.info(f"开始生成章节摘要: 章节={chapter_name}, slides数量={len(slides)}")

        if not slides:
            logger.warning("没有slides数据，无法生成摘要")
            return {"summary": "", "keywords": []}

        combined_text = ""
        for slide in slides:
            slide_content = slide.get('content', {})
            text_content = ""

            if isinstance(slide_content, dict):
                text_fields = ['text', 'content', 'raw_text', 'texts']
                for field in text_fields:
                    if field in slide_content:
                        content = slide_content[field]
                        if isinstance(content, list):
                            text_content = ' '.join(str(item) for item in content)
                        else:
                            text_content = str(content)
                        break
                if not text_content:
                    text_content = str(slide_content)
            else:
                text_content = str(slide_content)

            if text_content and text_content.strip() != "" and text_content != "{}":
                combined_text += f"\n\n{text_content}"

        if not combined_text.strip():
            logger.warning("提取的文本内容为空")
            return {"summary": "", "keywords": []}

        prompt = f"""你是一位经验丰富的大学教师，正在讲授课程《{self.course_name}》中的《{chapter_name}》这一章。

下面是本章节 PPT 的全部文本内容（已经按顺序拼接，可能有重复或格式噪声）：

---------------- PPT 内容开始 ----------------
{combined_text}
---------------- PPT 内容结束 ----------------

请你基于上述内容，为本章节生成一个面向本科生的中文摘要。注意：

1. 摘要内容要紧密围绕本 PPT 的实际内容，不要凭空添加教材之外的知识。
2. 摘要要涵盖本章节的核心概念、主要知识点和它们之间的大致关系。
3. 表达风格应当简洁、清晰，适合放在"章节介绍"中给学生阅读。
4. 不要逐条复述 PPT，而是做一个结构清晰的整体概括。
5. 长度建议在 150～300 字之间（不是硬性限制，只要完整清晰即可）。
6. 摘要中不要出现"在本 PPT 中我们介绍了……"之类的措辞，而是直接陈述知识内容。
7. 不要在摘要中出现课程名称和章节名称的重复长篇罗列，点到为止即可。

同时，请提炼出 5～10 个能够代表本章节主题的关键词（可以是核心概念、方法、重要对象等）。

请严格按照以下 JSON 格式输出（不要多写解释性文字）：

{{
  "summary": "这里是一段完整的中文摘要，长度大约 150～300 字。",
  "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}}

要求：
- 必须是合法的 JSON，对双引号等字符做好转义。
- 字段名必须是 summary 和 keywords。
- keywords 为字符串数组，每个元素是一个短语或词语。
"""

        try:
            response = self.deepseek.chat_completions_create([
                {"role": "user", "content": prompt}
            ], max_retries=2, max_tokens=2000)

            content = response['choices'][0]['message']['content']
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)

                summary = result.get('summary', '')
                keywords = result.get('keywords', [])

                logger.info(f"章节摘要生成成功: 摘要长度={len(summary)}字, 关键词数量={len(keywords)}个")
                return {"summary": summary, "keywords": keywords}
            else:
                logger.warning("无法从响应中提取JSON")
                return {"summary": "", "keywords": []}

        except Exception as e:
            logger.error(f"章节摘要生成失败: {str(e)}", exc_info=True)
            return {"summary": "", "keywords": []}

    @log_execution_time(logger)
    def extract_entities_from_multimodal(self, multimodal_data: Dict, document_name: str = "") -> ExtractedTriple:
        """从多模态数据中抽取实体关系 - 强化知识点间关系"""
        logger.info(f"开始实体抽取: 课程《{self.course_name}》, 文档={document_name}")

        all_entities = []
        all_relationships = []
        all_attributes = []

        slides = multimodal_data.get('slides', [])
        logger.info(f"处理文本文件: {len(slides)}个")

        batch_size = 10
        total_batches = (len(slides) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(slides))
            batch_slides = slides[start_idx:end_idx]

            logger.info(f"处理批次 {batch_idx + 1}/{total_batches}: 页面 {start_idx + 1}-{end_idx}")

            batch_entities, batch_relations = self._extract_from_slide_batch(batch_slides, document_name)
            all_entities.extend(batch_entities)
            all_relationships.extend(batch_relations)

            if batch_idx < total_batches - 1:
                import random
                delay = random.uniform(2, 3)
                logger.debug(f"批次间等待 {delay:.1f}秒")
                time.sleep(delay)

        images = multimodal_data.get('images', [])
        logger.info(f"处理图片文件: {len(images)}个")

        for i, image_data in enumerate(images):
            logger.debug(f"处理图片 {i + 1}/{len(images)}: {image_data.get('filename', '')}")
            img_entities = self._extract_from_image(image_data, document_name)
            all_entities.extend(img_entities)

        logger.info("开始全局关系挖掘")
        enhanced_relationships = self._discover_global_relationships(all_entities, all_relationships)
        all_relationships.extend(enhanced_relationships)

        all_entities = self._deduplicate_entities(all_entities)
        all_relationships = self._deduplicate_relationships(all_relationships)

        logger.info("开始 AI 实体清洗")
        all_entities, all_relationships = self._ai_clean_entities(
            all_entities, all_relationships, document_name=document_name
        )

        logger.info("开始过滤非知识关系")
        filtered_relationships = self._filter_knowledge_relationships(all_relationships, all_entities)

        logger.info(
            f"实体抽取完成: 实体={len(all_entities)}个, "
            f"原始关系={len(all_relationships)}个, 知识关系={len(filtered_relationships)}个"
        )

        return ExtractedTriple(
            entities=all_entities,
            relationships=filtered_relationships,
            attributes=all_attributes
        )

    @log_execution_time(logger)
    def _ai_clean_entities(self, entities: List[Dict], relationships: List[Dict],
                           document_name: str = "") -> Tuple[List[Dict], List[Dict]]:
        """使用大模型清洗垃圾实体"""
        logger.info(f"AI 实体清洗开始: 总实体数={len(entities)}, 总关系数={len(relationships)}")

        candidates = []
        candidate_indices = []

        for idx, entity in enumerate(entities):
            entity_type = entity.get('type', '')
            entity_source = entity.get('source', '')

            if (entity_type == 'image_concept' or
                    entity_source in ['image_description', 'image_ocr']):
                candidates.append(entity)
                candidate_indices.append(idx)

        candidate_count = len(candidates)
        logger.info(f"筛选出需要 AI 审查的候选实体: {candidate_count}个")

        if candidate_count == 0:
            logger.info("没有需要审查的候选实体，跳过 AI 清洗")
            return entities, relationships

        batch_size = 50
        total_batches = (candidate_count + batch_size - 1) // batch_size
        to_delete_names = set()

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, candidate_count)
            batch_candidates = candidates[start_idx:end_idx]

            logger.info(f"AI 清洗批次 {batch_idx + 1}/{total_batches}: 实体 {start_idx + 1}-{end_idx}")

            indexed_candidates = []
            for local_idx, entity in enumerate(batch_candidates, start=1):
                indexed_candidates.append({
                    "idx": local_idx,
                    "name": entity.get('name', ''),
                    "type": entity.get('type', ''),
                    "source": entity.get('source', ''),
                    "description": entity.get('description', '')
                })

            entity_list_text = ""
            for item in indexed_candidates:
                entity_list_text += f"{item['idx']}. name: {item['name']}\n"
                entity_list_text += f"   type: {item['type']}\n"
                entity_list_text += f"   source: {item['source']}\n"
                entity_list_text += f"   description: {item['description']}\n\n"

            prompt = f"""你是一位负责构建课程《{self.course_name}》知识图谱的专家，当前章节为《{document_name}》。

下面是一批候选实体。每个实体包含：
- idx：本批中的编号（整数）
- name：实体名称
- type：实体类型（例如：核心概念、硬件组件、参数规格、编程概念、image_concept 等）
- source：来源（例如：batch_text, image_description, image_ocr）
- description：实体的简要说明

你的任务是，从"知识图谱的知识承载能力"和"课程内容相关性"的角度，判断这些实体中哪些应该删除，不纳入最终的课程知识图谱。

【应当删除的典型情况】
- 名称非常笼统、缺乏具体语义，只表示图像或排版形式，例如：截图、示例图、概念图、统计图表、表格、插图、文本图像、系统架构图、分子结构、科学图表、实验装置（未指定具体实验或对象）等；
- type 为 image_concept，且 description 只是"从图片描述中识别: XXX"，并且 XXX 本身并不是一个清晰的课程知识点；
- 与课程《{self.course_name}》的核心内容几乎没有关系，只是一般性的视觉元素。

【应当保留的典型情况】
- 具体的物理量或数学量，如：角位移、线位移、控制偏差；
- 具体的硬件或系统，如：步进电机、光电编码器、反馈编码器、数控铣床；
- 具体的控制策略或算法，如：闭环控制、PID控制、Bang-Bang控制；
- 具有清晰语义和教学意义的实验或装置名称，如："步进电机位置控制实验装置"等。

请你只输出需要删除的实体编号列表。

【实体列表】
{entity_list_text}

【输出格式（必须严格遵守）】
只输出一个 JSON 对象，例如：

{{
  "delete": [2, 5, 7]
}}

要求：
- 顶层必须是一个对象，且只包含 "delete" 一个字段；
- "delete" 的值是一个整数数组，每个整数对应上面实体列表中的 idx；
- 如果本批所有实体都应该保留，则输出："delete": []；
- 不要在 JSON 外输出任何多余文本，不要使用 Markdown 代码块标记。
"""

            try:
                response = self.deepseek.chat_completions_create([
                    {"role": "user", "content": prompt}
                ], max_retries=2, max_tokens=1000)

                content = response['choices'][0]['message']['content']
                json_start = content.find('{')
                json_end = content.rfind('}') + 1

                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end]
                    try:
                        result = json.loads(json_str)
                        ids_to_delete = result.get('delete', [])

                        for local_idx in ids_to_delete:
                            if 1 <= local_idx <= len(indexed_candidates):
                                entity_name = indexed_candidates[local_idx - 1]['name']
                                to_delete_names.add(entity_name)
                                logger.debug(f"标记删除: idx={local_idx}, name={entity_name}")

                        logger.info(f"批次 {batch_idx + 1} 完成: 标记删除 {len(ids_to_delete)} 个实体")

                    except json.JSONDecodeError as e:
                        logger.error(f"AI 实体清洗 JSON 解析失败: {e}")
                        logger.error(f"原始内容: {json_str[:500]}")
                else:
                    logger.warning(f"批次 {batch_idx + 1} 无法从响应中提取 JSON")

            except Exception as e:
                logger.error(f"AI 实体清洗批次 {batch_idx + 1} 失败: {str(e)}", exc_info=True)

            if batch_idx < total_batches - 1:
                import random
                delay = random.uniform(1, 2)
                time.sleep(delay)

        deleted_count = len(to_delete_names)

        if deleted_count > 0:
            new_entities = [e for e in entities if e.get('name') not in to_delete_names]
            new_relationships = [
                r for r in relationships
                if r.get('source') not in to_delete_names and r.get('target') not in to_delete_names
            ]
            deleted_relationships_count = len(relationships) - len(new_relationships)

            logger.info("=" * 50)
            logger.info(f"AI 实体清洗完成:")
            logger.info(f"  候选实体: {candidate_count}个")
            logger.info(f"  删除实体: {deleted_count}个")
            logger.info(f"  删除关系: {deleted_relationships_count}个")
            logger.info(f"  剩余实体: {len(new_entities)}个")
            logger.info(f"  剩余关系: {len(new_relationships)}个")
            logger.info("=" * 50)

            logger.info("以下实体被 AI 判定为无知识价值，已删除：")
            for name in sorted(to_delete_names):
                logger.info(f"  - {name}")

            return new_entities, new_relationships
        else:
            logger.info(f"AI 实体清洗完成: 候选={candidate_count}个, 删除=0个")
            return entities, relationships

    def _extract_from_slide_batch(self, batch_slides: List[Dict], document_name: str = "") -> Tuple[
        List[Dict], List[Dict]]:
        """批量处理 - 强调知识点间关系"""
        if not batch_slides:
            logger.warning("批次中无幻灯片数据")
            return [], []

        combined_text = ""
        slide_info_list = []

        for slide in batch_slides:
            slide_content = slide.get('content', {})
            text_content = ""

            if isinstance(slide_content, dict):
                text_fields = ['text', 'content', 'raw_text', 'texts']
                for field in text_fields:
                    if field in slide_content:
                        content = slide_content[field]
                        if isinstance(content, list):
                            text_content = ' '.join(str(item) for item in content)
                        else:
                            text_content = str(content)
                        break
                if not text_content:
                    text_content = str(slide_content)
            else:
                text_content = str(slide_content)

            if text_content and text_content.strip() != "" and text_content != "{}":
                combined_text += f"\n\n{text_content}"
                slide_info_list.append({
                    "source_file": slide.get('source_file', document_name),
                    "page_number": slide.get('slide_number', 0)
                })

        if not combined_text.strip():
            logger.warning("批次中无有效文本内容")
            return [], []

        max_length = 8000
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "..."
            logger.info(f"批次文本过长，截取前{max_length}字符")

        prompt = f"""
你是一位经验丰富的讲授《{self.course_name}》课程的大学教师，正在为本科生讲授专业课程。
现在你拿到了本次课程的PPT内容，需要从中提取核心知识点和它们之间的关系，帮助学生构建完整的知识体系。

**你的任务**：
请**通读全文理解上下文后**，从教学角度，结合《{self.course_name}》领域的专业知识，抽取课程中的核心知识点（实体）和知识点之间的关系。
现在只抽取**知识点（实体）与知识点之间**的语义关系，暂时不要抽取"章节之间"的关系。

**核心要求**：
1. 综合分析全文，识别跨章节的知识关联和层次结构
2. 同一概念多次出现时合并理解，不要重复创建
3. 只抽取课程知识点，忽略"页面"、"幻灯片"、"第X章"等文档结构
4. 关系必须有实际教学价值，能帮助学生理解知识体系

**课程PPT内容**：
{combined_text}

---

**实体类型**：核心概念、硬件组件、技术特性、参数规格、编程概念、应用场景

**关系类型**（限制为以下几类，使用英文大写形式）：
- PREREQUISITE_OF（前置于）：学习A需要先学B
- APPLIES_TO（应用于）：A应用于B
- DEFINES（定义了）：A定义了B
- IS_DEFINED_AS（被定义为）：A被定义为B
- RELATED_TO（相关于）：A与B相关

**禁止**：
-  不创建"页面X"、"第X章"、"幻灯片X"等文档结构实体
-  不创建"页面→知识点"、"提到"、"出现在"等无教学意义的关系
-  不使用中文关系类型，必须使用上述英文大写形式

**输出格式**（确保JSON格式正确）：
{{
    "entities": [
        {{"name": "实体名称", "type": "类型", "description": "结合课程上下文的描述"}}
    ],
    "relationships": [
        {{"source": "源知识点", "target": "目标知识点", "relation": "PREREQUISITE_OF", "description": "关系依据"}}
    ]
}}

注意：所有字符串用双引号，最后元素后无逗号，特殊字符需转义。relation字段必须使用上述英文大写形式。
"""

        try:
            response = self.deepseek.chat_completions_create([
                {"role": "user", "content": prompt}
            ], max_retries=3, max_tokens=8192)

            content = response['choices'][0]['message']['content']
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)

                entities = result.get('entities', [])
                for entity in entities:
                    entity['source'] = 'batch_text'
                    if slide_info_list:
                        entity['source_file'] = slide_info_list[0]['source_file']
                        entity['page_number'] = slide_info_list[0]['page_number']

                relationships = result.get('relationships', [])
                for rel in relationships:
                    rel['source_type'] = 'knowledge_level'

                logger.info(f"批次抽取成功: 实体={len(entities)}个, 关系={len(relationships)}个")
                return entities, relationships

        except Exception as e:
            logger.error(f"批次实体抽取失败: {str(e)}", exc_info=True)

        return [], []

    def extract_chapter_relationships(self, chapter_summaries: List[Dict]) -> List[Dict]:
        """抽取章节-章节之间的关系"""
        logger.info(f"开始抽取章节关系: {len(chapter_summaries)}个章节")

        if not chapter_summaries:
            logger.warning("章节摘要列表为空")
            return []

        chapter_list = "\n".join([
            f"- {ch.get('chapter_name', '未命名章节')}: {ch.get('summary', '无摘要')[:200]}"
            for ch in chapter_summaries
        ])

        prompt = f"""
        你是一位经验丰富的讲授《{self.course_name}》课程的大学教师，现在正在以**教学设计者（curriculum designer）**的视角，为这门课设计一个清晰合理的章节学习路径图。

        以下是课程各章节的摘要信息（每一行包含章节名称和该章节的简要内容）：

        {chapter_list}

        请你只把这里出现的章节视为"实体"（chapter），**不要再抽取具体知识点**。你的任务是，从"课程结构"和"学生认知负荷"的角度，分析这些章节之间的关系。

        --------------------
        【任务说明】
        1. 判断章节之间的前后顺序（前置关系）
        2. 判断章节之间的内容相关性（平行或相关章节）

        【关系类型】（只允许以下两种，必须使用大写字母）：
        - PRECEDES：A 章节是 B 章节的前置（学习顺序），即在正常教学中学生应当先学 A 再学 B。
        - RELATED_CHAPTER：A 章节与 B 章节内容相关、互为补充或平行展开，但不存在严格的前后依赖顺序。

        --------------------
        【输出格式（必须严格遵守）】
        请**只输出**一个 JSON 对象，格式如下：

        {{
          "relationships": [
            {{
              "source": "章节A",
              "target": "章节B",
              "relation": "PRECEDES",
              "description": "一句话概括关系",
              "long_description": "2-4 句详细教学说明"
            }}
          ]
        }}
        """

        try:
            response = self.deepseek.chat_completions_create([
                {"role": "user", "content": prompt}
            ], max_retries=2, max_tokens=3000)

            content = response['choices'][0]['message']['content']
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)

                relationships = result.get('relationships', [])
                for rel in relationships:
                    rel['source_type'] = 'chapter_level'

                logger.info(f"章节关系抽取成功: {len(relationships)}个")
                return relationships

        except Exception as e:
            logger.error(f"章节关系抽取失败: {str(e)}", exc_info=True)

        return []

    def _discover_global_relationships(self, entities: List[Dict], existing_relationships: List[Dict]) -> List[Dict]:
        """全局关系挖掘"""
        logger.info(f"分析 {len(entities)} 个实体，发现隐含关系")

        entity_names = [e['name'] for e in entities]
        entity_types = {e['name']: e.get('type', 'unknown') for e in entities}

        if len(entity_names) > 50:
            logger.warning(f"实体过多，只分析前50个核心实体")
            entity_names = entity_names[:50]

        entity_list = "\n".join([f"- {name} ({entity_types.get(name, 'unknown')})" for name in entity_names])

        prompt = f"""
        你是一位经验丰富的讲授《{self.course_name}》课程的大学教师，现在需要站在**知识结构规划者**的角度，基于已有的实体列表，为本课程构建一个合理的知识点关系网络。

        以下是从课程材料中抽取出的实体列表：

        {entity_list}

        【可用关系类型】（relation 字段必须使用以下英文大写形式之一）：
        - PREREQUISITE_OF
        - APPLIES_TO
        - DEFINES
        - IS_DEFINED_AS
        - RELATED_TO

        【输出格式】
        {{
          "relationships": [
            {{
              "source": "实体A",
              "target": "实体B",
              "relation": "PREREQUISITE_OF",
              "description": "一句话依据",
              "confidence": 0.8
            }}
          ]
        }}

        严格要求：source/target 必须来自上述实体列表；relation 只能取上述 5 种之一；confidence 在 0~1 之间。
        """

        try:
            response = self.deepseek.chat_completions_create([
                {"role": "user", "content": prompt}
            ], max_retries=2, max_tokens=3000)

            content = response['choices'][0]['message']['content']
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)

                new_relationships = result.get('relationships', [])
                for rel in new_relationships:
                    rel['source_type'] = 'global_discovery'
                    rel['is_inferred'] = True

                logger.info(f"发现隐含关系: {len(new_relationships)}个")
                return new_relationships

        except Exception as e:
            logger.error(f"全局关系挖掘失败: {str(e)}", exc_info=True)

        return []

    def _filter_knowledge_relationships(self, relationships: List[Dict], entities: List[Dict]) -> List[Dict]:
        """过滤非知识关系"""
        logger.info(f"开始过滤关系: 原始关系={len(relationships)}个, 实体={len(entities)}个")

        entity_names = set()
        for e in entities:
            name = e['name'].lower().strip()
            name = name.replace('(', '').replace(')', '').replace('(', '').replace(')', '')
            name = name.replace(',', '').replace('，', '').replace('、', '').replace(' ', '')
            entity_names.add(name)

        blacklist_keywords = [
            'batch_text', 'batch', 'source', 'target', 'json', 'api', 'http',
            'output', 'input', 'data', 'file', 'path', 'url', 'id', 'index',
            '页面', 'page', 'slide', '幻灯片', 'ppt', '文档', 'document',
            '第', '章', 'chapter', 'section', '图', 'figure', 'img', 'image',
            'null', 'none', 'undefined', 'unknown', 'test', 'example',
            'source_type', 'global_discovery', 'is_inferred', 'confidence'
        ]

        filtered = []
        filtered_count = 0

        def smart_match(name: str) -> bool:
            if not name:
                return False
            name_lower = name.lower().strip()
            for keyword in blacklist_keywords:
                if keyword in name_lower:
                    return False
            name_clean = name_lower.replace('(', '').replace(')', '').replace('(', '').replace(')', '')
            name_clean = name_clean.replace(',', '').replace('，', '').replace('、', '').replace(' ', '')
            if name_clean in entity_names:
                return True
            for entity_name in entity_names:
                if len(entity_name) >= 4 and len(name_clean) >= 4:
                    if entity_name in name_clean or name_clean in entity_name:
                        is_blacklist_match = False
                        for keyword in blacklist_keywords:
                            if keyword in name_clean or keyword in entity_name:
                                is_blacklist_match = True
                                break
                        if not is_blacklist_match:
                            return True
            return False

        for rel in relationships:
            source = rel.get('source', '')
            target = rel.get('target', '')
            if smart_match(source) and smart_match(target):
                filtered.append(rel)
            else:
                filtered_count += 1
                logger.debug(f"过滤关系: {source} -> {target}")

        logger.info(f"过滤完成: 过滤={filtered_count}个, 保留={len(filtered)}个")
        return filtered

    def _extract_from_image(self, image_data: Dict, document_name: str = "") -> List[Dict]:
        """从图片数据中抽取实体"""
        entities = []
        image_path = image_data.get('image_path', '')
        filename = image_data.get('filename', '')
        page_number = image_data.get('slide_number', 0)

        logger.debug(f"从图片提取实体: {filename}")

        descriptions = image_data.get('descriptions', [])
        for desc_item in descriptions:
            desc_text = desc_item.get('description', '')
            confidence = desc_item.get('confidence', 0)
            if desc_text and confidence > 0.05:
                entities.append({
                    'name': desc_text,
                    'type': 'image_concept',
                    'description': f'从图片描述中识别: {desc_text}',
                    'confidence': confidence,
                    'source': 'image_description',
                    'source_file': document_name,
                    'page_number': page_number,
                    'image_path': image_path,
                    'filename': filename
                })

        ocr_text = image_data.get('ocr_text', '')
        if ocr_text:
            for keyword in self.arduino_keywords:
                if keyword.lower() in ocr_text.lower():
                    entities.append({
                        'name': keyword,
                        'type': 'hardware_component',
                        'description': f'从图片OCR中识别的{keyword}',
                        'source': 'image_ocr',
                        'source_file': document_name,
                        'page_number': page_number,
                        'image_path': image_path,
                        'filename': filename
                    })

        if 'arduino' in filename.lower():
            entities.append({
                'name': 'Arduino',
                'type': 'hardware_platform',
                'description': '从文件名识别的Arduino平台',
                'source': 'filename',
                'source_file': document_name,
                'page_number': page_number,
                'image_path': image_path,
                'filename': filename
            })

        logger.debug(f"图片实体提取完成: {filename}, 实体数={len(entities)}")
        return entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重"""
        original_count = len(entities)
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity['name'].lower(), entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        logger.info(f"实体去重: {original_count} -> {len(unique_entities)}")
        return unique_entities

    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:

        """关系去重"""
        original_count = len(relationships)
        seen = set()
        unique_relationships = []

        for rel in relationships:
            key = (rel['source'].lower(), rel['target'].lower(), rel['relation'])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        logger.info(f"关系去重: {original_count} -> {len(unique_relationships)}")
        return unique_relationships


from neo4j import GraphDatabase


class Neo4jKnowledgeGraph:
    """Neo4j知识图谱连接器 - 支持课程-章节-知识点三层结构"""

    RELATION_TYPE_MAPPING = {
        "前置于": "PREREQUISITE_OF",
        "precedes": "PREREQUISITE_OF",
        "先于": "PREREQUISITE_OF",
        "prerequisite_of": "PREREQUISITE_OF",
        "应用于": "APPLIES_TO",
        "applies_to": "APPLIES_TO",
        "定义了": "DEFINES",
        "defines": "DEFINES",
        "被定义为": "IS_DEFINED_AS",
        "is_defined_as": "IS_DEFINED_AS",
        "相关于": "RELATED_TO",
        "related_to": "RELATED_TO",
        "related_chapter": "RELATED_CHAPTER"
    }

    def __init__(self, uri: str, user: str, password: str):
        """初始化Neo4j连接"""
        self.driver = None
        try:
            logger.info(f"尝试连接Neo4j: uri={uri}, user={user}")

            config = {
                "keep_alive": True,
                "max_connection_lifetime": 3600,
                "max_connection_pool_size": 100
            }
            self.driver = GraphDatabase.driver(uri, auth=(user, password), **config)

            with self.driver.session() as session:
                session.run("RETURN 1")

            logger.info(f"Neo4j连接成功: {uri}")

        except Exception as e:
            logger.error(f"Neo4j连接失败: {str(e)}", exc_info=True)
            raise

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def clear_database(self):
        """清空数据库"""
        logger.warning("开始清空Neo4j数据库")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("数据库已清空")

    def ensure_course_chapter(self, course_name: str, chapter_name: str,
                              chapter_summary: str = "", chapter_keywords: List[str] = None):
        """确保课程、章节节点存在，并建立双向关系"""
        logger.info(f"确保上层结构存在: 课程={course_name}, 章节={chapter_name}")

        with self.driver.session() as session:
            query = """
            MERGE (course:Entity {name: $course_name, type: 'course'})
            SET course.layer = 1
            SET course.updated_at = datetime()

            MERGE (chapter:Entity {name: $chapter_name, type: 'chapter'})
            SET chapter.layer = 2
            SET chapter.updated_at = datetime()
            """

            params = {
                "course_name": course_name,
                "chapter_name": chapter_name
            }

            if chapter_summary:
                query += """
                SET chapter.summary = $chapter_summary
                SET chapter.summary_updated_at = datetime()
                """
                params["chapter_summary"] = chapter_summary

            if chapter_keywords:
                keywords_str = ", ".join(chapter_keywords)
                query += """
                SET chapter.keywords = $chapter_keywords
                """
                params["chapter_keywords"] = keywords_str

            query += """
            MERGE (course)-[:CONTAINS]->(chapter)
            MERGE (chapter)-[:BELONGS_TO]->(course)
            """

            session.run(query, params)

        logger.info(f"上层结构创建完成: {course_name} -> {chapter_name}")

    def bind_entity_to_chapter(self, entity_name: str, chapter_name: str):
        """将知识点实体与章节建立双向关系"""
        logger.debug(f"绑定知识点到章节: {entity_name} -> {chapter_name}")

        with self.driver.session() as session:
            query = """
            MATCH (ch:Entity {name: $chapter_name, type: 'chapter'})
            MATCH (e:Entity {name: $entity_name})
            WHERE e.type <> 'course' AND e.type <> 'chapter'
            MERGE (ch)-[:CONTAINS]->(e)
            MERGE (e)-[:BELONGS_TO]->(ch)
            """
            session.run(query, entity_name=entity_name, chapter_name=chapter_name)

    def normalize_relation_type(self, relation_raw: str) -> str:
        """规范化关系类型"""
        relation_clean = relation_raw.lower().strip().replace(' ', '_')

        if relation_clean in self.RELATION_TYPE_MAPPING:
            normalized = self.RELATION_TYPE_MAPPING[relation_clean]
            logger.debug(f"关系类型规范化: {relation_raw} -> {normalized}")
            return normalized

        if relation_raw.upper() in self.RELATION_TYPE_MAPPING.values():
            return relation_raw.upper()

        fallback = relation_raw.upper().replace(' ', '_')
        logger.debug(f"关系类型使用fallback: {relation_raw} -> {fallback}")
        return fallback

    def create_entity_node(self, entity_name: str, entity_type: str, description: str = "",
                           metadata: Dict = None):
        """创建实体节点 - 知识点自动设置layer=3"""
        logger.debug(f"创建实体节点: name={entity_name}, type={entity_type}")

        with self.driver.session() as session:
            query = """
            MERGE (e:Entity {name: $entity_name})
            SET e.type = $entity_type
            SET e.description = $description
            SET e.updated_at = datetime()
            """

            params = {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "description": description
            }

            if entity_type not in ['course', 'chapter']:
                query += "SET e.layer = 3 "

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        query += f"SET e.{key} = ${key} "
                        params[key] = value

            session.run(query, params)

    def create_relationship(self, source_name: str, target_name: str, relation_type: str,
                            properties: Dict = None):
        """创建关系"""
        logger.debug(f"创建关系: {source_name} -[{relation_type}]-> {target_name}")

        with self.driver.session() as session:
            session.run("""
                MERGE (s:Entity {name: $source_name})
                MERGE (t:Entity {name: $target_name})
            """, source_name=source_name, target_name=target_name)

            query = f"""
            MATCH (s:Entity {{name: $source_name}})
            MATCH (t:Entity {{name: $target_name}})
            MERGE (s)-[r:{relation_type}]->(t)
            SET r.created_at = datetime()
            """

            params = {"source_name": source_name, "target_name": target_name}

            if properties:
                for key, value in properties.items():
                    if isinstance(value, (str, int, float, bool)):
                        query += f"SET r.{key} = ${key} "
                        params[key] = value

            session.run(query, params)

    @log_execution_time(logger)
    def save_extracted_data(self, extracted_data, document_name: str, course_name: str, chapter_name: str,
                            chapter_summary: str = "", chapter_keywords: List[str] = None):
        """保存知识图谱 - 支持三层结构"""
        logger.info(f"开始保存知识图谱到Neo4j: 课程={course_name}, 文档={document_name}, 章节={chapter_name}")

        try:
            self.ensure_course_chapter(course_name, chapter_name, chapter_summary, chapter_keywords)

            logger.info(f"创建实体节点: {len(extracted_data.entities)}个")
            entity_count = 0

            for entity in extracted_data.entities:
                try:
                    entity_type = entity.get('type', 'unknown')

                    if entity_type in ['course', 'chapter']:
                        continue

                    metadata = {k: v for k, v in entity.items()
                                if k not in ['name', 'type', 'description'] and
                                isinstance(v, (str, int, float, bool))}

                    metadata['course_name'] = course_name
                    metadata['chapter_name'] = chapter_name
                    metadata['source_file'] = entity.get('source_file', document_name)

                    if 'page_number' in entity:
                        metadata['page_number'] = entity['page_number']

                    self.create_entity_node(
                        entity['name'],
                        entity_type,
                        entity.get('description', ''),
                        metadata
                    )
                    entity_count += 1

                    self.bind_entity_to_chapter(entity['name'], chapter_name)

                except Exception as e:
                    logger.warning(f"创建实体失败 {entity['name']}: {str(e)}")
                    continue

            knowledge_relations = []
            chapter_relations = []

            for rel in extracted_data.relationships:
                source_type = rel.get('source_type', 'knowledge_level')
                if source_type == 'chapter_level':
                    chapter_relations.append(rel)
                else:
                    knowledge_relations.append(rel)

            logger.info(f"关系分类: 知识点关系={len(knowledge_relations)}个, 章节关系={len(chapter_relations)}个")

            logger.info(f"创建知识点关系: {len(knowledge_relations)}个")
            knowledge_relation_count = 0

            for rel in knowledge_relations:
                try:
                    properties = {k: v for k, v in rel.items()
                                  if k not in ['source', 'target', 'relation'] and
                                  isinstance(v, (str, int, float, bool))}

                    relation_type_norm = self.normalize_relation_type(rel['relation'])

                    self.create_relationship(
                        rel['source'],
                        rel['target'],
                        relation_type_norm,
                        properties
                    )
                    knowledge_relation_count += 1

                except Exception as e:
                    logger.warning(f"创建知识点关系失败 {rel['source']}->{rel['target']}: {str(e)}")
                    continue

            logger.info(f"创建章节关系: {len(chapter_relations)}个")
            chapter_relation_count = 0

            for rel in chapter_relations:
                try:
                    properties = {k: v for k, v in rel.items()
                                  if k not in ['source', 'target', 'relation'] and
                                  isinstance(v, (str, int, float, bool))}

                    relation_type_norm = self.normalize_relation_type(rel['relation'])

                    self.create_relationship(
                        rel['source'],
                        rel['target'],
                        relation_type_norm,
                        properties
                    )
                    chapter_relation_count += 1

                except Exception as e:
                    logger.warning(f"创建章节关系失败 {rel['source']}->{rel['target']}: {str(e)}")
                    continue

            total_relations = knowledge_relation_count + chapter_relation_count

            logger.info(
                f"知识图谱保存完成: "
                f"实体={entity_count}/{len(extracted_data.entities)}, "
                f"知识点关系={knowledge_relation_count}, "
                f"章节关系={chapter_relation_count}, "
                f"总关系={total_relations}/{len(extracted_data.relationships)}"
            )

            return {
                'document': document_name,
                'course': course_name,
                'chapter': chapter_name,
                'entities_saved': entity_count,
                'relationships_saved': total_relations,
                'knowledge_relations': knowledge_relation_count,
                'chapter_relations': chapter_relation_count,
                'success': True
            }

        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}", exc_info=True)
            return {
                'document': document_name,
                'course': course_name,
                'chapter': chapter_name,
                'entities_saved': 0,
                'relationships_saved': 0,
                'success': False,
                'error': str(e)
            }

    def query_entities(self, limit: int = 10) -> List[Dict]:
        """查询实体"""
        logger.debug(f"查询实体: limit={limit}")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.name as name, e.type as type, e.description as description, 
                       e.layer as layer, e.source_file as source_file, e.page_number as page_number,
                       e.summary as summary, e.keywords as keywords
                LIMIT $limit
            """, limit=limit)

            return [dict(record) for record in result]

    def query_relationships(self, limit: int = 10) -> List[Dict]:
        """查询关系"""
        logger.debug(f"查询关系: limit={limit}")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Entity)-[r]->(t:Entity)
                RETURN s.name as source, type(r) as relation, t.name as target,
                       r.description as description, r.long_description as long_description
                LIMIT $limit
            """, limit=limit)

            return [dict(record) for record in result]

    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        logger.debug("获取数据库统计信息")

        with self.driver.session() as session:
            node_result = session.run("MATCH (n:Entity) RETURN count(n) as total_nodes")
            total_nodes = node_result.single()['total_nodes']

            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_relationships")
            total_relationships = rel_result.single()['total_relationships']

            entity_type_result = session.run("""
                MATCH (e:Entity)
                RETURN e.type as entity_type, count(e) as count
                ORDER BY count DESC
            """)
            entity_types = [dict(record) for record in entity_type_result]

            rel_type_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relation_type, count(r) as count
                ORDER BY count DESC
            """)
            relation_types = [dict(record) for record in rel_type_result]

            layer_result = session.run("""
                MATCH (e:Entity)
                WHERE e.layer IS NOT NULL
                RETURN e.layer as layer, count(e) as count
                ORDER BY e.layer
            """)
            layers = [dict(record) for record in layer_result]

            stats = {
                'total_nodes': total_nodes,
                'total_relationships': total_relationships,
                'entity_types': entity_types,
                'relation_types': relation_types,
                'layers': layers
            }

            logger.info(f"统计信息: 节点={total_nodes}, 关系={total_relationships}")
            return stats

    def search_entities_by_name(self, name_pattern: str, limit: int = 10) -> List[Dict]:
        """按名称搜索实体"""
        logger.debug(f"搜索实体: pattern={name_pattern}, limit={limit}")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($pattern)
                RETURN e.name as name, e.type as type, e.description as description, 
                       e.layer as layer, e.source_file as source_file
                LIMIT $limit
            """, pattern=name_pattern, limit=limit)

            return [dict(record) for record in result]

    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> Dict:
        """获取实体的邻居节点"""
        logger.debug(f"获取实体邻居: entity={entity_name}, depth={depth}")

        with self.driver.session() as session:
            result = session.run(f"""
                MATCH path = (e:Entity {{name: $entity_name}})-[*1..{depth}]-(neighbor)
                RETURN neighbor.name as name, neighbor.type as type, 
                       neighbor.description as description, neighbor.layer as layer
                LIMIT 20
            """, entity_name=entity_name)

            neighbors = [dict(record) for record in result]

            rel_result = session.run("""
                MATCH (e:Entity {name: $entity_name})-[r]-(neighbor)
                RETURN neighbor.name as neighbor, type(r) as relation,
                       CASE WHEN startNode(r).name = $entity_name 
                            THEN 'outgoing' ELSE 'incoming' END as direction
            """, entity_name=entity_name)

            relationships = [dict(record) for record in rel_result]

            return {
                'entity': entity_name,
                'neighbors': neighbors,
                'relationships': relationships
            }

    @log_execution_time(logger)
    def export_to_csv(self, course_name: str, chapter_name: str,
                      nodes_csv_path: str, edges_csv_path: str):
        """导出知识图谱到CSV文件（支持可选过滤）"""
        course_name = (course_name or "").strip()
        chapter_name = (chapter_name or "").strip()

        if not course_name and not chapter_name:
            export_scope = "全部数据"
        elif course_name and not chapter_name:
            export_scope = f"课程【{course_name}】"
        elif not course_name and chapter_name:
            export_scope = f"章节【{chapter_name}】"
        else:
            export_scope = f"课程【{course_name}】的章节【{chapter_name}】"

        logger.info(f"开始导出CSV: {export_scope}")

        try:
            with self.driver.session() as session:
                # 节点查询
                if not course_name and not chapter_name:
                    nodes_query = """
                    MATCH (e:Entity)
                    RETURN e.name AS name, e.type AS type, e.layer AS layer,
                      e.course_name AS course_name, e.chapter_name AS chapter_name,
                      e.source_file AS source_file, e.page_number AS page_number,
                      e.description AS description, e.summary AS summary, e.keywords AS keywords
                    ORDER BY e.layer, e.course_name, e.chapter_name, e.name
                    """
                    nodes_params = {}
                elif course_name and not chapter_name:
                    nodes_query = """
                    MATCH (e:Entity)
                    WHERE e.course_name = $course_name
                       OR (e.type = 'course' AND e.name = $course_name)
                    RETURN e.name AS name, e.type AS type, e.layer AS layer,
                      e.course_name AS course_name, e.chapter_name AS chapter_name,
                      e.source_file AS source_file, e.page_number AS page_number,
                      e.description AS description, e.summary AS summary, e.keywords AS keywords
                    ORDER BY e.layer, e.chapter_name, e.name
                    """
                    nodes_params = {"course_name": course_name}
                elif not course_name and chapter_name:
                    nodes_query = """
                    MATCH (e:Entity)
                    WHERE e.chapter_name = $chapter_name
                       OR (e.type = 'chapter' AND e.name = $chapter_name)
                    RETURN e.name AS name, e.type AS type, e.layer AS layer,
                      e.course_name AS course_name, e.chapter_name AS chapter_name,
                      e.source_file AS source_file, e.page_number AS page_number,
                      e.description AS description, e.summary AS summary, e.keywords AS keywords
                    ORDER BY e.layer, e.name
                    """
                    nodes_params = {"chapter_name": chapter_name}
                else:
                    nodes_query = """
                    MATCH (e:Entity)
                    WHERE e.course_name = $course_name AND e.chapter_name = $chapter_name
                       OR (e.type = 'chapter' AND e.name = $chapter_name)
                       OR (e.type = 'course' AND e.name = $course_name)
                    RETURN e.name AS name, e.type AS type, e.layer AS layer,
                      e.course_name AS course_name, e.chapter_name AS chapter_name,
                      e.source_file AS source_file, e.page_number AS page_number,
                      e.description AS description, e.summary AS summary, e.keywords AS keywords
                    ORDER BY e.layer, e.name
                    """
                    nodes_params = {"course_name": course_name, "chapter_name": chapter_name}

                nodes_result = session.run(nodes_query, nodes_params)
                nodes_data = [dict(record) for record in nodes_result]
                logger.info(f"查询到节点: {len(nodes_data)}个")

                if nodes_data:
                    with open(nodes_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        fieldnames = ['name', 'type', 'layer', 'course_name', 'chapter_name',
                                      'source_file', 'page_number', 'description', 'summary', 'keywords']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(nodes_data)
                    logger.info(f"节点CSV导出成功: {nodes_csv_path}")

                # 边查询
                if not course_name and not chapter_name:
                    edges_query = """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    RETURN s.name AS source, t.name AS target, type(r) AS relation_type,
                      r.description AS description, r.long_description AS long_description,
                      r.source_type AS source_type
                    """
                    edges_params = {}
                elif course_name and not chapter_name:
                    edges_query = """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE (s.course_name = $course_name OR s.type = 'course')
                      AND (t.course_name = $course_name OR t.type = 'course')
                    RETURN s.name AS source, t.name AS target, type(r) AS relation_type,
                      r.description AS description, r.long_description AS long_description,
                      r.source_type AS source_type
                    """
                    edges_params = {"course_name": course_name}
                elif not course_name and chapter_name:
                    edges_query = """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE (s.chapter_name = $chapter_name OR s.type = 'chapter')
                      AND (t.chapter_name = $chapter_name OR t.type = 'chapter')
                    RETURN s.name AS source, t.name AS target, type(r) AS relation_type,
                      r.description AS description, r.long_description AS long_description,
                      r.source_type AS source_type
                    """
                    edges_params = {"chapter_name": chapter_name}
                else:
                    edges_query = """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE (s.course_name = $course_name AND s.chapter_name = $chapter_name)
                       OR (s.type = 'chapter' AND s.name = $chapter_name)
                       OR (s.type = 'course' AND s.name = $course_name)
                    RETURN s.name AS source, t.name AS target, type(r) AS relation_type,
                      r.description AS description, r.long_description AS long_description,
                      r.source_type AS source_type
                    """
                    edges_params = {"course_name": course_name, "chapter_name": chapter_name}

                edges_result = session.run(edges_query, edges_params)
                edges_data = [dict(record) for record in edges_result]
                logger.info(f"查询到边: {len(edges_data)}个")

                if edges_data:
                    with open(edges_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        fieldnames = ['source', 'target', 'relation_type', 'description',
                                      'long_description', 'source_type']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(edges_data)
                    logger.info(f"边CSV导出成功: {edges_csv_path}")

                return {
                    'success': True,
                    'nodes_count': len(nodes_data),
                    'edges_count': len(edges_data),
                    'nodes_file': nodes_csv_path,
                    'edges_file': edges_csv_path,
                    'export_scope': export_scope
                }

        except Exception as e:
            logger.error(f"CSV导出失败: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}

    # ==================== ★ 新增：前端 JSON 导出 ====================

    @log_execution_time(logger)
    def export_to_frontend_json(self, course_names: Optional[List[str]] = None) -> Dict:
        """
        导出为前端图可视化格式（echarts-like）

        输出结构：
        {
            "课程名1": {
                "nodes": [
                    {"id": "...", "name": "...", "type": "course|chapter|knowledge",
                     "description": "...", "symbolSize": 80|60|40},
                    ...
                ],
                "links": [
                    {"source": "id_a", "target": "id_b"},
                    ...
                ]
            },
            "课程名2": {...}
        }

        Args:
            course_names: 要导出的课程名称列表；若为 None，则导出全部课程
        """
        logger.info(f"开始导出前端 JSON 格式: course_names={course_names}")

        result = {}

        try:
            with self.driver.session() as session:
                # 1. 获取要导出的课程列表
                if course_names:
                    courses = course_names
                else:
                    courses_res = session.run(
                        "MATCH (c:Entity {type: 'course'}) RETURN c.name AS name"
                    )
                    courses = [r["name"] for r in courses_res]

                logger.info(f"共需导出 {len(courses)} 个课程: {courses}")

                # 2. 逐个课程构造子图
                for course_name in courses:
                    logger.info(f"导出课程: {course_name}")

                    nodes = []
                    links = []
                    seen_node_ids = set()

                    # 2.1 查询课程节点
                    course_res = session.run(
                        "MATCH (c:Entity {name: $n, type: 'course'}) RETURN c",
                        n=course_name
                    )
                    course_record = course_res.single()
                    if not course_record:
                        logger.warning(f"课程不存在: {course_name}")
                        continue

                    # 课程节点 id 固定为 "course"
                    nodes.append({
                        "id": "course",
                        "name": course_name,
                        "type": "course",
                        "symbolSize": 80
                    })
                    seen_node_ids.add("course")
                    # 课程名 -> id 映射（方便后面查）
                    name_to_id = {course_name: "course"}

                    # 2.2 查询章节节点（课程 CONTAINS 章节）
                    chapters_res = session.run("""
                        MATCH (c:Entity {name: $n, type: 'course'})-[:CONTAINS]->(ch:Entity {type: 'chapter'})
                        RETURN ch.name AS name, ch.summary AS summary
                    """, n=course_name)
                    chapters = list(chapters_res)

                    for idx, ch in enumerate(chapters, start=1):
                        ch_name = ch["name"]
                        ch_id = f"chapter{idx}"
                        node = {
                            "id": ch_id,
                            "name": ch_name,
                            "type": "chapter",
                            "symbolSize": 60
                        }
                        if ch.get("summary"):
                            node["description"] = ch["summary"]
                        nodes.append(node)
                        seen_node_ids.add(ch_id)
                        name_to_id[ch_name] = ch_id

                        # 课程 -> 章节 link
                        links.append({"source": "course", "target": ch_id})

                        # 2.3 查询章节下的知识点
                        kp_res = session.run("""
                            MATCH (ch:Entity {name: $ch_name, type: 'chapter'})-[:CONTAINS]->(e:Entity)
                            WHERE e.type <> 'course' AND e.type <> 'chapter'
                            RETURN e.name AS name, e.description AS description, e.type AS type
                        """, ch_name=ch_name)

                        for kp_idx, kp in enumerate(kp_res, start=1):
                            kp_name = kp["name"]
                            # 知识点 id：chapter{idx}_k{kp_idx}
                            kp_id = f"chapter{idx}_k{kp_idx}"

                            # 避免同名实体重复
                            if kp_name in name_to_id:
                                continue

                            node = {
                                "id": kp_id,
                                "name": kp_name,
                                "type": "knowledge",
                                "symbolSize": 40
                            }
                            if kp.get("description"):
                                node["description"] = kp["description"]
                            nodes.append(node)
                            seen_node_ids.add(kp_id)
                            name_to_id[kp_name] = kp_id

                            # 章节 -> 知识点 link
                            links.append({"source": ch_id, "target": kp_id})

                    # 2.4 附加：知识点之间的关系（可选，用于展示跨知识点连接）
                    extra_rels_res = session.run("""
                        MATCH (s:Entity)-[r]->(t:Entity)
                        WHERE s.course_name = $n AND t.course_name = $n
                          AND type(r) <> 'CONTAINS' AND type(r) <> 'BELONGS_TO'
                        RETURN s.name AS source_name, t.name AS target_name, type(r) AS rel
                    """, n=course_name)

                    for rel in extra_rels_res:
                        s_name = rel["source_name"]
                        t_name = rel["target_name"]
                        if s_name in name_to_id and t_name in name_to_id:
                            links.append({
                                "source": name_to_id[s_name],
                                "target": name_to_id[t_name],
                                "relation": rel["rel"]
                            })

                    result[course_name] = {
                        "nodes": nodes,
                        "links": links
                    }

                    logger.info(f"  课程【{course_name}】: 节点={len(nodes)}, 连线={len(links)}")

            logger.info(f"前端 JSON 导出完成，共 {len(result)} 个课程")
            return result

        except Exception as e:
            logger.error(f"前端 JSON 导出失败: {str(e)}", exc_info=True)
            return {}

    # ==================== ★ 新增：6个 CRUD 方法 ====================

    def delete_entity(self, entity_name: str) -> Dict:
        """删除实体（及其所有关联关系）"""
        logger.info(f"[CRUD] 删除实体: {entity_name}")
        try:
            with self.driver.session() as session:
                # 先统计关联关系数
                res = session.run(
                    "MATCH (e:Entity {name: $n})-[r]-() RETURN count(r) AS c",
                    n=entity_name
                )
                rel_count = res.single()["c"]

                # 检查实体是否存在
                exist_res = session.run(
                    "MATCH (e:Entity {name: $n}) RETURN count(e) AS c",
                    n=entity_name
                )
                if exist_res.single()["c"] == 0:
                    logger.warning(f"实体不存在: {entity_name}")
                    return {"success": False, "error": "entity_not_found"}

                session.run("MATCH (e:Entity {name: $n}) DETACH DELETE e", n=entity_name)
                logger.info(f"实体删除成功: {entity_name}, 同时删除 {rel_count} 条关系")
                return {"success": True, "deleted_relationships": rel_count}
        except Exception as e:
            logger.error(f"删除实体失败: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def delete_relationship(self, source_name: str, target_name: str, relation_type: str) -> Dict:
        """删除指定关系"""
        rel = self.normalize_relation_type(relation_type)
        logger.info(f"[CRUD] 删除关系: {source_name} -[{rel}]-> {target_name}")
        try:
            with self.driver.session() as session:
                res = session.run(
                    f"MATCH (s:Entity {{name: $s}})-[r:{rel}]->(t:Entity {{name: $t}}) "
                    f"RETURN count(r) AS c",
                    s=source_name, t=target_name
                )
                count = res.single()["c"]

                if count > 0:
                    session.run(
                        f"MATCH (s:Entity {{name: $s}})-[r:{rel}]->(t:Entity {{name: $t}}) DELETE r",
                        s=source_name, t=target_name
                    )
                    logger.info(f"关系删除成功: {count}条")
                else:
                    logger.warning(f"关系不存在: {source_name} -[{rel}]-> {target_name}")

                return {"success": count > 0, "deleted_count": count}
        except Exception as e:
            logger.error(f"删除关系失败: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def update_entity(self, entity_name: str, updates: Dict) -> Dict:
        """更新实体属性"""
        logger.info(f"[CRUD] 更新实体: {entity_name}, updates={updates}")
        allowed = {"name", "type", "description", "summary", "keywords",
                   "course_name", "chapter_name", "source_file", "page_number"}
        filtered = {k: v for k, v in updates.items() if k in allowed}

        if not filtered:
            logger.warning("无有效更新字段")
            return {"success": False, "error": "no_valid_fields"}

        try:
            with self.driver.session() as session:
                # 重命名处理
                if "name" in filtered:
                    new_name = filtered.pop("name")
                    session.run(
                        "MATCH (e:Entity {name: $old}) SET e.name = $new, e.updated_at = datetime()",
                        old=entity_name, new=new_name
                    )
                    entity_name = new_name
                    logger.info(f"实体已重命名为: {new_name}")

                if filtered:
                    set_clause = ", ".join([f"e.{k} = ${k}" for k in filtered])
                    session.run(
                        f"MATCH (e:Entity {{name: $n}}) SET {set_clause}, e.updated_at = datetime()",
                        n=entity_name, **filtered
                    )

                logger.info(f"实体更新成功: {entity_name}")
                return {"success": True, "new_name": entity_name, "updated_fields": list(filtered.keys())}
        except Exception as e:
            logger.error(f"更新实体失败: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def update_relationship(self, source_name: str, target_name: str,
                            relation_type: str, updates: Dict) -> Dict:
        """更新关系属性"""
        rel = self.normalize_relation_type(relation_type)
        logger.info(f"[CRUD] 更新关系: {source_name} -[{rel}]-> {target_name}, updates={updates}")

        allowed = {"description", "long_description", "confidence", "source_type"}
        filtered = {k: v for k, v in updates.items() if k in allowed and isinstance(v, (str, int, float, bool))}

        if not filtered:
            return {"success": False, "error": "no_valid_fields"}

        try:
            with self.driver.session() as session:
                set_clause = ", ".join([f"r.{k} = ${k}" for k in filtered])
                res = session.run(
                    f"MATCH (s:Entity {{name: $s}})-[r:{rel}]->(t:Entity {{name: $t}}) "
                    f"SET {set_clause}, r.updated_at = datetime() "
                    f"RETURN count(r) AS c",
                    s=source_name, t=target_name, **filtered
                )
                count = res.single()["c"]
                logger.info(f"关系更新成功: {count}条")
                return {"success": count > 0, "updated_count": count}
        except Exception as e:
            logger.error(f"更新关系失败: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def add_entity(self, name: str, entity_type: str, course_name: str,
                   chapter_name: str, description: str = "",
                   metadata: Dict = None) -> Dict:
        """新增实体（自动绑定到章节）"""
        logger.info(f"[CRUD] 新增实体: name={name}, type={entity_type}, chapter={chapter_name}")
        try:
            meta = metadata.copy() if metadata else {}
            meta['course_name'] = course_name
            meta['chapter_name'] = chapter_name
            # 确保课程章节存在
            self.ensure_course_chapter(course_name, chapter_name)
            # 创建实体
            self.create_entity_node(name, entity_type, description, meta)
            # 绑定到章节
            if entity_type not in ['course', 'chapter']:
                self.bind_entity_to_chapter(name, chapter_name)
            logger.info(f"实体新增成功: {name}")
            return {"success": True, "name": name}
        except Exception as e:
            logger.error(f"新增实体失败: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def add_relationship(self, source_name: str, target_name: str,
                         relation_type: str, description: str = "",
                         long_description: str = "") -> Dict:
        """新增关系"""
        rel = self.normalize_relation_type(relation_type)
        logger.info(f"[CRUD] 新增关系: {source_name} -[{rel}]-> {target_name}")
        try:
            props = {}
            if description:
                props["description"] = description
            if long_description:
                props["long_description"] = long_description

            self.create_relationship(source_name, target_name, rel, props)
            logger.info(f"关系新增成功")
            return {"success": True, "source": source_name, "target": target_name, "relation": rel}
        except Exception as e:
            logger.error(f"新增关系失败: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ==================== CRUD 辅助方法 ====================

    def get_entity_detail(self, name: str) -> Dict:
        """获取实体详情"""
        with self.driver.session() as session:
            res = session.run("MATCH (e:Entity {name: $n}) RETURN e", n=name)
            record = res.single()
            return dict(record[0]) if record else {}

    def get_relationship_detail(self, source_name: str, target_name: str, relation_type: str) -> Dict:
        """获取关系详情"""
        rel = self.normalize_relation_type(relation_type)
        with self.driver.session() as session:
            res = session.run(
                f"MATCH (s:Entity {{name: $s}})-[r:{rel}]->(t:Entity {{name: $t}}) RETURN r",
                s=source_name, t=target_name
            )
            record = res.single()
            return dict(record[0]) if record else {}

    def preview_delete_entity(self, name: str) -> Dict:
        """预览删除实体将影响哪些关系"""
        with self.driver.session() as session:
            exist = session.run(
                "MATCH (e:Entity {name: $n}) RETURN count(e) AS c", n=name
            ).single()["c"]

            if exist == 0:
                return {"entity_exists": False, "relationship_count": 0, "relationships": []}

            res = session.run(
                "MATCH (e:Entity {name: $n})-[r]-(o) "
                "RETURN type(r) AS rel, o.name AS target, "
                "CASE WHEN startNode(r).name = $n THEN 'outgoing' ELSE 'incoming' END AS direction",
                n=name
            )
            rels = [dict(r) for r in res]
            return {
                "entity_exists": True,
                "relationship_count": len(rels),
                "relationships": rels
            }


# ==================== 工具函数 ====================

@log_execution_time(logger)
def reset_neo4j(neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    """清空Neo4j数据库"""
    logger.warning("=" * 60)
    logger.warning("准备清空Neo4j数据库")
    logger.warning("=" * 60)

    kg = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    try:
        kg.clear_database()
        logger.info("✅ 数据库清空成功")
    finally:
        kg.close()


@log_execution_time(logger)
def export_chapter_to_csv(neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                          course_name: str = "", chapter_name: str = "",
                          csv_output_dir: str = "csv_exports") -> Dict:
    """导出知识图谱到CSV文件"""
    course_name = (course_name or "").strip()
    chapter_name = (chapter_name or "").strip()

    if not course_name and not chapter_name:
        scope_desc = "全部数据"
        file_suffix = "ALL_COURSES_ALL_CHAPTERS"
    elif course_name and not chapter_name:
        scope_desc = f"课程【{course_name}】"
        safe_course_name = re.sub(r'[\\/:*?"<>|]', '_', course_name)
        file_suffix = f"{safe_course_name}_ALL_CHAPTERS"
    elif not course_name and chapter_name:
        scope_desc = f"章节【{chapter_name}】"
        safe_chapter_name = re.sub(r'[\\/:*?"<>|]', '_', chapter_name)
        file_suffix = f"ALL_COURSES_{safe_chapter_name}"
    else:
        scope_desc = f"课程【{course_name}】的章节【{chapter_name}】"
        safe_course = re.sub(r'[\\/:*?"<>|]', '_', course_name)
        safe_chapter = re.sub(r'[\\/:*?"<>|]', '_', chapter_name)
        file_suffix = f"{safe_course}_{safe_chapter}"

    logger.info(f"开始导出CSV: {scope_desc}")

    os.makedirs(csv_output_dir, exist_ok=True)

    nodes_csv_path = os.path.join(csv_output_dir, f"nodes_{file_suffix}.csv")
    edges_csv_path = os.path.join(csv_output_dir, f"edges_{file_suffix}.csv")

    kg = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    try:
        result = kg.export_to_csv(course_name, chapter_name, nodes_csv_path, edges_csv_path)

        if result['success']:
            logger.info(f"✅ CSV导出成功: 节点={result['nodes_count']}, 边={result['edges_count']}")
        else:
            logger.error(f"❌ CSV导出失败: {result.get('error', '未知错误')}")

        return result
    finally:
        kg.close()


@log_execution_time(logger)
def save_to_neo4j(extracted_data, document_name: str, course_name: str, chapter_name: str,
                  neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                  chapter_summary: str = "", chapter_keywords: List[str] = None):
    """保存数据到Neo4j的主函数"""
    logger.info(f"开始保存到Neo4j: 课程={course_name}, 文档={document_name}, 章节={chapter_name}")

    kg = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)

    try:
        result = kg.save_extracted_data(extracted_data, document_name, course_name, chapter_name,
                                        chapter_summary, chapter_keywords)
        stats = kg.get_statistics()
        logger.info(f"知识图谱统计: 实体={stats['total_nodes']}, 关系={stats['total_relationships']}")
        return result
    finally:
        kg.close()


# ==================== ★ 新增：前端 JSON 导出工具函数 ====================

@log_execution_time(logger)
def export_graph_to_frontend_json(
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        course_names: Optional[List[str]] = None,
        output_path: Optional[str] = None
) -> Dict:
    """
    导出知识图谱为前端图可视化 JSON（支持多课程）

    输出结构示例：
    {
        "课程A": {
            "nodes": [{"id": "course", "name": "...", "type": "course", "symbolSize": 80}, ...],
            "links": [{"source": "course", "target": "chapter1"}, ...]
        },
        ...
    }
    Args:
        neo4j_uri / user / password: Neo4j 连接信息
        course_names: 要导出的课程列表；None 表示导出全部课程
        output_path: 可选，若提供则把结果写入该路径 JSON 文件

    Returns:
        Dict: {课程名: {nodes: [...], links: [...]}}
    """
    logger.info("=" * 60)
    logger.info("开始导出前端 JSON 格式")
    logger.info(f"课程: {course_names if course_names else '全部'}")
    logger.info(f"输出路径: {output_path if output_path else '仅返回对象'}")
    logger.info("=" * 60)

    kg = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    try:
        data = kg.export_to_frontend_json(course_names)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 前端 JSON 已写入: {output_path}")

        # 简要统计
        total_nodes = sum(len(v.get("nodes", [])) for v in data.values())
        total_links = sum(len(v.get("links", [])) for v in data.values())
        logger.info(f"导出完成: 课程数={len(data)}, 总节点={total_nodes}, 总连线={total_links}")

        return data
    finally:
        kg.close()


@log_execution_time(logger)
def output_to_neo4j(
        output_dir: str,
        deepseek_api_key: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        course_name: str = "专业课程",
        chapter_name: str = "第1章",
        show_examples: bool = True,
        enable_redundancy: bool = True,
        num_passes: int = 2,
        chapter_summaries_path: str = None
) -> Dict:
    """从output目录提取实体关系并保存到Neo4j的主函数"""

    document_name = chapter_name

    logger.info("=" * 60)
    logger.info("开始知识图谱构建任务")
    logger.info(f"课程: {course_name}")
    logger.info(f"章节: {chapter_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Neo4j URI: {neo4j_uri}")
    if enable_redundancy:
        logger.info(f"重复抽取: 启用 ({num_passes}次)")
    logger.info("=" * 60)

    result = {
        'success': False,
        'entities_extracted': 0,
        'relationships_extracted': 0,
        'entities_saved': 0,
        'relationships_saved': 0,
        'chapter_summary_generated': False,
        'chapter_relations_extracted': 0,
        'error': None
    }

    kg = None

    try:
        logger.info("步骤0: 加载多模态数据")
        extractor = EntityExtractor(deepseek_api_key, course_name)
        multimodal_data = extractor.load_multimodal_data(output_dir)

        logger.info("步骤1: 生成章节摘要")
        chapter_summary_data = extractor.generate_chapter_summary(
            multimodal_data['slides'], document_name, chapter_name
        )
        chapter_summary = chapter_summary_data.get('summary', '')
        chapter_keywords = chapter_summary_data.get('keywords', [])

        if chapter_summary:
            result['chapter_summary_generated'] = True
            logger.info(f"章节摘要生成成功: {len(chapter_summary)}字, 关键词={len(chapter_keywords)}个")
        else:
            logger.warning("章节摘要生成失败或为空")

        if enable_redundancy:
            logger.info(f"步骤2: 实体关系抽取（重复{num_passes}次）")
            all_entities = []
            all_relationships = []

            for pass_idx in range(num_passes):
                logger.info("=" * 40)
                logger.info(f"第 {pass_idx + 1}/{num_passes} 轮抽取")
                logger.info("=" * 40)

                extracted_data_pass = extractor.extract_entities_from_multimodal(multimodal_data, document_name)

                logger.info(f"第 {pass_idx + 1} 轮结果: 实体={len(extracted_data_pass.entities)}个, "
                            f"关系={len(extracted_data_pass.relationships)}个")

                all_entities.extend(extracted_data_pass.entities)
                all_relationships.extend(extracted_data_pass.relationships)

                if pass_idx < num_passes - 1:
                    import random
                    delay = random.uniform(3, 5)
                    logger.info(f"轮次间等待 {delay:.1f}秒")
                    time.sleep(delay)

            logger.info("合并去重...")
            unique_entities = extractor._deduplicate_entities(all_entities)
            unique_relationships = extractor._deduplicate_relationships(all_relationships)

            extracted_data = ExtractedTriple(
                entities=unique_entities,
                relationships=unique_relationships,
                attributes=[]
            )
        else:
            logger.info("步骤2: 实体关系抽取（单次）")
            extracted_data = extractor.extract_entities_from_multimodal(multimodal_data, document_name)

        result['entities_extracted'] = len(extracted_data.entities)
        result['relationships_extracted'] = len(extracted_data.relationships)

        logger.info(
            f"抽取完成: 实体={result['entities_extracted']}, "
            f"关系={result['relationships_extracted']}"
        )

        if chapter_summaries_path and os.path.exists(chapter_summaries_path):
            logger.info("步骤3: 抽取章节关系")
            chapter_summaries = load_chapter_summaries(chapter_summaries_path)

            if chapter_summaries:
                chapter_rels = extractor.extract_chapter_relationships(chapter_summaries)
                extracted_data.relationships.extend(chapter_rels)
                result['chapter_relations_extracted'] = len(chapter_rels)
                logger.info(f"章节关系抽取完成: {len(chapter_rels)}个")
            else:
                logger.warning("章节摘要列表为空，跳过章节关系抽取")
        else:
            logger.info("步骤3: 跳过章节关系抽取（未提供章节摘要文件）")

        if result['entities_extracted'] == 0 and result['relationships_extracted'] == 0:
            logger.warning("未抽取到任何实体和关系")
            return result

        logger.info("步骤4: 连接Neo4j数据库")
        kg = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)

        logger.info("步骤5: 保存知识图谱到Neo4j")
        save_result = kg.save_extracted_data(
            extracted_data, document_name, course_name, chapter_name,
            chapter_summary, chapter_keywords
        )

        result['entities_saved'] = save_result['entities_saved']
        result['relationships_saved'] = save_result['relationships_saved']
        result['success'] = save_result['success']

        if not save_result['success']:
            result['error'] = save_result.get('error', '保存失败')
            return result

        logger.info("步骤6: 获取知识图谱统计信息")
        stats = kg.get_statistics()

        if show_examples:
            logger.info("步骤7: 显示知识图谱查询示例")

            entities = kg.query_entities(5)
            if entities:
                logger.info("实体示例 (前5个):")
                for i, entity in enumerate(entities):
                    layer_info = f" [Layer {entity.get('layer', 'N/A')}]" if entity.get('layer') else ""
                    source_info = f" [来源: {entity.get('source_file', 'N/A')}]" if entity.get('source_file') else ""
                    summary_info = ""
                    if entity.get('summary'):
                        summary_text = entity.get('summary', '')[:50]
                        summary_info = f" [摘要: {summary_text}...]"
                    logger.info(
                        f"  {i + 1}. {entity['name']} ({entity['type']}){layer_info}{source_info}{summary_info}")

            relationships = kg.query_relationships(10)
            if relationships:
                logger.info("关系示例 (前10个):")
                for i, rel in enumerate(relationships):
                    desc_info = f" - {rel.get('description', '')}" if rel.get('description') else ""
                    long_desc = rel.get('long_description', '')
                    long_desc_info = ""
                    if long_desc:
                        long_desc_info = f"\n     详细: {long_desc[:100]}..."
                    logger.info(
                        f"  {i + 1}. {rel['source']} -[{rel['relation']}]-> {rel['target']}{desc_info}{long_desc_info}")

        logger.info("=" * 60)
        logger.info("知识图谱构建完成!")
        logger.info(f"课程结构: {course_name} -> {chapter_name}")
        logger.info(f"实体抽取: {result['entities_extracted']}个")
        logger.info(f"关系抽取: {result['relationships_extracted']}个")
        logger.info(f"实体保存: {result['entities_saved']}/{result['entities_extracted']}")
        logger.info(f"关系保存: {result['relationships_saved']}/{result['relationships_extracted']}")
        if result['chapter_summary_generated']:
            logger.info(f"章节摘要: 已生成 ({len(chapter_summary)}字)")
        if result['chapter_relations_extracted'] > 0:
            logger.info(f"章节关系: {result['chapter_relations_extracted']}个")
        logger.info("=" * 60)

        result['success'] = True

    except Exception as e:
        logger.error(f"知识图谱构建失败: {str(e)}", exc_info=True)
        result['error'] = str(e)
        result['success'] = False

    finally:
        if kg:
            kg.close()

    return result


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    logger.info("程序启动")

    # Neo4j连接配置
    NEO4J_URI = "neo4j+s://8801ee35.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "WfWRJmI8o2kvWLSfDWLpHgI8Gsx49BL00beDkqhmzvk"
    DEEPSEEK_API_KEY = "sk-c28ec338b39e4552b9e6bded47466442"

    COURSE = "机电系统原理与实践"
    CHAPTER = "第十三章 机电系统在动力学与控制中的应用"

    # ========== 可选：清空数据库 ==========
    # reset_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # ========== 【示例 1】原有主流程：构建知识图谱 ==========
    # 如果只想跑 CRUD 和前端 JSON 导出的演示，把下面这段注释掉即可
    RUN_PIPELINE = False  # 改成 True 即可跑完整抽取流水线

    if RUN_PIPELINE:
        logger.info("\n" + "█" * 60)
        logger.info("【示例 1】执行知识图谱构建流水线")
        logger.info("█" * 60)

        result = output_to_neo4j(
            output_dir=r"C:\Users\Lin\PycharmProjects\PythonProject\output",
            deepseek_api_key=DEEPSEEK_API_KEY,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            course_name=COURSE,
            chapter_name=CHAPTER,
            show_examples=True,
            enable_redundancy=True,
            num_passes=2,
            chapter_summaries_path=None
        )

        if result['success']:
            logger.info("✅ 知识图谱构建成功!")
            logger.info(f"实体: {result['entities_saved']}/{result['entities_extracted']}")
            logger.info(f"关系: {result['relationships_saved']}/{result['relationships_extracted']}")
        else:
            logger.error(f"❌ 构建失败: {result.get('error', '未知错误')}")

    # ========== 【示例 2】6 种 CRUD 操作演示 ==========
    logger.info("\n" + "█" * 60)
    logger.info("【示例 2】6 种 CRUD 操作演示")
    logger.info("█" * 60)

    kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # 确保课程/章节存在（CRUD 演示的前置条件）
        kg.ensure_course_chapter(
            COURSE, CHAPTER,
            chapter_summary="本章讲解机电系统在动力学与控制中的典型应用，包括PID、Bang-Bang、模糊控制等策略。",
            chapter_keywords=["PID控制", "Bang-Bang控制", "模糊控制", "闭环控制"]
        )

        # ---------- ① add_entity：新增实体 ----------
        logger.info("\n--- [1/6] add_entity 新增实体 ---")
        kg.add_entity(name="模糊控制", entity_type="核心概念",
                      course_name=COURSE, chapter_name=CHAPTER,
                      description="一种基于模糊逻辑的非线性控制方法")
        kg.add_entity(name="PID控制", entity_type="核心概念",
                      course_name=COURSE, chapter_name=CHAPTER,
                      description="比例-积分-微分反馈控制策略")
        kg.add_entity(name="Bang-Bang控制", entity_type="核心概念",
                      course_name=COURSE, chapter_name=CHAPTER,
                      description="开关式（on-off）控制策略")
        logger.info(f"当前 PID控制 详情: {kg.get_entity_detail('PID控制')}")

        # ---------- ② add_relationship：新增关系 ----------
        logger.info("\n--- [2/6] add_relationship 新增关系 ---")
        kg.add_relationship(source_name="PID控制", target_name="模糊控制",
                            relation_type="应用于",
                            description="PID 常作为模糊控制的基线对照方法")
        kg.add_relationship(source_name="Bang-Bang控制", target_name="PID控制",
                            relation_type="相关于",
                            description="同属经典控制策略")

        # ---------- ③ update_entity：修改实体 ----------
        logger.info("\n--- [3/6] update_entity 修改实体 ---")
        kg.update_entity("PID控制", {
            "description": "经典闭环控制策略，在工业界应用极为广泛"
        })
        logger.info(f"更新后 PID控制 详情: {kg.get_entity_detail('PID控制')}")

        # ---------- ④ update_relationship：修改关系 ----------
        logger.info("\n--- [4/6] update_relationship 修改关系 ---")
        kg.update_relationship("PID控制", "模糊控制", "应用于", {
            "description": "在复杂非线性系统中常组合使用以提升鲁棒性",
            "long_description": "PID 提供基线线性控制效果，模糊控制在其基础上进行非线性补偿，两者结合可兼顾稳态精度与动态响应。"
        })
        logger.info(f"更新后的关系: {kg.get_relationship_detail('PID控制', '模糊控制', '应用于')}")

        # ---------- ⑤ delete_relationship：删除关系 ----------
        logger.info("\n--- [5/6] delete_relationship 删除关系 ---")
        del_rel_res = kg.delete_relationship("Bang-Bang控制", "PID控制", "相关于")
        logger.info(f"删除关系结果: {del_rel_res}")

        # ---------- ⑥ delete_entity：删除实体（带预览） ----------
        logger.info("\n--- [6/6] delete_entity 删除实体 ---")
        preview = kg.preview_delete_entity("Bang-Bang控制")
        logger.info(f"删除预览: 将影响 {preview['relationship_count']} 条关系")
        for r in preview.get("relationships", []):
            logger.info(f"  - [{r.get('direction')}] {r.get('rel')} -> {r.get('target')}")
        del_ent_res = kg.delete_entity("Bang-Bang控制")
        logger.info(f"删除实体结果: {del_ent_res}")

        logger.info("\n✅ 6 种 CRUD 操作全部演示完毕")

    except Exception as e:
        logger.error(f"CRUD 演示失败: {str(e)}", exc_info=True)
    finally:
        kg.close()

    # ========== 【示例 3】导出前端 JSON 格式 ==========

    logger.info("\n" + "█" * 60)
    logger.info("【示例 3】导出前端 JSON 格式（graphs.json 样式）")
    logger.info("█" * 60)

    # 3.1 导出单个课程

    logger.info("\n--- [3.1] 导出单个课程 ---")
    single_course_data = export_graph_to_frontend_json(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        course_names=[COURSE],
        output_path="exports/frontend_single_course.json"
    )
    for cname, cdata in single_course_data.items():
        logger.info(f"  课程【{cname}】: 节点={len(cdata['nodes'])}, 连线={len(cdata['links'])}")

    # 3.2 导出全部课程
    logger.info("\n--- [3.2] 导出全部课程 ---")
    all_courses_data = export_graph_to_frontend_json(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        course_names=None,  # None 表示全部
        output_path="exports/frontend_all_courses.json"
    )
    logger.info(f"全部课程共导出 {len(all_courses_data)} 个:")
    for cname, cdata in all_courses_data.items():
        logger.info(f"  - {cname}: 节点={len(cdata['nodes'])}, 连线={len(cdata['links'])}")

    # 3.3 打印出其中一个课程的 JSON 片段（前几个节点）作为示意
    if single_course_data:
        first_course = list(single_course_data.keys())[0]
        preview_data = {
            first_course: {
                "nodes": single_course_data[first_course]["nodes"][:3],
                "links": single_course_data[first_course]["links"][:3]
            }
        }
        logger.info("\n前端 JSON 片段预览:")
        logger.info(json.dumps(preview_data, ensure_ascii=False, indent=2))

    # ========== 【示例 4】CSV 导出（保留原有功能） ==========
    logger.info("\n" + "█" * 60)
    logger.info("【示例 4】CSV 导出（原有功能）")
    logger.info("█" * 60)

    csv_result = export_chapter_to_csv(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        course_name=COURSE,
        chapter_name="",  # 空字符串表示该课程全部章节
        csv_output_dir="csv_exports"
    )
    if csv_result.get('success'):
        logger.info(f"✅ CSV 导出成功: 节点={csv_result['nodes_count']}, 边={csv_result['edges_count']}")

    # ========== 【示例 5】从文件抽取知识图谱（抽取能力测试） ==========

    logger.info("\n" + "█" * 60)
    logger.info("【示例 5】从文件抽取知识图谱（抽取能力测试）")
    logger.info("█" * 60)

    # ---------------------------------------------------------------
    # 📁 在这里填写你的测试文件目录
    #    目录结构要求（与原有 output_to_neo4j 一致）：
    #    TEST_OUTPUT_DIR/
    #      └── text/
    #            ├── slide_01.json   ← 每个 json 里有 "text" 或 "content" 字段
    #            ├── slide_02.json
    #            └── ...
    #      └── images/              ← 可选，没有图片就留空目录
    # ---------------------------------------------------------------
    TEST_OUTPUT_DIR = r"C:\Users\Lin\PycharmProjects\PythonProject\output"  # ← 改成你的测试目录
    TEST_COURSE     = "机电系统原理与实践"
    TEST_CHAPTER    = "第十三章 机电系统在动力学与控制中的应用"

    # 是否在抽取前先清空数据库（建议测试时开启，避免和示例2的数据混在一起）
    CLEAN_BEFORE_EXTRACT = True
    if CLEAN_BEFORE_EXTRACT:
        logger.info("🧹 抽取前清空数据库...")
        reset_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        logger.info("✅ 数据库已清空")

    logger.info(f"📂 测试目录: {TEST_OUTPUT_DIR}")
    logger.info(f"📚 课程: {TEST_COURSE}")
    logger.info(f"📖 章节: {TEST_CHAPTER}")

    extract_result = output_to_neo4j(
        output_dir=TEST_OUTPUT_DIR,
        deepseek_api_key=DEEPSEEK_API_KEY,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        course_name=TEST_COURSE,
        chapter_name=TEST_CHAPTER,
        show_examples=True,       # 在日志里打印抽取到的实体/关系示例
        enable_redundancy=True,   # 重复抽取2次，提升召回率
        num_passes=2,
        chapter_summaries_path=None
    )

    # ---------- 结果汇报 ----------
    logger.info("\n" + "=" * 60)
    if extract_result['success']:
        logger.info("✅ 抽取测试完成！结果如下：")
        logger.info(f"  章节摘要生成: {'✅ 成功' if extract_result['chapter_summary_generated'] else '❌ 失败'}")
        logger.info(f"  实体抽取数量: {extract_result['entities_extracted']} 个")
        logger.info(f"  关系抽取数量: {extract_result['relationships_extracted']} 个")
        logger.info(f"  实体写入Neo4j: {extract_result['entities_saved']} 个")
        logger.info(f"  关系写入Neo4j: {extract_result['relationships_saved']} 个")
        if extract_result.get('chapter_relations_extracted', 0) > 0:
            logger.info(f"  章节间关系: {extract_result['chapter_relations_extracted']} 个")

        # 抽取完后，顺手导出一份 JSON 方便肉眼检查
        logger.info("\n📤 自动导出抽取结果到 JSON 文件...")
        export_graph_to_frontend_json(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            course_names=[TEST_COURSE],
            output_path="exports/extract_test_result.json"
        )
        logger.info("✅ 已导出到 exports/extract_test_result.json，可直接打开查看")

    else:
        logger.error(f"❌ 抽取测试失败: {extract_result.get('error', '未知错误')}")

    logger.info("=" * 60)


    logger.info("\n" + "█" * 60)
    logger.info("程序结束")
    logger.info("█" * 60)
