import json
import re
import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
from PIL import Image
import spacy
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, pipeline
import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading



@dataclass
class ExtractedTriple:
    entities: List[Dict]
    relationships: List[Dict]
    attributes: List[Dict]

class ContextBuffer:
    """上下文缓存，用于处理跨页实体关联"""

    def __init__(self, max_pages=3):
        self.buffer = {}  # {page_num: [entities]}
        self.max_pages = max_pages
        self.lock = threading.Lock()

    def add_entities(self, page_num: int, entities: List[Dict]):
        with self.lock:
            if page_num not in self.buffer:
                self.buffer[page_num] = []
            self.buffer[page_num].extend(entities)
            # 清理过期页面
            old_pages = [p for p in self.buffer if p < page_num - self.max_pages]
            for p in old_pages:
                del self.buffer[p]

    def get_recent_entities(self, current_page: int) -> List[Dict]:
        with self.lock:
            recent = []
            for page in range(max(1, current_page - self.max_pages + 1), current_page + 1):
                recent.extend(self.buffer.get(page, []))
            return recent

    def find_best_entity_for_attribute(self, attribute: Dict, current_page: int) -> Dict:
        """为属性找到最匹配的实体"""
        recent_entities = self.get_recent_entities(current_page)
        if not recent_entities:
            return None

        attr_text = attribute.get('evidence', attribute.get('name', ''))

        same_page_entities = [e for e in recent_entities if e.get('page') == current_page]
        if same_page_entities:
            for entity in same_page_entities:
                if entity['name'] in attr_text or attr_text in entity['name']:
                    return entity

        for entity in reversed(recent_entities):
            if entity['name'] in attr_text or attr_text in entity['name']:
                return entity

        return None


class DeepSeekClient:
    """DeepSeek API客户端，用于调用云端大模型"""

    def __init__(self, api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()  # 复用连接

    def chat_completions_create(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 1024) -> Dict:
        """调用DeepSeek API"""
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=300
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"DeepSeek API调用失败: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API请求异常: {str(e)}")


class EnhancedMultimodalExtractor:
    def __init__(self, use_clip=False, domain_config_path=None, use_deepseek_api=True, deepseek_model="deepseek-chat",
                 api_key=None):
        """ 初始化增强版多模态抽取器 """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_deepseek_api = use_deepseek_api

        # 加载DeepSeek云端API模型用于NER和属性抽取
        try:
            if use_deepseek_api and api_key:
                self.deepseek_client = DeepSeekClient(api_key=api_key, model=deepseek_model)
                print(f"使用DeepSeek云端API模型: {deepseek_model}")
                self.glm_model = None

                self.glm_tokenizer = None
            else:
                # 如果没有API Key，可以使用本地模型或其他方式
                model_name = "F:\\Models\\chatglm3-6b"
                self.glm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.glm_model = pipeline("text-generation", model=model_name, tokenizer=self.glm_tokenizer,
                                          device=0 if self.device == 'cuda' else -1)
                print("成功加载 ChatGLM3 模型")
        except Exception as e:
            print(f"加载大模型失败: {e}")
            self.glm_model = None
            self.glm_tokenizer = None
            self.deepseek_client = None

        # spaCy关系抽取（暂时保留，但不再使用）
        try:
            self.nlp_relation = spacy.load("zh_core_web_sm")
            print("成功加载zh_core_web_sm工具用于关系抽取")
        except:
            print("警告：无法加载spaCy中文模型用于关系抽取")
            self.nlp_relation = None

        # 多模态处理设置
        self.use_clip = use_clip
        if use_clip:
            self.clip_model = CLIPModel.from_pretrained("F:\\Models\\clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("F:\\Models\\clip-vit-base-patch32")

        # 加载领域配置
        self.domain_config = self._load_domain_config(domain_config_path) or {
            'term_terms': ["性能评估基准", "Benchmark", "标准化测试", "测试任务", "数据集", "模型性能", "准确率",
                           "推理速度", "计算效率"],
            'concept_terms': ["性能评估", "跨模型比较", "模型优化", "标准化"],
            'default_relations': {
                'definition': ["指的是", "指", "是", "means", "refers to"],
                'purpose': ["用来", "用于", "目的是", "used for", "purpose is"],
                'function': ["提供", "允许", "帮助", "指导", "provide", "allow", "help", "guide"],
                'characteristic': ["例如", "比如", "such as", "including"],
                'contains': ["包含", "包括", "含有"],
                'supports': ["支持", "允许", "帮助"],
                'ensures': ["确保", "保证"]
            }
        }

        # 缓存机制
        self.entity_cache = {}
        self.relation_patterns = self._compile_relation_patterns()

        # 上下文缓存
        self.context_buffer = ContextBuffer()

        # 结果缓存
        self.result_cache = {}
        self.cache_lock = threading.Lock()

    def _load_domain_config(self, path):
        """加载领域配置文件"""
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _compile_relation_patterns(self):
        """预编译关系匹配模式"""
        patterns = []
        for rel_type, keywords in self.domain_config.get('default_relations', {}).items():
            for kw in keywords:
                escaped_kw = re.escape(kw)
                pattern_str = rf'([^，。,.\n]{{1,30}}){escaped_kw}([^，。,.\n]{{1,50}})'
                patterns.append({
                    'type': rel_type,
                    'pattern': re.compile(pattern_str),
                    'keyword': kw
                })
        return patterns

    def _merge_attributes_into_entities(self, entities: List[Dict], attributes: List[Dict]) -> List[Dict]:
        """将属性合并到对应的实体中，保留所有实体"""
        # 复制所有实体，确保不修改原始数据
        merged_entities = [ent.copy() for ent in entities]

        # 创建 id 到实体的映射，方便快速查找
        id_to_entity = {ent['id']: ent for ent in merged_entities if ent.get('id')}

        # 将属性合并到对应实体
        for attr in attributes:
            entity_id = attr.get('entity_id')
            if entity_id and entity_id in id_to_entity:
                entity = id_to_entity[entity_id]
                attr_name = attr.get('name', 'attribute')
                attr_value = attr.get('value', '')
                # 以属性名作为键添加到实体中
                entity[f'attr_{attr_name}'] = attr_value
                # 也可以添加属性类型等信息
                entity[f'attr_{attr_name}_type'] = attr.get('type', '')

        # 返回所有实体（无论是否有id或属性）
        return merged_entities

    def process_multimodal_data(self, metadata_files: List[str], max_workers: int = 3) -> ExtractedTriple:
        """
        处理多模态数据入口方法 - 使用并发处理提升性能
        """
        all_entities = []
        all_relations = []
        all_attributes = []

        # 使用线程池并发处理多个文件
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in metadata_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        text_results, image_results = result

                        for entity in text_results.entities:
                            page = entity.get('page', 1)
                            self.context_buffer.add_entities(page, [entity])

                        attributed_attributes = self._assign_attributes_to_entities(
                            text_results.attributes,
                            text_results.entities,
                            []  # 这里需要实际的text_items
                        )

                        all_entities.extend(text_results.entities + image_results.entities)
                        all_relations.extend(text_results.relationships)
                        all_attributes.extend(attributed_attributes)

                        if self.use_clip:
                            cross_relations = self._link_cross_modal_entities(
                                text_results.entities,
                                image_results.entities
                            )
                            all_relations.extend(cross_relations)

                except Exception as e:
                    print(f"处理文件 {file_path} 出错: {str(e)}")
                    continue

        # 合并属性到实体中
        merged_entities = self._merge_attributes_into_entities(all_entities, all_attributes)

        # 不再进行去重操作，保留所有结果
        # final_entities = self._deduplicate_entities(merged_entities)
        # final_relations = self._filter_relations(all_relations)

        # 返回结果中不再包含attributes
        return ExtractedTriple(merged_entities, all_relations, [])

    def _process_single_file(self, file_path: str) -> Tuple:
        """处理单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            if isinstance(metadata, dict):
                metadata = [metadata]

            text_results = self._process_text_items(metadata)
            image_results = self._process_image_items([])

            return text_results, image_results
        except Exception as e:
            print(f"处理文件 {file_path} 出错: {str(e)}")
            return None, None

    def _assign_attributes_to_entities(self, attributes: List[Dict], entities: List[Dict], text_items: List[Dict]) -> \
    List[Dict]:
        """为属性分配实体归属"""
        attributed_attrs = []

        page_to_entities = defaultdict(list)
        for entity in entities:
            page = entity.get('page', 1)
            page_to_entities[page].append(entity)

        page_to_text = {}
        for item in text_items:
            page = item.get('page', 1)
            page_to_text[page] = item.get('raw_text', '')

        for attr in attributes:
            page = attr.get('page', 1)
            assigned_entity = None

            same_page_entities = page_to_entities.get(page, [])
            attr_text = attr.get('evidence', attr.get('name', ''))

            for entity in same_page_entities:
                if entity['name'] in attr_text:
                    assigned_entity = entity
                    break

            if not assigned_entity:
                assigned_entity = self.context_buffer.find_best_entity_for_attribute(attr, page)

            attr_copy = attr.copy()
            if assigned_entity:
                attr_copy['entity_id'] = assigned_entity['id']
                attr_copy['entity_name'] = assigned_entity['name']
            else:
                attr_copy['entity_id'] = None
                attr_copy['entity_name'] = '未分配'

            attributed_attrs.append(attr_copy)

        return attributed_attrs

    def _extract_json_from_response(self, response_text: str) -> str:
        """从模型响应中提取JSON内容"""
        # 首先尝试直接解析整个响应
        cleaned = self._clean_json_output(response_text)
        if cleaned.startswith('[') or cleaned.startswith('{'):
            return cleaned

        # 如果直接解析失败，尝试从代码块中提取
        lines = response_text.split('\n')
        json_lines = []
        in_json = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('```json'):
                in_json = True
                continue
            elif stripped.startswith('```'):
                in_json = False
                continue
            elif in_json:
                json_lines.append(line)

        if json_lines:
            return '\n'.join(json_lines).strip()

        # 如果还是没有找到，返回清理后的整个响应
        return cleaned

    def _safe_json_loads(self, json_text: str):
        """安全地解析JSON，包含错误处理"""
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"尝试解析的内容: {json_text[:200]}...")
            # 尝试修复常见的JSON问题
            fixed = json_text.strip()
            # 移除可能的前缀说明
            if fixed.startswith('当然可以') or fixed.startswith('好的') or fixed.startswith('以下是'):
                # 尝试找到第一个[或{的位置
                start_bracket = -1
                for i, char in enumerate(fixed):
                    if char in ['[', '{']:
                        start_bracket = i
                        break
                if start_bracket != -1:
                    fixed = fixed[start_bracket:]

            # 确保以[或{开始
            if not (fixed.startswith('[') or fixed.startswith('{')):
                # 尝试找到第一个[的位置
                bracket_pos = fixed.find('[')
                if bracket_pos != -1:
                    fixed = fixed[bracket_pos:]
                else:
                    # 尝试找到第一个{的位置
                    brace_pos = fixed.find('{')
                    if brace_pos != -1:
                        fixed = fixed[brace_pos:]

            # 确保以]或}结束
            if fixed.endswith(',') or fixed.endswith('，'):
                fixed = fixed[:-1]

            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                print(f"修复后仍无法解析JSON: {fixed[:200]}...")
                return None

    def _get_cache_key(self, text: str, task_type: str) -> str:
        """生成缓存键"""
        return f"{task_type}_{hash(text[:100])}"

    def _check_cache(self, cache_key: str):
        """检查缓存"""
        with self.cache_lock:
            return self.result_cache.get(cache_key)

    def _save_cache(self, cache_key: str, result):
        """保存到缓存"""
        with self.cache_lock:
            self.result_cache[cache_key] = result

    def _extract_entities_with_glm(self, text: str, source: Dict) -> List[Dict]:
        """使用大模型进行实体识别 - 增强版"""
        # 检查缓存
        cache_key = self._get_cache_key(text, "entities")
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result

        if not self.glm_model and not self.deepseek_client:
            return []

        # 改进的prompt
        prompt = f"""请仔细从以下文本中提取所有命名实体，并以JSON数组形式返回，每个实体包含'name'、'type'字段。
特别注意提取以下类型的实体：
- 技术术语：如"性能评估基准"、"Benchmark"等
- 性能指标：如"准确率"、"推理速度"、"计算效率"等
- 角色概念：如"研究者"、"开发者"、"跨模型比较"、"模型优化"等
- 业务概念：如"标准化测试"、"数据集"等

文本内容：
{text}

请严格按照以下JSON格式返回，不要添加其他说明：
[
  {{"name": "实体名称", "type": "实体类型"}},
  {{"name": "实体名称2", "type": "实体类型2"}}
]"""

        try:
            if self.use_deepseek_api and self.deepseek_client:
                response = self.deepseek_client.chat_completions_create(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500
                )
                output = response['choices'][0]['message']['content'].strip()
            else:
                response = self.glm_model(prompt, max_new_tokens=1024, do_sample=False)
                output = response[0]['generated_text'].replace(prompt, '').strip()

            # print(f"实体抽取模型输出: {output[:200]}...")  # 调试信息

            # 提取JSON内容
            json_content = self._extract_json_from_response(output)
            entities_list = self._safe_json_loads(json_content)

            if entities_list is None:
                print("无法解析实体抽取结果为JSON")
                # 回退到基于词典的方法
                return self._extract_term_entities(text, source)

            # 确保是列表格式
            if not isinstance(entities_list, list):
                if isinstance(entities_list, dict) and 'entities' in entities_list:
                    entities_list = entities_list['entities']
                else:
                    print("实体抽取结果不是列表格式")
                    return self._extract_term_entities(text, source)

            result = []
            for ent in entities_list:
                if isinstance(ent, dict) and 'name' in ent:
                    clean_word = str(ent['name']).strip()
                    entity_type = str(ent.get('type', '未知')).strip()
                    entity = self._create_entity(clean_word, entity_type, source)
                    if self._validate_entity(entity, text):
                        result.append(entity)

            # 保存到缓存
            self._save_cache(cache_key, result)
            return result
        except Exception as e:
            print(f"解析大模型实体抽取输出出错: {e}")
            # 回退到基于词典的方法
            return self._extract_term_entities(text, source)

    def _extract_relations_with_glm(self, text: str) -> List[Dict]:
        """使用大模型进行关系抽取 - 增强版"""
        # 检查缓存
        cache_key = self._get_cache_key(text, "relations")
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result

        if not self.glm_model and not self.deepseek_client:
            return []

        prompt = f"""
请从以下文本中抽取实体之间的关系，以JSON数组格式返回，每个关系包含以下字段：
- source: 关系的源实体
- target: 关系的目标实体  
- type: 关系类型（如定义、包含、支持、确保等）
- evidence: 支持该关系的原文句子

常见关系类型包括：
- definition: 定义关系（指的是、指、是）
- contains: 包含关系（包含、包括、含有）
- supports: 支持关系（支持、允许、帮助）
- ensures: 确保关系（确保、保证）
- purpose: 目的关系（用来、用于、目的是）

文本内容：
{text}

请严格按照以下JSON格式返回，不要添加其他说明：
[
  {{"source": "源实体", "target": "目标实体", "type": "关系类型", "evidence": "支持句子"}},
  {{"source": "源实体2", "target": "目标实体2", "type": "关系类型2", "evidence": "支持句子2"}}
]"""

        try:
            if self.use_deepseek_api and self.deepseek_client:
                response = self.deepseek_client.chat_completions_create(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                output = response['choices'][0]['message']['content'].strip()
            else:
                response = self.glm_model(prompt, max_new_tokens=1500, do_sample=False)
                output = response[0]['generated_text'].replace(prompt, '').strip()

            print(f"关系抽取模型输出: {output[:200]}...")  # 调试信息

            # 提取JSON内容
            json_content = self._extract_json_from_response(output)
            relations_list = self._safe_json_loads(json_content)

            if relations_list is None:
                print("无法解析关系抽取结果为JSON")
                return []

            # 确保是列表格式
            if not isinstance(relations_list, list):
                if isinstance(relations_list, dict) and 'relations' in relations_list:
                    relations_list = relations_list['relations']
                else:
                    print("关系抽取结果不是列表格式")
                    return []

            valid_relations = []
            for rel in relations_list:
                # 验证关系的基本字段
                if isinstance(rel, dict) and all(key in rel for key in ['source', 'target', 'type']):
                    valid_relations.append({
                        'source': str(rel['source']).strip(),
                        'target': str(rel['target']).strip(),
                        'type': str(rel['type']).strip(),
                        'evidence': str(rel.get('evidence', '')),
                        'confidence': 0.85  # 大模型抽取的置信度
                    })

            # 保存到缓存
            self._save_cache(cache_key, valid_relations)
            return valid_relations

        except Exception as e:
            print(f"大模型关系抽取失败: {e}")
            return []

    def _extract_attributes_with_glm(self, text: str, page: int = None) -> List[Dict]:
        """使用大模型进行属性抽取 - 增强版"""
        # 检查缓存
        cache_key = self._get_cache_key(text, "attributes")
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result

        if not self.glm_model and not self.deepseek_client:
            return []

        prompt = f"""
请从以下文本中抽取属性信息（如定义、特征、用途等），以JSON数组格式返回，每个属性包含以下字段：
- name: 属性名称
- value: 属性值
- type: 属性类型（如定义属性、特征属性、用途属性等）
- evidence: 支持该属性的原文句子

文本内容：
{text}

请严格按照以下JSON格式返回，不要添加其他说明：
[
  {{"name": "属性名称", "value": "属性值", "type": "属性类型", "evidence": "支持句子"}},
  {{"name": "属性名称2", "value": "属性值2", "type": "属性类型2", "evidence": "支持句子2"}}
]"""

        try:
            if self.use_deepseek_api and self.deepseek_client:
                response = self.deepseek_client.chat_completions_create(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500
                )
                output = response['choices'][0]['message']['content'].strip()
            else:
                response = self.glm_model(prompt, max_new_tokens=1500, do_sample=False)
                output = response[0]['generated_text'].replace(prompt, '').strip()

            print(f"属性抽取模型输出: {output[:200]}...")  # 调试信息

            # 提取JSON内容
            json_content = self._extract_json_from_response(output)
            attributes_list = self._safe_json_loads(json_content)

            if attributes_list is None:
                print("无法解析属性抽取结果为JSON")
                return []

            # 确保是列表格式
            if not isinstance(attributes_list, list):
                if isinstance(attributes_list, dict) and 'attributes' in attributes_list:
                    attributes_list = attributes_list['attributes']
                else:
                    print("属性抽取结果不是列表格式")
                    return []

            valid_attributes = []
            for attr in attributes_list:
                # 验证属性的基本字段
                if isinstance(attr, dict) and all(key in attr for key in ['name', 'value', 'type']):
                    valid_attributes.append({
                        'name': str(attr['name']).strip(),
                        'value': str(attr['value']).strip(),
                        'type': str(attr['type']).strip(),
                        'evidence': str(attr.get('evidence', '')),
                        'source': 'text',
                        'page': page
                    })

            # 保存到缓存
            self._save_cache(cache_key, valid_attributes)
            return valid_attributes

        except Exception as e:
            print(f"大模型属性抽取失败: {e}")
            return []

    def _clean_json_output(self, output: str) -> str:
        """清洗大模型输出的JSON内容"""
        # 去掉Markdown代码块标记
        if output.startswith("```json"):
            output = output[7:]
        if output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]

        return output.strip()

    def _process_text_items(self, text_items: List[Dict]) -> ExtractedTriple:
        """处理文本类型数据 - 并行处理提升性能"""
        entities = []
        relations = []
        attributes = []

        # 使用线程池并发处理不同类型的抽取任务
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            for item in text_items:
                raw_text = item.get('raw_text', '')

                # 并行提交三个抽取任务
                futures.append((
                    executor.submit(self._extract_entities_with_glm, raw_text, item),
                    executor.submit(self._extract_relations_with_glm, raw_text),
                    executor.submit(self._extract_attributes_with_glm, raw_text, item.get('page'))
                ))

            # 收集结果
            for entity_future, relation_future, attr_future in futures:
                try:
                    entities.extend(entity_future.result())
                except Exception as e:
                    print(f"实体抽取任务失败: {e}")

                try:
                    relations.extend(relation_future.result())
                except Exception as e:
                    print(f"关系抽取任务失败: {e}")

                try:
                    attributes.extend(attr_future.result())
                except Exception as e:
                    print(f"属性抽取任务失败: {e}")

        return ExtractedTriple(entities, relations, attributes)

    def _extract_term_entities(self, text: str, source: Dict) -> List[Dict]:
        """基于领域词典抽取术语实体 - 增强版"""
        entities = []
        processed_names = set()  # 避免重复

        # 处理术语
        term_terms = self.domain_config.get('term_terms', [])
        for term in term_terms:
            if term in text and term not in processed_names:
                entity = {
                    'id': f"ent_{len(self.entity_cache)}",
                    'name': term,
                    'type': '术语',
                    'source': 'text',
                    'page': source.get('page'),
                    'context': text[:100],
                    'confidence': 0.95
                }
                entities.append(entity)
                self.entity_cache[len(self.entity_cache)] = entity
                processed_names.add(term)

        # 处理概念
        concept_terms = self.domain_config.get('concept_terms', [])
        for concept in concept_terms:
            if concept in text and concept not in processed_names:
                entity = {
                    'id': f"ent_{len(self.entity_cache)}",
                    'name': concept,
                    'type': '概念',
                    'source': 'text',
                    'page': source.get('page'),
                    'context': text[:100],
                    'confidence': 0.9
                }
                entities.append(entity)
                self.entity_cache[len(self.entity_cache)] = entity
                processed_names.add(concept)

        return entities

    def _process_image_items(self, image_items: List[Dict]) -> ExtractedTriple:
        """处理图像类型数据"""
        if not self.use_clip:
            return ExtractedTriple([], [], [])

        entities = []
        for item in image_items:
            try:
                image = Image.open(item['path'])
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                    image_vector = features.cpu().numpy()[0]
                entity = {
                    'id': f"img_{item['id']}",
                    'name': f"图像:{item.get('caption', '')[:20]}",
                    'type': 'illustration',
                    'vector': image_vector.tolist(),
                    'source': 'image',
                    'page': item.get('page')
                }
                entities.append(entity)
            except Exception as e:
                print(f"处理图像 {item.get('path')} 出错: {str(e)}")
                continue
        return ExtractedTriple(entities, [], [])

    def _create_entity(self, text: str, label: str, source: Dict) -> Dict:
        """创建标准化实体结构"""
        entity_type = label
        # 更智能的类型映射
        type_mapping = {
            '准确率': '性能指标',
            '推理速度': '性能指标',
            '计算效率': '性能指标',
            'Benchmark': '技术术语',
            '性能评估基准': '技术术语',
            '标准化测试': '技术术语',
            '数据集': '技术术语',
            '研究者': '角色',
            '开发者': '角色',
            '跨模型比较': '概念',
            '模型优化': '概念',
            '标准化': '概念'
        }

        if text in type_mapping:
            entity_type = type_mapping[text]
        else:
            # 基于后缀的类型推断
            for domain, terms in self.domain_config.items():
                if domain.endswith('_terms') and text in terms:
                    entity_type = domain.replace('_terms', '')
                    break

        return {
            'id': f"ent_{len(self.entity_cache)}",
            'name': text,
            'type': entity_type,
            'source': 'text',
            'page': source.get('page'),
            'context': source.get('raw_text', '')[:100],
            'confidence': 0.9
        }

    def _validate_entity(self, entity: Dict, context: str) -> bool:
        """实体验证"""
        if len(entity['name']) < 2:
            return False
        '''
        if self.use_deepseek_api:
            return True  # 使用DeepSeek API时跳过本地验证
        domain_terms = self.domain_config.get(f"{entity['type']}_terms", [])
        if entity['name'] in domain_terms:
            return True
        if self.use_clip:
            return self._clip_validate(entity, context)
        '''
        return True

    def _clip_validate(self, entity: Dict, context: str) -> bool:
        """使用CLIP验证实体语义"""
        if not self.use_clip:
            return True
        try:
            prompts = [
                f"'{entity['name']}'是有效的{entity['type']}",
                f"'{entity['name']}'是随机字符"
            ]
            inputs = self.clip_processor(text=[context] + prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self.clip_model.get_text_features(** inputs)
                sim = torch.cosine_similarity(features[0:1], features[1:])
            return sim[0].item() > 0.7
        except:
            return True  # 出错时默认接受

    def _link_cross_modal_entities(self, text_ents: List[Dict], image_ents: List[Dict]) -> List[Dict]:
        """建立跨模态实体关联"""
        relations = []
        for text_ent in text_ents:
            for img_ent in image_ents:
                if text_ent['page'] == img_ent.get('page'):
                    relations.append({
                        'source': text_ent['id'],
                        'target': img_ent['id'],
                        'type': '同页关联',
                        'confidence': 0.8
                    })
                    if self.use_clip:
                        try:
                            sim = self._calculate_semantic_similarity(
                                text_ent['name'],
                                img_ent['name']
                            )
                            if sim > 0.6:
                                relations.append({
                                    'source': text_ent['id'],
                                    'target': img_ent['id'],
                                    'type': '语义关联',
                                    'confidence': sim
                                })
                        except:
                            pass
        return relations

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算文本间语义相似度"""
        if not self.use_clip:
            return 0.7
        try:
            inputs = self.clip_processor(
                text=[text1, text2],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_text_features(**inputs)
                sim = torch.cosine_similarity(features[0:1], features[1:2]).item()
            return max(0, min(1, sim))
        except:
            return 0.7

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重与合并"""
        unique_ents = {}
        for ent in entities:
            key = (ent['name'], ent['type'])
            if key not in unique_ents or ent['confidence'] > unique_ents[key]['confidence']:
                unique_ents[key] = ent
        return list(unique_ents.values())

    def _filter_relations(self, relations: List[Dict]) -> List[Dict]:
        """关系过滤与去重"""
        seen = set()
        filtered = []
        for rel in relations:
            key = (rel['source'], rel['target'], rel['type'])
            if key not in seen and rel.get('confidence', 0) > 0.5:
                filtered.append(rel)
                seen.add(key)
        return filtered

    def extract_keywords_with_deepseek(self, question: str) -> List[str]:
        """
        使用DeepSeek模型从输入句子中提取关键词

        Args:
            question (str): 输入的句子或问题

        Returns:
            List[str]: 提取出的关键词列表
        """
        try:
            # 使用类中已有的DeepSeek客户端
            if not hasattr(self, 'deepseek_client') or not self.deepseek_client:
                raise Exception("DeepSeek客户端未初始化")

            # 构建提示词
            prompt = f"""
            请从以下句子中提取出最重要的关键词，只返回关键词，不要其他内容。
            句子：{question}

            要求：
            1. 提取2-5个最重要的关键词
            2. 关键词应该是名词、专有名词或核心概念
            3. 用逗号分隔返回结果
            4. 只返回关键词，不要解释

            示例：
            句子：爱因斯坦的相对论对现代物理学有什么影响？
            返回：爱因斯坦,相对论,物理学,影响
            """

            # 调用DeepSeek API
            response = self.deepseek_client.chat_completions_create(
                messages=[
                    {"role": "system", "content": "你是一个专业的关键词提取助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )

            # 解析响应
            if response and 'choices' in response and len(response['choices']) > 0:
                keywords_text = response['choices'][0]['message']['content'].strip()
                # 分割关键词并清理
                keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
                return keywords
            else:
                print("DeepSeek API返回空响应")
                return []

        except Exception as e:
            print(f"使用DeepSeek提取关键词时出错: {e}")
            return []

    def _map_entity_type(self, spacy_label: str) -> str:
        """映射spaCy标签到自定义类型"""
        mapping = {
            'PERSON': '人物',
            'ORG': '机构',
            'GPE': '地点',
            'LOC': '地点',
            'PER': '人物',
            'ORGANIZATION': '机构',
            'LOCATION': '地点'
        }
        return mapping.get(spacy_label, '术语')

    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.result_cache.clear()


    # 将抽取结果转换为DataFrame格式
def convert_extracted_to_dataframes(entities, relations):
        """将抽取结果转换为DataFrame格式"""

        # 节点数据处理
        nodes_data = []
        for entity in entities:
            node = {
                'id': entity.get('id', ''),
                'name': entity.get('name', ''),
                'type': entity.get('type', 'Concept')
                # 可以根据需要添加其他属性
            }
            nodes_data.append(node)

        # 关系数据处理
        rels_data = []
        for rel in relations:
            relationship = {
                'source': rel.get('source', ''),
                'target': rel.get('target', ''),
                'type': rel.get('type', 'RELATED')
                # 可以根据需要添加其他属性
            }
            rels_data.append(relationship)

        # 转换为DataFrame
        nodes_df = pd.DataFrame(nodes_data)
        rels_df = pd.DataFrame(rels_data)

        return nodes_df, rels_df





# 使用示例
if __name__ == "__main__":
    # 设置API Key（可以通过环境变量或直接输入）
    import sys
    import io
    import os

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 获取DeepSeek API Key
    api_key = os.getenv("sk-c28ec338b39e4552b9e6bded47466442")
    if not api_key:
        api_key = input("请输入您的DeepSeek API Key: ")

    extractor = EnhancedMultimodalExtractor(
        use_clip=True,
        domain_config_path=r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\domain_config.json",
        use_deepseek_api=True,  # 启用DeepSeek云端API
        deepseek_model="deepseek-chat",  # 使用DeepSeek的chat模型
        api_key=api_key  # 传入API Key
    )

    test_path = r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\test_file.json"

    if os.path.exists(test_path):
        print("文件存在！！")
        result = extractor.process_multimodal_data([test_path])
    else:
        print(f" 文件不存在: {test_path}")

    # print(result)

    print("抽取的实体:")
    for ent in result.entities:
        print(f"- {ent['name']} ({ent['type']})")

    print("\n抽取的关系:")
    for rel in result.relationships:
        print(f"- {rel['source']} → {rel['target']}: {rel['type']}")

    print("\n抽取的属性:")
    for attr in result.attributes:
        print(f"- {attr['name']} (属于: {attr.get('entity_name', '未分配')})")

