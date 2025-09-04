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
from zhipuai import ZhipuAI  # 如果使用API方式调用智谱AI的GLM模型
import pandas as pd

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

    def add_entities(self, page_num: int, entities: List[Dict]):
        if page_num not in self.buffer:
            self.buffer[page_num] = []
        self.buffer[page_num].extend(entities)
        # 清理过期页面
        old_pages = [p for p in self.buffer if p < page_num - self.max_pages]
        for p in old_pages:
            del self.buffer[p]

    def get_recent_entities(self, current_page: int) -> List[Dict]:
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


class EnhancedMultimodalExtractor:
    def __init__(self, use_clip=False, domain_config_path=None, use_zhipuai=False):
        """ 初始化增强版多模态抽取器 """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_zhipuai = use_zhipuai

        # 加载GLM模型用于NER和属性抽取
        try:
            if not use_zhipuai:
                model_name = "F:\\Models\\chatglm3-6b"
                self.glm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.glm_model = pipeline("text-generation", model=model_name, tokenizer=self.glm_tokenizer,
                                          device=0 if self.device == 'cuda' else -1)
                print("成功加载 ChatGLM3 模型")
            else:
                api_key = os.getenv("ZHIPUAI_API_KEY")  # 需要环境变量设置API Key
                self.zhipu_client = ZhipuAI(api_key=api_key)
                print("使用智谱AI在线API模型")
                self.glm_model = None
                self.glm_tokenizer = None
        except Exception as e:
            print(f"加载 GLM 模型失败: {e}")
            self.glm_model = None
            self.glm_tokenizer = None

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
            'term_terms': ["性能评估基准", "Benchmark", "标准化测试", "测试任务", "数据集", "模型性能", "准确率"],
            'concept_terms': ["性能评估", "跨模型比较", "模型优化"],
            'default_relations': {
                'definition': ["指的是", "指", "是", "means", "refers to"],
                'purpose': ["用来", "用于", "目的是", "used for", "purpose is"],
                'function': ["提供", "允许", "帮助", "指导", "provide", "allow", "help", "guide"],
                'characteristic': ["例如", "比如", "such as", "including"]
            }
        }

        # 缓存机制
        self.entity_cache = {}
        self.relation_patterns = self._compile_relation_patterns()

        # 上下文缓存
        self.context_buffer = ContextBuffer()

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
        """将属性合并到对应的实体中"""
        # 创建实体映射字典，便于快速查找
        entity_map = {}
        for entity in entities:
            entity_id = entity.get('id')
            if entity_id:
                entity_map[entity_id] = entity.copy()  # 复制避免修改原数据

        # 将属性合并到对应实体
        for attr in attributes:
            entity_id = attr.get('entity_id')
            if entity_id and entity_id in entity_map:
                # 获取实体
                entity = entity_map[entity_id]
                # 将属性添加到实体中
                attr_name = attr.get('name', 'attribute')
                attr_value = attr.get('value', '')
                # 以属性名作为键添加到实体中
                entity[f'attr_{attr_name}'] = attr_value
                # 也可以添加属性类型等信息
                entity[f'attr_{attr_name}_type'] = attr.get('type', '')

        # 返回合并后的实体列表
        return list(entity_map.values())

    def process_multimodal_data(self, metadata_files: List[str]) -> ExtractedTriple:
        """
        处理多模态数据入口方法
        """
        all_entities = []
        all_relations = []
        all_attributes = []

        for file_path in metadata_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if isinstance(metadata, dict):
                    metadata = [metadata]

                text_results = self._process_text_items(metadata)
                image_results = self._process_image_items([])

                for entity in text_results.entities:
                    page = entity.get('page', 1)
                    self.context_buffer.add_entities(page, [entity])

                attributed_attributes = self._assign_attributes_to_entities(
                    text_results.attributes,
                    text_results.entities,
                    metadata
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

        final_entities = self._deduplicate_entities(merged_entities)
        final_relations = self._filter_relations(all_relations)

        # 返回结果中不再包含attributes
        return ExtractedTriple(final_entities, final_relations, [])

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

    def _extract_entities_with_glm(self, text: str, source: Dict) -> List[Dict]:
        """使用 GLM 模型进行实体识别"""
        if not self.glm_model and not self.use_zhipuai:
            return []

        prompt = f"请从以下文本中提取所有命名实体，并以JSON数组形式返回，每个实体包含'name'、'type'字段：\n\n{text}\n\n输出示例格式：[{{\"name\": \"实体名\", \"type\": \"类型\"}}]"

        if self.use_zhipuai:
            response = self.zhipu_client.chat.completions.create(
                model="glm-3-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            output = response.choices[0].message.content.strip()
        else:
            response = self.glm_model(prompt, max_new_tokens=512, do_sample=False)
            output = response[0]['generated_text'].replace(prompt, '').strip()

        # 清洗输出内容
        output = self._clean_json_output(output)

        try:
            entities_list = json.loads(output)
            result = []
            for ent in entities_list:
                clean_word = ent['name'].strip()
                entity = self._create_entity(clean_word, ent['type'], source)
                if self._validate_entity(entity, text):
                    result.append(entity)
            return result
        except Exception as e:
            print(f"解析 GLM 实体抽取输出出错: {e}")
            return []

    def _extract_relations_with_glm(self, text: str) -> List[Dict]:
        """使用GLM模型进行关系抽取"""
        if not self.glm_model and not self.use_zhipuai:
            return []

        prompt = f"""
请从以下文本中抽取实体之间的关系，以JSON数组格式返回，每个关系包含以下字段：
- source: 关系的源实体
- target: 关系的目标实体  
- type: 关系类型
- evidence: 支持该关系的原文句子

文本内容：
{text}

输出格式示例：
[
  {{"source": "实体A", "target": "实体B", "type": "关系类型", "evidence": "原文句子"}}
]

请只输出JSON，不要添加其他说明：
"""

        try:
            if self.use_zhipuai:
                response = self.zhipu_client.chat.completions.create(
                    model="glm-3-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                output = response.choices[0].message.content.strip()
            else:
                response = self.glm_model(prompt, max_new_tokens=1024, do_sample=False)
                output = response[0]['generated_text'].replace(prompt, '').strip()

            # 清洗输出内容
            output = self._clean_json_output(output)

            relations_list = json.loads(output)
            valid_relations = []

            for rel in relations_list:
                # 验证关系的基本字段
                if all(key in rel for key in ['source', 'target', 'type']):
                    valid_relations.append({
                        'source': rel['source'].strip(),
                        'target': rel['target'].strip(),
                        'type': rel['type'].strip(),
                        'evidence': rel.get('evidence', ''),
                        'confidence': 0.85  # GLM抽取的置信度
                    })

            return valid_relations

        except Exception as e:
            print(f"GLM关系抽取失败: {e}")
            return []

    def _extract_attributes_with_glm(self, text: str, page: int = None) -> List[Dict]:
        """使用GLM模型进行属性抽取"""
        if not self.glm_model and not self.use_zhipuai:
            return []

        prompt = f"""
请从以下文本中抽取属性信息（如定义、特征、用途等），以JSON数组格式返回，每个属性包含以下字段：
- name: 属性名称
- value: 属性值
- type: 属性类型（如定义属性、特征属性等）
- evidence: 支持该属性的原文句子

文本内容：
{text}

输出格式示例：
[
  {{"name": "属性名", "value": "属性值", "type": "属性类型", "evidence": "原文句子"}}
]

请只输出JSON，不要添加其他说明：
"""

        try:
            if self.use_zhipuai:
                response = self.zhipu_client.chat.completions.create(
                    model="glm-3-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                output = response.choices[0].message.content.strip()
            else:
                response = self.glm_model(prompt, max_new_tokens=1024, do_sample=False)
                output = response[0]['generated_text'].replace(prompt, '').strip()

            # 清洗输出内容
            output = self._clean_json_output(output)

            attributes_list = json.loads(output)
            valid_attributes = []

            for attr in attributes_list:
                # 验证属性的基本字段
                if all(key in attr for key in ['name', 'value', 'type']):
                    valid_attributes.append({
                        'name': attr['name'].strip(),
                        'value': attr['value'].strip(),
                        'type': attr['type'].strip(),
                        'evidence': attr.get('evidence', ''),
                        'source': 'text',
                        'page': page
                    })

            return valid_attributes

        except Exception as e:
            print(f"GLM属性抽取失败: {e}")
            return []

    def _clean_json_output(self, output: str) -> str:
        """清洗GLM输出的JSON内容"""
        # 去掉Markdown代码块标记
        if output.startswith("```json"):
            output = output[7:]
        if output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]

        return output.strip()

    def _process_text_items(self, text_items: List[Dict]) -> ExtractedTriple:
        """处理文本类型数据"""
        entities = []
        relations = []
        attributes = []

        for item in text_items:
            raw_text = item.get('raw_text', '')

            # 使用 GLM 进行实体抽取
            glm_entities = self._extract_entities_with_glm(raw_text, item)
            entities.extend(glm_entities)

            # 使用 GLM 进行关系抽取
            glm_relations = self._extract_relations_with_glm(raw_text)
            relations.extend(glm_relations)

            # 使用 GLM 进行属性抽取
            glm_attributes = self._extract_attributes_with_glm(raw_text, item.get('page'))
            attributes.extend(glm_attributes)

        return ExtractedTriple(entities, relations, attributes)

    def _extract_term_entities(self, text: str, source: Dict) -> List[Dict]:
        """基于领域词典抽取术语实体"""
        entities = []
        term_terms = self.domain_config.get('term_terms', [])
        for term in term_terms:
            if term in text:
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

        concept_terms = self.domain_config.get('concept_terms', [])
        for concept in concept_terms:
            if concept in text:
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
        if self.use_zhipuai:
            return True  # 使用智谱API时跳过本地验证
        domain_terms = self.domain_config.get(f"{entity['type']}_terms", [])
        if entity['name'] in domain_terms:
            return True
        if self.use_clip:
            return self._clip_validate(entity, context)
        return True

    def _clip_validate(self, entity: Dict, context: str) -> bool:
        """使用CLIP验证实体语义"""
        prompts = [
            f"'{entity['name']}'是有效的{entity['type']}",
            f"'{entity['name']}'是随机字符"
        ]
        inputs = self.clip_processor(text=[context] + prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            sim = torch.cosine_similarity(features[0:1], features[1:])
        return sim[0].item() > 0.7

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
        return relations

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算文本间语义相似度"""
        inputs = self.clip_processor(
            text=[text1, text2],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            sim = torch.cosine_similarity(features[0:1], features[1:2]).item()
        return max(0, min(1, sim))

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


# 使用示例
if __name__ == "__main__":
    # 设置API Key（可选，也可以用环境变量）
    import sys
    import io

    os.environ["ZHIPUAI_API_KEY"] = "2cee1433c4984becab7c30701b2c5fc1.bwFtIqU9qHpWKZDb"

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    extractor = EnhancedMultimodalExtractor(
        use_clip=True,
        domain_config_path=r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\domain_config.json",
        use_zhipuai=True  # 启用智谱AI API
    )

    test_path = r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\test_file.json"

    if os.path.exists(test_path):
        print("文件存在！！")
        result = extractor.process_multimodal_data([test_path])
    else:
        print(f" 文件不存在: {test_path}")

    print(result)

    print("抽取的实体:")
    for ent in result.entities:
        print(f"- {ent['name']} ({ent['type']})")

    print("\n抽取的关系:")
    for rel in result.relationships:
        print(f"- {rel['source']} → {rel['target']}: {rel['type']}")

    print("\n抽取的属性:")
    for attr in result.attributes:
        print(f"- {attr['name']} (属于: {attr.get('entity_name', '未分配')})")


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

