"""
图像处理模块
功能：图像增强、通过远程多模态 API 生成描述
"""

import os
import re
from PIL import Image, ImageEnhance
from typing import Optional, Dict, Any

try:
    from utils.path_config import workspace_relative
except ImportError:  # Package import used by tests and library callers.
    from ..utils.path_config import workspace_relative


IMAGE_DESCRIPTION_POLICY = "visible_evidence_v4_domain_neutral"


def _relative_path(path: Optional[str]) -> str:
    return workspace_relative(path)


class ImageProcessor:
    def __init__(self, logger=None, vlm_client=None):
        """初始化图像处理器。"""
        self.logger = logger
        self.vlm_client = vlm_client
    
    def enhance_image(self, image_path: str) -> str:
        """
        增强图像质量
        
        Args:
            image_path: 图像路径
            
        Returns:
            增强后的图像路径
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                if self.logger:
                    self.logger.warning(f"图像文件不存在: {image_path}")
                return image_path
            
            # 检查文件扩展名
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext in ['.svg', '.emf', '.wmf']:
                # 矢量图无法用PIL增强，跳过
                if self.logger:
                    self.logger.info(f"跳过矢量图增强: {image_path}")
                return image_path
            
            img = Image.open(image_path)
            
            # 处理调色板模式（Palette）图像的透明度问题
            if img.mode == 'P':
                # 如果有透明度信息，转换为RGBA
                if 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            
            # 转换为RGB模式（统一处理）
            if img.mode == 'RGBA':
                # RGBA需要处理透明度，创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 使用alpha通道作为mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 增强对比度
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # 增强锐度
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            # Only change the file name suffix. Replacing every "." also changes
            # model directories such as qwen3.7-plus.
            root, ext = os.path.splitext(image_path)
            enhanced_path = f"{root}_enhanced{ext}"
            img.save(enhanced_path)
            
            return enhanced_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"图像增强失败: {e}")
            return image_path
    
    def generate_description(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """
        生成图像描述（自动选择最佳方法）
        
        Args:
            image_path: 图像路径
            context: 上下文文本
            
        Returns:
            包含描述信息的字典
        """
        result = {
            "image_path": image_path,
            "description": "",
            "method": "none",
            "enhanced": False,
            "backend": "none",
            "provider": None,
            "model": None,
            "status": "not_started",
            "api_error": None,
            # Intended policy is available even when no model can run. Only a
            # successful returned description receives description_policy.
            "prompt_policy": IMAGE_DESCRIPTION_POLICY,
            "description_policy": None,
            "context_policy": "image_only_page_text_kept_separately",
            "response_metadata": {},
            "semantic_review_flags": [],
        }
        
        # Prefer the configured remote VLM API.
        if self.vlm_client:
            try:
                prompt = (
                    "请忠实描述这张教学材料图片，只记录图中可见且可确认的信息。\n"
                    "仅当可以确认图片只是校徽、水印、装饰、二维码或与教学内容无关的照片时，"
                    "请以[COURSE_IRRELEVANT]开头并简要说明原因。\n"
                    "其他图片第一句写“类型：示意图/照片/表格/公式/代码/其他”中的一种，"
                    "随后只记录可见的关键文字、对象、连线、坐标轴、步骤或空间关系。"
                    "对象的类别限定词、设备型号、端口号和关键符号应按图中原样保留，不要统一改写成泛称；"
                    "同属一类的对象也应保留各自的子类型与限定词。"
                    "文字或符号标签按原样记录，并说明可见位置；只有归属清楚时才把标签赋给某个对象。"
                    "方向、连通或状态必须有清晰可见的箭头、连接或标注支撑；归属或状态不明时只描述可见信息，不补推断。"
                    "页面标题与图中名称不同时，分别保留图中名称和可见说明，不能用页面主题覆盖局部图例或示例。"
                    "不要把图中所有名词判为教学知识点，不推断先修、同义或层级关系；仅描述明确可见的关系。"
                    "禁止使用Markdown，禁止根据常识推测用途、原理和课程结论；看不清或无法确认时写“不确定”。"
                    "若是表格、公式或代码，只概括主题并保留区分内容所需的关键标记，不在通用描述中完整转录，"
                    "专用流程会另行识别。总长度尽量控制在300个汉字以内，优先保留对象名称与区分信息。"
                )
                # A live test showed that full page text caused a wiring image
                # to be described as also containing a table elsewhere on the
                # page. Keep page text in the evidence handoff, outside this
                # image-only perception request. Retain `context` for callers.
                description = self.vlm_client.generate_description(image_path, prompt)
                result["description"] = description
                result["method"] = getattr(self.vlm_client, "provider_name", "vlm_api")
                result["backend"] = "api"
                result["provider"] = getattr(self.vlm_client, "provider", None)
                result["model"] = getattr(self.vlm_client, "model", None)
                result["status"] = "success"
                result["description_policy"] = IMAGE_DESCRIPTION_POLICY
                result["response_metadata"] = getattr(self.vlm_client, "last_response_metadata", {}) or {}
                return result
            except Exception as e:
                result["api_error"] = str(e)
                result["status"] = "api_failed"
                if self.logger:
                    self.logger.warning(f"VLM API描述生成失败: {e}")
        
        if result["status"] in {"not_started", "api_failed"}:
            result["status"] = "no_model_succeeded"
        return result
    
    def process_image(self, image_path: str, page_text: str = "") -> Dict[str, Any]:
        """
        完整的图像处理流程
        
        Args:
            image_path: 图像路径
            page_text: 页面文本（提供上下文）
            
        Returns:
            处理结果字典
        """
        result = {
            "original_path": _relative_path(image_path),
            "enhanced_path": None,
            "description": None,
            "metadata": {
                "generation": {
                    "prompt_policy": IMAGE_DESCRIPTION_POLICY,
                    "description_policy": None,
                    "context_policy": "image_only_page_text_kept_separately",
                }
            },
            "error": None,
            "method": "none",
            "backend": "none",
            "provider": None,
            "model": None,
            "status": "not_started",
            "api_error": None,
            "prompt_policy": IMAGE_DESCRIPTION_POLICY,
            "description_policy": None,
            "context_policy": "image_only_page_text_kept_separately",
            "response_metadata": {},
            "semantic_review_flags": [],
            "indexable": False,
            "review_required": False,
            "review_reason": None,
            "review_status": None,
            "course_relevant": None,
        }
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            error_msg = f"图像文件不存在: {image_path}"
            if self.logger:
                self.logger.warning(error_msg)
            result["error"] = error_msg
            return result
        
        # 检查文件格式，跳过无法处理的格式
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.svg', '.emf', '.wmf']:
            error_msg = f"跳过矢量图格式: {image_path}"
            if self.logger:
                self.logger.info(error_msg)
            result["error"] = error_msg
            return result
        
        # 1. 图像增强
        try:
            enhanced_path = self.enhance_image(image_path)
            result["enhanced_path"] = _relative_path(enhanced_path)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"图像增强失败: {e}")
            result["enhanced_path"] = _relative_path(image_path)
        
        # 2. 生成描述
        try:
            desc_image_path = enhanced_path if "enhanced_path" in locals() else image_path
            desc_result = self.generate_description(desc_image_path, page_text)
            # desc_result 是字典 {"description": "...", "method": "..."}
            result["description"] = desc_result.get("description", "")
            result["method"] = desc_result.get("method", "unknown")
            result["backend"] = desc_result.get("backend", "unknown")
            result["provider"] = desc_result.get("provider")
            result["model"] = desc_result.get("model")
            result["status"] = desc_result.get("status", "unknown")
            result["api_error"] = desc_result.get("api_error")
            result["prompt_policy"] = desc_result.get("prompt_policy", IMAGE_DESCRIPTION_POLICY)
            result["description_policy"] = desc_result.get("description_policy")
            result["context_policy"] = desc_result.get("context_policy")
            result["response_metadata"] = desc_result.get("response_metadata") or {}
            description = str(result["description"] or "").strip()
            # Live runs disagreed on arrows in the same magnetic-field image.
            # Keep the observation available, but make these claims explicit
            # review targets rather than treating API success as verification.
            if re.search(r"箭头|指向|顺时针|逆时针|流向|转向|极性", description):
                result["semantic_review_flags"].append("directional_claim_requires_visual_review")
            if re.search(r"(?<![A-Za-z])[NS](?![A-Za-z])|北极|南极", description):
                result["semantic_review_flags"].append("magnetic_pole_label_requires_visual_review")
            irrelevant_marker = bool(
                re.match(r"^\s*\[?COURSE[_ -]?IRRELEVANT\]?", description, re.IGNORECASE)
                or re.match(r"^\s*\[?课程无关\]?", description)
            )
            if result["status"] != "success":
                result["review_required"] = True
                result["review_reason"] = "api_failed"
            elif not description:
                result["review_required"] = True
                result["review_reason"] = "empty_description"
            elif result["response_metadata"].get("finish_reason") == "length":
                result["review_required"] = True
                result["review_reason"] = "truncated_description"
            elif irrelevant_marker:
                result["review_required"] = True
                result["review_reason"] = "course_irrelevant"

            result["review_status"] = "pending" if result["review_required"] else None
            result["course_relevant"] = False if irrelevant_marker else (
                True if result["status"] == "success" and description else None
            )
            result["indexable"] = bool(description) and not result["review_required"]
            result["metadata"]["generation"] = {
                "method": result["method"],
                "backend": result["backend"],
                "provider": result["provider"],
                "model": result["model"],
                "status": result["status"],
                "api_error": result["api_error"],
                "prompt_policy": result["prompt_policy"],
                "description_policy": result["description_policy"],
                "context_policy": result["context_policy"],
                "response_metadata": result["response_metadata"],
            }
            result["metadata"]["review"] = {
                "required": result["review_required"],
                "status": result["review_status"],
                "reason": result["review_reason"],
                "course_relevant": result["course_relevant"],
                "semantic_review_flags": result["semantic_review_flags"],
            }
        except Exception as e:
            error_msg = f"图像描述生成失败: {e}"
            if self.logger:
                self.logger.error(error_msg)
            result["error"] = error_msg
            result["status"] = "failed"
        
        return result
