#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档处理与知识图谱构建系统 - 环境配置脚本
自动安装所有必要的依赖包和配置环境
"""

import subprocess
import sys
import os
import platform
import json
from pathlib import Path
import urllib.request
import zipfile
import tarfile


class IntelligentKGInstaller:
    def __init__(self):
        self.failed_packages = []
        self.system = platform.system()
        self.python_version = sys.version_info
        self.project_root = Path.cwd()
        
    def print_header(self, title):
        """打印格式化标题"""
        print("\n" + "=" * 70)
        print(f" {title}")
        print("=" * 70)
    
    def print_step(self, step, total, description):
        """打印步骤信息"""
        print(f"\n[{step}/{total}] {description}")
        print("-" * 50)
    
    def check_system_requirements(self):
        """检查系统要求"""
        self.print_step(1, 10, "检查系统要求")
        
        # 检查Python版本
        if self.python_version < (3, 8):
            print(f"❌ Python版本过低: {sys.version}")
            print("   要求: Python 3.8+")
            return False
        else:
            print(f"✅ Python版本: {sys.version}")
        
        # 检查系统类型
        print(f"🖥️  操作系统: {self.system}")
        
        # 检查内存 (简单估算)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"💾 系统内存: {memory_gb:.1f}GB")
            if memory_gb < 8:
                print("⚠️  警告: 内存少于8GB，可能影响性能")
        except ImportError:
            print("ℹ️  无法检测内存信息")
        
        return True
    
    def upgrade_pip(self):
        """升级pip和基础工具"""
        self.print_step(2, 10, "升级pip和基础工具")
        
        packages_to_upgrade = ["pip", "setuptools", "wheel"]
        
        for package in packages_to_upgrade:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade", package
                ], stdout=subprocess.DEVNULL)
                print(f"   ✅ {package} 升级成功")
            except subprocess.CalledProcessError:
                print(f"   ⚠️  {package} 升级失败")
    
    def install_pytorch(self):
        """智能安装PyTorch"""
        self.print_step(3, 10, "安装PyTorch深度学习框架")
        
        # 检查CUDA可用性
        cuda_available = False
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                cuda_available = True
                print("🚀 检测到NVIDIA GPU，安装CUDA版本PyTorch")
            else:
                print("💻 未检测到GPU，安装CPU版本PyTorch")
        except FileNotFoundError:
            print("💻 未检测到nvidia-smi，安装CPU版本PyTorch")
        
        # 选择安装命令
        if cuda_available:
            torch_packages = ["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0"]
        else:
            torch_packages = [
                "torch==2.8.0+cpu", 
                "torchvision==0.23.0+cpu", 
                "torchaudio==2.8.0+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + torch_packages, stdout=subprocess.DEVNULL)
            print("   ✅ PyTorch安装成功")
            return True
        except subprocess.CalledProcessError:
            print("   ❌ PyTorch安装失败")
            self.failed_packages.extend(["torch", "torchvision", "torchaudio"])
            return False
    
    def install_core_packages(self):
        """安装核心依赖包"""
        self.print_step(4, 10, "安装核心依赖包")
        
        # 尝试从requirements.txt安装
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            print("   📋 从requirements.txt安装...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], stdout=subprocess.DEVNULL)
                print("   ✅ requirements.txt安装成功")
                return True
            except subprocess.CalledProcessError:
                print("   ⚠️  requirements.txt安装失败，尝试逐个安装...")
        
        # 核心包列表
        core_packages = [
            "transformers==4.55.2",
            "pandas==2.3.1",
            "numpy==2.2.1",
            "requests==2.32.4",
            "neo4j==5.28.2",
            "spacy==3.8.7",
            "sentence-transformers==5.1.0",
            "Pillow==11.0.0",
            "chardet==5.2.0",
            "tqdm==4.67.1",
            "colorama==0.4.6"
        ]
        
        success_count = 0
        for package in core_packages:
            package_name = package.split("==")[0]
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"   ✅ {package_name}")
                success_count += 1
            except subprocess.CalledProcessError:
                print(f"   ❌ {package_name}")
                self.failed_packages.append(package)
        
        print(f"   📊 核心包安装: {success_count}/{len(core_packages)}")
        return success_count > len(core_packages) * 0.8
    
    def install_optional_packages(self):
        """安装可选包"""
        self.print_step(5, 10, "安装可选依赖包")
        
        optional_packages = [
            "opencv-python==4.10.0.84",
            "openpyxl==3.1.5",
            "python-dotenv==1.0.1",
            "pyyaml==6.0.2",
            "loguru==0.7.2",
            "rich==13.9.4"
        ]
        
        for package in optional_packages:
            package_name = package.split("==")[0]
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"   ✅ {package_name} (可选)")
            except subprocess.CalledProcessError:
                print(f"   ⚠️  {package_name} (可选，安装失败)")
    
    def install_spacy_model(self):
        """安装spaCy中文模型"""
        self.print_step(6, 10, "安装spaCy中文语言模型")
        
        try:
            # 检查spacy是否已安装
            import spacy
            print("   📦 下载中文语言模型...")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "zh_core_web_sm"
            ], stdout=subprocess.DEVNULL)
            print("   ✅ spaCy中文模型安装成功")
            return True
        except ImportError:
            print("   ⚠️  spaCy未安装，跳过模型下载")
            return False
        except subprocess.CalledProcessError:
            print("   ⚠️  spaCy中文模型下载失败")
            return False
    
    def create_project_structure(self):
        """创建项目目录结构"""
        self.print_step(7, 10, "创建项目目录结构")
        
        directories = [
            "data/input",
            "data/output", 
            "data/models",
            "config",
            "examples",
            "docs",
            "logs"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   📁 {dir_path}/")
        
        print("   ✅ 项目目录结构创建完成")
    
    def create_config_files(self):
        """创建配置文件"""
        self.print_step(8, 10, "创建配置文件")
        
        # 创建领域配置文件
        domain_config = {
            "term_terms": [
                "性能评估基准", "Benchmark", "标准化测试", "测试任务", 
                "数据集", "模型性能", "准确率", "推理速度", "计算效率"
            ],
            "concept_terms": [
                "性能评估", "跨模型比较", "模型优化", "标准化"
            ],
            "default_relations": {
                "definition": ["指的是", "指", "是", "means", "refers to"],
                "purpose": ["用来", "用于", "目的是", "used for", "purpose is"],
                "function": ["提供", "允许", "帮助", "指导", "provide", "allow", "help", "guide"],
                "characteristic": ["例如", "比如", "such as", "including"],
                "contains": ["包含", "包括", "含有"],
                "supports": ["支持", "允许", "帮助"],
                "ensures": ["确保", "保证"]
            }
        }
        
        config_file = self.project_root / "config" / "domain_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(domain_config, f, ensure_ascii=False, indent=2)
        print("   ✅ 领域配置文件创建完成")
        
        # 创建API配置文件
        api_config_content = '''"""
API配置文件
请在此文件中配置您的API密钥和数据库连接信息
"""

import os

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Neo4j数据库配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your-password-here")

# 本地模型路径配置
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "data/models")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "data/models/clip-vit-base-patch32")
'''
        
        api_config_file = self.project_root / "config" / "api_config.py"
        with open(api_config_file, 'w', encoding='utf-8') as f:
            f.write(api_config_content)
        print("   ✅ API配置文件创建完成")
    
    def create_example_files(self):
        """创建示例文件"""
        self.print_step(9, 10, "创建示例文件")
        
        # 基础使用示例
        basic_example = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础使用示例
演示如何使用系统进行文档处理和知识图谱构建
"""

import os
import sys
sys.path.append('..')

from enhanced_multimodal_extractor import EnhancedMultimodalExtractor
from neo4j_team_collaborator import Neo4jTeamCollaborator
from config.api_config import *

def main():
    print("🚀 智能文档处理与知识图谱构建示例")
    
    # 1. 初始化抽取器
    extractor = EnhancedMultimodalExtractor(
        use_clip=True,
        use_deepseek_api=True,
        api_key=DEEPSEEK_API_KEY
    )
    
    # 2. 处理文档 (请确保有输入文件)
    input_files = ["data/input/example.json"]
    if os.path.exists(input_files[0]):
        result = extractor.process_multimodal_data(input_files)
        print(f"✅ 抽取完成: {len(result.entities)}个实体, {len(result.relationships)}个关系")
    else:
        print("⚠️  示例输入文件不存在，请添加文档到 data/input/ 目录")
    
    # 3. 连接知识图谱
    try:
        kg = Neo4jTeamCollaborator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        print("✅ 知识图谱连接成功")
        
        # 4. 搜索示例
        results = kg.semantic_search(["人工智能"])
        print(f"🔍 搜索结果: {len(results)}条")
        
        kg.close()
    except Exception as e:
        print(f"❌ 知识图谱连接失败: {e}")

if __name__ == "__main__":
    main()
'''
        
        example_file = self.project_root / "examples" / "basic_usage.py"
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(basic_example)
        print("   ✅ 基础示例文件创建完成")
    
    def check_dependencies(self):
        """检查关键依赖是否正确安装"""
        self.print_step(10, 10, "验证安装结果")
        
        critical_packages = [
            ("torch", "PyTorch深度学习框架"),
            ("transformers", "Transformers模型库"),
            ("neo4j", "Neo4j数据库驱动"),
            ("pandas", "数据处理库"),
            ("requests", "HTTP请求库")
        ]
        
        success_count = 0
        for package, description in critical_packages:
            try:
                __import__(package)
                print(f"   ✅ {description}")
                success_count += 1
            except ImportError:
                print(f"   ❌ {description} - 导入失败")
        
        # 检查CUDA可用性
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   🚀 GPU加速可用: {gpu_name}")
            else:
                print("   💻 使用CPU模式")
