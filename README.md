# 多模态文档处理与智能问答系统

一个基于深度学习的多模态文档处理系统，支持PDF和PPTX文件的智能解析、内容提取和问答功能。

## ✨ 主要功能

- 📄 **PDF文档处理**: 智能提取文本、图像、表格内容
- 📊 **PPTX演示文稿处理**: 解析幻灯片文本和图像
- 🖼️ **多模态内容理解**: 基于CLIP模型的图文联合理解
- 🔍 **OCR文字识别**: 支持中英文图像文字提取
- 📋 **表格智能解析**: 自动识别和提取表格数据
- 🤖 **智能问答**: 基于文档内容的问答系统

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 8GB+ RAM (推荐16GB)
- 网络连接 (首次运行需下载模型)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/multimodal-document-processor.git
cd multimodal-document-processor
自动安装环境
复制
python install_environment.py
手动安装 (可选)
复制
pip install -r requirements.txt
创建目录结构
复制
mkdir -p input output clip-model
运行程序
复制
python main_processor.py
📁 项目结构
复制
multimodal-document-processor/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── install_environment.py       # 自动环境配置脚本
├── main_processor.py           # 主处理程序
├── advanced_pptx_processor.py  # PPTX处理模块
├── input/                      # 输入文件目录
├── output/                     # 输出结果目录
└── clip-model/                 # 本地CLIP模型缓存
🔧 使用说明
基本使用
将PDF或PPTX文件放入 input/ 目录
运行 python main_processor.py
处理结果将保存在 output/ 目录
支持的文件格式
PDF文件: .pdf
PowerPoint文件: .pptx
输出内容
📝 提取的文本内容
🖼️ 图像文件和描述
📊 表格数据 (CSV格式)
📋 结构化JSON数据
⚙️ 配置选项
模型配置
系统会自动下载以下模型：

CLIP模型: openai/clip-vit-base-patch32 (~1GB)
OCR模型: EasyOCR中英文模型 (~100MB)
