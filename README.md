# 智能文档处理与知识图谱构建系统

一个基于深度学习和知识图谱的智能文档处理系统，支持多模态文档解析、实体关系抽取和知识图谱构建。

## ✨ 核心功能

### 📄 多模态文档处理
- **智能实体抽取**: 基于DeepSeek大模型的命名实体识别
- **关系抽取**: 自动识别实体间的语义关系
- **属性抽取**: 提取实体的属性和特征信息
- **多模态理解**: 结合CLIP模型进行图文联合理解

### 🔗 知识图谱构建
- **Neo4j集成**: 自动构建和管理知识图谱
- **批量数据导入**: 支持CSV/JSON格式的批量导入
- **语义搜索**: 基于关键词的智能搜索
- **统计分析**: 知识图谱的节点和关系统计

### 🤖 AI驱动的智能处理
- **DeepSeek API**: 云端大模型支持
- **本地模型**: 支持ChatGLM等本地模型
- **并发处理**: 多线程并发提升处理效率
- **缓存机制**: 智能缓存减少重复计算

## 🚀 快速开始

### 系统要求

- **Python**: 3.8+
- **内存**: 8GB+ (推荐16GB)
- **GPU**: NVIDIA GPU (可选，用于加速)
- **Neo4j**: 4.0+ 数据库
- **网络**: 需要访问DeepSeek API

### 环境准备

1. **安装Neo4j数据库**
```bash
# 下载并安装Neo4j Community Edition
# 启动Neo4j服务
neo4j start
克隆项目
复制
git clone https://github.com/your-username/intelligent-document-kg-system.git
cd intelligent-document-kg-system
自动环境配置
复制
python install_environment.py
手动安装依赖 (可选)
复制
pip install -r requirements.txt
配置说明
DeepSeek API配置
复制
# 方法1: 环境变量
export DEEPSEEK_API_KEY="your-api-key-here"

# 方法2: 代码中配置
api_key = "your-deepseek-api-key"
Neo4j数据库配置
复制
# 数据库连接参数
uri = "bolt://localhost:7687"  # 本地数据库
user = "neo4j"
password = "your-password"
📖 使用指南
1. 文档处理和实体抽取
复制
from enhanced_multimodal_extractor import EnhancedMultimodalExtractor

# 初始化抽取器
extractor = EnhancedMultimodalExtractor(
    use_clip=True,
    use_deepseek_api=True,
    api_key="your-deepseek-api-key"
)

# 处理文档
result = extractor.process_multimodal_data(["document.json"])

# 查看结果
print(f"抽取到 {len(result.entities)} 个实体")
print(f"抽取到 {len(result.relationships)} 个关系")
2. 知识图谱构建
复制
from neo4j_team_collaborator import Neo4jTeamCollaborator

# 连接数据库
kg = Neo4jTeamCollaborator(
    uri="bolt://localhost:7687",
    user="neo4j", 
    password="your-password"
)

# 导入数据
kg.import_data(
    node_file="nodes.csv",
    rel_file="relationships.csv",
    format_type="csv"
)

# 语义搜索
results = kg.semantic_search(["人工智能", "机器学习"])
3. 完整工作流程
复制
import os
from enhanced_multimodal_extractor import EnhancedMultimodalExtractor, convert_extracted_to_dataframes
from neo4j_team_collaborator import Neo4jTeamCollaborator

# 1. 文档处理
extractor = EnhancedMultimodalExtractor(
    use_deepseek_api=True,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 2. 抽取实体关系
result = extractor.process_multimodal_data(["input/document.json"])

# 3. 转换为DataFrame
nodes_df, rels_df = convert_extracted_to_dataframes(
    result.entities, 
    result.relationships
)

# 4. 构建知识图谱
kg = Neo4jTeamCollaborator("bolt://localhost:7687", "neo4j", "password")
kg.import_from_dataframe(nodes_df, rels_df)

# 5. 搜索和查询
search_results = kg.semantic_search(["关键词"])
📁 项目结构
复制
intelligent-document-kg-system/
├── README.md                           # 项目说明
├── requirements.txt                    # Python依赖
├── install_environment.py              # 环境配置脚本
├── enhanced_multimodal_extractor.py    # 多模态抽取器
├── neo4j_team_collaborator.py         # Neo4j知识图谱管理
├── config/
│   ├── domain_config.json             # 领域配置文件
│   └── api_config.py                  # API配置
├── data/
│   ├── input/                         # 输入文档目录
│   ├── output/                        # 输出结果目录
│   └── models/                        # 本地模型目录
├── examples/
│   ├── basic_usage.py                 # 基础使用示例
│   ├── advanced_workflow.py           # 高级工作流
│   └── batch_processing.py            # 批量处理示例
└── docs/
    ├── api_reference.md               # API参考文档
    └── troubleshooting.md             # 故障排除指南
⚙️ 配置选项
抽取器配置
复制
extractor = EnhancedMultimodalExtractor(
    use_clip=True,                     # 启用CLIP多模态理解
    domain_config_path="config/domain_config.json",  # 领域配置
    use_deepseek_api=True,             # 使用DeepSeek API
    deepseek_model="deepseek-chat",    # 模型选择
    api_key="your-api-key"             # API密钥
)
Neo4j配置
复制
kg = Neo4jTeamCollaborator(
    uri="bolt://localhost:7687",       # 数据库URI
    user="neo4j",                      # 用户名
    password="password"                # 密码
)
🔧 API参考
EnhancedMultimodalExtractor
主要方法
process_multimodal_data(files): 处理多模态文档
extract_keywords_with_deepseek(question): 关键词提取
clear_cache(): 清空缓存
返回结果
复制
@dataclass
class ExtractedTriple:
    entities: List[Dict]      # 实体列表
    relationships: List[Dict] # 关系列表
    attributes: List[Dict]    # 属性列表
Neo4jTeamCollaborator
主要方法
import_data(node_file, rel_file, format_type): 导入数据
semantic_search(keywords, limit): 语义搜索
advanced_search(keyword, node_types, rel_types): 高级搜索
get_node_statistics(): 节点统计
get_relationship_statistics(): 关系统计
