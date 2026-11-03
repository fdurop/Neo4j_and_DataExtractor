> 当前完整教学助手项目位于 [integrated_app](integrated_app/README.md)。请进入该目录，按说明配置环境并双击 `start.cmd`，再访问 `http://127.0.0.1:8769/login/`。

# 多模态教学知识图谱与 Graph RAG 系统

本项目用于将课程课件的结构化抽取结果构建为一个结合 **Neo4j 知识图谱**、**Pinecone 向量检索**和**多模态大模型**的教学问答系统。

系统支持处理课件中的：

- 页面文本；
- 图片与图片说明；
- 表格及其脚注；
- 代码片段；
- 数学公式；
- 概念及概念关系；
- 文档、页面和素材之间的层级结构。

最终可以根据用户问题检索相关课件页面，回溯对应知识图谱和多模态素材，并生成可发送给视觉语言模型的完整 Prompt。

---

## 主要功能

### 数据库构建

- 读取 extraction 流程生成的正式 JSON 文件；
- 校验文档、页面、素材节点及关系端点；
- 按页面调用 DeepSeek 抽取概念和语义关系；
- 合并完全同名概念；
- 生成疑似同义词候选并调用大模型审核；
- 统一概念描述和类别；
- 将素材层与知识层写入 Neo4j；
- 将已有向量写入 Pinecone；
- 导出供前端展示的图谱 JSON。

### Graph RAG 检索

- 调用 DashScope Embedding API 生成查询向量；
- 从 Pinecone 检索相关页面；
- 从 Neo4j 回溯页面、多模态素材、概念和关系；
- 优先加载增强后的课件图片；
- 完整读取表格正文与表格脚注；
- 组装适合多模态大模型使用的文本 Prompt；
- 返回需要随 Prompt 一起发送的本地图片路径。

---

## 系统架构

```text
extraction 正式输出
        │
        ├── course_manifest.json
        ├── documents.json
        ├── pages.json
        ├── multimodal_nodes.json
        ├── relationships.json
        └── vectors/
            ├── vector_matrix.npy
            └── vector_index.json
        │
        ▼
数据校验与知识抽取
        │
        ├── DeepSeek 概念抽取
        ├── 同义词审核
        ├── 描述统一
        └── 证据关系校验
        │
        ├──────────────────────┐
        ▼                      ▼
     Neo4j                  Pinecone
 双层知识图谱             多模态向量索引
        │                      │
        └──────────┬───────────┘
                   ▼
              Graph RAG 检索
                   │
        ├── 相关页面
        ├── 文本和表格
        ├── 图片和代码
        ├── 直接证据概念
        └── 概念语义关系
                   │
                   ▼
          多模态大模型回答
```

---

## Neo4j 双层图谱

系统将图谱划分为两个层级。

### 素材层

素材层的 `layer` 属性为 `2`，主要包括：

```text
Course
  └── Document
        └── Page
              ├── TextChunk
              ├── Figure
              ├── Table
              ├── Formula
              └── CodeBlock
```

结构节点之间使用 `CONTAINS` 关系连接。

### 知识层

知识层的 `layer` 属性为 `1`，主要由 `Concept` 节点组成。

```text
Concept ──HAS_EVIDENCE──> 素材节点
Concept ──语义关系──────> Concept
```

`HAS_EVIDENCE` 是概念与原始课件素材之间的权威溯源关系。

支持的概念关系包括：

- `IS_A`
- `PART_OF`
- `HAS_PART`
- `HAS_PARAMETER`
- `CONFIGURED_BY`
- `CONTROLS`
- `DRIVES`
- `CONNECTS_TO`
- `INPUTS`
- `OUTPUTS`
- `AFFECTS`
- `IMPLEMENTS`
- `REQUIRES`
- `PREREQUISITE_OF`
- `APPLIES_TO`
- `RELATED_TO`

---

## 项目文件

推荐的项目结构如下：

```text
project/
├── graph_pipeline.py
├── retriever.py
├── test.py
├── requirements.txt
├── README.md
├── key.env
└── vlm_api.yaml
```

文件作用：

| 文件 | 作用 |
|---|---|
| `graph_pipeline.py` | 构建 Neo4j、Pinecone，并导出前端图谱 |
| `retriever.py` | 执行向量检索、图谱回溯和 Prompt 组装 |
| `test.py` | 调用检索模块和多模态大模型进行本地测试 |
| `requirements.txt` | Python 第三方依赖 |
| `key.env` | API Key、数据库地址和索引配置 |
| `vlm_api.yaml` | Embedding 和视觉模型配置 |
| `README.md` | 项目说明 |

如果实际文件名不同，请根据自己的项目结构调整导入语句。

---

## 环境要求

建议使用：

- Python 3.10 或更高版本；
- Neo4j 5.x；
- Pinecone Serverless；
- Windows 10 或 Windows 11；
- 可访问 DashScope 和 DeepSeek API 的网络环境。

---

## 安装依赖

创建 `requirements.txt`：

```text
numpy>=1.24.0,<3.0.0
requests>=2.31.0,<3.0.0
PyYAML>=6.0.1,<7.0.0
neo4j>=5.20.0,<7.0.0
pinecone>=5.0.0,<8.0.0
```

安装依赖：

```bash
pip install -r requirements.txt
```

Windows 下也可以指定虚拟环境解释器：

```bash
C:\Users\Lin\PycharmProjects\PythonProject_fdu\.venv1\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 环境变量配置

在项目根目录创建 `key.env`：

```dotenv
# DeepSeek：用于知识抽取、同义词审核和描述统一
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# DashScope：用于查询文本向量化和多模态模型调用
DASHSCOPE_API_KEY=your_dashscope_api_key

# Neo4j
NEO4J_URI=neo4j+s://your-neo4j-host
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=multimodal-rag-index-v2
PINECONE_NAMESPACE=
PINECONE_DIMENSION=1024
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# 数据版本
DATA_VERSION=graph-rag-v2

# 可选：覆盖默认数据目录
KG_DATA_DIR=C:\path\to\kg_data
DEBUG_DIR=C:\path\to\debug
TEMP_DIR=C:\path\to\work_dir
```

> 不要将包含真实密钥的 `key.env` 提交到 Git。

建议在 `.gitignore` 中加入：

```gitignore
key.env
.env
.venv/
.venv1/
__pycache__/
*.pyc
.idea/
```

---

## YAML 配置

在项目根目录创建 `vlm_api.yaml`：

```yaml
embedding:
  api_key: ""
  base_url: "https://dashscope.aliyuncs.com/api/v1"
  model: "qwen3.7-text-embedding"
  dimension: 1024
  timeout: 60

vlm:
  api_key: ""
  model: "qwen-vl-max"
```

`api_key` 留空时，程序会尝试读取 `key.env` 中的 `DASHSCOPE_API_KEY`。

### 重要要求

查询时使用的 Embedding 模型必须与生成 `vector_matrix.npy` 时使用的模型一致：

```text
入库 Embedding 模型 = 查询 Embedding 模型
入库向量维度 = 查询向量维度 = Pinecone 索引维度
```

仅维度相同并不能保证不同模型生成的向量可以混用。

---

## 正式数据目录

`KG_DATA_DIR` 应至少包含以下文件：

```text
kg_data/
├── course_manifest.json
├── documents.json
├── pages.json
├── multimodal_nodes.json
├── relationships.json
└── vectors/
    ├── vector_matrix.npy
    └── vector_index.json
```

### 文件用途

| 文件 | 内容 |
|---|---|
| `course_manifest.json` | 课程编号和课程名称 |
| `documents.json` | 文档元数据 |
| `pages.json` | 页面正文、摘要和页码 |
| `multimodal_nodes.json` | 文本、图片、表格、公式和代码节点 |
| `relationships.json` | 正式素材层关系 |
| `vector_matrix.npy` | 多模态节点向量矩阵 |
| `vector_index.json` | 向量与素材节点之间的索引关系 |

---

## 构建数据库

直接运行数据库构建脚本：

```bash
python graph_pipeline.py
```

主程序默认执行：

1. 加载正式 extraction 数据；
2. 校验数据结构；
3. 构建或复用知识层；
4. 写入 Neo4j；
5. 写入 Pinecone；
6. 导出完整前端图谱。

### Python 调用

```python
from graph_pipeline import build_and_store_databases

report = build_and_store_databases()

print(report)
```

### 首次构建或强制重新抽取

```python
from graph_pipeline import build_and_store_databases

report = build_and_store_databases(
    force_reextract=True,
    reuse_final_concepts=False,
    write_neo4j=True,
    write_pinecone=True
)
```

这会删除当前工作目录中的旧知识抽取缓存，并重新调用 DeepSeek。

### 复用已有知识层

```python
report = build_and_store_databases(
    force_reextract=False,
    reuse_final_concepts=True
)
```

当 `concepts_graph_v2.json` 已存在且正式输入文件摘要没有变化时，程序会直接复用已有知识层，从而减少 API 调用和运行时间。

### 只写入 Neo4j

```python
report = build_and_store_databases(
    write_neo4j=True,
    write_pinecone=False
)
```

### 只写入 Pinecone

```python
report = build_and_store_databases(
    write_neo4j=False,
    write_pinecone=True
)
```

---

## 数据库构建参数

`build_and_store_databases()` 的常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `resume` | `True` | 是否继续未完成的页面抽取任务 |
| `force_reextract` | `False` | 是否删除缓存并重新抽取知识层 |
| `reuse_final_concepts` | `True` | 是否复用最终知识层文件 |
| `write_neo4j` | `True` | 是否写入 Neo4j |
| `write_pinecone` | `True` | 是否写入 Pinecone |
| `create_pinecone_index` | `True` | 索引不存在时是否自动创建 |
| `concept_confidence` | `0.55` | Concept 最低置信度 |
| `relation_confidence` | `0.55` | 语义关系最低置信度 |
| `max_relations_per_pair` | `2` | 任意两个 Concept 间最多保留的关系数 |
| `page_node_batch_size` | `12` | 每个页面抽取任务包含的最大素材节点数 |
| `data_version` | `graph-rag-v2` | 数据版本标识 |

---

## 导出 Neo4j 图谱

使用 `export_graph_from_neo4j()` 导出前端可读取的 JSON。

### 导出完整双层图谱

```python
from graph_pipeline import export_graph_from_neo4j

result = export_graph_from_neo4j(
    output_path="frontend_graph_v2.json",
    radius=0,
    layer=0,
    include_properties=True
)
```

### 只导出知识层

```python
result = export_graph_from_neo4j(
    output_path="concept_graph.json",
    radius=0,
    layer=1
)
```

### 只导出素材层

```python
result = export_graph_from_neo4j(
    output_path="material_graph.json",
    radius=0,
    layer=2
)
```

### 按中心节点导出局部图谱

```python
result = export_graph_from_neo4j(
    output_path="stepper_motor_graph.json",
    center_node_name="步进电机",
    radius=2,
    layer=1
)
```

`radius` 含义：

| 值 | 含义 |
|---:|---|
| `0` | 未指定中心时导出全部；指定中心时导出完整连通分量 |
| `1` | 中心节点及其一跳邻居 |
| `2` | 中心节点及其两跳邻居 |

`layer` 含义：

| 值 | 含义 |
|---:|---|
| `0` | 同时导出知识层和素材层 |
| `1` | 只导出 Concept 知识层 |
| `2` | 只导出 Page 和多模态素材层 |

---

## 执行 Graph RAG 检索

运行检索脚本：

```bash
python retriever.py
```

或在其他模块中调用：

```python
from retriever import (
    retrieve_materials,
    build_vlm_prompt
)

query = "步进电机驱动器的工作原理是什么？"

retrieved_data = retrieve_materials(
    query=query,
    vector_top_k=10,
    page_top_k=2,
    concept_expand_depth=1
)

prompt, image_paths = build_vlm_prompt(
    retrieved_data
)

print(prompt)
print(image_paths)
```

返回结果包括：

```text
query
parameters
retrieval_summary
pages
  ├── page_info
  ├── retrieval_hit
  ├── text_content
  ├── resolved_images
  ├── multimodal_nodes
  └── knowledge_graph
      ├── concepts
      └── relations
```

---

## 检索参数

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `vector_top_k` | `10` | Pinecone 返回的向量节点数量 |
| `page_top_k` | `2` 或 `3` | 最终保留的不同页面数量 |
| `concept_expand_depth` | `1` | 图谱概念扩展参数 |

当前实现中，建议将 `concept_expand_depth` 设置为 `1`。

如果设置为 `2`，高连接度 Concept 可能带回大量一跳邻居，例如：

- X、Y、Z 轴；
- 丝杆和齿带；
- 接近开关；
- 急停开关；
- 电位器；
- 外部中断和代码文件。

这会导致 Prompt 过长，并降低有效信息比例。

推荐的长期策略是：

1. 获取当前页面直接支持的 Concept；
2. 只保留这些 Concept 之间的关系；
3. 不对高连接度 Concept 进行无约束扩展。

---

## 表格处理

系统按以下优先级读取表格正文：

1. `table_markdown`
2. `markdown`
3. `table_csv`
4. `csv`
5. `table_content`
6. `content`
7. `embedding_text`

当前数据主要将完整表格保存于 `embedding_text`。

表格进入 Prompt 时会保留：

- 表格名称；
- 行数与列数；
- 识别置信度；
- 校验状态；
- 完整表格正文；
- 表格脚注；
- 数据语义提醒。

对步进电机驱动器表格，系统特别区分：

- `Peak`：峰值电流；
- `RMS`：有效值电流；
- `Pulse/rev`：电机旋转一圈所需的输入脉冲数。

`Pulse/rev` 不应直接等同于细分倍数。

---

## 图片处理

系统按以下优先级选择图片：

1. 增强图；
2. 原图。

增强图命名示例：

```text
original.jpeg
original_enhanced.jpeg
```

图片主要通过以下来源解析：

- `documents.json` 中的 `metadata.image_mapping`；
- `DEBUG_IMAGES_DIR` 下的文件名模式；
- Figure 节点中的 `source_path` 或 `enhanced_path`。

图片路径会与文本 Prompt 分开返回，需要由调用视觉模型的代码同时上传。

---

## 工作目录输出

默认会在工作目录生成：

```text
concepts_partial_v2.jsonl
concepts_raw_v2.json
concepts_graph_v2.json
graph_quality_report_v2.json
database_pipeline_report_v2.json
frontend_graph_v2.json
```

| 文件 | 说明 |
|---|---|
| `concepts_partial_v2.jsonl` | 页面级知识抽取的断点缓存 |
| `concepts_raw_v2.json` | 未完成全局归一化的概念和关系 |
| `concepts_graph_v2.json` | 最终知识层 |
| `graph_quality_report_v2.json` | 知识层数量和归一化统计 |
| `database_pipeline_report_v2.json` | 整体运行报告 |
| `frontend_graph_v2.json` | 前端图谱展示数据 |

---

## 缓存与重新构建

如果仅修改 Pinecone 索引名称，不需要重新执行知识抽取。

如果修改了以下内容，则需要重新抽取知识层：

- Concept 抽取 Prompt；
- aliases 规则；
- 同义词候选规则；
- 同义词审核 Prompt；
- 概念描述合成规则；
- 关系抽取规则。

推荐执行：

```python
build_and_store_databases(
    force_reextract=True,
    reuse_final_concepts=False
)
```

如果只修改以下内容，通常不需要重新抽取：

- Neo4j 入库 Cypher；
- `HAS_EVIDENCE` 建边逻辑；
- 图谱导出逻辑；
- Pinecone metadata；
- 检索侧图谱裁剪；
- Prompt 组装方式。

此时可以复用：

```text
concepts_graph_v2.json
```

---

## 已知注意事项

### 不会自动清空数据库

数据库构建函数只使用 `MERGE` 和 `SET`，不会主动清空 Neo4j，也不会删除 Pinecone 索引。

如果知识层 Concept 被重命名，旧 Concept 可能继续保留在 Neo4j 中，需要手动清理旧版本数据。

清理前请先备份数据库。

### 同义词合并需要谨慎

下列概念不能视为完全同义：

```text
步进电机驱动器
步进电机驱动器原理样机
商品化步进电机驱动器
```

推荐建模方式：

```text
驱动器原理样机
    ──IS_A──>
步进电机驱动器

商品化步进电机驱动器
    ──IS_A──>
步进电机驱动器
```

“可替代”“功能相似”和“属于同一类设备”不代表两个概念可以合并。

### 图谱不是高于原始素材的事实来源

推荐证据优先级：

```text
当前页面原始课件文本、表格和图片
    >
当前页面直接证据知识图谱
    >
扩展知识图谱
    >
一般工程知识
```

如果知识图谱与原始课件冲突，应优先使用原始课件，并将冲突标记为可能的抽取或归一化错误。

### Embedding 模型必须一致

如果查询模型和向量入库模型不同，即使维度均为 1024，检索结果也可能失去语义可靠性。

### 图谱扩展不宜过深

无约束的多跳查询会通过高连接度节点引入大量无关概念。教学问答场景一般优先使用页面直接证据 Concept。

---

## 常见问题

### Pinecone 没有返回页面编号

检查向量 metadata 是否包含：

```text
page_id
source_page_id
node_id
node_type
```

Page 节点至少应包含：

```text
node_type=Page
node_id=对应页面ID
```

### 查询结果仍然使用旧索引

检查 `key.env`：

```dotenv
PINECONE_INDEX_NAME=multimodal-rag-index-v2
```

注意系统环境变量优先于 `key.env`，因为默认加载方式不会覆盖已经存在的同名系统变量。

### 修改 Prompt 后没有生效

原因通常是程序复用了：

```text
concepts_graph_v2.json
```

使用：

```python
build_and_store_databases(
    force_reextract=True,
    reuse_final_concepts=False
)
```

### Prompt 中出现大量无关概念

将：

```python
concept_expand_depth=2
```

改为：

```python
concept_expand_depth=1
```

并优先采用“当前页面直接 Concept 内部关系”的检索策略。

### 图片存在但未发送给模型

检查：

- `documents.json` 中的 `image_mapping`；
- `DEBUG_IMAGES_DIR`；
- Figure 节点中的 `source_path`；
- 增强图片是否真实存在；
- 调用视觉模型时是否同时传入了 `image_paths`。

### Neo4j 连接失败

检查：

```dotenv
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
```

并确认 Neo4j 实例允许当前网络地址访问。

---

## 安全建议

- 不要将 API Key 写死在 Python 文件中；
- 不要将 `key.env` 提交到公开仓库；
- 不要在日志中输出完整认证信息；
- 删除 Neo4j 节点或 Pinecone 索引前先备份；
- 对大模型抽取的概念和关系保留证据节点；
- 不要将无证据的大模型推断直接写入知识图谱。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 key.env 和 vlm_api.yaml

# 3. 构建 Neo4j 和 Pinecone
python graph_pipeline.py

# 4. 测试 Graph RAG 检索
python retriever.py

# 5. 运行完整问答测试
python test.py
```

---

## 对外接口

数据库构建模块推荐只使用：

```python
from graph_pipeline import (
    build_and_store_databases,
    export_graph_from_neo4j
)
```

检索模块推荐使用：

```python
from retriever import (
    retrieve_materials,
    build_vlm_prompt
)
```

---

## License

当前项目尚未指定开源许可证。

如果计划公开发布，请根据实际用途添加合适的许可证，例如：

- MIT License；
- Apache License 2.0；
- GPL-3.0。

同时请确认课程课件、图片、模型输出和第三方服务的使用符合相应授权要求。
