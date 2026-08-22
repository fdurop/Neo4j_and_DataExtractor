# 🧠 Multimodal RAG & Dual-Layer Knowledge Graph System

基于多模态大模型与双层知识图谱的智能教育文档处理与问答系统。
本项目旨在将非结构化的教学课件（PDF/PPTX）转化为结构化的双层图谱，并支持基于向量检索与图谱拓展的精准语义问答（RAG）。

---

## ✨ 核心架构与亮点

### 1. 双层知识图谱架构 (Dual-Layer Graph)
打破传统扁平化存储，构建符合人类认知逻辑的立体图谱：
- **知识层 (Layer 1 - Concept)**：由 DeepSeek 抽取的抽象教学概念（如：步进电机、低通滤波）及逻辑关系（如 `PREREQUISITE_OF`, `RELATED_TO`）。
- **素材层 (Layer 2 - Material)**：物理文档的层级结构（Course -> Document -> Page -> Figure/TextChunk）。
- **跨层缝合**：通过 `HAS_EVIDENCE` 边，将抽象的知识点精准锚定到具体的 PPT 页面和多模态图片上。

### 2. 多模态 RAG 检索链路
- **语义召回**：使用 `qwen3.7-text-embedding` (1024维) 在 Pinecone 中进行高效向量检索。
- **图谱拓展**：在 Neo4j 中以命中的素材节点为起点，进行 N 层 BFS 知识点拓展，获取完整的上下文脉络。
- **多模态组装**：自动解析并挂载本地高清图片（支持增强版 `_enhanced.png`），为 VLM 提供丰富的视觉上下文。

### 3. 极致的工程化保障
- **断点续跑机制**：大模型抽取过程中实时追加落盘 (`.jsonl`)，断网/限流不丢失进度。
- **幻觉校验 (Hallucination Check)**：强制校验 LLM 生成的 `evidence_node_ids`，自动过滤编造的节点和关系。
- **图库批量写入优化**：使用 Neo4j `UNWIND` 语法进行批量并发入库，速度提升数十倍。

---

## 🛠️ 技术栈

- **核心语言**: Python 3.9+
- **大语言模型 (LLM)**: DeepSeek-Chat (用于复杂知识图谱抽取)
- **多模态视觉模型 (VLM)**: Qwen-3.7-Plus (用于多模态理解与最终答案生成)
- **向量模型 (Embedding)**: Qwen-3.7-Text-Embedding (1024维)
- **图数据库 (Graph DB)**: Neo4j AuraDB
- **向量数据库 (Vector DB)**: Pinecone Serverless

---

## 🚀 快速开始

### 1. 环境安装
克隆仓库并安装依赖：
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

### 2. 配置密钥 (重要)
本项目严格分离了代码与配置。请根据模板创建你自己的配置文件：

**配置环境变量 (`key.env`)**：
```bash
cp key.env.example key.env
```
在 `key.env` 中填入你的 `DEEPSEEK_API_KEY`、`PINECONE_API_KEY` 以及 `NEO4J` 的连接凭证。

**配置模型参数 (`vlm_api.yaml`)**：
```bash
cp vlm_api.yaml.example vlm_api.yaml
```
在 `vlm_api.yaml` 中填入阿里云百炼的 API Key。

### 3. 运行流水线

#### 阶段一：知识图谱构建与入库
执行核心构建脚本。该脚本会自动完成：DeepSeek 知识抽取 -> Neo4j 图谱入库 -> Pinecone 向量入库。
```bash
python build_pipeline.py
```
*(💡 提示：如果是首次从 384 维升级到 1024 维，请在代码中临时开启 `delete_pinecone_index()` 清理旧索引)*

#### 阶段二：多模态 RAG 检索测试
运行检索脚本，测试“向量召回 -> 图谱拓展 -> 素材组装”的完整链路：
```bash
python get_prompt.py
```
成功后，将在本地生成 `retrieved_materials_debug.json`，包含完整的结构化 Prompt 素材包。

---

## 📂 目录结构说明

```text
├── build_pipeline.py       # 核心流水线：知识抽取与双库入库
├── get_prompt.py           # RAG 检索端：向量召回与图谱上下文组装
├── vlm_api.yaml            # VLM 与 Embedding 模型配置文件
├── key.env                 # 敏感环境变量（需本地创建，不入库）
├── requirements.txt        # Python 依赖清单
├── .gitignore              # Git 忽略配置
└── README.md               # 项目说明文档
```

---

## ⚠️ 注意事项
1. **数据路径**：请确保代码中的 `KG_DATA_DIR` 和 `DEBUG_DIR` 指向正确的本地抽取结果目录。
2. **API 费用**：DeepSeek 抽取与 Qwen 向量化会产生少量 API 调用费用，测试时请注意控制批次大小。
3. **数据隐私**：请绝对不要将 `key.env` 或包含真实 API Key 的 `vlm_api.yaml` 提交到公开的 GitHub 仓库中！
