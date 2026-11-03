---
author: 昕煜 冷
---

# fdurop 课程图谱与教学问答助手

此分支保存截至 2026-11-03 的本地应用源码。应用入口是 `integrated_app/`，仓库根目录仍保留原知识提取工程。源码上传到 GitHub 不代表系统已部署为公网网站。

## Windows 启动

已经配置好本地环境的电脑：双击 `start.cmd`，保持启动窗口打开；在浏览器地址栏输入 `http://127.0.0.1:8769/login/`。

后台已运行时，直接访问上述地址，不必重复启动。教师和学生同时使用时，可分别在 Edge 普通窗口、InPrivate 隐私窗口登录；只需要一个后台服务。

首次下载：

```powershell
git clone --branch new --single-branch https://github.com/fdurop/Neo4j_and_DataExtractor.git
cd Neo4j_and_DataExtractor/integrated_app
powershell -ExecutionPolicy Bypass -File setup.ps1
Copy-Item config/local.env.example config/local.env
```

安装 Python 3.12。编辑 `config/local.env`，填写自己的 Neo4j、模型和 Pinecone 配置，按需要启用真实处理和向量检索，再双击 `start.cmd`。运行数据保存在 `runtime/`。

需要从 PowerShell 启动时：

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

本地界面/JSON 副本检查可使用 `DATABASE_MODE=local`、`ENABLE_PAID_PIPELINE=0`；完整真实课件处理需要 Neo4j 和模型配置。课程问答需要启用真实模型；向量模式还需要 Pinecone 配置。

## 当前功能

- PDF 上传、逐页多模态提取、知识建图、Neo4j 入库读回与课程图谱发布。PPT/PPTX 需先导出为 PDF。
- 教师和学生登录、课程管理、图谱浏览、节点和关系编辑、素材核对、出处查看与导出。
- 云端图谱编辑同步 Neo4j 原生节点/关系并保存审核快照和撤销历史；按课程 ID 限定读写范围。本地导入模式需导出 JSON 保存。
- 课程资料充分时回答并给出引用；资料不足时调用模型回答通用知识，并明确标注。检索/模型故障仍报错，不伪装成资料不足。
- 问题中明确提及的概念参与检索；相关教师编辑优先进入回答依据。同页课件材料合并，教师补充与课件来源分开标注，保留待确认状态。
- “删除课程”经确认后彻底清理课程所属节点、关系、图谱历史、上传处理资料和向量索引。处理中任务阻止删除，失败保留重试入口。共享预算记录和人工备份不属于活动课程清理范围。
- 数据库幂等导入有限重试，证据关系按素材标签走索引；模型短暂连接故障有限重试，保留预算和已完成结果。
- 向量请求最多尝试三次；已完成批次保存检查点，失败重试复用已有向量结果。课件页面提供“重试处理”按钮。

教师修改和自动抽取内容仍需核对，不能据此宣称每个回答都正确或整门课程已验收。

## 验证与说明

本地完整离线测试 229 项通过。真实课件流程已完成提取、Neo4j 入库读回、向量索引及网页问答检查；教师修订使用完全相同的问题验证进入回答和引用。独立测试课程验证过 Neo4j/Pinecone 删除及其他课程保持不变，测试数据已清理。

在 `integrated_app/` 下运行测试：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

课程隔离见 `COURSE_ISOLATION_CHECK.md`；删除与问答设计见 `COURSE_DELETE_AND_QA.md`；图谱原生存储见 `NATIVE_GRAPH_CRUD.md`。`delivery_manifest.json` 记录此交付的文件 SHA-256。

本分支不包含 `config/local.env`、API 密钥、账号数据库、课程运行数据或虚拟环境。下载后需要自行配置；不会自动带上本机账号和课程，也不会建立公网部署。
