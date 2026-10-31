---
author: 昕煜 冷
---

# 当前系统：课程图谱编辑演示

当前分支为 `graph/interactive-editor-20261031`，应用入口在本目录 `integrated_app/`。仓库根目录保留原知识提取工程；完整网页应用请从本目录启动。

这是可下载并在本机运行的源码，不是已经部署的在线演示网站。

## 已有功能

- PDF 上传、任务进度、知识提取、图谱构建和按课程存储。
- 登录、课程管理、图谱文字与关系编辑、课件出处核对、SVG/PNG 导出。
- 可拖动的力导向图谱、拖动后的弹性运动、节点固定与缩放。
- 点击节点后直接邻居环形展开，其余内容淡化；再次点击或点击空白处恢复原布局与视角。
- 紧凑的节点资料表、优先显示的关系编辑、按内容伸缩的说明框，以及默认折叠的来源证据。
- 教学层级与先修关系仍需教师核对。
- 跨课件同名概念累加来源与证据，并保留教师修改。
- 区分加载中、未登录、读取失败、课程无数据和筛选为空。

## 获取这个版本

```powershell
git clone --branch graph/interactive-editor-20261031 --single-branch https://github.com/fdurop/Neo4j_and_DataExtractor.git
cd Neo4j_and_DataExtractor/integrated_app
```

## Windows 启动

1. 安装 Python 3.12，在本目录运行 `powershell -ExecutionPolicy Bypass -File setup.ps1`。
2. 将 `config/local.env.example` 复制为 `config/local.env`，在本机填写自己的 Neo4j 配置。需要实际提取时再填写模型密钥并启用 `ENABLE_PAID_PIPELINE`。默认不启用收费流水线。
3. 双击 `start.cmd`，保持启动窗口打开。
4. 打开 <http://127.0.0.1:8769/login/>，注册或使用已有账号。

只检查本地界面与 JSON 副本时，可把配置中的 `DATABASE_MODE` 改为 `local`，保持 `ENABLE_PAID_PIPELINE=0`。从图谱页“文件与导出 → 导入图谱 JSON”打开自己的图谱副本。

这里的 localhost 地址指运行服务的电脑，不是已经部署好的公共网站。

## 当前图谱编辑的存储方式

云端模式的编辑保存到 Neo4j 的课程审核图谱快照 `Course.review_graph_json`，并提供版本检查和一步撤销。当前不会同步增删改 Neo4j 原生 `Concept` 节点和原生关系。2026-10-31 已通过真实浏览器操作与独立数据库查询核对快照中的新增、删除、修改和改回；保存/重载延迟与读取超时仍需优化。

本地 JSON 导入模式只编辑页面副本，需要导出 JSON 保存，不会自动写入云端。

图谱交互与回退说明见 [GRAPH_EDITOR.md](GRAPH_EDITOR.md)；独立只读删除核验脚本见 [scripts/VERIFY_GRAPH_DELETE.md](scripts/VERIFY_GRAPH_DELETE.md)。

## 验证范围与待办

已有一份完整 21 页课件的真实提取、建图、云库写入读回与网页联调验证。源码交付前已有 133 项离线测试通过；此前还做过 50 份模拟课件、5,000 节点和 10,000 条关系的图谱显示检查。

整门课程的真实抽取、跨课件同义词合并质量、教师确认的教学层级、服务器部署和 PPTX 上传入口仍需继续验证。模拟压力测试不能代替真实课程验收。

本分支不包含本机密钥、账号数据库、真实课程运行数据、虚拟环境及历史输出，因此拉取代码后不会自动出现原电脑上的课程数据。需要已有结果时，请由组内另外共享图谱 JSON 或配置授权数据库，不要把密码提交到仓库。

修改与验证记录见 `进度/`、`integration_status.json`；`delivery_manifest.json` 用于核对联调源码文件。

## 离线测试

在仓库根目录运行：

```powershell
& .\integrated_app\.venv\Scripts\python.exe -X utf8 -m unittest discover -s integrated_app/tests -v
```
