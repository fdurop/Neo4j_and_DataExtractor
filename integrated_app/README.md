---
author: 昕煜 冷
---

# 完整联调版本

本分支 `integration-demo-20261029` 保存目前测试的完整联调源码，入口在本目录 `integrated_app/`。仓库根目录保留原知识提取工程；完整网页应用请从本目录启动。

## 已有功能

- PDF 上传、任务进度、知识提取、图谱构建和按课程存储。
- 登录、课程管理、图谱文字与关系编辑、课件出处核对、SVG/PNG 导出。
- 全宽图谱画布、四档缩放展开、视野绘制和课件筛选。四档表示显示详略，教学层级与先修关系仍需教师核对。
- 跨课件同名概念累加来源与证据，并保留教师修改。
- 区分加载中、未登录、读取失败、课程无数据和筛选为空。

## Windows 启动

1. 安装 Python 3.12，在本目录运行 `powershell -ExecutionPolicy Bypass -File setup.ps1`。
2. 将 `config/local.env.example` 复制为 `config/local.env`，在本机填写自己的 Neo4j 配置。需要实际提取时再填写模型密钥并启用 `ENABLE_PAID_PIPELINE`。默认不启用收费流水线。
3. 双击 `start.cmd`，保持启动窗口打开。
4. 打开 <http://127.0.0.1:8769/login/>，注册或使用已有账号。

只检查本地界面与 JSON 副本时，可把配置中的 `DATABASE_MODE` 改为 `local`，保持 `ENABLE_PAID_PIPELINE=0`。从图谱页“文件与导出 → 导入图谱 JSON”打开自己的图谱副本。

这里的 localhost 地址指运行服务的电脑，不是已经部署好的公共网站。

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
