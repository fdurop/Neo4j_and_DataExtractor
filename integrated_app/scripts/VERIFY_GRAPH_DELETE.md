# 独立验证图谱删除

脚本直接连接 Neo4j，不经过网页接口，只执行读取，不会删除或修改数据库。
项目内也保存了一份：`scripts/verify_graph_delete.py`。

## 推荐：创建一个测试节点再删除

先在网页新增一个叫“删除验证专用节点”的节点，再在 PowerShell 运行：

```powershell
cd C:\Users\Lin\Desktop\study_documents\fdurop\python_doc
$courseId = 'course_4385a1b0c2b34e949778ae01b2cc66a9'
& .\.venv\Scripts\python.exe -X utf8 .\scripts\verify_graph_delete.py --course-id $courseId --name '删除验证专用节点' --out .\runtime\delete-before.json
```

这一步保存删除前记录。然后在网页删除该测试节点，再运行：

```powershell
& .\.venv\Scripts\python.exe -X utf8 .\scripts\verify_graph_delete.py --course-id $courseId --name '删除验证专用节点' --before .\runtime\delete-before.json --out .\runtime\delete-after.json
```

也可以用资料表中的完整节点编号替换 `--name ...`，改成 `--node-id ...`。前后两次必须使用相同课程和选择条件。

## 结果怎么看

- `reviewed_graph.matching_nodes`：网页实际使用的云端审核图谱中，还有几个匹配节点。
- `native_graph.matching_nodes`：Neo4j 原始提取图谱中，还有几个匹配节点。
- `revision`：审核图谱版本号。
- `deletion_confirmed_in_reviewed_graph`：提供 `--before` 时，核对“删除前存在、删除后不存在且版本已增加”。
- `references_to_requested_id`：按编号检查时，审核图谱中是否仍有关系引用该编号。

当前系统的编辑写入 `Course.review_graph_json` 审核快照。原始提取的 Concept 节点和原始关系保留，因此可能出现“审核图谱 0，原始节点 1”。脚本会明确报告这个区别。

没有删除前记录时，“未找到”只能说明当前不存在，不能证明它曾存在。查询报错、连接超时、课程或快照缺失也不能当作删除成功。

## 持续查看

在上述命令加上 `--watch 5`，每 5 秒查询一次，按 Ctrl+C 停止。
连接配置默认读取项目 `config/local.env`。也可以用 `--env '配置文件完整路径'` 指定另一份配置，兼容 `key(2).env` 中带 `_2` 的字段。

## 已做的真实只读检查

交付前检查了“磁极”：审核快照中 1 个，原始 Neo4j 节点 1 个，审核版本为 3；每一层各关联 4 条关系。
这份结果保存在 `磁极_删除前验证.json`。检查没有删除这个节点，报告也不包含密码。
