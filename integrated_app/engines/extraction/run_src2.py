"""API 多模态抽取流程入口。"""

import argparse
import json
import sys
import os
from pathlib import Path

# 统一以 extraction-main 为运行根目录。
extraction_root = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(extraction_root)
os.chdir(workspace_root)
print("工作区根目录: .")
print("抽取项目目录: extraction")

# 添加必要的路径
src2_path = os.path.join(extraction_root, 'src2')
sys.path.insert(0, src2_path)
sys.path.insert(0, extraction_root)

# 验证关键路径
print("\n检查关键路径:")
print(f"  src2目录: {os.path.exists(src2_path)} - extraction/src2")

input_path = os.path.join("extraction", "input")
print(f"  输入目录: {os.path.exists(input_path)} - {input_path}")
config_path = os.path.join("extraction", "config", "vlm_api.yaml")
print(f"  API配置: {os.path.exists(config_path)} - {config_path}")

print("\n当前工作目录: extraction-main")

# 导入并运行主程序
def parse_args():
    parser = argparse.ArgumentParser(description="运行 API 多模态抽取流程")
    parser.add_argument("--job-manifest", help="前端生成的抽取任务 JSON")
    parser.add_argument("--input-file", help="只处理一个 PDF/PPTX 文件")
    parser.add_argument("--input-dir", help="处理指定目录中的 PDF/PPTX")
    parser.add_argument("--output-dir", help="显式指定本次任务输出目录")
    parser.add_argument("--user-id")
    parser.add_argument("--username")
    parser.add_argument("--course-id")
    parser.add_argument("--course-name")
    parser.add_argument("--job-id")
    return parser.parse_args()


def _load_job_manifest(path):
    if not path:
        return {}
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = Path(workspace_root) / manifest_path
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("job manifest 必须是 JSON 对象")
    return data


def _argument_or_manifest(value, manifest, key):
    return value if value not in {None, ""} else manifest.get(key)


def run():
    args = parse_args()
    manifest = _load_job_manifest(args.job_manifest)
    values = {
        key: _argument_or_manifest(getattr(args, key), manifest, key)
        for key in (
            "input_file", "input_dir", "output_dir", "user_id", "username",
            "course_id", "course_name", "job_id",
        )
    }
    if not values["user_id"] and values["username"]:
        values["user_id"] = values["username"]

    print("\n" + "=" * 60)
    print("开始运行 API 多模态抽取流程")
    print("=" * 60 + "\n")

    from multimodal_preprocessor import main
    main(**values)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"\n[ERROR] 运行失败: {e}")
        import traceback
        traceback.print_exc()
