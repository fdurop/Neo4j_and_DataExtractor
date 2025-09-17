#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目环境配置脚本
自动安装所有必要的依赖包
"""

import subprocess
import sys


def install_package(package_name, version=None):
    """安装指定的包"""
    if version:
        package_spec = f"{package_name}=={version}"
    else:
        package_spec = package_name

    print(f"正在安装 {package_spec}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"✓ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package_name} 安装失败: {e}")
        return False


def install_spacy_model():
    """安装spaCy中文模型"""
    print("正在安装 spaCy 中文模型...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "zh_core_web_sm"])
        print("✓ spaCy 中文模型安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ spaCy 中文模型安装失败: {e}")
        return False


def main():
    print("=" * 50)
    print("开始配置项目环境...")
    print("=" * 50)

    # 依赖包列表（含版本号）
    packages = {
        "requests": "2.32.4",
        "torch": "2.8.0",
        "torchaudio": "2.8.0",
        "torchvision": "0.23.0",
        "transformers": "4.55.2",
        "spacy": "3.8.7",
        "pandas": "2.3.1",
        "chardet": "5.2.0",
        "neo4j": "5.28.2",
        "sentence-transformers": "5.1.0",
        "spacy-legacy": "3.0.12",
        "spacy-loggers": "1.0.5",
        "spacy_pkuseg": "1.0.1"
    }

    failed_packages = []
    for package, version in packages.items():
        if not install_package(package, version):
            failed_packages.append(f"{package}=={version}")

    # 安装 spaCy 中文模型
    if not install_spacy_model():
        print("警告: spaCy 中文模型安装失败，可能影响分词和实体识别功能")

    # 检查 CUDA 可用性
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用，检测到 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ 使用 CPU 模式运行")
    except ImportError:
        print("✗ PyTorch 导入失败")

    # 安装结果总结
    print("\n" + "=" * 50)
    print("环境配置完成！")
    print("=" * 50)

    if failed_packages:
        print("以下包安装失败：")
        for p in failed_packages:
            print(" -", p)
        print("请手动安装这些包或检查网络连接")
    else:
        print("所有依赖包安装成功！")

    print("\n下一步操作：")
    print("1. 确认 Neo4j 数据库已经安装并启动")
    print("2. 运行 'python main.py' 启动问答助手")
    print("3. 根据提示输入问题，测试问答功能")


if __name__ == "__main__":
    main()
