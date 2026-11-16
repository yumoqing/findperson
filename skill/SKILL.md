---
name: findperson-repo
version: 1.0.0
description: FindPerson
trigger_conditions:
  - 需要理解、修改或排障 findperson 的代码
  - 在 findperson 里定位入口、数据表、对外契约或构建方式
---

# findperson 仓库技能

> 由仓库实际内容生成的骨架（README/目录/模型定义/契约/构建脚本逐项取证）。新增判断与坑位请直接在下面对应小节补充。

## 定位

**FindPerson**

a module to save and find face in image see the requirements.txt

- 远端：`https://github.com/yumoqing/findperson.git`
- 默认分支：`main`（共 5 个提交）
- 最近提交：2026-11-03 `777fddc` chore: 忽略构建/测试临时文件
- 规模：14 个文件；Python 9，JSON 1

## 目录结构

- `findperson/` — Python 包源码（模块本体）（8 个文件）
- `app/` — 应用级初始化 / 菜单与页面装配（1 个文件）
- `conf/` — 配置（`config.json`；密码为 AES 密文）（1 个文件）
- 根文件：`.gitignore`、`README.md`、`pyproject.toml`、`setup.cfg`

## 注册入口

- `findperson/init.py` → `load_findperson()`

## 代码入口（顶层 Python）

- `findperson/image_imbedding.py`
- `findperson/imageface.py` — implements image to faces and to vector and save the face's vector to vector database
- `findperson/init.py`
- `findperson/milvus_utils.py`
- `findperson/utils_clip.py`
- `findperson/vectordb.py`
- `findperson/version.py`

## 构建与运行

- `pyproject.toml`：`?` v?，依赖 0 项
- Python 包 `findperson/`：`__init__.py`、`image_imbedding.py`、`imageface.py`、`init.py`、`milvus_utils.py`、`utils_clip.py`、`vectordb.py`、`version.py`

## README 要点与注意（摘自 `README.md`）

- 章节：dependent、installation

## 建议阅读顺序

1. `README.md`
2. `findperson/init.py`

