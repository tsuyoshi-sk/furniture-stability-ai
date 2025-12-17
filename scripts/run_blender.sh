#!/bin/bash
# pyenvの影響を回避してBlenderを実行するスクリプト
# 使用法: ./run_blender.sh script.py [args...]

BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"
env -u PYTHONHOME -u PYTHONPATH "$BLENDER_PATH" --background --python "$@"
