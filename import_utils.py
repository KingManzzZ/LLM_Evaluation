"""
统一的导入工具模块
解决所有子目录下的模块在直接运行或包导入时的路径问题
"""

import sys
import os

def setup_path():
    """
    自动配置 sys.path，使得子目录下的脚本能正确导入上级目录的模块
    在模块/脚本开头调用此函数即可
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（当前文件上两级）
    root_dir = os.path.dirname(current_dir)

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

# 自动调用
setup_path()

