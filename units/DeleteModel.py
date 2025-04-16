import shutil
from components.logger import get_logger  # 导入日志模块
from components.exceptions import SelfError  # 导入异常信息模块
from typing import Callable, Dict, Any
import os
import json

logging = get_logger()


async def DeleteModel(payload: Dict[str, Any]) -> Dict[str, Any]:
    logging.debug("  --- 调用删除模型功能  ---  ")

    # region 这里是删除模型的逻辑
    try:
        # 解析传过来的字典，检查要删除模型是否存在
        model_name = payload.get("name")
        if not model_name:
            raise SelfError("模型名称不能为空")
        # 拼接目标模型文件夹连接
        model_path = os.path.join("model", "Saved_models", model_name)
        # 检查目标模型文件夹是否存在
        if not os.path.exists(model_path):
            raise SelfError(f"目标模型文件夹不存在：{model_path}")

        shutil.rmtree(model_path)  # 递归删除文件夹，即：删除非空文件夹

    except AttributeError:
        raise SelfError("删除模型失败：payload 不是一个字典，无法调用 .get 方法", exc_info=True)

    except ValueError as ve:
        raise SelfError(f"删除模型失败：{ve}", exc_info=True)

    except Exception as e:
        raise SelfError(f"删除模型失败：发生未知异常 - {e}", exc_info=True)

    # endregion

    return {}
