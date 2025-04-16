from components.logger import get_logger  # 导入日志模块
from components.exceptions import SelfError  # 导入异常信息模块
from typing import Callable, Dict, Any
import os
import json

logging = get_logger()


async def ModifyModelInformation(payload: Dict[str, Any]) -> Dict[str, Any]:
    logging.debug("  --- 调用修改模型信息功能  ---  ")

    # region 修改模型信息的逻辑
    base_path = os.path.join("model", "Saved_models")
    model_path = os.path.join(base_path, payload.get("oldName", None))
    model_info_path = os.path.join(model_path, "model_info.json")

    if os.path.isdir(base_path):                 # model/Saved_models/    保存所有模型信息的文件夹
        if os.path.isdir(model_path):            # model/Saved_models/model_01_info  目标模型信息文件夹
            if os.path.isfile(model_info_path):  # model/Saved_models/model_01_info/model_info.json 模型信息文件
                """ 至此,找到了所有需要修改的位置 """
                try:
                    # 1: 读出json文件内容
                    with open(model_info_path, "r", encoding="utf-8") as f:
                        model_info = json.load(f)
                    # 2: 修改
                    model_info["model_name"] = payload.get("newName")
                    model_info["model_save_info"]["save_location"] = "Saved_models/" + payload.get("newName") + "/"
                    model_info["model_save_info"]["notes"] = payload.get("remark")
                    # 3: 写回文件
                    with open(model_info_path, "w", encoding="utf-8") as f:
                        json.dump(model_info, f, ensure_ascii=False, indent=2)
                    # 4: 修改文件夹
                    os.rename(model_path, os.path.join(base_path, payload.get("newName", "无名称")))
                except Exception as e:
                    raise SelfError(f"修改过程中出错")
            else:
                raise SelfError(f"目标模型信息文件不存在：{model_info_path}")
        else:
            raise SelfError(f"目标模型文件夹不存在：{model_path}")
    else:
        raise SelfError(f"存放已保存模型的文件夹没有找到：{base_path}")

    # endregion

    logging.debug(f"模型信息文件夹: {payload.get('oldName')} 改为: {payload.get('newName')}")
    return {
        "oldName": payload.get("oldName"),
        "newName": payload.get("newName"),
        "remark" : payload.get("remark")
    }
