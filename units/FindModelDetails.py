from components.logger import get_logger  # 导入日志模块
from typing import Callable, Dict, Any, List
import os
import json

logging = get_logger()


async def FindModelDetails(payload: Dict[str, Any]) -> Dict[str, Any]:
    logging.debug("  --- 调用寻找模型信息的功能  ---  ")
    # region 查询模型信息的处理逻辑
    base_path = os.path.join("model", "Saved_models")  # 你的模型保存路径
    model_list: List[Dict[str, Any]] = []

    if not os.path.exists(base_path):
        logging.warning(f"路径不存在：{base_path}")
        return {"models": []}

    for folder_name in os.listdir(base_path):
        if folder_name == ".gitkeep":
            continue
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            info_path = os.path.join(folder_path, "model_info.json")
            if os.path.isfile(info_path):
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        model_info = json.load(f)
                        model_list.append(model_info)
                        # logging.info(f"读取模型信息成功：{info_path}")
                except Exception as e:
                    logging.error(f"读取模型信息失败：{info_path}, 错误：{e}")
            else:
                logging.warning(f"未找到 model_info.json 文件：{info_path}")
        else:
            logging.warning(f"报错: 检查到 {folder_path} 并非模型文件夹, 这很不合理，需要人工查询一下功能日志")

    # endregion

    return model_list
