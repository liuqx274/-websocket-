from components.logger import get_logger  # 导入日志模块
from typing import Callable, Dict, Any

logging = get_logger()


async def DeleteModel(payload: Dict[str, Any]) -> Dict[str, Any]:
    logging.info("  --- 调用删除模型功能  ---  ")
    # 这里是你模型训练的逻辑
    return {"score": 0.95}
