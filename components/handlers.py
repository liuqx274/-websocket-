"""
    处理函数：根据前端所发送的 操作数不同 来调用不同的功能函数，并将结果保存，递交给主函数
"""
from components.logger import get_logger  # 导入日志模块
from components.exceptions import SelfError  # 导入异常信息模块
import importlib
from components.protocol import ClientRequestData, ServerResponseData

global request_id  # 使用全局变量
logging = get_logger()


# operation 调度函数
async def handle_operation(operation: str, data: ClientRequestData) -> ServerResponseData:
    print('进入调度函数')
    response: ServerResponseData = {
        "operation": operation,
        "status": True,
        "result": {}
    }

    # 操作参数为空
    if is_null_or_empty(operation):
        response["status"] = False
        response["result"]["message"] = f"操作参数 operation 不可为空"
        logging.debug(f"操作参数为空，拒绝执行")
        return response

    # 操作参数不为空
    try:
        ''' 
            根据 operation 构造需要引入的文件名称 module_path
            lower() 将字符串的所有字符转变为小写
        '''
        module_path = f"units.{operation}"  # 不要加 `.py`
        print(f"要找寻的模块为--{module_path}")
        try:
            module = importlib.import_module(module_path)
            logging.debug(f"已找到对应的模块-{module_path}")
        except ModuleNotFoundError:
            response["status"] = False
            response["result"]["message"] = f"操作 {operation} 对应的模块 {module_path} 不存在"
            logging.warning(f"寻找模块 - {module_path} 失败，执行终止")
            return response

        # 如果找到该模块，获取该模块下的函数并调用
        function_name = operation  # 函数名与操作名相同
        operation_function = getattr(module, function_name, None)  # getattr: 从指定的文件 module 中找出 function_name 函数
        if callable(operation_function):  # 判断是不是真的找到了一个能执行的函数
            logging.debug(f"已找到对应的函数-{operation_function}")
            try:
                result = await operation_function(data.get(operation, {}))
                response["result"][operation] = result
            except SelfError as e:
                response["status"] = False
                response["result"]["message"] = f"{e.message}"
                return response

        else:
            response["status"] = False
            response["result"]["message"] = f"模块 {operation} 中未找到有效的函数"
            logging.warning(f"未在模块中找到一个可调用的功能-{operation_function}, 执行终止")

    except Exception as e:
        response["status"] = False
        response["result"]["message"] = f"操作 {operation} 执行出错:"
        logging.warning(f"执行计算的中出现错误而终止")

    return response


# 判断是否为null或空字符串
def is_null_or_empty(string):
    # 使用 is None 判断是否为null
    if string is None:
        return True
    # 使用 len() 判断是否为空字符串
    if len(string) == 0:
        return True
    # 使用 isspace() 判断是否全由空格组成
    if string.isspace():
        return True
    # 其他情况返回 False
    return False
