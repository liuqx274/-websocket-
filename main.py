import asyncio  # python 的异步库
import os

import websockets  # websocket 库
import json
from components.logger import get_logger  # 导入日志模块
from components.protocol import ClientRequestData, ServerResponseData  # 导入数据结构
from components.handlers import handle_operation as handel  # 调控器

# 全局请求序号
request_id = 0

# 获取日志对象
logging = get_logger()


# 处理 WebSocket 连接
async def server(websocket):
    global request_id  # 使用全局变量
    logging.info("WebSocket 连接已建立")
    try:
        async for message in websocket:
            request_id += 1  # 每次接收到请求时递增序号
            logging.info(f"收到请求 #{request_id}: {message}")

            # 解析请求的操作及信息
            logging.debug(f"解析请求内容...")
            data: ClientRequestData = json.loads(message)
            operation = data.get("operation")

            # 执行对应操作，得出计算结果
            logging.debug(f"寻找相应处理模块...")
            response: ServerResponseData = await handel(operation, data)

            logging.debug(f"发送数据 - {response}")
            await websocket.send(json.dumps(response))

            logging.info(f"请求 #{request_id} 已解决")

    except websockets.exceptions.ConnectionClosed:
        logging.info("WebSocket 连接已关闭")


# 程序入口
async def main():
    logging.info("WebSocket 服务器启动中...")

    async with websockets.serve(server, "localhost", 5000):
        logging.info("服务启动在端口5000...")
        await asyncio.Future()  # 让服务器一直运行


asyncio.run(main())
