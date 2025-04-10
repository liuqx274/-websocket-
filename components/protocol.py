from typing import Optional, Literal, Dict, Any, TypedDict

# 所有操作类型
OperationType = Literal[
    "ModelTrain", "ModelSave", "FindModelDetails",
    "ModifyModelInformation", "DeleteModel",
    "ModelPredict", "OptimizationIssues"
]


# ✅ 接收到的客户端请求数据结构
class ClientRequestData(TypedDict, total=False):
    operation: OperationType
    ModelTrain: Optional[Dict[str, Any]]
    ModelSave: Optional[Dict[str, Any]]
    FindModelDetails: Optional[Dict[str, Any]]
    ModifyModelInformation: Optional[Dict[str, Any]]
    DeleteModel: Optional[Dict[str, Any]]
    ModelPredict: Optional[Dict[str, Any]]
    OptimizationIssues: Optional[Dict[str, Any]]


# ✅ 服务器的响应 result 字段（不同操作的返回数据）
class ResponseResult(TypedDict, total=False):
    ModelTrain: Any
    ModelSave: Any
    FindModelDetails: Any
    ModifyModelInformation: Any
    DeleteModel: Any
    ModelPredict: Any
    OptimizationIssues: Any
    message: str  # 可选的提示信息


# ✅ 返回给前端的数据结构
class ServerResponseData(TypedDict):
    operation: OperationType
    status: bool
    result: ResponseResult
