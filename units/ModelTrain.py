import json
import os
import pickle
import shutil
import time
from typing import Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from components.logger import get_logger  # 导入日志模块

logging = get_logger()


async def ModelTrain(payload: Dict[str, Any]) -> Dict[str, Any]:
    logging.debug("  --- 调用模型训练功能  ---  ")
    # region 模型训练的逻辑
    print(f"接收到的数据为:")
    print(payload)

    # region 解析 payload
    try:
        data = payload["data"]
        target = payload["target"]
        algorithm = payload["algorithm"]
        guidance = payload.get("guidance", {})
        use_existing = payload.get("useExistingModel", False)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, '..', 'model', 'temp_model')
        output_path = os.path.abspath(output_path)  # 转为绝对路径，保险
    except Exception as e:
        print("解析 payload 部分出现问题")
    # endregion

    # region 数据转换与清洗
    try:
        df = pd.DataFrame(data=data[1:], columns=data[0])
        # df = df.dropna()  # 简单删除缺失值
        df = df.drop(columns=["牌号"])
        # 强制所有列转为数值
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        # print(f"看一眼数据清洗后的数据样子")
        # print(df)
        X = df.drop(columns=[target])
        y = df[target]
    except Exception as e:
        print("数据转换与清洗")
    # endregion

    # region 相关性分析：删除与因变量相关性过低的列
    try:
        corr_df = df.corr(numeric_only=True)
        if target not in corr_df.columns:
            raise ValueError(f"目标列 '{target}' 不在可计算相关性的列中，可能不是数值类型。")

        corr_series = corr_df[target].drop(labels=[target])
        corr_sorted = corr_series.abs().sort_values(ascending=False)

        # print("\n特征与因变量的相关性（降序）:")
        # print(corr_sorted)

        threshold = 0.1
        dropped_features = corr_sorted[corr_sorted < threshold].index.tolist()
        # print(f"\n剔除以下相关性小于 {threshold} 的特征：{dropped_features}")

        X = X.drop(columns=dropped_features)

    except Exception as e:
        print(f"相关性分析出错: {e}")
    # endregion

    # region 划分训练集/测试集
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        # print(type(X_train))
        # print(type(X_test))
        # print(type(y_train))
        # print(type(y_test))

        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)
        y_train_1 = y_train.values
        y_test_1 = y_test.values
        y_train_2 = y_train_1.reshape(-1, 1)
        y_test_2 = y_test_1.reshape(-1, 1)
        # print(y_test)
        # print(y_test_1)
        # print(y_test_2)

        y_train_scaled = y_scaler.fit_transform(y_train_2)
        y_test_scaled = y_scaler.transform(y_test_2)

        # print(y_train_scaled)
        # print(y_test_scaled)
    except Exception as e:
        print(f"划分训练集/测试集 -- {e}")
    # endregion

    # region 创建输出目录
    try:
        model_dir = os.path.join(output_path, 'model_01')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(os.path.join(model_dir, 'data'))
    except:
        print("创建输出目录")
    # endregion

    # region 训练或加载模型
    try:
        if use_existing:
            # : 继续训练已有模型的逻辑（暂未实现）
            model = None
        else:
            start_time = time.time()
            if algorithm in ['随机森林', 'RandomForest']:
                # 随机森林回归 n_estimators:森林中多少棵树  random_state:随机数种子
                model = RandomForestRegressor(
                    n_estimators=int(guidance.get('n_estimators', 100)),
                    random_state=int(guidance.get('random_state', 42))
                )
                model.fit(X_train_scaled, y_train_scaled.ravel())

            elif algorithm in ['SVR']:
                # 支持向量回归
                '''
                    kernel 核函数 :  'linear'：线性核；'poly'：多项式核；'rbf'：径向基函数（默认、最常用）；'sigmoid'：类似神经网络中的 sigmoid 激活
                    C 惩罚系数 0.1， 1， 10， 100， 1000
                    epsilon 容忍范围  0.01、0.1、0.5
                '''
                model = SVR(
                    kernel=guidance.get('kernel', 'rbf'),
                    C=float(guidance.get('C', 10)),
                    epsilon=float(guidance.get('epsilon', 0.01))
                )

                model.fit(X_train_scaled, y_train_scaled.ravel())

            elif algorithm in ['XGBoost', 'xgboost']:
                # XGBoost 回归
                # n_estimators 弱学习器的数据，数量越多拟合能力越强 [50, 100, 200, 500]
                # learning_rate 学习率，每棵树（学习器）对最终结果的贡献 越小越稳定，收敛速度越慢 [0.01, 0.05, 0.1, 0.2]
                model = XGBRegressor(
                    n_estimators=int(guidance.get('XGBoost_n_estimators', 100)),
                    learning_rate=float(guidance.get('XGBoost_learning_rate', 0.1))
                )
                model.fit(X_train_scaled, y_train_scaled.ravel())

            elif algorithm in ['神经网络', 'NeuralNetwork']:
                # 自定义 PyTorch 神经网络
                '''
                    可选参数说明：
                    activationFunction：'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU'
                    optimizer：'adam'（推荐）, 'sgd'
                    initialLearningRate：初始学习率，如 0.01, 0.001
                    learning_rate_decay：L2 正则，如 0~0.01，对应 weight_decay
                    batch_size：训练每批数据量，如 16, 32, 64
                    epochs：训练轮数，如 10, 50, 100
                    dropout：Dropout 概率（0~1），如 0.5
                    scheduler_step_size：每多少轮衰减一次学习率
                    scheduler_gamma：每次衰减多少倍（如 0.1 表示减为十分之一）
                    validation_split：验证集比例，如 0.2 表示 20% 数据作验证
                    early_stopping_rounds：若 val_loss 连续 N 轮不下降则提前停止
                '''

                input_dim = X_train_scaled.shape[1]
                dropout_rate = float(guidance.get('nn_dropout', 0.0))
                activation = getattr(nn, guidance.get('nn_activationFunction', 'ReLU'))

                # 网络结构：两层 + dropout（可选）
                net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    activation(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 32),
                    activation(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(32, 1)
                )

                # 优化器与调度器
                opt_name = guidance.get('nn_optimizer', 'Adam').lower()
                lr = float(guidance.get('nn_initialLearningRate', 0.001))
                decay = float(guidance.get('nn_learning_rate_decay', 0))

                optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=decay) if opt_name == 'sgd' \
                    else optim.Adam(net.parameters(), lr=lr, weight_decay=decay)

                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(guidance.get('nn_scheduler_step_size', 10)),
                    gamma=float(guidance.get('nn_scheduler_gamma', 0.1))
                )

                criterion = nn.MSELoss()

                # 数据准备
                val_split = float(guidance.get('nn_validation_split', 0.2))
                val_size = int(len(X_train_scaled) * val_split)
                if val_size > 0:
                    X_val = X_train_scaled[:val_size]
                    y_val = y_train_scaled[:val_size]
                    X_train_scaled = X_train_scaled[val_size:]
                    y_train_scaled = y_train_scaled[val_size:]
                else:
                    X_val = y_val = None

                # 转 tensor
                X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)

                y_train_t = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)

                dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)

                loader = torch.utils.data.DataLoader(dataset, batch_size=int(guidance.get('nn_batch_size', 32)),
                                                     shuffle=True)

                # Early stopping 设置
                early_stop_rounds = int(guidance.get('nn_early_stopping_rounds', 0))
                best_val_loss = float('inf')
                no_improve_epochs = 0

                # 开始训练
                for epoch in range(int(guidance.get('nn_epochs', 10))):
                    net.train()

                    for X_batch, y_batch in loader:
                        optimizer.zero_grad()
                        pred = net(X_batch)
                        loss = criterion(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    # 验证
                    if val_size > 0:
                        net.eval()

                        with torch.no_grad():

                            X_val_t = torch.tensor(X_val, dtype=torch.float32)
                            y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
                            val_pred = net(X_val_t)
                            val_loss = float(criterion(val_pred, y_val_t).item())
                            print(
                                f"第 {epoch + 1} 轮训练，本轮结束后损失函数为: {val_loss}, 当前最好轮次损失函数为: {best_val_loss}")

                        # Early stopping 判断
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            no_improve_epochs = 0
                        else:
                            no_improve_epochs += 1
                            if 0 < early_stop_rounds <= no_improve_epochs:
                                print(f"早停触发：第 {epoch + 1} 轮，val_loss 已连续 {no_improve_epochs} 次未提升")
                                break

                # 最终预测
                net.eval()
                with torch.no_grad():
                    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
                    y_pred_scaled = net(X_test_t).numpy().flatten()
                    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

                model = net  # 保存模型对象

            elif algorithm in ['MLP']:
                # PyTorch 多层感知机 (单隐藏层)
                input_dim = X_train_scaled.shape[1]
                activation_name = guidance.get('mlp_activationFunction', 'ReLU')
                activation = getattr(nn, activation_name)
                dropout_rate = float(guidance.get('mlp_dropout', 0))
                hidden_size = int(guidance.get('mlp_hidden_layer_size', 64))
                # 构建模型（单隐藏层）
                net = nn.Sequential(
                    nn.Linear(input_dim, hidden_size),
                    activation(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_size, 1)
                )

                # 优化器
                opt_name = guidance.get('mlp_optimizer', 'Adam').lower()
                lr = float(guidance.get('mlp_initialLearningRate', 0.001))
                decay = float(guidance.get('mlp_learning_rate_decay', 0))
                optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=decay) \
                    if opt_name == 'sgd' else optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
                # 学习率调度器
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(guidance.get('mlp_scheduler_step_size', 10)),
                    gamma=float(guidance.get('mlp_scheduler_gamma', 0.1))
                )
                # 损失函数
                criterion = nn.MSELoss()
                # 数据准备
                val_split = float(guidance.get('mlp_validation_split', 0.2))
                X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train_scaled, test_size=val_split,
                                                            random_state=42)
                X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
                y_tr_t = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32)
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
                train_dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=int(guidance.get('mlp_batch_size', 32)), shuffle=True
                )
                # 早停
                early_stop_rounds = int(guidance.get('mlp_early_stopping_rounds', 0))
                best_val_loss = float('inf')
                no_improve_epochs = 0
                # 训练
                for epoch in range(int(guidance.get('mlp_epochs', 100))):
                    net.train()
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        pred = net(X_batch)
                        loss = criterion(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                    # 验证集评估
                    net.eval()
                    with torch.no_grad():
                        val_pred = net(X_val_t)
                        val_loss = criterion(val_pred, y_val_t).item()
                        print(
                            f"第 {epoch + 1} 轮训练，本轮结束后损失函数为: {val_loss}, 当前最好轮次损失函数为: {best_val_loss}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_epochs = 0
                        best_model_state = net.state_dict()
                    else:
                        no_improve_epochs += 1
                        if 0 < early_stop_rounds <= no_improve_epochs:
                            print(f"早停触发：第 {epoch + 1} 轮，val_loss 已连续 {no_improve_epochs} 次未提升")
                            break
                # 加载最佳模型
                net.load_state_dict(best_model_state)
                # 测试集预测
                net.eval()
                with torch.no_grad():
                    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
                    y_pred_scaled = net(X_test_t).numpy().flatten()
                    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
                model = net  # 保存最终模型

            # 计算训练时间
            train_time = time.time() - start_time
            # 如果用 sklearn 或 XGBoost 模型，上面未计算预测结果，需要单独预测
            if algorithm in ['SVR', 'XGBoost', '随机森林']:
                y_pred_scaled = model.predict(X_test_scaled)  # 一维数组
                y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))  # 转成二维后反归一化
            # 计算评价指标
            test_score = float(r2_score(y_test, y_pred))

            # 训练集预测并计算 R²
            if algorithm in ['SVR', 'XGBoost', '随机森林']:
                y_train_pred_scaled = model.predict(X_train_scaled)
                y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1))
            else:
                with torch.no_grad():
                    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
                    y_train_pred_scaled = model(X_train_t).numpy().flatten()
                    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1))

            train_score = float(r2_score(y_train, y_train_pred))
            print(f"训练集 R² 为: {train_score}")
            print(f"测试集 R² 为: {test_score}")

            # 保存模型文件
            with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 保存模型信息
            info = {
                "model_name": "model_01",
                "algorithm": algorithm,
                "train_R2": train_score,
                "test_R2": test_score,
                "parameters": guidance,
                "train_time": train_time if not use_existing else None,
                "data_size": len(df),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "correlation_ranking": corr_sorted.to_dict(),
                "dropped_features": dropped_features,

                "notes": "测试模型的备注"
            }
            with open(os.path.join(model_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=4)

            # 保存训练和测试数据
            train_df = X_train.copy()
            train_df[target] = y_train
            test_df = X_test.copy()
            test_df[target] = y_test
            train_df.to_excel(os.path.join(model_dir, 'data', 'train_data.xlsx'), index=False)
            test_df.to_excel(os.path.join(model_dir, 'data', 'test_data.xlsx'), index=False)

    except Exception as e:
        print(f"训练或加载模型-- {e}")
    # endregion

    return {"train_R2": train_score, "test_R2": test_score}
