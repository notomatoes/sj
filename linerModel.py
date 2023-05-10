"""
# File       : linerModel.py
# Time       ：2023/5/10 16:21
# Author     ：notomato
# Description：
# 
"""

import torch
import torch.nn as nn
from data import generate_batch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(7 * 3, 256)
        self.d1 = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.d2 = nn.Dropout(p=0.1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2 * 6)

    def forward(self, x):
        x = x.view(-1, 7 * 3)
        x = self.fc1(x)
        x = self.d1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.d2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x.view(-1, 2, 6)





if __name__ == '__main__':
    # testdata = generate_batch(6, 1)
    # km_data = torch.tensor([testdata[i][0] for i in range(len(testdata))], dtype=torch.float32)
    # zd_data = torch.tensor([testdata[i][1] for i in range(len(testdata))], dtype=torch.float32)
    #
    # print(km_data)
    # print(testdata[0])

    batch_size = 256
    num_epochs = 2000
    # 创建模型实例
    model = MyModel()
    model.train()
    # 数据
    data = generate_batch(6, 8196)
    km_data = torch.tensor([data[i][0] for i in range(len(data))], dtype=torch.float32)
    zd_data = torch.tensor([data[i][1] for i in range(len(data))], dtype=torch.float32)
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    # 定义损失函数
    criterion = nn.MSELoss()

    num_batches = len(km_data) // batch_size
    print('----------train-----------')
    for epoch in range(num_epochs):
        for i in range(num_batches):
            # 获取当前小批量数据
            km_batch = km_data[i * batch_size: (i + 1) * batch_size]
            zd_batch = zd_data[i * batch_size: (i + 1) * batch_size]

            # 向模型输入数据，并计算输出
            output = model(zd_batch)

            # 计算损失
            loss = criterion(output, km_batch)

            # 将梯度清零，反向传播计算梯度，使用优化器更新模型参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印损失
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    print('-------------eval------------')
    model.eval()
    for _ in range(1):
        testdata = generate_batch(6, 1)
        km_data = torch.tensor([testdata[i][0] for i in range(len(testdata))], dtype=torch.float32)
        zd_data = torch.tensor([testdata[i][1] for i in range(len(testdata))], dtype=torch.float32)

        with torch.no_grad():
            test_output = model(zd_data)

        print(km_data)
        # print(zd_data)
        print(test_output)
        test_loss = criterion(test_output, km_data)
        print(test_loss)
