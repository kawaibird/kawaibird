# Continuous XOR 예제로 이해하는 기본구조
# 자세한 것은 다음 링크를 참조하자.
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html

import torch

'''
    1) 신경망
'''

import torch.nn as nn


class MyModule(nn.Module):
    # Pytorch에선 신경망 자체가 하나의 모듈로 이미 만들어져 있다.
    # 즉 학습을 위해 필요한 요소들을 정의하고 그것들을 계산할 방법을 정해서 모듈에 대입하기만 하면 된다.
    # __init__에서 필요한 요소들(activation function, 변수 등등...)을 초기화하고 forward에선 그것들을 어떻게 계산할 것인지 logic을 구성한다.
    # 아래 SimpleClassfier에서 자세히 보자.
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class SimpleClassfier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = nn.Linear(x, self.input_dim, self.hidden_dim)
        x = nn.Tanh(x)
        x = nn.Linear(x, self.hidden_dim, self.output_dim)

        return x


'''
    2) 데이터
'''

# Pytorch는 훈련과 시험을 효율적으로 하기 위한 데이터에 관련한 여러 툴을 제공한다.
# torch.utils.data 패키지가 바로 그것이다. 잘 만들어진거 아껴서 뭐하냐 냉큼 주워서 사용하자.

import torch.utils.data as data


# data 패키지는 두 개의 인터페이스를 기반으로 작동한다.
# 첫 번째는 'Dataset'
# 두 번째는 'DataLoader'
# Dataset 클래스는 데이터에 일관적으로 접근하게 해준다.
# DataLoader 클래스는 학습이 진행되는 동안 데이터를 읽어들이고 쌓는 일괄처리를 효율적으로 해준다.
# 이 역시 아래에서 상세히 보자.


# 2-1) Dataset
# Dataset 클래스는 항상 두 개의 함수를 정의해야 한다.
# __getitem__과, __len__이다.
# getitem은 i번째 데이터를 얻는 함수
# len은 데이터 전체의 크기를 얻는 함수다.

class XORDataset(data.Dataset):
    """
    XOR 데이터를 만드는 클래스
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.get_xor()

    def get_xor(self):
        data_ = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data_.sum(dim=1) == 1).to(torch.long)

        self.data_ = data_
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        data_point = self.data_[item]
        data_label = self.label[item]
        return data_point, data_label


class ContinuousXORDataset(data.Dataset):
    """
       XOR 데이터를 만드는 클래스인데 분산을 끼얹은
    """

    def __init__(self, size, std=0.1):
        super().__init__()
        self.size = size
        self.std = std
        self.get_xor()

    def get_xor(self):
        data_ = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data_.sum(dim=1) == 1).to(torch.long)
        data_ += self.std * torch.randn(data_.shape)

        self.data_ = data_
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        data_point = self.data_[item]
        data_label = self.label[item]
        return data_point, data_label


# 2-2) DataLoader 클래스
# 자동 batching, 데이터 병렬 로딩 등 다양하게 지원해준다.

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = XORDataset(10000)
    data_loader = data.DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=False)
    model = SimpleClassfier(2, 4, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_module = nn.BCEWithLogitsLoss()

    epoches = 100
    for epoch in range(epoches):
        for x, y in data_loader:
            x=x.to(device)
            y=y.to(device)

            output = model(x)
            loss = loss_module(output, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print()
