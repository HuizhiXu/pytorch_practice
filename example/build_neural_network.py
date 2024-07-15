import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# Define the neural network
class NeuralNetwork(nn.Module):
    """
    需要实现__init__, __len__, __getitem__方法
    """
    def __init__(self):
        """
        在__init__里面需要定义网络的结构
        """
        super(NeuralNetwork, self).__init__() 
        # 从第一维到最后一维的张量展平
        self.flatten = nn.Flatten()
        # 线性和RELU叠加 3层
        # nn.Linear(input_size, output_size)
        # nn.ReLU() 代表激活函数
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )       

    def forward(self, x):
        """
        依次调用网络的每一层
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    print("Model structure: ", model, "\n\n" )
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        # 打印linear_relu_stack.0 linear_relu_stack.2 linear_relu_stack.4 的参数
        # 因为relu层没有参数
   
    # summary(model, (1, 28, 28))s

    # Load the data
    X = torch.rand(1, 28, 28).to(device)
    logits = model(X)
    print(logits)
    # nn.Softmax(dim=1) 代表对第1维进行softmax 归一化，得到概率
    pred_probab = nn.Softmax(dim=1)(logits)
    # 算出概率的最大值的class index
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")



