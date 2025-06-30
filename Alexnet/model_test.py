import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet


def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                              download=True)


    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_dataloader

def test_model_process(model, test_dataloader):
    # 将模型放入到模型设备中
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 初始化参数
    test_corrects = 0
    test_num = 0

    # 梯度置为0，仅进行前向传播得出结果
    with torch.no_grad():
        # 将一次数据放入到验证设备中
        for test_data_x,test_data_y in test_dataloader:
            # 将标签放入到验证设备中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据，输出为每个样本的预测值
            output = model(test_data_x)
            # 返回最大概率的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)
    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)


if __name__ == '__main__':
    # 获取模型
    model = AlexNet()
    # 测试模型
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载测试数据集
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)




    device ='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #设置模型为验证模型
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测的标签为：",classes[result], "真实值：", classes[label])




