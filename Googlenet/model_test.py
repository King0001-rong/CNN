import torch
import torch.utils.data as Data
from torchvision import transforms
from model import GoogLeNet, Inception
from torchvision.datasets import ImageFolder
# 读取图片的库
from PIL  import  Image


def test_data_process():
    ROOT_TRAIN = r'data\test'
    normalize = transforms.Normalize([0.162, 0.151, 0.138],
                                     [0.058, 0.052, 0.048])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)



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
    model = GoogLeNet(Inception)
    # 测试模型
    model.load_state_dict(torch.load('best_model.pth'))

    ## 模型测试准确率计算
    # 加载测试数据集
    # test_dataloader = test_data_process()
    # 加载模型测试的函数
    # test_model_process(model, test_dataloader)

    ## 遍历每张测试集图片，并打印预测结果
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    classes = ['猫', '狗']
    # with torch.no_grad():
    #     for b_x,b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         #设置模型为验证模型
    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim=1)
    #         result = pre_lab.item()
    #         label = b_y.item()
    #         print("预测的标签为：",classes[result], "真实值：", classes[label])



    ## 实际需求的图片预测
    image = Image.open(r'img_1.png')
    normalize = transforms.Normalize([0.162, 0.151, 0.138],
                                     [0.058, 0.052, 0.048])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image)

    # 添加批次维度
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
    print("预测的标签为：",classes[result])
