import os
from PIL import Image
import torch
from torchvision import transforms
from model import ResNet18, Residual
import time

def batch_predict(model, folder_path, transform, classes, device):
    """
    批量预测指定文件夹中的图片
    :param model: 加载好的模型
    :param folder_path: 图片所在文件夹路径
    :param transform: 图像预处理方法
    :param classes: 类别名称列表
    :param device: 使用设备（'cuda' 或 'cpu'）
    :return: 返回预测结果列表 [(图片路径, 预测类别), ...]
    """
    model = model.to(device)
    model.eval()

    results = []

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.jfif','.webp')

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(supported_extensions):
            img_path = os.path.join(folder_path, img_name)

            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(image_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    pred_class = classes[pred_idx]

                # 构造新文件名（保留原始扩展名）
                timestamp = str(int(time.time()))  # 添加时间戳
                file_root, file_ext = os.path.splitext(img_name)
                new_file_name = f"{pred_class}_{timestamp}_{file_root}{file_ext}"

                # 重命名文件
                new_img_path = os.path.join(folder_path, new_file_name)
                os.rename(img_path, new_img_path)

                results.append((img_name, new_file_name))

            except Exception as e:
                print(f"无法处理图片 {img_path}：{e}")

    return results


if __name__ == "__main__":
    # 定义类别
    classes = ['Mask','No Mask']

    # 图像预处理方式（必须与训练时一致）
    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451],
                                     [0.0736155,  0.06216329, 0.05930814])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # 加载模型
    model = ResNet18(Residual)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    # 设定设备
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 批量预测设置
    test_folder = r'test_images'  # 替换为你自己的测试图片文件夹路径
    predictions = batch_predict(model, test_folder, test_transform, classes, device)

    # 输出结果
    print("批量预测结果：")
    for img_name, pred_class in predictions:
        print(f"{img_name} -> 预测类别：{pred_class}")
