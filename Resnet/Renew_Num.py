import os


def rename_images_sequentially(folder_path, supported_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.jfif','.webp')):
    """
    将指定文件夹中的图像按名称排序，并从1开始按顺序重命名。

    :param folder_path: 图像所在文件夹路径
    :param supported_extensions: 支持的图像扩展名
    """
    # 获取所有支持格式的文件
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]

    # 按文件名排序（假设文件已按预测类别命名，例如 "Mask_时间戳_xxx.png"）
    files.sort()

    # 从1开始重命名
    for idx, filename in enumerate(files, start=1):
        file_root, file_ext = os.path.splitext(filename)
        new_file_name = f"{idx}{file_ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_file_name)

        try:
            os.rename(old_path, new_path)
            print(f"重命名 {filename} -> {new_file_name}")
        except Exception as e:
            print(f"无法重命名文件 {filename}: {e}")


if __name__ == "__main__":
    # 示例：批量预测后对结果进行重命名
    test_folder = r'test_images'  # 替换为你的预测文件夹路径
    rename_images_sequentially(test_folder)
