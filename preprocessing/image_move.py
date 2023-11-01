import os
import shutil

# 定义文件夹路径
bboxes_masks_path = '../dataset/bboxes_mask_label'
accident_path = os.path.join(bboxes_masks_path, 'accident')
no_accident_path = os.path.join(bboxes_masks_path, 'no_accident')
bboxes_frame_path = '../dataset/bboxes_frame_label'

# 获取两个子文件夹中的文件名并去除"_mask"后缀
accident_files = [file.replace('_mask', '') for file in os.listdir(accident_path)]
no_accident_files = [file.replace('_mask', '') for file in os.listdir(no_accident_path)]

# 确保在bboxes_frame文件夹中创建accident和no_accident子文件夹
os.makedirs(os.path.join(bboxes_frame_path, 'accident'), exist_ok=True)
os.makedirs(os.path.join(bboxes_frame_path, 'no_accident'), exist_ok=True)

# 遍历bboxes_frame文件夹中的文件
for file in os.listdir(bboxes_frame_path):
    # 检查文件是否在accident子文件夹中
    if file in accident_files:
        # 将文件移动到相应的子文件夹中
        shutil.move(os.path.join(bboxes_frame_path, file), os.path.join(bboxes_frame_path, 'accident', file))
    # 检查文件是否在no_accident子文件夹中
    elif file in no_accident_files:
        # 将文件移动到相应的子文件夹中
        shutil.move(os.path.join(bboxes_frame_path, file), os.path.join(bboxes_frame_path, 'no_accident', file))
    else:
        print(f"Warning: {file} not found in bboxes_masks folder.")
