from PIL import Image, ImageDraw
import os
import json

def read_bboxes(text_path):
    bboxes = {}

    with open(text_path, 'r') as files:
        for line in files:

            # 解析每行数据
            data = json.loads(line.strip())
            frame_idx = data["frame_idx"]

            bbox_id = list(data.keys())[1]
            bbox_coords = data[bbox_id]

            # 将bounding box数据存储到字典中
            if frame_idx not in bboxes:
                bboxes[frame_idx] = []

            bbox_data = {key: value for key, value in data.items() if key != "frame_idx"}
            bboxes[frame_idx].append(bbox_data)

        return bboxes

def fill_missing_frames(text_path, total_frames):
    # 读取文件中的所有帧信息
    bboxes = read_bboxes(text_path)

    # 填补缺失的帧信息
    for frame_idx in range(1, total_frames + 1):
        if frame_idx not in bboxes:
            # 寻找最近的帧信息
            nearest_frame_idx = None
            distance = float("inf")

            for available_frame_idx in bboxes:
                current_distance = abs(available_frame_idx - frame_idx)
                if current_distance < distance:
                    distance = current_distance
                    nearest_frame_idx = available_frame_idx

            # 如果找到了最近的帧，复制其信息
            if nearest_frame_idx is not None:
                bboxes[frame_idx] = bboxes[nearest_frame_idx]

    # 将填补后的帧信息写回文件
    with open(text_path, 'w') as file:
        for frame_idx, frame_data in sorted(bboxes.items()):
            if frame_data:  # 确保帧数据非空
                for bbox_data in frame_data:
                    data = {
                        "frame_idx": frame_idx,
                    }
                    data.update(bbox_data)
                    file.write(json.dumps(data) + "\n")

def draw_bboxes(frame_path, bboxes, frame_index, output_path):

    mask = Image.open(frame_path)
    # 创建一个画布，用于在图像上绘制边框
    draw = ImageDraw.Draw(mask)

    draw.rectangle([(0,  0), (1280, 720)], fill='black')

    # 遍历边框列表并绘制
    # for bbox_data in bboxes:
    # for bbox_id, bbox_coords in bboxes[frame_index]:
        # 将边框的坐标解压缩
    for coor in bboxes[frame_index]:

        coor_list = list(coor.values())
        x1, y1, x2, y2 = coor_list[0][0], coor_list[0][1], coor_list[0][2], coor_list[0][3]

        # 绘制边框
        draw.rectangle([(x1, y1), (x2, y2)], fill="white")

        mask.save(output_path)
        # mask.show()

    return

if __name__ == '__main__':

    org_text_path = "../dataset/bboxes_text/"
    org_frame_path = "../dataset/bboxes_frame/"

    for r, d, f in os.walk(org_text_path):
        for j in f:

            fill_missing_frames(org_text_path + j, 50)
            bboxes = read_bboxes(org_text_path + j)

            for r, d, f in os.walk(org_frame_path):
                for i in f:
                    if i[:6] == j[:6]:

                        frame_index = int(i[7:-4])
                        frame_path = org_frame_path + i

                        print(i[:-4], i[-4:])

                        output_path = "../dataset/bboxes_mask/" + i[:-4] + "_mask" + i[-4:]
                        draw_bboxes(frame_path, bboxes, frame_index, output_path)