import cv2
import os


def video_to_frames(video_path, output_folder, fps=10):
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 获取视频文件名
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_count = 1
    frame_index = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (video_fps // fps) == 0:
            # 使用视频文件名和帧序号作为输出帧图片的文件名
            output_path = os.path.join(output_folder, f"{video_name}_{frame_index}.png")
            cv2.imwrite(output_path, frame)
            frame_index += 1

        frame_count += 1

    cap.release()
    print(f"Extracted frames from {video_name}")


def process_videos(videos_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(videos_folder):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 您可以根据需要添加其他视频格式
            video_path = os.path.join(videos_folder, filename)
            video_to_frames(video_path, output_folder)

if __name__ == "__main__":
    videos_folder = "../dataset/CCD_videos/"  # 请替换为您的视频文件夹路径
    output_folder = "../dataset/current_frame/"  # 请替换为您的输出文件夹路径

    process_videos(videos_folder, output_folder)