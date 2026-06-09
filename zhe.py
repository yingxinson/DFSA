import cv2
import numpy as np
import random

def add_random_block_obstruction_to_video(
    input_video_path,
    output_video_path,
    block_count=5,  # 每帧随机遮挡块的数量
    block_size_range=(50, 150),  # 遮挡块的尺寸范围 (宽度, 高度)
    block_color=(0, 0, 0),  # 遮挡块颜色，默认黑色
):
    # 打开视频
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建视频写入对象
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 在帧中随机添加遮挡块
        for _ in range(block_count):
            # 随机选择遮挡块的宽度和高度
            block_width = random.randint(*block_size_range)
            block_height = random.randint(*block_size_range)

            # 随机选择遮挡块的位置（确保不超出图像边界）
            x = random.randint(0, width - block_width)
            y = random.randint(0, height - block_height)

            # 随机选择遮挡块颜色（如果需要可以更改）
            color = block_color  # 默认黑色，(0, 0, 0)

            # 在该区域填充遮挡块
            frame[y:y+block_height, x:x+block_width] = color

        # 写入处理后的帧
        out.write(frame)

    cap.release()
    out.release()
    print("处理完成，已保存为：", output_video_path)

if __name__ == "__main__":
    add_random_block_obstruction_to_video(
        input_video_path="./eval/lu/examples/videocrop5.mp4",
        output_video_path="./eval/lu/examples/videocrop5_blocks.mp4",
        block_count=1,  # 每帧10个遮挡块
        block_size_range=(40, 100),  # 遮挡块的尺寸范围
        block_color=(0, 0, 0)  # 黑色遮挡块
    )

