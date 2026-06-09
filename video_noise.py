import cv2
import numpy as np

def add_gaussian_noise_to_video(
    input_video_path,
    output_video_path,
    mean=0,
    sigma=15
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

        # 生成高斯噪声
        noise = np.random.normal(mean, sigma, frame.shape)

        # 加噪声（转成 float 防止溢出）
        noisy_frame = frame.astype(np.float32) + noise

        # 限制像素值范围
        noisy_frame = np.clip(noisy_frame, 0, 255)

        # 转回 uint8
        noisy_frame = noisy_frame.astype(np.uint8)

        # 写入视频
        out.write(noisy_frame)

    cap.release()
    out.release()
    print("处理完成，已保存为：", output_video_path)


if __name__ == "__main__":
    add_gaussian_noise_to_video(
        input_video_path="./eval/lu/examples/videocrop5.mp4 ",
        output_video_path="./eval/lu/examples/videocrop5_20.mp4",
        mean=0,
        sigma=10   # 三个档位5,10,20噪声强度，越大噪声越明显
    )
