# import cv2
# import dlib
# from moviepy.editor import VideoFileClip, AudioFileClip
#
# # 初始化dlib的人脸检测器和面部landmark预测器
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("D:/Python/team/DINet2/asserts/shape_predictor_68_face_landmarks.dat")  # 需从dlib官网下载
#
#
# def crop_face_centered(video_path, output_path):
#     # 先提取原始音频
#     try:
#         original_clip = VideoFileClip(video_path)
#         audio = original_clip.audio
#     except Exception as e:
#         print(f"音频提取失败: {str(e)}")
#         audio = None
#
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return
#
#     # 获取视频属性
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # 创建临时无声视频文件路径
#     temp_video = "temp_silent.mp4"
#
#     # 创建VideoWriter对象
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(temp_video, fourcc, fps, (416, 320))
#
#     prev_center = None  # 用于保存上一帧的面部中心位置
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 转换为灰度图像进行人脸检测
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray, 0)
#
#         if len(faces) > 0:
#             # 取最大的人脸（按面积计算）
#             face = max(faces, key=lambda rect: rect.width() * rect.height())
#             landmarks = predictor(gray, face)
#
#             # 计算68个特征点的中心点
#             sum_x = sum(landmarks.part(i).x for i in range(68))
#             sum_y = sum(landmarks.part(i).y for i in range(68))
#             center_x = int(sum_x / 68)
#             center_y = int(sum_y / 68)
#             prev_center = (center_x, center_y)
#         else:
#             if prev_center is None:
#                 continue  # 跳过没有面部且无历史数据的帧
#             center_x, center_y = prev_center  # 使用上一帧的位置
#
#         # 计算裁剪区域（416x320）
#         crop_width = 416
#         crop_height = 320
#         half_width = crop_width // 2
#         half_height = crop_height // 2
#
#         # 计算裁剪边界
#         x1 = max(0, center_x - half_width)
#         y1 = max(0, center_y - half_height)
#         x2 = min(width, center_x + half_width)
#         y2 = min(height, center_y + half_height)
#
#         # 处理边界情况（当裁剪区域超出原图时进行填充）
#         pad_left = max(0, half_width - center_x)
#         pad_top = max(0, half_height - center_y)
#         pad_right = max(0, (center_x + half_width) - width)
#         pad_bottom = max(0, (center_y + half_height) - height)
#
#         # 裁剪并填充
#         cropped = frame[y1:y2, x1:x2]
#         if any([pad_left, pad_right, pad_top, pad_bottom]):
#             cropped = cv2.copyMakeBorder(cropped,
#                                          pad_top,
#                                          pad_bottom,
#                                          pad_left,
#                                          pad_right,
#                                          cv2.BORDER_CONSTANT,
#                                          value=(0, 0, 0))
#
#         # 调整到目标尺寸
#         final = cv2.resize(cropped, (416, 320))
#         out.write(final)
#
#     cap.release()
#     out.release()
#     # 合并音频
#     if audio:
#         try:
#             video_clip = VideoFileClip(temp_video)
#             final_clip = video_clip.set_audio(audio)
#             final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
#             video_clip.close()
#         except Exception as e:
#             print(f"音频合并失败: {str(e)}")
#             # 如果合并失败，直接使用无声视频
#             import shutil
#             shutil.move(temp_video, output_path)
#     else:
#         import shutil
#         shutil.move(temp_video, output_path)
#
#     print(f"处理完成，输出视频已保存至：{output_path}")
#
#
# # 使用示例
# crop_face_centered("D:/Python/team/DINet2/asserts/examples/my/video.mp4", "D:/Python/team/DINet2/asserts/examples/my/videocrop.mp4")

import cv2
import dlib
import os
from moviepy.editor import VideoFileClip
import warnings

warnings.filterwarnings("ignore")  # 屏蔽moviepy的无关警告

# 初始化dlib的人脸检测器和面部landmark预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/Python/team/DINet2/asserts/shape_predictor_68_face_landmarks.dat")


def extract_audio_to_wav(video_path, output_wav_path=None):
    """
    提取视频中的音频并保存为WAV文件
    :param video_path: 输入视频路径
    :param output_wav_path: 输出音频路径（默认为视频同目录）
    :return: 成功返回True，失败返回False
    """
    try:
        # 设置默认输出路径
        if output_wav_path is None:
            base_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_wav_path = os.path.join(base_dir, f"{video_name}.wav")

        # 提取音频
        with VideoFileClip(video_path) as video:
            audio = video.audio
            if audio is None:
                print("警告: 视频中没有音频轨道")
                return False

            # 保存为WAV格式
            audio.write_audiofile(output_wav_path,
                                  fps=44100,  # 标准CD音质采样率
                                  codec='pcm_s16le',  # 16-bit PCM编码
                                  verbose=False)

            print(f"音频已保存至：{output_wav_path}")
            return True

    except Exception as e:
        print(f"音频提取失败: {str(e)}")
        return False


def crop_face_centered(video_path, output_path):
    # 先提取原始音频
    try:
        original_clip = VideoFileClip(video_path)
        audio = original_clip.audio
    except Exception as e:
        print(f"音频提取失败: {str(e)}")
        audio = None

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建临时无声视频文件路径
    temp_video = "temp_silent.mp4"

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (416, 320))

    prev_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图像进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        if len(faces) > 0:
            # 取最大的人脸（按面积计算）
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            landmarks = predictor(gray, face)

            # 计算68个特征点的中心点
            sum_x = sum(landmarks.part(i).x for i in range(68))
            sum_y = sum(landmarks.part(i).y for i in range(68))
            center_x = int(sum_x / 68)
            center_y = int(sum_y / 68)
            prev_center = (center_x, center_y)
        else:
            if prev_center is None:
                continue  # 跳过没有面部且无历史数据的帧
            center_x, center_y = prev_center  # 使用上一帧的位置

        # 计算裁剪区域（416x320）
        crop_width = 416
        crop_height = 320
        half_width = crop_width // 2
        half_height = crop_height // 2

        # 计算裁剪边界
        x1 = max(0, center_x - half_width)
        y1 = max(0, center_y - half_height)
        x2 = min(width, center_x + half_width)
        y2 = min(height, center_y + half_height)

        # 处理边界情况（当裁剪区域超出原图时进行填充）
        pad_left = max(0, half_width - center_x)
        pad_top = max(0, half_height - center_y)
        pad_right = max(0, (center_x + half_width) - width)
        pad_bottom = max(0, (center_y + half_height) - height)

        # 裁剪并填充
        cropped = frame[y1:y2, x1:x2]
        if any([pad_left, pad_right, pad_top, pad_bottom]):
            cropped = cv2.copyMakeBorder(cropped,
                                         pad_top,
                                         pad_bottom,
                                         pad_left,
                                         pad_right,
                                         cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))

        # 调整到目标尺寸
        final = cv2.resize(cropped, (416, 320))
        out.write(final)

    cap.release()
    out.release()

    # 合并音频
    if audio:
        try:
            video_clip = VideoFileClip(temp_video)
            final_clip = video_clip.set_audio(audio)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            video_clip.close()
        except Exception as e:
            print(f"音频合并失败: {str(e)}")
            import shutil
            shutil.move(temp_video, output_path)
    else:
        import shutil
        shutil.move(temp_video, output_path)

    # 清理临时文件
    if os.path.exists(temp_video):
        os.remove(temp_video)

    print(f"处理完成，输出视频已保存至：{output_path}")


# 使用示例
if __name__ == "__main__":
    input_video = "D:/Python/team/DINet2/eval/examples/video2-2.mp4"
    #output_video = "D:/Python/team/DINet2/asserts/examples/video2crop.mp4"

    # 同时进行视频裁剪和音频提取
    if extract_audio_to_wav(input_video):
        print("音频提取成功完成")
    #crop_face_centered(input_video, output_video)