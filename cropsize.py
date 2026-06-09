# import subprocess
# import argparse
# import os
#
#
# def get_video_dimensions(video_path):
#     """使用ffprobe获取视频尺寸"""
#     try:
#         width = subprocess.check_output(
#             ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
#              '-show_entries', 'stream=width', '-of', 'default=nw=1:nk=1', video_path],
#             text=True
#         ).strip()
#
#         height = subprocess.check_output(
#             ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
#              '-show_entries', 'stream=height', '-of', 'default=nw=1:nk=1', video_path],
#             text=True
#         ).strip()
#         return int(width), int(height)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"获取视频尺寸失败: {str(e)}")
#
#
# def resize_video(source_path, target_width, target_height, output_path, keep_ratio=True):
#     """调整视频尺寸"""
#     scale_filter = f"scale={target_width}:{target_height}"
#     if keep_ratio:
#         scale_filter = (
#             f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease,"
#             f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
#         )
#
#     try:
#         subprocess.run(
#             ['ffmpeg', '-i', source_path,
#              '-vf', f'autorotate,{scale_filter}',
#              '-c:a', 'copy',
#              '-y',  # 覆盖输出文件
#              output_path],
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"视频处理失败: {e.stderr.decode()}")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='调整视频尺寸')
#     parser.add_argument('video1',default='D:/Python/team/DINet2/eval/examples/videocrop1.mp4' ,help='D:/Python/team/DINet2/eval/examples/videocrop1.mp4')
#     parser.add_argument('video2',default='D:/Python/team/DINet2/eval/result/eamm/videocrop1.mp4',help='D:/Python/team/DINet2/eval/result/eamm/videocrop1.mp4')
#     parser.add_argument('-o', '--output', default='D:/Python/team/DINet2/eval/result/eamm/videocropsize1.mp4', help='D:/Python/team/DINet2/eval/result/eamm')
#     parser.add_argument('--force', action='store_false',
#                         help='强制拉伸模式（默认保持宽高比）')
#
#     args = parser.parse_args()
#
#     # 验证文件存在
#     for path in [args.video1, args.video2]:
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"文件不存在: {path}")
#
#     # 获取目标尺寸
#     try:
#         target_width, target_height = get_video_dimensions(args.video1)
#         print(f"目标尺寸: {target_width}x{target_height}")
#
#         # 执行尺寸调整
#         resize_video(
#             args.video2,
#             target_width,
#             target_height,
#             args.output,
#             keep_ratio=args.force
#         )
#
#         print(f"处理完成！输出文件: {args.output}")
#     except Exception as e:
#         print(f"错误发生: {str(e)}")
import argparse
import subprocess
import os
import sys


def get_video_dimensions(video_path):
    """获取视频分辨率（自动处理旋转元数据）"""
    try:
        # 获取原始分辨率
        width = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width', '-of', 'csv=p=0', video_path],
            text=True, stderr=subprocess.STDOUT
        ).strip()

        height = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=height', '-of', 'csv=p=0', video_path],
            text=True, stderr=subprocess.STDOUT
        ).strip()

        # 检测旋转元数据
        rotation = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream_side_data=rotation', '-of', 'csv=p=0', video_path],
            text=True, stderr=subprocess.STDOUT
        ).strip()

        # 处理旋转交换宽高
        if rotation in ['90', '270']:
            width, height = height, width

        return int(width), int(height)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"视频分析失败: {e.output}")


def resize_video(source_path, target_size, output_path, keep_ratio=True):
    """智能调整视频尺寸"""
    try:
        # 构建FFmpeg命令
        cmd = [
            'ffmpeg',
            '-i', source_path,
            '-vf', f'scale={target_size[0]}:{target_size[1]}' if not keep_ratio else
            f'scale=w={target_size[0]}:h={target_size[1]}:force_original_aspect_ratio=decrease,pad={target_size[0]}:{target_size[1]}:(ow-iw)/2:(oh-ih)/2',
            '-c:a', 'copy',
            '-y',  # 覆盖输出文件
            '-stats',  # 显示进度
            output_path
        ]

        # 显示友好进度（替换默认输出）
        print("▌" * 50 + " 开始处理视频 ")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # 实时输出处理进度
        for line in process.stdout:
            if 'frame=' in line:
                sys.stdout.write(f"\r▌ 处理进度: {line.strip()}")
                sys.stdout.flush()

        process.wait()
        print("\n" + "▌" * 50 + " 处理完成！ ")

    except Exception as e:
        raise RuntimeError(f"视频处理失败: {str(e)}")


if __name__ == "__main__":
    # 配置智能默认路径（根据项目结构自动生成）
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_config = {
        'video1': os.path.join(project_root, 'D:/Python/team/DINet2/eval/examples/mead1.mp4'),
        'video2': os.path.join(project_root, 'D:/Python/team/DINet2/eval/result/fsrt/mead.mp4'),
        'output': os.path.join(project_root, 'D:/Python/team/DINet2/eval/result/fsrt/meadsize.mp4')
    }

    # 参数解析器配置
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='PyCharm视频尺寸调整工具\n'
                    '自动将第二个视频调整为第一个视频的尺寸\n\n'
                    '默认路径配置：\n'
                    f'  参考视频：{default_config["video1"]}\n'
                    f'  输入视频：{default_config["video2"]}\n'
                    f'  输出路径：{default_config["output"]}'
    )

    parser.add_argument('video1', nargs='?', default=default_config['video1'],
                        help='参考视频路径（自动检测项目结构）')
    parser.add_argument('video2', nargs='?', default=default_config['video2'],
                        help='需要调整的视频路径')
    parser.add_argument('-o', '--output', default=default_config['output'],
                        help='输出文件路径（默认：项目目录下）')
    parser.add_argument('--force', action='store_true',
                        help='强制拉伸模式（默认智能填充黑边）')

    # 参数验证
    args = parser.parse_args()

    try:
        # 路径规范化处理
        args.video1 = os.path.normpath(args.video1)
        args.video2 = os.path.normpath(args.video2)
        args.output = os.path.normpath(args.output)

        # 自动创建输出目录
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        # 验证输入文件
        for path in [args.video1, args.video2]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")
            if not path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                raise ValueError(f"不支持的文件格式: {os.path.splitext(path)[1]}")

        # 获取目标尺寸
        print("▌ 正在分析参考视频尺寸...")
        target_size = get_video_dimensions(args.video1)
        print(f"▌ 目标分辨率: {target_size[0]}x{target_size[1]}")

        # 执行尺寸调整
        resize_video(
            source_path=args.video2,
            target_size=target_size,
            output_path=args.output,
            keep_ratio=not args.force
        )

        # 结果验证
        if os.path.getsize(args.output) == 0:
            raise RuntimeError("输出文件为空，请检查FFmpeg配置")

        print(f"✅ 处理成功！输出文件: {args.output}")
        print(f"   文件大小: {os.path.getsize(args.output) / 1024 / 1024:.2f}MB")

    except Exception as e:
        print(f"❌ 错误发生: {str(e)}")
        sys.exit(1)