import re
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".flv"}

def safe_name(name: str) -> str:
    # Windows/跨平台安全命名
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "video"

def extract_frames_ffmpeg(video_path: Path, out_dir: Path, fps: int = 25, img_ext: str = "jpg", jpg_q: int = 2):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"img_%06d.{img_ext}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",   # 想看过程可改 "info"
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
    ]

    # jpg质量：1最好，31最差（2-5通常很不错）
    if img_ext.lower() in ("jpg", "jpeg"):
        cmd += ["-q:v", str(jpg_q)]

    cmd += [pattern]
    subprocess.run(cmd, check=True)

def batch_extract_nested(input_root: str, output_root: str, fps: int = 25, img_ext: str = "jpg"):
    in_root = Path(input_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"未找到视频：{in_root}")
        return

    ok, fail = 0, 0
    for vp in videos:
        # 保持“子文件夹层级”
        rel_parent = vp.parent.relative_to(in_root)       # 例如：A/ 或 A/B/
        video_folder = safe_name(vp.stem)                 # 视频名文件夹
        out_dir = out_root / rel_parent / video_folder    # out/A/视频名/

        try:
            extract_frames_ffmpeg(vp, out_dir, fps=fps, img_ext=img_ext)
            ok += 1
            print(f"[OK] {vp} -> {out_dir}")
        except subprocess.CalledProcessError as e:
            fail += 1
            print(f"[FAIL] {vp}：{e}")

    print(f"\n完成：成功 {ok} 个视频，失败 {fail} 个。输出根目录：{out_root}")

if __name__ == "__main__":
    # 修改这里两行即可
    batch_extract_nested(
        input_root=r"D:/Python/team/DINet2/eval/result/idea2",   # 你的总文件夹（里面有很多子文件夹）
        output_root=r"D:/Python/team/DINet2/eval/25fps",   # 导出图片的总目录
        fps=25,
        img_ext="png"                  # 或 "png"
    )
