import pandas as pd
import matplotlib.pyplot as plt

csv_path = "H:/BaiduNetdiskDownload/frame128.csv"
df = pd.read_csv(csv_path)

# 排序+去重（避免折返）
df = df.sort_values("global_step")
df = df.groupby("global_step", as_index=False).mean()

# 平滑（可选）
for c in ["loss_g_total","loss_perc","loss_g_dI"]:
    if c in df.columns:
        df[c] = df[c].rolling(window=10, min_periods=1).mean()

x = df["global_step"]
ycols = [c for c in ["loss_g_total","loss_perc","loss_g_dI"] if c in df.columns]

# (a) 全程
plt.figure()
for c in ycols:
    plt.plot(x, df[c], label=c)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Generator losses (full range)")
plt.legend()
plt.tight_layout()
plt.savefig("G_full.png", dpi=400)
plt.savefig("G_full.pdf")
plt.close()

# (b) 放大 0–500
zoom_max = 6000
dfz = df[df["global_step"] <= zoom_max]

plt.figure()
for c in ycols:
    plt.plot(dfz["global_step"], dfz[c], label=c)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Generator losses (zoom: 0–500)")
plt.legend()
plt.tight_layout()
plt.savefig("G_zoom_0_6000.png", dpi=400)
plt.savefig("G_zoom_0_6000.pdf")
plt.close()
