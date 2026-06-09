#import torch
#print(torch.__version__)          # 应输出 1.12.1+cu113
#print(torch.cuda.is_available())    # 应输出 True
#print(torch.backends.cudnn.enabled) # 应输出 True
#print(torch.backends.cudnn.version()) # 应输出 (8, 2, 910)
import torch

# 查看 PyTorch 版本和 CUDA 版本
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")

# 检查 cuDNN 是否启用
print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
print(torch.cuda.is_available())  # 检查 CUDA 是否可用

print(torch.cuda.current_device())  # 打印当前使用的设备
print(torch.cuda.get_device_name(0))  # 打印 GPU 名称

x = torch.randn(1, 3, 224, 224).cuda()
conv = torch.nn.Conv2d(3, 64, kernel_size=3).cuda()
y = conv(x)  # 如果此处报错，则 cuDNN 存在问题
print("cuDNN 工作正常!")
