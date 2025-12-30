"""
环境验证脚本：
 - 检查关键依赖是否可导入并显示版本
 - 检查 PyTorch 的 CUDA 是否可用及设备信息
"""

import sys
from importlib import import_module


DEPENDENCIES = [
	("torch", "torch"),
	("numpy", "numpy"),
	("polars", "polars"),
	("tqdm", "tqdm"),
	("matplotlib", "matplotlib"),
]


def check_deps() -> tuple[bool, dict]:
	ok = True
	versions = {}
	for display, modname in DEPENDENCIES:
		try:
			m = import_module(modname)
			ver = getattr(m, "__version__", "未知")
			versions[display] = ver
			print(f"[INFO] 依赖 {display}：正常（版本 {ver}）")
		except Exception as e:
			ok = False
			print(f"[ERROR] 依赖 {display}：导入失败（{e}）")
	return ok, versions


def check_cuda() -> bool:
	try:
		import torch
	except Exception as e:
		print(f"[ERROR] CUDA：无法导入 torch（{e}）")
		return False

	if not torch.cuda.is_available():
		print(f"[ERROR] CUDA：不可用")
		return False

	count = torch.cuda.device_count()
	devices = []
	for i in range(count):
		try:
			devices.append(torch.cuda.get_device_name(i))
		except Exception:
			devices.append(f"设备{i}")
	print(f"[INFO] 可用（设备数 {count}，名称 {devices}）")
	return True


def main() -> int:
	all_ok, _ = check_deps()
	cuda_ok = check_cuda()
	if all_ok and cuda_ok:
		print("[INFO] 环境验证通过")
		return 0
	print("[ERROR] 环境验证失败")
	return 1


if __name__ == "__main__":
	sys.exit(main())
