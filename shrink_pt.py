import os
import torch
import numpy as np
from torch.serialization import add_safe_globals  # PyTorch 2.6+

# (신뢰한 파일이므로) NumPy reconstruct 허용
add_safe_globals([np.core.multiarray._reconstruct])

SRC = r""  # ← 파일 실제 위치
DST = r""  # 결과 파일 저장 위치

print("불러오는 중:", SRC)
# 신뢰한 파일이므로 weights_only=False 로드
ckpt = torch.load(SRC, map_location="cpu", weights_only=False)

# ckpt 형태에 유연하게 대응
model = None
if isinstance(ckpt, dict):
    model = ckpt.get("ema", ckpt.get("model", None))

if model is None:
    # state_dict만 저장된 형태일 수도 있으니 그대로 최소 저장
    torch.save(ckpt, DST)
else:
    # 불필요한 항목 제거
    for k in ["optimizer","updates","wandb_id","train_results","best_fitness",
              "git","date","wandb","ema","epoch"]:
        ckpt.pop(k, None)
    torch.save({"model": model}, DST)

print("저장 완료:", DST)
print("최종 파일 크기:", round(os.path.getsize(DST)/1024/1024, 2), "MB")