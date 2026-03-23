import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ========================================================
# ========================================================
# 1. BẢN ĐỒ THỰC TẾ: TẢI ẢNH TỪ FILE 
# ========================================================
img_path = 'columnCrack.JPG'
img_bgr = cv2.imread(img_path)

if img_bgr is None:
    raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {img_path}")
    
# Resize lại cho nhỏ gọn nếu ảnh quá lớn, để cửa sổ trượt (150px) lọt đúng khung hình
height, width = img_bgr.shape[:2]
if max(height, width) > 1000:
    scale = 1000 / max(height, width)
    img_bgr = cv2.resize(img_bgr, (int(width * scale), int(height * scale)))


# ========================================================
# 2. KHỞI TẠO MÔ HÌNH CNN NHẸ (LIGHTWEIGHT): SQUEEZENET 1.1
# Thiết kế hoàn hảo cho thiết bị Drone/Mobile nhờ giảm thiểu tỷ lệ dung lượng
# ========================================================
print("=> Khởi tạo mạng mạng siêu nhẹ SqueezeNet 1.1...")
model = models.squeezenet1_1(weights=None) # Lõi rỗng

# Hiệu chỉnh (Fine-tune) Mạng Neural xuống 2 classes: 0 (Negative) và 1 (Positive)
model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))

import os
import sys

weight_file = 'squeezenet_crack.pth'
if not os.path.exists(weight_file):
    print(f"\n[LỖI] Rỗng trí nhớ: Chưa tìm thấy tệp Não Nhân Tạo '{weight_file}'!")
    print("=> Xin vui lòng chạy file 'train_squeezenet.py' trước mặt để cỗ máy AI học cách phân tách dữ liệu bê tông nhé.")
    sys.exit(1)
else:
    print(f"=> Truy xuất thành công tệp tri thức trọng số từ quá trình Huấn Luyện thực tế: {weight_file}")
    model.load_state_dict(torch.load(weight_file, map_location='cpu'))

model.eval()

# Áp dụng bộ Transforms kinh điển mô tả trong sách: Resize -> ToTensor -> Normalize
preprocess = transforms.Compose([
    transforms.Resize((227, 227)), # SqueezeNet bắt buộc input Size: 227x227
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ========================================================
# 3. THỰC THI (INFERENCE) VỚI KỸ THUẬT SLIDING WINDOW
# ========================================================
window_size = 40  # Kích thước Cửa sổ (Pixels) - Siêu nhỏ để tăng độ phân giải cực đại
stride = 20       # Nhảy bước (20px) cho lưới dày đặc hơn

img_result = img_bgr.copy()
total_windows = 0
crack_found = 0

print("=> Cửa sổ bắt đầu trượt tuần tự quét bề ngoài vết nứt...")
for y in range(0, img_bgr.shape[0] - window_size + 1, stride):
    for x in range(0, img_bgr.shape[1] - window_size + 1, stride):
        total_windows += 1
        # Cắt ô nhỏ (Patch)
        patch = img_bgr[y:y+window_size, x:x+window_size]
        
        # Format Patch thành Tensor 4D (1, C, 227, 227) để đẩy vào Pytorch SqueezeNet
        patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(patch_pil).unsqueeze(0)
        
        # TIẾN HÀNH THỰC THI (INFERENCE) THẬT BẰNG MODEL ĐÃ TRAIN
        # Đút Patch bê tông đi vào Não AI SqueezeNet
        with torch.no_grad():
            output = model(input_tensor)
            # Hàm Softmax định dạng Probability [Prob(Negative), Prob(Positive)]
            prob_raw = torch.softmax(output, dim=1)
            # Truy xuất cái lớn nhất: (Tự tin, Ngăn kéo ID)
            prob_tensor, pred_idx = torch.max(prob_raw, 1)
            pred_class = pred_idx.item()
            prob = prob_tensor.item()
            
        # Ánh xạ theo chữ cái ABC: 
        # Folder N (Negative) xếp thứ 0, Folder P (Positive - Có vết nứt) xếp thứ 1
        if pred_class == 1:
            crack_found += 1
            # KHÔNG TÔ MÀU BÊN TRONG (Chỉ vẽ khung viền Đỏ đậm cực đoan bao quanh)
            cv2.rectangle(img_result, (x, y), (x+window_size, y+window_size), (0, 0, 255), 2)

print(f"=> XONG! Quét hết {total_windows} mảng ô cửa sổ.")
print(f"=> Báo động: Xác định {crack_found} vùng rạn nứt!")

# ========================================================
# 4. TRỰC QUAN HÓA SO SÁNH / HẠN CHẾ (ẢNH TRẮNG CHỮ)
# ========================================================
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
axs[1].axis('off')

plt.tight_layout(pad=0)
plt.savefig('squeezenet_sliding_window.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("=> File báo cáo trực quan cho Sách đã xuất tại: squeezenet_sliding_window.png (Hoàn toàn không chữ)")
