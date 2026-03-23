import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from ultralytics import YOLO

# ==========================================================
# 1. NẠP ẢNH MẪU CHUẨN TỪ MÁY (Cùng tấm ảnh Dog.jpg với Faster R-CNN)
# ==========================================================
print("=> Đang nạp ảnh mẫu từ file cục bộ...")
img_bgr = cv2.imread('dog.jpg')
if img_bgr is None:
    raise FileNotFoundError("Không tìm thấy file dog.jpg ở thư mục hiện tại.")
img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ==========================================================
# 2. KHỞI TẠO MÔ HÌNH YOLOv8 (ONE-STAGE DETECTOR TỐC ĐỘ CAO)
# ==========================================================
print("=> Đang tải mô hình kiến trúc YOLOv8n (Bản Nano - Anchor-Free)...")
# Note: Lệnh này sẽ nạp trực tiếp file yolov8n.pt ở máy của bạn
model = YOLO('yolov8n.pt') 

# ==========================================================
# 3. CHẠY SUY LUẬN (INFERENCE)
# ==========================================================
print("=> Đang phân tích hình ảnh (Chỉ 1 bước duy nhất)...")
results = model(img_np, verbose=False)[0]

# Trích xuất dữ liệu dự đoán
boxes = results.boxes.xyxy.cpu().numpy()  # Tọa độ [x1, y1, x2, y2]
scores = results.boxes.conf.cpu().numpy() # Độ tự tin (Confidence)
class_ids = results.boxes.cls.cpu().numpy() # ID nhãn (Class)
names = model.names

# ==========================================================
# 4. TRỰC QUAN HÓA GIÁO KHOA (DÀNH CHO SÁCH SHM)
# ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# -----------------------------------------------------------------
# HÌNH 1: MINH HỌA LÝ THUYẾT ANCHOR-FREE & DETECTION HEAD (Mục 5.3)
# -----------------------------------------------------------------
ax1.imshow(img_np)

# Lấy con chó làm ví dụ (thường là box đầu tiên hoặc box to nhất)
if len(boxes) > 0:
    box = boxes[0] 
    x1, y1, x2, y2 = box
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2  # Tâm dự đoán trên Feature Map
    
    # 4 khoảng cách (l, t, r, b) như lý thuyết trong sách
    l, t, r, b = xc - x1, yc - y1, x2 - xc, y2 - yc
    
    # 1. Đoạn mô phỏng lưới cắt Feature Map (Grid) của Backbone
    for i in range(0, img_np.shape[1], 50):
        ax1.axvline(i, color='white', alpha=0.15)
    for j in range(0, img_np.shape[0], 50):
        ax1.axhline(j, color='white', alpha=0.15)
        
    # 2. Vẽ Điểm Neo Dự đoán (Center point)
    ax1.plot(xc, yc, 'ro', markersize=12)
    ax1.text(xc - 110, yc + 35, "Điểm Neo (Gán nhãn động)", color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='red', alpha=0.8, pad=3))
    
    # 3. Kẻ 4 vector (l, t, r, b) phóng ra 4 cạnh Bounding Box
    ax1.annotate("", xy=(x1, yc), xytext=(xc, yc), arrowprops=dict(arrowstyle="->", color="lime", lw=4))
    ax1.text((x1+xc)/2, yc-15, "l (left)", color="lime", fontsize=16, fontweight='bold', bbox=dict(facecolor='black', alpha=0.4, pad=1))
    
    ax1.annotate("", xy=(x2, yc), xytext=(xc, yc), arrowprops=dict(arrowstyle="->", color="lime", lw=4))
    ax1.text((xc+x2)/2 - 10, yc-15, "r (right)", color="lime", fontsize=16, fontweight='bold', bbox=dict(facecolor='black', alpha=0.4, pad=1))
    
    ax1.annotate("", xy=(xc, y1), xytext=(xc, yc), arrowprops=dict(arrowstyle="->", color="cyan", lw=4))
    ax1.text(xc+10, (y1+yc)/2, "t (top)", color="cyan", fontsize=16, fontweight='bold', bbox=dict(facecolor='black', alpha=0.4, pad=1))
    
    ax1.annotate("", xy=(xc, y2), xytext=(xc, yc), arrowprops=dict(arrowstyle="->", color="cyan", lw=4))
    ax1.text(xc+10, (yc+y2)/2, "b (bottom)", color="cyan", fontsize=16, fontweight='bold', bbox=dict(facecolor='black', alpha=0.4, pad=1))
    
    # 4. Vẽ Box mờ để làm nền cho các vector l, t, r, b
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='yellow', facecolor='none', linestyle='--')
    ax1.add_patch(rect)

ax1.axis('off')

# -----------------------------------------------------------------
# HÌNH 2: KẾT QUẢ THỰC TẾ CỦA ONE-STAGE DETECTOR
# -----------------------------------------------------------------
ax2.imshow(img_np)
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i]
    conf = scores[i]
    cls_name = names[int(class_ids[i])]
    
    # Vẽ Khung giới hạn (Bounding Box)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=4, edgecolor='dodgerblue', facecolor='none')
    ax2.add_patch(rect)
    
    # Gắn nhãn
    text = f"{cls_name.upper()} {conf:.2f}"
    ax2.text(x1, y1-8, text, color='white', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='dodgerblue', edgecolor='none', alpha=0.9, pad=3))

ax2.axis('off')

# LƯU FILE - Bỏ hẳn chữ matplotlib để làm ảnh nền giáo khoa 
plt.tight_layout(pad=0)
plt.savefig('yolo_one_stage.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("=> XONG! Đã xuất đồ thị giải phẫu kiến trúc YOLOv8 (Anchor-Free) vào file: yolo_one_stage.png")
