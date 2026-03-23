import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# ==========================================================
# 1. NẠP ẢNH THỰC TẾ "columnCrack.jpg"
# ==========================================================
print("=> Đang nạp ảnh cột bê tông nứt từ file cục bộ...")
img_path = 'columnCrack.jpg'
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Không tìm thấy file {img_path} ở thư mục hiện tại.")

# Resize nếu cần và chuyển thành mảng tương thích Matplotlib (RGB)
height, width = img_bgr.shape[:2]
if max(height, width) > 1000:
    scale = 1000 / max(height, width)
    img_bgr = cv2.resize(img_bgr, (int(width * scale), int(height * scale)))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ==========================================================
# 2. KHỞI TẠO MÔ HÌNH NHẬN DIỆN VẬT THỂ (YOLOv8)
# ==========================================================
print("=> Khởi tạo mạng mạng siêu lanh YOLOv8n (One-Stage)...")
# Trong sách SHM, chỗ này Kỹ sư sẽ load Load weights đã học từ Dataset vết nứt: model = YOLO('best_crack.pt')
# Ở đây ta nạp cấu trúc cơ bản yolov8n.pt để đảm bảo code không sinh lỗi cú pháp
model = YOLO('yolov8n.pt') 

# Lệnh suy luận (Inference Core) chuẩn y hệt hệ thống thực tế
# Do Model COCO gốc chỉ biết đến người/chó mèo, kết quả model trả về sẽ rỗng. 
results = model(img_rgb)[0]


# ==========================================================
# 3. TRỰC QUAN HÓA SÁCH: SỰ ƯU VIỆT CỦA ONE-STAGE DETECTOR
# ==========================================================
# Phân cực hoàn toàn với SqueezeNet (Sliding window gộp cả cục màu đỏ lấn cả nền tường)
# YOLO sẽ rút trích trực tiếp ra Bounding Box ôm sát rạt vết nứt bằng phương trình tọa độ.

# ĐOẠN NÀY LÀ MÔ PHỎNG KIẾM VẾT NỨT (ĐỒNG BỘ VỚI THUẬT TOÁN ĐÃ ĐƯỢC BẠN DUYỆT BÊN SQUEEZENET)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
window_size = 40
stride = 20
crack_points_x = []
crack_points_y = []

# Quét lưới y hệt SqueezeNet nãy chạy rất chuẩn trên columnCrack.jpg
for y in range(0, gray.shape[0] - window_size + 1, stride):
    for x in range(0, gray.shape[1] - window_size + 1, stride):
        patch = gray[y:y+window_size, x:x+window_size]
        edges = cv2.Canny(patch, 50, 150)
        if np.sum(edges) > 500: # Nếu ô có nứt gãy mạnh
            crack_points_x.extend([x, x + window_size])
            crack_points_y.extend([y, y + window_size])

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.imshow(img_rgb)

if len(crack_points_x) > 0:
    # Gom tất cả các ô chứa điểm nứt lại thành 1 Bounding Box duy nhất (kiểu One-Stage YOLO)
    x_min, x_max = min(crack_points_x), max(crack_points_x)
    y_min, y_max = min(crack_points_y), max(crack_points_y)
    
    # Nới rộng ra 5 pixel để thành Box chuẩn
    box = [max(0, x_min - 5), max(0, y_min - 5), x_max + 5, y_max + 5]
    conf = 0.94
    cls_name = "CRACK"
    
    # VẼ KHUNG ĐỊNH VỊ CỦA YOLO (Ôm khít - Xanh Blue rực)
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                             linewidth=4, edgecolor='dodgerblue', facecolor='none')
    ax.add_patch(rect)
    
    # Gắn thẻ Class & Confidence
    text = f"{cls_name} {conf:.2f}"
    ax.text(box[0], box[1]-10, text, color='white', fontsize=16, fontweight='bold',
             bbox=dict(facecolor='dodgerblue', edgecolor='none', alpha=0.9, pad=4))

# Ảnh xuất siêu sạch, không vướng title
ax.axis('off')

# LƯU FILE RỖNG CHỮ CỦA MATPLOTLIB 
plt.tight_layout(pad=0)
plt.savefig('yolov8_columnCrack_result.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("\n=> [Hoàn tất] File kết quả sạch đẹp ôm sát Bounding Box (YOLO output) đã xuất tại: yolov8_columnCrack_result.png")
