import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. TẠO ẢNH GIẢ LẬP: MẶT ĐƯỜNG BÊ TÔNG (VỚI VẾT NỨT)
np.random.seed(42)
img_ground = np.random.normal(160, 15, (600, 600)).astype(np.uint8)
img_ground = cv2.GaussianBlur(img_ground, (3, 3), 0)
# Biến nó thành mảng 3 kênh RGB để vẽ màu
img_ground = cv2.cvtColor(img_ground, cv2.COLOR_GRAY2RGB)

# Vẽ một "Bản lề tham chiếu" (Mốc đo đạc - hình chữ nhật hoàn hảo 300x400 nằm ở vị trí X=150, Y=100)
# Ví dụ: Đây là một thước đo tỷ lệ được dán lên mặt đường, hoặc đường viền viên gạch
rect_ground = np.array([
    [150, 100], [450, 100], 
    [450, 500], [150, 500]
], dtype=np.float32)

# VẼ HÌNH CHỮ NHẬT LÊN CÙNG
cv2.polylines(img_ground, [rect_ground.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=5)

# NÉT ĐẶC TRƯNG: Thêm nội dung là một Vết nứt uốn lượn ở trong phạm vi hình chữ nhật
crack_pts = np.array([
    [200, 150], [280, 220], [250, 300], [350, 420], [310, 480]
], np.int32).reshape((-1, 1, 2))
cv2.polylines(img_ground, [crack_pts], isClosed=False, color=(255, 0, 0), thickness=8) # Vết nứt đỏ để dễ nhìn


# ==========================================
# 2. MÔ PHỎNG CAMERA CHỤP TỪ GÓC NGHIÊNG (PERSPECTIVE DISTORTION)
# ==========================================
# Giả sử ảnh trên là chụp thẳng từ flycam. Nhưng thực tế kỹ sư chụp bằng điện thoại từ vị trí chéo góc.
# Ta sẽ bóp méo phối cảnh ảnh gốc để biến nó thành ảnh góc nghiêng của camera
M_tilt = cv2.getPerspectiveTransform(
    np.array([[0,0], [600,0], [600,600], [0,600]], dtype=np.float32), 
    np.array([[100, 150], [500, 150], [600, 600], [0, 600]], dtype=np.float32) # Bóp méo viền trên
)
img_camera = cv2.warpPerspective(img_ground, M_tilt, (600, 600), borderValue=(50,50,50))

# Khung chữ nhật màu xanh lá trên mặt đất giờ đây khi chụp qua Camera đã bị biến dạng thành HÌNH THANG.
# Tọa độ 4 góc của "Hình thang" này trên ảnh chụp (Pixel coordinates x, y trên ảnh camera)
# Ta tính ngược để tìm tọa độ 4 điểm méo bằng M_tilt
pts_augmented = np.array([rect_ground], dtype=np.float32)
rect_camera = cv2.perspectiveTransform(pts_augmented, M_tilt)[0]


# ==========================================
# 3. ỨNG DỤNG BÀI TOÁN CỦA BẠN: IMAGE RECTIFICATION (HIỆU CHỈNH ẢNH BẰNG HOMOGRAPHY)
# ==========================================
# Giao diện cho máy tính xem:
# Input: Máy tính "thấy" khung xanh là một hình thang (rect_camera).
# Chân lý (Ground Truth): Kỹ sư ngoài đời biết khung xanh thực chất là một hình chữ nhật kích thước 300x400.
rect_true_flat = np.array([
    [100, 100], [400, 100],   # Phẳng, rộng 300
    [400, 500], [100, 500]    # Dài 400
], dtype=np.float32)

# BƯỚC CỐT LÕI (như mục 1.2.3 trong sách): Dùng 4 điểm để tìm Ma trận HOMOGRAPHY H (3x3)
H, status = cv2.findHomography(rect_camera, rect_true_flat)

# BƯỚC PHỤC HỒI (Warp Perspective): Nhân ma trận H này với TOÀN BỘ TỌA ĐỘ HÌNH ẢNH
# Kết quả s * [x, y, 1]^T = H * [X, Y, 1]^T
img_rectified = cv2.warpPerspective(img_camera, H, (500, 600))


# ==========================================
# TRỰC QUAN HÓA SO SÁNH (LƯU RA FILE)
# ==========================================
fig, axs = plt.subplots(1, 2, figsize=(18, 8))

# HÌNH 1: Ảnh chụp camera bị méo phối cảnh
axs[0].imshow(img_camera)
axs[0].set_title("1. Góc nhìn thực tế từ Camera\n(Vật thể bị bóp méo theo phối cảnh - Hình Thang)", fontsize=16, fontweight='bold', color='darkorange')
# Vẽ lại 4 góc bị méo
for i, pt in enumerate(rect_camera):
    axs[0].plot(pt[0], pt[1], 'ro', markersize=8)
    axs[0].text(pt[0]-30, pt[1]-15, f"P{i+1}\n(x,y)", color='white', fontweight='bold', bbox=dict(facecolor='red', alpha=0.5))
axs[0].axis('off')

# HÌNH 2: Ảnh sau khi áp dụng Ma trận Homography
axs[1].imshow(img_rectified)
axs[1].set_title("2. Áp dụng Ma trận Homography 3x3 (Hiệu chỉnh góc nhìn)\nĐưa ảnh về Rectified (Bird's Eye View) = Hình Chữ Nhật", fontsize=16, fontweight='bold', color='green')
for i, pt in enumerate(rect_true_flat):
    axs[1].plot(pt[0], pt[1], 'bo', markersize=8)
    axs[1].text(pt[0]-30, pt[1]-15, f"P{i+1}\n(X,Y)", color='white', fontweight='bold', bbox=dict(facecolor='blue', alpha=0.5))
axs[1].axis('off')

plt.suptitle("Ứng dụng cốt lõi của Homography trong SHM: IMAGE RECTIFICATION (HIỆU CHỈNH KÍCH THƯỚC TRỰC DIỆN)", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('homography_rectification.png', dpi=300)
print("=> Đã lưu kết quả Hiệu chỉnh Homography vào file: homography_rectification.png")
