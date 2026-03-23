import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. TẢI ẢNH GỐC TỪ FILE
img_path = 'crack.jpg'
img_original_color = cv2.imread(img_path)

if img_original_color is None:
    raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {img_path}")

# Thuật toán truyền thống trên cường độ sáng yêu cầu ảnh xám
img_original = cv2.cvtColor(img_original_color, cv2.COLOR_BGR2GRAY)


# ==========================================
# 2. XỬ LÝ THEO PHƯƠNG PHÁP TRUYỀN THỐNG (SHM)
# ==========================================

# A. Tiền xử lý (Bắt buộc): Làm mịn ảnh Gaussian để khử bớt hạt nhiễu của bê tông
img_smoothed = cv2.GaussianBlur(img_original, (5, 5), 0)

# B. Otsu Thresholding (Phân ngưỡng tự động)
# Sử dụng THRESH_BINARY_INV để nền xám biến thành đen (0), nứt đen thui rực lên thành trắng (255)
ret_otsu, thresh_otsu = cv2.threshold(img_smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# C. Canny Edge Detection (Toán tử phát hiện biên)
# Lấy ra các đường viền Gradient sắc bén nhất
edges_canny = cv2.Canny(img_smoothed, 40, 120)


# ==========================================
# 3. TRỰC QUAN HÓA SO SÁNH VÀ LƯU FILE
# ==========================================
fig, axs = plt.subplots(1, 4, figsize=(22, 6))

axs[0].imshow(img_original, cmap='gray', vmin=0, vmax=255)
axs[0].set_title("1. Bề mặt Bê tông (Nứt)", fontsize=16, fontweight='bold')
axs[0].axis('off')

axs[1].imshow(img_smoothed, cmap='gray', vmin=0, vmax=255)
axs[1].set_title("2. Tiền xử lý (Gaussian Filter)", fontsize=16, fontweight='bold')
axs[1].axis('off')

axs[2].imshow(thresh_otsu, cmap='gray')
axs[2].set_title(f"3. Otsu Thresholding\n(Ngưỡng tự động T = {int(ret_otsu)})", fontsize=16, fontweight='bold', color='blue')
axs[2].axis('off')

axs[3].imshow(edges_canny, cmap='gray')
axs[3].set_title("4. Canny Edge Detection\n(Rút trích đường biên mảnh)", fontsize=16, fontweight='bold', color='red')
axs[3].axis('off')

plt.tight_layout()
plt.savefig('crack_detection_traditional.png', dpi=300)
print("=> Đã lưu kết quả phân tích vết nứt truyền thống vào file: crack_detection_traditional.png")
