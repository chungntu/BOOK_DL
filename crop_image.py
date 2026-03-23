import cv2
import os
import tkinter as tk
from tkinter import filedialog

def main():
    # 1. Khởi tạo cửa sổ nền ẩn của Tkinter để dùng giao diện chọn file
    root = tk.Tk()
    root.withdraw()
    
    print("=> Vui lòng chọn file ảnh JPG cần cắt...")
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để cắt (Crop)",
        filetypes=[("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All Files", "*.*")]
    )
    
    if not file_path:
        print("Đã hủy thao tác chọn file.")
        return

    # 2. Đọc ảnh bằng OpenCV
    img = cv2.imread(file_path)
    if img is None:
        print("Lỗi: Không thể đọc được file ảnh. Vui lòng kiểm tra lại đường dẫn.")
        return

    print("\n--- HƯỚNG DẪN CẮT ẢNH ---")
    print("1. Kéo giữ chuột TRÁI để khoanh vùng (Region of Interest - ROI).")
    print("2. Nhấn phím SPACE hoặc ENTER để xác nhận vùng đã chọn.")
    print("3. Nhấn phím 'c' để vẽ lại nếu chọn sai.")
    print("4. Nhấn phím ESC để hủy bỏ toàn bộ.\n")
    
    # 3. Kích hoạt chế độ dùng chuột chọn vùng (ROI)
    # Tự động thay đổi kích thước cửa sổ nếu ảnh quá to để đỡ lấn màn hình
    cv2.namedWindow("Chon vung anh (Keo chuot -> Nhan SPACE)", cv2.WINDOW_NORMAL)
    roi_coords = cv2.selectROI("Chon vung anh (Keo chuot -> Nhan SPACE)", img, fromCenter=False, showCrosshair=True)
    
    # Lấy tọa độ (x, y) là gốc trái trên cùng, (w, h) là chiều rộng/cao
    x, y, w, h = int(roi_coords[0]), int(roi_coords[1]), int(roi_coords[2]), int(roi_coords[3])
    
    # Đóng giao diện ảnh sau khi đã chốt tọa độ
    cv2.destroyAllWindows()
    
    # Đoạn này xử lý trường hợp người dùng ấn ESC hoặc ấn Enter rỗng mà không kéo chuột (w=0, h=0)
    if w == 0 or h == 0:
        print("Đã hủy quá trình cắt ảnh (Không có vùng nào được chọn).")
        return

    # 4. Cắt rời vùng ảnh (Cắt ma trận theo cú pháp [y:y+h, x:x+w])
    cropped_img = img[y:y+h, x:x+w]
    
    # 5. Hỏi User vị trí và tên file muốn lưu thành
    print("=> Tốt lắm! Vui lòng chọn nơi lưu file ảnh mới...")
    save_path = filedialog.asksaveasfilename(
        title="Lưu vùng ảnh mới",
        defaultextension=".jpg",
        initialfile="cropped_" + os.path.basename(file_path),
        filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")]
    )
    
    if save_path:
        # Lưu file mới ra máy
        cv2.imwrite(save_path, cropped_img)
        print(f"\n=> XONG! Đã cắt và lưu file thành công tại:\n=> {save_path}")
    else:
        print("Hủy thao tác lưu file.")

if __name__ == "__main__":
    main()
