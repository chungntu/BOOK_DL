import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Khởi tạo đồ thị 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# --- ĐỊNH NGHĨA CÁC THÔNG SỐ VÀ TỌA ĐỘ ---
# Trục X sẽ là trục quang học (từ trái qua phải)
# Vật thể ở X > 0 (bên phải), Pinhole ở X = 0 (giữa), Mặt phẳng ảnh ở X < 0 (bên trái)

O = np.array([0, 0, 0])  # Tọa độ trung tâm camera (Pinhole)
f = 5                    # Tiêu cự (Mặt phẳng ảnh tại X = -f)
D = 10                   # Khoảng cách từ vật thể đến Pinhole (X = D)

# Định nghĩa một hình đơn giản: "Ngôi nhà" 2D nằm trên mặt phẳng X = 10
# Các điểm (X, Y, Z)
points_3d = np.array([
    [D, -1.5, 0],   # P1: Đáy trái
    [D, 1.5, 0],    # P2: Đáy phải
    [D, 1.5, 3],    # P3: Tường phải
    [D, -1.5, 3],   # P4: Tường trái
    [D, 0, 5]       # P5: Đỉnh mái nhà
])

# Các đường nối các điểm để tạo thành hình ngôi nhà
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Hình chữ nhật (thân nhà)
    (2, 4), (3, 4)                   # Mái nhà
]

# Tính toán các điểm trên mặt phẳng ảnh tại X = -f
# Theo tính chất đồng dạng: y / Y = -f / D => y = Y * (-f / D)
# z / Z = -f / D => z = Z * (-f / D)
scale = -f / D
points_2d = np.zeros_like(points_3d)
for i, P in enumerate(points_3d):
    points_2d[i] = [P[0] * scale, P[1] * scale, P[2] * scale]

# --- VẼ CÁC THÀNH PHẦN ---

# 1. Vẽ Mặt phẳng ảnh (Image Plane) ở X = -f (Bên Trái)
plane_y = np.array([-3, 3, 3, -3])
plane_z = np.array([-3, -3, 3, 3])
plane_x = np.array([-f, -f, -f, -f])
verts_img = [list(zip(plane_x, plane_y, plane_z))]
plane_img = Poly3DCollection(verts_img, alpha=0.3, facecolor='cyan', edgecolor='blue')
ax.add_collection3d(plane_img)
ax.text(-f, -3, 3.5, "Mặt phẳng ảnh", color='blue', fontsize=11, fontweight='bold')

# Bức tường (Pinhole Plane) ở X = 0 (Giữa)
plane_x_pinhole = np.array([0, 0, 0, 0])
verts_pinhole = [list(zip(plane_x_pinhole, plane_y, plane_z))]
plane_pinhole = Poly3DCollection(verts_pinhole, alpha=0.1, facecolor='gray', edgecolor='black')
ax.add_collection3d(plane_pinhole)
ax.text(0, -3.5, 3.5, "Mặt phẳng Pinhole", color='black', fontsize=11, fontweight='bold')

# 2. Vẽ trung tâm camera (Pinhole O)
ax.scatter([O[0]], [O[1]], [O[2]], color='black', s=50)
ax.text(O[0], O[1], O[2]-0.5, "O", color='black', fontsize=12, fontweight='bold')

# 3. Vẽ Trục quang học (Optical Axis - Trục X)
ax.plot([-f-2, D+2], [0, 0], [0, 0], color='gray', linestyle='-.', linewidth=1.5, label='Trục quang học (Trục X)')

# 4. Vẽ Vật thể thực (Bên Phải)
for edge in edges:
    pA, pB = points_3d[edge[0]], points_3d[edge[1]]
    ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], color='red', linewidth=3)
ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], color='darkred', s=40)
ax.text(D, 0, 6, "Vật thật", color='red', fontsize=12, fontweight='bold', ha='center')

# 5. Vẽ Ảnh trên mặt phẳng ảnh (Bên Trái) - Sẽ bị ngược
for edge in edges:
    pA, pB = points_2d[edge[0]], points_2d[edge[1]]
    ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], color='blue', linewidth=3)
ax.scatter(points_2d[:,0], points_2d[:,1], points_2d[:,2], color='darkblue', s=40)
ax.text(-f, 0, -4, "Ảnh", color='blue', fontsize=11, fontweight='bold', ha='center')

# 6. Vẽ các tia chiếu (Light rays)
for i in range(len(points_3d)):
    P = points_3d[i]
    p = points_2d[i]
    ax.plot([p[0], P[0]], [p[1], P[1]], [p[2], P[2]], color='orange', linestyle='--', linewidth=1)

# --- ĐỊNH DẠNG ĐỒ THỊ ---
ax.set_xlabel('Trục X (Quang học)')
ax.set_ylabel('Trục Y')
ax.set_zlabel('Trục Z')
ax.set_title('Mô hình Camera Pinhole', fontsize=15, fontweight='bold')

# Thiết lập giới hạn các trục
ax.set_xlim([-f-1, D+1])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 6])

# Thay đổi góc nhìn để dễ thấy sự đảo ngược và thứ tự Trái - Giữa - Phải
# elev (độ cao): 15 độ, azim (góc xoay quanh Z): -45 độ -> X hướng từ trái qua phải, Y hướng vào trong
ax.view_init(elev=15, azim=-60)

# Ẩn các lưới để nhìn rõ mô hình hơn (Tùy chọn)
ax.grid(True)

plt.tight_layout()
plt.savefig('pinhole_model_inverted.png', dpi=300)
print("Đã tạo thành công ảnh pinhole_model_inverted.png!")
plt.show()
