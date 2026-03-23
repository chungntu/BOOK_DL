import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

f, D, L = 3, 10, 4
theta_deg = 45
theta = np.radians(theta_deg) 

ax.plot(0, 0, 'ko', markersize=8)

ax.axhline(0, color='gray', linestyle='-.', linewidth=1.5)

x_img_plane = np.array([-f, -f])
y_img_plane = np.array([-L, L])
ax.plot(x_img_plane, y_img_plane, 'b-', linewidth=1.5)

# Vật vuông góc
y_obj_perp = np.array([L/2, -L/2])
x_obj_perp = np.array([D, D])
ax.plot(x_obj_perp, y_obj_perp, 'r-', linewidth=3, label='Vật')
y_img_perp = y_obj_perp * (-f / D)
ax.plot(np.array([-f, -f]), y_img_perp, 'red', linewidth=5, label="Ảnh", alpha=0.5)
for i in range(2):
    ax.plot([x_obj_perp[i], -f], [y_obj_perp[i], y_img_perp[i]], 'r--', alpha=0.4)

# Vật nghiêng
s = np.array([L/2, -L/2])
x_obj_tilt = D + s * np.sin(theta)
y_obj_tilt = s * np.cos(theta)
ax.plot(x_obj_tilt, y_obj_tilt, 'g-', linewidth=3, label=fr'Vật nghiêng $\theta={theta_deg}^\circ$')
y_img_tilt = y_obj_tilt * (-f / x_obj_tilt)
ax.plot(np.array([-f, -f]), y_img_tilt, 'green', linewidth=4, label="Ảnh nghiêng", alpha=0.8)
for i in range(2):
    ax.plot([x_obj_tilt[i], -f], [y_obj_tilt[i], y_img_tilt[i]], 'g--', alpha=0.4)

ax.set_aspect('equal')
ax.set_xlim([-f-3, D+3])
ax.set_ylim([-L-1, L+1])
ax.set_title('Scale Ratio', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig('scale_ratio_example.png', dpi=300)
print("Đã tạo đồ thị scale_ratio_example.png thành công!")
