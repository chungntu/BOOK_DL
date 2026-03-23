import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from collections import OrderedDict
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Danh sách nhãn vật thể của tập dữ liệu COCO
COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 1. Nạp ảnh mẫu chuẩn từ máy
print("Đang nạp ảnh mẫu từ file cục bộ...")
try:
    img = Image.open('dog.jpg').convert("RGB")
except FileNotFoundError:
    raise FileNotFoundError("Không tìm thấy file dog.jpg ở thư mục hiện tại.")
img_tensor = torchvision.transforms.functional.to_tensor(img)

# 2. Khởi tạo mạng Faster R-CNN
print("Đang khởi tạo mạng Faster R-CNN ResNet-50 FPN...")
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# Tiến hành giải phẫu 2 Giai đoạn
with torch.no_grad():
    # --- TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG CƠ SỞ ---
    images, _ = model.transform([img_tensor], None)
    
    # Backbone (ResNet + FPN) trích xuất Feature Maps
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])
        
    # --- GIAI ĐOẠN 1: REGION PROPOSAL NETWORK (RPN) ---
    print("\n=> Đang chạy Giai đoạn 1 (RPN)...")
    # Đầu ra: Hàng ngàn Anchor Boxes thô sơ (Region Proposals)
    proposals, _ = model.rpn(images, features, None)
    rpn_boxes = proposals[0].cpu().numpy()
    print(f"Giai đoạn 1 hoàn tất! RPN đề xuất {len(rpn_boxes)} vùng tiềm năng.")

    # --- GIAI ĐOẠN 2: R-CNN HEAD (RoI Pooling + Classifier + Regressor) ---
    print("\n=> Đang chạy Giai đoạn 2 (RoI Pooling & R-CNN Head)...")
    # Thay vì tự code RoI Pooling, ta gọi hàm roi_heads có sẵn của PyTorch.
    # Hàm này gọt ép các `proposals` lộn xộn về chuẩn 7x7, chạy qua classifier dán nhãn 
    # và chạy qua regressor tinh chỉnh lại độ khít của khung box.
    detections, _ = model.roi_heads(features, proposals, images.image_sizes, None)
    
    fast_rcnn_boxes = detections[0]['boxes'].cpu().numpy()
    fast_rcnn_labels = detections[0]['labels'].cpu().numpy()
    fast_rcnn_scores = detections[0]['scores'].cpu().numpy()
    print(f"Giai đoạn 2 hoàn tất! Mạng chốt hạ được {len(fast_rcnn_boxes)} đối tượng an toàn.")

# --- TRỰC QUAN HÓA SO SÁNH 2 GIAI ĐOẠN ---
orig_resized = images.tensors[0].cpu().numpy().transpose(1, 2, 0)
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
orig_resized = np.clip(orig_resized * std + mean, 0, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# HÌNH 1: KẾT QUẢ GIAI ĐOẠN 1 (RPN)
ax1.imshow(orig_resized)
top_k_rpn = 30
for i in range(top_k_rpn):
    box = rpn_boxes[i]
    is_top = i < 10
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                             linewidth=3.5 if is_top else 1.5, 
                             edgecolor='red' if is_top else 'lime', 
                             facecolor='none', alpha=0.9 if is_top else 0.5)
    ax1.add_patch(rect)
ax1.axis('off')

# HÌNH 2: KẾT QUẢ GIAI ĐOẠN 2 (R-CNN HEAD)
ax2.imshow(orig_resized)
# Chỉ hiển thị các box có độ tự tin cao > 70% sau phân loại
confidence_threshold = 0.7
drawn_count = 0
for i in range(len(fast_rcnn_boxes)):
    if fast_rcnn_scores[i] < confidence_threshold:
        continue
    drawn_count += 1
    box = fast_rcnn_boxes[i]
    label_idx = fast_rcnn_labels[i]
    label_name = COCO_NAMES[label_idx]
    score = fast_rcnn_scores[i]
    
    # Khung tinh chỉnh cuối cùng (Regressed Box)
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                             linewidth=4, edgecolor='dodgerblue', facecolor='none')
    ax2.add_patch(rect)
    
    # Gắn nhãn phân loại (Classification Label)
    text = f"{label_name.upper()} {score:.2f}"
    ax2.text(box[0], box[1]-8, text, color='white', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='dodgerblue', edgecolor='none', alpha=0.8, pad=2))

ax2.axis('off')

# LƯU FILE
plt.tight_layout(pad=0)
plt.savefig('two_stage_detection.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("\n=> Đã xuất đồ thị giải phẫu học HỆ DETECTOR 2 GIAI ĐOẠN (Trắng chữ tiêu đề) thành công vào file: two_stage_detection.png")
