import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import copy
import matplotlib.pyplot as plt
import csv

def main():
    print(f"\n{'='*60}")
    print(f"🚀 PHƯƠNG PHÁP: TRAIN FROM SCRATCH (Không dùng trọng số trước)")
    print(f"{'='*60}")
    
    data_dir = 'Concrete Crack Images for Classification'
    nested_dir = os.path.join(data_dir, 'Concrete Crack Images for Classification')
    if os.path.exists(nested_dir):
        data_dir = nested_dir
    if not os.path.exists(data_dir):
        print(f"LỖI: Không tìm thấy Dataset '{data_dir}'.")
        return

    print("=> Đang đọc Dataset hình ảnh vết nứt...")
    train_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(30/227, 30/227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir)
    num_classes = len(full_dataset.classes)
    
    total_size = len(full_dataset)
    # Giảm cực mạnh số lượng hình ảnh theo yêu cầu
    train_size = 10000
    val_size = 2000
    test_size = 2000
    remain_size = total_size - (train_size + val_size + test_size)
    
    # Cố định random seed để lúc chạy Transfer Learning bên kia cũng lấy ĐÚNG số ảnh này
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset, _ = random_split(
        full_dataset, 
        [train_size, val_size, test_size, remain_size],
        generator=generator
    )

    train_dataset.dataset = copy.deepcopy(full_dataset)
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms

    # Chỉnh Batch Size xuống 8 để mô hình vẫn cập nhật được nhiều lần dù dữ liệu ít
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=> Engine thực thi AI: {device}")

    print("=> Tải SqueezeNet...")
    model = models.squeezenet1_1(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Dùng SGD tốc độ chậm 
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 50
    
    print("\n[ĐÁNH GIÁ MÔ HÌNH TRƯỚC KHI HUẤN LUYỆN (EPOCH 0)]")
    model.eval()
    init_corrects = 0
    with torch.no_grad():
        for init_inputs, init_labels in val_loader:
            init_inputs = init_inputs.to(device)
            init_labels = init_labels.to(device)
            init_outputs = model(init_inputs)
            _, init_preds = torch.max(init_outputs, 1)
            init_corrects += torch.sum(init_preds == init_labels.data)
    init_acc = init_corrects.double() / val_size
    print(f"  => Độ chính xác khởi thủy (chưa học gì): {init_acc*100:.2f}% (kỳ vọng ~50% do đoán mò)")

    # Lưu kết quả để vẽ biểu đồ
    val_epochs = [0]
    val_accs = [init_acc.item() * 100]

    print("\n[BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN]")
    for epoch in range(epochs):
        print(f"\n- Kỷ nguyên (Epoch) {epoch+1}/{epochs}")
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()
                loader = train_loader
                dataset_size = train_size
            else:
                model.eval()
                loader = val_loader
                dataset_size = val_size
                
            running_loss = 0.0
            running_corrects = 0
            
            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                    print(f"    [{phase}] Đã chạy xong batch {i+1}/{len(loader)} | Loss hiện tại: {loss.item():.4f}")
                
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            print(f"  + {phase} >>> Hàm chi phí (Loss): {epoch_loss:.4f} | Độ chính xác (Acc): {epoch_acc*100:.2f}%")
            
            if phase == 'Val':
                val_epochs.append(epoch + 1)
                val_accs.append(epoch_acc.item() * 100)

    # ===== LƯU CSV =====
    csv_path = os.path.join(os.getcwd(), 'val_acc_curve_scratch.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_acc'])
        for ep, acc in zip(val_epochs, val_accs):
            writer.writerow([ep, f'{acc:.4f}'])
    print(f"=> [LƯU CSV] Đã lưu dữ liệu val_acc vào: {csv_path}")

    print("\n[BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST]")
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"    [Test] Đã xử lý batch {i+1}/{len(test_loader)} | Loss hiện tại: {loss.item():.4f}")
                
    total_test_loss = test_loss / test_size
    total_test_acc = test_corrects.double() / test_size
    print(f"\n=> KẾT QUẢ TEST Cuối Cùng >>> Hàm chi phí (Loss): {total_test_loss:.4f} | Độ chính xác (Acc): {total_test_acc*100:.2f}%\n")

    save_path = 'squeezenet_crack_scratch.pth'
    torch.save(model.state_dict(), save_path)
    print(f"=> XONG! Đã lưu trọng số mô hình vào: {save_path}")

    # ===== VE BIEU DO =====
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, val_accs, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
        plt.title('Validation Accuracy over 50 Epochs (Train from Scratch)\nDataset: 10000 Train / 2000 Val', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Accuracy (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 105)
        
        # Highlight Epoch 0 and Final Epoch
        plt.annotate(f'{val_accs[0]:.1f}%', xy=(val_epochs[0], val_accs[0]), xytext=(val_epochs[0]+1, val_accs[0]-5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
        plt.annotate(f'{val_accs[-1]:.1f}%', xy=(val_epochs[-1], val_accs[-1]), xytext=(val_epochs[-1]-5, val_accs[-1]-10),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
                     
        plot_path = os.path.join(os.getcwd(), 'val_acc_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"=> [VẼ BIỂU ĐỒ] Đã lưu biểu đồ thành công vào: {plot_path}")
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ: {e}")

if __name__ == '__main__':
    main()
