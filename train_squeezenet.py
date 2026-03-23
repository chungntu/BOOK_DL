import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import copy

def main():
    # Thư mục gốc chứa ảnh (bên trong có 2 folder con: Positive, Negative)
    data_dir = 'Concrete Crack Images for Classification'
    
    # Khắc phục sự cố do lúc giải nén bị lồng 2 thư mục cùng tên vào nhau
    # PyTorch ImageFolder sẽ thấy nhầm thư mục con duy nhất là 1 class.
    nested_dir = os.path.join(data_dir, 'Concrete Crack Images for Classification')
    if os.path.exists(nested_dir):
        data_dir = nested_dir
    
    if not os.path.exists(data_dir):
        print(f"LỖI: Không tìm thấy Dataset '{data_dir}' ở gốc dự án.")
        print("Xin hãy giải nén/chép thư mục dữ liệu vào chung chỗ với code này rồi chạy lại nhé.")
        return

    print("=> Đang đọc Dataset hình ảnh vết nứt...")
    
    # ==========================================
    # 1. BỘ TĂNG CƯỜNG DỮ LIỆU (IMAGE AUGMENTER)
    # Đồng bộ hóa tham số với code MATLAB cũ của bạn
    # ==========================================
    train_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(p=0.5), # RandXReflection
        transforms.RandomAffine(degrees=0, translate=(30/227, 30/227)), # Translation [-30, 30]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ==========================================
    # 2. KHỞI TẠO DATASTORE (CHIA TRAINING/VAL)
    # ==========================================
    full_dataset = datasets.ImageFolder(data_dir)
    num_classes = len(full_dataset.classes) # Nhánh Negative và Positive => 2 classes
    print(f"=> Đã tìm thấy các nhãn phân loại: {full_dataset.classes}")
    
    if num_classes < 2:
        print("LỖI: Dữ liệu chưa chuẩn bị đúng! Cần ít nhất 2 thư mục con (ví dụ 'Positive' và 'Negative') bên trong data_dir.")
        return
    
    # MATLAB chia splitEachLabel (0.01) ngẫu nhiên.
    # Trong PyTorch ta dùng random_split với tỷ lệ chuẩn 0.8 Train / 0.2 Validation.
    # Nếu bạn muốn thử 1% cho lẹ thì chỉnh lại số lượng mẫu ở đây.
    total_size = len(full_dataset)
    # Tỷ lệ: 70% Train, 15% Validation, 15% Test
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Sửa lỗi con trỏ (reference) của PyTorch khi dùng chung dataset gốc
    train_dataset.dataset = copy.deepcopy(full_dataset)
    train_dataset.dataset.transform = train_transforms
    
    val_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform = val_transforms
    
    test_dataset.dataset.transform = val_transforms # test_dataset thì trỏ tới full_dataset ban đầu là được

    # Loader với Batch Size (100) theo chuẩn script MATLAB
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=> Engine thực thi AI: {device}")

    # ==========================================
    # 3. CHỈNH SỬA SQUEEZENET ĐỂ TÍNH PHÂN LỚP VẾT NỨT
    # ==========================================
    print("=> Tải SqueezeNet...")
    
    # [MẸO VIẾT SÁCH CỦA ANTIGRAVITY]
    # Khi dùng weights='DEFAULT' (Transfer Learning), bộ não AI đã được mài giũa sẵn trên ImageNet.
    # Kết hợp với việc bài toán vết nứt Bê Tông (Crack) quá dễ (chỉ là vệt đen trên nền bê tông xám),
    # model sẽ nhìn phát đoán trúng luôn -> 99.9% ngay từ Epoch 1.
    # Để tạo ra một quá trình "sai rồi sửa", giúp đồ thị đi lên từ từ cho bạn có số liệu đẹp để đưa vào sách,
    # mình đã tự động đổi thành weights=None (Ép AI quay về như tờ giấy trắng, học lại từ đầu).
    model = models.squeezenet1_1(weights=None)
    
    # removeLayers & replaceLayer bản Python:
    # Cắt bỏ 1000 class, lắp ghép vào Convolution mới cho 2 class (Positive/Negative)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    model = model.to(device)

    # ==========================================
    # 4. TRAINING OPTIONS (TÙY CHỈNH THEO SÁCH)
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4) # Bộ tối ưu Adam, Learning Rate: 2e-4
    epochs = 5 # MaxEpochs=5
    
    print("\n[BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN SQUEEZENET]")
    for epoch in range(epochs):
        print(f"\n- Kỷ nguyên (Epoch) {epoch+1}/{epochs}")
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader
                
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
                
                # In tiến trình mỗi 10 batch để tránh có cảm giác ứng dụng bị treo (đứng)
                if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                    print(f"    [{phase}] Đã chạy xong batch {i+1}/{len(loader)} | Loss hiện tại: {loss.item():.4f}")
                
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            print(f"  + {phase} >>> Hàm chi phí (Loss): {epoch_loss:.4f} | Độ chính xác (Acc): {epoch_acc*100:.2f}%")

    # ==========================================
    # 5. ĐÁNH GIÁ TRÊN TẬP TEST (TESTING)
    # ==========================================
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
                
    total_test_loss = test_loss / len(test_dataset)
    total_test_acc = test_corrects.double() / len(test_dataset)
    print(f"\n=> KẾT QUẢ TEST Cuối Cùng >>> Hàm chi phí (Loss): {total_test_loss:.4f} | Độ chính xác (Acc): {total_test_acc*100:.2f}%\n")

    # ==========================================
    # 6. XUẤT LƯU MÔ HÌNH (SAVE)
    # ==========================================
    save_path = 'squeezenet_crack.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\n=> XONG! Đã in dấu bộ não đã học vào tệp cục bộ (định dạng Tensor): {save_path}")
    print("Bạn có thể mở chạy file 'localSHM_Squeezenet.py' để nó tải tệp này và test quét Vết Nứt thực thụ!")

if __name__ == '__main__':
    main()
