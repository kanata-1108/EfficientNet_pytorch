import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import os

os.chdir("/src/EfficientNet")

# データの前処理
trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias = True), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# テスト用データ
batchsize = 32
test_dataset = datasets.CIFAR10(root = ".", train = False, download = True, transform = trans)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batchsize, shuffle = False)

# テスト用データの数
data_num = len(test_dataset.data)

# 各クラスごとの枚数（各クラス1000枚）
class_num = (torch.bincount(torch.tensor(test_dataset.targets))).tolist()

# GPUが使えるなら使う
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# EfficientNetの読み込み
update_efficient_net = models.efficientnet_b0()

# １０クラス分類にするための出力層の調整
update_efficient_net.classifier = nn.Sequential(
    nn.Dropout(p = 0.5, inplace = True),
    nn.Linear(in_features = update_efficient_net.classifier[1].in_features, out_features = 10)
)

update_efficient_net.to(device)
update_efficient_net.load_state_dict(torch.load("update_efficientnet.pth", map_location = device))
update_efficient_net.eval()

acc_sum = 0
predicted_class = [0] * 10

with torch.no_grad():
    for img, label in test_loader:
        imgs, labels = img.to(device), label.to(device)
        outputs = update_efficient_net(imgs)
        for out, lbl in zip(outputs, labels):
            pre_index = torch.argmax(out)
            if pre_index == lbl:
                predicted_class[pre_index] += 1

print("-----accuracy-----")
for i in range(10):
    print("%10s : %2d %%" % (test_dataset.classes[i], 100 * (predicted_class[i] / class_num[i])))