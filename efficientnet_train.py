import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.chdir("/src/EfficientNet")

# データの前処理
trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias = True), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 訓練用データ
train_dataset = datasets.CIFAR10(root = ".", train = True, download = True, transform = trans)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

# テスト用データ
test_dataset = datasets.CIFAR10(root = ".", train = False, download = True, transform = trans)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True)

# EfficientNetの読み込み
efficient_net = models.efficientnet_b0()

# １０クラス分類にするための出力層の調整
efficient_net.classifier = nn.Sequential(
    nn.Dropout(p = 0.5, inplace = True),
    nn.Linear(in_features = efficient_net.classifier[1].in_features, out_features = 10)
)

def train_model(model, epochs, loader):

    # GPUが使えるなら使う
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # EfficientNetをGPUに転送
    model.to(device)
    
    model.train()

    # 誤差関数と最適化関数の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    # 学習回数ごとの誤差を格納するリスト
    losses = []

    # 学習回数ごとの正解率を格納するリスト
    accuracy = []

    for epoch in range(epochs):

        sum_loss = 0
        acc_sum = 0

        print("epoch : {}".format(epoch + 1))

        for img, label in tqdm(loader):
            imgs, labels = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            acc_sum += (outputs.argmax(1) == labels).sum().item()
        
        epoch_loss = sum_loss / 32
        train_accuracy = acc_sum / 50000
        losses.append(epoch_loss)
        accuracy.append(train_accuracy)
        print("loss : {} || accuracy : {}".format(epoch_loss, train_accuracy))

    return losses, accuracy, model

Epoch = 20
loss_list, acc_list, update_model = train_model(model = efficient_net, epochs = Epoch, loader = train_loader)
torch.save(update_model.state_dict(), "./update_efficientnet.pth")

plt.plot(range(Epoch), loss_list, c = "orangered")
plt.title("train Loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.xlim(0, Epoch)
plt.savefig("./loss_graph.png")
plt.clf()
plt.plot(range(Epoch), acc_list, c = "royalblue")
plt.title("train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.xlim(0, Epoch)
plt.savefig("./accuracy_graph.png")