import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import LeNet
from torch import nn
from torch import optim

if __name__ == '__main__':
    batch_size = 256
    learning_rate = 1e-1
    epoch = 100
    device = torch.device("cuda:0")  # 利用cuda加速
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])  # 最高正确率100%
    # transform = transforms.Compose([transforms.ToTensor()])
    # MNIST 训练集有 60000 张图片
    train_data = torchvision.datasets.MNIST(root='./MNIST', train=True, transform=transform)
    # MNIST 测试集有 10000 张图片
    test_data = torchvision.datasets.MNIST(root='./MNIST', train=False, transform=transform)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=train_data, batch_size=batch_size)  # 测试集合可以不用shuffle
    model = LeNet()
    model.to(device)
    # 设置损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    # 设置优化器(这里用随机梯度下降, 学习率为1e-1)
    optimizer = optim.SGD(model.parameters(), learning_rate)

    for current_epoch in range(epoch):
        model.train()
        for step, (train_imgs, train_labels) in enumerate(train_dataloader):  # for一次执行的是一个batch_size
            train_imgs = train_imgs.to(device)
            train_labels = train_labels.to(device)
            optimizer.zero_grad()  # 梯度清零
            predict_y = model(train_imgs)
            loss = loss_fn(predict_y, train_labels)  # 返回的是基于当前feed进模型所有数据所构成的loss
            if step % 10 == 0:
                print(f"current_epoch: {current_epoch + 1}, step: {step}, loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        with torch.no_grad():
            for step, (test_imgs, test_labels) in enumerate(test_dataloader):
                test_imgs = test_imgs.to(device)
                test_labels = test_labels.to(device)
                predict_y = model(test_imgs)  # predict_y的每一行的中数值大小表明该位置是其对应labels位置上的可能。数值越大越可能
                predict_y = predict_y.argmax(-1)  # 得到的是一维的tensor数据
                current_correct_num = predict_y == test_labels  # 首先返回predict_y == test_labels对应位置上的T或F
                all_correct_num += current_correct_num.sum().item()
                all_sample_num += test_labels.shape[0]
        acc = (all_correct_num / all_sample_num) * 100
        print(f'accuracy: {acc:.3f}')
        torch.save(model.state_dict(), f'./Models/mnist_{acc:.3f}.pth')  # torch.save() 若要存储指定路径，且路径包含文件夹必须手动创建。
