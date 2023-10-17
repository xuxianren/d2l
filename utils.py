from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def load_data_fashion_mnist(batch_size):
    trans = transforms.ToTensor()
    training_dateset = datasets.FashionMNIST(
        root="./data",
        transform=trans,
        train=True,
        download=True,
    )
    test_dataset = datasets.FashionMNIST(
        root="./data",
        transform=trans,
        train=False,
        download=True,
    )
    train_dateloader = DataLoader(training_dateset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dateloader, test_dataloader


def train(m, train_loader, loss_fn, optimizer, epoch_num):
    for epoch in range(epoch_num):
        for X, y in train_loader:
            y_hat = m(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch: {}, loss: {}".format(epoch, loss))