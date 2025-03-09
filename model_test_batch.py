import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

def test_data_process():
    test_dataset = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                              download=True
                              )

    test_dataloader = Data.DataLoader(dataset=test_dataset,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=6)

    return test_dataloader

def test_model_process(model,test_dataloader):
    cloth = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coar', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    test_correct = 0.0
    test_num = 0

    with torch.no_grad():
        for step,(test_data_x,test_data_y) in enumerate(test_dataloader):
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            pre_label = torch.argmax(output, dim=1)
            one_batch_acc = torch.sum(pre_label == test_data_y)
            test_correct += one_batch_acc
            print(f"Batch {step}: {one_batch_acc}/{test_data_x.size(0)}")
            test_num += test_data_x.size(0)
    test_acc = test_correct.double().item() / test_num
    print(f"准确率：{test_acc}")

if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(torch.load('./model/best_model4.pth'))
    test_dataloader = test_data_process()
    test_model_process(model,test_dataloader)
