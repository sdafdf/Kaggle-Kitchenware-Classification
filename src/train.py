from torch import optim,nn
from tqdm import tqdm
from dataset import create_dataloader
from network import resnet34
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def evaluate(model, loader):  # 计算每次训练后的准确率
#     correct = 0
#     total = len(loader.dataset)
#     for x, y in loader:
#         x = x.to(device)
#         y = y.to(device)
#         logits = model(x)
#         pred = torch.argmax(logits,1)  # 得到logits中分类值（要么是[1,0]要么是[0,1]表示分成两个类别）
#         correct += torch.eq(pred, y).sum().float().item()  # 用logits和标签label想比较得到分类正确的个数
#     return correct / total


def train():
    epochs=10
    best_acc=0
    net = resnet34(6).to(device)
    data_path = "F:\\PythonProject\\deep_code\\Kitchenware Classification\\dataset\\"
    dataloader = create_dataloader(data_path,mode="train",size=224,batch_size=64)
    test_dataloader = create_dataloader(data_path,"test",size=224,batch_size=64)
    optimizer = optim.Adam(net.parameters(),lr=0.01)
    cirteron = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss=0
        train_acc=[]
        train_losses=[]
        num_correct=0
        net.train()
        pbar = tqdm(enumerate(dataloader),total=len(dataloader))
        for i,(img,label) in pbar:
            out = net(img.to(device))
            label = label.to(device)
            predict_lab = torch.argmax(out,1)

            loss = cirteron(out,label.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            num_correct+=torch.eq(predict_lab,label).sum().float().item()
            
            pbar.set_description(f'Epoch [{epoch}/{epochs}]')
            pbar.set_postfix(loss=loss.item())
        train_acc.append(num_correct/len(dataloader.dataset))
        train_losses.append(train_loss/len(dataloader))
        print("train loss:",train_loss/len(dataloader))
        print("train acc: ",num_correct/len(dataloader.dataset))
        
        net.eval()
        val_losses=0
        val_acc=0
        val_loss_list = []
        val_acc_list = []
        pbar_val = tqdm(enumerate(test_dataloader),total=len(test_dataloader))
        for _,(img,lab) in pbar_val:
            img = img.to(device)
            lab = lab.to(device)
            output = net(img)

            val_loss = cirteron(output,lab)
            predict_val = torch.argmax(output,1)
            val_losses += val_loss.item()
            val_acc +=torch.eq(predict_val,lab).sum().item()
            pbar_val.set_description(f'Epoch [{epoch}/{epochs}]')
            pbar_val.set_postfix(val_loss=val_loss.item())
        val_loss_list.append(val_losses/len(test_dataloader))
        val_acc_list.append(val_acc/len(test_dataloader.dataset))
        print("val_acc:",val_acc/len(test_dataloader.dataset))
        print("val_loss:",val_losses/len(test_dataloader))

        torch.save(net,f'F:\\PythonProject\\deep_code\\Kitchenware Classification\\checkpoint\\epoch{epoch}_{val_acc/len(test_dataloader.dataset)}.pth')
        if best_acc<(val_acc/len(test_dataloader.dataset)):
            best_acc=(val_acc/len(test_dataloader.dataset))
            torch.save(net,"F:\\PythonProject\\deep_code\\Kitchenware Classification\\checkpoint\\best.pth")
            




if __name__=="__main__":
    train()



