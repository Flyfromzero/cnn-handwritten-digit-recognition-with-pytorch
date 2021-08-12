import torch
from matplotlib import pyplot as plt

from data_prepare import  train_loader
from model_cnn import net, criterion, optimizer

train_losses = []
train_counter = []

def train(epoch):
  net.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    #process the data to adjust to the net
    if data.shape == torch.Size([4,28,1,28]):
      data = data.permute(0,2,1,3)
    data = data.float()
    target = target.t()
    target = target.squeeze()
    #train the model
    output = net(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({(100. * batch_idx / len(train_loader)):.0f}%)]\tLoss:{loss.item():.6f}")
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*100) + ((epoch-1)*len(train_loader.dataset)))
      #save the parameters of model in params.pkl
      torch.save(net.state_dict(), 'params.pkl')
  # get the loss figure
  plt.plot(train_losses)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.show()
          
train(1)