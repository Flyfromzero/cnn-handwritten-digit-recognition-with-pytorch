import torch

from data_prepare import test_loader
from model_cnn import net, criterion

test_losses = []

def test():
  #load the parameters of model from params.pkl
  net.load_state_dict(torch.load('params.pkl'))
  net.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      #process the data to adjust to the net
      if data.shape == torch.Size([4,28,1,28]):
        data = data.permute(0,2,1,3)
      data = data.float()
      target = target.t()
      target = target.squeeze()
      #test and evaluate the model
      output = net(data)
      test_loss += criterion(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      print(pred)
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print(f'\nTest set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}({(100.* correct / len(test_loader.dataset)):.0f}%)\n')
    
test()