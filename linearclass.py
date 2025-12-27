from torch import nn
import torch
torch.manual_seed(1)
class linear_regression(nn.Module):
    def __init__(self,input_size,output_size):
        super(linear_regression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        yhat=self.linear(x)
        return yhat
model=linear_regression(1,10)
model(torch.tensor([1.0]))
# we can see the parameters
list(model.parameters())
# we can create a tensor with two rows representing one sample of data
x=torch.tensor([[1.0]])
# we can make a prediction
yhat=model(x)
yhat
# each row in the following tensor represents a different sample
X=torch.tensor([[1.0],[1.0],[3.0]])
# we can make a prediction using multiple samples
Yhat=model(X)
Yhat