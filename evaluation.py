import torch
import sys
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
np.set_printoptions(suppress=False)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear(x)
        return self.sigmoid(out)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
valid  = pd.read_csv("valid.csv")

xtrain = train[['age','anaemia','creatinine_phosphokinase','diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']].astype(np.float32)
ytrain = train['DEATH_EVENT'].astype(np.float32)

xtest = test[['age','anaemia','creatinine_phosphokinase','diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']].astype(np.float32)
ytest = test['DEATH_EVENT'].astype(np.float32)

xTrain = torch.from_numpy(xtrain.values)
yTrain = torch.from_numpy(ytrain.values.reshape(179,1))

xTest = torch.from_numpy(xtest.values)
yTest = torch.from_numpy(ytest.values)

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
learning_rate = 0.002
input_dim = 11
output_dim = 1

model = LogisticRegressionModel(input_dim, output_dim)
model.load_state_dict(torch.load('DEATH_EVENT.pth'))
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


prediction= model(xTest)


accuracy_score = accuracy_score(yTest, np.argmax(prediction.detach().numpy(), axis=1))
print("accuracy_score", accuracy_score)
print("F1", f1_score(yTest, np.argmax(prediction.detach().numpy(), axis=1), average=None))

with open("metrics.txt", 'w') as outfile:
    outfile.write("Accuracy: " + f1_score(yTest, np.argmax(prediction.detach().numpy(), axis=1), average=None) + "\n")