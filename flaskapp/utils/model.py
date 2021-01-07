import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentClassifier(nn.Module):
    def __init__(self, n_words, n_classes):
        super().__init__()
        self.dense_init = nn.Linear(n_words, 128)
        self.drop1 = nn.Dropout(0.5)
        self.dense = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.5)
        self.dense_final = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.dense_init(x))
        x = self.drop1(x)
        x =  F.relu(self.dense(x))
        x = self.drop2(x)
        x = self.dense_final(x)
        return x