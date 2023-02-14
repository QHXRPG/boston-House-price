import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

"""数据预处理（非常重要）"""
data = load_boston()
boston_data = pd.DataFrame(data=data.data, columns=data.feature_names)
boston_data['price'] = data.target
corr1 = boston_data.corr(method="pearson")
corr2 = boston_data.corr(method="kendall")
corr3 = boston_data.corr(method="spearman")
plt.figure(figsize=(10,10))
plt.subplot(3,3,1)
sns.heatmap(corr1)
plt.subplot(3,3,5)
sns.heatmap(corr2)
plt.subplot(3,3,9)
sns.heatmap(corr3)
plt.show()
x = boston_data.iloc[:,:-1]
y = boston_data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8)
(x_train, x_test, y_train, y_test )= (np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test))
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)
min_max_scaler = MinMaxScaler()
(x_train, x_test, y_train, y_test) = (min_max_scaler.fit_transform(x_train),
                                      min_max_scaler.fit_transform(x_test),
                                      min_max_scaler.fit_transform(y_train),
                                      min_max_scaler.fit_transform(y_test)) #数据规范化

#%%相关性分析
p = pearsonr(x_train,y_train)

#%%
pca = PCA(n_components=5)
x_train, x_test = pca.fit_transform(x_train), pca.fit_transform(x_test)
#%% 方法一
linear1 = LinearRegression()
linear1.fit(x_train,y_train)
y_p1 = linear1.predict(x_test)

#%%方法二
x_train_, x_test_, y_train_, y_test_ = np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)
(x_train_, x_test_, y_train_, y_test_) = (torch.from_numpy(x_train_).type(torch.Tensor),
                                          torch.from_numpy(x_test_).type(torch.Tensor),
                                          torch.from_numpy(y_train_).type(torch.Tensor),
                                          torch.from_numpy(y_test_).type(torch.Tensor))
train_data = Data.TensorDataset(x_train_,y_train_)
data_loader = Data.DataLoader(dataset=train_data, batch_size=20)
test_data = Data.TensorDataset(x_test_,y_test_)
data_loader_test = Data.DataLoader(dataset=test_data, batch_size=1000)

class Linear_predict(nn.Module):
    def __init__(self):
        super(Linear_predict, self).__init__()
        self.model = nn.Sequential(nn.Linear(5,20),
                                   nn.ReLU(),
                                   nn.Linear(20,10),
                                   nn.ReLU(),
                                   nn.Linear(10,1))
    def forward(self,x):
        return self.model(x)
linear2 = Linear_predict()
loss = nn.MSELoss()
opt = torch.optim.Adam(linear2.parameters(), lr=0.001)
for i in range(2000):
    for _,(x,y) in enumerate(data_loader):
        y_p = linear2(x)
        y = y.unsqueeze(1)
        l = loss(y_p,y)
        opt.zero_grad()
        l.backward()
        opt.step()
    if i % 50 == 0:
        print(l.item())
for i,(x_t,y_t) in enumerate(data_loader_test):
    y_p2 = linear2(x_t)

y_p2 = y_p2.detach().numpy()
#%% 检验
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("1预测的均方误差：", mean_squared_error(y_p1, y_test))
print("2预测的均方误差：", mean_squared_error(y_p2, y_test))
