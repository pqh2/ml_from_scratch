import pandas as pd
import numpy as np


def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.reshape(np.max(x, axis=1), (x.shape[0],1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis = 0)

data = pd.read_csv("data.csv")

y_train = data['income'].values
y_train = y_train == ">50K"
y_train = np.array([np.array([0, 1]) if label else np.array([1, 0]) for label in y_train]) 

age = data['age'].values
oc_l = list(set(data['occupation'].values))
oc_v = []

for v in data['occupation'].values:
    one_hot = [0 for j in range(len(oc_l))]
    for j in range(len(oc_l)):  
        if v == oc_l[j]:
            one_hot[j] = 1
    oc_v.append(np.array(one_hot))

oc_v = np.array(oc_v)

x_train = np.column_stack((age, oc_v, data['hours.per.week']))
x_train, y_train, x_valid, y_valid = x_train[:30000], y_train[:30000], x_train[30000:], y_train[30000:]

W1 = np.random.rand(x_train.shape[1], 10)
b1 = np.random.rand(1, 10)

W2 = np.random.rand(10, 2)
b2 = np.random.rand(1, 2)-0.5


lr = 0.0001
for i in range(100):
    x = x_train[i*32:(i+1) * 32]
    y = y_train[i*32:(i+1) * 32]
    z1 = np.matmul(x, W1) + b1
    #print z1
    rel1 = np.maximum(z1, 0)
    z2 = np.matmul(rel1, W2) + b2
    rel2 = np.maximum(z2, 0)
    softmax = stablesoftmax(rel2)
    #print softmax
    
    dLDRelu2 = rel2 - softmax / 32
    relu2_arr = np.array(z2 > 0, dtype=np.int32)
    print relu2_arr
    dLDz2 = np.multiply(dLDRelu2, relu2_arr)
    print dLDz2
    dLDw2 = np.matmul(rel1.transpose(), dLDz2)
    dLRelu1 = np.matmul(dLDz2, W2.transpose())
    relu1_arr = np.array(z1 > 0, dtype=np.int32)
    dLDz1 = np.multiply(dLRelu1, relu1_arr)
    dLDw1 = np.matmul(x.transpose(), dLDz1 )      
    #print dLDw1
    W2 = W2 - lr * dLDw2
    b2 = b2 - lr * sum(dLDw2)
    W1 = W1 - lr * dLDw1
    b1 = b1- lr * sum(dLDw1)
    z1 = np.matmul(x_valid, W1) + b1
    #print z1
    rel1 = np.maximum(z1, 0)
    z2 = np.matmul(rel1, W2) + b2
    rel2 = np.maximum(z2, 0)
    softmax = stablesoftmax(rel2)
    logit_err = -np.sum(np.sum(y_valid * np.log(softmax)))
    print logit_err
print dLDz
