import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
from opacus import PrivacyEngine
import warnings
from torch.utils.data import TensorDataset
warnings.filterwarnings("ignore", category=UserWarning)
import random
import torch.nn.functional as F
import tensorflow as tf
from PIL import Image
import numpy

# load dataset from disk
# im = np.array(Image.open('./chinese_data/Locate{1,1,1}.jpg'))
# im = im.reshape(1,64,64)
# target_label = [1]
# for i in range(1,101):
#     for j in range(1,11):
#         for k in range(1,16):
#             if i == 1 and j == 1 and k == 1:
#                 pass
#             else:
#                 image_name = "Locate{"+ str(i) + "," + str(j) + "," + str(k) + "}" + ".jpg"
#                 file_path = "./chinese_data/" + image_name
#                 new_im = np.array(Image.open(file_path))
#                 new_im = new_im.reshape(1,64,64)
#                 im = np.concatenate((im,new_im),axis=0)
#                 target_label.append(k)
#
# target_label = np.array(target_label)
# target_label = target_label - 1
# numpy.save("im.npy", im)
# numpy.save("target_label.npy", target_label)

im = np.load("im.npy")
target_label = np.load("target_label.npy")

im = im.astype("float64")
target_label = target_label.astype("int32")

train_index = random.sample(range(0,15000), 12000) # training set, set training size as 80%
test_index = []
for i in range(0,15000):
    if i not in train_index:
        test_index.append(i)

train_x = im[train_index[0]]
train_x = train_x.reshape(1,1,64,64)
train_y = [target_label[train_index[0]]]
timer = 0
tmp_target_label = list(target_label)

train_x = im[train_index]
train_y = target_label[train_index]
test_x = im[test_index]
test_y = target_label[test_index]

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

train_set = TensorDataset(train_x,train_y)
test_set = TensorDataset(test_x,test_y)

train_set_A=Subset(train_set,range(0,4000))
train_set_B=Subset(train_set,range(4000,8000))
train_set_C=Subset(train_set,range(8000,12000))

train_loader_A = dataloader.DataLoader(dataset=train_set_A,batch_size=1000,shuffle=True)
train_loader_B = dataloader.DataLoader(dataset=train_set_B,batch_size=1000,shuffle=True)
train_loader_C = dataloader.DataLoader(dataset=train_set_C,batch_size=1000,shuffle=True)

test_set=Subset(test_set,range(0,1000))
test_loader = dataloader.DataLoader(dataset=test_set,batch_size=1000,shuffle=True)

def train_and_test_1(train_loader, test_loader):
    class NeuralNet(nn.Module):
        def __init__(self, input_num, hidden_num, output_num):
            super(NeuralNet, self).__init__()

            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 13 * 13, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 15)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 13 * 13)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    epoches = 20  # 迭代20轮
    lr = 0.01  # 学习率，即步长
    input_num = 1024
    hidden_num = 600
    output_num = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_num, hidden_num, output_num)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()  # 损失函数的类型：交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化，也可以用SGD随机梯度下降法
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epoches):
        flag = 0
        for images, labels in train_loader:
            images = images.to(torch.float32)
            labels = labels.long()
            # images = images.reshape(-1, 32 * 32).to(device)
            labels = labels.to(device)

            images = images / 255

            output = model(images)

            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()  # 误差反向传播，计算参数更新值
            optimizer.step()  # 将参数更新值施加到net的parameters上

            # 以下两步可以看每轮损失函数具体的变化情况
            # if (flag + 1) % 10 == 0:
            # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, loss.item()))
            flag += 1

    params = list(model.named_parameters())  # 获取模型参数

    # 测试，评估准确率
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(torch.float32)
        labels = labels.long()
        # images = images.reshape(-1, 32 * 32).to(device)
        labels = labels.to(device)

        images = images / 255

        output = model(images)
        values, predicte = torch.max(output, 1)  # 0是每列的最大值，1是每行的最大值
        total += labels.size(0)
        # predicte == labels 返回每张图片的布尔类型
        correct += (predicte == labels).sum().item()
    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    return params

def train_and_test_2(train_loader,test_loader, com_para_conv1, com_para_conv2,com_para_fc1, com_para_fc2, com_para_fc3):
    class NeuralNet(nn.Module):
        def __init__(self, input_num, hidden_num, output_num, com_para_conv1, com_para_conv2, com_para_fc1, com_para_fc2, com_para_fc3):
            super(NeuralNet, self).__init__()

            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 13 * 13, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 15)
            self.conv1.weight = Parameter(com_para_conv1)
            self.conv2.weight = Parameter(com_para_conv2)
            self.fc1.weight = Parameter(com_para_fc1)
            self.fc2.weight = Parameter(com_para_fc2)
            self.fc3.weight = Parameter(com_para_fc3)
            nn.init.constant_(self.fc1.bias, val=0)
            nn.init.constant_(self.fc2.bias, val=0)
            nn.init.constant_(self.fc3.bias, val=0)
            nn.init.constant_(self.conv1.bias, val=0)
            nn.init.constant_(self.conv2.bias, val=0)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 13 * 13)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    epoches = 20
    lr = 0.01
    input_num = 1024
    hidden_num = 600
    output_num = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_num, hidden_num, output_num,com_para_conv1, com_para_conv2,com_para_fc1, com_para_fc2, com_para_fc3)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr)

    # use opacus to implement differencial privacy
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=0.1,  # 0.01
        alphas=[10, 100],
        noise_multiplier=0.3,  # 1.3, 0.5
        max_grad_norm=1.0,
    )
    privacy_engine.attach(optimizer)

    for epoch in range(epoches):
        flag = 0
        for images, labels in train_loader:
            # (images, labels) = data
            # images = images.reshape(-1, 32 * 32).to(device)
            images = images.to(torch.float32)
            labels = labels.long()
            labels = labels.to(device)
            images = images / 255
            output = model(images)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (flag + 1) % 10 == 0:
                # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, loss.item()))
            flag += 1
    params = list(model.named_parameters())#get the index by debuging

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(torch.float32)
        # images = images.reshape(-1, 32 * 32).to(device)
        labels = labels.long()
        labels = labels.to(device)

        images = images / 255

        output = model(images)
        values, predicte = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicte == labels).sum().item()
    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    torch.save(model.state_dict(), 'model3.pt') # In this project, the parameters of third model will be saved
    return params

def combine_params(para_A, para_B, para_C):

    conv1_wA = para_A[0][1].data
    conv1_wB = para_B[0][1].data
    conv1_wC = para_C[0][1].data

    conv2_wA = para_A[2][1].data
    conv2_wB = para_B[2][1].data
    conv2_wC = para_C[2][1].data

    fc1_wA = para_A[4][1].data
    fc1_wB = para_B[4][1].data
    fc1_wC = para_C[4][1].data

    fc2_wA = para_A[6][1].data
    fc2_wB = para_B[6][1].data
    fc2_wC = para_C[6][1].data

    fc3_wA = para_A[8][1].data
    fc3_wB = para_B[8][1].data
    fc3_wC = para_C[8][1].data

    com_para_conv1 = (conv1_wA + conv1_wB + conv1_wC) / 3
    com_para_conv2 = (conv2_wA + conv2_wB + conv2_wC) / 3
    com_para_fc1 = (fc1_wA + fc1_wB + fc1_wC) / 3
    com_para_fc2 = (fc2_wA + fc2_wB + fc2_wC) / 3
    com_para_fc3 = (fc3_wA + fc3_wB + fc3_wC) / 3

    return com_para_conv1, com_para_conv2, com_para_fc1, com_para_fc2, com_para_fc3

if __name__ == "__main__":

    para_A=train_and_test_1(train_loader_A,test_loader)
    para_B=train_and_test_1(train_loader_B,test_loader)
    para_C=train_and_test_1(train_loader_C,test_loader)
    for i in range(3): # 这个后面再改。。。
        print("The {} round to be federated!!!".format(i+1))
        com_para_conv1,com_para_conv2,com_para_fc1, com_para_fc2,com_para_fc3 = combine_params(para_A, para_B, para_C)
        para_A=train_and_test_2(train_loader_A,test_loader,com_para_conv1,com_para_conv2,com_para_fc1, com_para_fc2, com_para_fc3)
        para_B=train_and_test_2(train_loader_B,test_loader,com_para_conv1,com_para_conv2,com_para_fc1, com_para_fc2, com_para_fc3)
        para_C=train_and_test_2(train_loader_C,test_loader,com_para_conv1,com_para_conv2,com_para_fc1, com_para_fc2, com_para_fc3)

class NeuralNet(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)
        nn.init.constant_(self.fc1.bias, val=0)
        nn.init.constant_(self.fc2.bias, val=0)
        nn.init.constant_(self.fc3.bias, val=0)
        nn.init.constant_(self.conv1.bias, val=0)
        nn.init.constant_(self.conv2.bias, val=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_num = 1024
hidden_num = 600
output_num = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_num, hidden_num, output_num)
model.load_state_dict(torch.load('model3.pt')) # we load the parameters
model.eval()
model.to(device)

timer = 0
logits_train = 0
prob_train = 0
logits_test = 0
prob_test = 0
y_train = []
y_test = []
num_classes = 15
for images, labels in train_loader_C:
    images = images.to(torch.float32)
    labels = labels.long()
    y_train.append(labels)
    timer += 1
    # images = images.reshape(-1, 32 * 32).to(device)
    labels = labels.to(device)

    images = images / 255

    output = model(images)
    if timer == 1:
        logits_train = output
    else:
        logits_train = torch.cat((logits_train, output), 0)

timer = 0
for images, labels in test_loader:
    images = images.to(torch.float32)
    labels = labels.long()
    y_test.append(labels)
    timer += 1
    # images = images.reshape(-1, 32 * 32).to(device)
    labels = labels.to(device)

    images = images / 255

    output = model(images)
    if timer == 1:
        logits_test = output
    else:
        logits_test = torch.cat((logits_test, output), 0)

prob_train = F.softmax(logits_train,dim=1)
prob_train = prob_train.detach().numpy()
prob_test = F.softmax(logits_test,dim=1)
prob_test = prob_test.detach().numpy()

logits_train = logits_train.detach().numpy()
logits_test = logits_test.detach().numpy()

first_batch_train_y = y_train[0].tolist()
timer = 0
for i in y_train:
    if timer == 0:
        pass
    else:
        first_batch_train_y += i.tolist()
    timer += 1
y_train = first_batch_train_y

first_batch_test_y = y_test[0].tolist()
timer = 0
for i in y_test:
    if timer == 0:
        pass
    else:
        first_batch_test_y += i.tolist()
    timer += 1
y_test = first_batch_test_y

y_train = torch.Tensor(np.array(y_train))
y_test = torch.Tensor(np.array(y_test))

y_train = y_train.reshape(y_train.shape[0],1).numpy()
y_test = y_test.reshape(y_test.shape[0],1).numpy()

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

cce = tf.keras.backend.categorical_crossentropy
constant = tf.keras.backend.constant

constant_y_train = constant(y_train)
constant_prob_train = constant(prob_train)
loss_train = cce(constant_y_train, constant_prob_train, from_logits=False).numpy()
loss_test = cce(constant(y_test), constant(prob_test), from_logits=False).numpy()

"""## Run membership inference attacks.

We will now execute a membership inference attack against the previously trained CIFAR10 model. This will generate a number of scores, most notably, attacker advantage and AUC for the membership inference classifier.

An AUC of close to 0.5 means that the attack wasn't able to identify training samples, which means that the model doesn't have privacy issues according to this test. Higher values, on the contrary, indicate potential privacy issues.
"""
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting


labels_train = np.argmax(y_train, axis=1)
labels_test = np.argmax(y_test, axis=1)

input = AttackInputData(
  logits_train = logits_train,
  logits_test = logits_test,
  loss_train = loss_train,
  loss_test = loss_test,
  labels_train = labels_train,
  labels_test = labels_test
)
# Run several attacks for different data slices
attacks_result = mia.run_attacks(input,
                                 SlicingSpec(
                                     entire_dataset = True,
                                     by_class = True,
                                     by_classification_correctness = True
                                 ),
                                 attack_types = [AttackType.LOGISTIC_REGRESSION])  # AttackType.THRESHOLD_ATTACK

# Plot the ROC curve of the best classifier
fig = plotting.plot_roc_curve(
    attacks_result.get_result_with_max_auc().roc_curve)

# Print a user-friendly summary of the attacks
print(attacks_result.summary(by_slices = True))