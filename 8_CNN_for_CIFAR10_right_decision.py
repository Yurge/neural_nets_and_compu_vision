import matplotlib.pyplot as plt
from torch import device, optim, nn, manual_seed, cuda, backends, FloatTensor, LongTensor
import random
import torch
import numpy as np
from torchvision.datasets import CIFAR10


# ------------------------ свёрточная НС LeNet для задач классификации рукописных цифр --------------------------------

# Зафиксируем seed, чтобы эксперимент был воспроизводим многократно. То есть, перезапуская один и тот же скрипт, будем
# получать один и тот же результат
random.seed(0)
np.random.seed(0)
manual_seed(0)
cuda.manual_seed(0)
backends.cudnn.deterministic = True


# Загрузим известный датасет CIFAR10 с картинками, будем его использовать для классификации.
# Там 50000 картинок размером 32х32 пикселей и эти картинки цветные, то есть они трёх слойные
CIFAR10_train = CIFAR10('./', download=True, train=True)
CIFAR10_test = CIFAR10('./', download=True, train=False)
# print(CIFAR10_train.data.shape)
# посмотрим на какие классы поделены все изображения:
# print(CIFAR10_train.classes)

# разделим данные на train и test
X_train = FloatTensor(CIFAR10_train.data)
y_train = LongTensor(CIFAR10_train.targets)
X_test = FloatTensor(CIFAR10_test.data)
y_test = LongTensor(CIFAR10_test.targets)
# print(x_train.size(), '\n', len(y_test))


# Если мы посмотрим на макс. и мин. значения в картинках, то окажется, что это "0" и "255".
# С такими изображениями можно работать - есть сети, которые действительно работают с такими данными.
# Но мы, для удобства, отнормируем эти данные -- разделим каждый пиксель, каждое его значение, на 255 и получим, что
# в наших картинках будут лежать значения от нуля до единицы
X_train /= 255
X_test /= 255
# print(x_train.size())

# сейчас количество каналов стоит на последнем месте (torch.Size([50000, 32, 32, 3])), а torch хочет, чтобы
# кол-во каналов стояло перед размером картинки
X_train = X_train.permute(0, 3, 1, 2)
X_test = X_test.permute(0, 3, 1, 2)
# print(x_train.size())


#
class CIFARNet(nn.Module):
	def __init__(self):
		super(CIFARNet, self).__init__()
		self.batch_norm0 = nn.BatchNorm2d(3)

		self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
		self.act1 = nn.ReLU()
		self.batch_norm1 = nn.BatchNorm2d(16)
		self.pool1 = nn.MaxPool2d(2, 2)

		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.act2 = nn.ReLU()
		self.batch_norm2 = nn.BatchNorm2d(32)
		self.pool2 = nn.MaxPool2d(2, 2)

		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.act3 = nn.ReLU()
		self.batch_norm3 = nn.BatchNorm2d(64)

		self.fc1 = nn.Linear(8 * 8 * 64, 256)
		self.act4 = nn.Tanh()
		self.batch_norm4 = nn.BatchNorm1d(256)

		self.fc2 = nn.Linear(256, 64)
		self.act5 = nn.Tanh()
		self.batch_norm5 = nn.BatchNorm1d(64)

		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = self.batch_norm0(x)
		x = self.conv1(x)
		x = self.act1(x)
		x = self.batch_norm1(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.act2(x)
		x = self.batch_norm2(x)
		x = self.pool2(x)

		x = self.conv3(x)
		x = self.act3(x)
		x = self.batch_norm3(x)

		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		x = self.fc1(x)
		x = self.act4(x)
		x = self.batch_norm4(x)
		x = self.fc2(x)
		x = self.act5(x)
		x = self.batch_norm5(x)
		x = self.fc3(x)

		return x


def train(net, X_train, y_train, X_test, y_test):
	device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
	net = net.to(device)
	loss = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=1.0e-3)

	batch_size = 100

	test_accuracy_history = []
	test_loss_history = []

	X_test = X_test.to(device)
	y_test = y_test.to(device)

	for epoch in range(30):
		order = np.random.permutation(len(X_train))
		for start_index in range(0, len(X_train), batch_size):
			optimizer.zero_grad()
			net.train()

			batch_indexes = order[start_index:start_index + batch_size]

			X_batch = X_train[batch_indexes].to(device)
			y_batch = y_train[batch_indexes].to(device)

			preds = net.forward(X_batch)

			loss_value = loss(preds, y_batch)
			loss_value.backward()

			optimizer.step()

			X_batch

		net.eval()
		test_preds = net.forward(X_test)
		test_loss_history.append(loss(test_preds, y_test).data.cpu())

		accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
		test_accuracy_history.append(accuracy)

		print('epoch:', str(epoch + 1).rjust(2),  f'  accuracy: {accuracy * 100:.2f}')
	del net
	return test_accuracy_history, test_loss_history


accuracies = {}
losses = {}
accuracies['cifar_net'], losses['cifar_net'] = train(CIFARNet(), X_train, y_train, X_test, y_test)

for experiment_id in losses.keys():
	plt.plot(losses[experiment_id], label=experiment_id)
plt.legend()
plt.title('Validation Loss')
plt.show()
