
from torch import optim, nn, manual_seed, cuda, backends
import random
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.nn import Linear, Sigmoid, ReLU, CrossEntropyLoss


# ------------------------ полносвязная НС для задач классификации рукописных цифр ------------------------------------

# Зафиксируем seed, чтобы эксперимент был воспроизводим многократно. Т.е. перезапуская один и тот же скрипт будем
# получать один и тот же результат
random.seed(0)
np.random.seed(0)
manual_seed(0)
cuda.manual_seed(0)
backends.cudnn.deterministic = True


# Загрузим известный датасет MNIST с рукописными цифрами, будем его использовать для классификации.
# Там 60000 картинок размером 28х28 пикселей
MNIST_train = MNIST('./', download=True, train=True)
MNIST_test = MNIST('./', download=True, train=False)
# print(wine.data.shape)

# разделим данные на train и test
x_train = MNIST_train.data.float()
y_train = MNIST_train.targets
x_test = MNIST_test.data.float()
y_test = MNIST_test.targets
# print(x_train.shape)

# Посмотрим на первую картинку
# plt.imshow(x_train[0, :, :])
# plt.show()

# превратим 3D тензор 60000*28*28 в обычную матрицу 60000*784, то есть тензор 28х28 мы растянули в 1 строку
x_train = x_train.reshape([-1, 784])
x_test = x_test.reshape([-1, 784])
# print(x_train.size())


# архитектура НС
class MNISTNet(nn.Module):
	def __init__(self, n_hidden_neurons):
		super(MNISTNet, self).__init__()
		self.fc1 = Linear(784, n_hidden_neurons)
		self.act1 = Sigmoid()
		self.fc2 = Linear(n_hidden_neurons, 10)

	# пропускаем переменную последовательно через все слои
	def forward(self, x):
		x = self.fc1(x)
		x = self.act1(x)
		x = self.fc2(x)
		return x

mnist_net = MNISTNet(150)

# создаем функцию потерь и оптимизатор
loss = CrossEntropyLoss()
optimizer = optim.Adam(mnist_net.parameters(), lr=5e-4)


train_loss_history = []
test_loss_history = []
test_accuracy_history = []
batch_size = 150
for epoch in range(30):
	# создадим перемешанный список всех индексов x_train
	order = np.random.permutation(len(x_train))
	for start_index in range(0, len(x_train), batch_size):
		optimizer.zero_grad()
		# отбираем часть индексов
		batch_indexes = order[start_index:start_index + batch_size]
		# создаём batch - из train используем только отобранные индексы
		x_batch = x_train[batch_indexes]
		y_batch = y_train[batch_indexes]
		# прогоняем batch через НС, получаем значение функции потерь, делаем шаг ГС
		predicts = mnist_net(x_batch)
		loss_value = loss(predicts, y_batch)
		loss_value.backward()
		optimizer.step()
	# После каждой эпохи в НС сохранены определенные веса и смещения. Используя эти веса, будем считать значение
	# accuracy на тестовых данных и сохранять его в список, чтобы затем построить график
	train_preds = mnist_net(x_train)
	train_loss_history.append(loss(train_preds, y_train).item())

	test_preds = mnist_net(x_test)
	test_loss_history.append(loss(test_preds, y_test).item())

	accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
	test_accuracy_history.append(accuracy)
	print(accuracy)


# в конце всех эпох обучения НС посмотрим на график функции потерь (loss) и точности предсказаний (accuracy)
# f, ax = plt.subplots(1, 2, figsize=(12.5, 7))
# ax[0].plot(test_accuracy_history)
# ax[0].legend(['accuracy'])
# ax[1].plot(test_loss_history)
# ax[1].legend(['loss'])


# Посмотрим на loss по тестовой и train выборке.
# Если train_loss падает сильнее, test_loss от него отстает, значит у НС переобучение!
f, ax = plt.subplots(1, 1, figsize=(12.5, 7))
plt.plot(train_loss_history)
plt.plot(test_loss_history)
plt.grid(True)
ax.set_facecolor('black')
plt.legend(['train_loss', 'test_loss'])
plt.show()
