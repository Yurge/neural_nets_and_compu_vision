import matplotlib.pyplot as plt
from torch import optim, nn, manual_seed, cuda, backends, device
import random
import numpy as np
from torchvision.datasets import MNIST


# ------------------------ сверточная НС LeNet для задач классификации рукописных цифр --------------------------------

# Зафиксируем seed, чтобы эксперимент был воспроизводим многократно. То есть, перезапуская один и тот же скрипт, будем
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
# print(x_train.size(), '\n', len(y_test))

# Сейчас у нас 2D тензор (60000 картинок и каждая размером 28х28), а нам для свёртки нужно сделать, чтобы данные о
# картинке были в 3D формате (60000 картинок и каждая 1х28х28), то есть добавить информацию о количестве слоёв, после
# чего в итоге получим 60000х1х28х28
x_train = x_train.unsqueeze(1).float()
x_test = x_test.unsqueeze(1).float()
# print(x_train.size())


# свертка будет использоваться Conv2d, так как у нас двумерная картинка
class LeNet5(nn.Module):
	def __init__(self):
		super(LeNet5, self).__init__()
		# нужно, чтобы после первой свертки остался размер 28х28, поэтому добавим паддинг=2
		# применим 6 фильтров, чтобы на выходе было 6 каналов
		# в сети LeNet используется ф-я активации Tanh
		# сожмем изображение до 14х14 с помощью усредняющего пулинга AvgPool2d
		self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
		self.act1 = nn.Tanh()
		self.pool1 = nn.AvgPool2d(2, 2)
		# после второй свертки размер изображения станет 10х10, и после пулинга станет 5х5
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.act2 = nn.Tanh()
		self.pool2 = nn.AvgPool2d(2, 2)
		# в ф-ии forward изображение растянется в один вектор для входа в полносвязный слой (16*5*5=400)
		self.fc1 = nn.Linear(400, 100)
		self.act3 = nn.Tanh()
		#
		self.fc2 = nn.Linear(100, 30)
		self.act4 = nn.Tanh()
		#
		self.fc3 = nn.Linear(30, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.act1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.act2(x)
		x = self.pool2(x)
		# ф-я view растянет кубический тензор в однострочный вектор
		# было (batch_size, 16, 5, 5), стало (batch_size, 400)
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		x = self.fc1(x)
		x = self.act3(x)
		x = self.fc2(x)
		x = self.act4(x)
		x = self.fc3(x)
		return x

lenet5 = LeNet5()

# print(cuda.is_available())
# чтобы ускорить все процессы, переведем их на видеокарту
device = device('cuda:0' if cuda.is_available() else 'cpu')
lenet5 = lenet5.to(device)

# так как у нас классификация, значит ф-я потерь будет CrossEntropyLoss
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet5.parameters(), lr=0.0005)


train_loss_history = []
test_loss_history = []
test_accuracy_history = []
batch_size = 100
for epoch in range(20):
	# создадим перемешанный список всех индексов x_train
	order = np.random.permutation(len(x_train))
	for start_index in range(0, len(x_train), batch_size):
		optimizer.zero_grad()
		# отбираем часть индексов
		batch_indexes = order[start_index:start_index + batch_size]
		# создаём batch - из train используем только отобранные индексы и переводим процесс на видеокарту
		x_batch = x_train[batch_indexes].to(device)
		y_batch = y_train[batch_indexes].to(device)
		# прогоняем batch через НС, получаем значение функции потерь, делаем шаг ГС
		predicts = lenet5(x_batch)
		loss_value = loss(predicts, y_batch)
		loss_value.backward()
		optimizer.step()

	# После каждой эпохи в НС сохранены определенные веса и смещения. Используя эти веса, будем считать значение
	# accuracy на тестовых данных и сохранять его в список, чтобы затем построить график
	train_preds = lenet5(x_train)
	train_loss_history.append(loss(train_preds, y_train).data.cpu())

	test_preds = lenet5(x_test)
	test_loss_history.append(loss(test_preds, y_test).data.cpu())

	accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
	test_accuracy_history.append(accuracy)
	print('epoch:', str(epoch + 1).rjust(2),  f'  accuracy: {accuracy * 100:.2f}')

# print(train_loss_history, '\n', test_loss_history, '\n', test_accuracy_history)


# в конце всех эпох обучения НС посмотрим на график функции потерь (loss) и точности предсказаний (accuracy)
f, ax = plt.subplots(1, 2, figsize=(12.5, 7))
ax[0].plot(test_accuracy_history)
ax[0].legend(['accuracy'])
ax[1].plot(test_loss_history)
ax[1].legend(['loss'])
# plt.show()
