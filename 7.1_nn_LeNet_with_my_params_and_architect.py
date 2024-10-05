import matplotlib.pyplot as plt
from torch import optim, nn, manual_seed, cuda, backends, device, abs
import random
import numpy as np
from torchvision.datasets import MNIST


# ------------------------ свёрточная НС LeNet для задач классификации рукописных цифр --------------------------------

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
# print(MNIST_train.data.shape)

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
		activate = nn.ReLU()
		pooling = nn.MaxPool2d(2, 2)
		# картинка 28х28
		self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
		self.act1 = activate
		# после первой свертки получилось 28x28х8
		self.conv2 = nn.Conv2d(8, 16, 3)
		self.act2 = activate
		# после второй свертки получилось 26x26х16
		self.conv21 = nn.Conv2d(16, 24, 5)
		self.act21 = activate
		self.pool21 = pooling
		# после второй свертки получилось 11х11х24
		self.conv22 = nn.Conv2d(24, 32, 4)
		self.act22 = activate
		self.pool22 = pooling
		# после четвертой свертки получилось 4х4х32 = 512

		# в ф-ии forward метод view растянет изображение в один вектор (4х4х32=512) для входа в полносвязный слой
		self.fc1 = nn.Linear(in_features=512, out_features=10)
		self.act3 = activate
		#
		self.fc3 = nn.Linear(10, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.act1(x)
		x = self.conv2(x)
		x = self.act2(x)
		x = self.conv21(x)
		x = self.act21(x)
		x = self.pool21(x)
		x = self.conv22(x)
		x = self.act22(x)
		x = self.pool22(x)
		# ф-я view растянет кубический тензор в однострочный вектор
		# было (batch_size, 16, 5, 5), стало (batch_size, 400)
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		x = self.fc1(x)
		x = self.act3(x)
		x = self.fc3(x)
		return x

lenet5 = LeNet5()

# print(cuda.is_available())
# чтобы ускорить все процессы, переведем их на видеокарту
device = device('cuda:0' if cuda.is_available() else 'cpu')
lenet5 = lenet5.to(device)

# так как у нас классификация, значит ф-я потерь будет CrossEntropyLoss
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet5.parameters(), lr=0.0004)


train_loss_history = []
test_loss_history = []
test_accuracy_history = []
batch_size = 100
for epoch in range(30):
	# создадим перемешанный список всех индексов x_train
	order = np.random.permutation(len(x_train))
	for start_index in range(0, len(x_train), batch_size):
		optimizer.zero_grad()
		#lenet5.train()														# указываем, если есть батч-нормализация
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

	# Чтобы batch-norm слой менял параметры только при тренировке, нужно явно указать net.train и net.eval
	#lenet5.eval()
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


# Посмотрим на loss по тестовой и train выборке.
# Если train_loss падает сильнее, test_loss от него отстает, значит у НС переобучение!
f, ax = plt.subplots(1, 1, figsize=(12.5, 7))
plt.plot(train_loss_history)
plt.plot(test_loss_history)
plt.grid(True)
ax.set_facecolor('black')
plt.legend(['train_loss', 'test_loss'])
plt.show()
