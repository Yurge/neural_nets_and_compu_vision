
from torch import optim, nn, manual_seed, cuda, backends, FloatTensor, LongTensor
import random
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from torch.ao.nn.quantized import Softmax
from torch.nn import Linear, Sigmoid

# ------------------------------------ НС для задач классификации ------------------------------------

# Зафиксируем seed, чтобы эксперимент был воспроизводим многократно. Т.е. перезапуская один и тот же скрипт будем
# получать один и тот же результат
random.seed(0)
np.random.seed(0)
manual_seed(0)
cuda.manual_seed(0)
backends.cudnn.deterministic = True


# Загрузим известный датасет wine из библиотеки sklearn, будем его использовать для классификации. Там 178 бутылок,
# 13 параметров и будет 3 класса на выходе
wine = sklearn.datasets.load_wine()
Y = wine.target
# print(wine.data.shape)


# Разделим данные на train и test, возьмем из данных только 2 колонки (2 параметра)
X_train, X_test, y_train, y_test = train_test_split(wine.data[:, :2], Y, test_size=0.3, shuffle=True)

# Все данные обернем в тенсоры. Если данные дробные, то будет "float tensor"
X_train = FloatTensor(X_train)
X_test = FloatTensor(X_test)
y_train = LongTensor(y_train)
y_test = LongTensor(y_test)


# Создаём класс с архитектурой НС
class WineNet(nn.Module):
	def __init__(self, n_hidden_neurons):
		super(WineNet, self).__init__()
		# архитектура НС
		self.fc1 = Linear(2, n_hidden_neurons)
		self.act1 = Sigmoid()
		self.fc2 = Linear(n_hidden_neurons, n_hidden_neurons)
		self.act2 = Sigmoid()
		self.fc3 = Linear(n_hidden_neurons, 3)
		self.sm = Softmax(dim=1)

	# реализация графа нашей НС
	def forward(self, x):
		x = self.fc1(x)
		x = self.act1(x)
		x = self.fc2(x)
		x = self.act2(x)
		x = self.fc3(x)
		return x

	#
	def inference(self, x):
		x = self.forward(x)
		x = self.sm(x)
		return x

wine_net = WineNet(5)

# функция потерь
loss = nn.CrossEntropyLoss()
# оптимизатор
optimizer = optim.Adam(wine_net.parameters(), lr=1.0e-3)


# ------------------- Градиентный спуск по мини батчам ----------------------


# создать список перемешанных чисел от 0 до 4 : np.random.permutation(5)

for epoch in range(10000):
	batch_size = 10
	# Создадим список со всеми индексами датасета. Индексы будут перемешаны
	order = np.random.permutation(len(X_train))
	for start_index in range(0, len(X_train), batch_size):
		optimizer.zero_grad()

		batch_indexes = order[start_index:start_index + batch_size]

		x_batch = X_train[batch_indexes]
		y_batch = y_train[batch_indexes]
		# В crossentropy есть логарифм softmax, поэтому мы не используем softmax в ф-и inference, а вызываем forward
		predict = wine_net.forward(x_batch)

		loss_val = loss(predict, y_batch)
		loss_val.backward()

		optimizer.step()

	# каждые 250 эпох будем выводить среднее значение совпадений
	if epoch % 250 == 0:
		test_predict = wine_net.forward(X_test)
		# Оставим только один выход (он же нейрон, он же класс) из трёх. Выход с максимальным значением
		test_predict = test_predict.argmax(dim=1)
		print((test_predict == y_test).float().mean())



# Мы можем экспериментировать и сравнивать как меняется accuracy или скорость обучения НС: 
# можно поменять количество скрытых слоёв, колич нейронов в скрытом слое, learning rate, метод ГС, размер батча и др.
