from torch import optim, rand, randn, sin, linspace, nn
import matplotlib.pyplot as plt


# train dataset (данные для тренировки НС)
x_train = rand(100)							# тензор из 100 чисел от 0 до 1
x_train = x_train * 20 - 10					# увеличим числа, чтобы они были примерно от -10 до 10
# целевые данные
y_train = sin(x_train)
# добавим шум в целевые данные, чтобы график был не 100% sin(x)
# взяли размер тензора такой же как y_train и заполнили его случайными числами деленными на 5, затем добавили в y_train
noise = randn(y_train.shape) / 5
y_train = y_train + noise

# переведём данные в "много строк, одна колонка"
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

# график для train
fig = plt.figure(figsize=(12.5, 7))
# plt.axis([-11, 11, -1.2, 1.2])
# plt.plot(x_train, y_train, 'o')


# validation dataset (данные для валидации НС)
x_validation = linspace(-10, 10, 100)
y_validation = sin(x_validation.data)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

# график для валидации
# plt.plot(x_validation, y_validation, 'o')
# plt.legend(['train', 'valid'])
# plt.show()



# model construction (архитектура нашей НС)
# наш class сразу же наследует модуль torch.nn
class SineNet(nn.Module):
	def __init__(self, n_hidden_neurons):							# сеть получит параметр: кол-во скрытых нейронов
		super(SineNet, self).__init__()								# инициализируем родительский объект
		# Первый полносвязный слой. На входе один нейрон (одно значение Х) и кол-во связанных нейронов
		self.fc1 = nn.Linear(1, n_hidden_neurons)
		# ф-я активации первого слоя
		self.act1 = nn.Sigmoid()
		# Второй полносвязный слой. На выходе один нейрон, а по сути это будет ответ
		self.fc2 = nn.Linear(n_hidden_neurons, 1)

	def forward(self, x):
		# Сначала подаем значение на первый слой, затем полученное значение даем на ф-ю активации
		# и полученный ответ поступает во второй слой. Возвращаем итоговое значение
		x = self.fc1(x)
		x = self.act1(x)
		x = self.fc2(x)
		return x

# Создадим один объект класса SineNet и передадим ему кол-во скрытых нейронов 50 шт
sine_net = SineNet(50)


#
def loss(pred, target):
	# MSE
	squares = (pred - target) ** 2
	return squares.mean()


optimiser = optim.Adam(sine_net.parameters(), lr=0.01)

# Training Procedure
# Тренируем НС на 2000 эпохах
for epochs in range(2000):
	optimiser.zero_grad()
	y_pred = sine_net.forward(x_train)
	loss_func = loss(y_pred, y_train)
	loss_func.backward()
	optimiser.step()


# НС натренирована, теперь давайте посмотрим, что она предскажет на валидационной выборке
def predict(net, x, y):
	y_predict = net.forward(x)
	plt.plot(x, y)
	plt.plot(x, y_predict.data, 'o', c='r')
	plt.legend(['True', 'Predict'])
	plt.show()
predict(sine_net, x_validation, y_validation)


# Проверка осуществляется вызовом кода:
def metric(pred, target):
	return (pred - target).abs().mean()
print(metric(sine_net.forward(x_validation), y_validation).item())