from torch import tensor, optim
import numpy as np
import matplotlib.pyplot as plt


# функция для визуализации ГС
def show_contours(function, data):
	# рисуем график функции
	x_value = np.linspace(start=-10, stop=10, num=100)
	y_value = [function(elem) for elem in x_value]
	fig = plt.figure(figsize=(12, 7))
	plt.plot(x_value, y_value)
	# рисуем точки градиентного спуска
	x_grad = [(sum(item[0]) + sum(item[1])) / 4 for item in data]
	y_grad = [function(elem) for elem in x_grad]
	plt.scatter(x_grad, y_grad, color='red')
	plt.show()


# матрица весов
x = tensor([[5, 10], [1, 2]], requires_grad=True, dtype=float)
# Создаем оптимизатор, который осуществляет ГС. SGD - стохастический ГС
optimiser = optim.SGD([x], lr=0.1)

# функция сети
def func_nn(s):
	return 10 + (s ** 2)


# функция "потерь"
def function_parabola(variable):
	return func_nn(variable).sum()


# ГС
points = [x.tolist()]
for i in range(10):
	loss_function = function_parabola(x)
	loss_function.backward()
	optimiser.step()  						# один шаг ГС
	points += [x.tolist()]					# сохраняем новые координаты в список
	optimiser.zero_grad()  					# обнуляем градиент


# выводим график на экран
show_contours(func_nn, points)
