from sympy import symbols, ln, diff


# градиентный спуск при помощи sympy
w1, w2, w3, w4 = symbols('w1, w2, w3, w4', real=True)
items = [w1, w2, w3, w4]
f = ln(ln(w1 + 7)) * ln(ln(w2 + 7)) * ln(ln(w3 + 7)) * ln(ln(w4 + 7))

lr = 0.01
point = [5, 10, 1, 2]
points = ([5, 10, 1, 2],)
for _ in range(2):
	# с каждой итерацией меняем значения весов
	parameters = {item: point[num] for num, item in enumerate(items)}
	for num, item in enumerate(items):
		gradient = diff(f, item).subs(parameters)
		point[num] = round(point[num] - lr * gradient, 4)

print(point,'\n')


# градиентный спуск при помощи torch (#1)
from torch import tensor, log


# матрица весов
w = tensor([[5, 10], [1, 2]], requires_grad=True, dtype=float)
# скорость обучения (learning rate)
lr = 0.01
for num in range(1, 3):
	function = log(log(w + 7)).prod()			# функция (якобы потерь)
	function.backward()							# обратное распространение вычисления
	w.data -= lr * w.grad						# значения весов уменьшаем на градиент
	w.grad.zero_()								# обнуляем градиент, чтобы не накапливать его значения
	# print('GS  №', num, ', w_out : ', w_out, '\n')

print(w, '\n')


# градиентный спуск при помощи torch optimiser (#2)
from torch import optim


x = tensor([[5, 10], [1, 2]], requires_grad=True, dtype=float)
# Создаем оптимизатор, который осуществляет ГС. SGD - стохастический ГС
optimiser = optim.SGD([x], lr=0.01)


def function_log(variable):
	return log(log(variable + 7)).prod()


def make_gradient_step(function, variable):
	function_result = function(variable)
	function_result.backward()
	optimiser.step()								# один шаг ГС
	optimiser.zero_grad()							# обнуляем градиент


for i in range(2):
	make_gradient_step(function_log, x)

print(x, '\n')
