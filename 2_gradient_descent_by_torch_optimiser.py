from sympy import symbols, ln, diff
from torch import tensor, log, optim


# градиентный спуск при помощи sympy
w1, w2, w3, w4 = symbols('w1, w2, w3, w4', real=True)
items = [w1, w2, w3, w4]
f = ln(ln(w1 + 7)) * ln(ln(w2 + 7)) * ln(ln(w3 + 7)) * ln(ln(w4 + 7))

lr = 0.01
point = [5, 10, 1, 2]
points = ([5, 10, 1, 2],)
for _ in range(2):
	parameters = {item: point[num] for num, item in enumerate(items)}
	for num, item in enumerate(items):
		gradient = diff(f, item).subs(parameters)
		point[num] = round(point[num] - lr * gradient, 4)
	print(point)

print('\n')


# градиентный спуск при помощи torch (#1)
# матрица весов
w = tensor([[5, 10], [1, 2]], requires_grad=True, dtype=float)
print(w, '\n')
lr = 0.01
for num in range(1, 3):
	# функция (якобы потерь)
	function = log(log(w + 7)).prod()
	function.backward()
	w.data -= lr * w.grad
	w.grad.zero_()
	print('GS  №', num, ', w : ', w, '\n')

print('\n')


# градиентный спуск при помощи torch optimiser (#2)
x = tensor([[5, 10], [1, 2]], requires_grad=True, dtype=float)
#
optimiser = optim.SGD([x], lr=0.01)
#
def function_log(variable):
	return log(log(variable + 7)).prod()


def make_gradient_step(function, variable):
	function_result = function(variable)
	function_result.backward()
	optimiser.step()
	optimiser.zero_grad()


for i in range(2):
	make_gradient_step(function_log, x)

print(x, '\n')
