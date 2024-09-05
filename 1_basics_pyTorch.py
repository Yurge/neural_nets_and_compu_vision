import torch
import numpy as np

a = torch.zeros([3, 4])		# матрица размером 3х4 заполняется 0
# print(a)

# print(torch.ones(3))		# матрица размером 1х3 заполняется 1
# print(torch.rand([2, 3]))	# тензор 2х3 со случайными числами)

x = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(x, '\n')
# print(x.size())
# print(x[1, 2])			# выберем из второй строки третье число
# print(x[:, 1])			# выберем второй столбик (первое значение всех строк)
# print(x + 10)				# каждое значение матрицы увеличим на 10

# y = x + 5					# создание новой матрицы, в которой каждый элемент на 5 больше чем в матрице Х
# print(y, '\n')
# print(x + y, '\n')			# сложение поэлементно
print(x > 5)				# проверим поэлементно соответствие условию

mask = x > 5
print(x[mask])				# выборка из матрицы "х". список элементов по условию
# print(x[x > 5])				# то же самое, но без создания переменной


# функция, которая возвращает сумму (x.sum()) элементов тензора X, строго превышающих значение limit
X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = 4
larger_than_limit_sum = (X[X > limit]).sum()
print(larger_than_limit_sum)


# копирование тензоров осущ-я при помощи функции clone
z = x.clone()
z[0, 0] = 111
# print(z)

# изменение типа данных
x = x.double()
# print(x)
x = x.int()
# print(x)

# иногда данные приходят в виде numpy array. нужно перевести их в данные tensor
d = np.array([[1, 3, 5, 7], [2, 4, 6, 8]])
d = torch.from_numpy(d)
print(d)


#
f = torch.rand([2, 3])		# тензор 2х3 со случайными числами от 0 до 1
print(torch.cuda.is_available())

# можно перевести вычисления на мощности видеокарты (cuda) и выбрать первую видеокарту (cuda:0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)