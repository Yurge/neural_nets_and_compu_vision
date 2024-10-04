import torch


# --------------------------------------- операция свертки (свой алгоритм) ------------------------------------------

# На вход подается одно изображение (1 батч), 3х канальное, размером 5х5
input_tensor = torch.tensor(
	[
		[
		[[1, 1, 2, 1, 3],
    	 [1, 2, 4, 2, 1],
    	 [0, 1, 5, 1, 0],
		 [1, 2, 6, 0, 2],
		 [2, 0, 7, 1, 0]],

        [[1, 2, 8, 0, 2],
         [0, 3, 9, 1, 0],
         [0, 1, 10, 2, 1],
		 [1, 2, 11, 2, 1],
		 [0, 1, 12, 1, 0]],

		[[1, 1, 2, 1, 3],
    	 [1, 2, 4, 2, 1],
    	 [0, 1, 5, 1, 0],
		 [1, 2, 6, 0, 2],
		 [2, 0, 7, 1, 0]]
		]
	]
)


# один 3-х канальный фильтр
kernel = torch.tensor(
	[
	[
	[[0, 0, 0],
     [0, 2, 0],
     [0, 0, 0]],

	[[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]],

	[[0, 0, 0],
     [0, 2, 0],
     [0, 0, 0]]
	]
	]
).float()

# print(kernel.shape)

def padd():
	pass

# Если padding будет больше 0, то придется дописать функцию padd, которая обернет входное изображение нулями
padding = 0
stride = 1

batch_size  = input_tensor.shape[0]
input_channels = input_tensor.shape[1]
input_height = input_tensor.shape[2]
input_width = input_tensor.shape[3]
input_size = batch_size * input_channels * input_height * input_width


# Рассчитываем размеры и готовим балванку для тензора, который будет на выходе после свертки
output_channels = kernel.shape[0]
h_out = int((input_height - kernel.shape[1] + 2 * padding) / stride + 1)
w_out = int((input_height - kernel.shape[2] + 2 * padding) / stride + 1)
out_tensor = torch.zeros(batch_size, output_channels, h_out, w_out)


#
for num_img in range(batch_size):
	for channel in range(output_channels):
		for row in range(h_out):
			for col in range(w_out):
				r_start, r_end = row * stride, row * stride + kernel.shape[1]
				c_start, c_end = col * stride, col * stride + kernel.shape[2]
				input_tensor_slice = input_tensor[num_img, :, r_start:r_end, c_start:c_end]
				out_tensor[num_img, channel, row, col] = (input_tensor_slice * kernel).sum()


print(out_tensor)
