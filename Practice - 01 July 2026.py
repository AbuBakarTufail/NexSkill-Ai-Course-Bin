import numpy as np
price = np.genfromtxt('Datasets/RealEstate-USA.csv', delimiter=',', skip_header=1, dtype=None, encoding=None, usecols=(2))
print('Min: ' + str(np.min(price)))
print('Max: ' + str(np.max(price)))
# print(np.min(price))
# print(price)