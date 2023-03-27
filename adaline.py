import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

import random as rd

class Adaline(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        average cost in every epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.

    """
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
    
    def _initialize_weights(self, m):
        """ use a _initialize_weights method to initialize weights to zero 
            and after initialization set w_initialized to True """
        self.w_ = np.zeros(m + 1)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """This method perform one weight update for one training sample xi and calculate the error
        Since weights update will be used is both fit and partial fit methods, 
        it's better to seperate it out to be concise"""
        output = self.net_input(xi)
        error = target - output
        self.w_[0] += self.eta * error
        self.w_[1:] += self.eta * xi.dot(error)
        cost = 0.5 * error ** 2
        return cost
    
    def _shuffle(self, X, y):
        """Shuffle training data with np random permutation"""
        seq = np.random.permutation(len(y))
        return X[seq], y[seq]
    
    def fit(self, X, y):
        
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return self.net_input(X)
        
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

n_samples = 200
st_dev = 0.4

x1_data = np.random.normal(2, st_dev, n_samples)
y1_data = np.random.normal(2, st_dev, n_samples)
g1_data = 1 * np.ones(n_samples)

x2_data = np.random.normal(4, st_dev, n_samples)
y2_data = np.random.normal(4, st_dev, n_samples)
g2_data = -1 * np.ones(n_samples)

x_data = np.concatenate((x1_data, x2_data))
y_data = np.concatenate((y1_data, y2_data))
g_data = np.concatenate((g1_data, g2_data)).astype(int)

data = np.array([x_data, y_data]).T

adaline = Adaline()

adaline.fit(data, g_data)

grid_size = 0.01
grid = np.zeros((round(6 / grid_size), round(6 / grid_size)))
x_out = np.zeros(round(6 / grid_size))
y_out = np.arange(0, 6, grid_size)

for i in range(0, round(6 / grid_size), 1):
  for j in range(0, round(6 / grid_size), 1):
    grid[i, j] = adaline.net_input([i * grid_size, j * grid_size])

  # print(find_nearest(grid[i], 0))
  x_out[i] = find_nearest(grid[i], 0) * grid_size


# print(grid)
    
grid = np.rot90(grid, 3)

plt.plot(x1_data, y1_data, 'or', mfc='none')
plt.plot(x2_data, y2_data, 'ob', mfc='none')
plt.plot(x_out, y_out, '-', mfc='none')

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "red"], 2)

c = plt.imshow(grid.T, cmap=cmap, vmax = 3, vmin = -3, extent=[0, 6, 0, 6], interpolation='nearest', alpha=0.2)
plt.colorbar(c)

plt.show()