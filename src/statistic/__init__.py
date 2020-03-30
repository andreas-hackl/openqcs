import numpy as np 

def bootstrap(data, n_boot=10):
    means = np.zeros(n_boot, dtype=np.double)
    for i in range(n_boot):
        tmp = np.random.choice(data, size=data.shape[0], replace=True)
        means[i] = np.mean(tmp)

    return np.mean(means), np.std(means, ddof=1)
