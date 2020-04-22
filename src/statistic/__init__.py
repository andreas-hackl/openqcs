import numpy as np 

def std(data, p=1):
    N = data.shape[0]
    
    m = np.mean(data)

    var = 1/(N-p) * sum([(d-m)**2 for d in data])
    sigma = np.sqrt(var)
    return sigma 


def bootstrap(data, n_boot=100):
    means = np.zeros(n_boot, dtype=np.double)
    for i in range(n_boot):
        tmp = np.random.choice(data, size=data.shape[0], replace=False)
        means[i] = np.mean(tmp)

    return np.mean(means), std(means)
