import numpy as np
import stan


def run_model(filename, data):
    posterior = stan.build(open(filename).read(), data=data)
    fit = posterior.sample(num_chains=4, num_samples=1000, run)
    return fit.to_frame()  # pandas `DataFrame, requires pandas




data = {"N": 8, "y": [28, 8, -3, 7, -1, 1, 18, 12],
                "temperature": [15, 10, 16, 11, 9, 11, 10, 18]}
N = 100
y = np.random.normal(0, 1, N)
temperature = np.random.normal(0, 1, N)

data = {'N': N, 'y': y, 'temperature': temperature}
result = run_model('climate_model.stan', data)
print(np.mean(result['beta']))
