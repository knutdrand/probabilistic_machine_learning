import numpy as np
import stan

model_code ='''\
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  y[2:N] ~ normal(alpha + beta * y[1:(N - 1)], sigma);
}
'''

schools_data = {"N": 10,
                "y": np.arange(10)}

posterior = stan.build(model_code, data=schools_data)
fit = posterior.sample(num_chains=4, num_samples=1000)
eta = fit["alpha"]  # array with shape (8, 4000)
print(eta)
df = fit.to_frame()  # pandas `DataFrame, requires pandas
print(df)
