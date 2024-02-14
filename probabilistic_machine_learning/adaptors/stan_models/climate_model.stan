data {
  int<lower=0> N;
  vector[N] y;
  vector[N] temperature;
}
parameters {
  real alpha;
  real beta;
  vector[N] hidden;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  hidden[1] ~ normal(0, 10);
  hidden[2:N] ~ normal(alpha+ beta* temperature[1:(N-1)]+ hidden[1:(N-1)], sigma);
  y ~ normal(hidden*0.5, sigma);
}
