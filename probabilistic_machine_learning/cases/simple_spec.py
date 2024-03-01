class SimpleSpec(MosquitoModelSpec):
    def get_mosquito_maturation_rate(self, temp, human_I):
        P = self._params
        infection_rate = mosquito_infection_rate_func(P, human_I)
        maturation_level = get_maturation_rate_by_temp(P, temp)
        return [logit(maturation_level / 3), logit(maturation_level / 5),
                P['lo_mosqutito_beta'] * human_I]

    def get_human_params(self, mosquito_I):
        P = self._params
        beta_diff = P['alpha'] + P['beta'] * jnp.log(mosquito_I)
        return [beta_diff]

    def diff_distribution(self, state, exogenous):
        params = self._params
        mosquito_params = self.get_mosquito_maturation_rate(exogenous, state[..., 2])
        human_params = self.get_human_params(state[..., -1])
        param_array = jnp.array(jnp.broadcast_arrays(*(human_params + mosquito_params))).T
        scale = 0.1#jnp.exp(params['logscale'])
        return Logisitic(loc=param_array, scale=scale)

    def transition(self, states, logits):
        P = self._params
        human_state, mosquito_state = states[..., :4], states[..., 4:]
        human_logits = jnp.array(jnp.broadcast_arrays(*([logits[..., 0]] + P[p] for p in ['lo_gamma', 'lo_a', 'lo_mu']))).T
        human_diffs = human_state * expit(human_logits)
        death_rates = jnp.array(jnp.broadcast_arrays(*self.get_mosquito_death_rate(mosquito_state))).T
        mosquito_state = mosquito_state * (1 - expit(death_rates))
        mosquito_logits = logits[..., 1:]
        # mosquito_logits = jnp.array(jnp.broadcast_arrays(*(
        mosquito_diffs = mosquito_state * expit(mosquito_logits)
        new_eggs = jnp.exp(P['log_eggrate']) * jnp.sum(mosquito_state[..., 3:], axis=-1) * expit(
            mosquito_logits[..., -1])
        new_state = human_state - human_diffs + jnp.roll(human_diffs, 1, axis=-1)
        new_state = jnp.array([new_state[..., 0], new_state[..., 1], new_state[..., 2], new_state[..., 3],
                               mosquito_state[..., 0] - mosquito_diffs[..., 0] + new_eggs,
                               mosquito_state[..., 1] - mosquito_diffs[..., 1] + mosquito_diffs[..., 0],
                               mosquito_state[..., 2] - mosquito_diffs[..., 2] + mosquito_diffs[..., 1],
                               mosquito_state[..., 3] - mosquito_diffs[..., 3] + mosquito_diffs[..., 2],
                               mosquito_state[..., 4] - mosquito_diffs[..., 4] + mosquito_diffs[..., 3],
                               mosquito_state[..., 5] + mosquito_diffs[..., 4]]).T

        return new_state, new_state
