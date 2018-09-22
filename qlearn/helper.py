from copy import deepcopy
from collections import defaultdict
import random
import numpy as np

class TabularQLearner:
    def __init__(self,
                 gamma=0.99,
                 max_iter=1000,
                 c=16,
                 alpha=0.5,
                 verbose=10):
        self.gamma = gamma
        self.c = c
        self.alpha = alpha
        self.epsilon = 1.
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, env):
        terminal_states = deepcopy(env.pos_reward_states)
        terminal_states.extend(env.neg_reward_states)
        terminal_reward_map = deepcopy(env.pos_reward_vals)
        terminal_reward_map.update(env.neg_reward_vals)
        self._fit_helper(
            env.states,
            env.init_state,
            terminal_states,
            terminal_reward_map,
            env.state_action_state_probs,
            env.actions_available,
            env.get_reward
        )

    def get_next_state(self,
                       cur_state,
                       action_to_take,
                       state_action_state_probs):
        states = [
            state for state, _ in state_action_state_probs[cur_state][action_to_take]
        ]
        probs =[
            prob for _, prob in state_action_state_probs[cur_state][action_to_take]
        ]
        new_state_i = np.argmax(probs)
        return states[new_state_i]

    def fetch_max_q(self, state):
        return max(zip(self.Q[state].values(), self.Q[state].keys()))[0]

    def fetch_a_greedily(self, state):
        return max(zip(self.Q[state].values(), self.Q[state].keys()))[1]

    def sample_action(self, state, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        return self.fetch_a_greedily(state)

    def update_epsilon(self, iter):
        self.epsilon = self.c / (self.c + iter)

    def init_Q(self, states, action_fetcher, terminal_reward_map):
        self.Q = defaultdict(lambda: {})
        for state in states:
            if state in terminal_reward_map:
                self.Q[state][action_fetcher(state)[0]] = terminal_reward_map[state]
                continue
            for action in action_fetcher(state):
                self.Q[state][action] = 0.

    def _fit_helper(self,
                    states,
                    init_state,
                    terminal_states,
                    terminal_reward_map,
                    state_action_state_probs,
                    action_fetcher,
                    reward_fetcher):
        init_state_sampler = lambda: random.choice(list(set(states) - set(terminal_states)))
        self.init_Q(states, action_fetcher, terminal_reward_map)
        if init_state is None:
            init_state = init_state_sampler()

        cur_state = init_state

        for k in range(self.max_iter):
            possible_actions = action_fetcher(cur_state)
            action_to_take = self.sample_action(cur_state, possible_actions)
            new_state = self.get_next_state(cur_state,
                action_to_take, state_action_state_probs)
            if new_state in terminal_states:
                target = terminal_reward_map[new_state]
                new_state = init_state_sampler()
            else:
                cur_r = reward_fetcher(cur_state, action_to_take, new_state)
                discounted_q = self.gamma * self.fetch_max_q(new_state)
                target = cur_r + discounted_q
            cur_q_weight = (1-self.alpha) * self.Q[cur_state][action_to_take]
            target_q_weight = (self.alpha) * target
            self.Q[cur_state][action_to_take] = cur_q_weight + target_q_weight
            cur_state = new_state
            if k % self.verbose == 0:
                self.update_epsilon(k)
