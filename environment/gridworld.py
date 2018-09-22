from collections import defaultdict

class Grid:

    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    EXIT = 0

    def __init__(self,
                 init_state=None,
                 x_range=(1, 4),
                 y_range=(1, 3),
                 noise=0.2,
                 pos_reward_states=[(4, 3)],
                 neg_reward_states=[(4, 2)],
                 pos_reward_vals=[1.],
                 neg_reward_vals=[-1.],
                 blocked_states=[(2, 2)]):
        self.init_state = init_state
        self.x_range = x_range
        self.y_range = y_range
        self.noise = noise
        self.pos_reward_states = pos_reward_states
        self.neg_reward_states = neg_reward_states
        self.pos_reward_vals = dict(zip(pos_reward_states, pos_reward_vals))
        self.neg_reward_vals = dict(zip(neg_reward_states, neg_reward_vals))
        self.blocked_states = blocked_states

        self.define_states()
        self.define_sas_probs()

    def define_states(self):
        states = []
        for x in range(self.x_range[0], self.x_range[1]+1):
            for y in range(self.y_range[0], self.y_range[1]+1):
                states.append((x, y))
        self.states = list(set(states) - set(self.blocked_states))

    def define_sas_probs(self):
        self.state_action_state_probs = defaultdict(lambda: {})
        for state in self.states:
            if state in self.pos_reward_states or state in self.neg_reward_states:
                self.state_action_state_probs[state][Grid.EXIT] = [(state, 1.)]
                continue
            for action in self.actions_available(state):
                self.state_action_state_probs[state][action] = [
                    (self.move_given_action(state, action)[1], (1-self.noise)),
                    (self.move_lr_given_action(state, action)[0], (self.noise / 2)),
                    (self.move_lr_given_action(state, action)[1], (self.noise / 2)),
                ]

    def move_north(self, cur_state):
        new_state = (cur_state[0], cur_state[1]+1)
        if new_state in self.states: return new_state
        return cur_state

    def moving_north_lr(self, state):
        return [
            self.move_west(state),
            self.move_east(state)
        ]

    def move_south(self, cur_state):
        new_state = (cur_state[0], cur_state[1]-1)
        if new_state in self.states: return new_state
        return cur_state

    def moving_south_lr(self, state):
        return [
            self.move_east(state),
            self.move_west(state)
        ]

    def move_west(self, cur_state):
        new_state = (cur_state[0]-1, cur_state[1])
        if new_state in self.states: return new_state
        return cur_state

    def moving_west_lr(self, state):
        return [
            self.move_south(state),
            self.move_north(state)
        ]

    def move_east(self, cur_state):
        new_state = (cur_state[0]+1, cur_state[1])
        if new_state in self.states: return new_state
        return cur_state

    def moving_east_lr(self, state):
        return [
            self.move_north(state),
            self.move_south(state)
        ]

    def move_given_action(self, state, action):
        if action == Grid.EXIT:
            if state in self.pos_reward_states:
                res = self.pos_reward_vals[state]
            else:
                res = self.neg_reward_vals[state]
            return True, res

        if action == Grid.NORTH:
            return False, self.move_north(state)
        if action == Grid.SOUTH:
            return False, self.move_south(state)
        if action == Grid.EAST:
            return False, self.move_east(state)
        return False, self.move_west(state)

    def move_lr_given_action(self, state, action):
        if action == Grid.NORTH:
            return self.moving_north_lr(state)
        if action == Grid.SOUTH:
            return self.moving_south_lr(state)
        if action == Grid.EAST:
            return self.moving_east_lr(state)
        return self.moving_west_lr(state)

    def get_reward(self, old_state, action, cur_state):
        return 0.


    def actions_available(self, state):
        assert state in self.states
        if state in self.pos_reward_states or state in self.neg_reward_states:
            return [Grid.EXIT]
        else:
            return [Grid.NORTH,
                    Grid.SOUTH,
                    Grid.EAST,
                    Grid.WEST]

    def possible_states(self, state):
        if state in self.pos_reward_states or state in self.neg_reward_states:
            return []
        else:
            return [
                self.move_north(state),
                self.move_south(state),
                self.move_east(state),
                self.move_west(state)
            ]

    def display_world(self, cur_state=None):
        assert cur_state not in self.blocked_states
        print(' '*5, end='')
        for x in range(self.x_range[0], self.x_range[1]+1):
            print('|{:^5s}|'.format(str(x)), end='')
        print('\n', '-'*(7*(self.x_range[1] - self.x_range[0] + 1)+5))

        for y in range(self.y_range[1], self.y_range[0]-1, -1):
            print('{:>5s}'.format(str(y)), end='')
            for x in range(self.x_range[0], self.x_range[1]+1):
                if (x, y) == cur_state:
                    print('|{:^5s}|'.format('r'), end='')
                elif (x, y) in self.pos_reward_states:
                    print('|{:^5s}|'.format(str(self.pos_reward_vals[(x, y)])), end='')
                elif (x, y) in self.neg_reward_states:
                    print('|{:^5s}|'.format(str(self.neg_reward_vals[(x, y)])), end='')
                elif (x, y) in self.states:
                    print('|{:^5s}|'.format('a'), end='')
                else:
                    print('|{:^5s}|'.format('na'), end='')
            print()

    def display_world_q_vals(self, q_star):
        print(' '*5, end='')
        for x in range(self.x_range[0], self.x_range[1]+1):
            print('|{:^5s}|'.format(str(x)), end='')
        print('\n', '-'*(7*(self.x_range[1] - self.x_range[0] + 1)+5))

        for y in range(self.y_range[1], self.y_range[0]-1, -1):
            print('{:>5s}'.format(str(y)), end='')
            for x in range(self.x_range[0], self.x_range[1]+1):
                if (x, y) in self.blocked_states:
                    print('|{:^5s}|'.format('na'), end='')
                else:
                    print('|{:^5s}|'.format(
                        '{:.2f}'.format(max(q_star[(x, y)].values()))), end='')
            print()

    def get_max_q_star_state_action_map(self, q_star):
        pi = {}
        for state in q_star:
            pi[state] = max(zip(q_star[state].values(), q_star[state].keys()))[1]
        return pi

    def display_world_pi_vals(self, q_star):
        pi_star = self.get_max_q_star_state_action_map(q_star)
        print(' '*5, end='')
        for x in range(self.x_range[0], self.x_range[1]+1):
            print('|{:^5s}|'.format(str(x)), end='')
        print('\n', '-'*(7*(self.x_range[1] - self.x_range[0] + 1)+5))

        for y in range(self.y_range[1], self.y_range[0]-1, -1):
            print('{:>5s}'.format(str(y)), end='')
            for x in range(self.x_range[0], self.x_range[1]+1):
                if (x, y) in self.blocked_states:
                    print('|{:^5s}|'.format('na'), end='')
                elif (x, y) in self.pos_reward_states:
                    print('|{:^5s}|'.format(str(self.pos_reward_vals[(x, y)])), end='')
                elif (x, y) in self.neg_reward_states:
                    print('|{:^5s}|'.format(str(self.neg_reward_vals[(x, y)])), end='')
                else:
                    if pi_star[(x, y)] == Grid.NORTH:
                        print('|{:^5s}|'.format('^'), end='')
                    elif pi_star[(x, y)] == Grid.SOUTH:
                        print('|{:^5s}|'.format('_'), end='')
                    elif pi_star[(x, y)] == Grid.EAST:
                        print('|{:^5s}|'.format('->'), end='')
                    else:
                        print('|{:^5s}|'.format('<-'), end='')
            print()