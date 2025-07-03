import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.height = 5
        self.width = 5

        self.current_location = (4, 4)

        self.grid = np.zeros((self.height, self.width)) - 1
        self.grid[0, 0] = 100
        
        self.grid[1, 1] = -30
        self.grid[1, 2] = -30
        self.grid[2, 1] = -30
        self.grid[2, 2] = -30

        self.actions = ['U', 'D', 'L', 'R']

    def get_reward(self, new_location):
        return self.grid[new_location[0], new_location[1]]
    
    def make_move(self, action):
        last_location = self.current_location

        if action == "U":
            if self.current_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0]-1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        elif action == "D":
            if self.current_location[0] == self.height-1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0]+1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        elif action == "L":
            if self.current_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0], self.current_location[1]-1)
                reward = self.get_reward(self.current_location)
        elif action == "R":
            if self.current_location[1] == self.width-1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0], self.current_location[1]+1)
                reward = self.get_reward(self.current_location)

        return reward

    def check_state(self):
        if self.current_location == (0, 0):
            return 'END'

class Agent:
    def __init__(self, environment, epsilon=0.1, alpha=0.1, gamma=1):
        self.environment = environment
        self.q_table = dict()
        for x in range(self.environment.height):
            for y in range(self.environment.width):
                self.q_table[(x,y)] = {'U':0, 'D':0,'L':0, 'R':0}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def chose_action(self, available_actions):
        if np.random.uniform(0,1) < self.epsilon:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[self.environment.current_location]
            max_value = max(q_values_of_state.values())
            action = np.random.choice([key for key, value in q_values_of_state.items() if value == max_value])
        return action
    
    def learn(self, old_state, new_state, reward, action):
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]

        self.q_table[old_state][action] = (1-self.alpha)*current_q_value+self.alpha*(reward+self.gamma*max_q_value_in_new_state)

def play(environment, agent, trials = 500, max_steps = 1000, learn = False):
    reward_per_trial = []
    for trial in range(trials):
        cumulative_reward = 1
        step = 0
        game_over = False
        while step < max_steps and game_over == False:
            old_state = environment.current_location
            action = agent.chose_action(environment.actions)
            reward = environment.make_move(action)
            new_state = environment.current_location
            
            if learn == True:
                agent.learn(old_state, new_state, reward, action)

            cumulative_reward+=reward
            step+=1

            if environment.check_state() == 'END':
                environment.__init__()
                game_over = True

        reward_per_trial.append(cumulative_reward)

    return reward_per_trial

env = Environment()
agent = Agent(env)

reward_per_trial = play(env, agent, trials=10000, max_steps=1000, learn=True)
plt.plot(reward_per_trial)
#plt.scatter([i for i in range(len(reward_per_trial))], reward_per_trial)
plt.grid(True)
plt.show()
