import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode=None, is_slippery=True) #initialization - no GUI

### Q2.2

## global parameters
state_actions_table = {} # ex: state_actions[0] = {((0, 1), 1)}
transitions_table = {} # ex: transitions_counts[0] = {((0, 1, 4), 1)}
rewards_table = {} # ex: rewards[_] = {((0, 1, 4), 0), ..., ((14, 2, 15), 1)}

## helper function
# add to transition and state dicts
def addStates(dict, key):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1

# add to rewards_sum dicts
def addRewards(dict, key, reward):
    if key in dict:
        dict[key] += reward
    else:
        dict[key] = reward

# edge checker for each action
def validTransition(state, action, nextState):

    top_edge = [0, 1, 2, 3]
    left_edge = [0, 4, 8, 12]
    bottom_edge = [12, 13, 14, 15]
    right_edge = [3, 7, 11, 15]

    if action == 3:
        if state in top_edge and nextState != state:
            return False
    if action == 0:
        if state in left_edge and nextState != state:
            return False
    if action == 1:
        if state in bottom_edge and nextState != state:
            return False
    if action == 2:
        if state in right_edge and nextState != state:
            return False    
    
    # if passed all --> not staying in the corner
    return True

## collecting transition and reward values for the states from 1000 seperate runs
for _ in range(1000):
    observation, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample() # agent policy that uses the observation and info
        next_obs, reward, terminated, truncated, info = env.step(action)

        state_action = (observation, action)
        state_action_nextSt = (*state_action, next_obs)

        # put each observation in the parameters
        if validTransition(observation, action, next_obs):
            addStates(transitions_table, state_action_nextSt) # only add the valid transitions for t-table
        
        addStates(state_actions_table, state_action)
        addRewards(rewards_table, state_action_nextSt, reward)
        
        done = terminated or truncated
        observation = next_obs

# done collecting t-table and r-table --> updating transition and reward table for each state's T and R functions
for i in transitions_table:
    state = i[0]
    action = i[1]
    next_st = i[2]
    rewards_table[i] = rewards_table[i] / transitions_table[i]
    transitions_table[i] = transitions_table[i] / state_actions_table[(state, action)]

env.close()

# print(transitions_table)

### Q2.3

## global parameters
value_table = np.zeros(16)
discount_factor = 0.95
threshold = 0.0001

## helper functions
# V_k+1(state) value finder
def calculateStateVal(state):

    # temporary list for holding summation values per action
    action_summation = [] 

    for action in range(4):
        summation = 0

        for next_st in range(16):
            transition = transitions_table.get((state, action, next_st), 0)
            reward = rewards_table.get((state, action, next_st), 0)
            summation += transition * (reward + discount_factor * value_table[next_st]) # summation value of the state for every possible next_state
    
        action_summation.append(summation)
    
    # find the maximum summation value from the list
    return max(action_summation)

## iteration - convergence
while 1:

    # temporary variable for converging
    delta = 0

    for state in range(16):
        v = value_table[state]

        # update state value
        value_table[state] = calculateStateVal(state)

        # update convergence factor
        delta = max(delta, abs(v - value_table[state]))

    if delta < threshold:
        break

### Q2.4

## global parameters
policy = np.zeros(16, dtype=int)

## helper functions
# optimal policy finder
def findPolicy(state):

    # temporary list for holding summation values per action
    action_summation = [] 

    for action in range(4):
        summation = 0

        for next_st in range(16):
            transition = transitions_table.get((state, action, next_st), 0)
            reward = rewards_table.get((state, action, next_st), 0)
            summation += transition * (reward + discount_factor * value_table[next_st]) # summation value of the state for every possible next_state
    
        action_summation.append(summation)
    
    # find the action that gives the maximum state value
    return np.argmax(action_summation) # ex: action 0's state value is stored at action_summation[0] ... so far so on

## extraction
for state in range(16):
    policy[state] = findPolicy(state)

print(policy)

### Q2.5

# # setting up the environment again for GUI
# env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode='human', is_slippery=True) #initialization - no GUI
# observation, info = env.reset()
# wins = 0

# for _ in range(50):
#     action = policy[observation] # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         if reward > 0:
#             wins += 1
#         observation, info = env.reset()
        
# env.close()
# print('wins:', wins)
