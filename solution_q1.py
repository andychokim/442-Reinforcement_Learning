import gymnasium as gym
import numpy as np

env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human") # Initializing environments observation,

## parameters
learning_rate = 0.1
n_episodes = 100 # for training
eval_episodes = 10 # for evaluating
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
discount_factor = 0.95

# storing q-values for each state and action pair
# ex: q_vals[0] ==> [((state), action), q-val]
q_vals = {}

# win counter
wins = 0
loses = 0

## helper functions
# getter
def getQval(state, action):
    return q_vals.get((state, action), 0.0)

# policy
def actionPolicy(state, start_epsilon):
    # random exploration
    if np.random.random() < start_epsilon:
        return env.action_space.sample()
    # choose max_Q(s,a) - exploitation
    else:
        return np.argmax([getQval(state, action) for action in [0, 1]])
    
# updating q-vals
def updateQvals(state, nextState, action, reward, terminated):
    if terminated:
        v_star = 0
    else:
        v_star = max(getQval(nextState, action) for action in [0, 1])
    old_q = getQval(state, action)
    temporal_diff = (
        reward + discount_factor * v_star - old_q
    )

    return (old_q + learning_rate * temporal_diff)

## training
for _ in range(n_episodes):

    # Reset the environment to generate the first observation
    observation, info = env.reset()
    observation = tuple(observation)
    done = False

    while not done:
        
        # this is where you would insert your policy
        action = actionPolicy(observation, start_epsilon)

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = tuple(next_obs)

        # updating the q_vals
        q_vals[(observation, action)] = updateQvals(observation, next_obs, action, reward, terminated)

        # If the episode has ended then we can reset to start a new episode
        done = terminated or truncated
        observation = next_obs

    # epsilon decaying
    start_epsilon = max(final_epsilon, start_epsilon - epsilon_decay)

print('Training completed!')

## evalution
for _ in range(eval_episodes):
    observation, info = env.reset()
    observation = tuple(observation)
    done = False

    while not done:

        # iteration
        action = np.argmax([getQval(observation, action) for action in [0, 1]])
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = tuple(next_obs)

        # transition to next observation (state), without updating q-values
        done = terminated or truncated
        observation = next_obs

    # out of while loop: check if won or lost
    if reward == 1:
        print('Won')
        wins = wins + 1
    elif reward == -1:
        print('Lost')
        loses = loses + 1

## print the evaluation results
print('win counts and percentage:', wins, (wins / eval_episodes)*100)
print('lose counts and percentage:', loses, (loses / eval_episodes)*100)

env.close()