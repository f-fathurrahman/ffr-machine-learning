import gym

env = gym.make("FrozenLake-v0", is_slippery=False)
env.reset()
env.render()

# Print out all possible actions in this game
actions = env.action_space
print("Possible actions in Frozen Lake is ", actions)

# All possible states
states = env.observation_space
print("Possible states = ", states)

# Play the game until win or lose happens
istep = 0
while True:
    istep += 1
    print("\nStep ", istep)
    action = actions.sample()
    print("action = ", action)
    new_state, reward, done, info = env.step(action)
    env.render()
    #
    print("new_state = ", new_state)
    print("reward = ", reward)
    print("done = ", done)
    print("info  ", info)
    #
    if done == True:
        break

