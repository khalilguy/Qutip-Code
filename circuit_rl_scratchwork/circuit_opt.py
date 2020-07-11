import numpy as np

import circuit_learning, circuit_generators

def main():


    dimension = 4
    pixel_cutoff = 20
    max_evts = 1000
    dir_path = "/tmp"

    # Construct a dataset to train on; Not applicable, so I'll just use
    # random numbers for now
    n_data = 1000
    X = np.random.random(size=(n_data, 4))
    y = np.random.randint(2, size=n_data)

    # This is the interface for constructing the _initial_ circuit
    # Don't worry about the details of this implementation at all; all that
    # matters is that this object provides the RL agent prior knowledge about
    # how to compose a circuit
    new_generator = circuit_generators.PairwiseHardwareInspired

    # batch size optimized in the `explore_loss.py` script.
    env = circuit_learning.CircuitGeneratorEnv(X, y, new_generator,
                                               dimension, batch_size=10, render=True)

    dqn_agent = circuit_learning.DQN(env=env, gamma=0.9, epsilon=0.95)

    # import pdb; pdb.set_trace()
    trials = 100
    trial_len = 500
    for trial in range(trials):
        cur_state = env.reset()
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            env.render(mode='mpl')
            new_state, reward, done, _ = env.step(action)
            reward = reward if not done else -20
            print("REWARD", reward)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()
            dqn_agent.target_train()
            cur_state = new_state
            if done:
                break
        if step < trial_len - 1:
            print("Failed to complete trial {}".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            break


if __name__ == "__main__":
    main()
