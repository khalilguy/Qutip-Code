from qkm.utils import (circuit_learning, data_tools, backends,
                       circuit_generators)
import cirq
import matplotlib.pyplot as plt

import gym


def main():


    dimension = 4
    pixel_cutoff = 20
    max_evts = 1000
    dir_path = "/tmp"

    # Load and modify data according to
    data = data_tools.load_strong_lensing_pca(dir_path=dir_path,
                                              d=dimension,
                                              cutoff=pixel_cutoff,
                                              max_evts=max_evts)
    X = data["train_x"]
    y_mod = data["train_y"]
    y_mod[y_mod == 0] = -1

    new_generator = circuit_generators.PairwiseHardwareInspired

    # batch size optimized in the `explore_loss.py` script.
    env = circuit_learning.CircuitGeneratorEnv(X, y_mod, new_generator,
                                               dimension, batch_size=100, render=True)

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
