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

    # import pdb; pdb.set_trace()
    batch_sizes = range(10, 200, 10)
    dots = []
    trials = 10
    for batch_size in batch_sizes:
        for trial in range(trials):
            env = circuit_learning.CircuitGeneratorEnv(X,
                                                       y_mod,
                                                       new_generator,
                                                       dimension,
                                                       batch_size=batch_size,
                                                       render=False)
            dqn_agent = circuit_learning.DQN(env=env, gamma=0.9, epsilon=0.95)
            cur_state = env.reset()
            new_state, reward, done, _ = env.step(5)
            print("batch_size", batch_size)
            print("REWARD", reward)
            dots.append((batch_size, reward))

    x = [t[0] for t in dots]
    y = [t[1] for t in dots]
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    main()
