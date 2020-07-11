from qkm.utils import (circuit_learning, data_tools, backends,
                       circuit_generators)
import cirq
import matplotlib.pyplot as plt

dimension = 4
pixel_cutoff = 20
max_evts = 1000
dir_path = "/tmp"
data = data_tools.load_strong_lensing_pca(dir_path=dir_path,
                                          d=dimension,
                                          cutoff=pixel_cutoff,
                                          max_evts=max_evts)
backend = backends.HardwareBackend(hardware=cirq.Simulator())

new_generator = circuit_generators.PairwiseHardwareInspired

y_mod = data["train_y"]
y_mod[y_mod == 0] = -1

model = circuit_learning.DestructiveCircuitOptimizer(data["train_x"],
                                                     y_mod,
                                                     new_generator,
                                                     dimension,
                                                     seed=63335)

# Batch size tuned using `survey_loss` method...
opt_circuit, loss_history = model.train(backend, batch_size=50)
plt.plot(loss_history)
plt.show()

