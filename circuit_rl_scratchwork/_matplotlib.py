import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np

# For tests.
import time
import cirq


class CircuitRender:
    def __init__(self, circuit, qubits, fig=None, ax=None):
        """ `qubits` must preserve order."""
        self.qubits = qubits
        self.rows = len(self.qubits)
        self.cols = len(circuit)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        self.fig = fig
        self.ax = ax

        # static representation!
        self.circuit = circuit.copy()

        # This repo will house mpl objects that render the part of the circuit
        # in the position corresponding to the location of the object
        self._make_patch_repos()
        # construct a grid to render a circuit on top of
        self._make_grid()
        # plot the initial circuit
        self._plot_circuit()

        # Stage for modifications
        plt.pause(.01)

    def _make_grid(self):
        # Draw a grid
        self.ax.xaxis.set_ticks(np.arange(0, self.cols + 1))
        self.ax.yaxis.set_ticks(np.arange(0, self.rows + 1))
        self.ax.set_xlim((-0.5, self.cols - 0.5))
        self.ax.set_ylim((-0.5, self.rows - 0.5))
        for x in np.arange(-.5, self.cols - 0.5):
            self.ax.axvline(x, color="black", lw=0.5)
        for y in np.arange(-.5, self.rows - 0.5):
            self.ax.axhline(y, color="black", lw=0.5)
        self.grid = [g for g in self.ax.lines]


    def _make_patch_repos(self):
        self.text_repo = np.empty((self.cols, self.rows), dtype=object)
        for (x, y), _ in np.ndenumerate(self.text_repo):
            self.text_repo[x, y] = self.ax.text(x,
                                                y,
                                                "",
                                                ha='center',
                                                va='center',
                                                fontsize=8)
        self.line_repo = []
        self.cursor = patches.Rectangle(
            (0, 0),  #(bottom, left)
            1,  # width
            1,  # height
            linewidth=3,
            edgecolor='r',
            facecolor='none')
        self.ax.add_patch(self.cursor)

    def _render_op(self, col, op):

        opstr = str(op.gate)
        if '(' in opstr:
            opstr = opstr[:opstr.index('(')]
        if len(op.qubits) == 1:
            row = self.qubits.index(op.qubits[0])
            self.text_repo[col, row].set_text(opstr)

        if len(op.qubits) == 2:
            rows = [self.qubits.index(q) for q in op.qubits]
            # self.ax.lines.remove(self.text_repo[col, row][-1])
            for row in rows:
                self.text_repo[col, row].set_text(opstr)
            rows = sorted(rows)
            # store line in either ref loc
            line = mpl.lines.Line2D([col, col], [rows[0] + 0.2, rows[1] - 0.2],
                                    ls='--',
                                    c='k',
                                    alpha=0.2)
            self.ax.add_line(line)
            self.line_repo.append(line)

    def _plot_circuit(self):
        # Draw a circuit, simplified

        # clear previous lines
        for line in self.line_repo:
            line.remove()
        self.line_repo = []

        for row in range(self.rows):
            for col in range(self.cols):
                # Check if there is an operation to be rendered
                if self.circuit.operation_at(self.qubits[row], col):
                    for op in self.circuit[col].operations:
                        self._render_op(col, op)
                else:
                    self.text_repo[col, row].set_text("")

    def render(self, new_circuit, xy):
        """Only render changes from the previous circuit

        new cursor is in (x, y) format = (moment, qubit)
        """

        if new_circuit:
            self.circuit = new_circuit.copy()
            self._plot_circuit()

        self.cursor.set_xy([xy[0] - .5, xy[1] - 0.5])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def test_render():

    fig, ax = plt.subplots(figsize=(12, 3))

    circuit = cirq.testing.random_circuit(cirq.GridQubit.rect(4, 1), 10, .9)
    circuit_artist = CircuitRender(circuit, ax=ax, fig=fig)

    # plt.show()

    for i in range(10):

        print("...")
        q, m = np.random.randint(4), np.random.randint(10)
        to_remove = circuit.operation_at(cirq.GridQubit(q, 0), m)
        while not to_remove:
            q, m = np.random.randint(4), np.random.randint(10)
            to_remove = circuit.operation_at(cirq.GridQubit(q, 0), m)

        print(to_remove)
        circuit.batch_remove([(m, to_remove)])
        print(circuit, m, q)
        circuit_artist.render(circuit, (m, q))



if __name__ == "__main__":
    test_render()
