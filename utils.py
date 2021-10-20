from qmlt.numerical.losses import square_loss
from strawberryfields.ops import Dgate, BSgate
import strawberryfields as sf
from numpy import ndarray
from typing import List


# The function which representing the variational circuit is defined.
def circuit(X: ndarray, params: ndarray) -> List[float]:

    eng, q = sf.Engine(2)

    # Since X is a data stack, a circuit is defined for a single input.
    def single_input_circuit(x: ndarray) -> float:

        eng.reset()
        with eng:
            Dgate(x[0], 0.) | q[0]
            Dgate(x[1], 0.) | q[1]
            BSgate(phi=params[0]) | (q[0], q[1])
            BSgate() | (q[0], q[1])
        state = eng.run('fock', cutoff_dim=10, eval=True)

        # The output is defined as the probability of measuring |2,0> as opposed to |0,2>.
        p0 = state.fock_prob([0, 2])
        p1 = state.fock_prob([2, 0])
        normalization = p0 + p1 + 1e-10
        outp = p1 / normalization

        # print("x0:",x[0]," - x1:",x[1]," - Cikti:",outp)
        return outp

    # Implament of data stack to the circuit one by one.
    circuit_output = [single_input_circuit(x) for x in X]

    return circuit_output

# Defining the loss function: Comparing output of variational circuit with targets.


def myloss(circuit_output: ndarray, targets: ndarray) -> float:
    # Use of function of square_loss contained in the MLT (Machine Learning Tool).
    return square_loss(outputs=circuit_output, targets=targets)

# Round the output values of the circuit.
def outputs_to_predictions(circuit_output: float) -> int:
    return round(circuit_output)


def sonucugor(i: float) -> str:
    sonuc = ""
    if i == 1.0:
        sonuc = " May buy product."
    else:
        sonuc = " Dont may buy product."
    return sonuc
