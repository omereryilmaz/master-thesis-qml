def circuit(X: list[float], params: list[float]) -> float:

    eng, q = sf.Engine(2)
    
    # Since X is a data stack, a circuit is defined for a single input.
    def single_input_circuit(x: list) -> float:

        eng.reset()
        with eng:
            Dgate(x[0], 0.) | q[0]
            Dgate(x[1], 0.) | q[1]
            BSgate(phi=params[0]) | (q[0], q[1])
            BSgate() | (q[0], q[1])
        state = eng.run('fock', cutoff_dim=10, eval=True)
    
        # The output is defined as the probability of measuring |2,0> as opposed to |0,2>.
        p0 = state.fock_prob([0,2])
        p1 = state.fock_prob([2,0])
        normalization = p0 + p1 + 1e-10
        outp = p1 / normalization
        
        # print("x0:",x[0]," - x1:",x[1]," - Cikti:",outp)
        return outp

    # Implament of data stack to the circuit one by one.
    circuit_output = [single_input_circuit(x) for x in X]

    return circuit_output

# Defining the loss function: Comparing output of variational circuit with targets.
def myloss(circuit_output: float, targets: list[float
]) -> float:
    # Use of function of square_loss contained in the MLT (Machine Learning Tool).
    return square_loss(outputs=circuit_output, targets=targets)

# Round the output values of the circuit.
def outputs_to_predictions(circuit_output: float) -> float:
    return round(circuit_output)
    

def sonucugor(i: float) -> str:
    sonuc=""
    if i==1.0:
        sonuc=" May buy product."
    else:
        sonuc=" Dont may buy product."
    return sonuc
     


URL = 'https://qml.azurewebsites.net/api/Alisveris'

YELLOW = '#f1c40f'
BLACK = '#2d3436'
BLUE = '#2980b9'
RED = '#c0392b'
GREEN = '#2ecc71'
TP = "TP : "
TN = "TN : "
FP = "FP : "
FN = "FN : "
CLASSICAL = "Classical : "
QUANTUM = "Quantum   : "
ENTER = '\n'
LABEL_TRAIN = 'Train'
LABEL_TEST = 'Test'
SHOPING_XLABEL = 'Age'
SHOPPING_YLABEL = "'Person\'s Annual Average Shopping Count'"
YAS = 'YAS'
