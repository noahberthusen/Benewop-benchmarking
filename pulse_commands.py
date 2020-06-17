from __init__ import *
from qiskit.pulse import pulse_lib
from qiskit.compiler import assemble
import numpy as np

# define X, Y, Z, cR pulses.

Xp = lambda i: qiskit.pulse.commands.SamplePulse([p for p in defaults.pulse_library
                 if p.name=='Xp_d'+str(i)][0].samples, 'Xp'+str(i))

Xm = lambda i: qiskit.pulse.commands.SamplePulse(Xp(i).samples*-1.0,'Xm'+str(i))

Ym = lambda i: qiskit.pulse.commands.SamplePulse([p for p in defaults.pulse_library
                 if p.name=='Ym_d'+str(i)][0].samples, 'Ym'+str(i))



X90p = lambda i: qiskit.pulse.commands.SamplePulse([p for p in defaults.pulse_library
                 if p.name=='X90p_d'+str(i)][0].samples, 'X90p'+str(i))

X90m = lambda i: qiskit.pulse.commands.SamplePulse([p for p in defaults.pulse_library
                 if p.name=='X90m_d'+str(i)][0].samples, 'X90m'+str(i))

Y90m = lambda i: qiskit.pulse.commands.SamplePulse([p for p in defaults.pulse_library
                 if p.name=='Y90m_d'+str(i)][0].samples, 'Y90m'+str(i))

I = lambda N: qiskit.pulse.commands.SamplePulse([0j]*N, 'I_'+str(N))

def measure_and_acquire(qubit):
    inst_sched_map = defaults.instruction_schedule_map
    measure = inst_sched_map.get('measure', qubits=qubit)
    return measure