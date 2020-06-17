from qiskit import *

# Choose a real device run on
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy141')
backend = provider.get_backend('ibmq_armonk')
props, config, defaults = (backend.properties(), backend.configuration(), backend.defaults())
