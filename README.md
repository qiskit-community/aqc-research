![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-informational)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Latest-6133BD)](https://github.com/Qiskit/qiskit)
[![License](https://img.shields.io/github/license/qiskit-community/aqc-research?label=License)](LICENSE.txt)
[![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)

# Approximate Quantum Compiling for Quantum Simulation


### Table of Contents
* [About This Project](#about-this-project)
* [How to use](#how-to-use)
* [How to Give Feedback](#how-to-give-feedback)
* [Contribution Guidelines](#contribution-guidelines)
* [Acknowledgements](#acknowledgements)
* [References](#references)
* [License](#license)

---

### About This Project

In this project we demonstrate two closely related approaches:
1. Approximate Quantum Compiling (``AQC``) aims to find a quantum circuit that approximates the action of some unitary matrix (``target``). In the folder ``docs`` one can find the introductory notebooks. The notebook called ``problem_and_ansatz.ipynb`` explains what kind of problem is being solved and what approximating circuit (``ansatz``) is being used. The AQC computational model is based on papers [[1],[2]](#references) and implements the method described therein.
2. Approximate State Preparation (``ASP``) aims to reduce the depth of Trotter circuit for solving time-evolution problems. Likewise the first case, we optimize an ansatz circuit to reproduce the target quantum states from zero one ``|0>``. The notebook ``time_evolution.ipynb`` illustrates the method presented in the papers [[3],[4]](#references).

---

### How to use

The package can be installed from sources, please refer to [the contributing guideline](CONTRIBUTING.md).

Alternatively, on Linux system a *quick* installation can be done, where we suggest to use virtual environment:
```
mkdir -p ${HOME}/venv &&
python3 -m venv ${HOME}/venv/aqc &&
source ${HOME}/venv/aqc/bin/activate &&
python3 -m pip install --upgrade pip &&
pip3 install wheel &&
pip3 install qiskit[visualization] psutil joblib black ipython scikit-learn jupyter pylint &&
pip3 cache purge &&
cd /path/to/your/working/directory &&
git clone "https://github.com/qiskit-community/aqc-research.git" &&
echo "all done"
```
On Mac OS the quotations are needed ``'qiskit[visualization]'``, see in-depth installation guidelines on the [Qiskit official web-site](https://qiskit.org/documentation/getting_started.html).

There are three notebooks demonstrating: (1) the problem formulation for Approximate Quantum Compiling (``AQC``) ["problem_and_ansatz.ipynb"](./docs/problem_and_ansatz.ipynb), (2) how to find approximate circuit given a target unitary matrix ["aqc.ipynb"](./docs/aqc.ipynb), (3) how to use Approximate State Preparation (``ASP``) for time-evolution and shortening the circuit depth ["time_evolution.ipynb"](./docs/time_evolution.ipynb).

If running this notebook without installation, directly *from the root folder* of the package downloaded from the GitHub, the following ``bash`` snippet helps to resolve module paths, for example:
```bash
(export PYTHONPATH=`pwd` && echo "Python path: ${PYTHONPATH}" && jupyter notebook docs/problem_and_ansatz.ipynb)
```

For the large scale problems using notebooks might be cumbersome. In this case, user can run the approbate Python scripts straight on a (remote) powerful computer, see examples below. We encourage to tweak parameters directly in the script, which will be copied into the output directory along with simulation results. It should be always possible to reproduce the result by re-running the saved scripts and user settings.

#### Time Evolution example

Time evolution module illustrates the method described in [[4]](#references). It can be run either from the notebook ["time_evolution.ipynb"](./docs/time_evolution.ipynb) or directly in terminal using the script ``run_time_evol.py``. The latter script offers an opportunity for changing parameter settings. The class ``UserOptions`` in ``aqc_research/model_sp_lhs/user_options.py`` contains all tunable user options along with their description.

There are few generic command-line options when running in terminal:
```bash
cd /path/to/your/working/directory/
python3 run_time_evol.py -h

options:
  -h, --help            show this help message and exit
  -n , --num_qubits     number of qubits (default: 5)
  -t, --target_only     flag: compute target states and exit (default: False)
  -g , --tag            tag that makes simulation results distinguishable (default: )
  -f , --targets_file   path to a file with precomputed targets (default: )
```
The majority of the parameters are defined in the class ``UserOptions`` and should be tweaked in the launching script ``run_time_evol.py``, according to the example therein, before the start. 

Once desired parameters have been set, it is straightforward to run the simulation, for example: 
```bash
python3 run_time_evol.py --num_qubits 12 --tag test
```
At first, the target states will be computed or loaded, if already exist. The default path for cached target states is: ``./results/trotter_evol``. Next, the ansatz circuit will be optimized for several time horizons as described in [[4]](#references).
In case of default settings, the output results will be stored in the folder: ``./results/trotter_evol/12qubits/<Date_Time>_test``.
The notebook ["time_evolution.ipynb"](./docs/time_evolution.ipynb) helps to explore the content of that folder. 

Lastly, one can choose the rename the script ``run_time_evol.py`` to something else to keep the original one intact. Does not matter what the name is, the script will be copied to the output result directory for completeness and reproducibility of numerical experiments.

---

### How to Give Feedback
We encourage your feedback! You can share your thoughts with us by:
- [Opening an issue](https://github.com/qiskit-community/aqc-research/issues) in the repository
- [Starting a conversation on GitHub Discussions](https://github.com/qiskit-community/aqc-research/discussions)

---

### Contribution Guidelines
For information on how to contribute to this project, please take a look at [CONTRIBUTING.MD](CONTRIBUTING.md).

---

### Acknowledgements
This module is based on the theory and experiment described in [[4]](#references) and accompanying papers.

The code and tutorials, this module rests on, were written by Albert Akhriev, Niall Robertson and Anton Dekusar.
We would like to express the deepest gratitude and appreciation to Merav Aharoni from IBM Research Haifa for the support and guidance regarding Matrix Product State (MPS) package implemented in Qiskit.

---

### References
[1] L. Madden, and A. Simonetto, *Best approximate quantum compiling problems,* ACM Transactions on Quantum Computing, vol.3, no.2, pp.1-29, 2022, [arXiv:2106.05649](https://arxiv.org/pdf/2106.05649.pdf)

[2] L. Madden, A. Akhriev and A. Simonetto, *Sketching the Best Approximate Quantum Compiling Problem,* 2022 IEEE International Conference on Quantum Computing and Engineering (QCE), [arXiv:2205.04025](https://arxiv.org/abs/2205.04025)

[3] N.F. Robertson, A. Akhriev, J. Vala, S. Zhuk, *Escaping barren plateaus in approximate quantum compiling,* [arXiv:2210.09191](https://arxiv.org/abs/2210.09191)

[4] N.F. Robertson, A. Akhriev, J. Vala, S. Zhuk, *Approximate Quantum Compiling for Quantum Simulation: A Tensor Network based approach,* [arXiv:2301.08609](https://arxiv.org/abs/2301.08609)

### License
[Apache License 2.0](LICENSE.txt)
