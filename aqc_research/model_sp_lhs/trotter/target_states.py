# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities for computation, caching and loading of the target states.
"""

import os
import pickle
from typing import Any, List, Optional, Union
import numpy as np
from qiskit import QuantumCircuit
import aqc_research.utils as helper
import aqc_research.mps_operations as mpsop
import aqc_research.model_sp_lhs.trotter.trotter as trotop
import aqc_research.checking as chk

_logger = helper.create_logger(__file__)


def precise_multiplier() -> int:
    """
    Returns the coefficient that should multiply the number of Trotter steps
    in order to obtain a more precise ground-truth Trotter circuit comparing
    to the less accurate reference one.
    """
    return int(10)


# -----------------------------------------------------------------------------
# MPS-only target state.
# -----------------------------------------------------------------------------


class TargetMpsState:
    """
    Structure keeps the target state |t1> in MPS format and related data.
    """

    def __init__(
        self,
        *,
        opts: Any,
        num_qubits: int,
        num_trot_steps: int,
        evol_time: float,
        my_id: int,
        t1_gt: mpsop.QiskitMPS,
        t1: mpsop.QiskitMPS,
        second_order: bool,
    ):
        """
        Args:
            opts: user supplied options defined as a class or namespace.
            num_qubits: number of qubits.
            num_trot_steps: number of Trotter steps.
            evol_time: evolution time.
            my_id: unique integer identifier of this structure.
            t1_gt: ground-truth state |t1> from precise Trotter evolution.
            t1: state |t1> obtained from loose Trotter evolution.
            second_order: True, if the 2nd order Trotter is intended.
        """
        assert chk.is_int(num_qubits, num_qubits >= 2)
        assert chk.is_int(num_trot_steps, num_trot_steps in opts.trotter_steps)
        assert chk.is_float(evol_time, evol_time in opts.evol_times)
        assert chk.is_int(my_id, my_id >= 0)
        assert mpsop.check_mps(t1_gt)
        assert mpsop.check_mps(t1)
        assert isinstance(second_order, bool)

        self.num_qubits = int(num_qubits)  # #qubits
        self.num_trot_steps = int(num_trot_steps)  # #steps in reference state |t1>
        self.precise_multiplier = precise_multiplier()  # #steps*mult for ground-truth
        self.trunc_thr = float(opts.trunc_thr_target)  # MPS truncation threshold
        self.delta = float(opts.delta)  # parameter in Hamiltonian
        self.evol_time = float(evol_time)  # evolution time (aka time horizon)
        self.my_id = int(my_id)  # unique id, here the no. of time horizon
        self.t1_gt = t1_gt  # ground-truth target state
        self.t1 = t1  # less accurate reference state
        self.second_order = second_order  # True for the 2nd order Trotter

    @staticmethod
    def check_cached_data(opts: Any, num_qubits: int, data: List[Any]) -> bool:
        """
        Checks that the structure of cached target states and related data
        match the expectation.

        Args:
            opts: user supplied options defined as a class.
            num_qubits: number of qubits.
            data: list of instances of TargetMpsState class to be verified.

        Returns:
            non-zero if elements of ``data`` match expected pattern.
        """
        assert chk.is_int(num_qubits, num_qubits >= 2) and chk.is_list(data)
        for i in range(min(len(data), len(opts.evol_times), len(opts.trotter_steps))):
            dat, t, s = data[i], opts.evol_times[i], opts.trotter_steps[i]
            if not (
                isinstance(dat, TargetMpsState)
                and hasattr(dat, "num_qubits")
                and hasattr(dat, "num_trot_steps")
                and hasattr(dat, "precise_multiplier")
                and hasattr(dat, "trunc_thr")
                and hasattr(dat, "delta")
                and hasattr(dat, "evol_time")
                and hasattr(dat, "my_id")
                and hasattr(dat, "t1_gt")
                and hasattr(dat, "t1")
                and hasattr(dat, "second_order")
                and dat.num_qubits == num_qubits
                and dat.num_trot_steps == s
                and dat.precise_multiplier == precise_multiplier()
                and bool(np.isclose(dat.trunc_thr / opts.trunc_thr_target, 1))
                and bool(np.isclose(dat.delta / opts.delta, 1))
                and chk.is_float(dat.evol_time, bool(np.isclose(dat.evol_time / t, 1)))
                and chk.is_int(dat.my_id, dat.my_id == i)
                and mpsop.check_mps(dat.t1_gt)
                and mpsop.check_mps(dat.t1)
                and isinstance(dat.second_order, bool)
            ):
                return False
        return True


def generate_all_mps_targets(
    *, opts: Any, num_qubits: int, second_order: bool
) -> List[TargetMpsState]:
    """
    Computes all targets (in MPS format) by precise and normal Trotterization.
    We reuse MSP decompositions, obtained for the previous targets, to compute
    the new ones for the current targets. Computing MPS from scratch is very
    time-consuming, even intractable for the large number of qubits.

    Args:
        opts: user supplied options defined as a class.
        num_qubits: number of qubits.
        second_order: True, if the 2nd order Trotter is intended.

    Returns:
        list of TargetMpsState structures that represent targets for
            the different time horizons.
    """
    _logger.info("running the function: %s ...", generate_all_mps_targets.__name__)

    def _ini_state() -> QuantumCircuit:
        """Returns the generator of initial state defined in Options."""
        return opts.ini_state_func[0](num_qubits)

    trotter_steps = np.asarray(opts.trotter_steps)
    evol_times = np.asarray(opts.evol_times)

    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert evol_times.size == trotter_steps.size
    assert isinstance(second_order, bool)
    assert np.unique(np.diff(trotter_steps)).size == 1, "expects uniform stepping"
    assert np.allclose(np.diff(evol_times), evol_times[0]), "expects equal intervals"

    thr = opts.trunc_thr_target
    t1_gt = mpsop.mps_from_circuit(_ini_state(), trunc_thr=thr)
    t1 = mpsop.mps_from_circuit(_ini_state(), trunc_thr=thr)
    interval = evol_times[0]  # duration of an incremental Trotter qcircuit
    nsteps = trotter_steps[0]  # number of elementary Trotter steps inside the interval
    targets = list([])

    for i in range(max(evol_times.size, trotter_steps.size)):
        timer = helper.MyTimer()
        if i > 0:
            interval = evol_times[i] - evol_times[i - 1]
            nsteps = trotter_steps[i] - trotter_steps[i - 1]

        # QCircuit for precise time evolution between |t1_gt>_{i-1} and |t1_gt>_{i}.
        with timer("|t1_gt>"):
            # Create an (incremental) quantum circuit with several Trotter steps.
            qc = trotop.Trotter(
                num_qubits=num_qubits,
                evol_time=interval,
                num_steps=nsteps * precise_multiplier(),  # high accuracy
                delta=opts.delta,
                second_order=second_order,
            ).as_qcircuit(ini_state=QuantumCircuit(num_qubits))
            # Apply incremental qcircuit to the previous vector in MSP format.
            t1_gt = mpsop.qcircuit_mul_mps(qc, t1_gt, trunc_thr=thr)

        # QCircuit for normal time evolution between |t1>_{i-1} and |t1>_{i}.
        with timer("|t1>"):
            # Create an (incremental) quantum circuit with several Trotter steps.
            qc = trotop.Trotter(
                num_qubits=num_qubits,
                evol_time=interval,
                num_steps=nsteps,  # normal accuracy
                delta=opts.delta,
                second_order=second_order,
            ).as_qcircuit(ini_state=QuantumCircuit(num_qubits))
            # Apply incremental qcircuit to the previous vector in MSP format.
            t1 = mpsop.qcircuit_mul_mps(qc, t1, trunc_thr=thr)

        # Make a target structure and add it to the list of output results.
        with timer("preprocess"):
            targets.append(
                TargetMpsState(
                    opts=opts,
                    num_qubits=num_qubits,
                    num_trot_steps=trotter_steps[i],
                    evol_time=evol_times[i],
                    my_id=i,
                    t1_gt=t1_gt,
                    t1=t1,
                    second_order=second_order,
                )
            )

        # Print out execution times and fidelity between |t1_gt> and |t1>.
        metrics = timer.rounded_metrics(3)
        fid = trotop.fidelity(targets[-1].t1_gt, targets[-1].t1)
        _logger.info(
            "fidelity |t1_gt> vs |t1>: %0.6f, evol.time: %0.3f  |  exec.times: %s",
            fid,
            evol_times[i],
            metrics,
        )
    return targets


def get_target_mps_states(
    opts: Any, num_qubits: int, second_order: bool, input_file: Optional[str] = None
) -> List[TargetMpsState]:
    """
    Either loads a list of precomputed target states ``|t1_gt>``, ``|t1>`` from
    file or computes them from scratch (can be very slow) and saves into a file.

    Args:
        opts: user supplied options defined as a class.
        num_qubits: number of qubits.
        second_order: True, if the 2nd order Trotter is intended.
        input_file: path to a file with precomputed targets; if not specified,
                    default path will be generated.

    Returns:
        list of instances of TargetMpsState objects for all evolution times.
    """
    filename = os.path.join(opts.result_dir, f"target_mps_states_n{num_qubits}.pkl")
    if not bool(isinstance(input_file, str) and os.path.isfile(input_file)):
        input_file = filename
    if os.path.isfile(input_file):
        _logger.info("loading precomputed target MPS states |t1_gt> and |t1>")
        _logger.info("from the file: %s", input_file)
        with open(input_file, "rb") as fld:
            data = pickle.load(fld)
        if TargetMpsState.check_cached_data(opts, num_qubits, data):
            _logger.info("done, evol.times: %s", [round(d.evol_time, 3) for d in data])
            return data
        else:
            _logger.info("cached data don't match the expectation, recomputing ...")
            _logger.warning("altering some user options can cause recomputation")
    else:
        _logger.info("file of precomputed MPS targets was not found, recomputing ...")

    _logger.info("generating target states |t1_gt>, |t1> ..., can be slow")
    data = generate_all_mps_targets(opts=opts, num_qubits=num_qubits, second_order=second_order)
    assert TargetMpsState.check_cached_data(opts, num_qubits, data)

    _logger.info("storing precomputed target MPS states |t1_gt>, |t1>")
    _logger.info("to the file: %s", filename)
    with open(filename, "wb") as fld:
        pickle.dump(data, fld)

    return data


# -----------------------------------------------------------------------------
# Classic-only target state.
# -----------------------------------------------------------------------------


class TargetClassicState:
    """
    Structure keeps the target state |t1> as a classic vector and related data.
    """

    def __init__(
        self,
        *,
        opts: Any,
        num_qubits: int,
        num_trot_steps: int,
        evol_time: float,
        my_id: int,
        t1_gt: np.ndarray,
        t1: np.ndarray,
        second_order: bool,
    ):
        """
        Args:
            opts: user supplied options defined as a class or namespace.
            num_qubits: number of qubits.
            num_trot_steps: number of Trotter steps.
            evol_time: evolution time.
            my_id: unique integer identifier of this structure.
            t1_gt: ground-truth state |t1> from precise Trotter evolution.
            t1: state |t1> obtained from loose Trotter evolution.
            second_order: True, if the 2nd order Trotter is intended.
        """
        assert chk.is_int(num_qubits, num_qubits >= 2)
        assert chk.is_int(num_trot_steps, num_trot_steps in opts.trotter_steps)
        assert chk.is_float(evol_time, evol_time in opts.evol_times)
        assert chk.is_int(my_id, my_id >= 0)
        assert isinstance(t1_gt, np.ndarray)
        assert isinstance(t1, np.ndarray)
        assert isinstance(second_order, bool)

        self.num_qubits = int(num_qubits)  # #qubits
        self.num_trot_steps = int(num_trot_steps)  # #step in reference state |t1>
        self.precise_multiplier = precise_multiplier()  # #steps*mult for ground-truth
        self.delta = float(opts.delta)  # parameter in Hamiltonian
        self.evol_time = float(evol_time)  # evolution time (aka time horizon)
        self.my_id = int(my_id)  # unique id, here the no. of time horizon
        self.t1_gt = t1_gt  # ground-truth target state
        self.t1 = t1  # less accurate reference state
        self.second_order = second_order  # True for the 2nd order Trotter

    @staticmethod
    def check_cached_data(opts: Any, num_qubits: int, data: List[Any]) -> bool:
        """
        Checks that the structure of cached target states and related data
        match the expectation.

        Args:
            opts: user supplied options defined as a class.
            num_qubits: number of qubits.
            data: list of instances of TargetClassicState class to be verified.

        Returns:
            non-zero if elements of ``data`` match expected pattern.
        """
        assert chk.is_int(num_qubits, num_qubits >= 2) and chk.is_list(data)
        for i in range(min(len(data), len(opts.evol_times), len(opts.trotter_steps))):
            dat, t, s = data[i], opts.evol_times[i], opts.trotter_steps[i]
            if not (
                isinstance(dat, TargetClassicState)
                and hasattr(dat, "num_qubits")
                and hasattr(dat, "num_trot_steps")
                and hasattr(dat, "precise_multiplier")
                and hasattr(dat, "delta")
                and hasattr(dat, "evol_time")
                and hasattr(dat, "my_id")
                and hasattr(dat, "t1_gt")
                and hasattr(dat, "t1")
                and hasattr(dat, "second_order")
                and dat.num_qubits == num_qubits
                and dat.num_trot_steps == s
                and dat.precise_multiplier == precise_multiplier()
                and bool(np.isclose(dat.delta / opts.delta, 1))
                and chk.is_float(dat.evol_time, bool(np.isclose(dat.evol_time / t, 1)))
                and chk.is_int(dat.my_id, dat.my_id == i)
                and isinstance(dat.t1_gt, np.ndarray)
                and isinstance(dat.t1, np.ndarray)
                and isinstance(dat.second_order, bool)
            ):
                return False
        return True


def generate_classic_target(
    *,
    opts: Any,
    num_qubits: int,
    num_trot_steps: int,
    evol_time: float,
    my_id: int,
    second_order: bool,
) -> TargetClassicState:
    """
    Computes states using accurate and normal Trotterization.

    Args:
        opts: user supplied options defined as a class.
        num_qubits: number of qubits.
        num_trot_steps: number of time steps in Trotterization.
        evol_time: evolution time.
        my_id: unique integer identifier of the instance of TargetClassicState
               structure.
        second_order: True, if the 2nd order Trotter is intended.

    Returns:
        instance of TargetClassicState structure.
    """
    _logger.info("running the function: %s ...", generate_classic_target.__name__)

    def _ini_state() -> QuantumCircuit:
        """Returns the generator of initial state defined in Options."""
        return opts.ini_state_func[0](num_qubits)

    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(num_trot_steps, num_trot_steps >= 1)
    assert chk.is_float(evol_time, evol_time > 0)
    assert chk.is_int(my_id, my_id >= 0)
    assert isinstance(second_order, bool)

    timer = helper.MyTimer()

    # |t1_gt> = precise_Trotter(time) |ini_state>
    with timer("|t1_gt>"):
        # Create Trotter qcircuit covering time interval from 0 to current time.
        trot = trotop.Trotter(
            num_qubits=num_qubits,
            evol_time=evol_time,
            num_steps=num_trot_steps * precise_multiplier(),  # high accuracy
            delta=opts.delta,
            second_order=second_order,
        )
        # Apply qcircuit to the initial state vector.
        t1_gt = trot.as_vector(ini_state=_ini_state())

    # |t1> = reference_Trotter(time) |ini_state>
    with timer("|t1>"):
        # Create Trotter qcircuit covering time interval from 0 to current time.
        trot = trotop.Trotter(
            num_qubits=num_qubits,
            evol_time=evol_time,
            num_steps=num_trot_steps,  # normal accuracy
            delta=opts.delta,
            second_order=second_order,
        )
        # Apply qcircuit to the initial state vector.
        t1 = trot.as_vector(ini_state=_ini_state())

    metrics = timer.rounded_metrics(3)
    fid = trotop.fidelity(t1_gt, t1)
    _logger.info(
        "fidelity |t1_gt> vs |t1>: %0.6f, evol.time: %0.3f  |  exec.times: %s",
        fid,
        evol_time,
        metrics,
    )

    return TargetClassicState(
        opts=opts,
        num_qubits=num_qubits,
        num_trot_steps=num_trot_steps,
        evol_time=evol_time,
        my_id=my_id,
        t1_gt=t1_gt,
        t1=t1,
        second_order=second_order,
    )


def get_target_classic_states(
    opts: Any, num_qubits: int, second_order: bool, input_file: Optional[str] = None
) -> List[TargetClassicState]:
    """
    Either loads a list of precomputed target states ``|t1_gt>``, ``|t1>`` from
    file or computes them from scratch (can be very slow) and saves into a file.

    Args:
        opts: user supplied options defined as a class.
        num_qubits: number of qubits.
        second_order: True, if the 2nd order Trotter is intended.
        input_file: path to a file with precomputed targets; if not specified,
                    default path will be generated.

    Returns:
        list of instances of TargetClassicStates objects for all evolution times.
    """
    filename = os.path.join(opts.result_dir, f"target_classic_states_n{num_qubits}.pkl")
    if not bool(isinstance(input_file, str) and os.path.isfile(input_file)):
        input_file = filename
    if os.path.isfile(input_file):
        _logger.info("loading precomputed target classic states |t1_gt> and |t1>")
        _logger.info("from the file: %s", input_file)
        with open(input_file, "rb") as fld:
            data = pickle.load(fld)
        if TargetClassicState.check_cached_data(opts, num_qubits, data):
            _logger.info("done, evol.times: %s", [round(d.evol_time, 3) for d in data])
            return data
        else:
            _logger.info("cached data don't match the expectation, recomputing ...")
            _logger.warning("altering some user options can cause recomputation")
    else:
        _logger.info("file of precomputed targets was not found, recomputing ...")

    _logger.info("generating target states |t1_gt>, |t1> ..., can be slow")
    data = list([])
    for my_id, (nts, etm) in enumerate(zip(opts.trotter_steps, opts.evol_times)):
        data.append(
            generate_classic_target(
                opts=opts,
                num_qubits=num_qubits,
                num_trot_steps=nts,
                evol_time=etm,
                my_id=my_id,
                second_order=second_order,
            )
        )
    assert TargetClassicState.check_cached_data(opts, num_qubits, data)

    _logger.info("storing precomputed target classic states |t1_gt>, |t1>")
    _logger.info("to the file: %s", filename)
    with open(filename, "wb") as fld:
        pickle.dump(data, fld)

    return data


# -----------------------------------------------------------------------------
# Common utilities.
# -----------------------------------------------------------------------------


def get_target_states(
    opts: Any,
) -> Union[List[TargetClassicState], List[TargetMpsState]]:
    """
    Loads a list of precomputed target states or recomputes them.

    Args:
        opts: user supplied options defined as a class or namespace.

    Returns:
        list of target states.
    """
    if opts.use_mps:
        return get_target_mps_states(
            opts=opts,
            num_qubits=opts.num_qubits,
            second_order=opts.second_order_trotter,
            input_file=opts.targets_file,
        )
    else:
        return get_target_classic_states(
            opts=opts,
            num_qubits=opts.num_qubits,
            second_order=opts.second_order_trotter,
            input_file=opts.targets_file,
        )
