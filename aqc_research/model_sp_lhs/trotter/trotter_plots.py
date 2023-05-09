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
Plotting functions for numerical experiments.
"""

import os
from typing import List, Dict, Optional
import numpy as np
from matplotlib import rcParams, checkdep_usetex

rcParams["font.size"] = 12
_USETEX = checkdep_usetex(True)
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

# import matplotlib.font_manager as fontman
# print(fontman.get_font_names())


def plot_fidelity_profiles(
    results: List[Dict],
    output_dir: str,
    no_print_block_rep: bool = False,
    tag: Optional[str] = None,
):
    """
    Plots fidelity profiles for the approximations ``Trotter(|t1>)`` and
    ``|a1>`` vs the exact states ``|t1>`` for different evolution times.
    Stores the pictures in image files and optionally plots in a window.

    Args:
        results: simulation results.
        output_dir: output directory for the chart pictures.
        no_print_block_rep: enables to suppress printing of the number of
                            block repetitions in the title.
        tag: additional tag-string in file name to make it distinguishable.
    """
    plt.ioff()
    for blr in [1, 2, 3]:
        subresults = [res for res in results if res["block_reps"] == blr]
        if len(subresults) == 0:
            continue
        num_qubits = subresults[0]["num_qubits"]
        fig, ax_tm = plt.subplots(figsize=(10, 7), dpi=400)

        y_1 = [res["fid_a1_vs_gt"] for res in subresults]
        y_2 = [res["fid_t1_vs_gt"] for res in subresults]
        x_1 = [str(res["evol_time1"]) for res in subresults]
        x_2 = [str(res["num_layers"]) for res in subresults]
        x_3 = [str(res["num_trotter_steps"]) for res in subresults]

        ax_nl = ax_tm.twiny()  # axis for the number of layers in ansatz
        ax_ts = ax_tm.twiny()  # axis for the number of Trotter steps

        y_min = float(min(np.amin(y_1), np.amin(y_2)))
        decimals = min(-1, int(np.floor(np.log10(max(1.001e-5, 1 - y_min)))))
        y_min = np.round(max(0, y_min - 10**decimals), decimals=abs(decimals))
        num_y_ticks = 11

        ax_tm.set_ylim(y_min, 1.0)
        ax_tm.set_xlim(0, len(x_1) - 1)
        ax_tm.set_xticks(np.arange(len(x_1)), x_1)
        ax_tm.set_yticks(np.linspace(y_min, 1.0, num_y_ticks))
        ax_nl.set_xticks(np.arange(len(x_2)), x_2)
        ax_nl.set_xlim(0, len(x_2) - 1)
        ax_ts.set_xticks(np.arange(len(x_3)), x_3)
        ax_ts.set_xlim(0, len(x_3) - 1)

        ax_tm.set_xlabel("evolution time")
        ax_tm.set_ylabel("fidelity")
        ax_nl.set_xlabel("number of layers in ansatz")
        ax_ts.set_xlabel("number of Trotter steps")

        ansatz_color = plt.cm.colors.CSS4_COLORS["orangered"]
        trotter_color = plt.cm.colors.CSS4_COLORS["deepskyblue"]

        (p_1,) = ax_tm.plot(x_1, y_1, color=ansatz_color, label="Ansatz", lw=2)
        (p_2,) = ax_tm.plot(x_1, y_2, color=trotter_color, label="Trotter")
        ax_tm.legend(handles=[p_1, p_2], loc="best")
        ax_tm.grid()

        # Set up ticks of the additional axes.
        ax_nl.spines["bottom"].set_position(("outward", 40))
        ax_ts.spines["bottom"].set_position(("outward", 80))
        ax_nl.xaxis.set_label_position("bottom")
        ax_ts.xaxis.set_label_position("bottom")
        ax_nl.xaxis.set_ticks_position("bottom")
        ax_ts.xaxis.set_ticks_position("bottom")
        ax_nl.spines["bottom"].set_visible(True)
        ax_ts.spines["bottom"].set_visible(True)

        # Set colors of additional axes.
        # https://stackoverflow.com/questions/1982770/matplotlib-changing-the-color-of-an-axis
        ax_nl.tick_params(axis="x", colors=ansatz_color)
        ax_nl.xaxis.label.set_color(ansatz_color)
        ax_nl.spines["bottom"].set_color(ansatz_color)
        ax_ts.tick_params(axis="x", colors=trotter_color)
        ax_ts.xaxis.label.set_color(trotter_color)
        ax_ts.spines["bottom"].set_color(trotter_color)

        if _USETEX:
            title = (
                "Fidelity: $|a_1\\rangle$ and $|t_1\\rangle$ vs. "
                "ground-truth state $|t_{gt}\\rangle$,  $N_{qubits}$: " + f"{num_qubits}"
            )
        else:
            title = (
                f"Fidelity: |a1> vs |t1_gt> and |t1> vs ground-truth state |t1_gt>,  "
                f"num.qubits: {num_qubits}"
            )

        if not no_print_block_rep:
            title = title + f",  block repetition: {blr}"

        plt.title(title)
        fig.tight_layout()
        # plt.draw()
        name = f"fidelity_plot_n{num_qubits}_br{blr}"
        if isinstance(tag, str) and len(tag) > 0:
            name = name + "_" + tag
        fname = os.path.join(output_dir, name + ".png")
        plt.savefig(fname)
        plt.close(fig)
