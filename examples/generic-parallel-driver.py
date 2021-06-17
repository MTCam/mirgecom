"""Driver to demonstrate alternate vision for drivers."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import os
import logging
import pyopencl as cl

from mirgecom.mpi import mpi_entry_point

from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
)

from user_package import MySimulationObject

logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, casename="nozzle", user_input_file=None,
         snapshot_pattern="{casename}-{step:06d}-{rank:04d}.pkl",
         restart_step=None, restart_name=None,
         use_profiling=False, use_logmgr=False, use_lazy_eval=False):
    """Drive a generic example."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
        mode="wo", mpi_comm=comm)

    from mirgecom.simutil import create_array_context
    actx, queue = create_array_context(use_profiling, use_lazy_eval,
                                       cl_ctx=ctx_factory())
    logmgr_add_cl_device_info(logmgr, queue)

    if rank == 0:
        logging.info("Creating user simulation object.")

    usimu = MySimulationObject(
        array_context=actx, logmgr=logmgr, casename=casename,
        restart_name=restart_name, user_input_file=user_input_file
    )

    usimu.intialize_simuation()
    timestepper_function = usimu.timestepper_function()
    rhs_function = usimu.rhs_function()
    prestep_function = usimu.prestep_function()
    poststep_function = usimu.poststep_function()

    if rank == 0:
        logging.info("Getting current state from simulation object.")

    (current_step, current_t) = \
        usimu.get_current_step_time()

    (step_max, t_final) = usimu.get_step_limits()

    current_state = usimu.initial_state()

    if ((current_t >= t_final) or (current_step >= step_max)):
        raise ValueError("Nothing to do, loop limits already met.")

    if rank == 0:
        logging.info("Stepping.")

    while ((current_t < t_final) and (current_step < step_max)):

        state, dt = prestep_function(step=current_step,
                                     time=current_t,
                                     state=current_state)

        current_state = timestepper_function(state=state, t=current_t,
                                             dt=dt, rhs=rhs_function)

        current_t += dt
        current_step += 1

        poststep_function(step=current_step, time=current_t,
                          state=current_state)

    usimu.finalize_simuation()

    exit()


if __name__ == "__main__":
    import sys

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = \
        argparse.ArgumentParser(description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "nozzle"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    snapshot_pattern = "{casename}-{step:06d}-{rank:04d}.pkl"
    restart_step = None
    restart_name = None
    if(args.restart_file):
        print(f"Restarting from file {args.restart_file}")
        file_path, file_name = os.path.split(args.restart_file)
        restart_step = int(file_name.split("-")[1])
        restart_name = (file_name.split("-")[0]).replace("'", "")
        print(f"step {restart_step}")
        print(f"name {restart_name}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    main(restart_step=restart_step, restart_name=restart_name,
         user_input_file=input_file, snapshot_pattern=snapshot_pattern,
         use_profiling=args.profile, use_lazy_eval=args.lazy, use_logmgr=args.log)

# vim: foldmethod=marker
