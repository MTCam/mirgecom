"""Demonstrate simple scalar lump advection."""

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
import logging
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial
from pytools.obj_array import make_obj_array

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedBoundary
from mirgecom.initializers import MulticomponentLump
from mirgecom.eos import IdealSingleGas

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    set_sim_state
)

logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_leap=False,
         use_profiling=False, rst_step=None, rst_name=None,
         casename="lumpy-scalars", use_logmgr=True):
    """Drive example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 3
    nel_1d = 16
    order = 3
    t_final = 0.005
    current_cfl = 1.0
    current_dt = .001
    current_t = 0
    constant_cfl = False
    nstatus = 1
    nrestart = 5
    nviz = 1
    nhealth = 1
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    box_ll = -5.0
    box_ur = 5.0

    nspecies = 4
    centers = make_obj_array([np.zeros(shape=(dim,)) for i in range(nspecies)])
    spec_y0s = np.ones(shape=(nspecies,))
    spec_amplitudes = np.ones(shape=(nspecies,))
    eos = IdealSingleGas()

    velocity = np.ones(shape=(dim,))

    initializer = MulticomponentLump(dim=dim, nspecies=nspecies,
                                     spec_centers=centers, velocity=velocity,
                                     spec_y0s=spec_y0s,
                                     spec_amplitudes=spec_amplitudes)
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_step:  # read the grid from restart data
        rst_fname = rst_pattern.format(cname=casename, step=rst_step, rank=rank)

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_fname)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        box_ll = -5.0
        box_ur = 5.0
        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                             extract_vars_for_logging, units_for_logging)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    if rst_step:
        current_t = restart_data["t"]
        current_step = rst_step
        current_state = restart_data["state"]
        if logmgr:
            from logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_state = initializer(nodes)

    visualizer = make_visualizer(discr)
    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    get_timestep = partial(inviscid_sim_timestep, discr=discr,
                           cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_graceful_exit(step, t, state, do_viz=False, do_restart=False,
                         message=None):
        if rank == 0:
            logger.info("Errors detected; attempting graceful exit.")
        if do_viz:
            my_write_viz(step=step, t=t, state=state)
        if do_restart:
            my_write_restart(step=step, t=t, state=state)
        if message is None:
            message = "Fatal simulation errors detected."
        raise RuntimeError(message)

    def my_write_viz(step, t, state, dv=None, exact=None, resid=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        if exact is None:
            exact = initializer(x_vec=nodes, eos=eos, t=t)
        if resid is None:
            resid = state - exact
        viz_fields = [("cv", state),
                      ("dv", dv),
                      ("exact", exact),
                      ("resid", resid)]
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        rst_data = {
            "local_mesh": local_mesh,
            "state": state,
            "t": t,
            "step": step,
            "global_nelements": global_nelements,
            "num_parts": nparts
        }
        from mirgecom.restart import write_restart_file
        write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(state, dv, exact):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure) \
           or check_range_local(discr, "vol", dv.pressure, .9, 1.1):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        from mirgecom.simutil import compare_fluid_solutions
        component_errors = compare_fluid_solutions(discr, state, exact)
        exittol = .09
        if max(component_errors) > exittol:
            health_error = True
            if rank == 0:
                logger.info("Solution diverged from exact soln.")

        return health_error

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_pre_step(step, t, dt, state):
        dv = None
        exact = None
        pre_step_errors = False

        if logmgr:
            logmgr.tick_before()

        from mirgecom.simutil import check_step
        do_viz = check_step(step=step, interval=nviz)
        do_restart = check_step(step=step, interval=nrestart)
        do_health = check_step(step=step, interval=nhealth)
        do_status = check_step(step=step, interval=nstatus)

        if step == rst_step:  # don't do viz or restart @ restart
            do_viz = False
            do_restart = False

        if do_health:
            dv = eos.dependent_vars(state)
            exact = initializer(x_vec=nodes, eos=eos, t=t)
            health_errors = my_health_check(state, dv, exact)
            if comm is not None:
                health_errors = comm.allreduce(health_errors, op=MPI.LOR)
            if health_errors and rank == 0:
                logger.info("Fluid solution failed health check.")
            pre_step_errors = pre_step_errors or health_errors

        if do_restart:
            my_write_restart(step=step, t=t, state=state)

        if do_viz:
            if dv is None:
                dv = eos.dependent_vars(state)
            if exact is None:
                exact = initializer(x_vec=nodes, eos=eos, t=t)
            resid = state - exact
            my_write_viz(step=step, t=t, state=state, dv=dv, exact=exact,
                         resid=resid)

        if do_status:
            if exact is None:
                exact = initializer(x_vec=nodes, eos=eos, t=t)
            from mirgecom.simutil import compare_fluid_solutions
            component_errors = compare_fluid_solutions(discr, state, exact)
            status_msg = (
                "------- errors="
                + ", ".join("%.3g" % en for en in component_errors))
            if rank == 0:
                logger.info(status_msg)

        if pre_step_errors:
            my_graceful_exit(step=step, t=t, state=state,
                             do_viz=(not do_viz), do_restart=(not do_restart),
                             message="Error detected at prestep, exiting.")

        return state, dt

    def my_rhs(t, state):
        return euler_operator(discr, cv=state, t=t,
                              boundaries=boundaries, eos=eos)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step, dt=current_dt,
                      post_step_callback=my_post_step,
                      get_timestep=get_timestep, state=current_state,
                      t=current_t, t_final=t_final, eos=eos, dim=dim)

    finish_tol = 1e-16
    if np.abs(current_t - t_final) > finish_tol:
        my_graceful_exit(step=current_step, t=current_t, state=current_state,
                         do_viz=True, do_restart=True,
                         message="Simulation timestepping did not complete.")

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    final_dv = eos.dependent_vars(current_state)
    final_exact = initializer(x_vec=nodes, eos=eos, t=current_t)
    final_resid = current_state - final_exact
    my_write_viz(step=current_step, t=current_t, state=current_state, dv=final_dv,
                 exact=final_exact, resid=final_resid)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main(use_leap=False)

# vim: foldmethod=marker
