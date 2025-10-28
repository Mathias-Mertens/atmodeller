#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Non-linear solvers for chemical equilibrium and parameterised systems.

This module provides JAX-compatible solver utilities for efficiently handling both single-system
and batched systems of non-linear equations. The solvers are designed to integrate seamlessly with
JAX transformations support Equinox-based pytrees for flexible parameter handling.
"""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from equinox._enum import EnumerationItem
from jax import lax
from jaxmod.solvers import MultiTrySolution, batch_retry_solver
from jaxmod.utils import vmap_axes_spec
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray

from atmodeller.constants import TAU, TAU_MAX, TAU_NUM
from atmodeller.containers import Parameters
from atmodeller.engine import objective_function

LOG_NUMBER_DENSITY_VMAP_AXES: int = 0


# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def solver_single(
    initial_guess: Float[Array, "..."], parameters: Parameters, objective_function: Callable
) -> optx.Solution:
    """Solves a single system of non-linear equations.

    Args:
        initial_guess: Initial guess for the solution
        parameters: Model parameters required by the objective function and solver
        objective_function: Callable returning residuals for the system

    Returns:
        Optimistix solution object
    """
    sol: optx.Solution = optx.root_find(
        objective_function,
        parameters.solver_parameters.get_solver_instance(),
        initial_guess,
        args=parameters,
        throw=parameters.solver_parameters.throw,
        max_steps=parameters.solver_parameters.max_steps,
        options=parameters.solver_parameters.get_options(parameters.species.number_species),
    )

    return sol


def get_solver_individual(parameters: Parameters) -> Callable:
    """Gets a vmapped, JIT-compiled solver for independent batch systems.

    Wraps :func:`solver_single` with :func:`equinox.filter_vmap` and :func:`equinox.filter_jit` so
    that it can solve multiple independent systems in a batch efficiently. Each batch element is
    solved separately, producing per-element convergence statistics.

    Args:
        parameters: Model parameters required by the objective function and solver

    Returns:
        Callable
    """
    solver_fn: Callable = eqx.Partial(solver_single, objective_function=objective_function)

    return eqx.filter_jit(
        eqx.filter_vmap(
            solver_fn, in_axes=(LOG_NUMBER_DENSITY_VMAP_AXES, vmap_axes_spec(parameters))
        )
    )


def get_solver_batch(parameters: Parameters) -> Callable:
    """Gets a JIT-compiled solver for batched systems treated as a single problem.

    In this mode, the objective function is already vmapped across the batch dimension, so
    :func:`solver_single` sees the batch as one system. The solver returns a single convergence
    status and iteration count, which are broadcast to match the batch shape.

    Args:
        parameters: Model parameters required by the objective function and solver

    Returns:
        Callable
    """
    objective_vmap: Callable = eqx.filter_vmap(
        objective_function,
        in_axes=(LOG_NUMBER_DENSITY_VMAP_AXES, vmap_axes_spec(parameters)),
    )
    solver_fn: Callable = eqx.Partial(solver_single, objective_function=objective_vmap)

    @eqx.filter_jit
    def solver(solution: Array, parameters: Parameters) -> optx.Solution:
        sol: optx.Solution = solver_fn(solution, parameters)

        # FIXME: If want the arrays to be consistent with batch dimension then these need
        # broadcast and updating the sol object via tree surgery (tree_at)
        # Broadcast scalars to match batch dimension
        # batch_size: int = solution.shape[0]
        # solver_status_b: Bool[Array, " batch"] = jnp.broadcast_to(solver_status, (batch_size,))
        # solver_steps_b: Integer[Array, " batch"] = jnp.broadcast_to(solver_steps, (batch_size,))
        # return sol_value, solver_status_b, solver_steps_b
        return sol

    return solver


# @eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=2)
def solver_tau_step(
    solver_fn: Callable,
    initial_guess: Float[Array, "batch solution"],
    parameters: Parameters,
    key: PRNGKeyArray,
) -> MultiTrySolution:
    """Solves a batch of solutions for a sequence of tau values using a solver function.

    This function iterates over a set of tau values and applies the solver function (``solver_fn``)
    to the batch of solutions at each tau step. It dynamically updates the ``tau`` value in the
    solver parameters for each iteration. This function is intended to be used inside
    ``jax.lax.scan`` to efficiently sweep over multiple tau values in a single compiled loop.

    Args:
        solver_fn: Callable that performs a single solve and returns an ``optx.Solution`` object
        initial_guess: Batched array of initial guesses for the solver
        parameters: Template :class:`~atmodeller.containers.Parameters` object containing the
            full solver configuration. The ``tau`` leaf inside
            :class:`~atmodeller.containers.SolverParameters` will be replaced at each step.
        key: JAX PRNG key for reproducible random perturbations

    Returns:
        Callable
    """

    def solve_tau_step(carry: tuple, tau: Float[Array, " batch"]) -> tuple[tuple, tuple]:
        """Performs a single solver step for a given batch of tau values.

        This function is intended to be used inside :func``jax.lax.scan`` to iterate over multiple
        tau values efficiently. It updates the ``tau`` leaf in the parameters, calls the
        :func:`repeat_solver` for the current batch, and returns the updated carry and results.

        Args:
            carry: Tuple of carry values
            tau: Array of tau values for the current step in the scan.

        Returns:
            new carry tuple, output tuple
        """
        (key, solution) = carry
        key, subkey = jax.random.split(key)

        # Get new parameters with tau value
        get_leaf: Callable = lambda t: t.solver_parameters.tau  # noqa: E731
        new_parameters: Parameters = eqx.tree_at(get_leaf, parameters, tau)
        # jax.debug.print("tau = {out}", out=new_parameters.solver_parameters.tau)

        new_sol: MultiTrySolution = batch_retry_solver(
            solver_fn,
            solution,
            new_parameters,
            parameters.solver_parameters.multistart_perturbation,
            parameters.solver_parameters.multistart,
            subkey,
        )

        new_solution: Float[Array, "batch solution"] = new_sol.value
        new_result: optx.RESULTS = new_sol.result
        new_steps: Integer[Array, " batch"] = new_sol.stats["num_steps"]
        success_attempt: Integer[Array, " batch"] = new_sol.attempts

        new_carry: tuple[PRNGKeyArray, Float[Array, "batch solution"]] = (key, new_solution)

        # Output current solution for this tau step
        out: tuple[Array, ...] = (new_solution, new_result._value, new_steps, success_attempt)  # pyright: ignore

        return new_carry, out

    varying_tau_row: Float[Array, " tau"] = jnp.logspace(
        jnp.log10(TAU_MAX), jnp.log10(TAU), num=TAU_NUM
    )
    constant_tau_row: Float[Array, " tau"] = jnp.full((TAU_NUM,), TAU)
    tau_templates: Float[Array, "tau 2"] = jnp.stack([varying_tau_row, constant_tau_row], axis=1)

    # Create solver_status as a 1-D array of zeros with the same number of rows as initial_guess
    solver_status: jnp.ndarray = jnp.zeros(initial_guess.shape[0], dtype=int)

    tau_array: Float[Array, "tau batch"] = tau_templates[:, solver_status]

    initial_carry: tuple[Array, Array] = (key, initial_guess)

    _, results = jax.lax.scan(solve_tau_step, initial_carry, tau_array)
    solution, result_value, steps, attempts = results

    # Aggregate output
    # Solution and result for final TAU
    solution = solution[-1]
    result = cast(optx.RESULTS, EnumerationItem(result_value[-1], optx.RESULTS))  # pyright: ignore
    steps = jnp.sum(steps, axis=0)  # Sum steps for all tau
    attempts = jnp.max(attempts, axis=0)  # Max for all tau

    # Bundle the final solution into a single object
    sol_multi: MultiTrySolution = MultiTrySolution(
        solution, result, None, {"num_steps": steps}, None, attempts
    )

    return sol_multi


def make_solver(parameters: Parameters) -> Callable:
    """Solver function with JIT compilation. Handles multistart stability and generic solvers.

    Args:
        parameters: Parameters

    Returns:
        Solver
    """
    solver: Callable = get_solver_individual(parameters)

    # @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def solve_with_jit(
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "batch solution"],
        parameters: Parameters,
    ):
        """Wrapped version of the solve function with JIT compilation for branching logic."""

        # Define the condition to check if active stability is enabled
        condition: Bool[Array, ""] = jnp.any(parameters.species.active_stability)

        def multistart_stability(key):
            """Function for multistart with stability"""
            subkey = jax.random.split(key)[1]  # Split only once and pass subkey
            return solver_tau_step(solver, base_solution_array, parameters, subkey)

        def multistart_generic(key):
            """Function for generic multistart"""
            subkey = jax.random.split(key)[1]  # Split only once and pass subkey
            return batch_retry_solver(
                solver,
                base_solution_array,
                parameters,
                parameters.solver_parameters.multistart_perturbation,
                parameters.solver_parameters.multistart,
                subkey,
            )

        # Use jax.lax.cond to select the branch based on the condition
        multi_sol = lax.cond(
            condition,
            lambda _: multistart_stability(key),  # True: Use stability solver
            lambda _: multistart_generic(key),  # False: Use generic solver
            operand=None,  # Operand not used for decision making
        )

        return multi_sol

    return solve_with_jit
