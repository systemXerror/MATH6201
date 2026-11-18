"""MATH 6201: periodic 1D Allen–Cahn solvers.
The grid is [0, 1); x=1 is identified with x=0.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import splu

Array: TypeAlias = NDArray[np.float64]
Source: TypeAlias = Callable[[float], ArrayLike]
Advance: TypeAlias = Callable[[Array, ArrayLike], Array]


def _positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _vector(values: ArrayLike, name: str, *, min_size: int = 1) -> Array:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < min_size:
        raise ValueError(f"{name} must be a one-dimensional array with at least {min_size} entries.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _grid_size(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError("n must be an integer.")
    n = int(n)
    if n < 3:
        raise ValueError("Use at least three periodic nodes.")
    return n


def _step_source(source: ArrayLike, n: int) -> ArrayLike:
    forcing = np.asarray(source, dtype=float)
    if forcing.ndim != 0 and forcing.shape != (n,):
        raise ValueError(f"source must be a scalar or a one-dimensional array of length {n}.")
    return forcing


def grid(n: int) -> tuple[Array, float]:  
    """Return n distinct periodic nodes and their spacing."""
    n = _grid_size(n)
    return np.arange(n) / n, 1.0 / n


def periodic_laplacian(n: int) -> csr_matrix:
    """Return the sparse centered second difference, including wraparound."""
    n = _grid_size(n)
    _, h = grid(n)
    # Offsets ±(n-1) put the two missing neighbours in the corners.
    return diags(
        [1.0, 1.0, -2.0, 1.0, 1.0], [-(n - 1), -1, 0, 1, n - 1],
        shape=(n, n), format="csr",
    ) / h**2


def reaction(u: Array) -> Array:
    values = np.asarray(u, dtype=float)
    return values - values**3


def discrete_energy(u: Array, eps: float) -> float:
    """Periodic rectangle rule, with one forward-difference edge per node."""
    _positive_finite(eps, "eps")
    u = _vector(u, "u", min_size=3)
    h = 1.0 / len(u)
    gradient = (np.roll(u, -1) - u) / h   
    '''Produces the next neighbour and supplies one forward difference per edge. '''
    density = 0.5 * eps**2 * gradient**2 + 0.25 * (u**2 - 1)**2
    return float(h * np.sum(density))


def l2_error(u: ArrayLike, exact: ArrayLike) -> float:
    """Discrete L2 norm on the unit interval."""
    u = _vector(u, "u")
    exact = np.asarray(exact, dtype=float)
    if exact.ndim == 0:
        if not np.isfinite(exact):
            raise ValueError("exact must contain only finite values.")
        exact = np.full_like(u, exact)
    else:
        exact = _vector(exact, "exact")
    if u.shape != exact.shape:
        raise ValueError("u and exact must have the same shape.")
    return float(np.sqrt(np.mean((u - exact)**2)))


def stepper(n: int, eps: float, dt: float, method: str) -> Advance:
    """Build a fixed-step update accepting old u and optional old-time source."""
    n = _grid_size(n)
    _positive_finite(eps, "eps")
    _positive_finite(dt, "dt")
    diffusion = eps**2 * periodic_laplacian(n)
    if method == "explicit":
        def advance(u: Array, source: ArrayLike = 0.0) -> Array:
            u = np.asarray(u, dtype=float)
            if u.shape != (n,):
                raise ValueError(f"u must be a one-dimensional array of length {n}.")
            source = _step_source(source, n)
            return u + dt * (diffusion @ u + reaction(u) + source)
    elif method == "semi-implicit":
        # Same matrix at every step. Only the right-hand side changes.
        lu = splu((eye(n, format="csc") - dt * diffusion).tocsc())

        def advance(u: Array, source: ArrayLike = 0.0) -> Array:
            u = np.asarray(u, dtype=float)
            if u.shape != (n,):
                raise ValueError(f"u must be a one-dimensional array of length {n}.")
            source = _step_source(source, n)
            return lu.solve(u + dt * (reaction(u) + source))
    else:
        raise ValueError(f"Unknown time method: {method}")
    return advance


@dataclass
class Run:
    u: Array
    dt: float
    t: float
    completed: bool
    escaped: bool
    peak: float
    times: Array
    energies: Array
    snapshots: dict[float, Array]


def evolve(
    initial: Array, eps: float, final_time: float, nsteps: int, method: str,
    source: Source | None = None, *, record_energy: bool = False,
    snapshot_steps: tuple[int, ...] = (),
) -> Run:
    """Take nsteps equal steps to final_time, unless the guard stops the run.

    'escaped' means max |u| exceeded 2 initially or on a step, or a
    nonfinite value occurred. The separate guard stops at max |u| > 10 or
    nonfinite data. Both cutoffs are diagnostics, not stability theorems.
    """
    if isinstance(nsteps, bool) or not isinstance(nsteps, Integral) or nsteps < 1:
        raise ValueError("nsteps must be a positive integer.")
    _positive_finite(final_time, "final_time")
    u = _vector(initial, "initial", min_size=3).copy()
    if any(
        isinstance(k, bool) or not isinstance(k, Integral) or k < 0 or k > nsteps
        for k in snapshot_steps
    ):
        raise ValueError("snapshot_steps must contain integer step indices from 0 to nsteps.")
    dt = final_time / nsteps
    advance = stepper(len(u), eps, dt, method)
    times = [0.0] if record_energy else []
    energies = [discrete_energy(u, eps)] if record_energy else []
    wanted = set(snapshot_steps)
    snapshots = {0.0: u.copy()} if 0 in wanted else {}
    peak = float(np.max(np.abs(u)))
    escaped, completed = peak > 2, peak <= 10
    last_t = 0.0
    step_indices = range(1, nsteps + 1) if completed else ()
    for k in step_indices:
        forcing = 0.0 if source is None else _step_source(source((k - 1) * dt), len(u))
        with np.errstate(over="ignore", invalid="ignore"):
            trial = advance(u, forcing)
        magnitude = float(np.max(np.abs(trial)))
        if not np.isfinite(magnitude):
            escaped, completed, peak = True, False, np.inf
            break
        u, last_t = trial, k * dt
        peak = max(peak, magnitude)
        escaped |= magnitude > 2
        if record_energy:
            times.append(last_t)
            energies.append(discrete_energy(u, eps))
        if k in wanted:
            snapshots[last_t] = u.copy()
        if magnitude > 10:
            completed = False
            break
    return Run(
        u, dt, last_t, completed, escaped, peak, np.asarray(times),
        np.asarray(energies), snapshots,
    )


def exact_uniform(a: float, t: float) -> float:
    """Exact uniform solution: u^2 solves the logistic equation."""
    if not np.isfinite(a):
        raise ValueError("a must be finite.")
    if not np.isfinite(t) or t < 0:
        raise ValueError("t must be finite and nonnegative.")
    if t == 0:
        return float(a)
    if a == 0:
        return 0.0
    decay = np.exp(-2 * t)
    if abs(a) >= 1:
        denominator_squared = -np.expm1(-2 * t) + decay * (1 / a) ** 2
        return float(np.copysign(1 / np.sqrt(denominator_squared), a))
    denominator_squared = a**2 * -np.expm1(-2 * t) + decay
    return float(a / np.sqrt(denominator_squared))


def manufactured(x: Array, t: float) -> Array:
    """A smooth periodic profile with both spatial and temporal variation."""
    return 0.4 * np.exp(-t) * np.sin(2 * np.pi * x)


def manufactured_source(
    x: Array, eps: float, *, laplacian: csr_matrix | None = None,
) -> Source:
    """Source for the PDE (analytic u_xx) or fixed-grid ODE (discrete D u)."""
    _positive_finite(eps, "eps")
    x = _vector(x, "x")
    if laplacian is not None and laplacian.shape != (len(x), len(x)):
        raise ValueError("laplacian must have one row and column per grid point.")

    def source(t: float) -> Array:
        u = manufactured(x, t)
        uxx = -(2 * np.pi)**2 * u if laplacian is None else laplacian @ u
        return -u - eps**2 * uxx - reaction(u)
    return source


def diffusion_and_reaction_limits(n: int, eps: float) -> tuple[float, float]:
    """Explicit linear limits: diffusion only, and linearized about u=±1."""
    n = _grid_size(n)
    _positive_finite(eps, "eps")
    _, h = grid(n)
    # Even n gives exactly 4/h²; floor(n/2) also covers odd grids.
    qmax = 4 * eps**2 * np.sin(np.pi * (n // 2) / n)**2 / h**2  
    '''Calculates the magnitude of the most negative diffusion eigenvalue—the grid mode that most restricts explicit Euler. The sine formula handles both odd and even grid sizes.'''
    return float(2 / qmax), float(2 / (qmax + 2))


def energy_rises(energies: Array) -> int:
    """Count upward steps above a roundoff tolerance; test unforced runs only."""
    energies = _vector(energies, "energies", min_size=0)
    tolerance = 1e-12 * np.maximum(1.0, np.abs(energies[:-1]))
    return int(np.count_nonzero(np.diff(energies) > tolerance))
