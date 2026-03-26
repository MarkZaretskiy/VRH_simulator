import numpy as np
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class RRNResult:
    voltages: np.ndarray
    total_current: float
    effective_conductance: float
    effective_resistance: float
    conductance_matrix: csr_matrix
    laplacian_matrix: csr_matrix


class RRNSolver:
    """
    Random Resistor Network solver for hopping transport.
    Implementation based off https://arxiv.org/pdf/2601.01243
    
    Main idea:
      1. Build pairwise conductances G_ij
      2. Assemble Kirchhoff / graph Laplacian matrix L
      3. Impose Dirichlet boundary conditions on contact nodes
      4. Solve for node voltages
      5. Compute total injected current and effective conductance

    Conductance model:
        G_ij = (G0 / T) * exp(-2*r_ij/xi) * exp(-eps_ij/(kB*T))

    with
        eps_ij = 0.5 * (|eps_i| + |eps_j| + |eps_i - eps_j|)

    Here G0 represents the temperature-independent prefactor q^2 * v0 / kB
    from Eq. (1) of the paper, so the full conductance keeps the 1/T scaling.
    """

    def __init__(
        self,
        positions: np.ndarray,
        energies: np.ndarray,
        temperature: float,
        xi: float,
        G0: float = 1.0,
        kB: float = 8.617333262145e-5,  # eV/K
        cutoff_distance: float | None = None,
        max_neighbors: int | None = None,
        min_conductance: float = 1e-30,
    ):
        """
        Parameters
        ----------
        positions : (N, d) ndarray
            Spatial coordinates of sites.
        energies : (N,) ndarray
            Site energies in eV relative to Fermi level.
        temperature : float
            Temperature in K.
        xi : float
            Localization length in same spatial units as positions.
        G0 : float
            Temperature-independent conductance prefactor q^2 * v0 / kB.
        kB : float
            Boltzmann constant in eV/K.
        cutoff_distance : float or None
            If set, only pairs with r_ij <= cutoff_distance are connected.
        max_neighbors : int or None
            If set, only the closest max_neighbors nodes for each site are kept.
        min_conductance : float
            Numerical threshold; weaker edges are discarded.
        """
        self.positions = np.asarray(positions, dtype=float)
        self.energies = np.asarray(energies, dtype=float)
        self.temperature = float(temperature)
        self.xi = float(xi)
        self.G0 = float(G0)
        self.kB = float(kB)
        self.cutoff_distance = cutoff_distance
        self.max_neighbors = max_neighbors
        self.min_conductance = float(min_conductance)

        if self.positions.ndim != 2:
            raise ValueError("positions must have shape (N, d)")
        if self.energies.ndim != 1:
            raise ValueError("energies must have shape (N,)")
        if len(self.positions) != len(self.energies):
            raise ValueError("positions and energies must have the same length")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.xi <= 0:
            raise ValueError("xi must be > 0")

        self.n_sites = len(self.energies)

    def pairwise_distances(self) -> np.ndarray:
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def pairwise_energy_cost(self) -> np.ndarray:
        ei = self.energies[:, None]
        ej = self.energies[None, :]
        return 0.5 * (np.abs(ei) + np.abs(ej) + np.abs(ei - ej))

    def build_conductance_matrix(self) -> csr_matrix:
        """
        Build symmetric sparse conductance matrix C where C[i, j] = G_ij.
        """
        N = self.n_sites
        dist = self.pairwise_distances()
        eps = self.pairwise_energy_cost()

        with np.errstate(over="ignore", under="ignore"):
            G = (self.G0 / self.temperature) * np.exp(-2.0 * dist / self.xi) * np.exp(
                -eps / (self.kB * self.temperature)
            )

        np.fill_diagonal(G, 0.0)

        if self.cutoff_distance is not None:
            G[dist > self.cutoff_distance] = 0.0

        if self.max_neighbors is not None:
            # Keep only nearest max_neighbors for each row
            pruned = np.zeros_like(G, dtype=bool)
            for i in range(N):
                row = dist[i].copy()
                row[i] = np.inf
                nn_idx = np.argsort(row)[: self.max_neighbors]
                pruned[i, nn_idx] = True
            # Symmetrize neighbor mask
            keep = pruned | pruned.T
            G[~keep] = 0.0

        G[G < self.min_conductance] = 0.0

        return csr_matrix(G)

    @staticmethod
    def build_laplacian(C: csr_matrix) -> csr_matrix:
        """
        Graph Laplacian L:
            L_ii = sum_j G_ij
            L_ij = -G_ij, i != j
        """
        degrees = np.array(C.sum(axis=1)).ravel()
        L = -C.copy().tolil()
        L.setdiag(degrees)
        return L.tocsr()

    def solve(
        self,
        left_nodes: np.ndarray | list[int],
        right_nodes: np.ndarray | list[int],
        V_left: float = 1.0,
        V_right: float = 0.0,
    ) -> RRNResult:
        """
        Solve the RRN with Dirichlet boundary conditions on left/right contact nodes.
        """
        left_nodes = np.asarray(left_nodes, dtype=int)
        right_nodes = np.asarray(right_nodes, dtype=int)

        if left_nodes.size == 0 or right_nodes.size == 0:
            raise ValueError("left_nodes and right_nodes must be non-empty")
        if np.intersect1d(left_nodes, right_nodes).size > 0:
            raise ValueError("left_nodes and right_nodes must not overlap")

        N = self.n_sites
        all_nodes = np.arange(N)
        boundary = np.unique(np.concatenate([left_nodes, right_nodes]))
        internal = np.setdiff1d(all_nodes, boundary)

        C = self.build_conductance_matrix()
        L = self.build_laplacian(C)

        V_boundary = np.zeros(N, dtype=float)
        V_boundary[left_nodes] = V_left
        V_boundary[right_nodes] = V_right

        V = np.zeros(N, dtype=float)
        V[boundary] = V_boundary[boundary]

        if internal.size > 0:
            L_ii = L[internal][:, internal]
            L_ib = L[internal][:, boundary]
            b = -L_ib @ V_boundary[boundary]
            V_internal = spsolve(L_ii, b)
            V[internal] = V_internal

        # Total current injected from left contact:
        # I_left = sum_{i in left} sum_j G_ij * (V_i - V_j)
        total_current = 0.0
        for i in left_nodes:
            row = C.getrow(i)
            js = row.indices
            gij = row.data
            total_current += np.sum(gij * (V[i] - V[js]))

        delta_V = V_left - V_right
        if np.isclose(delta_V, 0.0):
            raise ValueError("V_left and V_right must be different")

        effective_conductance = total_current / delta_V
        effective_resistance = np.inf if np.isclose(effective_conductance, 0.0) else 1.0 / effective_conductance

        return RRNResult(
            voltages=V,
            total_current=float(total_current),
            effective_conductance=float(effective_conductance),
            effective_resistance=float(effective_resistance),
            conductance_matrix=C,
            laplacian_matrix=L,
        )


def make_random_sites(
    n_sites: int,
    length: float,
    dim: int = 2,
    energy_std: float = 0.1,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, length, size=(n_sites, dim))
    energies = rng.normal(0.0, energy_std, size=n_sites)
    return positions, energies


def contact_nodes_from_x(
    positions: np.ndarray,
    contact_width: float,
):
    """
    Define left/right contacts by x-coordinate.
    """
    x = positions[:, 0]
    x_min = np.min(x)
    x_max = np.max(x)

    left_nodes = np.where(x <= x_min + contact_width)[0]
    right_nodes = np.where(x >= x_max - contact_width)[0]

    return left_nodes, right_nodes


if __name__ == "__main__":
    # Example usage
    positions, energies = make_random_sites(
        n_sites=300,
        length=20.0,
        dim=1,
        energy_std=0.08,
        seed=42,
    )

    left_nodes, right_nodes = contact_nodes_from_x(positions, contact_width=1.5)

    solver = RRNSolver(
        positions=positions,
        energies=energies,
        temperature=300.0,
        xi=2.0,
        G0=1.0,
        cutoff_distance=6.0,   # helps sparsity / speed
        max_neighbors=20,      # optional extra pruning
        min_conductance=1e-18,
    )

    result = solver.solve(
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        V_left=1.0,
        V_right=0.0,
    )

    print(f"Number of sites: {solver.n_sites}")
    print(f"Left contact nodes: {len(left_nodes)}")
    print(f"Right contact nodes: {len(right_nodes)}")
    print(f"Total current: {result.total_current:.6e}")
    print(f"Effective conductance: {result.effective_conductance:.6e}")
    print(f"Effective resistance: {result.effective_resistance:.6e}")
