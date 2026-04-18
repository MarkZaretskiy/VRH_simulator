import numpy as np
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
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
        if self.G0 <= 0:
            raise ValueError("G0 must be > 0")

        self.n_sites = len(self.energies)

    def pairwise_distances(self) -> np.ndarray:
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def pairwise_energy_cost(self) -> np.ndarray:
        ei = self.energies[:, None]
        ej = self.energies[None, :]
        return 0.5 * (np.abs(ei) + np.abs(ej) + np.abs(ei - ej))

    def build_scaled_conductance_matrix(self) -> tuple[csr_matrix, float]:
        """
        Build a symmetric sparse conductance matrix scaled by a global factor.

        The matrix entries are assembled from
            log(G_ij) = log(G0 / T) - 2*r_ij/xi - eps_ij/(kB*T)
        and then shifted by the largest active log-conductance before
        exponentiation. Solving the network with a globally scaled conductance
        matrix leaves node voltages unchanged while improving numerical
        stability. The caller is responsible for rescaling currents by
        exp(log_shift).
        """
        N = self.n_sites
        dist = self.pairwise_distances()
        eps = self.pairwise_energy_cost()
        log_G = np.log(self.G0 / self.temperature) - 2.0 * dist / self.xi - (
            eps / (self.kB * self.temperature)
        )

        active_mask = np.ones_like(log_G, dtype=bool)
        np.fill_diagonal(active_mask, False)

        if self.cutoff_distance is not None:
            active_mask &= dist <= self.cutoff_distance

        if self.max_neighbors is not None:
            pruned = np.zeros_like(active_mask, dtype=bool)
            for i in range(N):
                row = dist[i].copy()
                row[i] = np.inf
                nn_idx = np.argsort(row)[: self.max_neighbors]
                pruned[i, nn_idx] = True
            keep = pruned | pruned.T
            active_mask &= keep

        if self.min_conductance > 0.0:
            active_mask &= log_G >= np.log(self.min_conductance)

        if not np.any(active_mask):
            return csr_matrix((N, N), dtype=float), 0.0

        log_shift = float(np.max(log_G[active_mask]))
        G_scaled = np.zeros_like(log_G)
        G_scaled[active_mask] = np.exp(log_G[active_mask] - log_shift)

        return csr_matrix(G_scaled), log_shift

    @staticmethod
    def rescale_sparse_matrix(matrix: csr_matrix, scale_factor: float) -> csr_matrix:
        if np.isclose(scale_factor, 1.0):
            return matrix.copy()
        scaled_matrix = matrix.copy()
        scaled_matrix.data *= scale_factor
        return scaled_matrix

    def build_conductance_matrix(self) -> csr_matrix:
        """
        Build symmetric sparse conductance matrix C where C[i, j] = G_ij.
        """
        C_scaled, log_shift = self.build_scaled_conductance_matrix()
        scale_factor = float(np.exp(log_shift))
        return self.rescale_sparse_matrix(C_scaled, scale_factor)

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

    @staticmethod
    def contacts_are_connected(
        C: csr_matrix,
        left_nodes: np.ndarray | list[int],
        right_nodes: np.ndarray | list[int],
    ) -> bool:
        """
        Return True if any left contact node shares a connected component
        with any right contact node in the sparse conductance graph.
        """
        left_nodes = np.asarray(left_nodes, dtype=int)
        right_nodes = np.asarray(right_nodes, dtype=int)

        if left_nodes.size == 0 or right_nodes.size == 0:
            return False
        if C.shape[0] == 0:
            return False

        _, labels = connected_components(C, directed=False, return_labels=True)
        left_labels = np.unique(labels[left_nodes])
        right_labels = np.unique(labels[right_nodes])
        return np.intersect1d(left_labels, right_labels).size > 0

    @staticmethod
    def nodes_connected_to_boundary(
        C: csr_matrix,
        boundary_nodes: np.ndarray | list[int],
    ) -> np.ndarray:
        """
        Return a boolean mask marking nodes that belong to any connected
        component touching the Dirichlet boundary.
        """
        boundary_nodes = np.asarray(boundary_nodes, dtype=int)
        if C.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        if boundary_nodes.size == 0:
            return np.zeros(C.shape[0], dtype=bool)

        _, labels = connected_components(C, directed=False, return_labels=True)
        boundary_labels = np.unique(labels[boundary_nodes])
        return np.isin(labels, boundary_labels)

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
        boundary = np.unique(np.concatenate([left_nodes, right_nodes]))

        C_scaled, log_shift = self.build_scaled_conductance_matrix()
        if not self.contacts_are_connected(C_scaled, left_nodes, right_nodes):
            raise ValueError(
                "no conductive path connects left_nodes and right_nodes"
            )
        scale_factor = float(np.exp(log_shift))
        L_scaled = self.build_laplacian(C_scaled)
        # Components that do not touch either contact have no effect on the
        # two-terminal conductance, but they do make the reduced Laplacian
        # singular. Exclude them from the Dirichlet solve.
        active_mask = self.nodes_connected_to_boundary(C_scaled, boundary)
        active_nodes = np.flatnonzero(active_mask)

        V_boundary = np.zeros(N, dtype=float)
        V_boundary[left_nodes] = V_left
        V_boundary[right_nodes] = V_right

        V = np.zeros(N, dtype=float)
        V[boundary] = V_boundary[boundary]

        internal = np.setdiff1d(active_nodes, boundary)
        if internal.size > 0:
            L_ii = L_scaled[internal][:, internal]
            L_ib = L_scaled[internal][:, boundary]
            b = -L_ib @ V_boundary[boundary]
            V_internal = spsolve(L_ii, b)
            V[internal] = V_internal

        # Total current injected from left contact:
        # I_left = sum_{i in left} sum_j G_ij * (V_i - V_j)
        total_current_scaled = 0.0
        for i in left_nodes:
            row = C_scaled.getrow(i)
            js = row.indices
            gij = row.data
            total_current_scaled += np.sum(gij * (V[i] - V[js]))

        delta_V = V_left - V_right
        if np.isclose(delta_V, 0.0):
            raise ValueError("V_left and V_right must be different")

        total_current = total_current_scaled * scale_factor
        effective_conductance = total_current / delta_V
        effective_resistance = np.inf if np.isclose(effective_conductance, 0.0) else 1.0 / effective_conductance
        C = self.rescale_sparse_matrix(C_scaled, scale_factor)
        L = self.rescale_sparse_matrix(L_scaled, scale_factor)

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
    x_min: float | None = None,
    x_max: float | None = None,
):
    """
    Define left/right contacts by x-coordinate.

    If x_min/x_max are provided, they are treated as the fixed device
    boundaries. Otherwise the bounds are inferred from the sampled sites.
    """
    x = positions[:, 0]
    if x_min is None:
        x_min = float(np.min(x))
    else:
        x_min = float(x_min)
    if x_max is None:
        x_max = float(np.max(x))
    else:
        x_max = float(x_max)
    if x_max < x_min:
        raise ValueError("x_max must be >= x_min")

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
