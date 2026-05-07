"""
A* pathfinding in 3D voxel grid for EGO-Planner guiding path generation.
Returns collision-free paths that naturally hug obstacle surfaces.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional

from .voxel_grid import LocalVoxelGrid


class _Node:
    """A* search node with priority queue support."""
    __slots__ = ['pos', 'g', 'h', 'f', 'parent']

    def __init__(self, pos: Tuple[int, int, int], g: float = 0.0,
                 h: float = 0.0, parent: Optional['_Node'] = None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f


class AStarPlanner:
    """3D A* path planner on a voxel grid."""

    def __init__(self, voxel_grid: LocalVoxelGrid,
                 neighborhood: int = 26,
                 safety_radius: float = 0.5,
                 heuristic_weight: float = 1.0):
        """
        Args:
            voxel_grid: the local voxel grid for collision checks
            neighborhood: 6 or 26 connected neighbors
            safety_radius: clearance radius for collision checking
            heuristic_weight: weight for heuristic (1.0 = optimal, >1 = faster but suboptimal)
        """
        self.grid = voxel_grid
        self.neighborhood = neighborhood
        self.safety_radius = safety_radius
        self.heuristic_weight = heuristic_weight
        self._neighbors = self._build_neighbors()

    def _build_neighbors(self) -> List[Tuple[int, int, int]]:
        """Build the set of neighbor offsets."""
        if self.neighborhood == 6:
            return [(1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1)]
        else:  # 26-connected
            neighbors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbors.append((dx, dy, dz))
            return neighbors

    def _world_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        return self.grid._world_to_voxel(point)

    def _voxel_to_world(self, vx: int, vy: int, vz: int) -> np.ndarray:
        return self.grid._voxel_to_world(vx, vy, vz)

    def _is_in_bounds(self, vx: int, vy: int, vz: int) -> bool:
        return self.grid._is_in_bounds(vx, vy, vz)

    def _is_free(self, vx: int, vy: int, vz: int) -> bool:
        """Check if a voxel is free (not occupied and in bounds)."""
        if not self._is_in_bounds(vx, vy, vz):
            return False
        if self.grid.grid[vx, vy, vz]:
            return False
        # Check safety radius
        world_pt = self._voxel_to_world(vx, vy, vz)
        return not self.grid._check_sphere(world_pt, self.safety_radius * 0.5)

    def _heuristic(self, a: Tuple[int, int, int],
                   b: Tuple[int, int, int]) -> float:
        """Euclidean distance heuristic in world space."""
        wa = self._voxel_to_world(*a)
        wb = self._voxel_to_world(*b)
        return np.linalg.norm(wa - wb) * self.heuristic_weight

    def plan(self, start: np.ndarray,
             goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Find a collision-free path from start to goal.

        Args:
            start: (3,) start position in world coordinates
            goal: (3,) goal position in world coordinates

        Returns:
            List of (N, 3) world-coordinate waypoints, or None if no path found.
        """
        sv = self._world_to_voxel(start)
        gv = self._world_to_voxel(goal)

        if not self._is_in_bounds(*sv):
            return None
        if not self._is_in_bounds(*gv):
            # Try to find nearest free voxel to goal
            gv = self._find_nearest_free(gv)
            if gv is None:
                return None

        if not self._is_free(*sv):
            return None
        if not self._is_free(*gv):
            return None

        # A* search
        start_node = _Node(sv, g=0.0, h=self._heuristic(sv, gv))
        open_set = [start_node]
        closed = set()
        g_scores = {sv: 0.0}

        while open_set:
            current = heapq.heappop(open_set)

            if current.pos == gv:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(self._voxel_to_world(*node.pos))
                    node = node.parent
                path.reverse()
                return path

            closed.add(current.pos)

            for dx, dy, dz in self._neighbors:
                nb_pos = (current.pos[0] + dx,
                          current.pos[1] + dy,
                          current.pos[2] + dz)

                if nb_pos in closed:
                    continue
                if not self._is_free(*nb_pos):
                    continue

                # Cost: Euclidean distance in world space
                cw = self._voxel_to_world(*current.pos)
                nw = self._voxel_to_world(*nb_pos)
                step_cost = np.linalg.norm(cw - nw)
                tentative_g = current.g + step_cost

                if tentative_g < g_scores.get(nb_pos, float('inf')):
                    g_scores[nb_pos] = tentative_g
                    h = self._heuristic(nb_pos, gv)
                    child = _Node(nb_pos, g=tentative_g, h=h, parent=current)
                    heapq.heappush(open_set, child)

        return None  # No path found

    def _find_nearest_free(self, target: Tuple[int, int, int],
                           max_radius: int = 10) -> Optional[Tuple[int, int, int]]:
        """BFS from target voxel to find nearest free voxel."""
        if self._is_free(*target):
            return target

        visited = {target}
        queue = [target]

        for radius in range(max_radius):
            next_queue = []
            for pos in queue:
                for dx, dy, dz in self._neighbors:
                    nb = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if nb in visited:
                        continue
                    if not self._is_in_bounds(*nb):
                        continue
                    visited.add(nb)
                    if self._is_free(*nb):
                        return nb
                    next_queue.append(nb)
            queue = next_queue
            if not queue:
                break

        return None

    def plan_straight_with_obstacle_check(self, start: np.ndarray,
                                          goal: np.ndarray,
                                          n_samples: int = 20) -> bool:
        """Check if the straight line from start to goal is collision-free."""
        for i in range(n_samples):
            t = i / (n_samples - 1) if n_samples > 1 else 0
            pt = start + t * (goal - start)
            if self.grid.is_occupied(pt, self.safety_radius * 0.5):
                return False
        return True
