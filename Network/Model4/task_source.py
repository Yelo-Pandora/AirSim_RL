import os
import sys

import airsim
import numpy as np

MODEL4_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL4_DIR not in sys.path:
    sys.path.insert(0, MODEL4_DIR)

import config
from airsim_nav_sampler import AirSimNavigationSampler


class Model4TaskSource:
    """
    Long-range task source that decomposes a global AirSim route into mid-range subgoals.
    """

    def __init__(self, client=None, vehicle_name=None, force_rebuild=False):
        self.client = client or airsim.MultirotorClient()
        self.client.confirmConnection()
        self.vehicle_name = vehicle_name or self._detect_vehicle_name()
        self.sampler = AirSimNavigationSampler(self.client, self.vehicle_name)
        self.sampler.load_or_build(force_rebuild=force_rebuild)
        self._points_array = np.array([p["point"] for p in self.sampler.points], dtype=np.float32)
        self._graph = self._build_graph()
        self._components, self._component_of = self._build_components()

    def _detect_vehicle_name(self):
        try:
            vehicles = self.client.listVehicles()
            return vehicles[0] if vehicles else "Drone1"
        except Exception:
            return "Drone1"

    def sample_task(self):
        route = self.sample_route_task()
        first_target = route["subgoals"][0] if route["subgoals"] else route["goal"]
        return {
            "start": route["start"].copy(),
            "target": np.array(first_target, dtype=np.float32).copy(),
            "start_region": route["start_region"],
            "goal_region": route["goal_region"],
            "distance": route["distance"],
            "route_subgoals": [np.array(p, dtype=np.float32).copy() for p in route["subgoals"]],
            "global_goal": route["goal"].copy(),
            "route_points": [np.array(p, dtype=np.float32).copy() for p in route["route_points"]],
        }

    def get_points(self):
        return self.sampler.points

    def sample_route_task(self):
        min_dist = float(getattr(config, "AIRSIM_NAV_GLOBAL_MIN_DISTANCE", 80.0))
        max_dist = float(getattr(config, "AIRSIM_NAV_GLOBAL_MAX_DISTANCE", 220.0))

        for _ in range(500):
            viable_components = [comp for comp in self._components if len(comp) >= 3]
            if not viable_components:
                break

            component = viable_components[np.random.randint(len(viable_components))]
            start_idx = component[np.random.randint(len(component))]
            start = self.sampler.points[start_idx]
            start_point = np.array(start["point"], dtype=np.float32)
            candidates = []
            for goal_idx in component:
                goal = self.sampler.points[goal_idx]
                if goal_idx == start_idx:
                    continue
                goal_point = np.array(goal["point"], dtype=np.float32)
                dist = float(np.linalg.norm(goal_point - start_point))
                if min_dist <= dist <= max_dist:
                    candidates.append((goal_idx, goal, dist))

            if not candidates:
                continue

            goal_idx, goal, pair_dist = candidates[np.random.randint(len(candidates))]
            route_indices = self._shortest_path(start_idx, goal_idx)
            if len(route_indices) < 2:
                continue

            route_points = [self._points_array[idx].copy() for idx in route_indices]
            subgoals = self._make_subgoals(route_points)
            if not subgoals:
                continue

            return {
                "start": start_point.copy(),
                "goal": np.array(goal["point"], dtype=np.float32).copy(),
                "start_region": start["region"],
                "goal_region": goal["region"],
                "distance": pair_dist,
                "route_points": route_points,
                "subgoals": subgoals,
            }

        raise RuntimeError("Failed to sample a valid long-range route task.")

    def _build_graph(self):
        neighbor_radius = float(getattr(config, "AIRSIM_NAV_GRAPH_NEIGHBOR_RADIUS", 35.0))
        max_neighbors = int(getattr(config, "AIRSIM_NAV_GRAPH_MAX_NEIGHBORS", 8))
        graph = {idx: [] for idx in range(len(self._points_array))}

        for idx, point in enumerate(self._points_array):
            deltas = self._points_array - point
            dists = np.linalg.norm(deltas, axis=1)
            neighbor_indices = np.argsort(dists)
            added = 0
            for nbr in neighbor_indices:
                if nbr == idx:
                    continue
                dist = float(dists[nbr])
                if dist > neighbor_radius:
                    break
                graph[idx].append((int(nbr), dist))
                added += 1
                if added >= max_neighbors:
                    break
        return graph

    def _build_components(self):
        components = []
        component_of = {}
        visited = set()

        for node in self._graph:
            if node in visited:
                continue

            stack = [node]
            visited.add(node)
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v, _ in self._graph.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)

            comp = sorted(comp)
            comp_idx = len(components)
            for n in comp:
                component_of[n] = comp_idx
            components.append(comp)

        return components, component_of

    def _shortest_path(self, start_idx, goal_idx):
        import heapq

        heap = [(0.0, int(start_idx))]
        dist = {int(start_idx): 0.0}
        prev = {}
        visited = set()

        while heap:
            cost, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == int(goal_idx):
                break
            for nbr, weight in self._graph.get(node, []):
                new_cost = cost + weight
                if new_cost < dist.get(nbr, float("inf")):
                    dist[nbr] = new_cost
                    prev[nbr] = node
                    heapq.heappush(heap, (new_cost, nbr))

        if int(goal_idx) not in dist:
            return []

        path = [int(goal_idx)]
        while path[-1] != int(start_idx):
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _make_subgoals(self, route_points):
        spacing = float(getattr(config, "AIRSIM_NAV_SUBGOAL_SPACING", 30.0))
        min_spacing = float(getattr(config, "AIRSIM_NAV_SUBGOAL_MIN_SPACING", 18.0))
        if len(route_points) < 2:
            return []

        subgoals = []
        accumulated = 0.0
        last_kept = route_points[0]

        for idx in range(1, len(route_points)):
            pt = route_points[idx]
            seg = float(np.linalg.norm(pt - last_kept))
            accumulated += seg

            is_final = idx == len(route_points) - 1
            if accumulated >= spacing or (is_final and accumulated >= min_spacing):
                subgoals.append(pt.copy())
                last_kept = pt
                accumulated = 0.0

        if not subgoals or np.linalg.norm(subgoals[-1] - route_points[-1]) > 1e-3:
            subgoals.append(route_points[-1].copy())

        return subgoals
