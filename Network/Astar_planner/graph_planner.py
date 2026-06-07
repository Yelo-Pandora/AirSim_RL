import csv
import heapq
import math

import numpy as np

import config


class WaypointGraphPlanner:
    """
    Upper decision layer.

    Given a global start and goal, build a waypoint graph from known feasible
    dataset points and run A* to produce local target points.
    """

    def __init__(self, dataset_csv=None, waypoint_filter=None):
        self.dataset_csv = dataset_csv or config.DATASET_CSV
        self.waypoint_filter = waypoint_filter
        self.dataset_nodes = self._load_dataset_nodes()

    def _load_dataset_nodes(self):
        nodes = []
        seen = set()
        with open(self.dataset_csv, encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                for suffix, point, region in self._extract_points_from_row(row):
                    key = tuple(round(float(value), 4) for value in point.tolist())
                    if key in seen:
                        continue
                    if self.waypoint_filter is not None and not self.waypoint_filter(point):
                        continue
                    seen.add(key)
                    nodes.append({
                        "id": f"data_{index}_{suffix}",
                        "point": point,
                        "region": region,
                    })
        if not nodes:
            raise RuntimeError(f"No waypoint nodes loaded from {self.dataset_csv}")
        return nodes

    def _extract_points_from_row(self, row):
        """
        Support both dataset formats:
        1. Single-point waypoint CSV: x,y,z[,region]
        2. Task-pair CSV: start_x,start_y,start_z,end_x,end_y,end_z
        """
        if {"x", "y", "z"}.issubset(row.keys()):
            point = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=np.float32,
            )
            region = str(int(float(row.get("region", 0))))
            return [("point", point, region)]

        if {"start_x", "start_y", "start_z", "end_x", "end_y", "end_z"}.issubset(row.keys()):
            start = np.array(
                [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])],
                dtype=np.float32,
            )
            end = np.array(
                [float(row["end_x"]), float(row["end_y"]), float(row["end_z"])],
                dtype=np.float32,
            )
            return [
                ("start", start, "pair_start"),
                ("end", end, "pair_end"),
            ]

        raise KeyError(
            "Unsupported CSV columns. Expected x,y,z[,region] or "
            "start_x,start_y,start_z,end_x,end_y,end_z."
        )

    def plan(self, start, goal):
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)

        last_error = None
        for k_neighbors, max_edge_distance in config.GRAPH_FALLBACKS:
            nodes = self._with_endpoints(start, goal)
            adjacency = self._build_adjacency(nodes, k_neighbors, max_edge_distance)
            path_indices = self._astar(nodes, adjacency, 0, 1)
            if path_indices is None:
                last_error = (k_neighbors, max_edge_distance)
                continue

            path = [nodes[index]["point"].copy() for index in path_indices]
            return {
                "points": path,
                "node_ids": [nodes[index]["id"] for index in path_indices],
                "regions": [nodes[index]["region"] for index in path_indices],
                "k_neighbors": k_neighbors,
                "max_edge_distance": max_edge_distance,
                "path_length": self.path_length(path),
                "planner": "csv",
            }

        raise RuntimeError(
            f"Upper A* failed after fallbacks. Last k/distance={last_error}; "
            f"dataset={self.dataset_csv}"
        )

    def _with_endpoints(self, start, goal):
        return [
            {"id": "global_start", "point": start, "region": "start"},
            {"id": "global_goal", "point": goal, "region": "goal"},
            *self.dataset_nodes,
        ]

    def _build_adjacency(self, nodes, k_neighbors, max_edge_distance):
        points = np.array([node["point"] for node in nodes], dtype=np.float32)
        adjacency = [[] for _ in nodes]

        for index, point in enumerate(points):
            deltas = points - point
            distances = np.linalg.norm(deltas, axis=1)
            order = np.argsort(distances)
            added = 0
            for neighbor in order:
                if neighbor == index:
                    continue
                distance = float(distances[neighbor])
                if distance > max_edge_distance:
                    continue
                if abs(float(points[neighbor][2] - point[2])) > config.GRAPH_MAX_Z_DIFF:
                    continue
                cost = self._edge_cost(nodes[index], nodes[neighbor], distance)
                adjacency[index].append((int(neighbor), cost))
                added += 1
                if added >= k_neighbors:
                    break

        return adjacency

    def _edge_cost(self, node_a, node_b, distance):
        z_cost = abs(float(node_a["point"][2] - node_b["point"][2])) * config.GRAPH_VERTICAL_COST_WEIGHT
        region_cost = 0.0 if node_a["region"] == node_b["region"] else config.GRAPH_REGION_CHANGE_COST
        return float(distance + z_cost + region_cost)

    def _astar(self, nodes, adjacency, start_index, goal_index):
        open_heap = [(0.0, start_index)]
        came_from = {}
        g_score = {start_index: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_index:
                return self._reconstruct(came_from, current)
            closed.add(current)

            for neighbor, edge_cost in adjacency[current]:
                tentative = g_score[current] + edge_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + self._heuristic(nodes[neighbor]["point"], nodes[goal_index]["point"])
                heapq.heappush(open_heap, (priority, neighbor))
        return None

    @staticmethod
    def _heuristic(point, goal):
        return float(np.linalg.norm(point - goal))

    @staticmethod
    def _reconstruct(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def path_length(points):
        if len(points) < 2:
            return 0.0
        return float(sum(np.linalg.norm(points[index] - points[index - 1]) for index in range(1, len(points))))
