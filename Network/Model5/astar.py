import heapq
import math


def _heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _neighbors(cell, allow_diagonal=True):
    x, y = cell
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in steps:
        yield x + dx, y + dy


def astar(grid, start, goal, allow_diagonal=True):
    """Run A* over an OccupancyGrid. Returns a list of cells or None."""
    if not grid.is_free(start) or not grid.is_free(goal):
        return None

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed.add(current)

        for nxt in _neighbors(current, allow_diagonal=allow_diagonal):
            if not grid.is_free(nxt):
                continue
            step_cost = _heuristic(current, nxt)
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative
            f_score = tentative + _heuristic(nxt, goal)
            heapq.heappush(open_heap, (f_score, nxt))

    return None


def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def simplify_path(cells):
    """Keep only turning points from a grid path."""
    if not cells or len(cells) <= 2:
        return cells or []

    simplified = [cells[0]]
    prev = cells[0]
    direction = (cells[1][0] - prev[0], cells[1][1] - prev[1])
    for i in range(1, len(cells) - 1):
        new_direction = (cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1])
        if new_direction != direction:
            simplified.append(cells[i])
            direction = new_direction
    simplified.append(cells[-1])
    return simplified

