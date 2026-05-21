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


def _line_of_sight(grid, cell_a, cell_b):
    """Bresenham line check: True if all cells between A and B are free."""
    x0, y0 = cell_a
    x1, y1 = cell_b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if (x, y) != cell_a and (x, y) != cell_b and not grid.is_free((x, y)):
            return False
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True


def simplify_path(cells, grid=None):
    """Keep only turning points from a grid path, with line-of-sight validation.

    When grid is provided, straight-line segments between turning points are
    validated against obstacles to prevent cutting through buildings.
    """
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

    # Validate and repair: if any segment cuts through obstacles, re-insert
    # the intermediate grid cells that were removed.
    if grid is not None and len(simplified) > 1:
        repaired = [simplified[0]]
        for i in range(len(simplified) - 1):
            if _line_of_sight(grid, simplified[i], simplified[i + 1]):
                repaired.append(simplified[i + 1])
            else:
                # Straight line hits obstacles — restore the original sub-path
                start_idx = cells.index(simplified[i])
                end_idx = len(cells) - cells[::-1].index(simplified[i + 1]) - 1
                repaired.extend(cells[start_idx + 1:end_idx + 1])
        return repaired

    return simplified

