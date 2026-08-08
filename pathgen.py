"""Algorithm 1 from the SpaceForge proposal: random returnable paths on a square grid.

Generates an outbound self-avoiding walk P_out, a BFS return route P_ret that
avoids reusing the outbound corridor, and optional chord corridors C that add
extra cycles (and therefore junctions with left/right decisions).

Cells are (x, y) with x = column, y = row; y grows downward (image convention),
matching the world/render coordinate system.
"""

import heapq
import random
from collections import deque


def neighbours(c, n):
    x, y = c
    return [(nx, ny) for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if 0 <= nx < n and 0 <= ny < n]


def shortest_path(a, b, allowed, n, turn_penalty=0.0):
    """Shortest path from a to b through `allowed` cells, endpoints inclusive.

    With turn_penalty == 0 this is plain BFS. With turn_penalty > 0 it is a
    Dijkstra over (cell, incoming-direction) states where each direction
    change costs extra, so routes prefer long straight runs — corridors look
    like corridors instead of staircases. Returns None if b is unreachable.
    """
    if a == b:
        return [a]
    if turn_penalty <= 0.0:
        prev = {a: None}
        q = deque([a])
        while q:
            c = q.popleft()
            for nb in neighbours(c, n):
                if nb in allowed and nb not in prev:
                    prev[nb] = c
                    if nb == b:
                        path = [b]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    q.append(nb)
        return None

    start = (a, (0, 0))
    dist = {start: 0.0}
    par = {}
    cnt = 0
    pq = [(0.0, cnt, start)]
    while pq:
        d, _, st = heapq.heappop(pq)
        c, dr = st
        if d > dist.get(st, 1e18) + 1e-12:
            continue
        if c == b:
            path = [c]
            while st in par:
                st = par[st]
                path.append(st[0])
            return path[::-1]
        for nb in neighbours(c, n):
            if nb not in allowed:
                continue
            ndr = (nb[0] - c[0], nb[1] - c[1])
            nd = d + 1.0 + (turn_penalty if dr != (0, 0) and ndr != dr else 0.0)
            key = (nb, ndr)
            if nd < dist.get(key, 1e18) - 1e-12:
                dist[key] = nd
                par[key] = (c, dr)
                cnt += 1
                heapq.heappush(pq, (nd, cnt, key))
    return None


def _ret_junctions(p_out, p_ret, chords):
    """Interior cells of the return route with corridor degree >= 3."""
    deg = {}
    for r in [p_out, p_ret, *chords]:
        for u, v in zip(r[:-1], r[1:]):
            deg.setdefault(u, set()).add(v)
            deg.setdefault(v, set()).add(u)
    return [c for c in p_ret[1:-1] if len(deg[c]) >= 3]


def _route_bends(route):
    """Interior cells where the route changes direction."""
    return {route[i] for i in range(1, len(route) - 1)
            if (route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1]) !=
               (route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])}


def generate_returnable_path(n, target_len, min_len, max_attempts, n_chords, rng,
                             chord_tries=120, min_return_len=2,
                             bias_return_junction=False, min_return_decisions=2,
                             straight_bias=0.0, turn_penalty=0.0):
    """Algorithm 1. rng is a random.Random instance.

    n = grid size N, target_len = L, min_len = Lmin, max_attempts = T,
    n_chords = k. Guards not in the pseudocode: `chord_tries` caps the
    chord-search loop (line 16 could otherwise spin forever on a saturated
    grid), and `min_return_len` rejects trivially short return routes.

    bias_return_junction (deviation from line 17) turns the chord phase into a
    loop-maker aimed at the return route: it repeatedly connects a *bend* of
    P_ret to another P_ret bend (falling back to any loop cell), so each chord
    closes a rectangular cycle whose corners are genuine left/right decisions
    on the way back. It keeps going past k chords until at least
    min_return_decisions bends carry a junction, and rejects the walk if it
    cannot place at least two such decisions (or one, when P_ret has a single
    bend). This yields more cycles and more left/right choices than the
    uniform line-17 sampling.

    straight_bias (deviation from line 8) keeps the outbound walk going
    straight with the given probability when possible, and turn_penalty makes
    the return/chord routes prefer straight runs — together they yield simpler,
    loopier layouts with fewer incidental zig-zags. Both at 0 (and
    bias_return_junction off) reproduce the paper's pseudocode exactly.

    Returns {"out": P_out, "ret": P_ret, "chords": C} or None on failure.
    """
    def l1(u, v):
        return abs(u[0] - v[0]) + abs(u[1] - v[1])
    cells = {(x, y) for x in range(n) for y in range(n)}
    for _ in range(max_attempts):                       # line 1
        s = rng.choice(sorted(cells))                   # line 2
        p_out = [s]
        visited = {s}
        c = s                                           # line 3
        prev_d = None
        while len(p_out) < target_len:                  # line 4
            m = [nb for nb in neighbours(c, n) if nb not in visited]  # line 5
            if not m:                                   # line 6: walk trapped itself
                break
            straight = (c[0] + prev_d[0], c[1] + prev_d[1]) if prev_d else None
            if straight in m and rng.random() < straight_bias:
                nxt = straight                          # biased line 8: keep going straight
            else:
                nxt = rng.choice(m)                     # line 8
            prev_d = (nxt[0] - c[0], nxt[1] - c[1])
            c = nxt
            p_out.append(c)                             # line 9
            visited.add(c)
        e = p_out[-1]                                   # line 11
        allowed = (cells - visited) | {s, e}            # line 12
        p_ret = shortest_path(e, s, allowed, n, turn_penalty)   # line 13
        if p_ret is not None and len(p_out) >= min_len and len(p_ret) >= min_return_len:  # line 14
            ret_bends = set(_route_bends(p_ret))        # junctions here => L/R decisions
            # reject returns too straight to host the requested left/right choices
            if bias_return_junction and len(ret_bends) < min(min_return_decisions, 2):
                continue
            w = visited | set(p_ret)                    # line 15
            chords = []
            tries = 0
            wl = sorted(w)
            want = min(min_return_decisions, len(ret_bends)) if bias_return_junction else 0
            hard_cap = n_chords + 3                      # bound total cycles on small grids

            def covered():                              # return bends that are now junctions
                return ret_bends & set(_ret_junctions(p_out, p_ret, chords))

            # line 16: keep adding chords until we have k of them AND enough
            # left/right decisions on the return route
            while (len(chords) < n_chords or len(covered()) < want) \
                    and len(chords) < hard_cap and tries < chord_tries:
                tries += 1
                uncov = sorted(ret_bends - covered())
                if bias_return_junction and uncov:
                    a = rng.choice(uncov)               # biased line 17: hit a P_ret bend
                    # prefer joining two bends -> one chord, two L/R decisions
                    partners = [c for c in uncov if l1(a, c) >= 4] \
                        or [c for c in wl if l1(a, c) >= 4]
                    if not partners:
                        continue
                    b = rng.choice(partners)
                else:
                    a, b = rng.choice(wl), rng.choice(wl)   # line 17
                    if l1(a, b) < 4:
                        continue
                bset = (cells - w) | {a, b}             # line 18
                g = shortest_path(a, b, bset, n, turn_penalty)  # line 19
                # require at least one unused interior cell, i.e. a genuine
                # corridor "through unused cells" rather than a knocked-out wall
                if g is not None and len(g) >= 3:       # line 20
                    chords.append(g)
                    w |= set(g)
                    wl = sorted(w)
            # reject walks that couldn't host the decisions we asked for
            if bias_return_junction and len(covered()) < min(want, 2):
                continue                                # try a new walk
            return {"out": p_out, "ret": p_ret, "chords": chords}  # line 23
    return None                                         # line 26


# ---------------------------------------------------------------------------
# Graph variant: Algorithm 1 on an arbitrary lattice graph (e.g. the wall-line
# graph of a packed floor plan) instead of the full N x N grid. Nodes are
# integer lattice points, so direction/turn arithmetic works unchanged.
# ---------------------------------------------------------------------------

def shortest_path_on(a, b, allowed, adj, turn_penalty=0.0):
    """shortest_path over an adjacency dict, restricted to `allowed` nodes."""
    if a == b:
        return [a]
    if turn_penalty <= 0.0:
        prev = {a: None}
        q = deque([a])
        while q:
            c = q.popleft()
            for nb in adj.get(c, ()):
                if nb in allowed and nb not in prev:
                    prev[nb] = c
                    if nb == b:
                        path = [b]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    q.append(nb)
        return None
    start = (a, (0, 0))
    dist = {start: 0.0}
    par = {}
    cnt = 0
    pq = [(0.0, cnt, start)]
    while pq:
        d, _, st = heapq.heappop(pq)
        c, dr = st
        if d > dist.get(st, 1e18) + 1e-12:
            continue
        if c == b:
            path = [c]
            while st in par:
                st = par[st]
                path.append(st[0])
            return path[::-1]
        for nb in adj.get(c, ()):
            if nb not in allowed:
                continue
            ndr = (nb[0] - c[0], nb[1] - c[1])
            nd = d + 1.0 + (turn_penalty if dr != (0, 0) and ndr != dr else 0.0)
            key = (nb, ndr)
            if nd < dist.get(key, 1e18) - 1e-12:
                dist[key] = nd
                par[key] = (c, dr)
                cnt += 1
                heapq.heappush(pq, (nd, cnt, key))
    return None


def generate_returnable_path_graph(adj, target_len, min_len, max_attempts,
                                   n_chords, rng, chord_tries=120,
                                   min_return_len=2, max_return_len=None,
                                   bias_return_junction=False,
                                   min_return_decisions=2,
                                   straight_bias=0.0, turn_penalty=0.0):
    """Algorithm 1 with the grid replaced by graph adjacency `adj`
    (node -> sorted neighbour list). Same line-for-line semantics and the same
    documented deviations as generate_returnable_path."""
    def l1(u, v):
        return abs(u[0] - v[0]) + abs(u[1] - v[1])

    nodes = sorted(adj)
    node_set = set(nodes)
    for _ in range(max_attempts):                       # line 1
        s = rng.choice(nodes)                           # line 2
        p_out = [s]
        visited = {s}
        c = s                                           # line 3
        prev_d = None
        while len(p_out) < target_len:                  # line 4
            m = [nb for nb in adj[c] if nb not in visited]      # line 5
            if not m:                                   # line 6
                break
            straight = (c[0] + prev_d[0], c[1] + prev_d[1]) if prev_d else None
            if straight in m and rng.random() < straight_bias:
                nxt = straight                          # biased line 8
            else:
                nxt = rng.choice(m)                     # line 8
            prev_d = (nxt[0] - c[0], nxt[1] - c[1])
            c = nxt
            p_out.append(c)                             # line 9
            visited.add(c)
        e = p_out[-1]                                   # line 11
        allowed = (node_set - visited) | {s, e}         # line 12
        p_ret = shortest_path_on(e, s, allowed, adj, turn_penalty)  # line 13
        if p_ret is not None and len(p_out) >= min_len and len(p_ret) >= min_return_len \
                and (max_return_len is None or len(p_ret) <= max_return_len):
            ret_bends = set(_route_bends(p_ret))
            if bias_return_junction and len(ret_bends) < min(min_return_decisions, 2):
                continue
            w = visited | set(p_ret)                    # line 15
            chords = []
            tries = 0
            wl = sorted(w)
            want = min(min_return_decisions, len(ret_bends)) if bias_return_junction else 0
            hard_cap = n_chords + 3

            def covered():
                return ret_bends & set(_ret_junctions(p_out, p_ret, chords))

            while (len(chords) < n_chords or len(covered()) < want) \
                    and len(chords) < hard_cap and tries < chord_tries:  # line 16
                tries += 1
                uncov = sorted(ret_bends - covered())
                if bias_return_junction and uncov:
                    a = rng.choice(uncov)               # biased line 17
                    partners = [q for q in uncov if l1(a, q) >= 3] \
                        or [q for q in wl if l1(a, q) >= 3]
                    if not partners:
                        continue
                    b = rng.choice(partners)
                else:
                    a, b = rng.choice(wl), rng.choice(wl)   # line 17
                    if l1(a, b) < 3:
                        continue
                bset = (node_set - w) | {a, b}          # line 18
                g = shortest_path_on(a, b, bset, adj, turn_penalty)  # line 19
                if g is not None and len(g) >= 3:       # line 20
                    chords.append(g)
                    w |= set(g)
                    wl = sorted(w)
            if bias_return_junction and len(covered()) < min(want, 2):
                continue
            return {"out": p_out, "ret": p_ret, "chords": chords}  # line 23
    return None                                         # line 26


# ---------------------------------------------------------------------------
# Corridor-graph analysis: junctions and ground-truth turn sequences
# ---------------------------------------------------------------------------

def route_edge_pairs(route):
    return list(zip(route[:-1], route[1:]))


def corridor_graph(path):
    """Adjacency dict cell -> sorted list of connected cells, over all corridors.

    Corridors are the *edges* walked by the routes, not mere cell adjacency:
    two parallel corridors one grid row apart stay separated by a wall.
    """
    adj = {}
    for route in [path["out"], path["ret"], *path["chords"]]:
        for a, b in route_edge_pairs(route):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return {c: sorted(v) for c, v in adj.items()}


def turn_label(din, dout):
    """Egocentric turn for incoming direction din and outgoing dout.

    y grows downward, so cross > 0 is a clockwise map turn = agent's right.
    """
    if din == dout:
        return "straight"
    cross = din[0] * dout[1] - din[1] * dout[0]
    if cross > 0:
        return "right"
    if cross < 0:
        return "left"
    return "u-turn"


def route_decisions(route, adj):
    """Ground-truth action at every junction (corridor degree >= 3) along a route."""
    out = []
    for i in range(1, len(route) - 1):
        c = route[i]
        if len(adj[c]) >= 3:
            din = (c[0] - route[i - 1][0], c[1] - route[i - 1][1])
            dout = (route[i + 1][0] - c[0], route[i + 1][1] - c[1])
            out.append({"cell": list(c), "action": turn_label(din, dout)})
    return out


def route_turns(route):
    """Every bend (direction change) along a route, junction or not."""
    out = []
    for i in range(1, len(route) - 1):
        din = (route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1])
        dout = (route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        lab = turn_label(din, dout)
        if lab != "straight":
            out.append({"cell": list(route[i]), "turn": lab})
    return out


if __name__ == "__main__":
    # facing east (+x): a turn toward +y (south, downward on the map) is a right turn
    assert turn_label((1, 0), (0, 1)) == "right"
    assert turn_label((1, 0), (0, -1)) == "left"
    assert turn_label((0, -1), (1, 0)) == "right"
    assert turn_label((1, 0), (1, 0)) == "straight"
    p = generate_returnable_path(12, 22, 14, 500, 2, random.Random(0), min_return_len=6)
    assert p is not None
    assert set(p["ret"][1:-1]).isdisjoint(set(p["out"][1:-1])), "return reuses outbound corridor"
    print("outbound", len(p["out"]), "return", len(p["ret"]), "chords", [len(c) for c in p["chords"]])
    print("pathgen self-test OK")
