import heapq
import math
import time

import networkx as nx


class ContractionHierarchy:
    def __init__(
        self,
        graph,
        weight_func=lambda u, v, d: d.get("weight", 1),
        alpha_ed=190,
        alpha_dn=120,
        alpha_vor=70,
        alpha_cc=10,
        witness_hop_limit=5,
        witness_search_limit=500,
    ):
        self.graph = self._simplify_graph(graph, weight_func)

        # heuristic weights
        self.alpha_ed = alpha_ed
        self.alpha_dn = alpha_dn
        self.alpha_vor = alpha_vor
        self.alpha_cc = alpha_cc

        # limits
        self.witness_hop_limit = witness_hop_limit
        self.witness_search_limit = witness_search_limit

        self.node_order = {}  # rank pi(v)
        self.rank_to_node = {}  # inverse pi
        self.contracted_nodes = set()
        self.shortcuts = {}  # Key: (u, v), Value: {weight, via}
        self.edge_to_shortcuts = {}  # Dependency tracking for updates

        # upwaed/downward graph separation
        self.upward_graph = nx.DiGraph()
        self.downward_graph = nx.DiGraph()

    def _simplify_graph(self, graph, weight_func):
        """converts to a simple DiGraph, keeping minimum weights"""
        simple = nx.DiGraph()
        simple.add_nodes_from(graph.nodes())
        for u, v, data in graph.edges(data=True):
            w = weight_func(u, v, data)
            if simple.has_edge(u, v):
                if w < simple[u][v]["weight"]:
                    simple[u][v]["weight"] = w
            else:
                simple.add_edge(u, v, weight=w)
        return simple

    def _compute_voronoi_regions(self):
        """approximate voronoi region sizes for heuristics."""
        region_counts = {
            n: 0 for n in self.graph.nodes() if n not in self.contracted_nodes
        }
        pq = []
        visited = {}  # node -> (dist, root)

        for n in region_counts:
            heapq.heappush(pq, (0, n, n))
            visited[n] = (0, n)

        while pq:
            d, root, u = heapq.heappop(pq)
            if d > visited[u][0]:
                continue

            if root in region_counts:
                region_counts[root] += 1

            for v in self.graph.neighbors(u):
                if v in self.contracted_nodes:
                    continue
                new_dist = d + self.graph[u][v]["weight"]
                if v not in visited or new_dist < visited[v][0]:
                    visited[v] = (new_dist, root)
                    heapq.heappush(pq, (new_dist, root, v))
        return region_counts

    def witness_search(self, u, v, via_node, max_weight):
        """limit search to check if a path <= max_weight exists without via_node"""
        if self.graph.has_edge(u, v) and self.graph[u][v]["weight"] <= max_weight:
            return True, 0

        pq = [(0, 0, u)]
        dists = {u: 0}
        settled = 0

        while pq and settled < self.witness_search_limit:
            d, hops, curr = heapq.heappop(pq)
            settled += 1

            if d > max_weight:
                break
            if curr == v:
                return True, settled
            if hops >= self.witness_hop_limit:
                continue

            for neighbor in self.graph.neighbors(curr):
                if neighbor == via_node or neighbor in self.contracted_nodes:
                    continue

                new_dist = d + self.graph[curr][neighbor]["weight"]
                if new_dist < dists.get(neighbor, float("inf")):
                    dists[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, hops + 1, neighbor))

        return False, settled

    def compute_priority(self, node, voronoi_size):
        """calculate contraction priority (lower = contract earlier)"""
        shortcuts = 0
        contraction_cost = 0

        predecessors = [
            p for p in self.graph.predecessors(node) if p not in self.contracted_nodes
        ]
        successors = [
            s for s in self.graph.neighbors(node) if s not in self.contracted_nodes
        ]

        edges_removed = len(predecessors) + len(successors)

        for u in predecessors:
            w_uv = self.graph[u][node]["weight"]
            for v in successors:
                if u == v:
                    continue
                path_cost = w_uv + self.graph[node][v]["weight"]
                has_witness, settled = self.witness_search(u, v, node, path_cost)
                contraction_cost += settled
                if not has_witness:
                    shortcuts += 1

        edge_diff = shortcuts - edges_removed
        deleted_neighbors = sum(
            1 for n in self.graph.successors(node) if n in self.contracted_nodes
        )
        deleted_neighbors += sum(
            1 for n in self.graph.predecessors(node) if n in self.contracted_nodes
        )

        return (
            self.alpha_ed * edge_diff
            + self.alpha_dn * deleted_neighbors
            + self.alpha_vor * math.sqrt(voronoi_size)
            + self.alpha_cc * contraction_cost
        )

    def contract_node(self, node, recontracting=False):
        """add shortcuts and marks node as contracted
        when recontracting=True, uses rank to determine neighbors instead of contracted_set"""
        if recontracting:
            # valid neighbors are higher in rank
            my_rank = self.node_order[node]
            predecessors = [
                p
                for p in self.graph.predecessors(node)
                if self.node_order.get(p, -1) > my_rank
            ]
            successors = [
                s
                for s in self.graph.neighbors(node)
                if self.node_order.get(s, -1) > my_rank
            ]
        else:
            # valid neighbors not yet contracted
            predecessors = [
                p
                for p in self.graph.predecessors(node)
                if p not in self.contracted_nodes
            ]
            successors = [
                s for s in self.graph.neighbors(node) if s not in self.contracted_nodes
            ]

        for u in predecessors:
            w_uv = self.graph[u][node]["weight"]
            for v in successors:
                if u == v:
                    continue
                path_cost = w_uv + self.graph[node][v]["weight"]

                # if path is cheaper than existing, it needs a shortcut
                has_witness, _ = self.witness_search(u, v, node, path_cost)

                if not has_witness:
                    self.shortcuts[(u, v)] = {"weight": path_cost, "via": node}

                    # update graph for future contractions
                    if self.graph.has_edge(u, v):
                        self.graph[u][v]["weight"] = min(
                            self.graph[u][v]["weight"], path_cost
                        )
                    else:
                        self.graph.add_edge(u, v, weight=path_cost)

                    # dependency tracking
                    self._register_shortcut_dependency(u, v, node)

        if not recontracting:
            self.contracted_nodes.add(node)
            rank = len(self.contracted_nodes)
            self.node_order[node] = rank
            self.rank_to_node[rank] = node

    def _register_shortcut_dependency(self, u, v, via):
        """map shortcut (u,v) to the edges (u,via) and (via,v) for updates"""
        if (u, v) not in self.edge_to_shortcuts:
            self.edge_to_shortcuts[(u, v)] = set()
        if (u, via) not in self.edge_to_shortcuts:
            self.edge_to_shortcuts[(u, via)] = set()
        if (via, v) not in self.edge_to_shortcuts:
            self.edge_to_shortcuts[(via, v)] = set()

        self.edge_to_shortcuts[(u, via)].add((u, v))
        self.edge_to_shortcuts[(via, v)].add((u, v))

    def build(self):
        """lazy update build process"""
        start_time = time.time()
        print("initializing voronoi regions and priorities")
        voronoi_map = self._compute_voronoi_regions()

        pq = []
        current_priorities = {}

        for node in self.graph.nodes():
            p = self.compute_priority(node, voronoi_map.get(node, 1))
            current_priorities[node] = p
            heapq.heappush(pq, (p, node))

        print("contracting nodes (lazy updates)")
        while pq:
            p, node = heapq.heappop(pq)

            # check if already contracted
            if node in self.contracted_nodes:
                continue

            # lazy check: is the priority stale?
            if p > current_priorities[node]:
                continue  # discard

            # in a highly optimized version, we might check neighbors status first
            actual_p = self.compute_priority(node, voronoi_map.get(node, 1))

            if actual_p > p:
                # priority increased (worse candidate)
                current_priorities[node] = actual_p
                heapq.heappush(pq, (actual_p, node))
                continue

            self.contract_node(node)

        self._finalize_search_graphs()
        print(
            f"built CH with {len(self.shortcuts)} shortcuts in {time.time() - start_time:.4f}s"
        )

    def _finalize_search_graphs(self):
        """filter upward and downward graphs for fast querying."""
        self.upward_graph = nx.DiGraph()
        self.downward_graph = nx.DiGraph()

        # using the modified graph
        for u, v, data in self.graph.edges(data=True):
            rank_u = self.node_order.get(u, -1)
            rank_v = self.node_order.get(v, -1)

            # upward: source rank < target rank
            if rank_u < rank_v:
                self.upward_graph.add_edge(u, v, weight=data["weight"])

            # downward: source rank > target rank
            if rank_u > rank_v:
                self.downward_graph.add_edge(v, u, weight=data["weight"])

    def _unpack_path(self, u, v):
        if (u, v) in self.shortcuts:
            via = self.shortcuts[(u, v)]["via"]
            return self._unpack_path(u, via)[:-1] + self._unpack_path(via, v)
        else:
            return [u, v]

    def query(self, source, target):
        if source == target:
            return 0, [source]

        # forward search
        dist_fwd = {source: 0}
        parent_fwd = {source: None}
        pq_fwd = [(0, source)]

        # backward search
        dist_bwd = {target: 0}
        parent_bwd = {target: None}
        pq_bwd = [(0, target)]

        min_dist = float("inf")
        meet_node = None

        while pq_fwd or pq_bwd:
            # forward step
            if pq_fwd:
                d, u = heapq.heappop(pq_fwd)
                if d <= min_dist:
                    if u in dist_bwd:
                        total = d + dist_bwd[u]
                        if total < min_dist:
                            min_dist = total
                            meet_node = u

                    if u in self.upward_graph:
                        for v, data in self.upward_graph[u].items():
                            w = data["weight"]
                            new_d = d + w
                            if new_d < dist_fwd.get(v, float("inf")):
                                dist_fwd[v] = new_d
                                parent_fwd[v] = u
                                heapq.heappush(pq_fwd, (new_d, v))

            # backward step
            if pq_bwd:
                d, v = heapq.heappop(pq_bwd)
                if d <= min_dist:
                    if v in dist_fwd:
                        total = d + dist_fwd[v]
                        if total < min_dist:
                            min_dist = total
                            meet_node = v

                    if v in self.downward_graph:
                        for u, data in self.downward_graph[v].items():
                            w = data["weight"]
                            new_d = d + w
                            if new_d < dist_bwd.get(u, float("inf")):
                                dist_bwd[u] = new_d
                                parent_bwd[u] = v
                                heapq.heappush(pq_bwd, (new_d, u))

        if meet_node is None:
            return None, []

        # path reconstruction
        path_up = []
        curr = meet_node
        while curr is not None:
            path_up.append(curr)
            curr = parent_fwd.get(curr)
        path_up.reverse()

        # trace back from meet_node to target
        path_down = []
        curr = parent_bwd.get(meet_node)
        while curr is not None:
            path_down.append(curr)
            curr = parent_bwd.get(curr)

        # unpack shortcuts
        full_path = []
        raw_path = path_up + path_down

        for i in range(len(raw_path) - 1):
            segment = self._unpack_path(raw_path[i], raw_path[i + 1])
            if full_path:
                full_path.extend(segment[1:])  # svoid duplicating nodes
            else:
                full_path.extend(segment)

        return min_dist, full_path

    def handle_dynamic_update(self, u, v, new_weight):
        print(f"dynamic update: ({u}, {v}) -> {new_weight}")

        # update vase and upward/downward graphs (if base edge exists)
        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] = new_weight
            rank_u, rank_v = self.node_order.get(u, -1), self.node_order.get(v, -1)

            # update upward/downward graphs if the edge exists there
            if rank_u < rank_v and self.upward_graph.has_edge(u, v):
                self.upward_graph[u][v]["weight"] = new_weight
            elif rank_u > rank_v and self.downward_graph.has_edge(v, u):
                self.downward_graph[v][u]["weight"] = new_weight
        else:
            return  # edge does not exist in CH

        # identify dependencies and via nodes
        affected_shortcuts = set()
        nodes_to_recontract = set([u, v])

        queue = [(u, v)]

        while queue:
            curr_u, curr_v = queue.pop(0)

            # look up shortcuts that depend on (curr_u, curr_v)
            if (curr_u, curr_v) in self.edge_to_shortcuts:
                for sc in self.edge_to_shortcuts[(curr_u, curr_v)]:
                    if sc not in affected_shortcuts:
                        affected_shortcuts.add(sc)
                        queue.append(sc)

                        # re-contract the via node that created this shortcut
                        if sc in self.shortcuts:
                            via_node = self.shortcuts[sc]["via"]
                            nodes_to_recontract.add(via_node)

        # remove affected shortcuts
        for sc_u, sc_v in affected_shortcuts:
            if (sc_u, sc_v) in self.shortcuts:
                del self.shortcuts[(sc_u, sc_v)]

                # remove from search graphs
                if self.upward_graph.has_edge(sc_u, sc_v):
                    self.upward_graph.remove_edge(sc_u, sc_v)
                if self.downward_graph.has_edge(sc_v, sc_u):
                    self.downward_graph.remove_edge(sc_v, sc_u)

                # remove from overlay graph
                if self.graph.has_edge(sc_u, sc_v):
                    self.graph.remove_edge(sc_u, sc_v)

        # re-contract nodes in rank order
        # sorting by rank ensures we rebuild the hierarchy bottom-up
        valid_nodes = [n for n in nodes_to_recontract if n in self.node_order]
        sorted_nodes = sorted(valid_nodes, key=lambda n: self.node_order[n])

        for node in sorted_nodes:
            self.contract_node(node, recontracting=True)

        # sync shortcuts to search graphs
        for (sc_u, sc_v), data in self.shortcuts.items():
            rank_u, rank_v = self.node_order[sc_u], self.node_order[sc_v]
            w = data["weight"]

            if rank_u < rank_v:
                if not self.upward_graph.has_edge(sc_u, sc_v):
                    self.upward_graph.add_edge(sc_u, sc_v, weight=w)
                else:
                    self.upward_graph[sc_u][sc_v]["weight"] = min(
                        w, self.upward_graph[sc_u][sc_v]["weight"]
                    )
            elif rank_u > rank_v:
                if not self.downward_graph.has_edge(sc_v, sc_u):
                    self.downward_graph.add_edge(sc_v, sc_u, weight=w)
                else:
                    self.downward_graph[sc_v][sc_u]["weight"] = min(
                        w, self.downward_graph[sc_v][sc_u]["weight"]
                    )
