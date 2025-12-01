import heapq
import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from ch import ContractionHierarchy


class InspectableCH(ContractionHierarchy):
    """extends CH to expose heuristic values."""

    def get_stats(self, node, voronoi_map):
        return {
            "Priority": self.compute_priority(node, voronoi_map.get(node, 1)),
            "Rank": self.node_order.get(node, -1),
        }


def setup_graph(place_name="Monaco"):
    print(f"\ndownloading real-world data for {place_name}...")
    try:
        G_raw = ox.graph_from_place(place_name, network_type="drive")
        G_raw = ox.truncate.largest_component(
            G_raw, strongly=True
        )  # no isolated islands
    except:
        G_raw = ox.graph_from_place(place_name, network_type="drive")
        largest = max(nx.strongly_connected_components(G_raw), key=len)
        G_raw = G_raw.subgraph(largest).copy()

    G = nx.DiGraph()
    for u, v, data in G_raw.edges(data=True):
        # Calculate time (seconds)
        speed = float(data.get("speed_kph", 30))
        if isinstance(speed, list):
            speed = speed[0]
        length = float(data.get("length", 100))
        weight = (length / (speed * 1000)) * 3600

        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], weight)
        else:
            G.add_edge(u, v, weight=weight)

    # coordinate data for nearest-node lookup
    for node, data in G_raw.nodes(data=True):
        if node in G:
            G.nodes[node]["x"] = data.get("x")
            G.nodes[node]["y"] = data.get("y")

    print(
        f"        graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )
    return G, G_raw


def get_route(G, G_raw):
    """
    s: near Louis II Stadium (West)
    t: near Monte-Carlo Country Club (East)
    """
    # West: Fontvieille (Stadium area)
    coords_start = (43.7276, 7.4154)
    # East: Saint-Roman / Country Club border
    coords_end = (43.7495, 7.4376)

    # find nearest network nodes to these points
    node_a = ox.distance.nearest_nodes(G_raw, coords_start[1], coords_start[0])
    node_b = ox.distance.nearest_nodes(G_raw, coords_end[1], coords_end[0])

    # check if nodes exist in our simplified DiGraph G
    if node_a not in G or node_b not in G:
        print("[W] nodes unaviable")
        nodes = list(G.nodes())
        return nodes[0], nodes[-1]

    return node_a, node_b


def _ground_truth(G, ch, s, t, scenario, results):
    """verify CH against standard Dijkstra and collect data for plotting."""
    # dijkstra
    t0 = time.time()
    try:
        dijkstra_path = nx.dijkstra_path(G, s, t, weight="weight")
        true_dist = nx.path_weight(G, dijkstra_path, weight="weight")
    except nx.NetworkXNoPath:
        dijkstra_path = []
        true_dist = float("inf")
    t_dijkstra = (time.time() - t0) * 1000

    # CH
    t0 = time.time()
    ch_dist, ch_path = ch.query(s, t)
    t_ch = (time.time() - t0) * 1000

    results[scenario] = {
        "dijkstra_time": t_dijkstra,
        "ch_time": t_ch,
        "dijkstra_path": dijkstra_path,
        "ch_path": ch_path if ch_path else [],
        "dijkstra_dist": true_dist,
        "ch_dist": ch_dist,
    }

    d_str = f"{true_dist:.2f}" if true_dist != float("inf") else "inf"
    c_str = f"{ch_dist:.2f}" if ch_dist else "inf"
    print(f"     dijkstra: {d_str}s ({t_dijkstra:.3f}ms)")
    print(f"     CH: {c_str}s ({t_ch:.3f}ms)")

    if true_dist == float("inf") and ch_dist is None:
        print("     [PASS] both found no path")
    elif abs(true_dist - ch_dist) < 0.01:
        print("     [PASS] exact match")
    else:
        print(f"     [FAIL] difference: {abs(true_dist - ch_dist):.4f}s")


def demo_advanced_dynamics(G, ch, s, t):
    results = {}

    # initial state
    print(f"fixed route: {s} -> {t}")
    _ground_truth(G, ch, s, t, "initial", results)

    path_nodes = results["initial"]["dijkstra_path"]
    if not path_nodes:
        return results

    mid_idx = len(path_nodes) // 2
    u, v = path_nodes[mid_idx], path_nodes[mid_idx + 1]
    original_w = G[u][v]["weight"]

    print(f"\n[a] traffic on edge ({u},{v})")
    new_w_jam = original_w * 50.0
    print(f"   old weight: {original_w:.2f}s -> new weight: {new_w_jam:.2f}s")
    G[u][v]["weight"] = new_w_jam
    ch.handle_dynamic_update(u, v, new_w_jam)
    _ground_truth(G, ch, s, t, "traffic", results)

    print(f"\n[b] restore")
    G[u][v]["weight"] = original_w
    ch.handle_dynamic_update(u, v, original_w)
    _ground_truth(G, ch, s, t, "restored", results)

    return results


def plot_results(G_raw, results):
    # routes on a map
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    scenarios = ["initial", "traffic", "restored"]
    titles = [
        "Initial Route (Fastest)",
        "Route During Traffic Jam",
        "Restored Route (Fastest)",
    ]
    colors = ["green", "red", "blue"]

    for i, scenario in enumerate(scenarios):
        path = results[scenario]["ch_path"]
        dist = results[scenario]["ch_dist"]
        if path:
            ox.plot_graph_route(
                G_raw,
                path,
                ax=ax[i],
                node_size=0,
                edge_linewidth=3,
                route_color=colors[i],
                show=False,
                close=False,
            )
        else:
            # if no path, just plot the graph
            ox.plot_graph(G_raw, ax=ax[i], node_size=0, show=False, close=False)
        ax[i].set_title(f"{titles[i]}\nDistance: {dist:.2f}s")

    plt.suptitle(
        "Contraction Hierarchy Paths in Monaco under Dynamic Conditions", fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("monaco_routes.png")
    print("Saved monaco_routes.png")

    # query times comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    dijkstra_times = [results[s]["dijkstra_time"] for s in scenarios]
    ch_times = [results[s]["ch_time"] for s in scenarios]

    x = range(len(scenarios))
    width = 0.35

    ax2.bar(
        [i - width / 2 for i in x],
        dijkstra_times,
        width,
        label="Dijkstra",
        color="skyblue",
    )
    ax2.bar([i + width / 2 for i in x], ch_times, width, label="CH", color="orange")

    ax2.set_ylabel("Query Time (milliseconds)", fontsize=12)
    ax2.set_title("Query Performance: Dijkstra vs. CH", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.capitalize() for s in scenarios], fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(axis="y", linestyle="--")

    plt.tight_layout()
    fig2.savefig("query_performance.png")
    print("Saved query_performance.png")


def main():
    G, G_raw = setup_graph("Monaco")

    print("building heirarchy")

    ch = InspectableCH(G, weight_func=lambda u, v, d: d.get("weight", 1))
    ch.build()

    # basic path query
    s, t = get_route(G, G_raw)
    print(f"\nquerying {s} -> {t}")
    dist, path = ch.query(s, t)

    # edge case for no path
    if dist is None:
        print("no path")
    else:
        print(f"distance: {dist:.2f}")
        print(f"psth nodes: {len(path)}")

    results = demo_advanced_dynamics(G, ch, s, t)
    plot_results(G_raw, results)


if __name__ == "__main__":
    main()
