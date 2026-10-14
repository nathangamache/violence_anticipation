import matplotlib.pyplot as plt
import numpy as np


def visualize_graph(graph):
    nodes = graph["nodes"][:, :2]
    edges = graph["edges"]

    plt.figure(figsize=(6, 6))
    plt.scatter(nodes[:, 0], nodes[:, 1], c="blue")

    for (i, j, _) in edges:
        plt.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            color="gray",
            linewidth=0.5,
        )

    plt.title("Graph Visualization")
    plt.gca().invert_yaxis()
    plt.show()
