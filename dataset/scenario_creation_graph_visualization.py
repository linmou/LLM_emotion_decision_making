import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the build_scenario_creation_graph function
from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
    build_scenario_creation_graph,
)


def visualize_scenario_creation_graph():
    """Visualize the scenario creation graph"""
    # Create the graph
    graph = build_scenario_creation_graph(debug_mode=False)

    # Create a networkx DiGraph for visualization
    G = nx.DiGraph()

    # Add nodes
    nodes = ["propose_scenario", "verify_scenario", "finalize_scenario", "END"]
    node_colors = ["#3498db", "#f39c12", "#2ecc71", "#e74c3c"]
    node_positions = {
        "propose_scenario": (0, 1),
        "verify_scenario": (1, 1),
        "finalize_scenario": (2, 1),
        "END": (3, 1),
    }

    # Add edges
    edges = [
        ("propose_scenario", "verify_scenario"),
        ("verify_scenario", "propose_scenario", "refine"),
        ("verify_scenario", "finalize_scenario", "finalize"),
        ("finalize_scenario", "END"),
    ]

    # Add nodes to the graph
    for node in nodes:
        G.add_node(node)

    # Add edges to the graph
    for edge in edges:
        if len(edge) == 2:
            G.add_edge(edge[0], edge[1])
        else:
            G.add_edge(edge[0], edge[1], label=edge[2])

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Draw the nodes
    nx.draw_networkx_nodes(
        G,
        pos=node_positions,
        nodelist=nodes,
        node_color=node_colors,
        node_size=3000,
        alpha=0.8,
    )

    # Draw the node labels
    nx.draw_networkx_labels(
        G, pos=node_positions, font_size=10, font_weight="bold", font_color="white"
    )

    # Draw the edges
    for edge in edges:
        if len(edge) == 2:
            # Regular edge
            arrow = FancyArrowPatch(
                node_positions[edge[0]],
                node_positions[edge[1]],
                connectionstyle="arc3,rad=0.0",
                arrowstyle="-|>",
                mutation_scale=20,
                lw=2,
                color="gray",
            )
            plt.gca().add_patch(arrow)
        else:
            # Edge with label
            if edge[2] == "refine":
                # Curved edge for the loop back
                arrow = FancyArrowPatch(
                    node_positions[edge[0]],
                    node_positions[edge[1]],
                    connectionstyle="arc3,rad=0.5",
                    arrowstyle="-|>",
                    mutation_scale=20,
                    lw=2,
                    color="blue",
                )
                plt.gca().add_patch(arrow)
                # Add edge label
                midpoint_x = (
                    node_positions[edge[0]][0] + node_positions[edge[1]][0]
                ) / 2
                midpoint_y = (
                    node_positions[edge[0]][1] + node_positions[edge[1]][1]
                ) / 2 + 0.25
                plt.text(midpoint_x, midpoint_y, edge[2], fontsize=10, ha="center")
            else:
                # Regular edge with label
                arrow = FancyArrowPatch(
                    node_positions[edge[0]],
                    node_positions[edge[1]],
                    connectionstyle="arc3,rad=0.0",
                    arrowstyle="-|>",
                    mutation_scale=20,
                    lw=2,
                    color="green",
                )
                plt.gca().add_patch(arrow)
                # Add edge label
                midpoint_x = (
                    node_positions[edge[0]][0] + node_positions[edge[1]][0]
                ) / 2
                midpoint_y = (
                    node_positions[edge[0]][1] + node_positions[edge[1]][1]
                ) / 2 - 0.15
                plt.text(midpoint_x, midpoint_y, edge[2], fontsize=10, ha="center")

    # Add title and remove axes
    plt.title("Scenario Creation Graph", fontsize=14)
    plt.axis("off")

    # Add a legend
    plt.text(
        0,
        0.5,
        "1. Propose Scenario",
        fontsize=10,
        ha="left",
        color="#3498db",
        fontweight="bold",
    )
    plt.text(
        0,
        0.4,
        "2. Verify Scenario",
        fontsize=10,
        ha="left",
        color="#f39c12",
        fontweight="bold",
    )
    plt.text(
        0,
        0.3,
        "3. Finalize Scenario",
        fontsize=10,
        ha="left",
        color="#2ecc71",
        fontweight="bold",
    )
    plt.text(
        0,
        0.2,
        "4. End Process",
        fontsize=10,
        ha="left",
        color="#e74c3c",
        fontweight="bold",
    )

    # Add a diagram caption
    plt.text(
        1.5,
        0,
        "Scenario Creation Flow: The process starts with scenario proposal, followed by verification.\n"
        "If the scenario needs improvement, it's sent back for refinement.\n"
        "Otherwise, it's finalized and the process ends.",
        fontsize=10,
        ha="center",
        va="center",
    )

    # Save the figure
    plt.savefig("doc/scenario_creation_graph.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Graph visualization saved to doc/scenario_creation_graph.png")


if __name__ == "__main__":
    visualize_scenario_creation_graph()
