import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import ArrowStyle


def draw_bipartite(
        weights=None,
        weight_max=None,
):
    # if weights is None:
    #     weights = np.array([[1, 19], [10, 10]])

    n_node = len(weights)
    if weight_max is None:
        weight_max = np.mean(np.sum(weights, 1)) * 2
    # weight_max = 20
    node_sizes = np.concatenate([np.sum(weights, 1), np.sum(weights, 0)])

    G = nx.DiGraph()
    for i in range(n_node):
        G.add_node(i, layer=0, count=node_sizes[i])
        G.add_node(i + n_node, layer=1, count=node_sizes[i + n_node])

    for src in range(n_node):
        for dst in range(n_node):
            G.add_weighted_edges_from([(src, dst + n_node, weights[src, dst])])

    c_max = 255
    cmap = plt.get_cmap('coolwarm', c_max)

    pos = nx.multipartite_layout(G, subset_key="layer")
    # plt.figure(figsize=(3, 3))

    nx.draw_networkx_nodes(
        G, pos, nodelist=np.arange(n_node ** 2),
        node_size=300 * node_sizes / weight_max,
        node_color=cmap(node_sizes / weight_max),
        cmap=cmap,
        alpha=0.5
    )
    wmax = np.amax(weights)
    for edge, v in G.edges.items():
        w = v['weight']
        nx.draw_networkx_edges(
            G, pos, edgelist=[edge],
            edge_color=cmap(w / weight_max * n_node),
            width=0,
            arrowstyle=ArrowStyle.Simple(
                head_length=w / weight_max * 2.1,
                head_width=w / weight_max * 1.4,
                tail_width=w / weight_max * 0.5,
            ),
            min_source_margin=wmax / 2 + weight_max * 0.2,
            min_target_margin=wmax / 2 + weight_max * 0.2,
        )
    edge_weights = {k: '%g' % v['weight'] for k, v in G.edges.items()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_weights,
        label_pos = 0.33
    )
    node_labels = {k: '%g' % v['count'] for k, v in G.nodes.items()}
    nx.draw_networkx_labels(
        G, pos, labels=node_labels
    )

    plt.axis("equal")
    plt.box(False)
    # plt.show()
    # print('--')

    return G, pos


if __name__ == '__main__':
    draw_bipartite()
