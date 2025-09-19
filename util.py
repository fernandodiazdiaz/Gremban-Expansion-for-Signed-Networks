import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


##########################################
### NETWORK CREATION AND VISUALIZATION ###
##########################################


def generate_signed_dcsbm(N, group_probs, theta, prop_plus, prop_minus):
    """
    Generates a signed degree‑corrected stochastic block model (DC‑SBM)
    with a single node‑activity parameter.

    Parameters
    ----------
    N : int
        Number of nodes.
    group_probs : array-like or int
        If array-like (summing to 1), defines the probability distribution
        over q groups. If int, nodes are assigned uniformly into that many groups.
    theta : array-like of length N
        Node-level degree parameters.
    prop_plus : 2D numpy array, shape (q, q)
        Block propensity matrix for positive edges.
    prop_minus : 2D numpy array, shape (q, q)
        Block propensity matrix for negative edges.

    Returns
    -------
    G : networkx.Graph
        A signed undirected graph. Each edge has attribute 'weight' = +1 or -1.
    """
    # Assign groups
    if isinstance(group_probs, (list, np.ndarray)):
        group_probs = np.array(group_probs) / np.sum(group_probs)
        q = len(group_probs)
        groups = np.random.choice(q, size=N, p=group_probs)
    elif isinstance(group_probs, int):
        q = group_probs
        groups = np.random.randint(q, size=N)
    else:
        raise ValueError("group_probs must be array-like or int")

    # Precompute block total propensities
    prop_total = prop_plus + prop_minus

    G = nx.Graph()
    for i in range(N):
        G.add_node(i, group=int(groups[i]), theta=float(theta[i]))

    for i in range(N):
        for j in range(i + 1, N):
            gi, gj = groups[i], groups[j]
            lambda_tot = theta[i] * theta[j] * prop_total[gi, gj]
            p_edge = 1 - np.exp(-lambda_tot) if lambda_tot > 0 else 0

            if np.random.rand() < p_edge:
                # Probability positive = prop_plus / (prop_plus + prop_minus)
                block_plus = prop_plus[gi, gj]
                block_total = prop_total[gi, gj]
                p_pos = block_plus / block_total if block_total > 0 else 0
                weight = 1 if np.random.rand() < p_pos else -1
                G.add_edge(i, j, weight=weight)

    return G





def draw_network(
    obj,
    ax=None,
    pos=None,
    labels=None,
    node_size=500,
    node_color="white",
    nodeedge_color="k",
    cmap_nodes=None,
    label_fontsize=12,
    pos_edge_width=2,
    neg_edge_width=2,
    pos_edgecolor="black",
    neg_edgecolor="red",
    pos_ls="-",
    neg_ls="--",
    with_labels=False,
    spines=False,
    differenciate_groups=False,
    group_assignments=None,
    group_markers=None,
    group_colors=None,
):
    """
    Draw a networkx graph with positive and negative edges.

    Parameters:
    -----------
    obj : networkx graph or numpy array
        The network or its adjacency matrix.
    ax : matplotlib.axes.Axes, optional
        Axis to draw the graph on.
    pos : dict, optional
        Dictionary of node positions. If None, a Kamada-Kawai layout is used.
    labels : dict, optional
        Node labels.
    node_size : int, default 500
        Size of the nodes.
    node_color : str, default "white"
        Default color of the nodes.
    nodeedge_color : str, default "k"
        Color of the node borders.
    cmap_nodes : matplotlib colormap, optional
        Colormap for the nodes.
    label_fontsize : int, default 12
        Font size for node labels.
    pos_edge_width : int, default 2
        Width for positive edges.
    neg_edge_width : int, default 2
        Width for negative edges.
    pos_ls : str, default "-"
        Linestyle for positive edges.
    neg_ls : str, default "--"
        Linestyle for negative edges.
    with_labels : bool, default False
        If True, draw node labels.
    spines : bool, default False
        If False, the axes spines and ticks are hidden.
    differenciate_groups : bool, default False
        If True, nodes will be drawn by group (using different markers and colors).
    group_assignments : dict, optional
        Dictionary mapping nodes to group labels. Required if differenciate_groups is True.
    group_markers : dict, optional
        Dictionary mapping group labels to marker styles. If not provided, default markers are used.
    group_colors : dict, optional
        Dictionary mapping group labels to colors. If not provided, default colors are used.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axis with the drawn graph.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Convert input to a networkx graph if needed.
    if isinstance(obj, nx.Graph):
        G = obj
    elif isinstance(obj, np.ndarray):
        G = nx.from_numpy_array(obj)
    else:
        raise ValueError("Input must be a networkx graph or a numpy array")

    # Compute positions if not provided.
    if pos is None:
        G_abs = nx.Graph()
        for u, v, w in G.edges(data="weight"):
            G_abs.add_edge(u, v)
        pos = nx.kamada_kawai_layout(G_abs)

    # Ensure all edges have a weight; if not, assign a default weight 1.
    for u, v, w in G.edges(data="weight"):
        if w is None:
            G[u][v]["weight"] = 1

    # Set edge attributes: color, linestyle, and width.
    edge_color = []
    ls = []
    width = []
    for u, v, w in G.edges(data=True):
        if w["weight"] > 0:
            edge_color.append(pos_edgecolor)
            ls.append(pos_ls)
            width.append(pos_edge_width)
        else:
            edge_color.append(neg_edgecolor)
            ls.append(neg_ls)
            width.append(neg_edge_width)

    # Draw nodes: either by groups or with default style.
    if differenciate_groups:
        if group_assignments is None:
            raise ValueError(
                "group_assignments must be provided when differenciate_groups is True"
            )
        # Organize nodes by group.
        groups = {}
        for node, group in group_assignments.items():
            groups.setdefault(group, []).append(node)
        # Set default colors if not provided.
        if group_colors is None:
            default_colors = plt.cm.tab10.colors
            group_colors = {
                grp: default_colors[i % len(default_colors)]
                for i, grp in enumerate(sorted(groups.keys()))
            }
        # Set default markers if not provided.
        if group_markers is None:
            default_markers = ["o", "s", "^", "D", "v", "p", "*", "X", "8"]
            group_markers = {
                grp: default_markers[i % len(default_markers)]
                for i, grp in enumerate(sorted(groups.keys()))
            }
        # Draw nodes for each group.
        for grp, nodes in groups.items():
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                ax=ax,
                nodelist=nodes,
                node_size=node_size,
                node_color=[group_colors[grp]],
                node_shape=group_markers[grp],
                edgecolors=nodeedge_color,
            )
    else:
        # Default node drawing.
        if cmap_nodes is not None:
            nodes = nx.draw_networkx_nodes(
                G,
                pos=pos,
                ax=ax,
                node_size=node_size,
                node_color=node_color,
                cmap=cmap_nodes,
            )
        else:
            nodes = nx.draw_networkx_nodes(
                G, pos=pos, ax=ax, node_size=node_size, node_color=node_color
            )
        nodes.set_edgecolor(nodeedge_color)

    # Draw edges.
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color=edge_color, style=ls, width=width
    )
    if with_labels:
        nx.draw_networkx_labels(
            G, pos=pos, ax=ax, labels=labels, font_size=label_fontsize
        )
    if not spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    return ax


def pos_gremban_expansion(pos_original, offset=None):
    """
    Computes the position dictionary for the Gremban expansion.
    Input:
        pos_original: dictionary of node positions
    Output:
        pos_expanded: dictionary of node positions for the expanded graph
    """
    # If no offset is provided, calculate a suitable offset
    if offset is None:
        y_coords = [y for _, y in pos_original.values()]
        y_min, y_max = min(y_coords), max(y_coords)
        offset = max(abs(y_max - y_min) * 0.2, 3)

    # Create the expanded position dictionary
    pos_expanded = {}
    for key, (x, y) in pos_original.items():
        pos_expanded[f"{key}⁺"] = (x, y)
        pos_expanded[f"{key}⁻"] = (x, -y - offset)
    return pos_expanded



###############################################
### MATRIX EXTRACTION & SPECTRAL CLUSTERING ###
###############################################


def adjacency_matrix(obj, nodelist=None):
    """
    Converts a NetworkX graph to an adjacency matrix if needed.
    Input:
        obj: NetworkX graph or NumPy array
    Output:
        A: adjacency matrix (NumPy array)
    """
    if isinstance(obj, nx.Graph):
        return nx.to_numpy_array(obj, nodelist=nodelist, weight="weight")
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        raise ValueError("Input must be a NetworkX graph or a NumPy array")




def gremban_expansion(obj):
    """
    Computes the Gremban expansion of the signed adjacency matrix.
    Input:
    ----------
    obj : networkx.Graph or np.ndarray
        A signed graph or its adjacency matrix. If a graph, nodes may have arbitrary labels.
    Output:
    -------
    If input is an np.ndarray, returns the expanded 2n x 2n matrix.
    If input is a networkx.Graph, returns the expanded graph with relabeled nodes (original⁺, original⁻).
    """
    if isinstance(obj, np.ndarray):
        A = obj
        Ap = np.maximum(A, 0)
        An = np.maximum(-A, 0)
        return np.block([[Ap, An], [An, Ap]])

    elif isinstance(obj, nx.Graph):
        nodelist = list(obj.nodes)
        A = adjacency_matrix(obj, nodelist=nodelist)
        Ap = np.maximum(A, 0)
        An = np.maximum(-A, 0)
        A_exp = np.block([[Ap, An], [An, Ap]])

        # Create labels: for each node 'x', create 'x⁺' and 'x⁻'
        labels_plus = [f"{node}⁺" for node in nodelist]
        labels_minus = [f"{node}⁻" for node in nodelist]
        full_labels = labels_plus + labels_minus

        G_exp = nx.from_numpy_array(A_exp)
        mapping = dict(zip(range(2 * len(nodelist)), full_labels))
        G_exp = nx.relabel_nodes(G_exp, mapping)
        return G_exp

    else:
        raise ValueError("Input must be a NumPy array or NetworkX graph")




def compute_laplacian(G, kind="unsigned", normalized=True):
    """
    Compute one of three Laplacians for a signed graph G, with optional normalization.

    Parameters
    ----------
    G : networkx.Graph
        Input graph, with signed edges (weight = ±1).
    kind : str
        One of:
        - "unsigned": combinatorial Laplacian ignoring signs, L = D - |A|.
        - "signed"  : signed/opposing Laplacian,   L = D -  A.
        - "gremban" : Laplacian of the 2n×2n Gremban expansion, L = D_exp - A_exp.
    normalized : bool, default=True
        If True, return the symmetric normalized Laplacian
        \(L_{\rm sym} = D^{-1/2} L D^{-1/2}\).
        If False, return the raw combinatorial Laplacian.

    Returns
    -------
    L_out : np.ndarray
        The requested (possibly normalized) Laplacian matrix.
    """
    # Build the raw combinatorial Laplacian
    A = adjacency_matrix(G).astype(float)
    K = np.diag(np.sum(np.abs(A), axis=0))

    if kind == "unsigned":
        L = K - np.abs(A)
    elif kind == "signed":
        L = K - A
    elif kind == "gremban":
        A = gremban_expansion(A)
        K = np.diag(np.sum(np.abs(A), axis=0))
        L = K - np.abs(A)
    else:
        raise ValueError("kind must be one of 'unsigned', 'signed', or 'gremban'")

    # Optionally normalize symmetrically: L_sym = D^{-1/2} L D^{-1/2}
    if normalized:
        deg = np.diag(K)
        # avoid division by zero
        with np.errstate(divide='ignore'):
            inv_sqrt = 1.0 / np.sqrt(deg)
        inv_sqrt[np.isinf(inv_sqrt)] = 0.0
        K_inv_sqrt = np.diag(inv_sqrt)
        L_out = K_inv_sqrt @ L @ K_inv_sqrt
        return L_out
    else:
        return L


