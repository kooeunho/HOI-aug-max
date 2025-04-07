import numpy as np
import networkx as nx
from scipy.optimize import linprog
import copy
import math
import random
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SGConv, GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, is_undirected, subgraph, to_networkx
import torch_geometric.transforms as T



def generating_graph_with_simplices(n_units, prob_mat2, seed):
    # n_units: list indicating the number of nodes in each class
    # prob_mat2: connection probability matrix (B_2)

    random.seed(seed)
    np.random.seed(seed)
    
    # Create a list of node indices
    V = [i for i in range(sum(n_units))]
    
    def connection_function(x, y, classes, prob_mat):
        # Determine the class of each node
        for i in range(len(classes)):
            if x in classes[i]:
                class_x = i
            if y in classes[i]:
                class_y = i
                
        # Determine whether to connect x and y based on inter-class probability
        return 1 if random.random() < prob_mat[class_x][class_y] else 0
    
    n_V = len(V)
    possible_edges = list(combinations(V, 2))  # All possible undirected edges

    # Split nodes into class-wise sets
    interval = np.cumsum(np.insert(n_units, 0, 0))
    classes = [set(np.arange(n_V)[interval[i]:interval[i+1]]) for i in range(len(n_units))]
    
    # Determine actual edges based on probabilities
    connect = [connection_function(possible_edges[i][0], possible_edges[i][1], classes, prob_mat2)
               for i in range(len(possible_edges))]
    
    real_edges = [possible_edges[i] for i in range(len(possible_edges)) if connect[i] == 1]
    
    # Create graph and add nodes/edges
    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_edges_from(real_edges)

    # Ensure the graph is connected
    if not nx.is_connected(G):
        connected_components = list(nx.connected_components(G))
        for component in connected_components[1:]:
            random_class = random.choice(list(classes))
            connect_node = np.random.choice(list(random_class))
            G.add_edge(connect_node, list(component)[0])
    
    # Assign class labels to nodes
    label = {}
    for idx, class_set in enumerate(classes):
        for node in class_set:
            label[node] = idx
    nx.set_node_attributes(G, label, 'label')
    
    # Extract simplices (cliques) of all sizes
    all_cliques = list(nx.enumerate_all_cliques(G))
    simplices = [[] for _ in range(len(all_cliques[-1]))]
    for clique in all_cliques:
        n = len(clique)
        simplices[n-1].append(clique)
    
    return G, simplices, classes, list(label.values())


def adjacency_matrix(G):
    # Return the adjacency matrix of the graph G as a NumPy array
    return nx.adjacency_matrix(G).toarray()


def linear_programming(A, method='highs'):
    """
    Solves a linear programming problem based on matrix A.

    Parameters:
    A : numpy.ndarray
        An F x F submatrix of the Laplacian matrix L.
    method : str, optional
        Solver method for linprog (default: 'highs').

    Returns:
    numpy.ndarray
        The solution vector X normalized by the optimal a.

    Raises:
    ValueError
        If the problem is infeasible or no solution is found.
    """

    n = A.shape[0]

    # Objective: minimize the last variable 'a'
    c = np.zeros(n + 1)
    c[-1] = 1

    # Inequality constraint: A * x <= a
    A_ub = np.hstack((A, -np.ones((n, 1))))
    b_ub = np.zeros(n)

    # Equality constraint: sum(x) = 1
    A_eq = np.ones((1, n + 1))
    A_eq[0, -1] = 0
    b_eq = np.array([1])

    # Variable bounds: 0 <= x_i <= 1, a >= 0
    bounds = [(0, 1)] * n + [(0, None)]

    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)

    if result.success:
        a_min = result.x[-1]
        X = result.x[:-1] / (a_min + 1e-10)  # Normalize by a
        return X
    else:
        raise ValueError("Linear programming failed to find a solution.")


def equilibrium_measure(F, L):
    """
    Computes the equilibrium measure supported on the node set F.

    Parameters:
    F : list
        Subset of nodes (indices) where the equilibrium measure is supported.
    L : numpy.ndarray
        Graph Laplacian matrix of shape (n, n).

    Returns:
    numpy.ndarray
        A vector of length n with non-zero values only on indices in F.
    """
    A = L[F][:, F]  # Extract the submatrix for nodes in F
    EM_tem = linear_programming(A)  # Solve for the equilibrium measure
    result = np.zeros(len(L))
    result[F] = EM_tem  # Assign non-zero values to F
    return result


def add_node(F, node):
    """
    Adds a node to a sorted list F while preserving the order.

    Parameters:
    F : list
        A sorted list of integers.
    node : int
        The node to insert.

    Returns:
    list
        A new sorted list including the added node.
    """
    position = np.searchsorted(np.array(F), node)
    return list(np.insert(np.array(F), position, node))


def encode_integer(i, classes):
    """
    Encodes a node index and its class into a one-hot vector.

    Parameters:
    i : int
        The node index.
    classes : list of lists
        Each sublist contains node indices for a specific class.

    Returns:
    numpy.ndarray
        A vector where the first element is the node index, and the rest is one-hot encoded label.
    """
    encoded = np.full(len(classes) + 1, 0, dtype=float)
    encoded[0] = i  # Store the node index

    for index, class_list in enumerate(classes):
        if i in class_list:
            encoded[index + 1] = 1  # One-hot for class
            break

    return encoded



def known_sets(classes, simplices, BP, seed):
    """
    Randomly selects a subset of nodes from each class as known (labeled) nodes.

    Parameters:
    classes : list of lists
        List of node indices per class.
    simplices : unused
        Placeholder for compatibility (not used here).
    BP : float
        The ratio of nodes to select per class.
    seed : int
        Random seed for reproducibility.

    Returns:
    known : list of lists
        Selected known nodes for each class.
    """
    np.random.seed(seed)
    
    n_L = len(classes)
    known = []

    for i in range(n_L):
        class_nodes = classes[i]
        n_class = len(class_nodes)
        n_select = max(1, math.ceil(n_class * BP))  # Ensure at least one node is selected

        if n_class <= n_select:
            selected = class_nodes[:]  # Select all if too few
        else:
            selected = np.random.choice(class_nodes, n_select, replace=False).tolist()

        known.append(selected)

    return known


def initialization(G, classes, simplices, BP, seed, epsilon=1e-10):
    """
    Initializes the label probability matrix using equilibrium measures.

    Parameters:
    G : networkx.Graph
        Input graph.
    classes : list of lists
        Node indices grouped by class.
    simplices : list
        List of simplices (cliques) — not used here but passed for consistency.
    BP : float
        Boundary percentage used to select labeled nodes.
    seed : int
        Random seed.
    epsilon : float, optional
        Small constant to ensure numerical stability.

    Returns:
    probability_matrix : ndarray
        The initialized probability matrix (n_nodes x n_classes).
    classification_result : ndarray
        Predicted labels (argmax over the probability matrix).
    x_known : ndarray
        One-hot encoded known labels (with node indices).
    """
    V = list(G.nodes())
    adj_matrix = adjacency_matrix(G)
    degree_matrix = np.diag(adj_matrix.sum(axis=1))
    laplacian = degree_matrix - adj_matrix

    # Select known (labeled) nodes from each class
    Sets = known_sets(classes, simplices, BP, seed)
    Boundary_union = set().union(*Sets)
    F = np.sort(list(set(V) - Boundary_union))  # Unlabeled nodes

    # Compute equilibrium measure on unlabeled nodes
    v_F = equilibrium_measure(F, laplacian)

    def process_set(Set):
        """Compute class-wise potential difference for known nodes."""
        probabilities = []
        for s_i in Set:
            F_U_s_i = add_node(F, s_i)
            v_F_U_x = equilibrium_measure(F_U_s_i, laplacian)

            denominator = max(abs(v_F_U_x[s_i]), epsilon)
            prob = (v_F_U_x - v_F) / denominator
            probabilities.append(prob)
        
        return sum(probabilities)

    # Compute class potentials
    Prob = [process_set(Set) for Set in Sets]

    probability_matrix = np.stack(Prob).T
    classification_result = np.argmax(probability_matrix, axis=1)
    known = sorted(Boundary_union)
    x_known = np.array([encode_integer(i, classes) for i in known])
    
    return probability_matrix, classification_result, x_known


# ALL
def generating_general_simplices(G):
    """
    Generates all cliques (simplices) in the graph and groups them by size.

    Parameters:
    G : networkx.Graph
        The input graph.

    Returns:
    general_simplices : list of lists
        general_simplices[k] contains all (k+1)-node cliques.
    """
    all_cliques = list(nx.enumerate_all_cliques(G))
    general_simplices = [[] for _ in range(len(all_cliques[-1]))]
    for clique in all_cliques:
        n = len(clique)
        general_simplices[n-1].append(clique)
    return general_simplices


# MAX
def generating_maximal_simplices(G):
    """
    Generates maximal cliques in the graph and groups them by size.

    Parameters:
    G : networkx.Graph
        The input graph.

    Returns:
    maximal_simplices : list of lists
        maximal_simplices[k] contains all (k+1)-node maximal cliques.
    """
    # Get all cliques to later include 0-simplices (nodes)
    all_cliques = list(nx.enumerate_all_cliques(G))
    general_simplices = [[] for _ in range(len(all_cliques[-1]))]
    for clique in all_cliques:
        n = len(clique)
        general_simplices[n-1].append(clique)

    # Find only maximal cliques
    maximal_cliques = list(nx.find_cliques(G))
    
    # Group maximal cliques by size
    maximal_simplices = []
    for clique in maximal_cliques:
        n = len(clique)
        while len(maximal_simplices) < n:
            maximal_simplices.append([])
        maximal_simplices[n-1].append(clique)

    # Add 0-simplices (nodes) for consistency with general_simplices
    maximal_simplices[0] = general_simplices[0]

    # Sort each clique by node index
    for i in range(len(maximal_simplices)):
        maximal_simplices[i] = [sorted(clique) for clique in maximal_simplices[i]]

    return maximal_simplices


# Aug-MAX
def augment_maximal_cliques(general_simplices, maximal_simplices):
    """
    Augments the set of maximal cliques by adding selected non-maximal cliques
    to balance node participation frequencies.

    Parameters:
    general_simplices : list of lists
        All cliques grouped by size.
    maximal_simplices : list of lists
        Maximal cliques grouped by size.

    Returns:
    augmented_simplices : list of lists
        Augmented maximal cliques with selected additions.
    """

    N = max(max(node for clique in cliques for node in clique) for cliques in general_simplices) + 1
    augmented_simplices = copy.deepcopy(maximal_simplices)

    # Compute participation count for each node
    node_participation = [0] * N
    for cliques in maximal_simplices:
        for clique in cliques:
            for node in clique:
                node_participation[node] += 1

    # Compute average participation and deviation
    total_participation = sum(node_participation)
    avg_participation = total_participation / N
    node_deviation = [p - avg_participation for p in node_participation]

    # Select non-maximal cliques to reduce participation variance
    for k in range(2, len(general_simplices)):
        for clique in general_simplices[k]:
            if clique not in maximal_simplices[k]:
                sum_deviation = sum(node_deviation[node] for node in clique)
                threshold = k * (k - N) / (2 * N)
                if sum_deviation < threshold:
                    augmented_simplices[k].append(clique)
                    for node in clique:
                        node_participation[node] += 1
                    avg_participation = sum(node_participation) / N
                    node_deviation = [p - avg_participation for p in node_participation]

    return augmented_simplices


def get_x_known(data, train_mask, val_mask, num_classes):
    """
    Constructs a label information matrix from train and validation masks.

    Parameters:
    data : torch_geometric.data.Data
        Graph data object containing node labels.
    train_mask : torch.Tensor
        Boolean mask indicating training nodes.
    val_mask : torch.Tensor
        Boolean mask indicating validation nodes.
    num_classes : int
        Number of classes.

    Returns:
    x_known : np.ndarray
        An array of shape (n_known, num_classes + 1) where each row contains
        the node index and its one-hot encoded label.
    """
    # Extract train node indices and labels
    train_indices = train_mask.nonzero().squeeze().numpy()
    train_labels = data.y[train_indices].numpy()
    train_one_hot = np.eye(num_classes)[train_labels]
    train_matrix = np.hstack([train_indices[:, np.newaxis], train_one_hot])

    # Extract validation node indices and labels
    val_indices = val_mask.nonzero().squeeze().numpy()
    val_labels = data.y[val_indices].numpy()
    val_one_hot = np.eye(num_classes)[val_labels]
    val_matrix = np.hstack([val_indices[:, np.newaxis], val_one_hot])

    # Combine train and val matrices
    x_known = np.vstack([train_matrix, val_matrix]).astype(np.int64)

    return x_known


def fac(integer):
    """
    Computes the factorial of an integer as a float tensor.

    Parameters:
    integer : int

    Returns:
    torch.Tensor
        Tensor containing the factorial value.
    """
    return torch.tensor([math.factorial(integer)], dtype=torch.float32)


def basis_mat(n_max, n_classes):
    """
    Generates tensor basis for label combinations across simplices.

    Parameters:
    n_max : int
        Maximum simplex size.
    n_classes : int
        Number of classes.

    Returns:
    list of lists of Tensors
        Each sublist contains all possible label combinations for k-simplices.
    """
    basis = [torch.eye(n_classes)[i].unsqueeze(-1) for i in range(n_classes)]
    mat = [basis] + [[] for _ in range(n_max - 1)]
    for k in range(n_max - 1):
        for j in range(n_classes):
            for i in range(n_classes ** (k + 1)):
                mat[k + 1].append(torch.cat([basis[j], mat[k][i]], dim=1))
    return mat


def prob_product(vectors):
    """
    Computes generalized tensor outer product of a list of probability vectors.

    Parameters:
    vectors : list of torch.Tensor
        Each tensor represents a class distribution for one node.

    Returns:
    torch.Tensor
        Flattened tensor representing the outer product.
    """
    result = vectors[0]
    for vector in vectors[1:]:
        result = torch.ger(result, vector).flatten()
    return result


def generalized_outer_product(P, index_lists):
    """
    Computes outer products of class probability vectors for simplices.

    Parameters:
    P : torch.Tensor
        Class probability matrix (n_nodes x n_classes).
    index_lists : list of lists
        Each sublist contains node indices in a simplex.

    Returns:
    torch.Tensor
        Matrix of outer product vectors for each simplex.
    """
    results = []
    for indices in index_lists:
        selected_vectors = [P[idx] for idx in indices]
        result = prob_product(selected_vectors)
        results.append(result)
    result_matrix = torch.stack(results)
    return result_matrix


# HOI objective
def objective(P, simplices, n_L, exp_base, device):
    """
    Computes the HOI (Higher-Order Interaction) objective function.

    Parameters:
    P : torch.Tensor
        Probability distribution matrix of shape (n_nodes, n_classes).
    simplices : list of lists
        List of simplices (cliques) grouped by size.
    n_L : int
        Number of classes (label types).
    exp_base : float
        Exponential base used for weighting higher-order cliques.
    device : torch.device
        Device on which tensors are computed.

    Returns:
    torch.Tensor
        The total objective value representing label consistency across simplices.
    """
    n_max = len(simplices)

    # Compute outer products of class probabilities for each k-simplex
    prob_prod_set = [generalized_outer_product(P, simplices[i]) for i in range(n_max)]

    # Generate label combination basis tensors
    mat = basis_mat(n_max, n_L)

    # Compute multinomial coefficients for each simplex-label combination
    coef = [0]
    for k in range(1, n_max):
        cvals = []
        for j in range(len(mat[k])):
            row_sums = mat[k][j].sum(1)  # Count of each label
            row_fac = torch.prod(torch.tensor([fac(int(row_sums[i])) for i in range(n_L)]))
            cvals.append(fac(k) / row_fac)
        coef.append(torch.tensor(cvals, device=device))

    # Assign weights for each simplex size
    clique_weight = torch.tensor([exp_base ** i for i in range(n_max)], device=device)

    # Compute weighted sum of objective terms over all simplex sizes
    multi_coef_applied_prob = sum([
        clique_weight[i] * (coef[i] * prob_prod_set[i]).sum()
        for i in range(1, n_max)
    ])

    return multi_coef_applied_prob


# HOI training model 
# HOI training model
class Model(nn.Module):
    def __init__(self, device, initial_data, x_known, exp_base):
        """
        Initializes the HOI model.

        Parameters:
        device : torch.device
            Computation device (CPU or GPU).
        initial_data : array-like or torch.Tensor
            Initial probability distribution (n_nodes x n_classes).
        x_known : np.ndarray
            Array of known node indices and their one-hot encoded labels.
        exp_base : float
            Exponential base for weighting higher-order cliques.
        """
        super(Model, self).__init__()
        self.device = device
        self.exp_base = exp_base

        if not isinstance(initial_data, torch.Tensor):
            initial_data = torch.tensor(initial_data, dtype=torch.float32, device=device)
        else:
            initial_data = initial_data.to(device)

        self.n_V, self.n_L = initial_data.shape  # number of nodes, number of classes

        # Fixed nodes (known labels)
        self.fixed_indices = x_known[:, 0].astype(int)
        fixed_values = torch.tensor(x_known[:, 1:], dtype=torch.float32, device=device)
        self.register_buffer('fixed_params', fixed_values)  # non-trainable, but saved with model

        # Trainable node indices (i.e., not fixed)
        mask = torch.ones(self.n_V, dtype=torch.bool, device=device)
        mask[self.fixed_indices] = False
        self.trainable_indices = torch.arange(self.n_V, device=device)[mask]

        # Initialize trainable parameters directly from initial probabilities
        trainable_init = initial_data[mask]
        self.trainable_params = nn.Parameter(trainable_init)  # shape = (#trainable, n_L)

    def forward(self, simplices):
        """
        Forward pass to compute the HOI objective.

        Steps:
        1. Merge fixed and trainable node distributions.
        2. Apply softmax to ensure valid probability distribution.
        3. Compute the objective value.
        """
        # Combine fixed and trainable parameters
        full_data = torch.zeros((self.n_V, self.n_L), device=self.device)
        full_data[self.fixed_indices] = self.fixed_params
        full_data[self.trainable_indices] = self.trainable_params

        # Normalize to probability distribution
        soft_P = F.softmax(full_data, dim=1)

        # Compute HOI loss
        return objective(soft_P, simplices, self.n_L, self.exp_base, self.device)

    def get_probability_distribution(self):
        """
        Returns the current full node probability matrix.

        - Fixed nodes use original one-hot vectors.
        - Trainable nodes use softmax over their parameters.
        """
        full_data = torch.zeros((self.n_V, self.n_L), device=self.device)
        full_data[self.fixed_indices] = self.fixed_params
        full_data[self.trainable_indices] = self.trainable_params

        soft_P = F.softmax(full_data, dim=1)
        soft_P[self.fixed_indices] = self.fixed_params  # ensure exact one-hot for known labels

        return soft_P


def HOI_training(epochs, device, simplices, initial_data, x_known, lr, exp_base):  
    """
    Trains the HOI model using higher-order clique interactions.

    Parameters:
    epochs : int
        Number of training epochs.
    device : torch.device
        Device to run training on (e.g., 'cpu' or 'cuda').
    simplices : list of lists
        Simplices grouped by size (e.g., edges, triangles, etc.).
    initial_data : array-like
        Initial probability matrix of shape (n_nodes, n_classes).
    x_known : np.ndarray
        Known label information (node index + one-hot label vector).
    lr : float
        Learning rate.
    exp_base : float
        Exponential base for weighting different simplex orders.

    Returns:
    final_P : np.ndarray
        Final probability distribution matrix (n_nodes x n_classes).
    pred : np.ndarray
        Predicted class indices (argmax over final_P).
    """
    model = Model(device, initial_data, x_known, exp_base).to(device)
    optimizer = optim.Adam([model.trainable_params], lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model(simplices)  # Computes HOI objective using current probability distribution
        loss.backward()
        optimizer.step()

        # Optional: print loss during training
        # if epoch % max(1, (epochs // 5)) == 0:
        #     print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # After training, get final softmax probabilities and predictions
    with torch.no_grad():
        final_P_torch = model.get_probability_distribution()
        final_P = final_P_torch.cpu().numpy()
        pred = np.argmax(final_P, axis=1)

    return final_P, pred



def generating_dataset(name_data):
    """
    Loads a Planetoid benchmark dataset (e.g., Cora, Citeseer, Pubmed) and applies normalization.

    Parameters:
    name_data : str
        Name of the dataset ('Cora', 'Citeseer', or 'Pubmed').

    Returns:
    dataset : torch_geometric.datasets.Planetoid
        The loaded and preprocessed dataset.
    """
    dataset = Planetoid(root='/tmp/' + name_data, name=name_data)
    dataset.transform = T.NormalizeFeatures()
    return dataset


##############################################################################################
# Planetoid
##############################################################################################


# Planetoid
class PlanetoidGCN(nn.Module):
    """
    A 2-layer GCN model for node classification, extended with node embeddings for context prediction.
    """

    def __init__(self, num_features, num_classes, embedding_dim=16, num_nodes=None):
        super(PlanetoidGCN, self).__init__()
        self.conv1 = GCNConv(num_features, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, num_classes)

        # Ensure the number of nodes is specified for embedding
        if num_nodes is None:
            raise ValueError("num_nodes must be provided.")
        self.embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))  # For context loss

    def forward(self, data):
        """
        Forward pass through two GCN layers with ReLU activation.

        Parameters:
        data : torch_geometric.data.Data
            Graph data object.

        Returns:
        torch.Tensor
            Log-softmax predictions over classes.
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embedding(self):
        """
        Returns node embeddings used in context loss.

        Returns:
        torch.nn.Parameter
        """
        return self.embedding


# Planetoid 
def sample_context(graph, labels, num_samples=100):
    """
    Samples context node pairs (i, j) based on 2-hop neighborhoods.

    Parameters:
    graph : networkx.Graph
        The input graph.
    labels : list or array-like
        Ground-truth labels for each node.
    num_samples : int
        Number of node pairs to sample.

    Returns:
    context_pairs : list of tuples
        List of sampled node index pairs (i, j).
    gamma_values : list of int
        List of gamma values: 1 if same label, else 0.
    """
    context_pairs = []
    gamma_values = []
    nodes = list(graph.nodes)

    for _ in range(num_samples):
        node = np.random.choice(nodes)
        neighbors = list(nx.single_source_shortest_path_length(graph, node, cutoff=2).keys())
        if len(neighbors) > 1:
            neighbor = np.random.choice(neighbors)
            context_pairs.append((node, neighbor))
            gamma_values.append(1 if labels[node] == labels[neighbor] else 0)

    return context_pairs, gamma_values


# Planetoid 
def context_loss(embedding, context_pairs, gamma_values):
    """
    Computes the unsupervised context prediction loss.

    Parameters:
    embedding : torch.Tensor
        Node embeddings from the model (num_nodes x dim).
    context_pairs : list of tuple
        Sampled node pairs (i, j) based on proximity in the graph.
    gamma_values : list of int (0 or 1)
        Similarity indicator for each pair: 1 if labels match, else 0.

    Returns:
    torch.Tensor
        Averaged context prediction loss over all sampled pairs.
    """
    loss = 0
    for (i, j), gamma in zip(context_pairs, gamma_values):
        score = torch.dot(embedding[i], embedding[j])  # Similarity score
        # Binary classification loss: similar → high score, dissimilar → low score
        loss += gamma * F.logsigmoid(score) + (1 - gamma) * F.logsigmoid(-score)

    return -loss / len(context_pairs) if len(context_pairs) > 0 else 0


# Planetoid 
def gcn_training(model, data, graph, device, lr, weight_decay, num_epochs, context_samples):
    """
    Trains a PlanetoidGCN model using both supervised and unsupervised objectives.

    Parameters:
    model : PlanetoidGCN
        The GCN model with additional node embeddings for context learning.
    data : torch_geometric.data.Data
        Graph dataset including features, edge index, labels, and masks.
    graph : networkx.Graph
        NetworkX version of the graph used for sampling context.
    device : torch.device
        Computation device ('cpu' or 'cuda').
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay (L2 regularization).
    num_epochs : int
        Number of training epochs.
    context_samples : int
        Number of context node pairs to sample per epoch.

    Returns:
    softmax_output : torch.Tensor
        Final softmax predictions over classes (n_nodes x n_classes).
    predictions : torch.Tensor
        Predicted class indices (argmax from softmax_output).
    """
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Supervised loss using labeled nodes
        out = model(data)
        loss_s = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        # Unsupervised context loss using sampled node pairs
        context_pairs, gamma_values = sample_context(graph, data.y, context_samples)
        loss_u = context_loss(model.get_embedding(), context_pairs, gamma_values)

        # Combine losses (weighted sum)
        loss = loss_s + 0.5 * loss_u
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}, Total Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data)
        softmax_output = F.softmax(logits, dim=1)
    predictions = torch.argmax(softmax_output, dim=1)
    return softmax_output, predictions

##############################################################################################
##############################################################################################



##############################################################################################
# SGC, GCN, GAT
##############################################################################################

class SGC(torch.nn.Module):
    """
    Simple Graph Convolution (SGC) model.
    Uses a single linear transformation after K-step propagation without non-linearity or dropout.
    """
    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv = SGConv(num_features, num_classes, K=2, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    """
    Two-layer Graph Convolutional Network (GCN).
    Includes ReLU activation between layers.
    """
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """
    Two-layer Graph Attention Network (GAT).
    Uses multi-head attention in the first layer and dropout for regularization.
    """
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, num_classes,
                             concat=False, heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# SGC, GCN, GAT 
def gat_training(model, data, device, lr, weight_decay, epochs, eval_interval):
    """
    Trains and evaluates a GAT (Graph Attention Network) model.

    Parameters:
    model : torch.nn.Module
        The GAT model instance.
    data : torch_geometric.data.Data
        Input graph data containing features, edge_index, labels, and masks.
    device : torch.device
        Device for computation ('cpu' or 'cuda').
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization factor.
    epochs : int
        Number of training epochs.
    eval_interval : int
        Interval (in epochs) at which to optionally print training status.

    Returns:
    softmax_output : torch.Tensor
        Final softmax probability matrix over classes (n_nodes x n_classes).
    predictions : torch.Tensor
        Predicted class indices (argmax over softmax_output).
    """
    # Move model and data to the specified device
    model = model.to(device)
    data = data.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)  # Forward pass
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # Supervised loss

        # Optional: print loss at intervals
        # if (epoch + 1) % eval_interval == 0:
        #     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data)
        softmax_output = F.softmax(logits, dim=1)
        predictions = torch.argmax(softmax_output, dim=1)

    return softmax_output, predictions



##############################################################################################
##############################################################################################

def calculate_accuracy(outputs, labels, mask):
    """
    Computes accuracy over masked nodes.

    Parameters:
    outputs : torch.Tensor or np.ndarray
        Model output — either softmax probabilities or predicted labels.
    labels : torch.Tensor or np.ndarray
        Ground truth labels.
    mask : torch.Tensor
        Boolean mask indicating which nodes to evaluate.

    Returns:
    accuracy : float
        The proportion of correctly predicted nodes.
    """
    # Extract indices where mask is True
    mask_indices = mask.nonzero(as_tuple=True)[0]
    
    # Case 1: outputs are softmax probabilities
    if len(outputs.shape) > 1 and outputs.shape[1] > 1:
        preds = outputs[mask_indices].max(1)[1]  # Get predicted class
    else:
        # Case 2 or 3: outputs are already predicted labels
        preds = outputs[mask_indices]
    
    # Extract ground truth labels
    true_labels = labels[mask_indices]
    
    # Calculate accuracy
    correct = (preds == true_labels).sum().item()
    accuracy = correct / mask.sum().item()
    
    return accuracy




