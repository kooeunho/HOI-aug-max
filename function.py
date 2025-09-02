import numpy as np
import networkx as nx
from scipy.optimize import linprog
import math
from itertools import combinations
import random
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ====================================================================
# 1. GRAPH GENERATION & INITIALIZATION (OPTIMIZED)
# ====================================================================

def generating_graph_with_simplices(n_units, prob_mat, seed):
    """그래프를 효율적으로 생성합니다."""
    random.seed(seed)
    np.random.seed(seed)
    
    num_nodes = sum(n_units)
    V = list(range(num_nodes))

    node_to_class = np.zeros(num_nodes, dtype=int)
    class_boundaries = np.cumsum(np.insert(n_units, 0, 0))
    classes = []
    for i in range(len(n_units)):
        class_nodes = set(range(class_boundaries[i], class_boundaries[i+1]))
        classes.append(class_nodes)
        node_to_class[class_boundaries[i]:class_boundaries[i+1]] = i

    G = nx.Graph()
    G.add_nodes_from(V)
    
    for i, j in combinations(V, 2):
        class_i = node_to_class[i]
        class_j = node_to_class[j]
        if random.random() < prob_mat[class_i][class_j]:
            G.add_edge(i, j)

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        if components:
            main_component = max(components, key=len)
            for component in components:
                if component != main_component:
                    G.add_edge(random.choice(list(main_component)), random.choice(list(component)))

    nx.set_node_attributes(G, {n: c for c, nodes in enumerate(classes) for n in nodes}, 'label')
    return G, classes

def known_sets(classes, simplices, BP):
    """collections.Counter를 사용해 노드의 Clique 참여 빈도를 효율적으로 계산합니다."""
    n_V = sum(len(c) for c in classes)
    all_nodes_in_cliques = (node for size_cliques in simplices for clique in size_cliques for node in clique)
    participation_counts = Counter(all_nodes_in_cliques)
    sorted_nodes = sorted(range(n_V), key=lambda n: participation_counts.get(n, 0), reverse=True)
    
    node_to_class_map = {node: i for i, class_set in enumerate(classes) for node in class_set}
    
    new_classes = [[] for _ in range(len(classes))]
    for node in sorted_nodes:
        new_classes[node_to_class_map[node]].append(node)
        
    total_to_pick = math.ceil(n_V * BP)
    nodes_per_class = math.ceil(total_to_pick / len(classes))
    known = [nc[:nodes_per_class] for nc in new_classes]
    return known

def linear_programming(A, method='highs'):
    """scipy의 최적화된 LP 솔버를 사용합니다."""
    n = A.shape[0]
    c = np.zeros(n + 1); c[-1] = 1
    A_ub = np.hstack((A, -np.ones((n, 1)))); b_ub = np.zeros(n)
    A_eq = np.ones((1, n + 1)); A_eq[0, -1] = 0; b_eq = np.array([1])
    bounds = [(0, 1)] * n + [(0, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
    if result.success:
        return result.x[:-1] / (result.x[-1] + 1e-10)
    else:
        raise ValueError("LP 문제 해결에 실패했습니다.")

def equilibrium_measure(F, L):
    """Equilibrium measure를 계산합니다."""
    A = L[F][:, F]
    EM_tem = linear_programming(A)
    result = np.zeros(len(L)); result[F] = EM_tem
    return result

def initialization(G, classes, simplices, BP, epsilon=1e-10):
    """최적화된 함수들을 호출하여 초기 확률 분포를 계산합니다."""
    V = list(G.nodes()); n_V = len(V)
    laplacian = nx.laplacian_matrix(G).toarray()
    
    Sets = known_sets(classes, simplices, BP)
    Boundary_union = set().union(*Sets)
    F = np.sort(list(set(V) - Boundary_union))
    
    v_F = equilibrium_measure(F, laplacian)
    
    def process_set(Set, F_nodes, base_v_F):
        probabilities = []
        for s_i in Set:
            F_U_s_i = np.sort(np.append(F_nodes, s_i))
            v_F_U_s_i = equilibrium_measure(F_U_s_i, laplacian)
            denominator = max(abs(v_F_U_s_i[s_i]), epsilon)
            probabilities.append((v_F_U_s_i - base_v_F) / denominator)
        return sum(probabilities)

    Prob = [process_set(Set, F, v_F) for Set in Sets]
    
    probability_matrix = np.stack(Prob).T
    classification_result = np.argmax(probability_matrix, axis=1)
    
    known_nodes = sorted(list(Boundary_union))
    node_to_class_map = {node: i for i, class_set in enumerate(classes) for node in class_set}
    
    x_known_list = []
    for node in known_nodes:
        encoded = np.zeros(len(classes) + 1)
        encoded[0] = node
        encoded[node_to_class_map[node] + 1] = 1
        x_known_list.append(encoded)
        
    return probability_matrix, classification_result, np.array(x_known_list)

# ====================================================================
# 2. CLIQUE (SIMPLEX) GENERATION (OPTIMIZED)
# ====================================================================

def group_by_size(cliques):
    if not cliques: return []
    max_len = len(max(cliques, key=len)) if cliques else 0
    grouped = [[] for _ in range(max_len)]
    for c in cliques:
        if len(c) > 0: grouped[len(c)-1].append(c)
    return grouped

def _calculate_variance(participation_counter, num_nodes):
    if not participation_counter: return 0.0
    freqs = np.array(list(participation_counter.values()))
    all_freqs = np.pad(freqs, (0, num_nodes - len(freqs)), 'constant')
    return np.var(all_freqs)

def generate_simplex_sets(G: nx.Graph, budget: int = 500):
    print("  1/3: Maximal Clique 탐색 중...")
    maximal_cliques_list = [sorted(c) for c in nx.find_cliques(G)]
    
    print("  2/3: All Clique 생성 중 (Maximal Clique 기반)...")
    all_cliques_set = {tuple(sorted(sub)) for m in maximal_cliques_list for k in range(1, len(m) + 1) for sub in combinations(m, k)}
    all_cliques_list = [list(c) for c in all_cliques_set]

    print(f"  3/3: Augmented-Maximal Clique 생성 중 (Greedy, Budget={budget})...")
    num_nodes = G.number_of_nodes()
    aug_max_set = {tuple(c) for c in maximal_cliques_list}
    node_part = Counter(n for c in aug_max_set for n in c)
    current_var = _calculate_variance(node_part, num_nodes)
    
    candidates = all_cliques_set - aug_max_set
    
    for _ in range(budget):
        if not candidates: break
        best_clique, min_var = None, current_var
        
        sample_size = min(len(candidates), 100)
        for cand in random.sample(list(candidates), sample_size):
            temp_part = node_part.copy(); temp_part.update(cand)
            pot_var = _calculate_variance(temp_part, num_nodes)
            if pot_var < min_var:
                min_var, best_clique = pot_var, cand
        
        if best_clique:
            aug_max_set.add(best_clique); candidates.remove(best_clique)
            node_part.update(best_clique); current_var = min_var
        else:
            break

    return (
        group_by_size(all_cliques_list),
        group_by_size(maximal_cliques_list),
        group_by_size([list(c) for c in aug_max_set])
    )

# ====================================================================
# 3. HOI MODEL AND TRAINING (FINAL VERSION with Weight Strategy)
# ====================================================================

def precompute_hoi_coefficients(n_max, n_classes, device):
    fact_lookup = torch.tensor([math.factorial(i) for i in range(n_max + 2)], dtype=torch.float32, device=device)
    mat = [[torch.eye(n_classes, device=device)[i] for i in range(n_classes)]]
    for k in range(n_max):
        next_mat = [torch.cat((mat[0][j], item.T)).T for j in range(n_classes) for item in mat[k]]
        mat.append(next_mat)

    coef = [torch.empty(0, device=device)]
    for k in range(1, n_max + 2):
        cvals = []
        if k-1 < len(mat) and mat[k-1]:
            for item in mat[k-1]:
                row_sums = item.sum(1)
                row_fac = torch.prod(fact_lookup[row_sums.long()])
                cvals.append(fact_lookup[k] / row_fac)
        coef.append(torch.tensor(cvals, device=device))
    return coef

def generalized_outer_product(P, index_lists):
    if not index_lists: return torch.empty(0, device=P.device)
    def prob_product(vectors):
        res = vectors[0]
        for v in vectors[1:]: res = torch.ger(res, v).flatten()
        return res
    return torch.stack([prob_product([P[idx] for idx in indices]) for indices in index_lists])

def objective_efficient(P, simplices, device, precomputed_coef, weight_strategy='constant'):
    n_max = len(simplices)
    if weight_strategy == 'linear':
        clique_weight = torch.arange(1, n_max + 1, device=device, dtype=torch.float32)
    else:
        clique_weight = torch.ones(n_max, device=device, dtype=torch.float32)
            
    total_obj = 0.0
    for i in range(1, n_max):
        clique_size = i + 1
        if not simplices[i] or clique_size >= len(precomputed_coef): continue
        prob_prod = generalized_outer_product(P, simplices[i])
        coef_slice = precomputed_coef[clique_size]
        if coef_slice.shape[0] == prob_prod.shape[1]:
            term_sum = (coef_slice * prob_prod).sum()
            total_obj += clique_weight[i] * term_sum
    return total_obj

class HOIModel(nn.Module):
    def __init__(self, device, initial_data, x_known, precomputed_coef):
        super().__init__()
        self.device, self.precomputed_coef = device, precomputed_coef
        init_tensor = torch.tensor(initial_data, dtype=torch.float32, device=device)
        self.n_V, self.n_L = init_tensor.shape
        self.fixed_indices = torch.from_numpy(x_known[:, 0].astype(int)).to(device)
        self.fixed_params = torch.tensor(x_known[:, 1:], dtype=torch.float32, device=device)
        mask = torch.ones(self.n_V, dtype=torch.bool); mask[self.fixed_indices.cpu()] = False
        self.trainable_indices = torch.arange(self.n_V)[mask].to(device)
        self.trainable_params = nn.Parameter(init_tensor[mask])

    def forward(self, simplices, weight_strategy):
        full_data = torch.zeros((self.n_V, self.n_L), device=self.device)
        full_data[self.fixed_indices] = self.fixed_params
        full_data[self.trainable_indices] = self.trainable_params
        soft_P = F.softmax(full_data, dim=1)
        return objective_efficient(soft_P, simplices, self.device, self.precomputed_coef, weight_strategy)

    def get_probability_distribution(self):
        with torch.no_grad():
            full_data = torch.zeros((self.n_V, self.n_L), device=self.device)
            full_data[self.fixed_indices] = self.fixed_params
            full_data[self.trainable_indices] = self.trainable_params
            return F.softmax(full_data, dim=1)

def HOI_training(epochs, device, simplices, initial_data, x_known, lr, precomputed_coef, weight_strategy):
    model = HOIModel(device, initial_data, x_known, precomputed_coef).to(device)
    optimizer = optim.Adam([model.trainable_params], lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model(simplices, weight_strategy)
        loss.backward()
        optimizer.step()
    final_P = model.get_probability_distribution(); pred = torch.argmax(final_P, axis=1)
    return final_P.cpu().numpy(), pred.cpu().numpy()

# ====================================================================
# 4. PERFORMANCE METRICS CALCULATION
# ====================================================================

def calculate_all_metrics(y_true, y_pred, y_prob, n_labels):
    if len(np.unique(y_pred)) < 2:
        return {'accuracy': accuracy_score(y_true, y_pred), 'macro_f1': 0.0, 'roc_auc': 0.5, 'kappa': 0.0}
    y_true_bin = label_binarize(y_true, classes=range(n_labels))
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'roc_auc': roc_auc_score(y_true_bin, y_prob, multi_class='ovr'),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }