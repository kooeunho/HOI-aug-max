Title: Node Classification via Simplicial Interaction with Augmented Maximal Clique Selection

Abstract: Considering higher-order interactions allows for a more comprehensive understanding
 of network structures beyond simple pairwise connections. While leveraging all cliques
 in a network to handle higher-order interactions is intuitive, it often leads to
 computational inefficiencies due to overlapping information between higher-order and
 lower-order cliques. To address this issue, we propose an augmented maximal clique
 strategy. Although using only maximal cliques can reduce unnecessary overlap and
 provide a concise representation of the network, certain nodes may still appear in
 multiple maximal cliques, resulting in imbalanced training data. Therefore, our
 augmented maximal clique approach selectively includes some non-maximal cliques to
 mitigate the overrepresentation of specific nodes and promote more balanced learning
 across the network. Comparative analyses on synthetic networks and real-world
 citation datasets demonstrate that our method outperforms approaches based on
 pairwise interactions, all cliques, or only maximal cliques. Additionally, by integrating
 this strategy into GNN-based semi-supervised learning, we establish a link between
 maximal clique-based methods and GNNs, showing that incorporating higher-order
 structures improves predictive accuracy. As a result, the augmented maximal clique
 strategy offers a computationally efficient and effective solution for higher-order
 network learning.

Python version: 3.12.4

Pytorch version: 2.4.0

Files demonstrate the structure of the proposed augmented maximal clique strategy as well as optimization procedure

Synthetic experiments.ipynb : Experiment threshold on balanced and imbalanced setting

GNN experiments.ipynb : All real dataset experiments

function.py : utils and functions for the synthetic experiments
