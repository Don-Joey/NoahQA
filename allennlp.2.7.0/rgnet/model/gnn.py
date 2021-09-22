import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

from allennlp.nn import util

import numpy as np


def loss_nodes(y_pred_nodes, y_nodes, node_cw):
    """
    Loss function for node predictions.

    Args:
        y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
        y_nodes: Targets for nodes (batch_size, num_nodes)
        node_cw: Class weights for nodes loss

    Returns:
        loss_nodes: Value of loss function

    """
    # Node loss
    y = F.log_softmax(y_pred_nodes, dim=2)  # B x V x voc_nodes_out
    y = y.permute(0, 2, 1)  # B x voc_nodes x V
    loss_nodes = nn.NLLLoss(node_cw)(y, y_nodes)
    return loss_nodes


class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)

        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
        return e_bn


class NodeFeatures(nn.Module):
    """Convnet features for nodes.

    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]

    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """

    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H
        Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        gateVx = edge_gate * Vx  # B x V x V x H
        if self.aggregation == "mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation == "sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  # Extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = Ue + Vx + Wx
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H
        # Compute edge gates
        edge_gate = F.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate)
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y


class CharmGNN(nn.Module):
    def __init__(self, node_dim, reasoning_steps=1):
        super(CharmGNN, self).__init__()

        self.node_dim = node_dim
        self.reasoning_steps = reasoning_steps

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)

        self.nodes_embedding = nn.Linear(node_dim, int(node_dim / 2), bias=False)
        self.edges_embedding = nn.Embedding(3, int(node_dim / 2))

        gcn_layers = []
        for layer in range(self.reasoning_steps):
            gcn_layers.append(ResidualGatedGCNLayer(int(node_dim / 2), "mean"))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.mlp_edges = MLP(int(node_dim / 2), 2, 2)

    def forward(self,
                p_node: torch.LongTensor,
                q_node: torch.LongTensor,
                question_node: torch.LongTensor,
                p_node_mask: torch.LongTensor,
                q_node_mask: torch.LongTensor,
                question_node_mask: torch.LongTensor,
                pp_graph: torch.LongTensor,
                qq_graph: torch.LongTensor,
                pq_graph: torch.LongTensor,
                question_p_graph: torch.LongTensor,
                question_q_graph: torch.LongTensor,
                question_node_graph: torch.LongTensor,
                pp_graph_evidence: torch.LongTensor,
                qq_graph_evidence: torch.LongTensor,
                pq_graph_evidence: torch.LongTensor,
                question_p_graph_evidence: torch.LongTensor,
                question_q_graph_evidence: torch.LongTensor,
                question_node_graph_evidence: torch.LongTensor,
                ):
        p_len = p_node.size(1)
        q_len = q_node.size(1)
        graph = torch.cat((pp_graph, pq_graph, question_p_graph.transpose(1, 2)), dim=-1)
        graph = torch.cat(
            (graph, torch.cat((pq_graph.transpose(1, 2), qq_graph, question_q_graph.transpose(1, 2)), dim=-1)), dim=1)
        graph = torch.cat((graph, torch.cat((question_p_graph, question_q_graph, question_node_graph), dim=-1)),
                          dim=1).long()

        graph_evidence = torch.cat((pp_graph_evidence, pq_graph_evidence, question_p_graph_evidence.transpose(1, 2)),
                                   dim=-1)
        graph_evidence = torch.cat(
            (graph_evidence, torch.cat(
                (pq_graph_evidence.transpose(1, 2), qq_graph_evidence, question_q_graph_evidence.transpose(1, 2)),
                dim=-1)), dim=1)
        graph_evidence = torch.cat((graph_evidence, torch.cat(
            (question_p_graph_evidence, question_q_graph_evidence, question_node_graph_evidence), dim=-1)),
                                   dim=1).long()

        edge_labels = graph_evidence.cpu().numpy().flatten()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        edge_cw = torch.Tensor(edge_cw).type(torch.FloatTensor).cuda(device=graph_evidence.device)

        nodes = torch.cat((p_node, q_node, question_node), 1)
        x = self.nodes_embedding(nodes)
        e = self.edges_embedding(graph)

        for layer in range(self.reasoning_steps):
            x, e = self.gcn_layers[layer](x, e)
        y_pred_edges = self.mlp_edges(e)

        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges

        y = y.permute(0, 3, 1, 2)

        log_marginal = nn.NLLLoss(edge_cw)(y, graph_evidence)

        p_node = x[:, :p_len, :]
        q_node = x[:, p_len:(p_len + q_len), :]
        question_node = x[:, (p_len + q_len):, :]

        return p_node, q_node, question_node, log_marginal, y_pred_edges