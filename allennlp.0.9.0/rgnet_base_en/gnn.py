import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn import util

class CharmGNN(nn.Module):
    def __init__(self, node_dim, reasoning_steps = 1):
        super(CharmGNN, self).__init__()

        self.node_dim = node_dim
        self.reasoning_steps = reasoning_steps

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._pp_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._question_p_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._question_q_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

        self._predice_p_node = torch.nn.Conv1d(128, 3, kernel_size=1, bias=True)
        self._predice_q_node = torch.nn.Conv1d(128, 3, kernel_size=1, bias=True)

        self.output_probs = nn.Softmax(dim=-1)

        self.q_node_Linear = torch.nn.Linear(node_dim,128)
        self.p_node_Linear = torch.nn.Linear(node_dim,128)
        self.question_node_Linear = torch.nn.Linear(node_dim,128)
    def forward(self,
                p_node: torch.LongTensor,
                q_node: torch.LongTensor,
                question_node: torch.LongTensor,
                p_node_mask: torch.LongTensor,
                q_node_mask: torch.LongTensor,
                question_node_mask: torch.LongTensor,
                pp_graph: torch.LongTensor,
                qq_graph: torch.LongTensor,
                passage_evidence: torch.LongTensor,
                question_evidence: torch.LongTensor
                ):
        p_node_len = p_node.size(1)
        q_node_len = q_node.size(1)
        #print("p_node",p_node.size(),p_node[0,:,:])
        #print("q_node",q_node.size(),q_node[0,:,:])
        #print("pp_graph", pp_graph.size(), pp_graph[0, :, :])

        pp_graph = p_node_mask.unsqueeze(1) * p_node_mask.unsqueeze(-1) * pp_graph
        #print("pp_graph", pp_graph.size(), pp_graph[0, :, :])

        qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * qq_graph
        #print("qq_graph", qq_graph.size(), qq_graph[0, :, :])

        qp_graph = (p_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1)).float()
        #print("qp_graph", qp_graph.size(),qp_graph[0,:,:])

        question_p_graph = p_node_mask.unsqueeze(1) * question_node_mask.unsqueeze(-1)
        question_q_graph = q_node_mask.unsqueeze(1) * question_node_mask.unsqueeze(-1)
        #print("question_pq_graph", question_p_graph.size(), question_p_graph[0,:,:])

        p_node_neighbor_num = pp_graph.sum(-1)
        p_node_neighbor_num_mask = (p_node_neighbor_num >= 1).long()
        p_node_neighbor_num = util.replace_masked_values(p_node_neighbor_num.float(), p_node_neighbor_num_mask, 1)

        q_node_neighbor_num = qp_graph.sum(-1)  + qq_graph.sum(-1)
        q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        q_node_neighbor_num = util.replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1)

        question_neighbor_num = question_p_graph.sum(-1) + question_q_graph.sum(-1)
        question_neighbor_num_mask = (question_neighbor_num >= 1).long()
        question_neighbor_num = util.replace_masked_values(question_neighbor_num.float(),question_neighbor_num_mask,1)
        for step in range(self.reasoning_steps):
            p_node_weight = torch.sigmoid(self._node_weight_fc(p_node)).squeeze(-1)
            q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)
            question_weight = torch.sigmoid(self._node_weight_fc(question_node).squeeze(-1))

            self_p_node_info = self._self_node_fc(p_node)
            self_q_node_info = self._self_node_fc(q_node)
            self_question_info = self._self_node_fc(question_node)

            pp_node_info = self._pp_node_fc_right(p_node)
            qp_node_info = self._qd_node_fc_right(p_node)
            qq_node_info = self._qq_node_fc_right(q_node)
            question_p_info = self._question_p_node_fc_right(p_node)
            question_q_info = self._question_q_node_fc_right(q_node)

            pp_node_weight = util.replace_masked_values(
                p_node_weight.unsqueeze(1).expand(-1, p_node_len, -1),
                pp_graph,
                0)

            qp_node_weight = util.replace_masked_values(
                p_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qp_graph,
                0)

            qq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qq_graph,
                0)

            question_p_weight = util.replace_masked_values(
                p_node_weight.unsqueeze(1).expand(-1, question_node.size(1), -1),
                question_p_graph,
                0)

            question_q_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, question_node.size(1), -1),
                question_q_graph,
                0)

            pp_node_info = torch.matmul(pp_node_weight, pp_node_info)
            qp_node_info = torch.matmul(qp_node_weight, qp_node_info)
            qq_node_info = torch.matmul(qq_node_weight, qq_node_info)
            question_p_info = torch.matmul(question_p_weight, question_p_info)
            question_q_info = torch.matmul(question_q_weight, question_q_info)
            #print("pp_node_info",pp_node_info.size(),pp_node_info[0,:,:])
            #print("qp_node_info", qp_node_info.size(), qp_node_info[0, :, :])
            #print("qq_node_info", qq_node_info.size(), qq_node_info[0, :, :])
            #print("question_p_info", question_p_info.size(), question_p_info[0, :, :])
            #print("question_q_info", question_q_info.size(), question_q_info[0, :, :])

            agg_p_node_info = (pp_node_info) / p_node_neighbor_num.unsqueeze(-1)
            agg_q_node_info = (qp_node_info + qq_node_info) / q_node_neighbor_num.unsqueeze(-1)
            agg_question_node_info = (question_p_info + question_q_info) / question_neighbor_num.unsqueeze(-1)

            p_node = F.relu(self_p_node_info + agg_p_node_info)
            q_node = F.relu(self_q_node_info + agg_q_node_info)
            question_node = F.relu(self_question_info + agg_question_node_info)


        p_node = self.p_node_Linear(p_node)
        q_node = self.q_node_Linear(q_node)
        question_node = self.question_node_Linear(question_node)
        '''
        predict_p_nodes = self.output_probs(self._predice_p_node(p_node.permute(0,2,1)).permute(0,2,1))
        predict_q_nodes = self.output_probs(self._predice_q_node(q_node.permute(0,2,1)).permute(0,2,1))
        
        predict_nodes = torch.cat([predict_p_nodes, predict_q_nodes],dim = 1)
        evidence = torch.cat([passage_evidence, question_evidence], dim = 1)
        predict_nodes_seqfirst = predict_nodes.permute(1,0,2)
        target_node_seqfirst = evidence.permute(1,0).long()
        node_pad_mask = (target_node_seqfirst>0).long()
        log_loss_support_list = []
        for _i in range(predict_nodes_seqfirst.size(0)):
            predicted_node_probs = predict_nodes_seqfirst[_i]
            predicted_node_probs = predicted_node_probs.squeeze(1)
            true_node_labels = target_node_seqfirst[_i]
            mask_node_labels = node_pad_mask[_i]
            selected_node_probs = torch.gather(predicted_node_probs, 1 ,true_node_labels.unsqueeze(1))
            selected_node_probs = selected_node_probs.squeeze(1)
            selected_node_logprobs = torch.log(selected_node_probs)
            log_loss_support_list.append(selected_node_logprobs * mask_node_labels)
        log_likelihood_for_node = torch.stack(log_loss_support_list, 1)
        log_marginal_likelihood_for_node = log_likelihood_for_node.sum(1, keepdim=True)
        '''
        return p_node, q_node, question_node#, log_marginal_likelihood_for_node
