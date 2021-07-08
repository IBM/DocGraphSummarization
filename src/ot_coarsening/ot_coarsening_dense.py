##Method2: Directly compare original features and coarsed features (with the same dimension,
# GNN(fe, hidden) - Coarsen - GNN(hidden, fe) ).
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, JumpingKnowledge
from sinkhorn import sinkhorn_loss_default
from pytorch_memlab import profile, set_target_gpu, profile_every

class CoarsenBlock(torch.nn.Module):
    def __init__(self, in_channels, assign_ratio, num_output_sentences):
        super(CoarsenBlock, self).__init__()

        self.gcn_att = DenseGCNConv(in_channels, 1, bias=True)
        self.num_output_sentences = num_output_sentences
        self.assign_ratio = assign_ratio

    def normalize_batch_adj(self, adj):  # adj shape: batch_size * num_node * num_node, D^{-1/2} (A+I) D^{-1/2}
        dim = adj.size()[1]
        A = adj + torch.eye(dim, device=adj.device)
        deg_inv_sqrt = A.sum(dim=-1).clamp(min=1).pow(-0.5)

        newA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        newA = (adj.sum(-1)>0).float().unsqueeze(-1).to(adj.device) * newA
        return newA

    def get_topk_range_for_sentences(self, alpha_vec, num_sentences):
        alpha_vec = alpha_vec.detach().cpu().numpy()
        full_argsort = np.argsort(alpha_vec)[::-1]
        topk_ind = []
        sentences_so_far = 0
        for index in full_argsort:
            if sentences_so_far >= self.num_output_sentences:
                break
            is_sentence = index < num_sentences
            if is_sentence:
                sentences_so_far += 1
            topk_ind.append(index)

        temptopk = torch.Tensor(alpha_vec[topk_ind])
        topk_ind = torch.Tensor(topk_ind).int()
        # return temptopk and topkind
        return temptopk, topk_ind

    def reset_parameters(self):
        self.gcn_att.reset_parameters()

    def forward(self, x, adj, num_sentences=None):
        # alpha_vec = F.softmax(self.att(x).sum(-1), -1)
        alpha_vec = F.sigmoid(torch.pow(self.gcn_att(x,adj),2)).squeeze() # b*n*1 --> b*n
        if len(alpha_vec.shape) < 2:
            alpha_vec = alpha_vec.unsqueeze(0)
        norm_adj = self.normalize_batch_adj(adj)
        batch_size = x.size()[0]
        cut_value = torch.zeros_like(alpha_vec[:, 0])
        batch_topk_ind = []
        for j in range(batch_size):
            if num_sentences is None:
                # This is in the train phase
                num_coarse_nodes = int(num_nodes * self.assign_ratio) + 1
                temptopk, topk_ind = alpha_vec.topk(num_coarse_nodes, dim=-1)
                cut_value = temptopk[-1]
            else:
                # This is in the evaluation phase
                temptopk, topk_ind = self.get_topk_range_for_sentences(alpha_vec[j] , num_sentences[j])
                cut_value = temptopk[-1]
                batch_topk_ind.append(topk_ind)
        # cut_alpha_vec = torch.mul( ((alpha_vec - torch.unsqueeze(cut_value, -1))>=0).float(), alpha_vec)  # b * n
        cut_value = cut_value.to(x.device)
        cut_alpha_vec = F.relu(alpha_vec+0.0000001 - torch.unsqueeze(cut_value, -1))
        S = torch.mul(norm_adj, cut_alpha_vec.unsqueeze(1))  # repeat rows of cut_alpha_vec, #b * n * n
        # temp_rowsum = torch.sum(S, -1).unsqueeze(-1).pow(-1)
        # # temp_rowsum[temp_rowsum > 0] = 1.0 / temp_rowsum[temp_rowsum > 0]
        # S = torch.mul(S, temp_rowsum)  # row-wise normalization
        S = F.normalize(S, p=1, dim=-1)

        embedding_tensor = torch.matmul(torch.transpose(S, 1, 2),
                                        x)  # equals to torch.einsum('bij,bjk->bik',...)
        new_adj = torch.matmul(torch.matmul(torch.transpose(S, 1, 2), adj), S)  # batched matrix multiply

        return embedding_tensor, new_adj, S, batch_topk_ind

class Coarsening(torch.nn.Module):
    def __init__(self, dataset, hidden, ratio=0.5, epsilon=1.0, opt_epochs=100, num_output_sentences=3): # we only use 1 layer for coarsening
        super(Coarsening, self).__init__()
        self.ratio = ratio
        self.epsilon = epsilon
        self.opt_epochs = opt_epochs
        self.num_output_sentences = num_output_sentences
        # self.embed_block1 = GNNBlock(dataset.num_features, hidden, hidden)
        self.embed_block1 = DenseGCNConv(dataset.num_features, hidden)
        self.coarse_block1 = CoarsenBlock(hidden, ratio, num_output_sentences)
        self.embed_block2 = DenseGCNConv(hidden, dataset.num_features)
        self.jump = JumpingKnowledge(mode='cat')

        # self.lin1 = Linear(hidden + dataset.num_features, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.coarse_block1.reset_parameters()
        self.jump.reset_parameters()
        # self.lin1.reset_parameters()
        # self.lin2.reset_parameters()

    def forward(self, data, p=2):
        x, adj, mask = data.x, data.adj, data.mask
        num_sentences = data.num_sentences
        batch_num_nodes = data.mask.sum(-1)
        x1 = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        # xs = [x1.mean(dim=1)]
        coarse_x, new_adj, S, output_indices= self.coarse_block1(x1, adj, num_sentences=num_sentences)
        xs = [coarse_x.mean(dim=1)]
        x2 = F.tanh(self.embed_block2(coarse_x, new_adj, mask, add_loop=True))
        xs.append(x2.mean(dim=1))

        opt_loss = 0.0
        for i in range(len(x)):
            x3 = self.get_nonzero_rows(x[i])
            x4 = self.get_nonzero_rows(x2[i])
            opt_loss += sinkhorn_loss_default(x3, x2[i], self.epsilon, niter=self.opt_epochs, p=p)

        return xs, new_adj, S, opt_loss, output_indices

    def predict(self, xs):
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def get_nonzero_rows(self, M):# M is a matrix
        # row_ind = M.sum(-1).nonzero().squeeze() #nonzero has bugs in Pytorch 1.2.0.........
        #So we use other methods to take place of it
        MM, MM_ind = M.sum(-1).sort()
        N = (M.sum(-1)>0).sum()
        return M[MM_ind[:N]]

    def __repr__(self):
        return self.__class__.__name__

class MultiLayerCoarsening(torch.nn.Module):
    def __init__(self, dataset, hidden, num_layers=2, ratio=0.5):
        super(MultiLayerCoarsening, self).__init__()

        self.embed_block1 = DenseGCNConv(dataset.num_features, hidden)
        self.coarse_block1 = CoarsenBlock(hidden, ratio)
        self.embed_block2 = DenseGCNConv(hidden, dataset.num_features)
        # self.embed_block2 = GNNBlock(hidden, hidden, dataset.num_features)

        self.num_layers = num_layers

        self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear( hidden + dataset.num_features, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.coarse_block1.reset_parameters()
        self.embed_block2.reset_parameters()

        self.jump.reset_parameters()
        # self.lin1.reset_parameters()

    def forward(self, data, epsilon=0.01, opt_epochs=100):
        x, adj, mask = data.x, data.adj, data.mask
        batch_num_nodes = data.mask.sum(-1)
        new_adjs = [adj]
        Ss = []
        x1 = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [x1.mean(dim=1)]
        new_adj = adj
        coarse_x = x1
        # coarse_x, new_adj, S = self.coarse_block1(x1, adj, batch_num_nodes)
        # new_adjs.append(new_adj)
        # Ss.append(S)

        for i in range(self.num_layers):
            coarse_x, new_adj, S = self.coarse_block1(coarse_x, new_adj, batch_num_nodes)
            new_adjs.append(new_adj)
            Ss.append(S)
        x2 = self.embed_block2(coarse_x, new_adj, mask, add_loop=True)#should not add ReLu, otherwise x2 could be all zero.
        xs.append(x2.mean(dim=1))
        opt_loss = 0.0
        for i in range(len(x)):
            x3 = self.get_nonzero_rows(x[i])
            x4 = self.get_nonzero_rows(x2[i])
            if x3.size()[0]==0:
                continue
            if x4.size()[0]==0:
                # opt_loss += sinkhorn_loss_default(x3, x2[i], epsilon, niter=opt_epochs).float()
                continue
            opt_loss += sinkhorn_loss_default(x3, x4, epsilon, niter=opt_epochs).float()

        return xs, new_adjs, Ss, opt_loss

    def get_nonzero_rows(self, M):# M is a matrix
        # row_ind = M.sum(-1).nonzero().squeeze() #nonzero has bugs in Pytorch 1.2.0.........
        # So we use other methods to take place of it
        MM, MM_ind = torch.abs(M.sum(-1)).sort()
        N = (torch.abs(M.sum(-1)) > 0).sum()
        return M[MM_ind[:N]]

    def predict(self, xs):
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
