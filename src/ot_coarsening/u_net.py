from typing import Union, Optional, Callable
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import spspmm
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import FastRGCNConv, RGCNConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops, softmax)
from torch_geometric.utils.repeat import repeat

def topk(score, indices, ratio, batch, min_score=None, tol=1e-7, num_output_sentences=None):
    if indices is None:
        indices = torch.arange(score.shape[0])
    assert score.shape == indices.shape

    sentence_ratio = ratio
    word_ratio = ratio
    num_nodes = scatter_add(batch.new_ones(score.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
   
    index  = torch.arange(batch.size(0), dtype=torch.long, device=score.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = score.new_full((batch_size * max_num_nodes, ),
                         torch.finfo(score.dtype).min)
    dense_x[index] = score
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    if not num_output_sentences is None:
        k = torch.ones(batch_size) * num_output_sentences
    elif isinstance(ratio, int):
        k = num_nodes.new_full((num_nodes.size(0), ), ratio)
        k = torch.min(k, num_nodes)
    else:
        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

    mask = [
        torch.arange(k[i], dtype=torch.long, device=score.device) +
        i * max_num_nodes for i in range(batch_size)
    ]
    mask = torch.cat(mask, dim=0)
    perm = perm[mask]
    indices_perm = indices[perm]
    return indices_perm

def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i
    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

def convert_y_to_onehot(y, num_sentences):
    """
        Converts the y vector to a one hot encoded vector
    """
    batch_size = num_sentences.shape[0]
    onehot_vectors = []
    for index in range(batch_size):
        current_num_sentences = num_sentences[index].int()
        current_y = y[index]
        current_y = current_y[torch.nonzero(current_y > 0)].squeeze().long()
        base_vector = torch.zeros(current_num_sentences).to(y.device)
        base_vector[current_y] = 1
        onehot_vectors.append(base_vector)

    return onehot_vectors

class TopKPooling(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None, multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh):
        super(TopKPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data = torch.rand(1, self.in_channels, device=self.device)

    def __repr__(self):
        return '{}({}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None, num_sentences=None, num_output_sentences=3, is_last_layer=True):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)
        # get seperate permutation for sentences and the rest (word and document nodes)
        print("score requires grad")
        print(score.requires_grad)
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0).to(x.device)
        num_sentence_nodes = num_sentences.to(x.device) # the number of sentence nodes in each batch graph 
        cumulative_num_nodes  = torch.cat(
                [num_nodes.new_zeros(1),
                num_nodes.cumsum(dim=0)[:-1]]
            , dim=0).to(x.device)
        print(cumulative_num_nodes.requires_grad)
        batch_size = num_nodes.size(0)
        # get the sentence perm
        sentence_indices = [torch.arange(num_sentence_nodes[i], device=x.device) + cumulative_num_nodes[i] for i in range(batch_size)]
        sentence_indices = torch.cat(sentence_indices).long()
        print("sentence indices req")
        print(sentence_indices.requires_grad)
        sentence_scores = score[sentence_indices]
        sentence_batch = [torch.ones(num_sentence_nodes[i]) * i for i in range(len(num_sentence_nodes))]
        sentence_batch = torch.cat(sentence_batch).long().to(x.device)
        print("sentence batch requires grad")
        print(sentence_batch.requires_grad)
        print("sentence scores requires grad")
        print(sentence_scores.requires_grad) 
        sentence_perm = topk(sentence_scores, sentence_indices, self.ratio, sentence_batch, self.min_score, num_output_sentences=num_output_sentences)
        # get the other perm
        num_other_nodes = num_nodes - num_sentence_nodes
        other_indices = [torch.arange(num_other_nodes[i], device=x.device) + cumulative_num_nodes[i] + num_sentence_nodes[i] for i in range(batch_size)]
        other_indices = torch.cat(other_indices).long().to(x.device)
        other_scores = score[other_indices]
        other_batch = [torch.ones(num_other_nodes[i]) * i for i in range(len(num_other_nodes))]
        other_batch = torch.cat(other_batch).long().to(x.device)
        other_perm = topk(other_scores, other_indices, self.ratio, other_batch, self.min_score)
        # combine the other and sentence perms
        perm = torch.cat((sentence_perm, other_perm)) 
        # process the batch
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        print(perm.requires_grad)

        return x, edge_index, edge_attr, batch, perm, score[perm]

class GraphUNetCoarsening(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        activation (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, activation=F.relu, supervised=False):
        super(GraphUNetCoarsening, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.supervised = supervised
        self.activation = activation
        self.sum_res = sum_res
        self.loss = torch.nn.MSELoss()
        self.supervised_loss = torch.nn.BCELoss()

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GATConv(in_channels, channels, heads=1))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GATConv(channels, channels, heads=1))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GATConv(in_channels, channels, heads=1))
        self.up_convs.append(GATConv(in_channels, out_channels, heads=1))
        #self.output_conv = GCNConv(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)
        
    """
        Computes the supervised loss
    """
    def compute_supervised_loss(self, data, coarse_indices, batch):
        max_summary_length = 10
        num_sentences = data.num_sentences
        y = data.y
        y = torch.reshape(y, (-1, max_summary_length))
        onehot_y = convert_y_to_onehot(y, num_sentences)
        batch_size = batch.max() + 1
        # go through each element
        cumulative_loss = 0.0
        for index in range(batch_size):
            label_onehot = onehot_y[index]
            current_coarse = coarse_indices[torch.nonzero(batch == index)].squeeze()
            current_num_sentences = [num_sentences[index].int()]
            sentence_indices = torch.nonzero(current_coarse < current_num_sentences[0]).squeeze()
            sentence_indices = current_coarse[sentence_indices][None, :]
            onehot_predicted = convert_y_to_onehot(sentence_indices, torch.stack(current_num_sentences))[0]
            # compute the supervised loss
            current_loss = self.supervised_loss(onehot_predicted[None, :], label_onehot[None, :])
            cumulative_loss += current_loss

        mean_loss = cumulative_loss / batch_size

        return mean_loss

    """
        Computes a basic MSE between an input graph and a 
        reconstructed graph. 
    """
    def compute_loss(self, original_x, reconstructed_x, num_sentences=None, sentence_only=False, batch=None):
        if not num_sentences is None and sentence_only:
            x = original_x
            # computes the loss solely based on the sentence nodes
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0).to(x.device)
            num_sentence_nodes = num_sentences.to(x.device) # the number of sentence nodes in each batch graph 
            cumulative_num_nodes  = torch.cat(
                    [num_nodes.new_zeros(1),
                    num_nodes.cumsum(dim=0)[:-1]]
                , dim=0).to(x.device)
            batch_size = num_nodes.size(0)
            # get the sentence perm
            sentence_indices = [torch.arange(num_sentence_nodes[i], device=x.device) + cumulative_num_nodes[i] for i in range(batch_size)]
            sentence_indices = torch.cat(sentence_indices).long()
            # select the correct nodes
            original_sentence_x = original_x[sentence_indices]
            reconstructed_sentence_x = reconstructed_x[sentence_indices]
            loss = self.loss(original_sentence_x, reconstructed_sentence_x)
        else:
            loss = self.loss(original_x, reconstructed_x)
        return loss

    """
        Performs forward pass 
    """
    def forward(self, data, num_output_sentences=3, batch=None):
        if batch is None:
            batch = data.batch
        original_batch = batch.clone()
        # unpack the data
        x, y, edge_index, edge_attr, num_sentences = data.x, data.y, data.edge_index, data.edge_attr, data.num_sentences
        # convert edge_attr to edge_weight
        edge_weight = edge_attr.squeeze().float()
        edge_index = edge_index.long()
        x = x.float()
        # save a copy of the input x
        original_x = x.clone()
        # perform first layer down convolution 
        x = self.down_convs[0](x, edge_index)
        x = self.activation(x)
        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []
        last_batch = None
        # perform downward pooling and convolutions
        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            is_last_layer = not i < self.depth
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                                                             x,
                                                             edge_index,
                                                             edge_attr=edge_weight, 
                                                             num_sentences=num_sentences, 
                                                             num_output_sentences=num_output_sentences, 
                                                             batch=batch,
                                                             is_last_layer=is_last_layer)

            x = self.down_convs[i](x, edge_index)
            x = self.activation(x)
            print("perm requires grad")
            print(perm.requires_grad)
            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]
            last_batch = batch
        coarsened_indices = perms[-1]
        # perform upward pooling and convolutions
        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index)
            x = self.activation(x) if i < self.depth - 1 else x
        # perform GCN to map the output graph wiht zero embeddings 
        # to an attempted reconstruction of the input graph
        
        # compute the loss
        reconstructed_x = x
        if self.supervised:
            loss = self.compute_supervised_loss(data, coarsened_indices, last_batch)
        else:
            loss = self.compute_loss(original_x, reconstructed_x, num_sentences=num_sentences, batch=original_batch)
        # figure out the coarsened indices
        S = None
        return x, edge_index, edge_attr, S, loss, coarsened_indices

