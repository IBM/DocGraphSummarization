from torch_geometric.nn.models.graph_unet import GraphUNet

"""
    Class that extends the GraphUNet model. 

    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/graph_unet.html#GraphUNet
"""
class UNetCoarsening(GraphUNet):

    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(UNetCoarsening, self).__init__(in_channels, hidden_channels, 
                                            out_channels, depth, pool_ratios=pool_ratios, 
                                            sum_res=sum_res, act=act)
        
        
    """
        Overridden forward function. I need to implement the behavior
        for returning the coarsened sentence indices.
    """
    def forward(self, x, edge_index, batch=None):
        pass
        
    
    
