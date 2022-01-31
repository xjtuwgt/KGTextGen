from torch import nn
from codes.gnn_layers import GATConv


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 topk,
                 topp,
                 sparse_mode,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        assert sparse_mode in {'top_k', 'top_p', 'no_sparse'}
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_feats=in_dim, out_feats=num_hidden, num_heads=heads[0],
            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
            residual=False, activation=self.activation, top_p=topp, top_k=topk, sparse_mode=sparse_mode))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                in_feats=num_hidden * heads[l - 1], out_feats=num_hidden, num_heads=heads[l],
                feat_drop=feat_drop, attn_drop=attn_drop, top_k=topk, top_p=topp, sparse_mode=sparse_mode,
                negative_slope=negative_slope, residual=residual, activation=self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            in_feats=num_hidden * heads[-2], out_feats=num_classes, num_heads=heads[-1],
            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, top_k=topk,
            top_p=topp, sparse_mode=sparse_mode,
            residual=residual, activation=None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
