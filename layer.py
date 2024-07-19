from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from core_qnn.quaternion_layers import QuaternionConv, QuaternionLinear
from qgnn.q4gnn import Quaternion_mul, make_quaternion_GCN_adj_mul, make_quaternion_mul

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, Ar,Ai,Aj,Ak):
        x = make_quaternion_GCN_adj_mul(Ar, Ai, Aj, Ak, x)
        return x.contiguous()

# class nconv(nn.Module):
#     def __init__(self):
#         super(nconv,self).__init__()

#     def forward(self,x, A):
#         x = torch.einsum('ncwl,wv->ncvl',(x,A))
#         return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho

#下面这个是初始的MTGNN
# class mixprop(nn.Module):
#     def __init__(self,c_in,c_out,gdep,dropout,alpha):
#         super(mixprop, self).__init__()
#         self.nconv = nconv()
#         self.mlp = linear((gdep+1)*c_in,c_out)
#         self.gdep = gdep
#         self.dropout = dropout
#         self.alpha = alpha


#     def forward(self,x,adj):
#         adj = adj + torch.eye(adj.size(0)).to(x.device)
#         d = adj.sum(1)
#         h = x
#         out = [h]
#         a = adj / d.view(-1, 1)
#         for i in range(self.gdep):
#             h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
#             out.append(h)
#         ho = torch.cat(out,dim=1)
#         ho = self.mlp(ho)
#         return ho

#下面这个是改为四元数的GNN
# class mixprop(nn.Module):
#     def __init__(self, c_in, c_out, gdep, dropout,alpha):
#         super(mixprop, self).__init__()
#         self.nconv = nconv()
#         self.c_in = c_in
#         self.c_out = c_out
#         self.gdep = gdep
#         self.dropout = dropout
#         self.alpha = alpha
#         self.weight = Parameter(torch.FloatTensor(gdep+1, self.c_in//4, self.c_out))
#         self.bias = nn.Parameter(torch.FloatTensor(c_out))
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
#     def forward(self, x, gso):
#         gso = gso + torch.eye(gso.size(0)).to(x.device)
#         d = gso.sum(1)
#         h = x
#         out = [h]
#         a = gso / d.view(-1, 1)
#         for i in range(self.gdep):
#             h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
#             out.append(h)
#         x = torch.stack(out, dim=2)
#         hamilton = make_quaternion_mul(self.weight)
#         cheb_graph_conv = torch.einsum('bikht,kij->bjht', x, hamilton)
        
#         return cheb_graph_conv


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.c_in = c_in
        self.c_out = c_out
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.weight = Parameter(torch.FloatTensor(gdep+1, self.c_in//4, self.c_out))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x, adp_r, adp_i, adp_j, adp_k):
        a_r = adp_r
        adp_i = adp_i + torch.eye(adp_i.size(0)).to(x.device)
        d = adp_i.sum(1)
        a_i = adp_i / d.view(-1, 1)
        adp_j = adp_j + torch.eye(adp_j.size(0)).to(x.device)
        d = adp_j.sum(1)
        a_j = adp_j / d.view(-1, 1)
        adp_k = adp_k + torch.eye(adp_k.size(0)).to(x.device)
        d = adp_k.sum(1)
        a_k = adp_k / d.view(-1, 1)
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a_r,a_i,a_j,a_k)
            out.append(h)
        x = torch.stack(out, dim=2)
        hamilton = make_quaternion_mul(self.weight)
        cheb_graph_conv = torch.einsum('bikht,kij->bjht', x, hamilton)
        
        return cheb_graph_conv

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, kernel_size):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [kernel_size]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(QuaternionConv(cin, cout, kernel_size=(1,kern), stride=1))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class graph_constructor_Quaternion(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor_Quaternion, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,4*dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))
        dim = nodevec1.size(1)//4
        r, i, j, k = torch.split(nodevec1, [dim, dim, dim, dim], dim=1)
        r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3
        i2 = torch.cat([i, r, -k, j], dim=1)    # 1, 0, 3, 2
        j2 = torch.cat([j, k, r, -i], dim=1)    # 2, 3, 0, 1
        k2 = torch.cat([k, -j, i, r], dim=1)    # 3, 2, 1, 0
        hamilton1 = torch.cat([r2, i2, j2, k2], dim=0)
        a =torch.mm(hamilton1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        graph_dim = adj.size(0)//4
        adj_r, adj_i, adj_j, adj_k = torch.split(adj, [graph_dim, graph_dim, graph_dim, graph_dim], dim=0)

        mask1 = torch.zeros(adj_r.size(0), adj_r.size(1)).to(self.device)
        mask1.fill_(float('0'))
        s1,t1 = (adj_r + torch.rand_like(adj_r)*0.01).topk(self.k,1)
        mask1.scatter_(1,t1,s1.fill_(1))
        adj_r = adj_r*mask1

        mask2 = torch.zeros(adj_i.size(0), adj_i.size(1)).to(self.device)
        mask2.fill_(float('0'))
        s1,t1 = (adj_i + torch.rand_like(adj_i)*0.01).topk(self.k,1)
        mask2.scatter_(1,t1,s1.fill_(1))
        adj_i = adj_i*mask2

        mask3 = torch.zeros(adj_j.size(0), adj_j.size(1)).to(self.device)
        mask3.fill_(float('0'))
        s1,t1 = (adj_j + torch.rand_like(adj_j)*0.01).topk(self.k,1)
        mask3.scatter_(1,t1,s1.fill_(1))
        adj_j = adj_j*mask3

        mask4 = torch.zeros(adj_k.size(0), adj_k.size(1)).to(self.device)
        mask4.fill_(float('0'))
        s1,t1 = (adj_k + torch.rand_like(adj_k)*0.01).topk(self.k,1)
        mask4.scatter_(1,t1,s1.fill_(1))
        adj_k = adj_k*mask4

        return adj_r, adj_i, adj_j, adj_k

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
