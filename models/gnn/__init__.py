"""GNN encoders for graph representation learning"""

from .lightgcn import LightGCN
from .graphsage import GraphSAGE
from .base_gnn import BaseGNN

__all__ = ['LightGCN', 'GraphSAGE', 'BaseGNN']
