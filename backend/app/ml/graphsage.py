"""Exact GraphSAGE class definition used by the training notebook."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class RiskGraphSAGE(nn.Module):
    def __init__(self, metadata, hidden_dim=64, embedding_dim=32):
        super().__init__()
        _, edge_types = metadata
        self.conv1=HeteroConv({e:SAGEConv((-1,-1),hidden_dim) for e in edge_types},aggr="mean")
        self.conv2=HeteroConv({e:SAGEConv((-1,-1),hidden_dim) for e in edge_types},aggr="mean")
        self.customer_projection=nn.Linear(hidden_dim,embedding_dim)
        self.classifier=nn.Linear(embedding_dim,1)
    def encode(self,x_dict,edge_index_dict):
        x_dict={k:F.relu(v) for k,v in self.conv1(x_dict,edge_index_dict).items()}
        x_dict={k:F.relu(v) for k,v in self.conv2(x_dict,edge_index_dict).items()}
        return self.customer_projection(x_dict["customer"])
    def forward(self,x_dict,edge_index_dict):
        emb=self.encode(x_dict,edge_index_dict)
        return emb,self.classifier(emb).squeeze(-1)

def load_saved_graphsage(path):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    edge_types=[("customer",f"uses_{t}",t) for t in ("device","network","address","payment")]+[(t,f"rev_uses_{t}","customer") for t in ("device","network","address","payment")]
    metadata=(["customer","device","network","address","payment"],edge_types)
    model=RiskGraphSAGE(metadata,payload.get("hidden_dim",64),payload.get("embedding_dim",32))
    x={k:torch.ones((1,2)) for k in metadata[0]}; edges={e:torch.zeros((2,0),dtype=torch.long) for e in edge_types}
    # One edge per relation initializes PyG lazy layers without needing the full dataset.
    edges={e:torch.tensor([[0],[0]]) for e in edge_types}
    model(x,edges)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
