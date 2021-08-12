#%% Imports
import torch
torch.manual_seed(44)
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import DataLoader, dataloader
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.nn import global_mean_pool
from custom_dataset import ToxDataset
from tqdm import tqdm
from IPython.display import clear_output
import plotly.express as px
# %%
class GCN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # obtain node embedding
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # readout layer
        x = global_mean_pool(x, batch)
        # final classifier
        X = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trainset = ToxDataset(root="data/", filename='toxicity-train-resampled.csv')
testset = ToxDataset(root="data/", filename='toxicity-test.csv', test=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

model = GCN(trainset.num_node_features, 128, 1).to(device)
print(model)
# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    running_loss = 0.0
    step = 0
    for _, data in enumerate(tqdm(trainloader)):
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(torch.squeeze(out), data.y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
    return running_loss/step

# %%
all_loss = []
for epoch in range(500):
    train_loss = train()
    all_loss.append(train_loss)
print("Done!")

