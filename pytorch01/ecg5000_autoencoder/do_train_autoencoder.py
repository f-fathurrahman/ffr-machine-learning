from arff2pandas import a2p
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from sklearn.model_selection import train_test_split
import copy

device = "cpu"
#device = "cuda"

sns.set(font_scale=1.2)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

with open('DATASET/ECG5000_TRAIN.arff') as f:
    train_data = a2p.load(f)

with open('DATASET/ECG5000_TEST.arff') as f:
    test_data = a2p.load(f)

df = train_data.append(test_data)
df = df.sample(frac=1.0) # shuffle?
print(df.shape)

CLASS_NORMAL = 1
class_names = ["Normal", "R on T", "PVC", "SP", "UB"]

# rename the last column to target , so its easier to reference it
new_columns = list(df.columns) # convert to list
new_columns[-1] = "target"
df.columns = new_columns

# how many examples for each heartbeat class do we have:
print( df.target.value_counts() )

#ax = sns.countplot(df.target)
#ax.set_xticklabels(class_names)
#plt.savefig("IMG_target_sample.pdf")


def plot_time_series_class(data, class_name, ax, Nsteps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(Nsteps).mean()
    path_deviation = 2*time_series_df.rolling(Nsteps).std()

    underline = (smooth_path - path_deviation)[0]
    overline = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path)
    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
        path_deviation.index,
        underline,
        overline,
        alpha=.125
    )
    ax.set_title(class_name)

classes = df.target.unique()
fig, axs = plt.subplots(
    nrows=len(classes) // 3 + 1,
    ncols=3,sharey=True,
    figsize=(14, 8)
)
for i, cls in enumerate(classes):
    ax = axs.flat[i]
    data = df[df.target == cls]\
        .drop(labels='target', axis=1)\
        .mean(axis=0)\
        .to_numpy()
    #plot_time_series_class(data, class_names[i], ax)

#fig.delaxes(axs.flat[-1])
#fig.tight_layout()
#plt.savefig("IMG_time_series_class.pdf")


# Data preprocessing

# Get all normal heartbeats and drop the target (class) column
normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
print(normal_df.shape)

# We’ll merge all other classes and mark them as anomalies
anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)
print(anomaly_df.shape)

# Split the normal examples into train, validation and test sets
train_df, val_df = train_test_split(
    normal_df,
    test_size=0.15,
    random_state=RANDOM_SEED
)
val_df, test_df = train_test_split(
    val_df,
    test_size=0.33,
    random_state=RANDOM_SEED
)

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

# We need to convert our examples into tensors, so we can use them to train our Autoencoder.
# Let’s write a helper function for that:

def create_dataset(df):
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features

train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):
    
    def __init__(self, seq_len, input_dim=64, n_features=1):
        
        super(Decoder, self).__init__()
        
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#model = RecurrentAutoencoder(seq_len, n_features, 128)
model = RecurrentAutoencoder(seq_len, n_features, 32)
model = model.to(device)

def train_model(model, train_dataset, val_dataset, n_epochs):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #criterion = nn.L1Loss(reduction='sum').to(device)
    criterion = nn.MSELoss().to(device)
    history = dict(train=[], val=[])
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    
    for epoch in range(1, n_epochs + 1):
        print("Enter epoch ", epoch)
        model = model.train()
        train_losses = []
        for i,seq_true in enumerate(train_dataset):
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            print("seq_true i = %d is done" % (i))

        val_losses = []
        model = model.eval()

        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    
    model.load_state_dict(best_model_wts)
    return model.eval(), history

model, history = train_model(
    model,
    train_dataset,
    val_dataset,
    n_epochs=150
)

