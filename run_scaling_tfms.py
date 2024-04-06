import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import Transformer

from ezmup import Ezmup, get_coord_data, plot_coord_data


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        # The query, key, and value layers now map from 'hidden_dim' to 'hidden_dim'
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.hidden_dim
        attention = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention, V)
    
    
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        # self.layers = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim*4),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim*4, hidden_dim),
        #     nn.ReLU(),
        # )
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class MyModel(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim, hidden_dim, num_layers):
        super(MyModel, self).__init__()
        # self.layers = nn.ModuleList(
        #     [AttentionLayer(hidden_dim) for _ in range(num_layers)],
        # )
        self.embed = nn.Embedding(vocab_size, input_dim)
        self.fin = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([layer for _ in range(num_layers) for layer in [AttentionLayer(hidden_dim), MLP(hidden_dim)]])
        self.unembed = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.fin(x)
        x = nn.ReLU()(x)
        for layer in self.layers:
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.unembed(x)
        
        # x.mean(dim=[0, 1]).abs().mean()
        l1_norm = torch.abs(x).mean()
        wandb.log({f"{self.log_name}": l1_norm})        
        return x
    

# class MyModel(nn.Module):
#     def __init__(self, vocab_size, input_dim, output_dim, hidden_dim, num_layers):
#         super(MyModel, self).__init__()
#         # self.layers = nn.ModuleList(
#         #     [AttentionLayer(hidden_dim) for _ in range(num_layers)],
#         # )
#         self.embed = nn.Embedding(vocab_size, input_d)
#         self.net = nn.Linear(hidden_dim, hidden_dim)
#         self.fout = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.embed(x)
#         x = nn.ReLU()(self.net(x))
#         x = self.fout(x)
#         return x



from datasets import load_dataset
from transformers import BloomTokenizerFast
from torch.utils.data import DataLoader

BATCH_SIZE = 128

MODEL_NAME = "bigscience/bloom"
DATA_NAME = "CohereForAI/aya_dataset"
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

torch.cuda.empty_cache()


model = MyModel(vocab_size=len(tokenizer), input_dim=41, output_dim=len(tokenizer), hidden_dim=47, num_layers=1)
ref_model = MyModel(vocab_size=len(tokenizer), input_dim=41, output_dim=len(tokenizer), hidden_dim=8192, num_layers=1)
model.log_name = "mup_l1_norm"
ref_model.log_name = "ref_l1_norm"


model.to("cuda:0")
ref_model.to("cuda:0")

# model = nn.Transformer(nhead=4, num_encoder_layers=4, dim_feedforward=16)


mup_engine = Ezmup(47, model, init_std=1.0)
mup_engine.change_width_as(8192)


# def loss_fn(batch, model):
#     x, y = batch
#     y_pred = model(x)
#     return F.mse_loss(y_pred, y)


def loss_fn(batch, model):
    def loss_func(logits, targets):
        func = nn.CrossEntropyLoss()
        # logits = outputs.logits.squeeze(dim=1)
        # targets = input_ids.squeeze(dim=1)
        logits = logits[:, :-1, :].contiguous()
        return func(logits.view(-1, logits.shape[-1]), targets.view(-1))

    x, y = batch
    y_pred = model(x)
    return loss_func(y_pred, y)


mup_engine.forward = loss_fn

offset = 4*33*41
from torch.optim import Adam

mup_optim = mup_engine.get_optimizer(Adam, lr=1e-3)
ref_optim = Adam(ref_model.parameters(), lr=1e-3)

import wandb

assert sum(p.numel() for p in mup_engine.model.parameters()) == sum(p.numel() for p in ref_model.parameters())

wandb.init(
    project="ezmup",
    name="ezmup",
    group="ezmup",
    config={
        "model": "ezmup",
        "dataset": "coord",
        "optimizer": "Adam",
        "lr": 1e-3,
        "batch_size": 4,
        "num_epochs": 1000,
        "width": 8192,
        "hidden_dim": 47,
        "num_layers": 4,
        "init_std": 1.0,
        "num_params": sum(p.numel() for p in mup_engine.model.parameters()),
    },

)

mup_engine.model.train()
ref_model.train()


dataset = load_dataset(DATA_NAME)
dataset = dataset.map(
    lambda x: tokenizer(x["inputs"], padding="max_length", truncation=True, max_length=16, return_tensors="pt")
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataloaders = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)



N_EPOCHS = 100


# def record(model):
#     for name, module in model.named_modules():
#         module.register_forward_hook(
#             _record_coords(
#                 df,
#                 width,
#                 name,
#                 j,
#                 output_fdict=None,
#                 input_fdict=None,
#                 param_fdict=None,
#             ),
#         )

for epoch in range(N_EPOCHS):
    for step, batch in enumerate(dataloaders):

        # x = torch.arange(4*33*41).view(4, 33, 41).float().to("cuda:0") + ((step+1)*offset)
        # y = torch.cos(x).to("cuda:0")
        
        batch = {k: v.squeeze(dim=1).to("cuda") for k, v in batch.items()}
        targets = batch["input_ids"][:, 1:].contiguous()
        
        # mup_loss = mup_engine.forward((x, y), mup_engine.model)
        mup_loss = mup_engine.forward((batch["input_ids"], targets), mup_engine.model)
                        
        mup_loss.backward()
        mup_optim.step()
        mup_optim.zero_grad()

        # ref_loss = loss_fn((x, y), ref_model)
        ref_loss = loss_fn((batch["input_ids"], targets), ref_model)
        ref_loss.backward()
        ref_optim.step()
        ref_optim.zero_grad()
        
        print(f"step: {step}")
        wandb.log({"mup_loss": mup_loss.item(), "ref_loss": ref_loss.item()})


    # plot_coord_data(
    #     df,
    #     y="l1",
    #     save_to="contents/coord-check_new.png",
    #     suptitle=None,
    #     x="width",
    #     hue="module",
    # )


# batch = next(iter(dataloaders))
# batch = {k: v.squeeze(dim=1).to("cuda") for k, v in batch.items()}
# targets = batch["input_ids"][:, 1:].contiguous()

# df = get_coord_data(mup_engine, (batch["input_ids"], targets), n_seeds=1, n_steps=3)
# df.to_csv("contents/example.csv")


# plot_coord_data(
#     df,
#     y="l1",
#     save_to="contents/coord-check_tfms.png",
#     suptitle=None,
#     x="width",
#     hue="module",
# )
