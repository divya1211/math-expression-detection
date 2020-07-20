import torch

class RobertaClassificationHead(torch.nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1, num_labels=2):
        super().__init__()
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
