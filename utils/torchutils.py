from torch.nn import functional as F


def compute_probs_tensor(output, targets):
   return F.softmax(output, dim=1)[:, targets]

