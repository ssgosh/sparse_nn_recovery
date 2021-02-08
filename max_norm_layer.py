import torch
import torch.nn as nn


# MaxNormLayer
#
# returns max(0, d - ||x - v||_\infty)
# that is, max(0, d - max(|x - v|))
#  (modulo is element-wise, second max is over the vector elements)
# That is, linear within a bounding box with hyperplanes parallel to the
# feature hyperplanes
# v and d are parameters. 
#
# Can also independently scale each dimension, with scaling vector 's':
# max(0, d - ||(x - v) / s||_\infty)
#  division is element-wise. When s_i is close to 0, |x_i - v_i| / s is large
#  very soon, and it gives a tighter squeeze along that dimension.
class MaxNormLayer(nn.Module):
    def __init__(self, size_in, size_out, lambd):
        super().__init__()
        d = torch.empty(size_out)
        nn.init.uniform_(d, 0, 10)
        self.d = nn.Parameter(d)
        #print(self.d)
        v = torch.randn(size_out, size_in)
        #v = torch.tensor([
        #    [ 0, 0, 0, ], 
        #    [ 0, 1, 0, ],
        #    [ 0, 0, 1, ],
        #    [ 1, 0, 0. ]])

        self.v = nn.Parameter(v)

        # Initially keep the scaling equal to 1
        s = torch.ones(size_out, size_in)
        self.s = nn.Parameter(s)

        self.lambd = lambd


    # Return 0 if outside the bounding box, else distance from closest face
    # x is of shape (N, size_in)
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        mn = torch.norm((x - self.v) / self.s, float('inf'), dim=-1)
        r = self.d - mn
        zeros = torch.zeros(mn.shape, device=x.device)
        return torch.max(zeros, r)
        #leaky = 0.1 * r
        #return torch.max(leaky, r)


    def get_weight_decay(self):
        return self.lambd * (torch.mean(self.d * self.d) +
                torch.mean(self.s * self.s))
        #return 0.


if __name__ == '__main__':
    layer = MaxNormLayer(3, 4)
    x = torch.tensor([[.1, .2, .3,], [.4, .5, .6]])
    #x = torch.randn(2, 3)
    print(x)
    out = layer(x)
    print(out.shape)
    print(out)
