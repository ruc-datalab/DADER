import torch

def KL_divergence(p,q):
    d = p/q
    d = torch.log(d)
    d = p*d
    return torch.sum(d)


def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*KL_divergence(p, M)+0.5*KL_divergence(q, M)
