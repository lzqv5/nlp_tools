import torch
import torch.nn.functional as F

#* simces_loss
#* 输入 embeds 为 形如 [2*batch_size, dim] 的 embeddings
#* 其中正样本相邻排列, i.e. (0,1), (2,3), (4,5), ... 互为正样本
#* output 为 平均之后的 对比损失
def simcse_loss(embeds, tau):
    N, dim = embeds.shape
    idxs = torch.arange(0, N)
    idxs_1 = idxs[None,:]   # 等价于 idxs_1 = idxs.expand((1,-1)), 将 shape 从 [B]=>[1,B]
    idxs_2 = (idxs+1 - idxs % 2 * 2)[:,None]    # [B]=>[B,1]
    labels = idxs_1.expand((N,N))==idxs_2.expand((N,N))
    embeds = torch.normal
    embeds = F.normalize(embeds, p=2 ,dim=1)    # L2-normalization
    similarities = torch.matmul(embeds,embeds.T)
    similarities = similarities - torch.eye(N)*1e12
    similarities = similarities/tau
    loss = F.cross_entropy(input=similarities, target=labels)
    return loss