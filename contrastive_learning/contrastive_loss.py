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
    labels = (idxs_1.expand((N,N))==idxs_2.expand((N,N))).float()
    embeds = F.normalize(embeds, p=2 ,dim=1)    # L2-normalization
    similarities = torch.matmul(embeds,embeds.T)
    similarities = similarities - torch.eye(N)*1e12
    similarities = similarities/tau
    loss = F.cross_entropy(input=similarities, target=labels)
    return loss

#* 参考: https://ojs.aaai.org/index.php/AAAI/article/view/21426
#* Modified frequency-aware weighted contrastive loss
#* 输入 embeds 为 形如 [2*batch_size, dim] 的 embeddings
#* 其中正样本相邻排列, i.e. (0,1), (2,3), (4,5), ... 互为正样本
#* freqs 为每个样本对应的 "频率"
#* tau 为温度系数 temperature
#* gamma 为调整 weight大小的超参数
def freq_aware_contrastive_loss(embeds, tau, freqs, device, gamma = 1.4):
    N, dim = embeds.shape
    idxs = torch.arange(0, N)
    idxs_1 = idxs[None,:]   # 等价于 idxs_1 = idxs.expand((1,-1)), 将 shape 从 [2B]=>[1,2B]
    idxs_2 = (idxs+1 - idxs % 2 * 2)[:,None]    # [2B]=>[2B,1]
    labels_bool = (idxs_1.expand((N,N))==idxs_2.expand((N,N)))
    embeds = F.normalize(embeds, p=2 ,dim=1)    # L2-normalization
    similarities = torch.matmul(embeds,embeds.T)    #^ 相似度矩阵 logits
    similarities = similarities - torch.eye(N).to(device)*1e12
    similarities = similarities/tau
    similarities_exp = torch.exp(similarities)
    numerators = similarities_exp[labels_bool]  #^ 根据 mask 取所有正样本pair [2B]
    freqs = (1-torch.log(freqs)/torch.max(torch.log(freqs)))
    weights = freqs[:,None]*freqs[None,:]*gamma #^ 计算权重矩阵 [2B, 2B]
    #^ 计算每一行的均值时, 忽略对角元
    means = torch.mean(weights[torch.eye(N)!=1].reshape(N,-1),dim=1,keepdim=True)
    weights = weights + (1-means)   #^ 使 weights 的每一行的均值为 1
    #^ 因为对角元的exp是0, 所以即使乘上weight, 对角元仍是0
    denominators = torch.sum(similarities_exp*weights, dim=1)
    loss = -torch.log(numerators/denominators)
    return torch.mean(loss)