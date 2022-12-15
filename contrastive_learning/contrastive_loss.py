import torch
import torch.nn.functional as F

#* simces_loss
#* 输入 embeds 为 形如 [2*batch_size, dim] 的 embeddings
#* 其中正样本相邻排列, i.e. (0,1), (2,3), (4,5), ... 互为正样本
#* output 为 平均之后的 对比损失
def simcse_loss(embeds, tau, device):
    N, dim = embeds.shape
    idxs = torch.arange(0, N)
    idxs_1 = idxs[None,:]   # 等价于 idxs_1 = idxs.expand((1,-1)), 将 shape 从 [B]=>[1,B]
    idxs_2 = (idxs+1 - idxs % 2 * 2)[:,None]    # [B]=>[B,1]
    labels = (idxs_1.expand((N,N))==idxs_2.expand((N,N))).float()
    embeds = F.normalize(embeds, p=2 ,dim=1)    # L2-normalization
    similarities = torch.matmul(embeds,embeds.T)
    similarities = similarities - torch.eye(N).to(device)*1e12
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

#* 每个正负样本的对比使用不同的 tau
#* 后续该函数会有进一步的改进和完善 
def adaptive_tau_contrastive_loss(embeds, priorities, max_priority, device, tau_min=0.05, tau_max=0.15):
    N, dim = embeds.shape
    idxs = torch.arange(0, N)
    idxs_1 = idxs[None,:]   # 等价于 idxs_1 = idxs.expand((1,-1)), 将 shape 从 [B]=>[1,B]
    idxs_2 = (idxs+1 - idxs % 2 * 2)[:,None]    # [B]=>[B,1]
    labels_bool = (idxs_1.expand((N,N))==idxs_2.expand((N,N))).to(device)
    labels = labels_bool.float()
    embeds = F.normalize(embeds, p=2 ,dim=1)    # L2-normalization
    similarities = torch.matmul(embeds,embeds.T)
    similarities = similarities - torch.eye(N).to(device)*1e12

    taus = ((priorities[:,None]*priorities[None,:])**0.5)
    taus.masked_fill_(labels_bool==False, tau_max)
    taus[labels_bool] = tau_min + (1-taus[labels_bool]/max_priority)*(tau_max-tau_min) 

    loss = F.cross_entropy(input=similarities/taus, target=labels)
    return loss

#* FlatNCE loss
#* 其产生的梯度等价于 InfoNCE, 但梯度的分布具有更小的方差
#* 最后返回的值本质上是 InfoNCE (因为 FlatNCE 里, loss 始终是常数, variance=0)
#* 假设一个 batch 内每个 pair为 (x_i_1, x_i_2, ... , x_i_m), 那么 num_elements=m, embeddings 为这 batch_size*num_elements 个 vector 按顺序拼接起来
def flat_nce_loss(embeddings, device, batch_size=16, num_elements=2, temperature=0.05):    
    labels = torch.cat([torch.arange(batch_size) for i in range(num_elements)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # labels.shape = [num_elements*batch_size, num_elements*batch_size]
    labels = labels.to(device)  # labels_ij = 1 表示 x_i 和 x_j 形成正样本对
    # embeddings.shape = [num_elements*batch_size, dimension]
    embeddings = F.normalize(embeddings, dim=1)
    # 计算 相似度矩阵, # similarity_matrix.shape = [num_elements*batch_size, num_elements*batch_size]
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    # 丢弃 labels 和 相似度矩阵的对角元, 也即后续计算时不考虑 某样本和它自己形成的 样本对
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1) # shape=[num_elements*batch_size, num_elements*batch_size-1]
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # shape=[num_elements*batch_size, num_elements*batch_size-1]
    # 筛选出所有 正样本对 和 负样本对 所对应的 logits (similarities) 
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # shape=[num_elements*batch_size, 1]
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # shape=[num_elements*batch_size, num_elements*batch_size-2]
    # 进一步计算 logits 和 labels
    logits = (negatives - positives)/temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    v = torch.logsumexp(logits, dim=1, keepdim=True)
    loss_vec = torch.exp(v-v.detach())  # 将 FlatNCE loss 归一化到 1 (此时所有项)
    assert loss_vec.shape == (len(logits),1)
    dummy_logits = torch.cat([torch.zeros(logits.size(0),1).to(device), logits],1)
    true_loss_val = F.cross_entropy(dummy_logits, labels)
    #^ 这里需要说明, 在反向传播时, 只有 loss_vec 能起作用(因为它在计算图中)
    #^ F.cross_entropy(dummy_logits,labels).detach() 对 反向传播 没效果, 它只是反映当前模型 loss 的实际值 (因为 FlatNCE loss 恒为1, 不反应loss的变化趋势)
    loss = loss_vec.mean()-1 + F.cross_entropy(dummy_logits,labels).detach()
    return loss