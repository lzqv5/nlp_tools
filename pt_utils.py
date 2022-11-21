import torch
from torch import nn
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class DictDataset(torch.utils.data.Dataset):
	def __init__(self, inp, labels, device='cuda', keys=None):
		if keys is None: keys = set(inp.keys())
		self.x = {k:v.to(device) for k,v in inp.items() if k in keys}
		self.labels = labels.to(device)
	def __getitem__(self, idx):
		item = {key:val[idx] for key, val in self.x.items()}
		item['labels'] = self.labels[idx]
		return item
	def __len__(self):
		return len(self.labels)

from transformers import BertTokenizer, BertModel, BertForMaskedLM
# 获取 tokenizer
def getTokenizer(plm='hfl/chinese-roberta-wwm-ext'):
    return BertTokenizer.from_pretrained(plm)
# 获取 预训练模型
def getBertModel(plm='hfl/chinese-roberta-wwm-ext'):
    return BertModel.from_pretrained(plm)
# 获取 mlm 模型
def getBertForMaskedLM(plm='hfl/chinese-roberta-wwm-ext'):
    return BertForMaskedLM.from_pretrained(plm)

def pad_to_fixed_length(x, length, value=0):
	s = x.shape # [B, T, x, x, ...]
	lpad = length - x.shape[1]  # 计算还需要 pad 的长度
	if lpad > 0: 
        # (s[0], lpad)+s[2:] 即把 形状进行一个拼接1
		pad = torch.zeros((s[0], lpad)+s[2:], dtype=x.dtype) + value
		x = torch.cat([x, pad], dim=1)
	return x[:,:length]

class BERTClassification(nn.Module):
	def __init__(self, n_tags, cls_only=False, plm='hfl/chinese-roberta-wwm-ext') -> None:
		super().__init__()
		self.n_tags = n_tags    # BERT分类模型 - 类别数
		self.bert = BertModel.from_pretrained(plm)  # BERT分类模型 - 主体模型
		self.fc = nn.Linear(768, n_tags)    # 分类器全连接层
		self.cls_only = cls_only    # 是否只对 [CLS] 分类
	def forward(self, x, seg=None):
        # seg - 即 segmentation, 表明当前输入的句子哪些部分是第一句，哪些部分是第二句
		if seg is None: seg = torch.zeros_like(x)
		z = self.bert(x, token_type_ids=seg).last_hidden_state
		if self.cls_only: z = z[:,0]    # 取 batch 内每个句子的第一个 token, i.e. [CLS]
		out = self.fc(z)
		return out

# 计算 多个二分类 任务的 metrics
class MultiBinaryClassification():
	def __init__(self):
		self.cri = nn.BCELoss()
	def get_optim_and_sche(self, model, lr, epochs, dl_train):
		total_steps = epochs * len(dl_train)
		return get_bert_optim_and_sche(model, lr, total_steps)
	def collate_fn(self, items):
		xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
		yy = nn.utils.rnn.pad_sequence([y for x,y in items], batch_first=True)
		return xx, yy.float()
	def train_func(self, model, ditem):
		x, y = ditem[0].cuda(), ditem[1].cuda()
		out = model(x)
		loss = self.cri(out, y)
		oc = (out > 0.5).float()
		prec = (oc + y > 1.5).sum() / max(oc.sum().item(), 1)
		reca = (oc + y > 1.5).sum() / max(y.sum().item(), 1)
		f1 = 2 * prec * reca / (prec + reca)
		return {'loss': loss, 'prec': prec, 'reca': reca, 'f1':f1}
	def dev_func(self, model, dl_dev, return_str=True):
		outs = [];  ys = []
		for x, y in dl_dev:
			out = (model(x.cuda()) > 0.5).long().detach().cpu()
			outs.append(out)
			ys.append(y)
		outs = torch.cat(outs, 0)
		ys = torch.cat(ys, 0)
		accu = (outs == ys).float().mean()
		prec = (outs + ys == 2).float().sum() / outs.sum()
		reca = (outs + ys == 2).float().sum() / ys.sum()
		f1 = 2 * prec * reca / (prec + reca)
		if return_str: return f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.4f}'
		return accu, prec, reca, f1

# 后续写 train_model 时可以借鉴该函数的思路
def train_model(model, optimizer, train_dl, epochs=3, train_func=None, test_func=None, 
				scheduler=None, save_file=None, accelerator=None, epoch_len=None):
	for epoch in range(epochs):
		model.train()
		print(f'\nEpoch {epoch+1} / {epochs}:')
        # tqdm 返回关于相关迭代器的 progress bar (pbar)
		if accelerator:
			pbar = tqdm(train_dl, total=epoch_len, disable=not accelerator.is_local_main_process)
		else: 
			pbar = tqdm(train_dl, total=epoch_len)
		metricsums = {}
		iters, accloss = 0, 0
        # 这里 pbar 就等价于 包装了 tqdm 的 dataloader
        # ditem 就相当于每次从 dataloader 里提取的一个 batch
		for ditem in pbar:
			metrics = {}
            # 使用 train_func 计算模型在当前批次 ditem 内的 metrics 和 loss
			loss = train_func(model, ditem)
			if type(loss) is type({}):	# 如果 loss 是字典类型
				metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
				loss = loss['loss']
			# 更新迭代次数 以及 累计求和 loss
			iters += 1; accloss += loss
            # 清空参数的梯度
			optimizer.zero_grad()
            # 反向传播, 更新参数的梯度信息
			if accelerator: 
				accelerator.backward(loss)
			else: 
				loss.backward()
            # 进行一步优化
			optimizer.step()
            # 优化调度器
            # 根据当前的训练进度，调度器调整优化中的 config
			if scheduler:
				if accelerator is None or not accelerator.optimizer_step_was_skipped:
					scheduler.step()
			for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
			infos = {'loss': f'{accloss/iters:.4f}'}
			for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
            # 给进度条设置前缀 (增加打印信息)
			pbar.set_postfix(infos)
			if epoch_len and iters > epoch_len: break
        # 关掉进度条
		pbar.close()
		if save_file:
			if accelerator:
				accelerator.wait_for_everyone()
				unwrapped_model = accelerator.unwrap_model(model)
				accelerator.save(unwrapped_model.state_dict(), save_file)
			else:
				torch.save(model.state_dict(), save_file)
		if test_func:
			if accelerator is None or accelerator.is_local_main_process: 
				model.eval()
				test_func()

def get_bert_adamw(model, lr=2e-5, decay_factor=0.9, layerwise_lr=0):
	no_decay = ['bias', 'LayerNorm.weight']
    #* 对于 no_decay 内的参数, 不进行 weight decay, weight_decay 设置为 0;
	#* transformers.AdamW 优化器内具有属性 .param_groups, 其对应一个 list, 里面的每个元素都是一组需要优化的参数;
	#* 每组参数可以独立设置 initial_lr, weight_decay, eps, betas, correct_bias 等优化参数
	weight_decay = 0.01
	if layerwise_lr:
		numLayers = len(model.bert.encoder.layer)
		optimizer_grouped_parameters = [	# 每个 encoder layer 中的，带有 weight decay 的参数 
			{'params':[p for n, p in model.named_parameters() if f'encoder.layer.{l}.' in n and (not any(nd in n for nd in no_decay)) and p.requires_grad], 'lr':lr*(decay_factor**(numLayers-1-l)), 
			'weight_decay': weight_decay} for l in range(numLayers)
		] + [	# 每个 encoder layer 中的，不带有 weight decay 的参数 
			{'params':[p for n, p in model.named_parameters() if f'encoder.layer.{l}.' in n and any(nd in n for nd in no_decay) and p.requires_grad], 'lr':lr*(decay_factor**(numLayers-1-l)), 
			'weight_decay': 0.0} for l in range(numLayers)
		] + [
			{'params': [p for n, p in model.named_parameters() if 'encoder.layer' not in n and 'embeddings'not in n and not any(nd in n for nd in no_decay) and p.requires_grad], 'lr':lr,
			'weight_decay': weight_decay},
			{'params': [p for n, p in model.named_parameters() if 'encoder.layer' not in n and 'embeddings'not in n and any(nd in n for nd in no_decay) and p.requires_grad], 'lr':lr,
			'weight_decay': 0.0},
			{'params': [p for n, p in model.named_parameters() if 'embeddings' in n and not any(nd in n for nd in no_decay) and p.requires_grad], 'lr':lr*(decay_factor**(numLayers-1)),
			'weight_decay': weight_decay},
			{'params': [p for n, p in model.named_parameters() if 'embeddings' in n and any(nd in n for nd in no_decay) and p.requires_grad], 'lr':lr*(decay_factor**(numLayers-1)),
			'weight_decay': 0.0},
		]
		optimzer = AdamW(optimizer_grouped_parameters)
	else:
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
		]
		optimzer = AdamW(optimizer_grouped_parameters, lr=lr)
	return optimzer

def get_bert_optim_and_sche(model, lr, total_steps, decay_factor=0.9, layerwise_lr=0):
	# optimizer = get_bert_adamw(model, lr=lr)
	optimizer = get_bert_adamw(model, lr=lr, decay_factor=decay_factor, layerwise_lr=layerwise_lr)
	#* 可以通过 scheduler.get_lr() 获取当前状态下, scheduler 所控制的 optimizer 的 learning rate
	#* 可以通过 scheduler.state_dict() 获取当前状态下, scheduler 的状态信息 (e.g. base_lrs, _step_count, _last_lr, etc.)
	# scheduler = get_linear_schedule_with_warmup(optimizer, total_steps//10, total_steps)
	scheduler = get_linear_schedule_with_warmup(optimizer, total_steps//20, total_steps)
	return optimizer, scheduler

# 计算 一个 batch 内的 topk 准确率 
# output.shape = [B, C]
# target.shape = [B]
def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)	# pred 为 前k大的value所对应的 indices, 形状为 [B, K]
    pred = pred.t()	# 进行转置, 形状为 [k, B]
	# 接下来将 ground_truth 从 [B] 扩展为 [k, B], 从而来进行 match
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k / batch_size)
    return res

# 计算 一个 batch 内的 topk 命中个数
def topk_correct_num(output, target, topk=(1,5)):
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)	# pred 为 前k大的value所对应的 indices, 形状为 [B, k]
	pred = pred.t()	# [k, B]
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].sum().item()
		res.append(correct_k)
	return torch.tensor(res)