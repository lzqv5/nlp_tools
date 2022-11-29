import jieba
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

#* group_texts 对给定文本 tokenization 并将 token ids 分成多个 chunks
#* 输入一个文本列表 (a list of strings) 和 一个 tokenizer
#* 输出一个字典, 其中包含了 input_ids 以及对应的 mlm labels
#^ 具体可参考 https://huggingface.co/course/chapter7/3?fw=pt
def group_texts_fast(texts:list, tokenizer=None, chunk_size=512):
    assert tokenizer != None and tokenizer.is_fast
    CLS = tokenizer.cls_token
    CLS_ID = tokenizer.cls_token_id
    SEP = tokenizer.sep_token
    SEP_ID = tokenizer.sep_token_id
    long_text = (SEP+CLS).join(texts)
    # 将所有文本拼接起来，形成用于 mlm 的 corpora
    tokenized_texts = tokenizer([long_text])
    # 为每个 text 添加相应的 word_ids, 以标注哪些 tokens 同属一个 word
    tokenized_texts["word_ids"] = [tokenized_texts.word_ids(i) for i in range(len(tokenized_texts["input_ids"]))]
    for key,val in tokenized_texts.items():
        # 去除不必要的嵌套索引
        tokenized_texts[key] = val[0]
    # 将所有 [CLS], [SEP] 对应的 word_id 替换为 None
    for idx, token_id in  enumerate(tokenized_texts["input_ids"]):
        if token_id in [CLS_ID, SEP_ID]:
            tokenized_texts["word_ids"][idx] = None
    # 计算总 token 数 (以便后续将 corpora 进行分段操作)
    total_length = len(tokenized_texts["input_ids"])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len - 分块化
    result = {
        key: [val_list[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for key, val_list in tokenized_texts.items()
    }
    # Create a new labels column - 这将作为后续进行 training 的 mini-batch 对应的 labels
    result["labels"] = result["input_ids"].copy()
    return result
    
#* 省略了chunk的步骤,直接每一句话当成一个chunk
#* 输入一个文本列表 (a list of strings) 和 一个 tokenizer, 以及填充的最大长度
#* 输出一个字典, 其中包含了 input_ids 以及对应的 word_ids (方便后续做 wwm for alphabetics)
def fast_tokenize_texts_for_alphabetic_wwm(texts:list, tokenizer=None, max_length=512):
    assert type(texts) == list and tokenizer != None and tokenizer.is_fast
    tokenized_texts = tokenizer(texts,return_tensors='pt',truncation=True,max_length=max_length, padding='max_length')
    tokenized_texts["word_ids"] = [tokenized_texts.word_ids(i) for i in range(len(tokenized_texts["input_ids"]))]
    return tokenized_texts

#* 该 wwm 主要是用于以字母letters为单位的西文做 整词掩蔽 
#* 对于中文这种以 字 为基本单位的语言，需要额外使用 分词工具来完成 wwm
#* 给定已经分好块的 token ids (或者每一行 token ids 都对应一句话, 但每一行都填充到了同样长度), 进行 randomly masking.
#* Note: 一个 [MASK] 对应一个 token, i.e. [MASK]不具备拓展性 (e.g. RoBERTa, BERT 这种非生成类的模型) 
#^ 具体可参考 https://huggingface.co/course/chapter7/3?fw=pt
def whole_word_masking_alphabetic_data_collator(chunked_texts, mask_token_id, wwm_probability=0.15): 
    word_ids_mat = chunked_texts.pop("word_ids")    # it is a list
    chunked_texts = {
        key: torch.tensor(val_list) for key, val_list in chunked_texts.items()
    }

    # 二项分布, 随机概率, mask 掉 whole word
    binomial_sampler = torch.distributions.binomial.Binomial(total_count = 1,
                            probs = wwm_probability*torch.ones_like(chunked_texts["input_ids"]))
    mask = binomial_sampler.sample().long() # dtype 转换成 Long
    mask_type = torch.rand(mask.shape)
    labels = chunked_texts["labels"] if "labels" in chunked_texts else chunked_texts["input_ids"].clone()
    new_labels = -100*torch.ones_like(labels)
    rows,cols = chunked_texts["input_ids"].shape
    # 构建 new labels 和 new input_ids
    for r in range(rows):
        for c in range(cols):
            if mask[r,c] == 1:  # masked
                new_labels[r,c] = labels[r,c]
                if mask_type[r,c] > 0.9:    # unchange
                    pass
                elif mask_type[r,c] < 0.1:  # random id
                    chunked_texts["input_ids"][r,c] = np.random.choice(vocab_size-1, size=1)[0]+1
                else:   # mask token
                    chunked_texts["input_ids"][r,c] = mask_token_id
            elif word_ids_mat[r][c]!=None:   # mask[r,c] == 0 & it is not special token
                if c>0 and word_ids_mat[r][c]==word_ids_mat[r][c-1]:  # 和前一个 token 同属一个 word
                    if mask[r,c-1] == 1:  # 前一个 token 被 mask 掉了
                        mask[r,c] = 1   # 当前 token 也需要被 mask 掉
                        mask_type[r,c] = mask_type[r,c-1]
                        new_labels[r,c] = labels[r,c]
                        if mask_type[r,c] > 0.9:    # unchange
                            pass
                        elif mask_type[r,c] < 0.1:  # random id
                            chunked_texts["input_ids"][r,c] = np.random.choice(vocab_size-1, size=1)[0]+1
                        else:   # mask token
                            chunked_texts["input_ids"][r,c] = mask_token_id
                elif c==0 and r>0 and word_ids_mat[r][c]==word_ids_mat[r-1][-1]:
                    if mask[r-1,-1] == 1:  # 前一个 token 被 mask 掉了
                        mask[r,c] = 1   # 当前 token 也需要被 mask 掉
                        mask_type[r,c] = mask_type[r-1,-1]
                        new_labels[r,c] = labels[r,c]
                        if mask_type[r,c] > 0.9:    # unchange
                            pass
                        elif mask_type[r,c] < 0.1:  # random id
                            chunked_texts["input_ids"][r,c] = np.random.choice(vocab_size-1, size=1)[0]+1
                        else:   # mask token
                            chunked_texts["input_ids"][r,c] = mask_token_id
    # new labels 也可以使用 torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100) 来构建
    chunked_texts["labels"] = new_labels

    return chunked_texts


#* input 为已经 MASK 好的文本列表
#* 输出为 各个 [MASK] 所对应 token 的 top-k predictions
def mlm_topk_predict(masked_texts:list, tokenizer, model, k=5, language='zh', device="cuda"):
    assert type(masked_texts) == list
    model.eval()
    model = model.to(device)
    inputs = tokenizer(masked_texts, padding="longest", max_length=512, truncation=True, return_tensors="pt")
    total_mask = torch.where(inputs["input_ids"]==tokenizer.mask_token_id, True, False)
    # process whole data batch by batch
    dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], total_mask)
    sampler = torch.utils.data.SequentialSampler(dataset)
    params = {"batch_size": 16, "sampler": sampler}
    dataloader = torch.utils.data.DataLoader(dataset, **params)
    word_predictions = []
    token_predictions = []
    prob_predictions = []
    refreshed_texts = []
    for batch in dataloader:
        mask = batch[3].to(device=device)
        batch_input_ids = batch[0].to(device=device)
        try:
            mask_token_probs = F.softmax(model(batch_input_ids, batch[1].to(device=device), 
                                            batch[2].to(device=device)).logits[mask],dim=1)
        except:
            print(batch_input_ids.shape)
            continue
        topk_probs, topk_indices = mask_token_probs.topk(k,largest=True)   # topk_indices.shape = [#mask_tokens, k]
        num_mask_tokens_for_each_text = mask.sum(dim=1) # shape = [#texts]
        num_processed_tokens = 0 
        for idx in range(num_mask_tokens_for_each_text.shape[0]):
            num = num_mask_tokens_for_each_text[idx]
            word_prediction = [ [tokenizer.decode(token_id) for token_id in cur_topk_indices] for cur_topk_indices in topk_indices[num_processed_tokens:num_processed_tokens+num] ]
            token_prediction = [ [token_id.item() for token_id in cur_topk_indices] for cur_topk_indices in topk_indices[num_processed_tokens:num_processed_tokens+num] ]
            prob_prediction = [ [prob.item() for prob in cur_topk_probs] for cur_topk_probs in topk_probs[num_processed_tokens:num_processed_tokens+num] ]
            word_predictions.append(word_prediction)
            token_predictions.append(token_prediction)
            prob_predictions.append(prob_prediction)
            num_processed_tokens += num
        batch_input_ids[mask] = topk_indices[:,0]
        new_texts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        if language=="zh":  # 去除解码后的中文文本内的多余空格
            new_texts = [text.replace(" ","") for text in new_texts]
        for new_text in new_texts:
            refreshed_texts.append(new_text)
    return {
        'words': word_predictions,
        'tokens': token_predictions,
        'probs': prob_predictions,
        'texts':refreshed_texts,   
    }

#^ 参考: https://github.com/ZhuiyiTechnology/GAU-alpha/blob/main/train.py
#* 输入一个中文文本, 利用分词工具将整个句子其划分为数个词(一个词对应一个或多个字)
#* 输出是该文本对应的经过 whole word masking 之后的 input_ids, 以及其对应的 labels
#* -100 代表计算 cross-entropy loss 时，忽略该项
#* Note: 一个 [MASK] 对应一个 token, i.e. [MASK]不具备拓展性 (e.g. RoBERTa, BERT 这种非生成类的模型) 
def wwm_mlm_zh_encode(text, tokenizer, max_length=None, wwm_probability=0.15):
    assert max_length==None or type(max_length)==int
    special_tokens = {tokenizer.mask_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
    mask_token_id = tokenizer.mask_token_id
    words = jieba.lcut(text)
    rands = np.random.rand(len(words))
    source, target = [], []
    for r,w in zip(rands, words):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if w in special_tokens: # 说明 text 内部可能包含 special tokens
            source.extend(ids)  # ids = [special_token_id]
            target.extend([-100])
        elif r < wwm_probability*0.8:
            source.extend([mask_token_id]*len(ids))
            target.extend(ids)
        elif r < wwm_probability*0.9:
            source.extend(ids) 
            target.extend(ids)
        elif r < wwm_probability:
            source.extend(
                np.random.choice(tokenizer.vocab_size-1, size=len(ids))+1
            )
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([-100]*len(ids))
    if max_length is None:
        return [tokenizer.cls_token_id] + source + [tokenizer.sep_token_id],[-100] + target + [-100]
    else:
        return [tokenizer.cls_token_id] + source[:max_length-2] + [tokenizer.sep_token_id],\
                [-100] + target[:max_length-2] + [-100] 

#* 主要针对中文文本做 whole word masking
#* 输入是中文文本列表 (相当于每个中文文本自己就相当于一个 chunk), 以及相应的 tokenizer, 填充最大长度, 以及掩盖率
#* 输出是各个中文文本对应的 input_ids, attention_mask 和 labels
def whole_word_masking_zh(texts:list, tokenizer, max_length=None, wwm_probability=0.15):
    tokenized_texts = {
        'input_ids': [0]*len(texts),
        'labels': [0]*len(texts),
        # 'attention_mask': [0]*len(texts),
    }
    for idx,text in enumerate(texts):
        source, target = wwm_mlm_zh_encode(text, tokenizer, max_length=max_length, wwm_probability=wwm_probability)
        tokenized_texts['input_ids'][idx] = torch.tensor(source)
        tokenized_texts['labels'][idx] = torch.tensor(target)
    tokenized_texts['input_ids'] = pad_sequence(tokenized_texts['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
    tokenized_texts['labels'] = pad_sequence(tokenized_texts['labels'], batch_first=True, padding_value=-100)
    tokenized_texts['attention_mask'] = (tokenized_texts['input_ids'] != tokenizer.pad_token_id).long()
    return tokenized_texts


#* input 为带有 emojis 的文本列表 (emoji 用 [...] 表示, 中括号[]里的内容代表表情内容, 整个[...]表示一个表情)
#* 输出为 各个 将各个emoji转换成对应[MAKS]的文本列表
def convert_emoji_to_mask(texts:list, tokenizer, language='zh'):
    left_token_id = tokenizer.encode("[", add_special_tokens=False)[0]
    right_token_id = tokenizer.encode("]", add_special_tokens=False)[0]
    tokenized_texts_tokenIds = tokenizer(texts, add_special_tokens=False)["input_ids"]
    MASK_ID = tokenizer.mask_token_id
    for r,token_ids in enumerate(tokenized_texts_tokenIds):
        on = False
        for c, token_id in enumerate(token_ids):
            if token_id == left_token_id:   # 当前是左括号 [
                on = True   # 表示当前位置处于括号中间(表情)
            elif token_id == right_token_id:    # 当前是右括号 ] 
                on = False  # 表示当前位置处于括号外(非表情)
            else:   # 当前是其他字符
                if on:
                    tokenized_texts_tokenIds[r][c] = MASK_ID
    new_texts = tokenizer.batch_decode(tokenized_texts_tokenIds)
    if language=="zh":  # 去除解码后的中文文本内的多余空格
        new_texts = [text.replace(" ","") for text in new_texts]
    return new_texts