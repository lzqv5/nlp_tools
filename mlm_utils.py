import torch

#* group_texts 对给定文本 tokenization 并将 token ids 分成多个 chunks
#* 输入一个文本列表 (a list of strings) 和 一个 tokenizer
#* 输出一个字典, 其中包含了 input_ids 以及对应的 mlm labels
#^ 具体可参考 https://huggingface.co/course/chapter7/3?fw=pt
def group_texts(texts:list, tokenizer=None, chunk_size=512):
    assert tokenizer != None
    CLS = tokenizer.cls_token
    CLS_ID = tokenizer.cls_token_id
    SEP = tokenizer.sep_token
    SEP_ID = tokenizer.sep_token_id
    long_text = (SEP+CLS).join(texts)
    # 将所有文本拼接起来，形成用于 mlm 的 corpora
    tokenized_texts = tokenizer([long_text])
    if tokenizer.is_fast:   # 为每个 text 添加相应的 word_ids, 以标注哪些 tokens 同属一个 word
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


#* 给定已经分好块的 token ids, 进行 randomly masking.
#* chunked_texts 为已经分好块的 token ids
#^ 具体可参考 https://huggingface.co/course/chapter7/3?fw=pt
def whole_word_masking_data_collator(chunked_texts, mask_token_id, wwm_probability=0.15): 
    word_ids_mat = chunked_texts.pop("word_ids")    # it is a list
    chunked_texts = {
        key: torch.tensor(val_list) for key, val_list in chunked_texts.items()
    }

    # 二项分布, 随机概率, mask 掉 whole word
    binomial_sampler = torch.distributions.binomial.Binomial(total_count = 1,
                            probs = wwm_probability*torch.ones_like(chunked_texts["input_ids"]))
    mask = binomial_sampler.sample().long() # dtype 转换成 Long
    labels = chunked_texts["labels"] if "labels" in chunked_texts else chunked_texts["input_ids"].clone()
    new_labels = -100*torch.ones_like(labels)
    rows,cols = chunked_texts["input_ids"].shape
    # 构建 new labels 和 new input_ids
    for r in range(rows):
        for c in range(cols):
            if mask[r,c] == 1:  # masked
                new_labels[r,c] = labels[r,c]
                chunked_texts["input_ids"][r,c] = mask_token_id
            elif word_ids_mat[r][c]!=None:   # mask[r,c] == 0 & it is not special token
                if c>0 and word_ids_mat[r][c]==word_ids_mat[r][c-1]:  # 和前一个 token 同属一个 word
                    if mask[r,c-1] == 1:  # 前一个 token 被 mask 掉了
                        mask[r,c] = 1   # 当前 token 也需要被 mask 掉
                        new_labels[r,c] = labels[r,c]
                        chunked_texts["input_ids"][r,c] = mask_token_id
                elif c==0 and r>0 and word_ids_mat[r][c]==word_ids_mat[r-1][-1]:
                    if mask[r-1,-1] == 1:  # 前一个 token 被 mask 掉了
                        mask[r,c] = 1   # 当前 token 也需要被 mask 掉
                        new_labels[r,c] = labels[r,c]
                        chunked_texts["input_ids"][r,c] = mask_token_id
    # new labels 也可以使用 torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100) 来构建
    chunked_texts["labels"] = new_labels

    return chunked_texts