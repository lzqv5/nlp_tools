import torch

def span_f1(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken):
    all_span_idxs = all_span_idxs.tolist()
    tmp_spans = []
    for i in all_span_idxs:
        tmp = []
        for j in i: tmp.append((j[0], j[1]))
        tmp_spans.append(tmp)

    pred_label = torch.max(predicts, dim=-1)[1]
    span_probs = predicts.tolist()
    pred_label = span_prune(pred_label, tmp_spans, span_probs).cuda()
    pred_label_mask = (pred_label != 0)

    all_correct = pred_label == span_label_ltoken
    all_correct = all_correct*pred_label_mask*real_span_mask_ltoken.bool()
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(pred_label != 0)
    total_golden = torch.sum(span_label_ltoken != 0)

    return torch.stack([correct_pred, total_pred, total_golden]), pred_label

def span_prune(pred_label_idx, all_span_idxs, span_probs):
    # 输出去除预测重叠span后的预测的label
    _, n_span = pred_label_idx.size(0), pred_label_idx.size(1)
    res = []

    for i, (labels, spans) in enumerate(zip(pred_label_idx, all_span_idxs)):
        span2label = {}
        span2prob = {}
        span2coordinate = {}
        tmp_spans = []
        tmp = [0] * n_span
        for j, (label, span) in enumerate(zip(labels, spans)):
            label = int(label.item())
            if label == 0: continue
            span2label[span] = label
            span2prob[span] = span_probs[i][j][label]
            span2coordinate[span] = j
            tmp_spans.append(span)
        tmp_spans = clean_overlapping_span(tmp_spans, span2prob)
        for span in tmp_spans:
            tmp[span2coordinate[span]] = span2label[span]
        res.append(tmp)
    res = torch.LongTensor(res)
    return res

def clean_overlapping_span(spans, span2prob):
    # 输入一个span列表和对应的span的预测概率的字典，清洗其中预测重叠的span
    n = len(spans)
    res = []
    if n == 0: return res
    elif n == 1: return spans

    spans.sort(key=lambda x: span2prob[x], reverse=True)
    for i in range(n):
        span1 = spans[i]
        flag = True
        for j in range(len(res)):
            span2 = res[j]
            if overlap(span1, span2):
                flag = False
                break
        if flag:
            res.append(span1)

    return res

def overlap(span1, span2):
    # 判断两个span是否有重叠
    if span1[0] < span2[0]:
        x, y = span1, span2
    else:
        x, y = span2, span1
    if x[1] < y[0]: return False
    else: return True
