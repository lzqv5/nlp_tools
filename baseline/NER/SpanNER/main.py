import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from model import SpanNER
from dataset import SpanNERDataset, load_examples, collate_fn
from utils.metric import span_f1
from utils.save import predictions_save

def get_args():
    # 准备参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=4396, type=int)
    parser.add_argument(
        '--model_path',
        default='/home/jgc22/bert-large-uncased',
        type=str
    )

    parser.add_argument(
        '--data_path',
        default='/home/jgc22/OOV/data/WNUT2017/',
        type=str
    )

    parser.add_argument(
        '--output_path',
        default='/home/jgc22/OOV/output/WNUT2017/',
        type=str
    )

    parser.add_argument(
        '--max_seq_len',
        default=128,
        type=int
    )

    parser.add_argument(
        '--max_span_len',
        default=4,
        type=int
    )

    parser.add_argument(
        '--span_len_dim',
        default=50,
        type=int
    )

    parser.add_argument('--bert_lr', default=1e-5, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    with open(os.path.join(args.data_path, './labels.txt'), 'r') as f:
        labels = f.read().splitlines()
    if 'O' not in labels: labels.append('O')
    args.labels = labels

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return args

def set_seed(seed):
    # 设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_optimizer_and_scheduler(args, model, training_steps):
    no_decay = ['bias', 'LayerNorm.weight']
    bert_params = eval('model.bert').named_parameters()
    other_params = list(model.span_layer.named_parameters()) + list(model.classifier.named_parameters())

    optimizer_params = [
        {
            'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.lr
        },
        {
            'params': [p for n, p in other_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.lr
        },
        {
            'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.bert_lr
        },
        {
            'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.bert_lr
        },
    ]

    optimizer = AdamW(optimizer_params, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=training_steps
    )
    return optimizer, scheduler

def train(args, model, tokenizer):
    train_examples = load_examples(args, mode='train', tokenizer=tokenizer)
    training_steps = (len(train_examples) - 1 / args.epoch + 1) * args.epoch
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, training_steps)

    global_step = 0
    best_epoch, best_score = 0, 0.0
    tr_loss = 0.0
    model.zero_grad()
    epoch_num = 0

    for _ in range(args.epoch):
        epoch_num += 1
        print('Epoch {}:'.format(epoch_num))
        train_dataset = SpanNERDataset(
            train_examples, args, tokenizer
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, collate_fn=collate_fn
        )

        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        # train
        for _, batch in enumerate(epoch_iterator):
            model.train()
            feature_tensors = {k: v.to(args.device) for k, v in batch[0].items()}
            _, loss = model(feature_tensors)

            loss.backward()

            tr_loss += loss.item()

            epoch_iterator.set_description(
                'loss: {}'.format(round(loss.item(), 5))
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
        # evaluate
        res, _ = evaluate(args, model, tokenizer)
        print('Evaluate result: f1 {} \t precision {} \t recall {}'.format(res['span_f1'], res['span_precision'], res['span_recall']))
        if best_score < res['span_f1']:
            print('Saving the best checkpoint...')
            best_score = res['span_f1']
            best_epoch = epoch_num

            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            model_to_save = (model.module if hasattr(model, 'module') else model)
            model_to_save.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)

    print('Best Epoch:{}, Best score:{:.5f}'.format(best_epoch, best_score))
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, mode='dev'):
    eval_examples = load_examples(args, mode=mode, tokenizer=tokenizer)
    eval_dataset = SpanNERDataset(eval_examples, args, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2*args.batch_size, collate_fn=collate_fn)

    model.eval()
    outputs = []
    for batch in tqdm(eval_dataloader, desc='Evaluating' if mode=='dev' else 'Testing'):
        feature_tensors = {k: v.to(args.device) for k, v in batch[0].items()}
        with torch.no_grad():
            output, _ = model(feature_tensors)

            span_f1s, pred_label_idx = span_f1(
                feature_tensors['span_word_idxes'],
                output[0],
                feature_tensors['span_labels'],
                feature_tensors['span_masks']
            )
            outputs.append({
                'span_f1': span_f1s,
                'pred_label_idx': pred_label_idx,
                'all_span_idx': feature_tensors['span_word_idxes'],
                'span_label_ltoken': feature_tensors['span_labels']
            })

    all_counts = torch.stack([x[f'span_f1'] for x in outputs]).sum(0)
    correct_pred, total_pred, total_golden = all_counts
    precision = correct_pred / (total_pred + 1e-10)
    recall = correct_pred / (total_golden + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    res = {
        'span_precision': round(precision.cpu().numpy().tolist(), 5),
        'span_recall': round(recall.cpu().numpy().tolist(), 5),
        'span_f1': round(f1.cpu().numpy().tolist(), 5)
    }
    return res, outputs

def main():
    args = get_args()
    set_seed(args.seed)

    # config
    config = AutoConfig.from_pretrained(
        args.model_path,
        id2label={i:label for i, label in enumerate(args.labels)},
        label2id={label:i for i, label in enumerate(args.labels)},
        cache_dir=args.model_path,
        output_attentions=True
    )
    args.model_type = config.model_type.lower()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=args.model_path
    )

    # model
    model = SpanNER.from_pretrained(
        args.model_path,
        config=config,
        args=args
    )
    model.to(args.device)

    global_step, tr_loss = train(args, model, tokenizer)

    config = AutoConfig.from_pretrained(
        args.output_path,
        id2label={i:label for i, label in enumerate(args.labels)},
        label2id={label:i for i, label in enumerate(args.labels)}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.output_path
    )
    model = SpanNER.from_pretrained(
        args.output_path,
        config=config,
        args=args
    )
    model.to(args.device)

    res, outputs = evaluate(args, model, tokenizer, mode='test')

    print('Test result: f1 {} \t precision {} \t recall {}'.format(res['span_f1'], res['span_precision'], res['span_recall']))

    test_path = os.path.join(args.data_path, 'test.txt')
    output_predictions_path = os.path.join(args.output_path, 'test_predictions.txt')
    predictions_save(test_path, outputs, output_predictions_path, args.labels)


if __name__ == '__main__':
    main()