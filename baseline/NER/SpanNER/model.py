import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor

class SpanLayer(nn.Module):
    def __init__(self, hidden_size, span_len_dim, max_span_len):
        super().__init__()
        self.span_extractor = EndpointSpanExtractor(input_dim=hidden_size)
        self.span_len_embedding = nn.Embedding(
            max_span_len + 1, span_len_dim, padding_idx=0
        )

    def forward(self, sequences_embed, span_token_idxes, span_lens):
        span_endpoints_rep = self.span_extractor(sequences_embed, span_token_idxes.long())
        span_len_rep = self.span_len_embedding(span_lens)

        span_rep = torch.cat((span_endpoints_rep, span_len_rep), dim=-1)
        return span_rep


class SpanNER(BertPreTrainedModel):
    def __init__(self, config, args):
        super(SpanNER, self).__init__(config)

        self.bert = BertModel(config)
        self.n_class = len(args.labels)

        self.span_layer = SpanLayer(
            hidden_size=config.hidden_size,
            span_len_dim=args.span_len_dim,
            max_span_len=args.max_span_len
        )

        span_rep_dim = 2*config.hidden_size + args.span_len_dim
        self.classifier = nn.Sequential(
            nn.Linear(span_rep_dim, span_rep_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(span_rep_dim, self.n_class)
        )

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def forward(self, features):
        encoding = {}
        encoding['span_rep'] = self.span_encoding(**features)
        encoding['logits'] = self.classifier(encoding['span_rep'])

        loss = self.compute_loss(features, encoding)
        outputs = [self.softmax(encoding['logits'])]

        return outputs, loss

    def span_encoding(self, input_ids, input_mask, segment_ids, span_token_idxes, span_lens, **kwargs):
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids
        )
        span_rep = self.span_layer(
            sequence_output[0],
            span_token_idxes.long(),
            span_lens
        )
        return span_rep

    def compute_loss(self, features, encoding):
        batch_size, n_span = features['span_labels'].size()
        predicts = encoding['logits'].view(-1, self.n_class)
        labels = features['span_labels'].view(-1)

        loss = self.criterion(predicts, labels)
        loss = loss.view(batch_size, n_span) * features['span_weights']

        loss = torch.masked_select(loss, features['span_masks'].bool()).mean()

        return loss