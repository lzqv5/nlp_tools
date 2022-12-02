def predictions_save(origin_file, predictions, output_file, labels):
    pred_label_idx = [x['pred_label_idx'].tolist() for x in predictions]
    all_span_idxs = [x['all_span_idx'].tolist() for x in predictions]
    span_label_ltoken = [x['span_label_ltoken'].tolist() for x in predictions]

    idx2label = {k: v for k, v in enumerate(labels)}

    with open(output_file, "w") as writer:
        sentence_words = []
        sentence_labels = []
        with open(origin_file, "r") as f:
            words = []
            labels = []

            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        sentence_words.append(words)
                        sentence_labels.append(labels)
                        words = []
                        labels = []
                else:
                    splits = line.split()
                    words.append(splits[0])
                    label = splits[-1].replace('\n', '')
                    if label[0:2] in ['B-', 'I-']:
                        label = label[2:]
                    labels.append(label)
            if words:
                sentence_words.append(words)
                sentence_labels.append(labels)
            f.close()

        cnt = 0
        for i in range(len(pred_label_idx)):
            for spans, predict_labels, true_labels in zip(all_span_idxs[i], pred_label_idx[i], span_label_ltoken[i]):
                words = sentence_words[cnt]
                for span, predict_label, _ in zip(spans, predict_labels, true_labels):
                    if predict_label != 0:
                        predict_label = idx2label[int(predict_label)]
                        begin, end = span
                        for k in range(int(begin), int(end) + 1):
                            sentence_labels[cnt][k] += ' {}'.format(predict_label)
                for k in range(len(words)):
                    if len(sentence_labels[cnt][k].strip().split()) == 1:
                        sentence_labels[cnt][k] += ' O'
                    sentence_labels[cnt][k] = words[k] + ' ' + sentence_labels[cnt][k]
                text = '\n'.join(sentence_labels[cnt])
                writer.write(text + '\n\n')
                cnt += 1

        writer.close()