#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import json
import torch
import time
import random
import datetime
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import RobertaConfig
from transformers import BertConfig
from transformers import BertForPreTraining
from transformers import BertModel
from transformers import BertForTokenClassification
from transformers import BertForSequenceClassification
# from transformers import RobertaForTokenClassification
# from transformers import RobertaForSequenceClassification
from transformers import BertPreTrainedModel
from transformers import BertForQuestionAnswering
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from config import model_path, train_file_4


label_dict = {'疾病和诊断': 1, '影像检查': 3,
              '解剖部位': 5, '手术': 7, '药物': 9, '实验室检验': 11}
label_map_reverse = {
    0: 'O',
    1: 'B-disease',
    2: 'I-disease',
    3: 'B-check',
    4: 'I-check',
    5: 'B-part',
    6: 'I-part',
    7: 'B-operation',
    8: 'I-operation',
    9: 'B-drug',
    10: 'I-drug',
    11: 'B-assay',
    12: 'I-assay',
}


def process_text(text):
    text = text.replace("Ⅰ", '一')
    text = text.replace("Ⅱ", '二')
    text = text.replace("Ⅲ", '三')
    text = text.replace("Ⅳ", '四')
    text = text.replace("Ⅴ", '五')
    text = text.replace("Ⅵ", '六')
    text = text.replace("Ⅶ", '七')
    text = text.replace("Ⅷ", '八')
    text = text.replace("Ⅸ", '九')
    text = text.replace("Ⅹ", '十')
    text = text.replace("℃", "度")
    text = text.replace("㎝", "m")
    text = text.replace("㎡", "m")
    text = text.replace("“", "\"")
    text = text.replace("”", "\"")
    text = text.replace("缬", "结")
    text = text.replace("冸", "泮")
    text = text.replace("莨", "良")
    text = text.replace("菪", "宕")

    return text


def transform_entities_to_label(text, entities, sep_sentence, sentence_max_length):
    """
    原文，实体，wwm ===> 标签(支持词组encode)
    :param text: originalText
    :param entities: [{"start_pos": 10, "end_pos": 13, "label_type": "疾病和诊断"}]
    :param sep_sentence: [word, word]
    :return: [0, 0, 1, 0]
    """
    char_label = np.array([0 for i in range(len(text))])
    out = np.array([0 for i in range(sentence_max_length)])

    for i in entities:
        char_label[i["start_pos"]:i["end_pos"]] = label_dict[i["label_type"]]
        char_label[i["end_pos"] - 1] = label_dict[i["label_type"]] + 1

    current_idx = 0
    pad_num = 0
    while '<pad>' in sep_sentence:
        sep_sentence.remove("<pad>")
        pad_num += 1

    for i, j in enumerate(sep_sentence[1:-2]):
        out[i + pad_num + 1] = max(char_label[current_idx:current_idx + len(j)])

        if j == "<unk>":
            current_idx = current_idx + 1
        else:
            current_idx = current_idx + len(j)

    return out.tolist()


def load_train_data(train_file_path, model_file_path, is_sequence, sentence_max_length):
    config = BertConfig.from_pretrained(model_file_path)
    tokenizer = BertTokenizer.from_pretrained(model_file_path)

    train_input_ids, train_attention_masks = [], []

    if not is_sequence:  # 分类
        # TODO 句对分类
        train_df = pd.read_csv(train_file_path, header=None, sep='\t', index_col=0)
        train_df.columns = ['text', 'label']

        label_map = dict(zip(train_df['label'].value_counts().index,
                             range(len(train_df['label'].value_counts()))))
        config.num_labels = len(label_map)
        train_df['label'] = train_df[['label']].applymap(lambda x: label_map[x])
        train_sentences = train_df[['text']].values
        train_labels = train_df.label.values
        data_size = train_sentences.shape[0]

        for sent in train_sentences:
            encoded_dict = tokenizer.encode_plus(
                text=sent[0],
                text_pair=None,
                add_special_tokens=True,
                max_length=sentence_max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            train_input_ids.append(encoded_dict['input_ids'])
            train_attention_masks.append(encoded_dict['attention_mask'])

    else:  # 序列标注
        config.num_labels = len(label_map_reverse)
        train_labels = []
        with open(train_file_path, mode="r", encoding="utf-8") as f1:
            for line in f1.readlines():
                data = json.loads(line.strip())

                originalText = process_text(data["originalText"])
                entities = data["entities"]

                encoded_dict = tokenizer.encode_plus(
                    originalText,
                    text_pair=None,
                    add_special_tokens=True,
                    max_length=sentence_max_length,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                input_ids = encoded_dict['input_ids']
                sep_sentence = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
                seq_lab = transform_entities_to_label(originalText, entities, sep_sentence,
                                                      sentence_max_length=sentence_max_length)
                labels = torch.tensor(seq_lab).unsqueeze(0)

                train_attention_masks.append(encoded_dict["attention_mask"])
                train_input_ids.append(input_ids)
                train_labels.append(labels)
        data_size = len(train_labels)

    # 数据切分
    train_input_ids = torch.cat(train_input_ids, dim=0)
    train_attention_masks = torch.cat(train_attention_masks, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    train_size = 8 * data_size // 10

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,
                                                                [train_size, data_size - train_size])

    # model = BertForSequenceClassification(config=config)  # 文本分类
    model = BertForTokenClassification(config=config)  # 序列标注
    model.cuda()

    return train_dataset, test_dataset, model


# 数据处理
train_dataset, test_dataset, model = load_train_data(train_file_4, model_path,
                                                     is_sequence=True, sentence_max_length=50)


batch_size = 8

train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)

test_dataloader = DataLoader(
    test_dataset,  # The validation samples.
    sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

params = list(model.named_parameters())

print('The model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(test_dataloader)
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
