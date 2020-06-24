import sys
import json
import time
from tqdm import tqdm, trange
import logging
import argparse
from typing import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)


def load_jsonl_data(data_dir: str) -> List:
    data = []
    with open(data_dir, 'r') as lf:
        for line in lf:
            data.append(json.loads(line))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_pred_file", type=str, default='dev_predictions.jsonl',
                        help='dev set')
    parser.add_argument("--test_pred_file", type=str, default='test_predictions.jsonl',
                        help='test set')

    args = parser.parse_args()

    t0 = time.time()

    log.info('... loading data (dev+test) ...')

    dev_data = load_jsonl_data(args.dev_file)
    dev_samples = [d['label_probs'] for d in dev_data]
    dev_labels = [d['label'] for d in dev_data]

    test_data = load_jsonl_data(args.test_file)
    test_samples = [d['label_probs'] for d in test_data]
    test_labels = [d['label'] for d in test_data]

    acc_max = 0.0
    eps_best = 0.0
    f1_macro_max = 0.0
    dev_len = len(dev_labels)
    log.info('... dev ...')
    for i in tqdm(range(1, 1000, 1)):
        eps = i / 1000
        dev_preds = [1 if score > eps else 0 for score in dev_samples]
        correct_count = sum(1 for pred, tf in zip(dev_preds, dev_labels) if pred == tf)
        acc = correct_count / dev_len
        f1_macro = f1_score(dev_labels, dev_preds, average='macro')
        if f1_macro > f1_macro_max:
            f1_macro_max = f1_macro
            eps_best = eps
            correct_count = sum(1 for pred, tf in zip(dev_preds, dev_labels) if pred == tf)
            acc = correct_count / dev_len
            acc_max = acc
    log.info('... threshold = %.3f ...' % eps_best)
    log.info('... F1 dev = %.3f ...' % f1_macro_max)
    print(confusion_matrix(dev_labels, dev_preds, labels=[False, True]))

    log.info('... on dev set ...')
    dev_len = len(dev_labels)
    dev_preds = [1 if score >= eps_best else 0 for score in dev_samples]
    dev_f1_macro = f1_score(dev_labels, dev_preds, average='macro')
    log.info('... F1 dev = %.3f ...' % dev_f1_macro)
    print(classification_report(dev_labels, dev_preds, digits=3))
    print(confusion_matrix(dev_labels, dev_preds, labels=[False, True]))

    log.info('Total time: %.4f (s)' % (time.time() - t0))

    log.info('... on test set ...')
    test_len = len(test_labels)
    test_preds = [1 if score >= eps_best else 0 for score in test_samples]
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    log.info('... F1 Test = %.3f ...' % test_f1_macro)
    print(classification_report(test_labels, test_preds, digits=3))
    print(confusion_matrix(test_labels, test_preds, labels=[False, True]))

    log.info('Total time: %.4f (s)' % (time.time() - t0))
