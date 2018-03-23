from collections import defaultdict, OrderedDict

import math
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support


def latex_classification_report(y_true, y_pred, average='weighted', labels=None, style='booktabs', caption='',
                                tbl_label='', derivations=defaultdict(int)):
    _labels = set(y_true) if labels is None else labels
    values = OrderedDict()
    for label in _labels:
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[label])
        values[label] = {'precision': precision[0], 'recall': recall[0], 'f1': f1[0],
                         'precision_der': derivations['precision'][label], 'recall_der': derivations['recall'][label],
                         'f1_der': derivations['f1'][label]}

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=average,
                                                                     labels=list(_labels))
    values['average'] = {'precision': precision, 'recall': recall, 'f1': f1,
                         'precision_der': derivations['precision']['avg'], 'recall_der': derivations['recall']['avg'],
                         'f1_der': derivations['f1']['avg']}

    table_str = '\\begin{{table}}[h] \n \centering \n \caption{{ {} }} \n \label{{ {} }}'.format(caption, tbl_label)
    table_str += '\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n' if style is 'booktabs' else '\\begin{tabular}{lrrrr}\n'
    table_str += ' \t&\t precision &\t recall &\t f1 score  \\\ \midrule \n' if style is 'booktabs' else ' \t&\t precision &\t recall &\t f1 score  \\\ \hline \n'

    for k, v in values.items():
        table_str += '{}\t&\t {:04.2f} ({:04.2f}) &\t {:04.2f} ({:04.2f}) &\t {:04.2f} ({:04.2f})  \\\ \n'.format(k, v[
            'precision'], v['precision_der'], v['recall'], v['recall_der'], v['f1'], v['f1_der'])

    table_str += '\\bottomrule\n' if style is 'booktabs' else '\hline\n'
    table_str += '\end{tabular}\n\end{table}'
    return table_str


def get_std_derivations(true_pred_list, labels, average='weighted'):
    prec_means = _get_means(true_pred_list, labels, precision_score, average)
    recall_means = _get_means(true_pred_list, labels, recall_score, average)
    f1_means = _get_means(true_pred_list, labels, f1_score, average)
    prec_der = _get_derivation(true_pred_list, labels, prec_means, precision_score, average)
    recal_der = _get_derivation(true_pred_list, labels, recall_means, recall_score, average)
    f1_der = _get_derivation(true_pred_list, labels, f1_means, f1_score, average)
    return {'precision': prec_der, 'recall': recal_der, 'f1': f1_der}


def get_best_fold(true_pred_list, average='weighted', metric=f1_score):
    lst = []
    for y_true, y_pred in true_pred_list:
        lst.append((metric(y_true, y_pred, average=average), (y_true, y_pred)))
    lst = sorted(lst, key=lambda x: x[0], reverse=True)
    return lst[0][1]


def _get_derivation(true_pred_list, labels, means, metric, average='weighted'):
    derivation = defaultdict(int)
    for y_true, y_pred in true_pred_list:
        derivation['avg'] += (metric(y_true, y_pred, average=average) - means['avg']) ** 2
        for label in labels:
            derivation[label] += (metric(y_true, y_pred, average=average, labels=[label]) - means[
                label]) ** 2
    derivation = {k: math.sqrt(v / len(true_pred_list)) for k, v in derivation.items()}
    return derivation


def _get_means(true_pred_list, labels, metric, average='weighted'):
    prec_means = defaultdict(int)
    for y_true, y_pred in true_pred_list:
        prec_means['avg'] += metric(y_true, y_pred, average=average)
        for label in labels:
            prec_means[label] += metric(y_true, y_pred, average=average, labels=[label])
    return {k: v / len(true_pred_list) for k, v in prec_means.items()}
