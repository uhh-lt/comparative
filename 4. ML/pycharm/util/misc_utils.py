from sklearn.metrics import precision_recall_fscore_support
import logging
from datetime import datetime


def latex_table(res, cap=''):
    try:
        b_row = '\\texttt{BETTER}\t & '
        w_row = '\\texttt{WORSE}\t & '
        n_row = '\\texttt{NONE}\t & '
        a_row = 'average\t & '
        for v in res:
            precision, recall, f1, support = precision_recall_fscore_support(v[0], v[1], average=None,
                                                                             labels=['BETTER', 'WORSE', 'NONE'])
            b_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision[0], recall[0], f1[0])
            w_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision[1], recall[1], f1[1])
            n_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision[2], recall[2], f1[2])
            precision, recall, f1, support = precision_recall_fscore_support(v[0], v[1], average='weighted',
                                                                             labels=['BETTER', 'WORSE', 'NONE'])
            a_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision, recall, f1)

        b_row = b_row[:-1] + '\\\ '
        w_row = w_row[:-1] + '\\\ '
        n_row = n_row[:-1] + '\\\ \midrule '
        a_row = a_row[:-1] + '\\\ \\bottomrule'
        tbl = """ \\begin{table}[h]
                        \centering"""

        tbl += '\caption{{ {} }}'.format(cap)
        tbl += '\label{{tbl:{} }}'.format(cap.lower().replace(' ', '_'))

        tbl += """ \\begin{tabular}{@{}lccccccccc@{}}
                      \\toprule
                       & \multicolumn{3}{c}{Worst} & \multicolumn{3}{c}{Average} & \multicolumn{3}{c}{Best}  \\\ \midrule
                       & Precision  & Recall & F1   & Precision  & Recall  & F1    & Precision & Recall & F1   \\\ \\toprule"""
        tbl += b_row
        tbl += w_row
        tbl += n_row
        tbl += a_row
        tbl += """
            \end{tabular}
        \end{table}
            """
        return tbl
    except Exception as e:
        return 'kaputt'


def latex_table_bin(res, cap=''):
    b_row = '\\texttt{ARG}\t & '
    n_row = '\\texttt{NONE}\t & '
    a_row = 'average\t & '
    for v in res:
        precision, recall, f1, support = precision_recall_fscore_support(v[0], v[1], average=None,
                                                                         labels=['ARG', 'NONE'])
        b_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision[0], recall[0], f1[0])
        n_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision[1], recall[1], f1[1])
        precision, recall, f1, support = precision_recall_fscore_support(v[0], v[1], average='weighted',
                                                                         labels=['ARG', 'NONE'])
        a_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision, recall, f1)

    b_row = b_row[:-1] + '\\\ '
    n_row = n_row[:-1] + '\\\ \midrule '
    a_row = a_row[:-1] + '\\\ \\bottomrule'
    print(""" \\begin{table}[h]
                \centering""")
    print('\caption{{ {} }}'.format(cap))
    print('\label{{tbl:{} }}'.format(cap.lower().replace(' ', '_')))

    print(""" \\begin{tabular}{@{}lccccccccc@{}}
              \\toprule
               & \multicolumn{3}{c}{Worst} & \multicolumn{3}{c}{Average} & \multicolumn{3}{c}{Best}  \\\ \midrule
               & Precision  & Recall & F1   & Precision  & Recall  & F1    & Precision & Recall & F1   \\\ \\toprule""")
    print(b_row)
    print(n_row)
    print(a_row)
    print("""
    \end{tabular}
\end{table}
    """)


def res_table(res, logger):
    logger.info("""\\begin{table}[h]
\centering
\caption{My caption}
\label{my-label}
\\begin{tabularx}{\\textwidth}{Xrrrr}
\\toprule
Feature & Worst (F1) & Average (F1) & Best (F1) & Mean F1 \\\ \midrule
""")
    for data in res:
        logger.info(
            '{} ({}) &\t {:04.2f} &\t {:04.2f}&\t {:04.2f}&\t{:04.2f} \\\ '.format(data['name'], data['comment'],
                                                                                   data['worst'], data['avg'],
                                                                                   data['best'],
                                                                                   (data['worst'] + data['avg'] +
                                                                                    data['best']) / 3))
    logger.info("""\\bottomrule\end{tabularx}
\end{table}""")


def get_logger(name):
    now = datetime.now()
    s = '{}-{}-({}_{})'.format(name, now.day, now.hour, now.minute)
    logger = logging.getLogger(s)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(s + '.log')
    ch = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
