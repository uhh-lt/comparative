from sklearn.metrics import precision_recall_fscore_support


def latex_table(res, cap=''):
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
    print(""" \\begin{table}[h]
                \centering""")
    print('\caption{{ {} }}'.format(cap))
    print('\label{{tbl:{} }}'.format(cap.lower().replace(' ', '_')))

    print(""" \\begin{tabular}{@{}lccccccccc@{}}
              \\toprule
               & \multicolumn{3}{c}{Worst} & \multicolumn{3}{c}{Average} & \multicolumn{3}{c}{Best}  \\\ \midrule
               & Precision  & Recall & F1   & Precision  & Recall  & F1    & Precision & Recall & F1   \\\ \\toprule""")
    print(b_row)
    print(w_row)
    print(n_row)
    print(a_row)
    print("""
    \end{tabular}
\end{table}
    """)

