import numpy as np

def get_accuracy(labels, logits, n_class):
    conf_mat = np.zeros([n_class,n_class])
    for i, pred in zip(range(len(labels)),logits):
        conf_mat[int(labels[i]),pred] += 1

    precision = np.zeros([n_class])
    recall = np.zeros([n_class])
    for i in range(n_class):
        true_pos = conf_mat[i,i]

        indices = np.array(list(filter(lambda x: x!=i, np.arange(0,n_class))))
        false_pos = np.sum(conf_mat[indices,i])
        false_neg = np.sum(conf_mat[i,indices])

        if true_pos+false_pos != 0:
            precision[i] = true_pos/(true_pos+false_pos)
        else:
            precision[i] = 0

        if true_pos+false_neg != 0:
            recall[i] = true_pos/(true_pos + false_neg)
        else:
            recall[i] = 0
    # f1_score = 2 * ((precision * recall)/(precision + recall))
    f1_score = precision + recall
    f1_score[f1_score == 0] = 1e-9
    f1_score = 2 * (precision * recall)/f1_score
    return conf_mat, precision, recall, f1_score
