from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class evaluation_metrics():
    def __init__(self, embs, labels, args, train_idx=None, val_idx=None, test_idx=None):

        self.embs = embs
        self.args = args
        if args.dataset in ['Aminer','photo','computers', 'cs', 'physics']:
            self.trX, self.trY = self.embs[train_idx], np.array(labels[train_idx])
            self.valX, self.valY = self.embs[val_idx], np.array(labels[val_idx])
            self.tsX, self.tsY = self.embs[test_idx], np.array(labels[test_idx])
            self.n_label = int(max(labels)-min(labels)+1)
        else:
            train, val, test = labels
            self.trX, self.trY = self.embs[np.array(train)[:,0]], np.array(train)[:,1]
            self.valX, self.valY = self.embs[np.array(val)[:,0]], np.array(val)[:,1]
            self.tsX, self.tsY = self.embs[np.array(test)[:,0]], np.array(test)[:,1]
            self.n_label = len(set(self.tsY))
    def evalutation(self, args):
        fis, fas = 0,0
        for rs in [0, 1, 2, 3, 4]:
            lr = LogisticRegression(max_iter=500, random_state=rs, solver='sag')
            lr.fit(self.trX, self.trY)
            Y_pred = lr.predict(self.tsX)
            f1_micro = metrics.f1_score(self.tsY, Y_pred, average='micro')
            f1_macro = metrics.f1_score(self.tsY, Y_pred, average='macro')
            fis+=f1_micro
            fas+=f1_macro
        print('\t[Classification] f1_macro=%.5f, f1_micro=%.5f' % (fas/5, fis/5))

        return fis/5, fas/5




