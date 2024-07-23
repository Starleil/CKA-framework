import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import softmax
from aggregator.nn import UninormAggregator
from aggregator.util import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, \
    recall_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from threading import Thread
import pickle


class TrainThread(Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        super(TrainThread, self).__init__(group, target, name, args, kwargs)
        self.trainset = args[0]
        self.testset = args[1]
        self.modelfile = args[2]
        self.outputfile = args[3]
        self.score_range = args[4]
        self.args = args[5]
        self._return = 0

    def run(self):
        net = train(self.trainset, self.modelfile, self.score_range, self.args)
        self._return = test(net, self.testset, self.outputfile, self.score_range)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def train(dataset, testset, modelfile, score_range, args):
    net = UninormAggregator(score_range, tnorm=args.tnorm, normalize_neutral=args.normalize_neutral,
                            init_neutral=args.init_neutral)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 20, 25], gamma=args.lr_gamma)
    if args.loss == "kl":
        criterion = kl_div_loss
    else:
        criterion = cross_entropy_loss

    score_weight = None

    max_accuracy = 0.0
    max_loss = 0.0

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        num = 0

        optimizer.zero_grad()
        for x, label in dataset:
            y = net(x).view(1, -1)
            loss = criterion(y, label, use_sord=args.use_sord, zero_score_gap=args.zero_score_gap, weight=score_weight)
            loss.backward()
            running_loss += loss.item()
            running_accuracy += label == torch.argmax(y)
            num += 1
        running_accuracy /= num
        running_loss /= num
        print("parameters")
        net.print_parameters()
        print("gradient")
        net.print_gradient()
        print('[%d] train accuracy: %.5f' % (epoch + 1, running_accuracy))
        print('[%d] train loss: %.5f' % (epoch + 1, running_loss))

        # val
        val_loss = 0.0
        val_accuracy = 0.0
        val_num = 0

        net.eval()
        with torch.no_grad():
            for x, label in testset:
                y = net(x).view(1, -1)
                loss = criterion(y, label, use_sord=args.use_sord, zero_score_gap=args.zero_score_gap, weight=score_weight)
                val_loss += loss.item()
                val_accuracy += label == torch.argmax(y)
                val_num += 1
        val_accuracy /= val_num
        val_loss /= val_num
        print('[%d] val accuracy: %.5f' % (epoch + 1, val_accuracy))
        print('[%d] val loss: %.5f' % (epoch + 1, val_loss))
        max_accuracy, max_loss = save_checkpoint(net, modelfile, args.earlystop, val_accuracy, val_loss,
                                                 max_accuracy, max_loss)

        optimizer.step()
        if not args.normalize_neutral:
            net.clamp_params()
        lr_scheduler.step()

    if args.earlystop != "last":
        net.load_state_dict(torch.load(modelfile))

    print("learned parameters")
    net.print_parameters()

    return net


def save_checkpoint(net, modelfile, earlystop, running_accuracy, running_loss, max_accuracy, max_loss):
    if running_accuracy > max_accuracy:
        max_accuracy = running_accuracy
        if earlystop == 'train_acc':
            torch.save(net.state_dict(), modelfile)
    if running_loss > max_loss:
        max_loss = running_loss
        if earlystop == 'train_loss':
            torch.save(net.state_dict(), modelfile)
    return max_accuracy, max_loss


def test(net, testset, outputfile, score_range):
    inputs = []
    labels = []
    outputs = []

    with torch.no_grad():
        for x, label in testset:
            inputs.append(x)
            labels.append(label[0])
            outputs.append(net(x))

        print("Predictor results")
        preds = [torch.argmax(o) for o in outputs]
        print_results(labels, preds, score_range)
        print("auc = %.5f" % roc_auc_score(labels,[torch.sum(softmax(o, dim=0)[1:]).detach().numpy() for o in outputs]))
        save_predictions(labels, preds, outputfile)

    baselines = (argmax_mean,
                 max_argmax)

    for baseline in baselines:
        baseline_preds = []
        for x, label in testset:
            a = baseline(x)
            baseline_preds.append(baseline(x))
        print("Baseline <%s> results" % baseline.__name__)
        print_results(labels, baseline_preds, score_range)
        print("auc = %.5f" % roc_auc_score(labels, [torch.sum(softmax(o, dim=0)[1:]).detach().numpy() for o in outputs]))

    return inputs, labels, outputs, preds


def print_results(labels, preds, score_range):
    print(labels)
    print(preds)
    print(confusion_matrix(labels, preds))
    print("\nweighted f1 = %.5f" % f1_score(labels, preds))
    print("weighted pre = %.5f" % precision_score(labels, preds))
    print("weighted rec = %.5f" % recall_score(labels, preds))
    print("accuracy = %.5f" % accuracy_score(labels, preds))


def save_predictions(labels, preds, outputfile):
    with open(outputfile, "w") as f:
        np.savetxt(f, torch.tensor([labels, preds], dtype=torch.int8).t(), fmt='%d')
