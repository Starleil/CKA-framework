import torch
# import torch.nn as nn
from .data import infoDataset
from .trainer import train, test
from .nn import UninormAggregator
import argparse
from datetime import datetime
from os import path, makedirs

parser = argparse.ArgumentParser()
parser.add_argument("--datafile", default=r'')
parser.add_argument("--outputdir", default='')
parser.add_argument("--infos", default=None, help="If there is no datafile, need to initialize infos")
parser.add_argument("--PATH",default=r'')
parser.add_argument("--score_range", default=2, type=int, help="num of class")
parser.add_argument("--testfile",
					default=r'')
parser.add_argument("--expname", default="trial")
parser.add_argument("--tnorm", default="product", choices=['lukasiewicz', 'product'])
parser.add_argument("--off_diagonal", default="min", choices=['min', 'mean', 'max'])
parser.add_argument("--loss", default="ce", choices=['ce', 'kl'])
parser.add_argument("--earlystop", default="train_acc", choices=['last', 'train_loss', 'train_acc'])
parser.add_argument('--setting', default='traintest', choices=['traintest'])
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--normalize_neutral", action='store_true', default=False)
parser.add_argument("--use_majority_label", action='store_true', default=False)
parser.add_argument("--use_binary_labels", action='store_true', default=False)
parser.add_argument("--use_score_hierarchy", action='store_true', default=False)
parser.add_argument("--rebalance_scores", action='store_true', default=False)
parser.add_argument("--use_sord", action='store_true', default=True)
parser.add_argument("--multithread", action='store_true', default=False)
parser.add_argument("--stratified", action='store_true', default=False)
parser.add_argument("--zero_score_gap", default=0.5, type=float)
parser.add_argument("--init_neutral", default=1/3, type=float)
parser.add_argument("--lr_gamma", default=1/3, type=float)
parser.add_argument("--datatype", default="normal", choices=['normal', 'softmax', '1_score'])
args = parser.parse_args()

dataset = infoDataset(args.datafile, args.infos, args.score_range, args.datatype)

workdir = path.join(args.outputdir, datetime.now().isoformat().replace(':', '_'))
makedirs(workdir)
outprefix = path.join(workdir, args.expname)
print('init_neutral', args.init_neutral, 'lr_gamma', args.lr_gamma, 'datatype', args.datatype)
with open (outprefix + "_config.txt", "w") as f:
	f.write(str(args).replace(',', ',\n') + "\n")
# TERRIBLE PATCH BEFORE IMPLEMENTING PROPER LOGGING
# import sys
# sys.stdout = open(outprefix + ".log", 'w')

if args.testfile:
	testset = infoDataset(args.testfile, args.infos, args.score_range, args.datatype)
else:
	testset = dataset

score_range = args.score_range

#
if args.PATH:
	print('loading model...')
	net = UninormAggregator(score_range, tnorm=args.tnorm, normalize_neutral=args.normalize_neutral,
								init_neutral=args.init_neutral)
	net.load_state_dict(torch.load(args.PATH))
	net.eval()
else:
	net = train(dataset, testset, outprefix + "_model", score_range, args)
test(net, testset, outprefix + "_preds", score_range)
