import time
import argparse # 命令行参数模块
from sklearn.model_selection import train_test_split # 数据划分模块

from model import MODEL # 定义了GNN模型的结构
from layers import * # GNN所需要的层
from utlis import * # 辅助函数
#os.environ["CUDA_LAUNCH_BLOCKING"]="0"


"""
   Paper: FRAUDRE: Fraud Detection Dual-Resistant toGraph Inconsistency and Imbalance
   Source: https://github.com/FraudDetection/FRAUDRE
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='yelp', help='The dataset name. [Amazon_demo, Yelp_demo, amazon,yelp]')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size 1024 for yelp, 256 for amazon.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]') # 初始学习率
parser.add_argument('--lambda_1', type=float, default=1e-4, help='Weight decay (L2 loss weight).') # 权重衰减（L2损失的权重）
parser.add_argument('--embed_dim', type=int, default=64, help='Node embedding size at the first layer.') # 第一层节点嵌入的大小
parser.add_argument('--num_epochs', type=int, default=61, help='Number of epochs.') # 训练轮数
parser.add_argument('--test_epochs', type=int, default=10, help='Epoch interval to run test set.') # 测试轮数
parser.add_argument('--seed', type=int, default=123, help='Random seed.') # 随机数种子
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.') # 是否使用GPU进行训练

if(torch.cuda.is_available()):
	print("cuda is available")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if(args.cuda):
	print("runing with GPU")

print(f'run on {args.data}')

# load topology, feature, and label
homo, relation1, relation2, relation3, feat_data, labels = load_data(args.data)

# set seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# train_test split
if args.data == 'yelp':

	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify = labels, test_size = 0.80,
															random_state = 2, shuffle = True)

	# set prior 设置先验概率
	num_1= len(np.where(y_train==1)[0])
	num_2= len(np.where(y_train==0)[0])
	p0 = (num_1/(num_1+num_2))
	p1 = 1- p0
	prior = np.array([p1, p0])

    # 转换为numpy数组并添加一个微小值（1e-8）以避免出现零概率
	if args.cuda:
		prior = (torch.from_numpy(prior +1e-8)).cuda() 
	else:
		prior = (torch.from_numpy(prior +1e-8))

elif args.data == 'amazon':

	# 0-3304 are unlabeled nodes
	index = list(range(3305, len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify = labels[3305:],
															test_size = 0.90, random_state = 2, shuffle = True)

	num_1 = len(np.where(y_train == 1)[0])
	num_2 = len(np.where(y_train == 0)[0])
	p0 = (num_1 / (num_1 + num_2))
	p1 = 1 - p0
	prior = np.array([p1, p0])
	if args.cuda:
		prior = (torch.from_numpy(prior +1e-8)).cuda()
	else:
		prior = (torch.from_numpy(prior +1e-8))
	#prior = np.array([0.9, 0.1])


# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1]) # (输入维度，输出维度)
feat_data = normalize(feat_data) # 归一化
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad = False) # 将feat_data的值复制给嵌入层的权重
if args.cuda:
	features.cuda()

# set input graph topology
adj_lists = [relation1, relation2, relation3]


# build model

# the first neural network layer (ego-feature embedding module)
# 初始化一个mlp
mlp = MLP_(features, feat_data.shape[1], args.embed_dim, cuda = args.cuda)

#first convolution layer
intra1_1 = IntraAgg(cuda = args.cuda)
intra1_2 = IntraAgg(cuda = args.cuda)
intra1_3 = IntraAgg(cuda = args.cuda)
# 将mlp函数传入
agg1 = InterAgg(lambda nodes: mlp(nodes), args.embed_dim, adj_lists, [intra1_1, intra1_2, intra1_3], cuda = args.cuda)


#second convolution layer
intra2_1 = IntraAgg(cuda = args.cuda)
intra2_2 = IntraAgg(cuda = args.cuda)
intra2_3 = IntraAgg(cuda = args.cuda)

#def __init__(self, features, embed_dim, adj_lists, intraggs, cuda = False):
agg2 = InterAgg(lambda nodes: agg1(nodes), args.embed_dim*2, adj_lists, [intra2_1, intra2_2, intra2_3], cuda = args.cuda)
gnn_model = MODEL(2, 2, args.embed_dim, agg2, prior)
# gnn_model in one convolution layer
#gnn_model = MODEL(1, 2, args.embed_dim, agg1, prior, cuda = args.cuda)


if args.cuda:
	gnn_model.cuda()

# 使用优化器优化模型参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_1)
performance_log = []

# train the model

overall_time = 0
for epoch in range(args.num_epochs):

	# gnn_model.train()
	# shuffle
	random.shuffle(idx_train)
	num_batches = int(len(idx_train) / args.batch_size) +1 #每个mini_batch包含batch_size个节点

	loss = 0.0
	epoch_time = 0

	#mini-batch training
	for batch in range(num_batches):

		print(f'Epoch: {epoch}, batch: {batch}')

		i_start = batch * args.batch_size
		i_end = min((batch + 1) * args.batch_size, len(idx_train))

		batch_nodes = idx_train[i_start:i_end] # 训练节点

		batch_label = labels[np.array(batch_nodes)] # 训练标签

		optimizer.zero_grad() # 对优化器进行梯度清零

		start_time = time.time()

        # 计算损失
		if args.cuda:
			loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
		else:
			loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))

		end_time = time.time()

		epoch_time += end_time - start_time

		loss.backward()
		optimizer.step()
		loss += loss.item()

	print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')
	overall_time += epoch_time

	#testing the model for every $test_epoch$ epoch
    # 每个测试轮进行测试
	if epoch % args.test_epochs == 0:

		#gnn_model.eval()
		auc, precision, a_p, recall, f1 = test_model(idx_test, y_test, gnn_model)
		performance_log.append([auc, precision, a_p, recall, f1])

print("The training time per epoch")
print(overall_time/args.num_epochs)

