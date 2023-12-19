import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score, precision_score
from collections import defaultdict

def sparse_to_adjlist(sp_matrix):

	"""Transfer sparse matrix to adjacency list"""
    # 将稀疏矩阵转换为邻接列表的形式
    # 输入：稀疏矩阵 sp_matrix
    # 输出：邻接表（一个字典，其中字典的键是矩阵中的节点，对应的值是与该节点相邻的节点集合）

	#add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0]) #加self loop: 对角线加上单位矩阵，使节点自己连接自己
	#creat adj_list
	adj_lists = defaultdict(set) # 类似字典的数据结构，但是访问不存在的见，会自动创建默认值：空set
	edges = homo_adj.nonzero()
	
	for index, node in enumerate(edges[0]): # 遍历稀疏矩阵的非零元素，建立邻接表
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
    #限制每个节点的邻居数量不超过 10 个，限制每个节点的度数，并控制邻接表的大小
	adj_lists = {keya:random.sample(adj_lists[keya],10) if len(adj_lists[keya])>=10 else adj_lists[keya] for i, keya in enumerate(adj_lists)}

	return adj_lists

def load_data(data):

	"""load data"""
    #输入：字符串参数 data，用于指定加载哪个数据集
    #输出：自环图邻接表、不同关系类型的邻接表、节点特征矩阵和节点标签

	if data == 'yelp':

		yelp = loadmat('data/YelpChi.mat')
		homo = sparse_to_adjlist(yelp['homo'])
		relation1 = sparse_to_adjlist(yelp['net_rur'])
		relation2 = sparse_to_adjlist(yelp['net_rtr'])
		relation3 = sparse_to_adjlist(yelp['net_rsr'])
		feat_data = yelp['features'].toarray()
		labels = yelp['label'].flatten()

	elif data == 'amazon':

		amz = loadmat('data/Amazon.mat')
		homo = sparse_to_adjlist(amz['homo'])
		relation1 = sparse_to_adjlist(amz['net_upu'])
		relation2 = sparse_to_adjlist(amz['net_usu'])
		relation3 = sparse_to_adjlist(amz['net_uvu'])
		feat_data = amz['features'].toarray()
		labels = amz['label'].flatten()


	return homo, relation1, relation2, relation3, feat_data, labels


def normalize(mx):

	"""Row-normalize sparse matrix"""
    #输入：稀疏矩阵mx
    #输出：归一化后的稀疏矩阵

	rowsum = np.array(mx.sum(1)) #行和
	r_inv = np.power(rowsum, -1).flatten() #行和倒数
	r_inv[np.isinf(r_inv)] = 0. # 无穷大值替换为0
	r_mat_inv = sp.diags(r_inv) # 创建对角矩阵，对角元素为r_inv
	mx = r_mat_inv.dot(mx) # 每个值乘行和倒数
	return mx


def test_model(test_cases, labels, model):
	"""
	test the performance of model
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	"""
    # 输入：test_cases 是一个测试节点的列表，labels 是这些测试节点对应的标签，model 是待测试的图神经网络模型
    # 打印：ROC AUC、Precision、Average Precision、Recall 和 F1-score
	gnn_prob = model.to_prob(test_cases, train_flag = False) # 测试集输入到GNN模型中，得到模型预测的节点标签概率

	auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
	precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
	a_p = average_precision_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
	recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
	f1 = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

	#print(gnn_prob.data.cpu().numpy().argmax(axis=1))

	print(f"GNN auc: {auc_gnn:.4f}")
	print(f"GNN precision: {precision_gnn:.4f}")
	print(f"GNN a_precision: {a_p:.4f}")
	print(f"GNN Recall: {recall_gnn:.4f}")
	print(f"GNN f1: {f1:.4f}")

	return auc_gnn, precision_gnn, a_p, recall_gnn, f1
