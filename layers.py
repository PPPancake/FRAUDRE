import email
from unittest import result
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

### 公式3
def weight_inter_agg(num_relations, neigh_feats, embed_dim, alpha, n, cuda): # 实现基于权重的多关系邻居聚合
	
	"""
	Weight inter-relation aggregator
	:param num_relations: number of relations in the graph 关系的数量
	:param neigh_feats: intra_relation aggregated neighbor embeddings for each aggregation 内关系聚合的邻居嵌入
	:param embed_dim: the dimension of output embedding 输出嵌入的维度
	:param alpha: weight paramter for each relation 每个关系的权重参数
	:param n: number of nodes in a batch 一个批次中节点的数量
	:param cuda: whether use GPU 是否使用GPU计算
	"""

	neigh_h = neigh_feats.t()

	w = F.softmax(alpha, dim = 1)
	
	if cuda:
		aggregated = torch.zeros(size=(embed_dim, n)).cuda() #
	else:
		aggregated = torch.zeros(size=(embed_dim, n))

	for r in range(num_relations):
        # 对应的权重向量*对应的邻居嵌入向量
		aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1,n), neigh_h[:, r*n:(r+1)*n])

	return aggregated.t()

### 拓扑不可知嵌入层
class MLP_(nn.Module): # 学习节点特征

	"""
	the ego-feature embedding module
	"""

	def __init__(self, features, input_dim, output_dim, cuda = False):
		super(MLP_, self).__init__()

		self.features = features # 节点特征矩阵
		self.input_dim = input_dim # 输入特征维度
		self.output_dim = output_dim # 输出特征维度
		print(self.input_dim, self.output_dim)
		self.cuda = cuda
		self.mlp_layer = nn.Linear(self.input_dim, self.output_dim)
    
	def forward(self, nodes):
        # 接受一个节点列表作为输入，根据节点列表得到相应的节点特征
		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(nodes))
		else:
			batch_features = self.features(torch.LongTensor(nodes))

		if self.cuda:
			self.mlp_layer.cuda()

        # 然后将节点特征传递给线性层进行特征变换，最后使用ReLU作为激活函数输出
        ### 公式1
		result = self.mlp_layer(batch_features)

		result = F.relu(result)

		# print(result)


		return result

class InterAgg(nn.Module): # 外关系聚合
    # 把目标节点的邻居节点找出来，在不同的关系上进行聚合，然后再聚合不同关系的嵌入，使用注意力机制为不同的关系赋予权重

	"""
	the fraud-aware convolution module
	Inter aggregation layer
	"""

	def __init__(self, features, embed_dim, adj_lists, intraggs, cuda = False):

		"""
		Initialize the inter-relation aggregator
		:param features: the input embeddings for all nodes 输入节点的嵌入特征
		:param embed_dim: the dimension need to be aggregated 需要聚合的维度
		:param adj_lists: a list of adjacency lists for each single-relation graph 单一关系图的邻接列表
		:param intraggs: the intra-relation aggregatore used by each single-relation graph 用于每个单一关系图的内关系聚合器
		:param cuda: whether to use GPU
		"""

		super(InterAgg, self). __init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda

        #初始化用于计算注意力权重的参数aplha
		if self.cuda:
			self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim*2, 3)).cuda()

		else:
			self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim*2, 3))

		init.xavier_uniform_(self.alpha)


	def forward(self, nodes, train_flag = True):

		"""
		nodes: a list of batch node ids
		"""
		
		if (isinstance(nodes,list)==False):
			nodes = nodes.cpu().numpy().tolist() # nodes是当前mini-batch中包含的节点列表
		
		to_neighs = []

		#adj_lists = [relation1, relation2, relation3]

		for adj_list in self.adj_lists: # 找到输入节点的邻居节点
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		#to_neighs: [[set, set, set], [set, set, set], [set, set, set]]
		
		#find unique nodes and their neighbors used in current batch   #set(nodes)
        # 存储所有出现在邻居节点列表中的节点
		unique_nodes =  set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),set.union(*to_neighs[2], set(nodes)))

		#id mapping
        # 创建节点ID到新的连续ID的映射
		unique_nodes_new_index = {n: i for i, n in enumerate(list(unique_nodes))}
		
        # 转换为张量形式便于运算
		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))
        
		# print(type(batch_features))
        
		# print(batch_features)


        # 进行ID的映射
		#get neighbor node id list for each batch node and relation
		r1_list = [set(to_neigh) for to_neigh in to_neighs[0]] # [[set],[set],[ser]]  //   [[list],[list],[list]]
		r2_list = [set(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [set(to_neigh) for to_neigh in to_neighs[2]]

		center_nodes_new_index = [unique_nodes_new_index[int(n)] for n in nodes]
		'''
		if self.cuda and isinstance(nodes, list):
			self_feats = self.features(torch.cuda.LongTensor(nodes))
		else:
			self_feats = self.features(index)
		'''

		#center_feats = self_feats[:, -self.embed_dim:]
		
        # 找到对应的节点特征
		self_feats = batch_features[center_nodes_new_index]

        # 三组关系分别进行聚合，内关系聚合
		r1_feats = self.intra_agg1.forward(batch_features[:, -self.embed_dim:], nodes, r1_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])
		r2_feats = self.intra_agg2.forward(batch_features[:, -self.embed_dim:], nodes, r2_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])
		r3_feats = self.intra_agg3.forward(batch_features[:, -self.embed_dim:], nodes, r3_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])

        # 将三组关系的新特征表示拼接在一起
		neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim = 0)

		n=len(nodes)

        # 为不同关系赋予权重并聚合
		attention_layer_outputs = weight_inter_agg(len(self.adj_lists), neigh_feats, self.embed_dim * 2, self.alpha, n, self.cuda)

        # 将self_feats和聚合后的邻居特征拼接
        ### 中间表示组合模块，公式4
		result = torch.cat((self_feats, attention_layer_outputs), dim = 1)

		return result

class IntraAgg(nn.Module): # 内关系聚合
    # 每个节点聚合它的邻居节点的特征，从而更新自身的特征

	"""
	the fraud-aware convolution module
	Intra Aggregation Layer
	"""

	def __init__(self, cuda = False):

		super(IntraAgg, self).__init__()

		self.cuda = cuda

	def forward(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):

		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch # 所有节点的id列表
		:param embedding: embedding of all nodes in a batch 所有节点的特征向量
		:param neighbor_lists: neighbor node id list for each batch node in one relation # [[list],[list],[list]] 每个节点的邻居节点ID列表
		:param unique_nodes_new_index 所有节点的ID到索引的映射，用于将节点ID转化为embedding中的索引
		"""

		#find unique nodes
		unique_nodes_list = list(set.union(*neighbor_lists))

		#id mapping
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # 根据新的ID创建新的邻接矩阵
		mask = Variable(torch.zeros(len(neighbor_lists), len(unique_nodes)))

		# print("+++++++++++++++++")
		# print(len(neighbor_lists), len(unique_nodes))
		
		column_indices = [unique_nodes[n] for neighbor_list in neighbor_lists for n in neighbor_list ]
		row_indices = [i for i in range(len(neighbor_lists)) for _ in range(len(neighbor_lists[i]))]

		mask[row_indices, column_indices] = 1


		num_neigh = mask.sum(1,keepdim=True) # 计算每个节点的邻居数
		#mask = torch.true_divide(mask, num_neigh)
		mask = torch.div(mask, num_neigh) # 每个节点邻居的权重，

        # 根据索引提取特征向量
		neighbors_new_index = [unique_nodes_new_index[n] for n in unique_nodes_list ]

		embed_matrix = embedding[neighbors_new_index]
		
		embed_matrix = embed_matrix.cpu()

		print(embed_matrix.shape[0], embed_matrix.shape[1])

        ###公式2
		_feats_1 = mask.mm(embed_matrix)
		if self.cuda:
			_feats_1 = _feats_1.cuda()

		#difference 
		_feats_2 = self_feats - _feats_1
		return torch.cat((_feats_1, _feats_2), dim=1)
