import torch
import torch.nn as nn
from torch.nn import init
import math

class MODEL(nn.Module):

	def __init__(self, K, num_classes, embed_dim, agg, prior):
		super(MODEL, self).__init__()

		"""
		Initialize the model
		:param K: the number of CONVOLUTION layers of the model
		:param num_classes: number of classes (2 in our paper)
		:param embed_dim: the output dimension of MLP layer
		:agg: the inter-relation aggregator that output the final embedding
		:lambad 1: the weight of MLP layer (ignore it)
		:prior:prior 先验概率
		"""

		self.agg = agg 
		#self.lambda_1 = lambda_1

		self.K = K #how many layers 卷积层数量
		self.prior = prior # 先验概率
		self.xent = nn.CrossEntropyLoss() # 交叉熵损失函数
		self.embed_dim = embed_dim # MLP输出维度
		self.fun = nn.LeakyReLU(0.3) # LeakyReLU激活函数


		self.weight_mlp = nn.Parameter(torch.FloatTensor(self.embed_dim, num_classes)) #Default requires_grad = True
		self.weight_model = nn.Parameter(torch.FloatTensor((int(math.pow(2, K+1)-1) * self.embed_dim), 64))

		self.weight_model2 = nn.Parameter(torch.FloatTensor(64, num_classes))

		# print(self.embed_dim, num_classes)
		# print((int(math.pow(2, K+1)-1) * self.embed_dim), 64)
		# print(64, num_classes)

		
		init.xavier_uniform_(self.weight_mlp) # 使用Xavier初始化方法对模型权重初始化
		init.xavier_uniform_(self.weight_model)
		init.xavier_uniform_(self.weight_model2) 

	def forward(self, nodes, train_flag = True):

		embedding = self.agg(nodes, train_flag) # 输入节点得到节点嵌入

        # 2个全连接层
		scores_model = embedding.mm(self.weight_model) # 线性变换
		scores_model = self.fun(scores_model) # 激活函数
		scores_model = scores_model.mm(self.weight_model2)
		#scores_model = self.fun(scores_model)

        # 对embedding的前self.embed_dim维度进行线性变换和激活函数处理
		scores_mlp = embedding[:, 0: self.embed_dim].mm(self.weight_mlp)
		scores_mlp = self.fun(scores_mlp) # 送入多层感知机

		return scores_model, scores_mlp
		#dimension, the number of center nodes * 2
	
	def to_prob(self, nodes, train_flag = False):# 测试集输入到GNN模型中，得到模型预测的节点标签概率

		scores_model, scores_mlp = self.forward(nodes, train_flag)
		scores_model = torch.sigmoid(scores_model)# 将模型的输出分数通过 sigmoid 函数转换为概率
		return scores_model


	def loss(self, nodes, labels, train_flag = True):

		#the classification module

		scores_model, scores_mlp = self.forward(nodes, train_flag)

        # 分别与先验概率相加，并通过交叉熵损失函数计算模型的损失
		scores_model = scores_model + torch.log(self.prior)
		scores_mlp = scores_mlp + torch.log(self.prior)

		loss_model = self.xent(scores_model, labels.squeeze())
		#loss_mlp = self.xent(scores_mlp, labels.squeeze())
		final_loss = loss_model #+ self.lambda_1 * loss_mlp
		return final_loss

