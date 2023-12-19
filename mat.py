from scipy.io import loadmat

amz = loadmat('data/Amazon.mat')
# print(amz['homo'])
# print(amz['net_upu'])
# print(amz['net_usu'])
# print(amz['net_uvu'])
feat_data = amz['features'].toarray()
# print(feat_data)
labels = amz['label'].flatten()
print(labels)