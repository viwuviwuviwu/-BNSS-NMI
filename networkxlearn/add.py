from collections import Counter
import numpy as np
import networkx as nx

__all__ = [
    'NMI',
    'degree_centrality_weight',
    'communities_inter_degree',
    'communities_outer_degree'

]
def NMI(x,y):
    # 计算x中每个元素出现的次数
    count_x = Counter(x)
    # 计算x中每个元素出现的概率
    p_x = dict()
    for i in count_x.keys():
        p_x[i] = count_x[i] / len(x)
    H_x = 0
    for i in p_x.keys():
        H_x += p_x[i] * np.log(p_x[i])
    # 计算y中每个元素出现的次数
    count_y = Counter(y)
    # 计算y中每个元素出现的概率
    p_y = dict()
    for i in count_y.keys():
        p_y[i] = count_y[i] / len(y)
    H_y = 0
    for i in p_y.keys():
        H_y += p_y[i] * np.log(p_y[i])

    # 计算每一个（x，y）元素出现的次数
    d_xy = dict()
    for i in range(len(x)):
        if (x[i], y[i]) in d_xy:
            d_xy[x[i], y[i]] += 1
        else:
            d_xy[x[i], y[i]] = 1

    # 计算每一个（x，y）元素出现的概率
    p_xy = dict()
    for xy in d_xy.keys():
        p_xy[xy] = d_xy[xy] / len(x)

    # 初始化互信息值为0
    mi = 0
    for xy in p_xy.keys():
        mi += p_xy[xy] * np.log(p_xy[xy] / (p_x[xy[0]] * p_y[xy[1]]))  # 互信息公式
    # 归一化
    nmi = mi / np.sqrt(H_x * H_y)
    return nmi

def degree_centrality_weight(G):
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {}
    sum_weight = 0
    for n in G.nodes:
        for e in G.edges:
            if n in e:
                sum_weight += nx.get_edge_attributes(G, 'weight')[e]
        centrality[n] = sum_weight
        sum_weight = 0
    return centrality


# 计算社团内部边数（连接节点到其社区内其他节点的边的和）
def communities_inter_degree(G,communities):
    inter_degree = {}
    for community in communities:
        sum_inter_degree = 0.0
        for u, v, w in G.edges(community, data='weight', default=1):
            if v in community:
                sum_inter_degree += w
                x = G.nodes(data='community')[v]
        inter_degree[x] = sum_inter_degree
    return inter_degree


# 计算社团外部边数（连接节点与来自其他社区的节点的边之和）
def communities_outer_degree(G,communities):
    outer_degree = {}
    for community in communities:
        sum_degree = 0.0
        for node in community:
            sum_degree += G.degree(node, weight='weight')
            x = G.nodes(data='community')[node]
        outer_degree[x] = sum_degree - communities_inter_degree(G,communities)[x]*2
    return outer_degree
