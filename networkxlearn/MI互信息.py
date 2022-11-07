import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import add


df = pd.read_excel('./案例数据/MI.xlsx')
x = df['BNSS1']
y = df['BNSS2']

nmi = add.NMI(x,y)
# print(nmi)
# print(normalized_mutual_info_score(x,y))
# 赋值nodes列表
BNSS = list(df.columns)
print(BNSS)
# 生成完全网络
G = nx.complete_graph(BNSS)


# 赋值连边权重
for u,v,d in G.edges(data=True):
     d['weight'] = normalized_mutual_info_score(df[u],df[v])
weight_BNSS = list(nx.get_edge_attributes(G,'weight').values())
# 按照连边权重设置连边粗细
edgewidth=[]
for i in range(len(weight_BNSS)):
    edgewidth.append(weight_BNSS[i]*3)

# 计算宏观属性
# 密度
density = nx.density(G)
# 平均最短路径长度
shortest_path_lenth = nx.average_shortest_path_length(G,weight='weight')
# 平均集聚系数
cluster = nx.average_clustering(G)



# 计算微观属性：网络中心性指标
dc = add.degree_centrality_weight(G)
bc = nx.betweenness_centrality(G,weight='weight')
cl = nx.clustering(G,weight='weight')
dc_list = list(dc.values())
bc_list = list(bc.values())
cl_list = list(cl.values())




cmap = plt.get_cmap('YlGn')
nx_att = {
    'font_size':5,
    'node_size':300,
    'edge_color':'#DCDCDC',
    'width':edgewidth,
    'cmap':cmap,
    'alpha': 0.7
}

fig = plt.figure(figsize=[18, 6])
ax1 = fig.add_subplot(2,2,1)
# ax1 = plt.subplot(2,2,1)
nx.draw_networkx(G,node_color=dc_list,**nx_att)
norm_1= mpl.colors.Normalize(vmin=np.min(dc_list),vmax=np.max(dc_list))
plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=norm_1),
             label='degree centrality',extend='both',ax=ax1)

ax2 = fig.add_subplot(2,2,2)
nx.draw_networkx(G,node_color=bc_list,**nx_att)
norm_2= mpl.colors.Normalize(vmin=np.min(bc_list),vmax=np.max(bc_list))
plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=norm_2),
             label='betweenness centrality',extend='both',ax=ax2)

ax3 = fig.add_subplot(2,2,3)
nx.draw_networkx(G,node_color=cl_list,**nx_att)
norm_3= mpl.colors.Normalize(vmin=np.min(cl_list),vmax=np.max(cl_list))
plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=norm_3),
             label='clustering',extend='both',ax=ax3)


ax4 = fig.add_subplot(2,2,4)
x = range(len(BNSS))
plt.scatter(x,dc_list,c='blue')
plt.scatter(x,bc_list,c='green')
plt.scatter(x,cl_list,c='red')
plt.xticks(x,BNSS,rotation=45,fontsize=7)
plt.title('Microscopic network properties of controls')
plt.legend(["degree centrality","betweenness centrality",'clustering'],fontsize=7)
plt.show()