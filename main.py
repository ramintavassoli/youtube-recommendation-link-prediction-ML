import pandas as pd
import numpy as np
import networkx as nx
from multiprocessing import Pool

df = pd.read_csv('train_data.csv', header=0, sep=',').fillna(-1)
recommendation = df[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']].as_matrix()
#cat = df['category'].values.tolist()
nodes = df['video_id'].values.tolist()
node_dict = {}

for i in range(960000):
    node_dict[nodes[i]] = i

k = 1
for i in range(960000):
    for j in range(20):
        if recommendation[i][j] == -1:
            continue
        else:
            try:
                recommendation[i][j] = node_dict[recommendation[i][j]]
            except KeyError:
                node_dict[recommendation[i][j]] = 960000 + k
                recommendation[i][j] =  node_dict[recommendation[i][j]]
                k += 1
rec = recommendation.tolist()

dft = pd.read_csv('test_data.csv', header=0, sep=',')
t_nodes = dft[['source','target']].as_matrix()

def ftn(row):
    A = []
    B = []
    try:
        if node_dict[row[0]]<960000:
            A = rec[node_dict[row[0]]]
            for j in range(960000)
                if node_dict[row[0]] in rec[j]:
                    A.append(j)
                    A.extend(rec[j])
        else:
            for j in range(960000):
                if node_dict[row[0]] in rec[j]:
                    A.append(j)
                    A.extend(rec[j])
    except KeyError:
        return -1
    try:
        if node_dict[row[1]]<960000:
            B = rec[node_dict[row[1]]]
            for j in range(960000):
                if node_dict[row[1]] in rec[j]:
                    B.append(j)
                    B.extend(rec[j])
        else:
            for j in range(960000):
                if node_dict[row[1]] in rec[j]:
                    B.append(j)
                    B.extend(rec[j])
    except KeyError:
        return -1
    Jacc  = len(list(set(A) & set(B)))
    return Jacc

if __name__ == '__main__':
    a = t_nodes.tolist()
    pool = Pool()
    result = pool.map(ftn, a)
ins = np.asarray(result)
np.savetxt("Jacc.csv", ins, delimiter=",")
            
#building the graph usinng networkx
#adding the keys of the dictionary into a list of nodes to insert into DG
node_lst = []
for key, value in node_dict.iteritems():
    node_lst.append(value)

G = nx.MultiGraph()
G.add_nodes_from(node_lst)

for i in range(960000):
    edge_weights = [1]*20
    node_i = [i for j in range(20)]
    neighbor_i = [x for x in rec[i]]
    null_indecies = [j for j,x in enumerate(neighbor_i) if x == -1]
    #removing null indicies
    node_i= [x for j,x in enumerate(node_i) if j not in null_indecies]
    edge_weights = [x for j,x in enumerate(edge_weights) if j not in null_indecies]
    neighbor_i = [x for j,x in enumerate(neighbor_i) if j not in null_indecies]
    zipped = zip(node_i, neighbor_i, edge_weights)
    G.add_weighted_edges_from(zipped, weight= 'weight')

dft = pd.read_csv('test_data.csv', header=0, sep=',').fillna(-1)
t_nodes = dft[['source','target']].as_matrix()
for node in t_nodes:
    for j in range(2):
        try:
            node[j] = node_dict[node[j]]
        except KeyError:
            node[0] = 92
            node[1] = 92 #ensuring that there is a path between them of length 0

def ftn (row):
    try:
        length = nx.shortest_path_length(G, source = row[0], target = row[1], weight = 'weight')
    except nx.NetworkXNoPath:
        length = 10000
    return length

if __name__ == '__main__':
    a = t_nodes.tolist()
    pool = Pool()
    result = pool.map(ftn, a)
#
#def ftn (duo):
#    decision = 0
#    try:
#        if component_dict[duo[0]] != component_dict[duo[1]]:
#            decision = 0
#        else:
#            if component_dict[duo[0]] != 0:
#                decision = 1
#            elif:
#                try:
#                    if len(nx.shortest_path_length(G, source = duo[0], target = duo[1], weight = 'weight'))< 50:
#                        decision = 1
#                except NetworkXNoPath:
#                    pass
#    except KeyError:
#        pass
#    return decision

#=========================Conductance Minimization================================
# for node in G.nodes():
#     if G.degree(node) == 0:
#         G.remove_node(node)
# 
# con_components = list(nx.connected_component_subgraphs(G))
# component_number = 0
# component_dict = {}
# 
# for component in con_components:
#     nodes_ = component.nodes()
#     for node in nodes_:
#         component_dict[node] = component_number
#     component_number += 1
#     
# def component_zero(comp):
#     label_dict = {}
#     LM = nx.laplacian_matrix(comp)
#     eigenValues,eigenVectors = sparse.linalg.eigsh(LM,k=3,which='SM')
#     idx = eigenValues.argsort()
#     eigenVectors = eigenVectors[:,idx[1:3]]
#     nodes_ = comp.nodes()
#     clust_list = []
#     for j in range(len(nodes_)):
#         clust_list.append(eigenVectors[j])
#     clust_list_arr = np.asarray(clust_list)
#     kmeans = KMeans(n_clusters= 100, random_state=0).fit(clust_list_arr)
#     i = 0
#     for node in nodes:
#         label_dict[node] = kmeans.labels_[i]
#         i += 1
#     return label_dict
# 
# label_dict = component_zero(con_components[0])
# 
# dft = pd.read_csv('test_data.csv', header=0, sep=',').fillna(-1)
# test_nodes = dft[['source','target']].as_matrix()
# final_solution = []
# for i in range(100000):
#     try:
#         if component_dict[node_dict[test_nodes[i][0]]] != component_dict[node_dict[test_nodes[i][1]]]:
#             final_solution.append([i,0])
#         else:
#             if component_dict[node_dict[test_nodes[i][1]]] != 0:
#                 final_solution.append([i,1])
#             else:
#                 if label_dict[node_dict[test_nodes[i][0]]] == label_dict[node_dict[test_nodes[i][1]]]:
#                     final_solution.append([i,1])
#     except KeyError:
#         final_solution.append([i,0])
# 
# a = np.asarray(final_solution)
# np.savetxt("foo.csv", a, delimiter=",")
#==============================================================================

# for component in con_components:
#     fiedler[i] = nx.fiedler_vector(component)
#     i += 1
# getting the tuples for the edges for DG
# temp_node_i = []
# edge_weights = [np.exp(i) for i in reversed(range(20))]
# for i in range(26901):; 
#     temp_node_i = [i for j in range(20)]
#     temp_neigh = rec[i]
#     zipped = zip(temp_node_i, temp_neigh, edge_weights)
#     DG.add_weighted_edges_from(zipped)
# nx.write_gml(DG,"test.gml")
