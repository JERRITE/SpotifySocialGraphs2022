# This class is containing helper functions for the spotify project 2022 DTU
# -----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# for converting of data files
from matplotlib.colors import LogNorm
import pandas as pd
import ast
import numpy as np

# communities
import community.community_louvain as cl
import csv
import nltk

# coloring
from matplotlib import colors as mcolors

# graph
import networkx as nx
import matplotlib.pyplot as plt
from fa2 import ForceAtlas2
from scipy.stats import mode

# -----------------------------------------------------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------------------------------------------------

# all possible colors for community coloring
colors = list(dict(mcolors.XKCD_COLORS))  # all possible colors

# color shift in the list of colors
# the value 15 is for shifting the color scheme of the visualization since the first values in colors are black and white(very bad to see on the screen)
color_shift = 15

# shapes
shapes = ['o', 's', 'p',]

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS NETWORK BUILDING
# -----------------------------------------------------------------------------------------------------------------------

def get_edges_artists_relations(artists_relations):
    """Returns the edges of all related artists depending on the artists_relations dataframe"""
    # get all edges from artists_relations
    edges_related = []
    for c in range(len(artists_relations['id'])):
        artist_id = artists_relations['id'][c]
        related = artists_relations['related_artists'][c]
        related = ast.literal_eval(related)
        edges_related = edges_related + [(artist_id, i) for i in related]

    return edges_related


def get_edges_artists_colab(artists_data):
    """Returns the edges of all colaborating artists depending on the artists_data dataframe and if they worked together on a song"""
    # get all edges from the artists which worked together
    edges_colab = []
    names_overall = []
    id_overall = []

    for a in range(len(artists_data)):
        name = artists_data['id'][a]
        info = artists_data['top_tracks'][a]
        info = ast.literal_eval(info)

        # get all connections to artists
        outgoing = []
        for i in info:
            for j in i['artists']:
                if j != name:
                    outgoing.append(j)

        # only unique values
        outgoing = list(set(outgoing))

        # convert names to the id
        outgoing_id = []
        for i in outgoing:
            outgoing_id = outgoing_id + \
                artists_data['id'][artists_data.index[artists_data['name'] == i]].to_list()

        # save names overall
        id_overall = id_overall + outgoing_id

        # add to edges
        edges_colab = edges_colab + [(name, i) for i in outgoing_id]

    return edges_colab


def get_weighted_edges_from_list_of_edges(artists, edges):
    """Returns the weighted edges in form of ("id_a", "id_b", weight) based on the list of edges and artists data"""
    weight_edges = edges

    # create adj_matrix
    adj_matrix = []
    w, h = len(artists), len(artists)
    adj_matrix = [[0 for x in range(w)] for y in range(h)]

    # fill matrix
    for i, j in weight_edges:
        ind_i = artists.index[artists['id'] == i][0]
        ind_j = artists.index[artists['id'] == j][0]
        adj_matrix[ind_i][ind_j] += 1
        if ind_i != ind_j:
            adj_matrix[ind_j][ind_i] += 1

    # get edges from adj_matrix
    edges = []
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[0]) - i):
            # + 1 # we dont want to allow (1,4) and (4,1) cause undirected (go from the diagonal of the matrix)
            j = j+i
            # +1 because we don't want self edges
            # note that we substract i in the header of the for loop
            if adj_matrix[i][j] > 0:
                id_i = artists['id'][i]
                id_j = artists['id'][j]
                weight_value = adj_matrix[i][j]
                # save in edge
                edges.append((id_i, id_j, weight_value))

    return edges

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTION PLOTTING
# -----------------------------------------------------------------------------------------------------------------------

def get_heat_map(x_axis, y_axis, bin_size=25):
    """Returns the heatmap of data in x_axis and y_axis lists with number of bins set to bin_size"""
    heatmap, xedges, yedges = np.histogram2d(x_axis, y_axis, bins=bin_size)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    return heatmap, extent

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS COMMUNITIES
# -----------------------------------------------------------------------------------------------------------------------

def get_communities(graph, seed):
    """Returns communities in a graph"""
    # choose partition with as few communities as possible
    partition_related = cl.best_partition(graph, random_state=seed)
    return partition_related


def export_communities_to_file(graph, seed, name_str):
    """Exports communities in graph to a file name_str"""
    # get communities
    partition = get_communities(graph, seed)

    # instead dict of id:community_id get something like community_id:[id,id,...]
    partition = conv_community_dict(partition)

    # saving to file
    with open(str(name_str)+'.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for key, value in partition.items():
            writer.writerow([key, value])

def conv_community_dict(dict_com):
    """Converts and returns a dict of type {id: community_id} to: {community_id: [id, id, ...]}"""
    new_dict = {}

    for i in dict_com:
        id_com = dict_com[i]
        if id_com in new_dict:
            new_dict[id_com].append(i)
        else:
            new_dict[id_com] = [i]

    # get largest community sorted first
    def value_getter(item):
        return len(item[1])

    # sorted
    partition_sorted = sorted(
        new_dict.items(), key=value_getter, reverse=True)
    sorted_dict = {}
    for key, value in partition_sorted:
        sorted_dict[key] = value

    return sorted_dict


def give_amount_communities(graph, seed, bi_map, node_map):
    """Gets information about the number of nodes activated in bi_map per community and returns percentage of nodes per community"""
    # get communities
    com = get_communities(graph, seed)
    com = conv_community_dict(com)

    amount_dict = {}
    
    # calculate how many nodes are in a community
    for i in range(len(bi_map)):
        for j in list(com.keys()):
                x = node_map[i]
                y = com[j]
                if x in y:
                    if j in list(amount_dict.keys()):
                        # insert node
                        if bi_map[i] > 0.99:
                                amount_dict[j]["included"] = amount_dict[j]["included"] + 1
                    else:
                        amount_dict[j] = {}
                        if bi_map[i] > 0.99:
                                amount_dict[j]["included"] = 1
                        else:
                            amount_dict[j]["included"] = 0
                        amount_dict[j]["total"] = len(com[j])
                        
    # calculate a procentage
    for i in list(amount_dict.keys()):
        proc = (amount_dict[i]["included"]/amount_dict[i]["total"])*100
        amount_dict[i]["percentage"] = proc
    
    # get largest sorted first
    def value_getter(item):
        return item[0]

    amount_dict_sorted = sorted( amount_dict.items(), key=value_getter, reverse=False)
    sorted_dict = {}
    for key, value in amount_dict_sorted:
        sorted_dict[key] = value
        
    return sorted_dict


def get_most_genre(artists_wMood):
    """"Returns most frequent genres from artists_wMood"""
    # get all genre
    all_genre = get_all_genre(artists_wMood)

    for i in range(len(all_genre)):
        all_genre[i] = all_genre[i].replace("hip hop", "hip-hop") # necessary for recognizion as one word
        all_genre[i] = all_genre[i].replace("trap", "tr-ap") # necessary for recognizion trap not a rap -> is subgenre of hip-hop

    # find the 5 most frquent one
    def most_frequent_words(text):
        stop = nltk.corpus.stopwords.words()
        words = []
        for i in text:
            if not(i in stop) and any(c.isalpha() for c in i):
                words.append(i)
        fd = nltk.FreqDist(words)
        return [(w,num) for w, num in fd.most_common(50)]

    str_genre = []
    for i in all_genre:
        str_genre += i.split()
    most_genre = most_frequent_words(str_genre)

    return most_genre[:5], all_genre


def get_amount_genre_communities(artists_wMood, graph, seed):
    """Prints information about the amount of genres inside communities"""
    # get genres
    most_genre, all_genre = get_most_genre(artists_wMood)

    # create dict of all associated subgenres to the main(most_genre) genres
    all_genre_cat = {}
    for g in range(5):
        all_genre_cat[most_genre[g][0]] = []
        for i in all_genre:
            if most_genre[g][0] in i:
                all_genre_cat[most_genre[g][0]].append(i)

    for i in list(all_genre_cat.keys()):
        for j in range(len(all_genre_cat[i])):
            all_genre_cat[i][j] = all_genre_cat[i][j].replace("hip-hop", "hip hop") # replace again
            all_genre_cat[i][j] = all_genre_cat[i][j].replace("tr-ap", "trap") # replace again

    for i in list(all_genre_cat.keys()):
        print("Calculate node_count for genre: " + str(i) + "\n")
        # calculate color map depending on mood
        genre_bi_map = calculate_binar_map_genre(graph, artists_wMood, all_genre_cat[i])

        sorted_dict = give_amount_communities(graph, seed, genre_bi_map, list(graph.nodes()))
        
        print([(i, sorted_dict[i]) for i in list(sorted_dict.keys())])
        print("\n\n")


def get_amount_mood_communities(artists_wMood, graph, seed):
    """Prints information about the mood inside communities"""
    # get community - mood information about nodes
    all_moods = get_all_moods(artists_wMood)

    # print for each mood the amount of nodes inside every community
    for i in all_moods:
        print("Calculate node_count for mood: " + str(i) + "\n")
        # calculate color map depending on mood
        color_mood_map_related, alpha_mood_map_related = calculate_color_map_mood(graph, artists_wMood, mood=i)
        
        sorted_dict = give_amount_communities(graph, seed, alpha_mood_map_related, list(graph.nodes()))
        
        print([(i,sorted_dict[i]) for i in list(sorted_dict.keys())])
        print("\n\n")

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS COLORING
# -----------------------------------------------------------------------------------------------------------------------

def calculate_color_map_popularity(graph, artists_data):
    """Returns color map for nodes depending on the popularity of artists"""
    # color nodes
    color_map_popularity = []  # Use color_map_popularity as draw_networkx_nodes() input
    bias_popularity = 5
    scale_popularity = 1

    for i in graph.nodes():
        try:
            color_map_popularity.append(
                bias_popularity + (artists_data['popularity'][artists_data.index[artists_data['id'] == i].tolist()[0]])/scale_popularity)
        except:
            print("Error")
            color_map_popularity.append(0)

    return color_map_popularity


def calculate_color_map_genre(graph, artists_data, genre, node_color_map='blue', alpha=0.2, color_genre_node='red'):
    """Returns the color map for nodes depending on artist's genre from list of genres together with alpha values"""
    color_map_genre = calculate_binar_map_genre(graph, artists_data, genre)

    color_map_genre_alpha = []

    # alpha values
    for i in range(len(color_map_genre)):
        if color_map_genre[i] == 1:
            color_map_genre_alpha.append(1.0)
        else:
            color_map_genre_alpha.append(alpha)

    # color map
    map_color = []
    for i in range(len(color_map_genre)):
        if color_map_genre[i] == 1:
            map_color.append(color_genre_node)
        else:
            if isinstance(node_color_map, str):
                map_color.append(node_color_map)
            else:
                map_color.append(node_color_map[i])

    return map_color, color_map_genre_alpha


def calculate_color_map_communities(graph, seed):
    """Return color map for nodes depending on community they belong to"""
    # get communities
    partition = get_communities(graph, seed)
    partition = conv_community_dict(partition)

    # get largest community sorted first
    def value_getter(item):
        return len(item[1])

    # sorted
    partition_sorted = sorted(
        partition.items(), key=value_getter, reverse=True)
    sorted_dict = {}
    for key, value in partition_sorted:
        sorted_dict[key] = value

    # to list for indices
    part_sort = []
    for i in sorted_dict:
        part_sort.append(sorted_dict[i])

    # color
    color_map_communities = []
    color_count = 0

    for i in graph.nodes():
        for j in range(len(part_sort)):
            if i in part_sort[j]:
                try:
                    color_map_communities.append(colors[j + color_shift])
                except:
                    color_map_communities.append('black')
                break

    return color_map_communities


def calculate_color_map_mood(graph, artists_wMood, node_color_map=None, to_set_node_color='red', mood=None, alpha=0.2):
    """Returns the color map for nodes depending on artist's mood and alpha map if one mood is specified"""
    # all moods
    mood_map = calculate_map_mood(graph, artists_wMood)

    # all possible moods unique values
    moods = list(set(mood_map))

    color_mood_map = []
    alpha_map = []
    counter = 0

    for i in mood_map:
        for m in range(len(moods)):
            if str(i) == str(moods[m]):
                # node color
                if node_color_map == None:
                    color_mood_map.append(colors[m + color_shift])
                else:
                    if mood != None and i == mood:
                        color_mood_map.append(to_set_node_color)
                    else:
                        color_mood_map.append(node_color_map[counter])

                # if mood is given
                if mood != None and i == mood:
                    # if node_color_map != None:
                    # color_mood_map.append(to_set_node_color)
                    alpha_map.append(1)
                else:
                    # if node_color_map != None:
                    # color_mood_map.append(node_color_map[counter])
                    alpha_map.append(alpha)

        counter += 1

    return color_mood_map, alpha_map


def calculate_alpha_degree(graph):
    """Returns alpha map for nodes: 1 when degree larger than 0, otherwise 0"""
    # get alpha map
    alpha_degree = []

    for i in list(graph.nodes()):
        if graph.degree(i) > 0:
            alpha_degree.append(1.0)
        else:
            alpha_degree.append(0)

    return alpha_degree

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS NODESIZE
# -----------------------------------------------------------------------------------------------------------------------

def calculate_node_size_map_popularity(graph, artists_data):
    """Returns size map of nodes depending on popularity of artists"""
    # size nodes popularity
    size_map_poplarity = []  # Use size_map_popularity as draw_networkx_nodes() input
    bias_popularity = 5
    scale_popularity = 1

    for i in graph.nodes():
        try:
            size_map_poplarity.append(
                bias_popularity + (artists_data['popularity'][artists_data.index[artists_data['id'] == i].tolist()[0]])/scale_popularity)
        except:
            print("Error")
            size_map_poplarity.append(0)

    return size_map_poplarity


def calculate_node_size_map_degree(graph):
    """Returns size map of nodes depending on degrees"""
    # size nodes degree
    size_map_degree = []  # Use size_map_degree as draw_networkx_nodes() input
    scale_degree = 2
    bias_degree = 5

    for i in graph.nodes():
        size_map_degree.append(
            bias_degree + (scale_degree * len(graph.edges(i))))

    return size_map_degree

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS DEGREE DISTRIBUTION
# -----------------------------------------------------------------------------------------------------------------------

def get_degree_distribution(graph):
    """Calculates and returns degree distribution (histogram and bins) of the whole graph"""
    # get degree distribution
    in_degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    v_in = np.arange(min(in_degree_sequence), max(in_degree_sequence)+2, 1)
    hist_in_related, bin_edges_in_related = np.histogram(
        in_degree_sequence, bins=v_in)
    return hist_in_related, bin_edges_in_related


def get_degree_distribution_nodes(graph, node_list):
    """Calculates and returns degree distribution (histogram and bins) for the list of nodes from graph"""
    # get degree distribution
    degree_nodes = [graph.degree(n) for n in node_list]
    in_degree_sequence = sorted(degree_nodes, reverse=True)
    v_in = np.arange(min(in_degree_sequence), max(in_degree_sequence)+2, 1)
    hist_in_related, bin_edges_in_related = np.histogram(
        in_degree_sequence, bins=v_in)
    return hist_in_related, bin_edges_in_related


def draw_degree_distribution(graph, function=get_degree_distribution, node_list=None, title="Degree distributions of the network"):
    """Draws degree distribution of the graph"""
    # get degree distriution
    if node_list != None:
        hist_in_related, bin_edges_in_related = function(graph, node_list)
    else:
        hist_in_related, bin_edges_in_related = function(graph)

    # draw
    # %matplotlib inline
    plt.figure(figsize=(6, 6))
    plt.scatter(bin_edges_in_related[:-1],
                hist_in_related, label="degree distribution")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("degree")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.show()

    
def get_most_connected_artists(graph, artists_data, n=5):
    """Prints n most connected artists in graph"""
    degree_sequence = sorted([(d, i) for i, d in graph.degree()], reverse=True)
    for deg, id_ in degree_sequence[:n]:
        print("Name: " + artists_data[artists_data["id"] == id_]["name"].values[0] + ", degree: " + str(deg))


def get_degree_stats(graph):
    """"Returns mean, mode, median, minimum and maximum degree of nodes in graph"""
    degree_sequence = [d for n, d in graph.degree()]
    mean_ = np.mean(degree_sequence)
    median_ = np.median(degree_sequence)
    mode_ = mode(degree_sequence)
    min_ = min(degree_sequence)
    max_ = max(degree_sequence)
    return {"mean": mean_, "median": median_, "mode": mode_, "min": min_, "max": max_}


def calculate_centrality(graph, centrality_type = "degree_centrality"):
    """Calculate centrality given by centrality type of artists in graph"""
    if centrality_type == "degree_centrality":
        return [(c, i) for i, c in dict(nx.degree_centrality(graph)).items()]
    elif centrality_type == "betweenness_centrality":
        return [(c, i) for i, c in dict(nx.betweenness_centrality(graph)).items()]
    elif centrality_type == "eigenvector_centrality":
        return [(c, i) for i, c in dict(nx.eigenvector_centrality(graph, max_iter=200)).items()]  


def get_most_central_artist(graph, artists_data, centrality_sequence, n=5, centrality_type="degree_centrality"):
    """Return n most central artists in graph according to centrality type"""  
    sorted_centrality = sorted(centrality_sequence, reverse=True)
    for centrality, id_ in sorted_centrality[:n]:
        print("Name: {name}, {cent_type}: {cent}".format(name = artists_data[artists_data["id"] == id_]["name"].values[0], 
                                                         cent_type = centrality_type.replace("_", " ", ), cent = str(centrality)))

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTION NETWORK
# -----------------------------------------------------------------------------------------------------------------------

def calculate_position(graph, iterations=2000):
    """Calculates positions of nodes with forceatlas2"""
    # using Force Atlas 2 Algorithm
    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=True,  # Dissuade hubs
        linLogMode=False,  # NOT IMPLEMENTED
        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
        edgeWeightInfluence=1.0,

        # Performance
        jitterTolerance=1.0,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,  # NOT IMPLEMENTED

        # Tuning
        scalingRatio=1.0,
        strongGravityMode=False,
        gravity=10.0,

        # Log
        verbose=True,
    )

    positions = forceatlas2.forceatlas2_networkx_layout(
        graph, pos=None, iterations=iterations)

    return positions


def draw_network(graph, positions, seed=None, alpha_nodes=1, color_map_nodes='b', node_shape=shapes[0], node_size=5, alpha_edges=0.1, color_map_edges='black', labels=False, font_color='b'):
    """Draws the network"""
    # draw the network
    plt.figure(figsize=(18, 15), dpi=80)

    nx.draw_networkx_nodes(graph, positions, alpha=alpha_nodes, node_shape=node_shape,
                           node_color=color_map_nodes, node_size=node_size, cmap=plt.cm.Blues)
    nx.draw_networkx_edges(
        graph, positions, edge_color=color_map_edges, alpha=alpha_edges)

    if labels:
        partition_related = get_communities(graph, seed)
        nx.draw_networkx_labels(
            graph, pos=positions, labels=partition_related, font_size=10, font_color=font_color)

    plt.axis('off')
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTION BINARY
# -----------------------------------------------------------------------------------------------------------------------

def calculate_binar_map_mood(graph, artists_wMood, mood):
    """Returns a binary map: 1 if the node has this mood and otherwise 0"""
    mood_map = calculate_map_mood(graph, artists_wMood)

    binar_map = []
    for i in mood_map:
        if i == mood:
            binar_map.append(1)
        else:
            binar_map.append(0)

    return binar_map


def calculate_map_mood(graph, artists_wMood):
    """Returns a binary map of mood: 1 if it's node's mood and 0 otherwise"""
    mood_map = []

    for i in graph.nodes():
        try:
            mood_map.append(
                (artists_wMood['mood'][artists_wMood.index[artists_wMood['id'] == i]].to_list())[0])
        except:
            mood_map.append("")

    return mood_map


def calculate_binar_map_genre(graph, artists_data, genre):
    """Returns a color map of genres"""
    color_map_genre = []
    count = 0
    for i in graph.nodes():
        color_map_genre.append(0)
        for gen in genre:
            if gen in ast.literal_eval((artists_data['genres'][artists_data.index[artists_data['id'] == i]].to_list())[0]):
                color_map_genre[count] = color_map_genre[count] + 1
        count += 1

    return color_map_genre


def get_all_nodes_with_sentiment(artists_wMood, sentiment):
    """Return all nodes with sentiment in specified range"""
    sentiment_nodes = artists_wMood['id'][(sentiment[0] <= artists_wMood['avg_lyrics_sentiment']) & (artists_wMood['avg_lyrics_sentiment'] <= sentiment[1])].tolist()
    
    return sentiment_nodes


def get_node_id_from_binary_map(graph, binary_map):
    """Returns id of nodes from hte binary map where values are greater than zero"""
    list_nodes = []
    graph_nodes = list(graph)

    for i in range(len(binary_map)):
        if binary_map[i] > 0:
            list_nodes.append(graph_nodes[i])

    return list_nodes

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTION HELPER
# -----------------------------------------------------------------------------------------------------------------------

def get_all_genre(artists_data):
    """Returns list of all genres in artists data"""
    all_genres = []
    genres_data = artists_data['genres'].tolist()
    genres_data = [ast.literal_eval(i) for i in genres_data]

    for i in range(len(genres_data)):
        all_genres += genres_data[i]

    all_genres = list(set(all_genres))
    return all_genres


def get_all_moods(artists_wMood):
    """Returns all moods from artists data"""
    all_moods = artists_wMood['mood'].tolist()
    all_moods = list(set(all_moods))
    for i in range(len(all_moods)):
        if str(all_moods[i]) == "nan":
            del all_moods[i]
            break
    return all_moods


def remove_suffix(input_string, suffix):
    """Removes suffix from input_string. Needed in lower python versions."""
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string
# -----------------------------------------------------------------------------------------------------------------------
