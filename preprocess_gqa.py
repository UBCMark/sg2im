import json
from itertools import chain
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('GTK3Agg')

def json_load(file_dir):
    with open(file_dir, 'r') as f:
        data = json.load(f)
    return data

def json_save(data, file_dir):
    with open(file_dir, 'w') as f:
        json.dump(data, f)

def get_object_id(graph):
    # input graph : dictionary
    # return object id :  dictionary
    #   key : Object Id, value : Object Name
    object_dict = {}
    for imgID in graph.keys():
        image = graph[imgID]
        objects = image["objects"]
        for object_id in objects.keys():
            object_name = objects[object_id]["name"]
            object_dict[object_id] = object_name

    return object_dict

def get_whole_relation_graph(graph, id_dict):
    ##### input
    # graph : dictionary
    # id_dict : dictionary
    ##### output
    # relation_dict : dictionary
    #   key : object name
    #   value : dictionary
    #               key : target object name
    #               value : relation name
    relation_dict = {}
    attribute_name_counter = Counter()
    for imgID in graph.keys():
        img = graph[imgID]
        objects = img["objects"]
        for object_id in objects.keys():
            source_name = objects[object_id]["name"]
            relation_candidate = objects[object_id]["relations"]        # list [ {"object" : "name"} , { "object" : "name" } ,  ..... ]

            # for candidate in relation_candidate:
            #     target_id = candidate["object"]
            #     target_name = id_dict[target_id]
            #     relation =  candidate["name"]
            #
            #     # if there is no key value, add it.
            #     if source_name not in relation_dict.keys():
            #         relation_dict[source_name] = defaultdict(list)
            #
            #     #
            #     relation_dict[source_name][target_name].append(relation)
            #     relation_dict[source_name][target_name] = list(set(relation_dict[source_name][target_name]))
            names = set()
            for name in objects[object_id]["attributes"]:
                names.add(name.strip(' .').lower())
            attribute_name_counter.update(names)

    attribute_names = []
    min_attribute_instances = 1000
    for v, (name, count) in enumerate(attribute_name_counter.most_common()):
        if count >= min_attribute_instances and v < 128:
            attribute_names.append(name)
    print('Found %d attribute categories with >= %d training instances' %
          (len(attribute_names), min_attribute_instances))

    attribute_name_to_idx = {}
    attribute_idx_to_name = []
    for idx, name in enumerate(attribute_names):
        attribute_name_to_idx[name] = idx
        attribute_idx_to_name.append(name)

    print(attribute_name_counter.most_common(128))
    labels, values = zip(*attribute_name_counter.most_common(128))

    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()


    return attribute_name_counter

if __name__ == '__main__':
    # 1. graph data load
    graph_data = {}
    train_graph = json_load('../sg2im/data/sceneGraphs/train_sceneGraphs.json')
    val_graph = json_load('../sg2im/data/sceneGraphs/val_sceneGraphs.json')

    graph_data.update(val_graph)
    graph_data.update(train_graph)  #최종적으로 train_graph 기준으로 dictionary가 update가 된다.

    print('total number of ImageId : ', len(list(graph_data.keys())))

    # 2. Collect object name and id -> dictionary ( key : ObjectID , value : Object Name )
    object_id_dict = get_object_id(train_graph)

    # 3. Object , relation, target 으로 dictionary 만들기.
    relation_graph = get_whole_relation_graph(graph_data, object_id_dict)

    # 4. save the results into json file
    # json_save(object_id_dict, './result/object_id.json')
    # json_save(relation_graph, './result/relation_graph.json')