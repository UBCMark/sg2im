from collections import Counter, defaultdict
import json, os

print('Loading attributes from /scratch/zhaobo/Datasets/vg')
with open("/scratch/zhaobo/Datasets/vg/attributes.json", 'r') as f:
    attributes = json.load(f)
attribute_name_counter = Counter()
len(attribute_name_counter)
aaa = set()
for image in attributes:
    # if image['image_id'] not in image_ids:
    #     continue

    for attribute in image['attributes']:
        names = set()
        try:
            for name in attribute['attributes']:
                aaa.add(name)
                names.add(name)
            attribute_name_counter.update(names)
        except KeyError:
            pass
print("len of original atts %d" % len(attribute_name_counter))
print("len of original aaa %d" % len(aaa))

min_attribute_instances = 5
attribute_names = []
for name, count in attribute_name_counter.most_common():
    if count >= min_attribute_instances:
        attribute_names.append(name)
print('Found %d attribute categories with >= %d training instances' %
      (len(attribute_names), min_attribute_instances))

print(
      sum(value == 1 for value in attribute_name_counter.values()))

