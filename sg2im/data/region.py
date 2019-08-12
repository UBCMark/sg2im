import json
with open('/scratch/markma/Datasets/vg/region_descriptions.json', 'r') as f:
  regions = json.load(f)

object_ids = [i['regions'] for i in regions if i['id'] == 2407890]

print(object_ids)
print(len(object_ids[0]))
