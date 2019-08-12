import json

with open('/scratch/zhaobo/Datasets/vg/objects.json', 'r') as f:
  objects = json.load(f)

object_ids = [i['objects'] for i in objects if i['image_id'] == 2407890]

print(object_ids) 
print(len(object_ids[0]))
