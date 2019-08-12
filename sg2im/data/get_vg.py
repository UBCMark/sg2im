from vg import VgSceneGraphDataset
import json

with open("vocab.json", 'r') as f:
    vocab = json.load(f)
dataset = VgSceneGraphDataset(vocab=vocab, h5_path="train.h5", image_dir="/scratch/zhaobo/Datasets/vg/images/VG_100K")

index = [0,10]
out = dataset.getitem() 
