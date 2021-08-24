import json
import os
import numpy as np
from tree_utils import *
import pickle

datadir = '/afs/inf.ed.ac.uk/group/project/prosody/rnng-pytorch/data/swbd/turn'
datafile = os.path.join(datadir,'swbd-dev.json')
idfile = os.path.join(datadir,'dev_sent_ids.txt')
treefile = os.path.join(datadir,'dev.trees')

trees,ids = load_trees_with_idx(treefile, idfile)

id2tree = dict(zip(ids,trees))

turns = {}
with open(datafile,'r') as f:
    for line in f.readlines():
        turndict = json.loads(line)
        if 'idnum' in turndict:
            turnid = turndict['idnum']
            turns[turnid] = turndict

def get_constituent_len(const):
    counter = 0
    for leaf in const.leaves():
        counter += 1
    return counter
    
def find_sent_ends(tree):
    sent_end_idxs = []
    idx = 0
    children = tree.children
    for child in children:
        idx += get_constituent_len(child) 
        sent_end_idxs.append(idx-1)
    return sent_end_idxs


sent_final_resets = []
elsewhere_resets = []

for turn in turns:
    tree = id2tree[turn]
    pitch = [np.array(tok) for tok in turns[turn]['pitch']]
    sent_final_idxs = find_sent_ends(tree)
    for i,tok in enumerate(pitch):
        if i < len(pitch)-1:
            final_pit = tok[:,-1]
            initial_pit = pitch[i+1][:,0]
            if i in sent_final_idxs:
                sent_final_resets.append(abs(final_pit-initial_pit))
            else:
                elsewhere_resets.append(abs(final_pit-initial_pit))

print('mean sent final resets:')
print(np.mean(sent_final_resets,axis=0))
print('mean elsewhere resets:')
print(np.mean(elsewhere_resets,axis=0))

with open('b0f0-turn2IPSUconf.pickle','rb') as f:
    b0f0IPSU = pickle.load(f)
with open('b0f1-turn2IPSUconf.pickle','rb') as f:
    b0f1IPSU = pickle.load(f)


fixable_mistakes = []
unfixable_mistakes = []
    
for turn in turns:
    tree = id2tree[turn]
    pitch = [np.array(tok) for tok in turns[turn]['pitch']]
    if turn in b0f0IPSU:
        idxs = b0f0IPSU[turn]
        for idx in idxs:
            if not turn in b0f1IPSU:
                tok1 = pitch[idx][:,-1]
                tok2 = pitch[idx+1][:,0]
                fixable_mistakes.append(abs(tok2-tok1))
            elif not idx in b0f1IPSU[turn]:
                tok1 = pitch[idx][:,-1]
                tok2 = pitch[idx+1][:,0]
                fixable_mistakes.append(abs(tok2-tok1))
    if turn in b0f1IPSU:
        idxs = b0f1IPSU[turn]
        for idx in idxs:
            tok1 = pitch[idx][:,-1]
            tok2 = pitch[idx+1][:,0]
            unfixable_mistakes.append(abs(tok2-tok1))
        
print('fixable resets:')
print(np.mean(fixable_mistakes,axis=0))
print('unfixable resets:')
print(np.mean(unfixable_mistakes,axis=0))
import pdb;pdb.set_trace()
