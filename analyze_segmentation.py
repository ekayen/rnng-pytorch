import os
import tree_utils
import pickle

import tempfile

tmpdir = tempfile.TemporaryDirectory(prefix="evalb-")

"""
Analyze where the parser puts sentence breaks, relative to certain prosodic feature
or alternatively, relative to certain constituents.

"""


turndir = "/afs/inf.ed.ac.uk/group/project/prosody/prosody_nlp/data/input_features/turn_pause_dur_fixed"
turn2part = pickle.load(open(os.path.join(turndir,'turn_dev_partition.pickle'),'rb'))
turn2pitch = pickle.load(open(os.path.join(turndir,'turn_dev_pitch.pickle'),'rb'))
turn2fbank = pickle.load(open(os.path.join(turndir,'turn_dev_fbank.pickle'),'rb'))
turn2pause = pickle.load(open(os.path.join(turndir,'turn_dev_pause.pickle'),'rb'))
turn2dur = pickle.load(open(os.path.join(turndir,'turn_dev_duration.pickle'),'rb'))
turn_ids = [l.strip() for l in open(os.path.join(turndir,'turn_dev_sent_ids_medium.txt'),'r').readlines()]
turn_trees = [l.strip() for l in open(os.path.join(turndir,'turn_dev_medium.trees'),'r').readlines()]



def get_wd_len(constituent):
    leaves = 0
    for leaf in constituent.leaves():
        leaves += 1
    return leaves

def get_sent_break_idx(tree):
    if len(tree.children) == 1:
        return False
    else:
        idxs = []
        for i,child in enumerate(tree.children):
            left_leaf_len = sum([get_wd_len(tree.children[j]) for j in range(i+1)])
            idxs.append(left_leaf_len-1)
        return idxs

def get_break_pauses(idx,pauses):
    aft = pauses['pause_aft']
    break_pauses = [aft[i] for i in idx]
    return break_pauses

def count_pauses(pauses):
    pause_counts = {}
    for pause in pauses:
        if pause in pause_counts:
            pause_counts[pause] += 1
        else:
            pause_counts[pause] = 1
    return pause_counts

def count_leaves(constituent):
    constituent = constituent.replace('(','').replace(')','').replace('£','').replace('$','')
    constituent = ''.join(ch for ch in constituent if not ch.isupper())
    return len(constituent.split())


def get_edit_idxs(tree):
    num_leaves = len(list(tree.leaves()))
    tree_string = tree.linearize()
    
    if 'EDITED' in tree_string:
        pre_edit_idx = []
        post_edit_idx = []

        tree_string = tree_string.replace('EDITED','£')
        for i,char in enumerate(tree_string):
            if char == '£':
                prefix = tree_string[:i]
                pre_edit_idx.append(count_leaves(prefix)-1)
                edited_span = []
                open_paren_stack = ['(']
                j = 1
                while open_paren_stack:
                    next_char = tree_string[i+j]
                    edited_span.append(next_char)
                    if next_char == '(':
                        open_paren_stack.append('(')
                    elif next_char == ')':
                        open_paren_stack.pop()
                    j += 1
                
                edited_span = ''.join(edited_span)
                post_edit_idx.append(count_leaves(prefix)+count_leaves(edited_span)-1)
        return pre_edit_idx,post_edit_idx
    return None

def intersection_size(lst1, lst2):
    return len(list(set(lst1) & set(lst2)))

def add_top(line):
    if not line.startswith('(TOP (TURN'):
        new_line = f'(TOP (TURN {line} ))'
    else:
        new_line = line
    return new_line

def main():

    #output_dir = "/afs/inf.ed.ac.uk/group/project/prosody/rnng-pytorch/output/context"
    output_dir = "/afs/inf.ed.ac.uk/group/project/prosody/rnng-pytorch/output"

    datadir = "/afs/inf.ed.ac.uk/group/project/prosody/rnng-pytorch/data/swbd/turn"
    pred_tree_file = os.path.join(output_dir,"turn_speech.pred")
    #pred_tree_file = os.path.join(output_dir,"b1f0.pred")
    gold_tree_file = os.path.join(datadir,"dev.trees")
    id_file = os.path.join(datadir,"dev_sent_ids.txt")

    print(pred_tree_file.split('/')[-1])
    pred_trees,ids = tree_utils.load_trees_with_idx(pred_tree_file,id_file)
    gold_trees,ids = tree_utils.load_trees_with_idx(gold_tree_file,id_file)

    correct_splits = 0
    incorrect_splits = 0    
    total_pred_splits = 0
    total_gold_splits = 0
    
    turn_med_breaks = 0
    sent_break_pauses = []

    pre_edit_breaks = 0
    post_edit_breaks = 0

    total_turn_medial_positions = 0 # number of positions between words that sentence breaks could go. sum(len(turn)-1) over all turns
    
    total_pred_breaks = 0
    total_gold_edits = 0

    top_remains = 0
    post_edit_top_remains = 0

    turn2IPSUconf = {}
    
    for i,tree in enumerate(pred_trees):
        gold = gold_trees[i]
        gold_num_sents = len(gold.children)
        pred_num_sents = len(tree.children)

        total_pred_splits += pred_num_sents - 1
        total_gold_splits += gold_num_sents - 1
        pred_leaves = []
        gold_leaves = [] 
        pred_break_idxs = set()
        gold_break_idxs = set()         
        for j in range(pred_num_sents-1):
            pred_leaves.extend([leaf for leaf in tree.children[j].leaves()])
            pred_break_idxs.add(len(pred_leaves))
        for j in range(gold_num_sents-1):
            gold_leaves.extend([leaf for leaf in gold.children[j].leaves()])
            gold_break_idxs.add(len(gold_leaves))                
        correct_splits += len(pred_break_idxs.intersection(gold_break_idxs))
        incorrect_splits += len(pred_break_idxs-gold_break_idxs)

            
        if tree.label == 'TOP': top_remains += 1
        total_turn_medial_positions += (get_wd_len(tree)-1)
        break_idxs = get_sent_break_idx(tree)
        gold_edit_idxs = get_edit_idxs(gold_trees[i])
        if gold_edit_idxs: total_gold_edits += len(gold_edit_idxs[0])
        if break_idxs:
            total_pred_breaks += len(break_idxs)-1
            turn_id = ids[i]
            sent_break_pauses.extend(get_break_pauses(break_idxs[:-1],turn2pause[turn_id])) #Q: what pauses happen at sent breaks?
            turn_med_breaks += len(break_idxs[:-1])
        if gold_edit_idxs and break_idxs:
            pre_edit_idxs = gold_edit_idxs[0]
            post_edit_idxs = gold_edit_idxs[1]
            pre_edit_breaks += intersection_size(pre_edit_idxs,break_idxs)
            post_edit_breaks += intersection_size(post_edit_idxs,break_idxs)
            if  intersection_size(post_edit_idxs,break_idxs) >0 and tree.label=='TOP':
                post_edit_top_remains += 1

            idnum = ids[i]
            if intersection_size(post_edit_idxs,break_idxs) > 0:
                turn2IPSUconf[idnum] = set(post_edit_idxs).intersection(set(break_idxs))
                
    with open('b0f0-turn2IPSUconf.pickle','wb') as f:
        pickle.dump(turn2IPSUconf,f)

                
    print(f'{correct_splits} correct splits')
    print(f'{incorrect_splits} incorrect splits')

    precision = correct_splits / total_pred_splits
    recall = correct_splits / total_gold_splits
    f1 = (2*precision*recall)/(precision+recall)
    
    print(f'seg. precision: {precision}')
    print(f'seg. recall: {recall}')
    print(f'seg. f1: {f1}')

    pause_counts = count_pauses(sent_break_pauses)
    print(pred_tree_file.split('/')[-1])
    print(f'Trees where TOP remains (predicted early TURN closure): {top_remains}')
    print(f'Predicted early closure + predicted break post-edit: {post_edit_top_remains}')
    print('Pauses at turn-internal sentence breaks:')
    print(pause_counts)
    print(f'Turn-medial breaks: {turn_med_breaks}')
    print(f'Total edits: {total_gold_edits}')
    print(f'Predicted breaks pre-edits: {pre_edit_breaks}')
    print(f'Predicted breaks post-edit: {post_edit_breaks}')
    print(f'Total turn medial positions: {total_turn_medial_positions}')


if __name__=='__main__':
    main()
