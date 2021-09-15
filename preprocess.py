#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create data files
"""

import os
import sys
import argparse
import numpy as np
import pickle
import itertools
from collections import defaultdict
import utils
import re
import shutil
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from multiprocessing import Pool
import itertools
import torch
from skimage.measure import block_reduce

from data import Sentence, Vocabulary
from action_dict import TopDownActionDict, InOrderActionDict

pad = '<pad>'
unk = '<unk>'

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')

def get_next_bracket_index(line, start_idx):
    for i in range(start_idx+1, len(line)):
        char = line[i]
        if char == '(' or char == ')':
            return i
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)    
    return ''.join(output)

def get_tags_tokens_lowercase(line):
    output = []
    line = line.rstrip()
    for i in range(len(line)):
        if i == 0:
            assert line[i] == '('    
        if line[i] == '(' and not(is_next_open_bracket(line, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        # print(terminal, terminal_split)
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word        
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]

def transform_to_subword_tree(line, sp):
    line = line.rstrip()
    tags, tokens, _ = get_tags_tokens_lowercase(line)
    pieces = sp.encode(' '.join(tokens), out_type=str)
    end_idxs = [i+1 for i, p in enumerate(pieces) if '▁' in p]
    begin_idxs = [0] + end_idxs[:-1]
    spans = list(zip(begin_idxs, end_idxs))  # map from original token idx to piece span idxs.

    def get_piece_preterms(tok_i):
        tag = tags[tok_i]
        b, e = spans[tok_i]
        span_pieces = pieces[b:e]
        return ' '.join(['({} {})'.format(tag, p) for p in span_pieces])

    new_preterms = [get_piece_preterms(i) for i in range(len(tokens))]
    orig_token_spans = []
    for i in range(len(line)):
        if line[i] == '(':
            next_bracket_idx = get_next_bracket_index(line, i)
            found_bracket = line[next_bracket_idx]
            if found_bracket == '(':
                continue  # not terminal -> skip
            orig_token_spans.append((i, next_bracket_idx+1))
    assert len(new_preterms) == len(orig_token_spans)
    ex_span_ends = [span[0] for span in orig_token_spans] + [len(line)]
    ex_span_begins = [0] + [span[1] for span in orig_token_spans]
    parts = []
    for i in range(len(new_preterms)):
        parts.append(line[ex_span_begins[i]:ex_span_ends[i]])
        parts.append(new_preterms[i])
    parts.append(line[ex_span_begins[i+1]:ex_span_ends[i+1]])
    return ''.join(parts)

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1  
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx
    return output_actions

def find_nts_in_tree(tree):
    tree = tree.strip()
    return re.findall(r'(?=\(([^\s]+)\s\()', tree)

def pad_tok_frames(tok_frames,padded_len=350):
    if padded_len >= tok_frames.shape[-1]:
        padded = np.pad(tok_frames,((0,0),(0,padded_len-tok_frames.shape[-1])))
    else:
        print(f'pad size too small: {tok_frames.shape}')
        padded = tok_frames
    return padded


def get_sent_info(arg):
    if len(arg) == 3:
        tree, setting, idnum = arg
    else:
        tree, setting = arg
        idnum = ''
    tree = tree.strip()
    lowercase, replace_num, vocab, sp, action_dict, io_action_dict,speech_feats = setting
    if sp is not None:
        # use sentencepiece
        tree = transform_to_subword_tree(tree, sp)
    subword_tokenized = sp is not None
    tags, tokens, tokens_lower = get_tags_tokens_lowercase(tree)
    top_down_actions = get_actions(tree)
    in_order_actions = utils.get_in_order_actions(tree, subword_tokenized)
    top_down_max_stack_size = utils.get_top_down_max_stack_size(top_down_actions)
    assert len([a for a in in_order_actions if a == 'SHIFT']) == len(tokens)
    in_order_max_stack_size = utils.get_in_order_max_stack_size(
        in_order_actions, tokens, subword_tokenized)
    tags, tokens, tokens_lower = get_tags_tokens_lowercase(tree)
    orig_tokens = tokens[:]

    if sp is None:
        # these are not applied with sentencepiece
        if lowercase:
            tokens = tokens_lower
        if replace_num:
            tokens = [utils.clean_number(w) for w in tokens]

        token_ids = [vocab.get_id(t) for t in tokens]
        conved_tokens = [vocab.i2w[w_i] for w_i in token_ids]
    else:
        token_ids = sp.piece_to_id(tokens)
        conved_tokens = tokens

    top_down_action_ids = action_dict.to_id(top_down_actions)
    in_order_action_ids = io_action_dict.to_id(in_order_actions)

    
    return_dict =  {
        'orig_tokens': orig_tokens,
        'tokens': conved_tokens,
        'token_ids': token_ids,
        'tags': tags,
        'actions': top_down_actions,
        'action_ids': top_down_action_ids,
        'max_stack_size': top_down_max_stack_size,
        'in_order_actions': in_order_actions,
        'in_order_action_ids': in_order_action_ids,
        'in_order_max_stack_size': in_order_max_stack_size,
        'tree_str': tree    }

    if idnum:
        return_dict['idnum'] = idnum
        #return_dict['pitch'] = [pad_tok_frames(feats).tolist() for feats in speech_feats[idnum]['pitch']]
        #return_dict['fbank'] = [pad_tok_frames(feats).tolist() for feats in speech_feats[idnum]['fbank']]
        return_dict['pitch'] = [feats.tolist() for feats in speech_feats[idnum]['pitch']]
        return_dict['fbank'] = [feats.tolist() for feats in speech_feats[idnum]['fbank']]
        return_dict['pause'] = speech_feats[idnum]['pause']
        return_dict['dur'] = [feats.tolist() for feats in speech_feats[idnum]['dur']]
    
    return return_dict

def make_vocab(textfile, seqlength, minseqlength, lowercase, replace_num,
               vocabsize, vocabminfreq, unkmethod, jobs, apply_length_filter=True):
    w2c = defaultdict(int)
    with open(textfile, 'r') as f:
        trees = [tree.strip() for tree in f]
    with Pool(jobs) as pool:
        for tags, sent, sent_lower in pool.map(get_tags_tokens_lowercase, trees):
            assert(len(tags) == len(sent))
            if lowercase:
                sent = sent_lower
            if replace_num:
                sent = [utils.clean_number(w) for w in sent]
            if (len(sent) > seqlength and apply_length_filter) or len(sent) < minseqlength:
                continue

            for word in sent:
                w2c[word] += 1
    if unkmethod == 'berkeleyrule' or unkmethod == 'berkeleyrule2':
        conv_method = utils.berkeley_unk_conv if unkmethod == 'berkeleyrule' else utils.berkeley_unk_conv2
        berkeley_unks = set([conv_method(w) for w, c in w2c.items()])
        specials = list(berkeley_unks)
    else:
        specials = [unk]
    if vocabminfreq:
        w2c = dict([(w, c) for w, c in w2c.items() if c >= vocabminfreq])
    elif vocabsize > 0 and len(w2c) > vocabsize:
        sorted_wc = sorted(list(w2c.items()), key=lambda x:x[1], reverse=True)
        w2c = dict(sorted_wc[:vocabsize])
    return Vocabulary(list(w2c.items()), pad, unkmethod, unk, specials)

def learn_sentencepiece(textfile, output_prefix, args, apply_length_filter=True):
    import sentencepiece as spm
    with open(textfile, 'r') as f:
        trees = [tree.strip() for tree in f]
    user_defined_symbols = args.subword_user_defined_symbols or []
    if args.keep_ptb_bracket:
        user_defined_symbols += ['-LRB-', '-RRB-']
    with NamedTemporaryFile('wt') as tmp:
        with Pool(args.jobs) as pool:
            for tags, sent, sent_lower in pool.map(get_tags_tokens_lowercase, trees):
                assert(len(tags) == len(sent))
                if (len(sent) > args.seqlength and apply_length_filter) or len(sent) < args.minseqlength:
                    continue
                tmp.write(' '.join(sent)+'\n')
        tmp.flush()
        spm.SentencePieceTrainer.train(
            input=tmp.name,
            model_prefix=output_prefix,
            vocab_size=args.vocabsize,
            model_type=args.subword_type,
            treat_whitespace_as_suffix=True,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            user_defined_symbols=user_defined_symbols,
        )
    return spm.SentencePieceProcessor(model_file='{}.model'.format(output_prefix))

def get_data(args):
    def get_nonterminals(textfiles, jobs=-1):
        nts = set()
        for fn in textfiles:
            with open(fn, 'r') as f:
                lines = [line for line in f]
            with Pool(jobs) as pool:
                local_nts = pool.map(find_nts_in_tree, lines)
                nts.update(list(itertools.chain.from_iterable(local_nts)))
        nts = sorted(list(nts))
        print('Found nonterminals: {}'.format(nts))
        return nts

    def convert(textfile, lowercase, replace_num, seqlength, minseqlength,
                outfile, vocab, sp, action_dict, io_action_dict, apply_length_filter=True, jobs=-1,speech_feats={},id_file=''):
        dropped = 0
        num_sents = 0
        conv_setting = (lowercase, replace_num, vocab, sp, action_dict, io_action_dict,speech_feats)
        def process_block(tree_with_settings, f):
            _dropped = 0
            with Pool(jobs) as pool:
                for sent_info in pool.map(get_sent_info, tree_with_settings):
                    tokens = sent_info['tokens']
                    if (apply_length_filter and
                        (len(tokens) > seqlength or len(tokens) < minseqlength)):
                        _dropped += 1
                        continue
                    sent_info['key'] = 'sentence'
                    f.write(json.dumps(sent_info)+'\n')
            return _dropped
        if id_file: ids = [l.strip() for l in open(id_file,'r').readlines()]
        with open(outfile, 'wt') as f, open(textfile, 'r') as in_f:
            block_size = 100000
            tree_with_settings = []
            for idx,tree in enumerate(in_f):
                if tree.startswith('(TOP'):
                    tree = tree.replace('(TOP','').strip()[:-1]
                if id_file:
                    idnum = ids[idx].strip()
                    tree_with_settings.append((tree, conv_setting,idnum))
                else:
                    tree_with_settings.append((tree, conv_setting))
                if len(tree_with_settings) >= block_size:
                    dropped += process_block(tree_with_settings, f)
                    num_sents += len(tree_with_settings)
                    tree_with_settings = []
            if len(tree_with_settings) > 0:
                process_block(tree_with_settings, f)
                num_sents += len(tree_with_settings)

            others = {"vocab": vocab.to_json_dict() if vocab is not None else None,
                      "nonterminals": nonterminals,
                      "pad_token": pad,
                      "unk_token": unk,
                      "args": args.__dict__}
            for k, v in others.items():
                f.write(json.dumps({'key': k, 'value': v})+'\n')

        print("Saved {} sentences (dropped {} due to length/unk filter)".format(
            num_sents, dropped))

    print("First pass through data to get nonterminals...")
    nonterminals = get_nonterminals([args.trainfile, args.valfile, args.testfile], args.jobs)
    action_dict = TopDownActionDict(nonterminals)
    io_action_dict = InOrderActionDict(nonterminals)

    if args.unkmethod == 'subword':
        if args.vocabfile != '':
            print('Loading pre-trained sentencepiece model from {}'.format(args.vocabfile))
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=args.vocabfile)
            sp_model_path = '{}-spm.model'.format(args.outputfile)
            print('Copy sentencepiece model to {}'.format(sp_model_path))
            shutil.copyfile(args.vocabfile, sp_model_path)
        else:
            print('unkmethod subword is selected. Running sentencepiece on the training data...')
            sp = learn_sentencepiece(args.trainfile, args.outputfile+'-spm', args)
        vocab = None
    else:
        if args.vocabfile != '':
            print('Loading pre-specified source vocab from ' + args.vocabfile)
            vocab = Vocabulary.load(args.vocabfile)
        else:
            print("Second pass through data to get vocab...")
            vocab = make_vocab(args.trainfile, args.seqlength, args.minseqlength,
                               args.lowercase, args.replace_num, args.vocabsize, args.vocabminfreq,
                               args.unkmethod, args.jobs)
        vocab.dump(args.outputfile + ".vocab")
        print("Vocab size: {}".format(len(vocab.i2w)))
        sp = None

    def load_sp_feats(directory,split,turn=False,back_context=0,for_context=0,tok_frame_len=100,context_strat="all"):
        if turn:
            prefix = 'turn_'
        else:
            prefix = ''
        partition = pickle.load(open(os.path.join(directory,f'{prefix}{split}_partition.pickle'),'rb'))
        pitch = pickle.load(open(os.path.join(directory,f'{prefix}{split}_pitch.pickle'),'rb'))
        fbank = pickle.load(open(os.path.join(directory,f'{prefix}{split}_fbank.pickle'),'rb'))
        pause = pickle.load(open(os.path.join(directory,f'{prefix}{split}_pause.pickle'),'rb'))
        duration = pickle.load(open(os.path.join(directory,f'{prefix}{split}_duration.pickle'),'rb'))
        out_dict = {}
        for idnum in pause: # iterate over all sents or turns
            out_dict[idnum] = {}
                
            part = partition[idnum]
            pit = pitch[idnum]
            fb = fbank[idnum]
            pau = pause[idnum]
            dur = duration[idnum]
            pit_tok = []
            fb_tok = []
            pau_tok = []
            dur_tok = []
            init_idx = 0
            final_idx = len(part)-1
            for idx,prt in enumerate(part):
                if idx > 0 and prt[0]>part[idx-1][-1]:
                    print(part)
                    import pdb;pdb.set_trace()
                pau_tok.append(pau['pause_aft'][idx])
                dur_tok.append(dur[:,idx])
                if context_strat == 'tok_only':
                    tmp_pit = []
                    tmp_fb = []
                    tok_start_idx = max(0,idx-back_context)
                    tok_end_idx = min(len(part)-1,idx+for_context)
                    for j in range(tok_start_idx,tok_end_idx+1):
                        start = part[j][0]
                        end = part[j][-1]
                        tmp_pit.append(pit[:,start:end])
                        tmp_fb.append(fb[:,start:end])
                    tmp_pit = np.concatenate(tmp_pit,axis=1)
                    tmp_fb = np.concatenate(tmp_fb,axis=1)

                    pit_tok.append(tmp_pit)
                    fb_tok.append(tmp_fb)

                elif context_strat == 'all': # include all frames for prev/following tokens
                    start_idx = max(idx-back_context,init_idx)
                    end_idx = min(idx+for_context,final_idx)
                    start = part[start_idx][0]
                    end = part[end_idx][-1]
                    tmp_pit = pit[:,start:end]
                    tmp_fb = fb[:,start:end]
                    while tmp_pit.shape[-1] > args.tok_frame_len:
                        tmp_pit = tmp_pit[:,::2]
                        tmp_fb = tmp_pit[:,::2]

                    pit_tok.append(tmp_pit)
                    fb_tok.append(tmp_fb)
                elif context_strat == 'pool': # Use mean pooling on prev/following tokens
                    start = part[idx][0]
                    end = part[idx][-1]
                    tmp_pit = pit[:,start:end]
                    tmp_fb = fb[:,start:end]
                    # Get any trailing tokens, mean pool them, and then append:
                    bk_pit = []
                    bk_fb = []
                    fr_pit = []
                    fr_fb = []
                    if back_context > 0:
                        for i in range(back_context):
                            prev_idx = idx-i-1
                            if prev_idx >= 0:
                                prev_start = part[prev_idx][0]
                                prev_end = part[prev_idx][-1]

                                prev_pit = pit[:,prev_start:prev_end]

                                if not prev_start==prev_end:
                                    prev_pit = block_reduce(prev_pit,block_size=(1,prev_pit.shape[-1]),func=np.mean)
                                    bk_pit = [prev_pit] + bk_pit
                                
                                    prev_fb = fb[:,prev_start:prev_end]
                                    prev_fb = block_reduce(prev_fb,block_size=(1,prev_fb.shape[-1]),func=np.mean)
                                    bk_fb = [prev_fb] + bk_fb
                                
                    if for_context > 0:
                        for i in range(for_context):
                            next_idx = idx+i+1
                            if next_idx < len(part):
                                next_start = part[next_idx][0]
                                next_end = part[next_idx][-1]

                                next_pit = pit[:,next_start:next_end]

                                next_pit = block_reduce(next_pit,block_size=(1,next_pit.shape[-1]),func=np.mean)
                                fr_pit.append(next_pit)

                                next_fb = fb[:,next_start:next_end]
                                next_fb = block_reduce(next_fb,block_size(1,next_fb.shape[-1]), func=np.mean)
                                fr_fb.append(next_pit)
                    tmp_pit = np.concatenate(bk_pit+[tmp_pit]+fr_pit,axis=1)
                    tmp_fb = np.concatenate(bk_fb+[tmp_fb]+fr_fb,axis=1)
                    pit_tok.append(tmp_pit)
                    fb_tok.append(tmp_fb)
                elif context_strat == 'leading': #use the first few leading frames for following (prev?) tokens
                    start = part[idx][0]
                    end = part[idx][-1]
                    tmp_pit = pit[:,start:end]
                    tmp_fb = fb[:,start:end]

                    N = 5 # How many leading frames to take
                    assert back_context == 0 #FOR NOW, only use this setting to test out the 'one following token' condition
                    assert for_context == 1
                    next_idx = idx+1
                    if next_idx < len(part):
                        next_start = part[next_idx][0]
                        next_end = min(part[next_idx][-1],next_start+N) # only take first N frames
                        leading_pit = pit[:,next_start:next_end]
                        leading_fb = fb[:,next_start:next_end]
                        tmp_pit = np.concatenate([tmp_pit,leading_pit],axis=1)
                        tmp_fb = np.concatenate([tmp_fb,leading_fb],axis=1)
                    pit_tok.append(tmp_pit)
                    fb_tok.append(tmp_fb)
                elif context_strat == 'noise': # non-current token frames are replaced with noise of same shape.
                    assert back_context == 0
                    assert for_context == 1 # FOR NOW, only implement for the one following token condition.
                    next_idx = idx + 1

                    if next_idx < len(part):
                        next_start = part[next_idx][0]
                        next_end = part[next_idx][-1]
                        next_pit = pit[:,next_start:next_end]
                        next_fb = fb[:,next_start:next_end]
                        noise_pit = np.random.normal(size=next_pit.shape)
                        noise_fb = np.random.normal(size=next_fb.shape)
                        tmp_pit = np.concatenate([tmp_pit,noise_pit],axis=1)
                        tmp_fb = np.concatenate([tmp_fb,noise_fb],axis=1)

                    pit_tok.append(tmp_pit)
                    fb_tok.append(tmp_fb)
            
            #if pit[:,start:end].shape[-1] >100:
            #    import pdb;pdb.set_trace()        
            out_dict[idnum]['pause'] = pau_tok
            out_dict[idnum]['pitch'] = pit_tok
            out_dict[idnum]['fbank'] = fb_tok
            out_dict[idnum]['dur'] = dur_tok
        return out_dict

        
    if args.speech_dir:
        train_sp_feats = load_sp_feats(args.speech_dir,'train',args.turn,args.back_context,args.for_context,args.tok_frame_len,args.context_strat) if args.speech_dir else {}
        dev_sp_feats = load_sp_feats(args.speech_dir,'dev',args.turn,args.back_context,args.for_context,args.tok_frame_len,args.context_strat) if args.speech_dir else {}
        test_sp_feats = load_sp_feats(args.speech_dir,'test',args.turn,args.back_context,args.for_context,args.tok_frame_len,args.context_strat) if args.speech_dir else {}

        if args.turn:
            prefix = 'turn_'
            infix = '_medium'
        else:
            prefix = infix = ''
        train_id_file = os.path.join(args.speech_dir,f'{prefix}train_sent_ids{infix}.txt')
        dev_id_file = os.path.join(args.speech_dir,f'{prefix}dev_sent_ids{infix}.txt')
        test_id_file = os.path.join(args.speech_dir,f'{prefix}test_sent_ids{infix}.txt')
    else:
        train_sp_feats = {}
        dev_sp_feats = {}
        test_sp_feats = {}

        train_id_file = ''
        dev_id_file = ''
        test_id_file = ''

    print(train_id_file)

    """
    convert(args.testfile, args.lowercase, args.replace_num,
            0, args.minseqlength, args.outputfile + "-test.json",
            vocab, sp, action_dict, io_action_dict, 0, args.jobs,test_sp_feats,test_id_file)
    """
    convert(args.valfile, args.lowercase, args.replace_num,
            args.seqlength, args.minseqlength, args.outputfile + "-dev.json",
            vocab, sp, action_dict, io_action_dict, 0, args.jobs,dev_sp_feats,dev_id_file)
    """
    convert(args.trainfile, args.lowercase, args.replace_num,
            args.seqlength,  args.minseqlength, args.outputfile + "-train.json",
            vocab, sp, action_dict, io_action_dict, 1, args.jobs,train_sp_feats,train_id_file)
    """
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocabsize',
                        help="Size of vocabulary or subword vocabulary. "
                        "When unkmethod is not subword, vocab is constructed "
                        "by taking the top X most frequent words and "
                        "rest are replaced with special UNK tokens. "
                        "If unkmethod=subword, this defines the subword vocabulary size. ",
                        type=int, default=100000)
    parser.add_argument('--vocabminfreq',
                        help="Minimum frequency for vocab. "
                        "When this value > 0, this value is used instead of vocabsize.",
                        type=int, default=0)
    parser.add_argument('--unkmethod', help="How to replace an unknown token.",
                        choices=['unk', 'berkeleyrule', 'berkeleyrule2', 'subword'],
                        default='unk')
    parser.add_argument('--subword_type', help="Segmentation algorithm in sentencepiece. Note that --treat_whitespace_as_suffix for sentence_piece is always True.",
                        choices=['bpe', 'unigram'], default='bpe')
    parser.add_argument('--keep_ptb_bracket', action='store_true',
                        help='Recommended for English PTB-like dataset. Do not segment -LRB- and -RRB- into subwords. (other brackets, such as -LCB-, are kept (may be segmented), considering their low frequency.)')
    parser.add_argument('--subword_user_defined_symbols', nargs='*',
                        help='--user_defined_symbols for sentencepiece. These tokens are not segmented into subwords.')
    parser.add_argument('--lowercase', help="Lower case", action='store_true')
    parser.add_argument('--replace_num', help="Replace numbers with N", action='store_true')
    parser.add_argument('--trainfile', help="Path to training data.", required=True)
    parser.add_argument('--valfile', help="Path to validation data.", required=True)
    parser.add_argument('--testfile', help="Path to test validation data.", required=True)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                            "than this are dropped.", type=int, default=300)
    parser.add_argument('--minseqlength', help="Minimum sequence length. Sequences shorter "
                                               "than this are dropped.", type=int, default=0)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str,
                        required=True)
    parser.add_argument('--vocabfile', help="If working with a preset vocab, "
                                            "then including this will ignore srcvocabsize and use the"
                                            "vocab provided here. "
                                            "If unkmethod=subword, this argument specifies learned sentencepiece model.",
                                            type = str, default='')
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--id_file',type=str,default='')
    parser.add_argument('--speech_dir',type=str,default='')
    parser.add_argument('--turn',action='store_true')
    parser.add_argument('--back_context',type=int,default=0)
    parser.add_argument('--for_context',type=int,default=0)
    parser.add_argument('--tok_frame_len',type=int,default=100)
    parser.add_argument('--context_strat',type=str,default='all',help='Strategy for processing extra context: all = keep all frames, pool = mean pool frames, leading = keep leading n frames')
    args = parser.parse_args(arguments)
    if args.jobs == -1:
        args.jobs = len(os.sched_getaffinity(0))
    # np.random.seed(3435)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
