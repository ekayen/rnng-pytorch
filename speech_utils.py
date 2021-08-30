import torch
import numpy as np
from skimage.measure import block_reduce
###################################
## Speech handling fns
## Elizabeth Nielsen
## enielsen@ed.ac.uk
###################################


def pad_tok_frames(tok_frames,padded_len = 100):
  if padded_len >= tok_frames.shape[-1]:
    padded = np.pad(tok_frames,((0,0),(0,padded_len-tok_frames.shape[-1])))
  else:
    subsampled = tok_frames
    while subsampled.shape[-1] > padded_len:
      subsampled = subsampled[:,::2]
    padded = np.pad(subsampled,((0,0),(0,padded_len-subsampled.shape[-1])))
  return padded

def pad_batch_frames(feats,back_context,for_context,device):
  batched = []
  num_channel_in = 1
  max_toks = max([len(sent_feats) for sent_feats in feats])
  zero_tok = np.zeros(feats[0][0].shape)
  for i,sent in enumerate(feats):
    while len(feats[i]) < max_toks:
      feats[i].append(zero_tok)
  batched = []
  for i in range(max_toks):
    all_tok = np.stack([sent[i] for sent in feats],axis=0)
    bat_shape = all_tok.shape
    all_tok = torch.tensor(all_tok,requires_grad=True).type(torch.float).view(bat_shape[0],num_channel_in,bat_shape[1],bat_shape[2])
    if device:
      all_tok = all_tok.to(device)
    batched.append(all_tok)
  return batched


def frame_batch_w_context(batch_feats,back_context,for_context,context_strat,tok_frame_len):
  feats_w_context = []
  for sent in batch_feats:
    sent_w_context = []
    for curr_idx,tok in enumerate(sent):
      if context_strat == 'all':
        start_idx = max(0,curr_idx-back_context)
        end_idx = min(len(sent)-1,curr_idx+for_context)
        w_context = sent[start_idx:end_idx+1]
        w_context = np.concatenate(w_context,axis=1)
        w_context = pad_tok_frames(w_context,tok_frame_len)
        sent_w_context.append(w_context)

      elif context_strat == 'pool':
        start_idx = max(0,curr_idx-back_context)
        end_idx = min(len(sent)-1,curr_idx+for_context)
        if back_context > 0:
          prev_toks = sent[start_idx:curr_idx]
          prev_toks = [block_reduce(tok,block_size=(1,tok.shape[-1]),func=np.mean) for tok in prev_toks]
        else:
          prev_toks = []
        if for_context > 0:
          next_toks = sent[curr_idx+1:end_idx+1]
          next_toks = [block_reduce(tok,block_size=(1,tok.shape[-1]),func=np.mean) for tok in next_toks]
        else:
          next_toks = []
        w_context = np.concatenate(prev_toks+[tok]+next_toks,axis=1)
        sent_w_context.append(w_context)
      elif context_strat == 'leading':
        N = 5 # NUMBER OF LEADING FRAMES
        assert for_context == 1
        assert back_context == 0
        if curr_idx < len(sent)-1:
          leading_frames = sent[curr_idx+1][:,:N]
          w_context = np.concatenate((tok,leading_frames),axis=1)
        else:
          w_context = tok
        sent_w_context.append(w_context)

    feats_w_context.append(sent_w_context)

  return feats_w_context
  
def bat_frame_feats(speech_feats,pitch=True,fbank=True,tok_frame_len=100,device=None,back_context=0,for_context=0,context_strat='all'):
  if pitch:
    pitch_feats = [feats['pitch'] for feats in speech_feats]
    feats_w_context = []
    if back_context == 0 and for_context == 0:
      pitch_batched = [[pad_tok_frames(tok,tok_frame_len) for tok in feats] for feats in pitch_feats]
    else:
      pitch_batched = frame_batch_w_context(pitch_feats,back_context,for_context,context_strat,tok_frame_len)
    pitch_batched = pad_batch_frames(pitch_batched,back_context,for_context,device)
  if fbank:
    fbank_feats = [feats['fbank'] for feats in speech_feats]
    feats_w_context = []
    if back_context == 0 and for_context == 0:
      fbank_batched  = [[pad_tok_frames(tok,tok_frame_len) for tok in feats] for feats in fbank_feats]
    else:
      fbank_batched = frame_batch_w_context(fbank_feats,back_context,for_context,context_strat,tok_frame_len)
    fbank_batched = pad_batch_frames(fbank_batched,back_context,for_context,device)
  if fbank and pitch:
    frame_batched = [torch.cat([pit,fb],dim=2) for pit,fb in zip(pitch_batched,fbank_batched)]
  elif fbank:
    frame_batched = fbank_batched
  elif pitch:
    frame_batched = pitch_batched
    
  return frame_batched

    
def bat_pause_feats(speech_feats,back_context,for_context,device):
  max_num_toks = 0
  pause_batched = []
  back = [[] for _ in range(back_context)]
  forward = [[] for _ in range(for_context)]
  for feats in speech_feats:
    max_num_toks = max(max_num_toks,len(feats['pause']))
    pause_batched.append(np.array(feats['pause']))
    if back_context > 0:
      for i in range(1,back_context+1):
        padding = i if i <= max_num_toks else max_num_toks
        back_shifted = [0]*padding + feats['pause'][:-i]
        back[-i].append(np.array(back_shifted))
    if for_context > 0:
      for i in range(1,for_context+1):
        padding = i if i <= max_num_toks else max_num_toks
        for_shifted = feats['pause'][i:] + [0]*padding 
        forward[i-1].append(np.array(for_shifted))
  pause_batched = torch.tensor(np.stack([np.pad(sent,((0,max_num_toks-sent.shape[0])),constant_values=0) for sent in pause_batched])).type(torch.int)
  if back_context > 0:
    back = [torch.tensor(np.stack([np.pad(sent,((0,max_num_toks-sent.shape[0])),constant_values=0) for sent in shift])).type(torch.int).to(device) for shift in back]
  if for_context > 0:
    forward = [torch.tensor(np.stack([np.pad(sent,((0,max_num_toks-sent.shape[0])),constant_values=0) for sent in shift])).type(torch.int).to(device) for shift in forward]
  if back_context == 0 and for_context == 0:
    return pause_batched.to(device)
  else:
    return pause_batched.to(device),back,forward
 
def bat_dur_feats(speech_feats,back_context,for_context):
  max_num_toks = 0
  dur_batched = []
  for feats in speech_feats:
    max_num_toks = max(max_num_toks,len(feats['dur']))
    dur_batched.append(np.stack(feats['dur']))
  dur_batched = [np.pad(sent,((0,max_num_toks-sent.shape[0]),(0,0)),constant_values=0) for sent in dur_batched]
  dur_batched = np.stack(dur_batched)
  back = []
  forward = []
  if back_context > 0 or for_context > 0:
    if back_context > 0:
      back = []
      for i in range(1,back_context+1):
        padding = i if i <= max_num_toks else max_num_toks
        shifted = dur_batched[:,:-i,:]
        shifted = np.pad(shifted,((0,0),(padding,0),(0,0)),constant_values=0)
        back = [shifted] + back
    if for_context > 0:
      forward = []
      for i in range(1,for_context+1):
        padding = i if i <= max_num_toks else max_num_toks
        shifted = dur_batched[:,i:,:]
        shifted = np.pad(shifted,((0,0),(0,padding),(0,0)),constant_values=0)
        forward.append(shifted)
    dur_batched = np.concatenate(back+[dur_batched]+forward,axis=-1)
  dur_batched = torch.tensor(dur_batched)
  return dur_batched

def get_sp_feats(args,speech_feats,device,speech_feat_types=[],tok_frame_len=None):
  if not speech_feat_types:
    speech_feat_types = args.speech_feat_types
  if not tok_frame_len:
    tok_frame_len = args.tok_frame_len


  try:
    back_context = args.back_context
    for_context = args.for_context
    context_strat = args.context_strat
  except:
    back_context = 0
    for_context = 0
    context_strat = 0
    
  
  if speech_feat_types:
    if 'pause' in speech_feat_types:
      pause = bat_pause_feats(speech_feats,back_context,for_context,device)
    else:
      pause = None
    if 'dur' in speech_feat_types:
      dur = bat_dur_feats(speech_feats,back_context,for_context).to(device)
    else:
      dur = None
    if 'pitch' and 'fbank' in speech_feat_types:
      frames = bat_frame_feats(speech_feats,pitch=True,fbank=True,tok_frame_len=tok_frame_len,device=device,back_context=back_context,for_context=for_context,context_strat=context_strat)
    elif 'pitch' in speech_feat_types:
      frames = bat_frame_feats(speech_feats,pitch=True,fbank=False,tok_frame_len=tok_frame_len,device=device,back_context=back_context,for_context=for_context,context_strat=context_strat)
    elif 'fbank' in speech_feat_types:
      frames = bat_frame_feats(speech_feats,pitch=False,fbank=True,tok_frame_len=tok_frame_len,device=device,back_context=back_context,for_context=for_context,context_strat=context_strat)
    else:
      frames = None
  else:
      pause = dur = frames = None
  return pause,dur,frames  

