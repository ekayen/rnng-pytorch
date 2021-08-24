import torch
import numpy as np

## Speech handling fns:

def pad_tok_frames(tok_frames,padded_len = 100):
  if padded_len >= tok_frames.shape[-1]:
    padded = np.pad(tok_frames,((0,0),(0,padded_len-tok_frames.shape[-1])))
  else:
    subsampled = tok_frames
    while subsampled.shape[-1] > padded_len:
      subsampled = subsampled[:,::2]
    padded = np.pad(subsampled,((0,0),(0,padded_len-subsampled.shape[-1])))
  return padded

def pad_batch_frames(feats,device):
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

    
def bat_frame_feats(speech_feats,pitch=True,fbank=True,tok_frame_len=100,device=None):
  if pitch:
    pitch_feats = [feats['pitch'] for feats in speech_feats]
    pitch_batched = [[pad_tok_frames(tok,tok_frame_len) for tok in feats] for feats in pitch_feats]
    pitch_batched = pad_batch_frames(pitch_batched,device)

  if fbank:
    fbank_feats = [feats['fbank'] for feats in speech_feats]
    fbank_batched  = [[pad_tok_frames(tok,tok_frame_len) for tok in feats] for feats in fbank_feats]
    fbank_batched = pad_batch_frames(fbank_batched,device)

  if fbank and pitch:
    frame_batched = [torch.cat([pit,fb],dim=2) for pit,fb in zip(pitch_batched,fbank_batched)]
  elif fbank:
    frame_batched = fbank_batched
  elif pitch:
    frame_batched = pitch_batched
    
  return frame_batched

    
def bat_pause_feats(speech_feats):
  max_num_toks = 0
  pause_batched = []
  for feats in speech_feats:
    max_num_toks = max(max_num_toks,len(feats['pause']))
    pause_batched.append(np.array(feats['pause']))
  pause_batched = torch.tensor(np.stack([np.pad(sent,((0,max_num_toks-sent.shape[0])),constant_values=0) for sent in pause_batched])).type(torch.int)
  return pause_batched

def bat_dur_feats(speech_feats):
  max_num_toks = 0
  dur_batched = []
  for feats in speech_feats:
    max_num_toks = max(max_num_toks,len(feats['dur']))
    dur_batched.append(np.stack(feats['dur']))
  dur_batched = [np.pad(sent,((0,max_num_toks-sent.shape[0]),(0,0)),constant_values=0) for sent in dur_batched]
  dur_batched = torch.tensor(np.stack(dur_batched))
  return dur_batched

def get_sp_feats(args,speech_feats,device,speech_feat_types=[],tok_frame_len=None):
  if not speech_feat_types:
    speech_feat_types = args.speech_feat_types
  if not tok_frame_len:
    tok_frame_len = args.tok_frame_len

  back_context = args.back_context
  for_context = args.for_context
  context_strat = args.context_strat
  
  if speech_feat_types:
    if 'pause' in speech_feat_types:
      pause = bat_pause_feats(speech_feats).to(device)
    else:
      pause = None
    if 'dur' in speech_feat_types:
      dur = bat_dur_feats(speech_feats).to(device)
    else:
      dur = None
    if 'pitch' and 'fbank' in speech_feat_types:
      frames = bat_frame_feats(speech_feats,pitch=True,fbank=True,tok_frame_len=tok_frame_len,device=device)
    elif 'pitch' in speech_feat_types:
      frames = bat_frame_feats(speech_feats,pitch=True,fbank=False,tok_frame_len=tok_frame_len,device=device)
    elif 'fbank' in speech_feat_types:
      frames = bat_frame_feats(speech_feats,pitch=False,fbank=True,tok_frame_len=tok_frame_len,device=device)
    else:
      frames = None
  else:
      pause = dur = frames = None
  return pause,dur,frames  

