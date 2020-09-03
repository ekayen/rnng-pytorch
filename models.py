import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils import *
from TreeCRF import ConstituencyTreeCRF
from torch.distributions import Bernoulli
import itertools

class RNNLM(nn.Module):
  def __init__(self, vocab=10000,
               w_dim=650,
               h_dim=650,
               num_layers=2,
               dropout=0.5):
    super(RNNLM, self).__init__()
    self.h_dim = h_dim
    self.num_layers = num_layers    
    self.word_vecs = nn.Embedding(vocab, w_dim)
    self.dropout = nn.Dropout(dropout)
    self.rnn = nn.LSTM(w_dim, h_dim, num_layers = num_layers,
                       dropout = dropout, batch_first = True)      
    self.vocab_linear =  nn.Linear(h_dim, vocab)
    self.vocab_linear.weight = self.word_vecs.weight # weight sharing

  def forward(self, sent):
    word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))
    h, _ = self.rnn(word_vecs)
    log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2) # b x l x v
    ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
    return ll.sum(1)
  
  def generate(self, bos = 2, eos = 3, max_len = 150):
    x = []
    bos = torch.LongTensor(1,1).cuda().fill_(bos)
    emb = self.dropout(self.word_vecs(bos))
    prev_h = None
    for l in range(max_len):
      h, prev_h = self.rnn(emb, prev_h)
      prob = F.softmax(self.vocab_linear(self.dropout(h.squeeze(1))), 1)
      sample = torch.multinomial(prob, 1)
      emb = self.dropout(self.word_vecs(sample))
      x.append(sample.item())
      if x[-1] == eos:
        x.pop()
        break
    return x

class SeqLSTM(nn.Module):
  def __init__(self, i_dim = 200,  # word embedding dim
               h_dim = 0,  # internal hidden dim
               num_layers = 1,
               dropout = 0):
    super(SeqLSTM, self).__init__()    
    self.i_dim = i_dim
    self.h_dim = h_dim
    self.num_layers = num_layers
    self.linears = nn.ModuleList([nn.Linear(h_dim + i_dim, h_dim*4) if l == 0 else
                                  nn.Linear(h_dim*2, h_dim*4) for l in range(num_layers)])
    self.dropout = dropout
    self.dropout_layer = nn.Dropout(dropout)

  def forward(self, x, prev_h = None):
    if prev_h is None:
      prev_h = [(x.new(x.size(0), self.h_dim).fill_(0),
                 x.new(x.size(0), self.h_dim).fill_(0)) for _ in range(self.num_layers)]  # (n_layers, 2, batch_size, h_dim)
    curr_h = []
    for l in range(self.num_layers):
      input = x if l == 0 else curr_h[l-1][0]
      if l > 0 and self.dropout > 0:
        input = self.dropout_layer(input)
      concat = torch.cat([input, prev_h[l][0]], 1)  # (batch_size, h_dim*2) or (batch_size, h_dim+i_dim)
      all_sum = self.linears[l](concat)  # (batch_size, h_dim*4)
      i, f, o, g = all_sum.split(self.h_dim, 1)  # (batch_size, h_dim) for each
      c = torch.sigmoid(f)*prev_h[l][1] + torch.sigmoid(i)*torch.tanh(g)
      h = torch.sigmoid(o)*torch.tanh(c)
      curr_h.append((h, c))
    return curr_h  # (n_layers, 2, batch_size, h_dim)

class TreeLSTM(nn.Module):
  def __init__(self, dim = 200):
    super(TreeLSTM, self).__init__()
    self.dim = dim
    self.linear = nn.Linear(dim*2, dim*5)

  def forward(self, x1, x2, e=None):
    if not isinstance(x1, tuple):
      x1 = (x1, None)    
    h1, c1 = x1 
    if x2 is None: 
      x2 = (torch.zeros_like(h1), torch.zeros_like(h1))
    elif not isinstance(x2, tuple):
      x2 = (x2, None)    
    h2, c2 = x2
    if c1 is None:
      c1 = torch.zeros_like(h1)
    if c2 is None:
      c2 = torch.zeros_like(h2)
    concat = torch.cat([h1, h2], 1)
    all_sum = self.linear(concat)
    i, f1, f2, o, g = all_sum.split(self.dim, 1)

    c = torch.sigmoid(f1)*c1 + torch.sigmoid(f2)*c2 + torch.sigmoid(i)*torch.tanh(g)
    h = torch.sigmoid(o)*torch.tanh(c)
    return (h, c)

class RNNG(nn.Module):
  def __init__(self, vocab = 100,
               w_dim = 20, 
               h_dim = 20,
               num_layers = 1,
               dropout = 0,
               q_dim = 20,
               max_len = 250):
    super(RNNG, self).__init__()
    self.S = 0 #action idx for shift/generate
    self.R = 1 #action idx for reduce
    self.emb = nn.Embedding(vocab, w_dim)
    self.dropout = nn.Dropout(dropout)    
    self.stack_rnn = SeqLSTM(w_dim, h_dim, num_layers = num_layers, dropout = dropout)
    self.tree_rnn = TreeLSTM(w_dim)
    self.vocab_mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, vocab))
    self.num_layers = num_layers
    self.q_binary = nn.Sequential(nn.Linear(q_dim*2, q_dim*2), nn.ReLU(), nn.LayerNorm(q_dim*2),
                                  nn.Dropout(dropout), nn.Linear(q_dim*2, 1))
    self.action_mlp_p = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, 1))
    self.w_dim = w_dim
    self.h_dim = h_dim
    self.q_dim = q_dim    
    self.q_leaf_rnn = nn.LSTM(w_dim, q_dim, bidirectional = True, batch_first = True)
    self.q_crf = ConstituencyTreeCRF()
    self.pad1 = 0 # idx for <s> token from ptb.dict
    self.pad2 = 2 # idx for </s> token from ptb.dict 
    self.q_pos_emb = nn.Embedding(max_len, w_dim) # position embeddings
    self.vocab_mlp[-1].weight = self.emb.weight #share embeddings

  def get_span_scores(self, x):
    #produces the span scores s_ij
    bos = x.new(x.size(0), 1).fill_(self.pad1)
    eos  = x.new(x.size(0), 1).fill_(self.pad2)
    x = torch.cat([bos, x, eos], 1)  # (batch_size, n_words+2)
    x_vec = self.dropout(self.emb(x))
    pos = torch.arange(0, x.size(1)).unsqueeze(0).expand_as(x).long().cuda()  # (batch_size, n_words)
    x_vec = x_vec + self.dropout(self.q_pos_emb(pos))
    q_h, _ = self.q_leaf_rnn(x_vec)
    fwd = q_h[:, 1:, :self.q_dim]  # (batch_size, n_words+1, q_dim)
    bwd = q_h[:, :-1, self.q_dim:]  # (batch_size, n_words+1, q_dim)
    fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
    bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
    concat = torch.cat([fwd_diff, bwd_diff], 3)
    scores = self.q_binary(concat).squeeze(3)
    return scores  # (batch_size, n_words, n_words)

  def get_action_masks(self, actions, length):
    #this masks out actions so that we don't incur a loss if some actions are deterministic
    #in practice this doesn't really seem to matter
    mask = actions.new(actions.size(0), actions.size(1)).fill_(1)
    for b in range(actions.size(0)):      
      num_shift = 0
      stack_len = 0
      for l in range(actions.size(1)):
        if stack_len < 2:
          mask[b][l].fill_(0)
        if actions[b][l].item() == self.S:
          num_shift += 1
          stack_len += 1
        else:
          stack_len -= 1
    return mask

  def forward(self, x, samples = 1, is_temp = 1., has_eos=True):
    #For has eos, if </s> exists, then inference network ignores it. 
    #Note that </s> is predicted for training since we want the model to know when to stop.
    #However it is ignored for PPL evaluation on the version of the PTB dataset from
    #the original RNNG paper (Dyer et al. 2016)
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:]
    batch, length = x.size(0), x.size(1)
    if has_eos: 
      parse_length = length - 1
      parse_x = x[:, :-1]
    else:
      parse_length = length
      parse_x = x
    word_vecs =  self.dropout(self.emb(x))
    scores = self.get_span_scores(parse_x)
    self.scores = scores
    scores = scores / is_temp
    self.q_crf._forward(scores)
    self.q_crf._entropy(scores)
    entropy = self.q_crf.entropy[0][parse_length-1]
    crf_input = scores.unsqueeze(1).expand(batch, samples, parse_length, parse_length)
    crf_input = crf_input.contiguous().view(batch*samples, parse_length, parse_length)
    for i in range(len(self.q_crf.alpha)):
      for j in range(len(self.q_crf.alpha)):
        self.q_crf.alpha[i][j] = self.q_crf.alpha[i][j].unsqueeze(1).expand(
          batch, samples).contiguous().view(batch*samples)        
    _, log_probs_action_q, tree_brackets, spans = self.q_crf._sample(crf_input, self.q_crf.alpha)
    actions = []
    for b in range(crf_input.size(0)):    
      action = get_actions(tree_brackets[b])
      if has_eos:
        actions.append(action + [self.S, self.R]) #we train the model to generate <s> and then do a final reduce
      else:
        actions.append(action)
    actions = torch.Tensor(actions).float().cuda()
    action_masks = self.get_action_masks(actions, length) 
    num_action = 2*length - 1
    batch_expand = batch*samples
    contexts = []
    log_probs_action_p = [] #conditional prior
    init_emb = init_emb.unsqueeze(1).expand(batch, samples, self.w_dim)
    init_emb = init_emb.contiguous().view(batch_expand, self.w_dim)
    init_stack = self.stack_rnn(init_emb, None)
    x_expand = x.unsqueeze(1).expand(batch, samples, length)
    x_expand = x_expand.contiguous().view(batch_expand, length)
    word_vecs = self.dropout(self.emb(x_expand))
    word_vecs = word_vecs.unsqueeze(2)
    word_vecs_zeros = torch.zeros_like(word_vecs)
    stack = [init_stack]
    stack_child = [[] for _ in range(batch_expand)]
    stack2 = [[] for _ in range(batch_expand)]
    for b in range(batch_expand):
      stack2[b].append([[init_stack[l][0][b], init_stack[l][1][b]] for l in range(self.num_layers)])
    pointer = [0]*batch_expand
    for l in range(num_action):
      contexts.append(stack[-1][-1][0])
      stack_input = []
      child1_h = []
      child1_c = []
      child2_h = []
      child2_c = []
      stack_context = []
      for b in range(batch_expand):
        # batch all the shift/reduce operations separately
        if actions[b][l].item() == self.R:
          child1 = stack_child[b].pop()
          child2 = stack_child[b].pop()
          child1_h.append(child1[0])
          child1_c.append(child1[1])
          child2_h.append(child2[0])
          child2_c.append(child2[1])
          stack2[b].pop()
          stack2[b].pop()
      if len(child1_h) > 0:
        child1_h = torch.cat(child1_h, 0)
        child1_c = torch.cat(child1_c, 0)
        child2_h = torch.cat(child2_h, 0)
        child2_c = torch.cat(child2_c, 0)
        new_child = self.tree_rnn((child1_h, child1_c), (child2_h, child2_c))

      child_idx = 0
      stack_h = [[[], []] for _ in range(self.num_layers)]
      for b in range(batch_expand):
        assert(len(stack2[b]) - 1 == len(stack_child[b]))
        for k in range(self.num_layers):
          stack_h[k][0].append(stack2[b][-1][k][0])
          stack_h[k][1].append(stack2[b][-1][k][1])
        if actions[b][l].item() == self.S:          
          input_b = word_vecs[b][pointer[b]]
          stack_child[b].append((word_vecs[b][pointer[b]], word_vecs_zeros[b][pointer[b]]))
          pointer[b] += 1          
        else:
          input_b = new_child[0][child_idx].unsqueeze(0)
          stack_child[b].append((input_b, new_child[1][child_idx].unsqueeze(0)))
          child_idx += 1
        stack_input.append(input_b)
      stack_input = torch.cat(stack_input, 0)
      stack_h_all = []
      for k in range(self.num_layers):
        stack_h_all.append((torch.stack(stack_h[k][0], 0), torch.stack(stack_h[k][1], 0)))
      stack_h = self.stack_rnn(stack_input, stack_h_all)
      stack.append(stack_h)
      for b in range(batch_expand):
        stack2[b].append([[stack_h[k][0][b], stack_h[k][1][b]] for k in range(self.num_layers)])
      
    contexts = torch.stack(contexts, 1) #stack contexts
    action_logit_p = self.action_mlp_p(contexts).squeeze(2) 
    action_prob_p = torch.sigmoid(action_logit_p).clamp(min=1e-7, max=1-1e-7)
    action_shift_score = (1 - action_prob_p).log()
    action_reduce_score = action_prob_p.log()
    action_score = (1-actions)*action_shift_score + actions*action_reduce_score
    action_score = (action_score*action_masks).sum(1)
    
    word_contexts = contexts[actions < 1]
    word_contexts = word_contexts.contiguous().view(batch*samples, length, self.h_dim)

    log_probs_word = F.log_softmax(self.vocab_mlp(word_contexts), 2)
    log_probs_word = torch.gather(log_probs_word, 2, x_expand.unsqueeze(2)).squeeze(2)
    log_probs_word = log_probs_word.sum(1)
    log_probs_word = log_probs_word.contiguous().view(batch, samples)
    log_probs_action_p = action_score.contiguous().view(batch, samples)
    log_probs_action_q = log_probs_action_q.contiguous().view(batch, samples)
    actions = actions.contiguous().view(batch, samples, -1)
    return log_probs_word, log_probs_action_p, log_probs_action_q, actions, entropy

  def forward_actions(self, x, actions, has_eos=True):
    # this is for when ground through actions are available
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:]    
    if has_eos:
      new_actions = []
      for action in actions:
        new_actions.append(action + [self.S, self.R])
      actions = new_actions
    batch, length = x.size(0), x.size(1)
    word_vecs =  self.dropout(self.emb(x))
    actions = torch.Tensor(actions).float().cuda()
    action_masks = self.get_action_masks(actions, length)  # for not incurring loss when next action is deterministic (shift is deterministic when stack size <= 1)
    num_action = 2*length - 1
    contexts = []
    log_probs_action_p = [] #prior
    init_stack = self.stack_rnn(init_emb, None)
    word_vecs = word_vecs.unsqueeze(2)
    word_vecs_zeros = torch.zeros_like(word_vecs)
    stack = [init_stack]  # keeps each step's top stack element
    stack_child = [[] for _ in range(batch)]  # batch to current stack elements (used for TreeLSTM)
    stack2 = [[] for _ in range(batch)]  # batch to current stack elements (used for stack LSTM)
    pointer = [0]*batch
    for b in range(batch):
      stack2[b].append([[init_stack[l][0][b], init_stack[l][1][b]] for l in range(self.num_layers)])
    for l in range(num_action):
      contexts.append(stack[-1][-1][0])  # stack top state (hidden)
      stack_input = []
      child1_h = []
      child1_c = []
      child2_h = []
      child2_c = []
      stack_context = []
      for b in range(batch):
        if actions[b][l].item() == self.R:
          child1 = stack_child[b].pop()
          child2 = stack_child[b].pop()
          child1_h.append(child1[0])
          child1_c.append(child1[1])
          child2_h.append(child2[0])
          child2_c.append(child2[1])
          stack2[b].pop()
          stack2[b].pop()
      if len(child1_h) > 0:
        child1_h = torch.cat(child1_h, 0)
        child1_c = torch.cat(child1_c, 0)
        child2_h = torch.cat(child2_h, 0)
        child2_c = torch.cat(child2_c, 0)
        new_child = self.tree_rnn((child1_h, child1_c), (child2_h, child2_c))
      child_idx = 0
      stack_h = [[[], []] for _ in range(self.num_layers)]  # (n_layers, 2, batch_size, h_dim); last layer of stack2
      for b in range(batch):
        assert(len(stack2[b]) - 1 == len(stack_child[b]))
        for k in range(self.num_layers):
          stack_h[k][0].append(stack2[b][-1][k][0])
          stack_h[k][1].append(stack2[b][-1][k][1])
        if actions[b][l].item() == self.S:          
          input_b = word_vecs[b][pointer[b]]
          stack_child[b].append((word_vecs[b][pointer[b]], word_vecs_zeros[b][pointer[b]]))  # correspond to (h, c)
          pointer[b] += 1          
        else:
          input_b = new_child[0][child_idx].unsqueeze(0)
          stack_child[b].append((input_b, new_child[1][child_idx].unsqueeze(0)))
          child_idx += 1
        stack_input.append(input_b)
      stack_input = torch.cat(stack_input, 0)
      stack_h_all = []
      for k in range(self.num_layers):
        stack_h_all.append((torch.stack(stack_h[k][0], 0), torch.stack(stack_h[k][1], 0)))
      stack_h = self.stack_rnn(stack_input, stack_h_all)  # Each step always update the top element of stack. This updates the stack with that new element.
      stack.append(stack_h)
      for b in range(batch):
        stack2[b].append([[stack_h[k][0][b], stack_h[k][1][b]] for k in range(self.num_layers)])
    contexts = torch.stack(contexts, 1)  # (batch_size, n_actions, h_dim)
    action_logit_p = self.action_mlp_p(contexts).squeeze(2)
    action_prob_p = torch.sigmoid(action_logit_p).clamp(min=1e-7, max=1-1e-7)
    action_shift_score = (1 - action_prob_p).log()
    action_reduce_score = action_prob_p.log()
    action_score = (1-actions)*action_shift_score + actions*action_reduce_score
    action_score = (action_score*action_masks).sum(1)
    
    word_contexts = contexts[actions < 1]
    word_contexts = word_contexts.contiguous().view(batch, length, self.h_dim)
    log_probs_word = F.log_softmax(self.vocab_mlp(word_contexts), 2)
    log_probs_word = torch.gather(log_probs_word, 2, x.unsqueeze(2)).squeeze(2).sum(1)
    log_probs_action_p = action_score.contiguous().view(batch)
    actions = actions.contiguous().view(batch, 1, -1)
    return log_probs_word, log_probs_action_p, actions
  
  def forward_tree(self, x, actions, has_eos=True):
    # this is log q( tree | x) for discriminative parser training in supervised RNNG
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:-1]
    batch, length = x.size(0), x.size(1)
    scores = self.get_span_scores(x)
    crf_input = scores
    gold_spans = scores.new(batch, length, length)
    for b in range(batch):
      gold_spans[b].copy_(torch.eye(length).cuda())  # eye sets up token (single word) spans
      spans = get_spans(actions[b])
      for span in spans:
        gold_spans[b][span[0]][span[1]] = 1
    self.q_crf._forward(crf_input)
    log_Z = self.q_crf.alpha[0][length-1]
    span_scores = (gold_spans*scores).sum(2).sum(1)
    ll_action_q = span_scores - log_Z
    return ll_action_q
    
  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
      return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
      return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d    

class LSTMComposition(nn.Module):
  def __init__(self, dim, dropout):
    super(LSTMComposition, self).__init__()
    self.dim = dim
    self.rnn = nn.LSTM(dim, dim, bidirectional=True, batch_first=True)
    self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim*2, dim), nn.ReLU())

  def forward(self, children, ch_lengths, nt, nt_id, stack_state):
    lengths = ch_lengths + 2
    nt = nt.unsqueeze(1)
    elems = torch.cat([nt, children, torch.zeros_like(nt)], dim=1)
    for b, l in enumerate(lengths):
      elems[b][l-1] = nt[b]

    packed = pack_padded_sequence(elems, lengths, batch_first=True, enforce_sorted=False)
    h, _ = self.rnn(packed)
    h, _ = pad_packed_sequence(h, batch_first=True)

    gather_idx = (lengths - 2).unsqueeze(1).expand(-1, h.size(-1)).unsqueeze(1)
    fwd = h.gather(1, gather_idx).squeeze(1)[:, :self.dim]
    bwd = h[:, 1, self.dim:]
    c = torch.cat([fwd, bwd], dim=1)

    return self.output(c), None, None

class AttentionComposition(nn.Module):
  def __init__(self, w_dim, dropout, num_labels = 10):
    super(AttentionComposition, self).__init__()
    self.w_dim = w_dim
    self.num_labels = num_labels
    self.dropout = nn.Dropout(dropout)

    self.rnn = nn.LSTM(w_dim, w_dim, bidirectional=True, batch_first=True)

    self.V = nn.Linear(2*w_dim, 2*w_dim, bias=False)
    self.nt_emb = nn.Embedding(num_labels, w_dim)  # o_nt in the Kuncoro et al. (2017)
    self.nt_emb2 = nn.Sequential(nn.Embedding(num_labels, w_dim*2), self.dropout)  # t_nt in the Kuncoro et al. (2017)
    self.gate = nn.Sequential(nn.Linear(w_dim*4, w_dim*2), nn.Sigmoid())
    self.output = nn.Sequential(nn.Linear(w_dim*2, w_dim), nn.ReLU())

  def forward(self, children, ch_lengths, nt, nt_id, stack_state):  # children: (batch_size, n_children, w_dim)

    packed = pack_padded_sequence(children, ch_lengths, batch_first=True, enforce_sorted=False)
    h, _ = self.rnn(packed)
    h, _ = pad_packed_sequence(h, batch_first=True)  # (batch, n_children, 2*w_dim)

    rhs = torch.cat([self.nt_emb(nt_id), stack_state], dim=1) # (batch_size, w_dim*2, 1)
    logit = (self.V(h)*rhs.unsqueeze(1)).sum(-1)  # equivalent to bmm(self.V(h), rhs.unsqueeze(-1)).squeeze(-1)
    len_mask = length_to_mask(ch_lengths)
    logit[len_mask != 1] = -float('inf')
    attn = F.softmax(logit)
    weighted_child = (h*attn.unsqueeze(-1)).sum(1)
    weighted_child = self.dropout(weighted_child)

    nt2 = self.nt_emb2(nt_id)  # (batch_size, w_dim)
    gate_input = torch.cat([nt2, weighted_child], dim=-1)
    g = self.gate(gate_input)  # (batch_size, w_dim)
    c = g*nt2 + (1-g)*weighted_child  # (batch_size, w_dim)

    return self.output(c), attn, g

class TopDownRNNG(nn.Module):
  def __init__(self, action_dict,
               vocab = 100,
               padding_idx = 0,
               w_dim = 20,
               h_dim = 20,
               num_layers = 1,
               dropout = 0,
               attention_composition=False,
               max_open_nts = 100,
               max_cons_nts = 8,
  ):
    super(TopDownRNNG, self).__init__()
    self.action_dict = action_dict
    self.padding_idx = padding_idx
    self.action_criterion = nn.CrossEntropyLoss(reduction='none',
                                                ignore_index=action_dict.padding_idx)
    self.word_criterion = nn.CrossEntropyLoss(reduction='none',
                                              ignore_index=padding_idx)

    self.dropout = nn.Dropout(dropout)
    self.emb = nn.Sequential(
      nn.Embedding(vocab, w_dim, padding_idx=padding_idx), self.dropout)
    self.nt_emb = nn.Sequential(nn.Embedding(action_dict.num_nts(), w_dim), self.dropout)
    self.stack_rnn = SeqLSTM(w_dim, h_dim, num_layers=num_layers, dropout=dropout)
    self.stack_to_hidden = nn.Sequential(self.dropout, nn.Linear(h_dim, w_dim), nn.ReLU())
    self.vocab_mlp = nn.Linear(w_dim, vocab)
    self.num_layers = num_layers
    self.num_actions = action_dict.num_actions()  # num_labels + 2
    self.action_mlp = nn.Linear(w_dim, self.num_actions)
    self.w_dim = w_dim
    self.h_dim = h_dim
    self.vocab_mlp.weight = self.emb[0].weight

    self.composition = (AttentionComposition(w_dim, dropout, self.action_dict.num_nts())
                        if attention_composition else
                        LSTMComposition(w_dim, dropout))
    self.max_open_nts = max_open_nts
    self.max_cons_nts = max_cons_nts

    self.initial_emb = nn.Sequential(nn.Embedding(1, w_dim), self.dropout)

  def forward(self, x, actions, initial_stack = None):
    assert isinstance(x, torch.Tensor)
    assert isinstance(actions, torch.Tensor)

    states = self.initial_states(x, initial_stack)
    word_vecs = self.emb(x)
    action_contexts = self.unroll_states(states, word_vecs, actions)

    a_loss, _ = self.action_loss(actions, self.action_dict, action_contexts)
    w_loss, _ = self.word_loss(x, actions, self.action_dict, action_contexts)
    loss = (a_loss.sum() + w_loss.sum())
    return loss, a_loss, w_loss, states

  def unroll_states(self, states, word_vecs, actions):
    hs = [self.stack_top_h(states)]
    step = 0
    _batch_idx = word_vecs.new_zeros(1, word_vecs.size(0), dtype=torch.long)
    while not all(state.finished() for state in states):
      assert step < actions.size(1)

      action = actions[:, step]
      self.update_stack_rnn_train(states, action, word_vecs)
      hs.append(self.stack_top_h(states))
      step += 1
    batch_first_hs = torch.stack(hs[:-1], dim=1)
    action_contexts = self.stack_to_hidden(batch_first_hs)
    return action_contexts

  def action_loss(self, actions, action_dict, hiddens):
    assert hiddens.size()[:2] == actions.size()
    actions = actions.view(-1)
    hiddens = hiddens.view(actions.size(0), -1)

    action_mask = actions != action_dict.padding_idx
    idx = action_mask.nonzero().squeeze(1)
    actions = actions[idx]
    hiddens = hiddens[idx]

    logit = self.action_mlp(hiddens)
    loss = self.action_criterion(logit, actions)
    return loss, logit

  def word_loss(self, x, actions, action_dict, hiddens):
    assert hiddens.size()[:2] == actions.size()
    actions = actions.view(-1)
    hiddens = hiddens.view(actions.size(0), -1)

    action_mask = actions == action_dict.a2i['SHIFT']
    idx = action_mask.nonzero().squeeze(1)
    hiddens = hiddens[idx]

    x = x.view(-1)
    assert x.size(0) == hiddens.size(0)
    logit = self.vocab_mlp(hiddens)
    loss = self.word_criterion(logit, x)
    return loss, logit

  def get_initial_stack(self, x):
    iemb = self.initial_emb(x.new_zeros(x.size(0)))
    return self.stack_rnn(iemb, None)

  def word_sync_beam_search(self, x, beam_size, word_beam_size = 0, shift_size = 0):
    self.eval()

    if word_beam_size <= 0:
      word_beam_size = beam_size

    beam = self.initial_beam(x)
    word_completed = [[] for _ in range(x.size(0))]
    word_vecs = self.emb(x)
    word_marginal_ll = [[] for _ in range(x.size(0))]

    for pointer in range(x.size(1) + 1):
      forced_completions = [0 for _ in range(x.size(0))]
      bucket_i = 0
      while not all(len(batch_beam) == 0 for batch_beam in beam):
        successors, added_forced_completions = self.get_successors(
          x, pointer, beam, beam_size, shift_size)
        new_beam = [[beam[batch_i][b].next_incomplete_item(a, s) for (b, a, s) in batch_succs]
                    for batch_i, batch_succs in enumerate(successors)]
        for i, c in enumerate(added_forced_completions):
          forced_completions[i] += c

        all_items, batch_idx = self._flatten_items(new_beam)

        self.update_stack_rnn_beam(all_items, word_vecs, batch_idx, pointer)

        beam_lengths = [len(batch_beam) for batch_beam in new_beam]
        self.update_beam_and_word_completed(
          beam, word_completed, all_items, beam_lengths, pointer == x.size(1))

        for b in range(len(beam)):
          # Condition to stop beam for current word. This follows original implementation (rnng-bert),
          # but following the paper, the true condition might be
          # `len(word_completed[b]) >= word_beam_size` or
          # `len(word_completed[b]) - forced_completions[b] >= word_beam_size`?
          if len(word_completed[b]) - forced_completions[b] >= beam_size:
            beam[b] = []
        bucket_i += 1

      for b in range(len(beam)):
        word_completed[b].sort(key=lambda x: x.score, reverse=True)
        beam[b] = word_completed[b][:word_beam_size]
        word_completed[b] = []

      marginal = self._get_marginal_ll(beam)
      for b, s in enumerate(marginal):
        word_marginal_ll[b].append(s)

    parses = [[(item.parse_actions(), item.score) for item in  batch_beam]
              for batch_beam in beam]
    surprisals = [[] for _ in range(len(beam))]
    for b in range(len(beam)):
      for i in range(0, len(word_marginal_ll[b])-1):
        surprisals[b].append(-word_marginal_ll[b][i] - (-word_marginal_ll[b][i-1] if i > 0 else 0))
    return parses, surprisals

  def variable_beam_search(self, x, K, original_reweight=False):
    self.eval()

    beam = self.initial_particle_beam(x, K)
    word_completed = [[] for _ in range(x.size(0))]
    word_vecs = self.emb(x)
    word_marginal_ll = [[] for _ in range(x.size(0))]

    for pointer in range(x.size(1) + 1):
      bucket_i = 0
      while not all(len(batch_beam) == 0 for batch_beam in beam):
        new_beam = self.get_successors_by_particle_filter(x, pointer, beam)
        all_items, batch_idx = self._flatten_items(new_beam)
        self.update_stack_rnn_beam(all_items, word_vecs, batch_idx, pointer)
        beam_lengths = [len(batch_beam) for batch_beam in new_beam]
        self.update_beam_and_word_completed(
          beam, word_completed, all_items, beam_lengths, pointer == x.size(1))
        bucket_i += 1

      for b in range(len(beam)):
        beam[b] = self.reweight_and_filter_particles(
          word_completed[b], K, original_reweight=original_reweight)
        word_completed[b] = []

      marginal = self._get_marginal_ll(beam)
      for b, s in enumerate(marginal):
        word_marginal_ll[b].append(s)

    parses = [sorted([(item.parse_actions(), item.score) for item in batch_beam],
                     key=lambda x:x[1], reverse=True)
              for batch_beam in beam]
    surprisals = [[] for _ in range(len(beam))]
    for b in range(len(beam)):
      for i in range(0, len(word_marginal_ll[b])-1):
        surprisals[b].append(-word_marginal_ll[b][i] - (-word_marginal_ll[b][i-1] if i > 0 else 0))
    return parses, surprisals

  def initial_beam(self, x):
    states = self.initial_states(x)
    return [[BeamItem.from_initial_state(state)] for state in states]

  def initial_particle_beam(self, x, K):
    states = self.initial_states(x)
    return [[ParticleBeamItem.from_initial_state(state, K)] for state in states]

  def initial_states(self, x, initial_stack = None):
    initial_hs = self._initial_hs(x, initial_stack)
    return [TopDownState.from_initial_stack(h) for h in initial_hs]

  def _initial_hs(self, x, initial_stack = None):
    initial_stack = initial_stack or self.get_initial_stack(x)
    return [[(stack_layer[0][b], stack_layer[1][b]) for stack_layer in initial_stack]
            for b in range(x.size(0))]

  def get_successors(self, x, pointer, beam, beam_size, shift_size):
    all_items, _ = self._flatten_items(beam)
    beam_lengths = [len(batch_beam) for batch_beam in beam]
    states = [item.state for item in all_items]

    if pointer < x.size(1):
      next_x = [x[b, pointer].expand(beam_lengths[b]) for b in range(x.size(0))]
      next_x = torch.cat(next_x)  # (total beam size)
    else:
      next_x = None

    action_mask = self.valid_action_mask(all_items, x.size(1))  # (total beam size, n_actions)
    log_probs = self.action_log_probs(states, action_mask, next_x)  # tensor of size (total beam, n_actions)
    log_probs += log_probs.new_tensor([item.score for item in all_items]).unsqueeze(1)

    log_probs = self._deflatten_items(log_probs, beam_lengths)

    successors, forced_completions = self.scores_to_successors(
      x, pointer, beam, log_probs, beam_size, shift_size)
    return successors, forced_completions

  def get_successors_by_particle_filter(self, x, pointer, beam):
    all_items, _ = self._flatten_items(beam)
    beam_lengths = [len(batch_beam) for batch_beam in beam]
    states = [item.state for item in all_items]

    if pointer < x.size(1):
      next_x = [x[b, pointer].expand(beam_lengths[b]) for b in range(x.size(0))]
      next_x = torch.cat(next_x)  # (total beam size)
    else:
      next_x = None

    action_mask = self.valid_action_mask(all_items, x.size(1))  # (total beam size, n_actions)
    log_probs, disc_log_probs = self.action_log_probs(states, action_mask, next_x, return_disc_probs=True)
    new_K = (disc_log_probs.exp() * log_probs.new_tensor(
      [item.particle_path.K for item in all_items]).unsqueeze(1)).round_()
    new_K = new_K.view(-1)
    mask = new_K > 0.0  # (total beam size * n_actions)
    # action id can be obtained by (idx % n_actions)

    offset = 0
    successors = []
    num_actions = log_probs.size(1)  # (total_beam_size*n_actions, 1)
    log_probs = log_probs.view(-1)
    disc_log_probs = disc_log_probs.view(-1)
    # actions = torch.arange(0, mask.size(0), dtype=torch.long,
    #                        device=new_K.device) % num_actions
    for batch_i, beam_length in enumerate(beam_lengths):
      b, e = offset*num_actions, (offset+beam_length)*num_actions
      batch_active_idx = mask[b:e].nonzero().squeeze(1)
      beam_id = (batch_active_idx // num_actions).cpu().numpy()
      action_id = (batch_active_idx % num_actions).cpu().numpy()
      active_idx = batch_active_idx + offset*num_actions
      batch_K = new_K[active_idx].cpu().numpy()
      batch_log_probs = log_probs[active_idx].cpu().numpy()
      batch_disc_log_probs = disc_log_probs[active_idx].cpu().numpy()

      successors.append([beam[batch_i][beam_id[i]].next_incomplete_item(
        action_id[i], batch_K[i], batch_log_probs[i], batch_disc_log_probs[i])
                         for i in range(len(beam_id))])
      offset += beam_length

    return successors

  def reweight_and_filter_particles(self, batch_beam, K, upper_lex_size=1000,
                                    original_reweight=False):
    if original_reweight:
      scores = torch.tensor([[b.particle_path.K,
                              b.particle_path.gen_ll,
                              b.particle_path.disc_ll] for b in batch_beam])
      scores = torch.t(scores)
      log_weights = scores[0].log() + scores[1] - scores[2]
    else:
      log_weights = torch.tensor([b.particle_path.gen_ll for b in batch_beam])

    denom = torch.logsumexp(log_weights, 0)
    unround_Ks = (log_weights - denom).exp() * K
    new_Ks = unround_Ks.round()

    active = (new_Ks > 0).nonzero().squeeze(1)
    if active.size(0) > upper_lex_size:
      active_Ks = unround_Ks[active]
      _, sort_idx = torch.sort(active_Ks, descending=True)
      active = active[sort_idx[:upper_lex_size]].cpu().numpy()
    new_Ks = new_Ks.cpu().numpy()
    new_beam = []
    for i in active:
      b = batch_beam[i]
      b.particle_path.reweight(new_Ks[i])
      new_beam.append(b)

    return new_beam

  def action_log_probs(self, states, action_mask, next_x = None, return_disc_probs = False):
    """
    states: tensor of size (batch, h_dim)
    new_x: tensor of size (batch)
    """
    hiddens = self.stack_top_h(states)  # (total beam size, h_dim)
    hiddens = self.stack_to_hidden(hiddens)  # (total beam size, h_dim)

    action_logit = self.action_mlp(hiddens)
    action_logit[action_mask != 1] = -float('inf')

    log_probs = F.log_softmax(action_logit)  # (total beam size, num_actions)
    if return_disc_probs:
      disc_log_probs = log_probs.clone()

    if next_x is not None:  # shift is valid for next action
      word_logit = self.vocab_mlp(hiddens)
      shift_idx = self.action_dict.a2i['SHIFT']
      assert next_x.size(0) == hiddens.size(0)
      word_log_probs = self.word_criterion(word_logit, next_x) * -1.0  # (total beam size)
      log_probs[:, shift_idx] += word_log_probs

    if return_disc_probs:
      return (log_probs, disc_log_probs)
    else:
      return log_probs

  def valid_action_mask(self, items, sent_len):
    mask = torch.ones((len(items), self.num_actions), dtype=torch.uint8)
    mask[:, self.action_dict.padding_idx] = 0
    for b, item in enumerate(items):
      state = item.state
      prev_action = item.action
      if state.pointer == sent_len and state.nopen_parens == 0:  # finished
        mask[b, :] = 0
        continue
      if state.nopen_parens == 0:  # only nt
        assert state.pointer == 0 and len(state.stack) == 1
        self.action_dict.mask_shift(mask, b)
        self.action_dict.mask_reduce(mask, b)
      if state.pointer == sent_len:  # only reduce
        self.action_dict.mask_nt(mask, b)
        self.action_dict.mask_shift(mask, b)
      if (state.nopen_parens > self.max_open_nts or
          state.ncons_nts >= self.max_cons_nts):  # no more open
        self.action_dict.mask_nt(mask, b)

      if ((state.nopen_parens == 1 and state.pointer < sent_len) or # cannot reduce the unique open element at intermediate steps.
          self.action_dict.is_nt(prev_action) or # cannot reduce immediately after nt
          len(state.stack) < 3):
        self.action_dict.mask_reduce(mask, b)
    mask = mask.to(items[0].state.stack[0][0][0].device)

    return mask

  def stack_top_h(self, states):
    return torch.stack([state.stack[-1][-1][0] for state in states], dim=0)

  def scores_to_successors(self, x, pointer, beam, total_scores, beam_size, shift_size):
    successors = [[] for _ in range(len(total_scores))]
    forced_completions = [0 for _ in range(len(total_scores))]
    for batch, batch_total_scores in enumerate(total_scores):
      if batch_total_scores.size(0) == 0:
        continue  # this batch is finished.
      _, num_actions = batch_total_scores.size()

      scores = batch_total_scores.view(-1)
      sorted_scores, sort_idx = torch.sort(scores, descending=True)

      active_span = (sorted_scores != -float('inf')).nonzero()[-1][0] + 1
      sorted_scores = sorted_scores[:active_span]
      sort_idx = sort_idx[:active_span]

      beam_id = sort_idx // num_actions
      action_id = sort_idx % num_actions

      beam_id_np = beam_id.cpu().numpy()
      action_id_np = action_id.cpu().numpy()
      sorted_scores_np = sorted_scores.cpu().numpy()

      successors[batch] = [(beam_id_np[i], action_id_np[i], sorted_scores_np[i])
                           for i in range(min(len(beam_id_np), beam_size))]
      if pointer < x.size(1):
        # shift is target for being forced.
        shift_in_successors = (action_id[:beam_size] == self.action_dict.a2i['SHIFT']).nonzero().size(0)
        if shift_in_successors < shift_size:
          # find and add additional forced shift successors
          additional = ((action_id[beam_size:] == self.action_dict.a2i['SHIFT'])
                        .nonzero()[:shift_size - shift_in_successors].squeeze(1)).cpu()
          additional += beam_size  # add offset
          assert all(action_id_np[i] == self.action_dict.a2i['SHIFT'] for i in additional)
          successors[batch] += [(beam_id_np[i], action_id_np[i], sorted_scores_np[i])
                                for i in additional]
          forced_completions[batch] = additional.size(0)
      else:
        # At the end, save final actions.
        finish_actions = [(b, a, s) for (b, a, s) in successors[batch]
                          if self._is_last_action(a, beam[batch][b].state, True)]

        if len(finish_actions) < shift_size:
          remain = shift_size - len(finish_actions)
          additional = []
          for i in range(beam_size, len(beam_id_np)):
            b = beam_id_np[i]
            a = action_id_np[i]
            if self._is_last_action(a, beam[batch][b].state, True):
              additional.append(i)
            if len(additional) >= remain:
              break
          successors[batch] += [(beam_id_np[i], action_id_np[i], sorted_scores_np[i])
                                for i in additional]
          forced_completions[batch] = len(additional)

    return successors, forced_completions

  def update_stack_rnn_train(self, states, action, word_vecs):
    shift_batch = (action == self.action_dict.a2i['SHIFT']).nonzero().squeeze(1)
    pointer = action.new_tensor([state.pointer for state in states])
    shift_batch_word_vecs = word_vecs[shift_batch]  # (num_shifted, sent len, w_dim)
    shift_batch_pointer = pointer[shift_batch]
    idx = shift_batch_pointer.view(-1, 1, 1).expand(-1, 1, word_vecs.size(-1))
    shifted_embs = torch.gather(shift_batch_word_vecs, 1, idx).squeeze(1)

    self.update_stack_rnn(states, action, shift_batch, shifted_embs)

  def update_stack_rnn_beam(self, items, word_vecs, batch_idx, pointer):
    """
    This method does some preparation for update_stack_rnn, which is used also
    for training.

    items is a flattened tensor. batch_idx is mapping from index in flattend
    list to the original batch, which is necessary to get correct mapping from
    word_vecs, which is not flattened and has size of (batch_size, num words).
    """
    states = [item.state for item in items]
    actions = word_vecs.new_tensor([item.action for item in items], dtype=torch.long)

    batch_idx_tensor = word_vecs.new_tensor(batch_idx, dtype=torch.long)
    shifts = (actions == self.action_dict.a2i['SHIFT']).nonzero().squeeze(1)

    shifted_batch_idx = batch_idx_tensor[shifts]
    if pointer < word_vecs.size(1):
      shifted_words = word_vecs[shifted_batch_idx, pointer]
    else:
      # At end of sentence. Since shift is not allowed, shifted_words will become
      # empty. We have to process this condition separately.
      assert shifted_batch_idx.size(0) == 0
      shifted_words = word_vecs[shifted_batch_idx, -1]

    self.update_stack_rnn(states, actions, shifts, shifted_words)

  def update_stack_rnn(self, states, actions, shift_idx, shifted_embs):
    assert actions.size(0) == len(states)
    assert shift_idx.size(0) == shifted_embs.size(0)
    if len(states) == 0:
      return

    reduces = (actions == self.action_dict.a2i['REDUCE']).nonzero().squeeze(1)
    nts = (actions >= self.action_dict.nt_begin_id()).nonzero().squeeze(1)

    new_stack_input = shifted_embs.new_zeros(actions.size(0), self.w_dim)

    if reduces.size(0) > 0:
      reduce_idx = reduces.cpu().numpy()
      reduce_states = [states[i] for i in reduce_idx]
      children, ch_lengths, nt, nt_id = self._collect_children_for_reduce(reduce_states)
      reduce_context = self.stack_top_h(reduce_states)
      reduce_context = self.stack_to_hidden(reduce_context)
      new_child, _, _ = self.composition(children, ch_lengths, nt, nt_id, reduce_context)
      new_stack_input[reduces] = new_child

    new_stack_input[shift_idx] = shifted_embs


    nt_ids = (actions[nts] - self.action_dict.nt_begin_id())
    nt_embs = self.nt_emb(nt_ids)
    new_stack_input[nts] = nt_embs

    stack_top_context = self._collect_stack_top_context(states)
    new_stack_top = self.stack_rnn(new_stack_input, stack_top_context)

    for b in range(len(states)):
      new_stack_top_b = [[layer[0][b], layer[1][b]] for layer in new_stack_top]
      states[b].update_stack(new_stack_top_b, new_stack_input[b])
      states[b].do_action(actions[b].item(), self.action_dict)

  def update_beam_and_word_completed(self, beam, word_completed, items, beam_lengths, last_token=False):
    accum = 0
    for b in range(len(beam)):
      beam[b] = []
      l = beam_lengths[b]
      for item in items[accum:accum+l]:
        if (item.state.finished() or self.action_dict.is_shift(item.action)):
          # Cases where shifting last+1 token should be pruned by action masking.
          word_completed[b].append(item)
        else:
          beam[b].append(item)
      accum += l
    assert accum == sum(beam_lengths)

  def _is_last_action(self, action, state, shifted_all):
    return (shifted_all and self.action_dict.is_reduce(action) and
            state.can_finish_by_reduce())

  def _flatten_items(self, items):
    flatten_items = []
    idxs = []
    for i, batch_items in enumerate(items):
      flatten_items += batch_items
      idxs += [i] * len(batch_items)
    return flatten_items, idxs

  def _deflatten_items(self, items, beam_lengths):
    de_items = []
    accum_idx = 0
    for b, b_len in enumerate(beam_lengths):
      de_items.append(items[accum_idx:accum_idx+b_len])
      accum_idx += b_len
    assert accum_idx == items.size(0)
    return de_items  # List[Tensor]

  def _collect_children_for_reduce(self, reduce_states):
    children = []
    nt = []
    nt_ids = []
    ch_lengths = []
    for state in reduce_states:
      reduced_nt, reduced_trees, nt_id = state.reduce_stack()
      nt.append(reduced_nt)
      nt_ids.append(nt_id)
      children.append(reduced_trees)
      ch_lengths.append(len(reduced_trees))

    max_ch_len = max(ch_lengths)
    zero_child = children[0][0].new_zeros(children[0][0].size())
    for b, b_child in enumerate(children):
      if len(b_child) < max_ch_len:
        children[b] += [zero_child] * (max_ch_len - len(b_child))
      children[b] = torch.stack(children[b], 0)

    children = torch.stack(children, 0)
    nt = torch.stack(nt, 0)
    nt_ids = nt.new_tensor(nt_ids, dtype=torch.long)
    ch_lengths = nt_ids.new_tensor(ch_lengths)

    return (children, ch_lengths, nt, nt_ids)

  def _collect_stack_top_context(self, states, idx = None, offset = 1):
    if idx is None:
      idx = range(len(states))
    stack_h_all = []
    for l in range(self.num_layers):
      h = [states[i].stack[-offset][l][0] for i in idx]
      c = [states[i].stack[-offset][l][1] for i in idx]
      stack_h_all.append((torch.stack(h, 0), torch.stack(c, 0)))
    return stack_h_all

  def _get_marginal_ll(self, beam):
    ll = []
    for b in range(len(beam)):
      scores = [item.score for item in beam[b]]
      ll.append(torch.logsumexp(torch.tensor(scores), 0).item())
    return ll

class TopDownState:
  def __init__(self,
               pointer = 0,
               stack = None,
               stack_trees = None,
               nopen_parens = 0,
               ncons_nts = 0,
               nt_index = None,
               nt_ids = None):
    self.pointer = pointer
    self.stack = stack or []
    self.stack_trees = stack_trees or []
    self.nopen_parens = nopen_parens
    self.ncons_nts = ncons_nts
    self.nt_index = nt_index or []
    self.nt_ids = nt_ids or []

  def stack_str(self, action_dict):
    stack_str = ''
    stack_str = ['<dummy>'] + ['C' for _ in range(len(self.stack)-1)]
    for i, nt in zip(self.nt_index, self.nt_ids):
      stack_str[i] = '(' + action_dict.nonterminals[nt]
    assert self.nopen_parens == len(self.nt_ids) == len(self.nt_index)

    stack_str = ' '.join(stack_str)
    return '{{[ {} ], pointer={}}}'.format(stack_str, self.pointer)

  @classmethod
  def from_initial_stack(cls, initial_stack_elem):
    return cls(stack=[initial_stack_elem])

  def can_finish_by_reduce(self):
    return self.nopen_parens == 1

  def finished(self):
    return self.pointer > 0 and self.nopen_parens == 0

  def copy(self):
    return TopDownState(self.pointer, self.stack[:], self.stack_trees[:],
                        self.nopen_parens, self.ncons_nts, self.nt_index[:], self.nt_ids[:])

  def do_action(self, a, action_dict):
    if action_dict.is_shift(a):
      self.pointer += 1
      self.ncons_nts = 0
    elif action_dict.is_nt(a):
      nt_id = action_dict.nt_id(a)
      self.nopen_parens += 1
      self.ncons_nts += 1
      self.nt_index.append(len(self.stack) - 1)
      self.nt_ids.append(nt_id)
    elif action_dict.is_reduce(a):
      self.nopen_parens -= 1
      self.ncons_nts = 0

  def reduce_stack(self):
    open_idx = self.nt_index.pop()
    nt_id = self.nt_ids.pop()
    self.stack = self.stack[:open_idx]
    reduce_trees = self.stack_trees[open_idx-1:]
    self.stack_trees = self.stack_trees[:open_idx-1]
    return reduce_trees[0], reduce_trees[1:], nt_id

  def update_stack(self, new_stack_top, new_tree_elem):
    self.stack.append(new_stack_top)
    self.stack_trees.append(new_tree_elem)

class ActionPath:
  def __init__(self, prev=None, action=0, score=0.0):
    self.prev = prev
    self.action = action
    self.score = score

  def add_action(self, action, score):
    return ActionPath(self, action, score)

  def foreach(self, f):
    f(self)
    if self.prev is not None:
      self.prev.foreach(f)

class BeamItem:
  def __init__(self, state, action_path):
    self.state = state
    self.action_path = action_path
    self.action = self.action_path.action
    self.score = self.action_path.score

  def do_action(self, action_dict):
    self.state.do_action(self.action, action_dict)

  @staticmethod
  def from_initial_state(initial_state):
    path = ActionPath()  # initial action is 0 (pad)
    return BeamItem(initial_state, path)

  def parse_actions(self):
    actions = []
    def add_action(path):
      actions.append(path.action)
    self.action_path.foreach(add_action)
    assert actions[-1] == 0  # last (= initial after revsed) is pad (dummy)
    return list(reversed(actions[:-1]))

  def next_incomplete_item(self, action, score):
    state = self.state.copy()
    path = self.action_path.add_action(action, score)
    return BeamItem(state, path)

class ParticlePath:
  def __init__(self, K, prev=None, action=0, gen_ll=0.0, disc_ll=0.0):
    self.K = K
    self.prev = prev
    self.action = action
    self.gen_ll = gen_ll
    self.disc_ll = disc_ll

  def add_action(self, action, K, gen_local_score, disc_local_score):
    return ParticlePath(K, self, action, self.gen_ll+gen_local_score, self.disc_ll+disc_local_score)

  def normalized_weight(self):
    return self.K * np.exp(self.gen_ll - self.disc_ll)

  def reweight(self, new_K):
    self.K = new_K

  def foreach(self, f):
    f(self)
    if self.prev is not None:
      self.prev.foreach(f)

class ParticleBeamItem:
  def __init__(self, state, particle_path):
    self.state = state
    self.particle_path = particle_path
    self.action = self.particle_path.action
    self.score = self.particle_path.gen_ll

  @staticmethod
  def from_initial_state(initial_state, K):
    path = ParticlePath(K)  # initial action is 0 (pad)
    return ParticleBeamItem(initial_state, path)

  def parse_actions(self):
    actions = []
    def add_action(path):
      actions.append(path.action)
    self.particle_path.foreach(add_action)
    assert actions[-1] == 0  # last (= initial after revsed) is pad (dummy)
    return list(reversed(actions[:-1]))

  def next_incomplete_item(self, action, K, log_prob, disc_log_prob):
    state = self.state.copy()
    path = self.particle_path.add_action(action, K, log_prob, disc_log_prob)
    return ParticleBeamItem(state, path)
