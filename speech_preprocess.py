"""
Creates a dictionary where 
   Key = sentence ID
   Value = sentence speech feats

   The value is a dictionary where:
      key = <pause,dur,pitch,fbank,pitchfbank>
      value = (list of?) np arrays of that feature, one array per token

Different settings:

- Number of backward tokens in context for each token
- Number of forward tokens in context for each token
- Which features are included in the context window extension (all vs. just frame-based)
"""
import os
import pickle


def main():
    outdir = '/afs/inf.ed.ac.uk/group/project/prosody/rnng-pytorch/data/speech'
    
    turndir = '/afs/inf.ed.ac.uk/group/project/prosody/prosody_nlp/data/input_features/turn_pause_dur_fixed'
    sentdir = '/afs/inf.ed.ac.uk/group/project/prosody/prosody_nlp/data/input_features/sentence_short'

    dirs = {'sent':sentdir,'turn':turndir}

    dir_prefix = {'sent':'','turn':'turn_'}

    splits = ['dev','train','test']

    for di in dirs:
        directory = dirs[di]
        for split in splits:
            partition = pickle.load(open(os.path.join(directory,f'{dir_prefix[di]}{split}_partition.pickle'),'rb'))
            pitch = pickle.load(open(os.path.join(directory,f'{dir_prefix[di]}{split}_pitch.pickle'),'rb'))
            fbank = pickle.load(open(os.path.join(directory,f'{dir_prefix[di]}{split}_fbank.pickle'),'rb'))
            pause = pickle.load(open(os.path.join(directory,f'{dir_prefix[di]}{split}_pause.pickle'),'rb'))
            duration = pickle.load(open(os.path.join(directory,f'{dir_prefix[di]}{split}_duration.pickle'),'rb'))

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
                for idx,prt in enumerate(part):
                    start,end = prt
                    pau_tok.append(pau['pause_aft'][idx])
                    dur_tok.append(dur[:,idx])
                    pit_tok.append(pit[:,start:end])
                    fb_tok.append(fb[:,start:end])
                    
                out_dict[idnum]['pause'] = pau_tok
                out_dict[idnum]['pitch'] = pit_tok
                out_dict[idnum]['fbank'] = fb_tok
                out_dict[idnum]['dur'] = dur_tok

            with open(os.path.join(outdir,di,f'{split}.pickle'),'wb') as f:
                pickle.dump(out_dict,f)

if __name__=="__main__":
    main()

