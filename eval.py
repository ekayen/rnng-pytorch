from PYEVALB import scorer

sent_gold = "data/swbd/sent/dev.trees"
turn_gold = "data/swbd/turn/dev.trees"
sent_preds = ["sent_speech.pred","sent_text.pred"]
turn_preds = ["turn_speech.pred","turn_text.pred"]


#pred = 'b0f1-frames-only-d3-lr0.0001.pred'
#pred = 'b0f1_boundary.pred'
pred = 'b0f1_all_feats.pred'
#pred = 'b0f1-frames-only-d1-lr0.001.pred'
#pred = 'b0f1-tok-only.pred'
#pred = 'b0f1-frames-only-preproc.pred'
#pred = 'turn_speech_rerun.pred'
scr = scorer.Scorer()
scr.evalb(turn_gold,pred,pred.replace('pred','eval'))


"""
lrs = ['0.00001','0.0001','0.001','0.01','0.1']
drpts = ['0']#['0.1','0.3','0.5','0.7']
for lr in lrs:
    for drpt in drpts:

        pred = f'sent_text_lr{lr}_d{drpt}.pred'
        print(pred)
        scr = scorer.Scorer()
        scr.evalb(sent_gold,pred,pred.replace('pred','eval'))
"""
"""
for pred in sent_preds:
  print(pred)
  scr = scorer.Scorer()
  scr.evalb(sent_gold,pred,pred.replace('pred','eval'))

for pred in turn_preds:
  print(pred)  
  scr = scorer.Scorer()
  scr.evalb(turn_gold,pred,pred.replace('pred','eval'))

"""
