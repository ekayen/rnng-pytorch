from PYEVALB import scorer

sent_gold = "data/swbd/sent/dev.trees"
turn_gold = "data/swbd/turn/dev.trees"
sent_preds = ["sent_speech.pred","sent_text.pred"]
turn_preds = ["turn_speech.pred","turn_text.pred"]

pred = 'lookahead.pred'
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
"""
"""
for pred in turn_preds:
  print(pred)  
  scr = scorer.Scorer()
  scr.evalb(turn_gold,pred,pred.replace('pred','eval'))
"""
