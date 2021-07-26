from PYEVALB import scorer

sent_gold = "data/swbd/sent/dev.trees"
turn_gold = "data/swbd/turn/dev.trees"
sent_preds = ["sent_speech.pred","sent_text.pred"]
turn_preds = ["turn_speech.pred","turn_text.pred"]

for pred in sent_preds:
  print(pred)
  scr = scorer.Scorer()
  scr.evalb(sent_gold,pred,pred.replace('pred','eval'))
for pred in turn_preds:
  print(pred)  
  scr = scorer.Scorer()
  scr.evalb(turn_gold,pred,pred.replace('pred','eval'))
