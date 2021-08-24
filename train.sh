#!/bin/bash

python train.py --train_file data/swbd/sent/swbd-train.json --val_file data/swbd/sent/swbd-dev.json --save_path sent_speech.pt --batch_size 64 --fixed_stack --strategy in_order --dropout 0.3 --optimizer adam --lr 0.001 --speech_feat_types pitch,fbank,pause,dur

python train.py --train_file data/swbd/sent/swbd-train.json --val_file data/swbd/sent/swbd-dev.json --save_path sent_text.pt --batch_size 64 --fixed_stack --strategy in_order --dropout 0.3 --optimizer adam --lr 0.001

python train.py --train_file data/swbd/turn/swbd-train.json --val_file data/swbd/turn/swbd-dev.json --save_path turn_text.pt --batch_size 64 --fixed_stack --strategy in_order --dropout 0.3 --optimizer adam --lr 0.001

python train.py --train_file data/swbd/turn/swbd-train.json --val_file data/swbd/turn/swbd-dev.json --save_path turn_speech.pt --batch_size 128 --fixed_stack --strategy in_order --dropout 0.3 --optimizer adam --lr 0.001 --speech_feat_types pitch,fbank,pause,dur

