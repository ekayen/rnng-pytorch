#!/bin/bash

#python beam_search.py --test_file data/swbd/sent/swbd-dev.json --model_file sent_speech.pt --batch_size 20  --block_size 100000 > sent_speech.pred
#python beam_search.py --test_file data/swbd/sent/swbd-dev.json --model_file sent_text.pt --batch_size 20  --block_size 100000 > sent_text.pred
python beam_search.py --test_file data/swbd/turn/swbd-dev.json --model_file turn_speech.pt --batch_size 10  --block_size 100000 > turn_speech.pred
python beam_search.py --test_file data/swbd/turn/swbd-dev.json --model_file turn_text.pt --batch_size 10  --block_size 100000 > turn_text.pred
