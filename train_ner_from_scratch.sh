rm lm_outputs_test/* -rf
python lm_train/train_new_lm.py
python ner_lm/train_complex_dataset.py
