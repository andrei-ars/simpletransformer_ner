rm lm_outputs_test/* -rf
python lm_train/train_new_lm.py
python ner_lm/train_complex_dataset.py

# Two training steps.
# 1) train new LM, save it into lm_outputs_test/from_scratch/best_model
# 2) train on NER task and save it into outputs_nlp_complex (or into another folder depending on modelname)
