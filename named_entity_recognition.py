import numpy as np
import pandas as pd
from scipy.special import softmax

from simpletransformers.ner import NERModel

# Creating train_df  and eval_df for demonstration
train_data = [
    [0, "Simple", "B-MISC"],
    [0, "Transformers", "I-MISC"],
    [0, "started", "O"],
    [0, "with", "O"],
    [0, "text", "O"],
    [0, "classification", "B-MISC"],
    [1, "Simple", "B-MISC"],
    [1, "Transformers", "I-MISC"],
    [1, "can", "O"],
    [1, "now", "O"],
    [1, "perform", "O"],
    [1, "NER", "B-MISC"],
]
train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

eval_data = [
    [0, "Simple", "B-MISC"],
    [0, "Transformers", "I-MISC"],
    [0, "was", "O"],
    [0, "built", "O"],
    [0, "for", "O"],
    [0, "text", "O"],
    [0, "classification", "B-MISC"],
    [1, "Simple", "B-MISC"],
    [1, "Transformers", "I-MISC"],
    [1, "then", "O"],
    [1, "expanded", "O"],
    [1, "to", "O"],
    [1, "perform", "O"],
    [1, "NER", "B-MISC"],
]
eval_df = pd.DataFrame(eval_data, columns=["sentence_id", "words", "labels"])

# Create a NERModel
model_args = {
    'overwrite_output_dir': True,
    'reprocess_input_data': True,
    
    'save_eval_checkpoints': False,
    'save_steps': -1,
    'save_model_every_epoch': False,
    
    'train_batch_size': 10,
    'num_train_epochs': 100,   # 5
    'max_seq_length': 256,
    'gradient_accumulation_steps': 8
}

model = NERModel("bert", "bert-base-cased", use_cuda=False, args=model_args)
    # args={"overwrite_output_dir": True, "reprocess_input_data": True}

train_df = eval_df

# # Train the model
model.train_model(train_df)

# # Evaluate the model
result, model_outputs, predictions = model.eval_model(eval_df)


# Predictions on arbitary text strings
sentences = ["Some arbitary sentence", "Simple Transformers sentence"]
predictions, raw_outputs = model.predict(sentences)

print(predictions)

# More detailed preditctions
for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
    print("\n___________________________")
    print("Sentence: ", sentences[n])
    for pred, out in zip(preds, outs):
        key = list(pred.keys())[0]
        new_out = out[key]
        preds = list(softmax(np.mean(new_out, axis=0)))
        print(key, pred[key], preds[np.argmax(preds)], preds)
