import numpy as np
import pandas as pd
from scipy.special import softmax
from simpletransformers.ner import NERModel


class NerModel:
    def __init__(self, modelname="", dataset=None, use_saved_model=False):
        self.dataset = dataset
        #labels_list = ["O", "B-ACT",  "I-ACT", "B-OBJ", "I-OBJ", "B-VAL", "I-VAL", "B-VAR", "I-VAR"]
        #labels_list = dataset.get_labels_list()
        labels_list = dataset['labels_list']

        output_dir = "outputs_{}".format(modelname)
        # Create a NERModel
        model_args = {
            'output_dir': output_dir,
            'overwrite_output_dir': True,
            'reprocess_input_data': True,
            
            'save_eval_checkpoints': False,
            'save_steps': -1,
            'save_model_every_epoch': False,
            
            'train_batch_size': 10, # 10
            'num_train_epochs': 10,   # 5
            'max_seq_length': 256,
            'gradient_accumulation_steps': 8,

            'labels_list': labels_list
        }
                
        if use_saved_model:
            self.model = NERModel("bert", output_dir, use_cuda=False, args=model_args)
        else:
            self.model = NERModel("bert", "bert-base-cased", use_cuda=False, args=model_args)
            # args={"overwrite_output_dir": True, "reprocess_input_data": True}

    def train(self):
        # # Train the model
        if self.dataset:
            self.model.train_model(self.dataset['train'])
        else:
            raise Exception("dataset is None")

    def eval(self):
        # # Evaluate the model
        if self.dataset:
            result, model_outputs, predictions = self.model.eval_model(self.dataset['val'])
            print("Evaluation result:", result)
        else:
            raise Exception("dataset is None")

    def simple_test(self):
        # Predictions on arbitary text strings
        sentences = ["Some arbitary sentence", "Simple Transformers sentence"]
        predictions, raw_outputs = self.model.predict(sentences)
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

    def predict(self, sentences):
        predictions, raw_outputs = self.model.predict(sentences)
        return predictions

