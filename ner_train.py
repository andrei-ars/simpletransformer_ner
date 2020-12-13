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



class NerModel:
    def __init__(self, dataset=None):

        if dataset:
            self.dataset = dataset
        else:
            raise Exception("Dataset is not set")

        labels_list=["O", "B-ACT",  "I-ACT", "B-OBJ", "I-OBJ", "B-VAL", "I-VAL", "B-VAR", "I-VAR"]

        # Create a NERModel
        model_args = {
            'overwrite_output_dir': True,
            'reprocess_input_data': True,
            
            'save_eval_checkpoints': False,
            'save_steps': -1,
            'save_model_every_epoch': False,
            
            'train_batch_size': 10, # 10
            'num_train_epochs': 5,   # 5
            'max_seq_length': 256,
            'gradient_accumulation_steps': 8,

            'labels_list': labels_list
        }

        self.model = NERModel("bert", "bert-base-cased", use_cuda=False, args=model_args)
            # args={"overwrite_output_dir": True, "reprocess_input_data": True}

    def train(self):
        # # Train the model
        self.model.train_model(self.dataset['train'])

    def eval(self):
        # # Evaluate the model
        result, model_outputs, predictions = self.model.eval_model(self.dataset['val'])

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


class NerDataset:
    def __init__(self, path):
        self.data = []

        with open(path) as fp:
            count = 0
            for line in fp:
                ls = line.split()
                if len(ls) == 2:
                    token, tag = ls
                    if token == ".":
                        count += 1
                    else:
                        print("{}: {} - {}".format(count, token, tag))
                        self.data.append([count, token, tag])
                else:
                    print("len != 2")
                    raise Exception("NerDataset.init: len != 2")

        #train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=["sentence_id", "words", "labels"])

    #def as_dict(self):
    #    dc = {'train': self.dataset.train,
    #          'val': self.dataset.val,
    #          'test': self.dataset.test,
    #            }


if __name__ == "__main__":
    dataset = {}
    dataset['train'] = NerDataset("dataset/train.txt").to_dataframe()
    dataset['val'] = NerDataset("dataset/valid.txt").to_dataframe()
    dataset['test'] = NerDataset("dataset/test.txt").to_dataframe()
    model = NerModel(dataset)
    model.train()
    model.eval()

