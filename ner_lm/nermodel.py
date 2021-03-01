import sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from simpletransformers.ner import NERModel
from sklearn.metrics import f1_score, accuracy_score



def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


class NerModel:
    def __init__(self, modelname="", dataset=None, use_saved_model=False):
        
        pretrained_model_name = "lm_outputs/from_scratch/best_model"
        pretrained_model_name = "lm_outputs_test/from_scratch/best_model"
        #pretrained_model_name = f"../lm_outputs/from_scratch/best_model"
        #pretrained_model_name = f"../lm_outputs_test/from_scratch/best_model"
        #pretrained_model_name = f"../lm_outputs_test/from_scratch_"

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
            #'no_save' : True,
            #'no_cache': True,
            
            'num_train_epochs': 10,   # 5
            'train_batch_size': 10, # 10
            'eval_batch_size' : 10,
            'evaluate_during_training' : True,

            'max_seq_length': 128, #256,
            'gradient_accumulation_steps': 8,

            'labels_list': labels_list
        }

        self.model = NERModel("bert", pretrained_model_name, use_cuda=False, args=model_args)
        #self.model = NERModel("bert", "bert-base-uncased", use_cuda=False, args=model_args)
        #self.model = NERModel("electra", 'google/electra-small-generator', use_cuda=False, args=model_args)
        #self.model = NERModel("layoutlm", 'microsoft/layoutlm-base-uncased', use_cuda=False, args=model_args)

        #self.model = NERModel("distilbert", "distilbert-base-cased-distilled-squad", use_cuda=False, args=model_args)

        """
        if use_saved_model:
            self.model = NERModel("bert", output_dir, use_cuda=False, args=model_args)
        else:
            self.model = NERModel("bert", pretrained_model_name, use_cuda=False, args=model_args)
            # args={"overwrite_output_dir": True, "reprocess_input_data": True}
        """

    def train(self):
        # # Train the model
        if self.dataset:
            self.model.train_model(self.dataset['train'], eval_data=self.dataset['val'])
        else:
            raise Exception("dataset is None")

    def eval(self):
        # # Evaluate the model
        if self.dataset:
            result, model_outputs, predictions = self.model.eval_model(
                                                    self.dataset['val'])
            print("Evaluation result:", result)
        else:
            raise Exception("dataset is None")

    def test(self):
        sentence_id = self.dataset['test']['sentence_id']
        words = self.dataset['test']['words']
        labels = self.dataset['test']['labels']
        
        prev_id = 0
        s_words = []
        s_labels = []

        samples = []

        for i in range(len(sentence_id)):
            s_id = sentence_id[i]
            word = words[i]
            label = labels[i]

            if s_id != prev_id:
                sentence = " ".join(s_words)
                print("sentence id={}: {}".format(prev_id, sentence))
                samples.append({'text': sentence, 'tokens': s_words, 'labels': s_labels})
                #print("s_labels: {}".format(s_labels))
                s_words = []
                s_labels = []
                prev_id = s_id

            s_words.append(words[i])
            s_labels.append(labels[i])
            #print("i={}, word={}, label={}".format(s_id, word, label))

        sentence = " ".join(s_words)
        print("sentence id={}: {}".format(prev_id, sentence))
        samples.append({'text': sentence, 'tokens': s_words, 'labels': s_labels})

        texts = [sample['text'] for sample in samples]
        predictions, raw_outputs = self.model.predict(texts)
        #print(predictions)

        acc_list = []
        success_list = []

        # More detailed preditctions
        for i, (preds, raw_outs) in enumerate(zip(predictions, raw_outputs)):
            print()
            print("text: ", texts[i])
            #print("\npreds: ", preds)
            pred_labels = [list(t.values())[0] for t in preds]
            print("pred_labels: ", pred_labels)
            true_labels = samples[i]['labels']
            print("true_labels: ", true_labels)
            #print("raw_outs: ", raw_outs)
            
            if len(true_labels) != len(pred_labels):
                raise Exception("len(true_labels) != len(pred_labels)")
            comp = [true_labels[i] == pred_labels[i] for i in range(len(pred_labels))]
            acc1sentence = np.mean(comp)
            print("acc={:.3f}".format(acc1sentence))
            acc_list.append(acc1sentence)
            success = 1 if acc1sentence == 1.0 else 0
            success_list.append(success)

        avg_acc = np.mean(acc_list)
        print()
        print("avg acc={:.3f}".format(avg_acc))
        avg_success = np.mean(success_list)
        print("avg success={:.3f}".format(avg_success))

            #for pred, out in zip(preds, outs):
                #print("pred:", pred)
                #print("out:", out)
                #key = list(pred.keys())[0]
                #new_out = out[key]
                #preds = list(softmax(np.mean(new_out, axis=0)))
                #print(key, pred[key], preds[np.argmax(preds)], preds)


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
        print("raw_outputs:", raw_outputs)
        return predictions

