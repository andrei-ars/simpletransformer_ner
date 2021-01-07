import os
import sys
import pandas as pd
import random
import logging

from simpletransformers.classification import ClassificationModel


class ActionDataset():

    def __init__(self, data_path):

        self.index_to_label = {}
        self.label_to_index = {}

        line_by_line = True
        if os.path.isdir(data_path):
            if line_by_line:
                data = self.load_docs_line_by_line_from_files_in_directory(data_path)
            else:
                data = self.load_docs_from_directory(data_path)
        elif os.path.isfile(data_path):
            data = self.load_docs_from_file(data_path)
        else:
            raise Exception("data_path {} is not a file or a directory.".format(data_path))

        data = [[sample[0], sample[2]] for sample in data]
        # We will get:
        # [['Click zzdo .', 22], ['fill somedata .', 18], ['verify last BOQ zzxu zzpu EOQ .', 16],...

        random.shuffle(data)
        data_size = len(data)
        train_part = 0.7
        train_size = int(train_part * data_size)
        train_data = data[:train_size]
        val_data = data[train_size:]

        self.data = {'train': train_data, 'val': val_data}


    def load_docs_line_by_line_from_files_in_directory(self, data_path):
        """ Each line in each file in data_path is a document.
        """
        labels = [f for f in os.listdir(data_path)] #  if f.endswith('.txt')
        self.index_to_label = {index: label for index, label in enumerate(labels)}
        self.label_to_index = {label: index for index, label in enumerate(labels)}

        data = []
        self.text_number_to_label = {}
        for index, label in enumerate(labels):
            with open(data_path + '/' + label, 'r') as f:
                for line in f:
                    text = line.strip()
                    logging.debug('{}: {}, len={}'.format(index, label, len(text)))
                    data.append([text, label, index])
                    #self.text_number_to_label[index] = doc
        return data

    #def load_docs_from_directory(self, data_path):
    #    """ Each file in data_path is a whole document.
    #    """
    #    doc_labels = [f for f in os.listdir(data_path)] #  if f.endswith('.txt')
    #    data = []
    #    self.text_number_to_label = {}
    #    for i, doc in enumerate(doc_labels):
    #        with open(data_path + '/' + doc, 'r') as f:
    #            text = f.read()
    #            text = text.replace('\n', ' ')
    #            logging.debug('{}: {}, len={}'.format(i, doc, len(text)))
    #            data.append(text)
    #            self.text_number_to_label[i] = doc
    #    return data

    def load_docs_from_file(self, filepath):
        """ A single file filepath contains all docs. It should have the following form:
                    label ::: word word word
        """
        df = pd.read_csv(filepath, sep=' ::: ', names=['label','text'], engine='python')
        data = df.text
        self.text_number_to_label = {i:l for i,l in enumerate(list(df.label))}
        return data


    def __getitem__(self, index):
        if index == "train":
            return self.data['train']
        elif index == "val":
            return self.data['val']
        else:
            logging.warning("Bad index {}".format(index))
            return None


def train_model(model, dataset):

    # Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.
    #train_data = [
    #    ["Example sentence belonging to class 1", 1],
    #    ["Example sentence belonging to class 0", 0],
    #    ["Example eval senntence belonging to class 2", 2],
    #]
    train_df = pd.DataFrame(dataset['train'])

    #eval_data = [
    #    ["Example eval sentence belonging to class 1", 1],
    #    ["Example eval sentence belonging to class 0", 0],
    #    ["Example eval senntence belonging to class 2", 2],
    #]
    eval_df = pd.DataFrame(dataset['val'])

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print("result:", result)
    print("model_outputs:", model_outputs)
    print("wrong_predictions:", wrong_predictions)
    return result, model_outputs, wrong_predictions

if __name__ == "__main__":

    dataset = ActionDataset(data_path="./dataset_action")
    print("train dataset size:", len(dataset['train']))
    print("val dataset size:", len(dataset['val']))
    print("index_to_label:", dataset.index_to_label)
    print("Example of train data:")
    print(dataset['train'][:10])

    num_labels = dataset.index_to_label
    print("num_labels:", num_labels)

    # Create a ClassificationModel
    model = ClassificationModel(
        "bert", "bert-base-cased", 
        num_labels=num_labels,
        args={"reprocess_input_data": True, "overwrite_output_dir": True},
        use_cuda=False
    )
    train_model(model, dataset)



predictions, raw_outputs = model.predict(["Some arbitary sentence"])
