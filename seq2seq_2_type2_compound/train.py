# code from https://simpletransformers.ai/docs/seq2seq-specifics/

import os
import sys
import logging
import pandas as pd


os.system("rm -rf outputs/")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = []
train_file = "./dataset/train.txt"
with open(train_file) as fp:
    for i, line in enumerate(fp):
        line = line.strip()
        if i % 3 == 0:
            input_text = line
        elif i % 3 == 1:
            target_text  = line
        elif i % 3 == 2:
            train_data.append([input_text, target_text])

"""
train_data = [
    [
        "click photo",
        "{'label': 'photo', 'action': 'clickables', 'element_type': None}",
    ],
]
"""

train_df = pd.DataFrame(
    train_data, columns=["input_text", "target_text"]
)
print(train_df)

eval_data = [
    [
        "click house after name",
        "{'label': 'house', 'action': 'clickables', 'element_type': None}",
    ],
    [
        "hover the user photo next to sim",
        "{'label': 'the user photo', 'action': 'hoverables', 'element_type': None}",
    ],
]

eval_df = pd.DataFrame(
    eval_data, columns=["input_text", "target_text"]
)


from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 20
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=False,
)


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum(
        [
            1 if label == pred else 0
            for label, pred in zip(labels, preds)
        ]
    )


# Train the model
model.train_model(
    train_df, eval_data=eval_df, matches=count_matches
)

# # Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction
print("\n Predictions:")
results = model.predict(
        [
            "Click on Task icon and click on Create Journal",
            "click name after userlogin and click password",
            "Enter name, password",
            "Enter userlogin and click submit",
            "Enter userlogin, click submit",
        ]
    )
print(results)
for i, result in enumerate(results):
    print("{}: {}".format(i, result))
"""
["{'label': 'anything is', 'action': 'clickables', 'element_type", 
"{'label': 'action': 'clickables', 'element_type': None}"]
"""