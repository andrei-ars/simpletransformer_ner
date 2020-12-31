# code from https://simpletransformers.ai/docs/seq2seq-specifics/

import os
import logging

import pandas as pd
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)


os.system("rm -rf outputs/")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = [
    [
        "Navigate to leads page by clicking on Next step",
        "click on Next step",
    ],
    [
        "Navigate to the next page by clicking on Go exit",
        "click on Go exit",
    ],
    [
        "Navigate to the next website by clicking on Exit",
        "click on Exit",
    ],    
    [
        "Login to the application by clicking on abc",
        "click on abc",
    ],
    [
        "Login to the website by clicking on Sigh",
        "click on Sigh",
    ],    
    [
        "Login to website by clicking on qwerty",
        "click on qwerty",
    ],
    [
        "Click on something in login window",
        "Click on something",
    ],
    [
        "Click on OK in login window",
        "Click on OK",
    ],
    [
        "Click on OKAY in home page",
        "Click on OKAY",
    ],    
    [
        "click on login on left side of the screen",
        "click on login",
    ],
    [
        "click on username on left side of the display",
        "click on username",
    ],
]

train_df = pd.DataFrame(
    train_data, columns=["input_text", "target_text"]
)

eval_data = [
    [
        "Navigate to the next application by clicking on Quit",
        "click on Quit",
    ],
    [
        "click on login on the right side of the screen",
        "click on login",
    ],
]

eval_df = pd.DataFrame(
    eval_data, columns=["input_text", "target_text"]
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
            "Navigate to the next application by clicking on Quit",
            "click on login on the right side of the screen"
        ]
    )
print(results)
for i, result in enumerate(results):
    print("{}: {}".format(i, result))
"""
["{'label': 'anything is', 'action': 'clickables', 'element_type", 
"{'label': 'action': 'clickables', 'element_type': None}"]
"""