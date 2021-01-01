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
        "Click on Next step",
    ],
    [
        "Navigate to the next page by clicking on Go exit",
        "Click on Go exit",
    ],
    [
        "Navigate to the next website by clicking on Exit",
        "Click on Exit",
    ],    
    [
        "Login to the application by clicking on abc",
        "Click on abc",
    ],
    [
        "Change web page by doing task1",
        "Do task1",
    ],
    [
        "Find out what happened by typing Search",
        "Type Search",
    ],
    [
        "Buy this product by entering full info",
        "Enter full info",
    ],    
    [
        "Login to the website by clicking on Sigh",
        "Click on Sigh",
    ],    
    [
        "Login to website by clicking on qwerty",
        "Click on qwerty",
    ],
    [
        "Click at something in login window",
        "Click at something",
    ],
    [
        "Click on OK in login window",
        "Click on OK",
    ],
    [
        "Click at OKAY in home page",
        "Click at OKAY",
    ],    
    [
        "Click on login on left side of the screen",
        "Click on login",
    ],
    [
        "Click on username on left side of the display",
        "Click on username",
    ],
]

train_df = pd.DataFrame(
    train_data, columns=["input_text", "target_text"]
)

eval_data = [
    [
        "Navigate to the next application by clicking on Quit",
        "Click on Quit",
    ],
    [
        "Click at login on the right side of the screen",
        "Click at login",
    ],
    [
        "Login to this website by entering username and password",
        "Enter username and password",
    ],
    [
        "Log out from the website by pressing OK button",
        "Press OK button",
    ],
]

eval_df = pd.DataFrame(
    eval_data, columns=["input_text", "target_text"]
)

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    #encoder_decoder_name="facebook/bart-large",
    encoder_decoder_name="facebook/bart-base",
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
            "Click on login on the right side of the screen",
            "Login to this website by entering username and password",
            "Log out from the website by pressing OK button"
        ]
    )
print(results)
for i, result in enumerate(results):
    print("{}: {}".format(i, result))
"""
["{'label': 'anything is', 'action': 'clickables', 'element_type", 
"{'label': 'action': 'clickables', 'element_type': None}"]
"""