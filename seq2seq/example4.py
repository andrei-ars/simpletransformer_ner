# code from https://simpletransformers.ai/docs/seq2seq-specifics/

import os
import logging
import random

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
        "Test this input field by typing ABC",
        "Type ABC",
    ],
    [
        "Test this audio box by saying Hello",
        "Say hello",
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
        "Login to the google mail by typing your password",
        "Type your password",
    ],  
    [
        "Buy this product by writing full info",
        "Write full info",
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
        "Hover on OK in login window",
        "Hover on OK",
    ],
    [
        "Click at OKAY in home page",
        "Click at OKAY",
    ],
    [
        "Select OKAY in the home page",
        "Select OKAY",
    ],
    [
        "Click on login on left side of the screen",
        "Click on login",
    ],
    [
        "Hover on username on left side of the display",
        "Hover on username",
    ],
]

random.shuffle(train_data)
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
        "Login to this website by typing username and password",
        "Type username and password",
    ],
    [
        "Log out by typing Exit",
        "Type Exit",
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
    encoder_decoder_name="facebook/bart-large",
    #encoder_decoder_name="facebook/bart-base",
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
input_texts = 
        [
            "Navigate to the next application by clicking on Quit",
            "Click at login on the right side of the screen",
            "Login to this website by typing username and password",
            "Log out by typing Exit",
            "Log out from the website by pressing OK button",
        ]
output_texts = model.predict(input_texts)
print(output_texts)
for i, pair in enumerate(zip(input_texts, output_texts)):
    print("{}: {} --> {}".format(i, pair[0], pair[1]))
"""
1) 
["{'label': 'anything is', 'action': 'clickables', 'element_type", 
"{'label': 'action': 'clickables', 'element_type': None}"]

2) bart-large
            "Navigate to the next application by clicking on Quit",
            "Click on login on the right side of the screen",
            "Login to this website by entering username and password",
            "Log out from the website by pressing OK button"
0: Click on
1: Click on login
2: Click to this website
3: Click OK button

3) bart-base:
0: Click on Quit
1: Click on login
2: Click on username and password
3: Click on OK button


            "Navigate to the next application by clicking on Quit",
            "Click on login on the right side of the screen",
            "Login to this website by entering username and password",
            "Log out by typing Exit",
            "Log out from the website by pressing OK button",
0: Click on Quit
1: Click on login
2: Click username and password
3: Type Exit
4: Click OK button

Login to this website by typing username and password
result: ['Click username and password']
Click at login on the right side of the screen
result: ['Click at login']
"""



print("\nManually input")
while True:
    input_text = input("Input text: ")
    if input_text == 'q':
        break
    if "|" in input_text:
        input_text, input_data = input_text.split("|")

    result = model.predict([input_text])
    print("result:", result)