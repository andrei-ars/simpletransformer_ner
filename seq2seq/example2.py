# code from https://simpletransformers.ai/docs/seq2seq-specifics/

import logging

import pandas as pd
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = [
    [
        "click photo",
        "{'label': 'photo', 'action': 'clickables', 'element_type': None}",
    ],
    [
        "click last row",
        "{'label': 'row', 'action': 'clickables', 'element_type': None}",
    ],
    [
        "click the third user photo.",
        "{'label': 'the third user photo .', 'action': 'clickables', 'element_type': None}",
    ],
    [
        "click anything for login",
        "{'label': 'anything', 'action': 'clickables', 'element_type': None}",
    ],
    [
        "click anything before login.",
        "{'label': 'anything', 'action': 'clickables', 'element_type': None}",
    ],
    [
        "click anything1 after login1.",
        "{'label': 'anything1', 'action': 'clickables', 'element_type': None}",
    ],

    [
        "click anything2 next to login2",
        "{'label': 'anything2 login2', 'action': 'clickables', 'element_type': None}",
    ],
    [
        "click anything where age is greater than 10",
        "{'action': 'clickables', 'query': 'WHERE \"AGE IS\" > \"10\"', 'column_data_type': 'number', 'header_in_query': 'age is', 'header_to_do_action': '', 'value_in_query': '10', 'table_xpath': '', 'element_type': None}",
    ],
    [
        "click hello where value is greater than 33",
        "{'action': 'clickables', 'query': 'WHERE \"VALUE IS\" > \"33\"', 'column_data_type': 'number', 'header_in_query': 'value is', 'header_to_do_action': '', 'value_in_query': '33', 'table_xpath': '', 'element_type': None}",
    ],
    [
        "click world where hight is greater than 0",
        "{'action': 'clickables', 'query': 'WHERE \"HIGHT IS\" > \"0\"', 'column_data_type': 'number', 'header_in_query': 'hight is', 'header_to_do_action': '', 'value_in_query': '0', 'table_xpath': '', 'element_type': None}",
    ],    
]

train_df = pd.DataFrame(
    train_data, columns=["input_text", "target_text"]
)

eval_data = [
    [
        "click house where hight is greater than 15.",
        "",
    ],
    [
        "click name after userlogin.",
        "",
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
print(
    model.predict(
        [
            "click something where width is greater than 99.",
            "click name after userlogin."
        ]
    )
)