import numpy as np
import pandas as pd

from nermodel import NerModel
from nerdataset import NerPartDataset, NerDataset
from nerdataset import get_labels_list


if __name__ == "__main__":

    mode = "train"
    #mode = "infer"

    modelname = "nlp_complex"
    complex_dataset_names = ["table", "table_nq", "nlp_ext", "nlp_ext_nq"]

    test_sentences = [
        "Double Click on a BOQ calendar EOQ from the list on the left side of the screen.",
        "Enter text into the BOQ password EOQ on the bottom left of the screen",
        "Double Click on a calendar from the list on the left side of the screen.",
        "Enter text into the password on the bottom left of the screen",
        "Enter text into the second name box on the bottom left of the screen",
        "click on Find an Agent after Renters Insurance",
        "Hover over login after forgot your password?",
        "click on Get a Quote next to Motor Home Insurance",
        "Click on Lifestyle next to Bihar Election 2020",
        "click on Renters Insurance before Find an Agent",
        "click on Images next to Maps",
        "click on Yes radio for Are you Hispanic or Latino?",
        "click on Letters where Numbers greater than 3",
        "Click on Manage External User button for Contact Detail",
        "Click on Black Where ends with c"
        ]

    if mode == "train":
        dataset = NerDataset(complex_dataset_names).as_dict()
    else:
        dataset = NerDataset(complex_dataset_names, labels_only=True).as_dict()
        #dataset['labels_list'] = get_labels_list("dataset/{}/tag.dict".format(modelname))

    if mode == "train":
        model = NerModel(modelname=modelname, dataset=dataset)
        model.train()
        model.eval()

    if mode in {"infer"}:
        model = NerModel(modelname=modelname, dataset=dataset, use_saved_model=True)

    if mode in {"train", "infer"}:
        result = model.predict(test_sentences)
        print("result:", result)

    print("\nManually input")
    while True:
        input_text = input("Input text: ")
        if input_text == 'q':
            break
        if "|" in input_text:
            input_text, input_data = input_text.split("|")

        result = model.predict([input_text])
        print("result:", result)