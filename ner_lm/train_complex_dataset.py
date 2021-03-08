import numpy as np
import pandas as pd

from nermodel import NerModel
from nerdataset import NerPartDataset, NerDataset
from nerdataset import get_labels_list
from ner_slot_filling import ner_slot_filling


if __name__ == "__main__":

    #mode = "train"
    #mode = "test"
    mode = "infer"

    modelname = "nlp_complex"
    #complex_dataset_names = ["table", "table_nq", "nlp_ext", "nlp_ext_nq"]
    complex_dataset_names = ["nlp_ext", "nlp_ext_nq"]

    """
    test_sentences = [
        "Double Click on a BOQ calendar EOQ from the list on the left side of the screen.",
        "Enter text into the BOQ password EOQ on the bottom left of the screen",
        "Double Click on a calendar from the list on the left side of the screen.",
        "Enter text into the password on the bottom left of the screen",
        "Enter text into the second name box on the bottom left of the screen",
        "Enter text into the Input name on the bottom left of the screen",
        "click on Find an Agent after Renters Insurance",
        "Hover over login after forgot your password?",
        "click on Get a Quote next to Motor Home Insurance",
        "Click on Lifestyle next to Bihar Election 2020",
        "click on Renters Insurance before Find an Agent",
        "click on Images next to Maps",
        "click on Simple Images next to Hello Maps",
        "click on Yes radio for Are you Hispanic or Latino?",
        "click on Letters where Numbers greater than 3",
        "Click on Manage External User button for Contact Detail",
        "Click on Black Where ends with c",
        "Navigate to leads page by clicking on Next step",
        "Change the page by clicking on Next button",
        "Extract information by clicking on Next button",
        ]
    """

    test_sentences = [
        "Double Click on a calendar from the list on the left side of the screen.",
        "Enter text into the password on the bottom left of the screen",
        "Enter text into the second name box on the bottom left of the screen",
        "Log out from the website by clicking on Log out from website",
        "Click on Basket in home window",
        "Click on Basket button in the left side of the screen",
        "Click on BOQ Basket EOQ button in the left side of the screen",

        "Double Click on a calendar from the list on the left side of the screen",
        "Click on calendar from the list on the left side of the screen",
        "Click on YYY from the list on the left side of the screen",
        "Double click on YYY aaa from the list on the left side of the screen",
        ]

    if mode == "train":
        dataset = NerDataset(complex_dataset_names).as_dict()
    else:
        dataset = NerDataset(complex_dataset_names).as_dict()
        #dataset = NerDataset(complex_dataset_names, labels_only=True).as_dict()
        #dataset['labels_list'] = get_labels_list("dataset/{}/tag.dict".format(modelname))

    if mode == "train":
        model = NerModel(modelname=modelname, dataset=dataset)
        model.train()
        model.eval()
    else:
        model = NerModel(modelname=modelname, dataset=dataset, use_saved_model=True)

    if mode in {"train", "test"}:
        print("\nMODEL.TEST:")
        model.test()
        #model.eval()
        #predictions = model.predict(test_sentences)
        #for i in range(len(predictions)):
        #    text = test_sentences[i]
        #    print("text: {}\noutput: {}\n".format(text, predictions[i]))

    if mode in {"infer"}:
        result = model.raw_predict(test_sentences)
        predictions = result['predictions']
        raw_outputs = result['raw_outputs']
        for i in range(len(predictions)):
            text = test_sentences[i]
            print("text: {}\noutput: {}".format(text, predictions[i]))
            slots = ner_slot_filling(text, predictions[i])
            print("slots: {}\n".format(slots))

    print("\nManually input")
    while True:
        input_text = input("Input text: ")
        if input_text == 'q':
            break
        if "|" in input_text:
            input_text, input_data = input_text.split("|")

        result = model.predict([input_text])
        print("result:", result)