'''
    Developed by
    Nobal Niraula, Boeing Research & Technology
    Daniel Whyatt, Boeing Research & Technology
    Hai Nguyen, Boeing Enterprise Safety Data Analytics
'''

from sklearn.model_selection import train_test_split
import pandas as pd
from os import path
import json

def prepare_datasets():
    if "yes" == input("Overriding data sets? Type 'yes' to confirm"):
        work_dir = r".\experiments\depressurization\data_sets"
        annotated_file = r".\experiments\depressurization\SDR_Depressurization_Gold_label_v3.csv"
        df = pd.read_csv(annotated_file)
        print("Total Model")
        print(df.Label.value_counts())

        boeing_annotated = df[~df.Year.notnull()]
        faa_annotated = df[df.Year.notnull()]
        print("\nBoeing Annotated:", boeing_annotated.shape)
        print(boeing_annotated.Label.value_counts())
        print("\nFAA Annotated:", faa_annotated.shape)
        print(faa_annotated.Label.value_counts())
        train, test = train_test_split(df, test_size=0.3)
        faa_annotated.to_csv(path.join(work_dir, "faa_annotated.csv"))
        boeing_annotated.to_csv(path.join(work_dir, "boeing_annotated.csv"))
        train.to_csv(path.join(work_dir, "train-0.7.csv"))
        test.to_csv(path.join(work_dir, "test-0.3.csv"))
    else:
        print("Nothing is done")


def convert2jsonl():
    train_file = r".\experiments\depressurization\data_sets\train-0.7.csv"
    test_file = r".\experiments\depressurization\data_sets\test-0.3.csv"
    train_jsonl = r".\experiments\depressurization\data_sets\train-0.7.jsonl"
    test_jsonl = r".\experiments\depressurization\data_sets\test-0.3.jsonl"

    with open(train_jsonl,"w") as fo:
        df = pd.read_csv(train_file)
        for i, row in df.iterrows():
            d = row.to_dict()
            x = {}
            x["prompt"] = d["Text"].lower().replace(r"\n"," ") + "\n\nEvent Type:"
            x["completion"] = " " + ("Depressurization" if d["Label"] == "Yes" else "None") + "\nEND"
            fo.write(f"{json.dumps(x)}\n")

    with open(test_jsonl,"w") as fo:
        df = pd.read_csv(test_file)
        for i, row in df.iterrows():
            d = row.to_dict()
            x = {}
            x["prompt"] = d["Text"].lower().replace(r"\n"," ") + "\n\nEvent Type:"
            x["completion"] = " " + ("Depressurization" if d["Label"] == "Yes" else "None") + "\nEND"
            fo.write(f"{json.dumps(x)}\n")

if __name__ == "__main__":
    pass
    # convert2jsonl()
    # prepare_datasets()
    # input("wait here")