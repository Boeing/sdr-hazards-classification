'''
    Boeing Proprietary.
    Developed by Nobal Niraula, Boeing Research & Technology
    Developed by Hai Nguyen, Enterprise Safety
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from os import path
import pickle

# Default column name  in DataFrame
from sklearn.svm import SVC

from prep_utils import PreprocessingUtils
from vectorizers import Vectorizers


prep_util = PreprocessingUtils()


class TextClassifier:

    @staticmethod
    def df2text_sdr(in_df, text_field):
        data = [r[text_field] for i, r in in_df.iterrows()]
        return data

    @staticmethod
    def predict_sdr_labels(model, model_config, records):
        prep_records = prep_util.preprocess_records(model_config, records)
        vectorizers = model_config["vectorizers"]
        model_name = model_config["model_name"]
        records_transformed = Vectorizers.transform_with_vectorizers(prep_records, vectorizers)
        predictions = model.predict(records_transformed)
        if model_name == "lg":
            predictions_probabilities = model.predict_proba(records_transformed)[:, 1]
        elif model_name == "xgb":
            predictions_probabilities = predictions
            predictions = [round(value) for value in predictions_probabilities]
        else:
            predictions_probabilities = None
        return predictions, predictions_probabilities

    @staticmethod
    def train_text_classifier(criteria, all_df , text_field, label_field, model_type, stop_words=[], over_sample=1, holdout_testsize=.1):
        print("Criteria Name: ", criteria)
        evaluate_model = False if holdout_testsize <= 0 else True
        test_data, test_labels = None, None

        le = LabelEncoder()
        le.fit(all_df[label_field].unique())
        label_dict = dict(zip(le.transform(le.classes_), le.classes_))
        # print("Classes Distribution:", all_df.groupby(label_field).size())
        # print(f"number of positive examples:{len(positive_df)}")
        # print(f"number of negative examples:{len(negative_df)}")
        print(f"Training {model_type} classifier ...")
        all_data = TextClassifier.df2text_sdr(all_df, text_field)
        # all_df = pd.concat([positive_df, negative_df])

        if evaluate_model:
            print(f"Evaluation is enabled with holdout test size {holdout_testsize}.")
            train_df, test_df, y_train, y_test = train_test_split(all_df, le.transform(all_df[label_field]), test_size=0.1, stratify=all_df[label_field], random_state=42)
            print("Train size:")
            print(len(train_df))
            print("Test size:")
            print(len(test_df))
            test_data = [x for x in TextClassifier.df2text_sdr(test_df, text_field)]
            # test_labels = [1 if int(x) == 1 else 0 for x in test_df[label_field]]
            # test_labels = [label_dict[x] for x in test_df[label_field]]
            test_labels = y_test
            final_df = train_df
        else:
            final_df = all_df

        train_data = TextClassifier.df2text_sdr(final_df, text_field)
        # train_labels = [1 if int(x) == 1 else 0 for x in final_df["label"]]
        # train_labels = [label_dict[x] for x in final_df[label_field]]
        train_labels =  le.transform(final_df[label_field])

        model_config = {}
        model_config["feature"] = ["words", "characters"]
        model_config["model_name"] = model_type
        model_config["model_info"] =  f"{model_config['model_name']}; {'; '.join(model_config['feature'])}"
        model_config["remove_general_stop_words"] = True
        model_config["custom_stopwords"] = stop_words
        model_config["lower_case"] = True
        model_config["remove_punctuations"] = True
        model_config["remove_duplicate_sentences"] = False
        model_config["use_mad_preprocessing"] = False
        model_config["feature"] = ["words", "characters"]
        model_config["label_dict"] = label_dict

        train_data = prep_util.preprocess_records(model_config, train_data)

        vectorizers = Vectorizers.fit_vectorizers(all_data, options=model_config["feature"])
        X_train = Vectorizers.transform_with_vectorizers(train_data, vectorizers)
        model_config["vectorizers"] = vectorizers

        model = None
        if model_config["model_name"] == "lg":
            model = LogisticRegression(solver='liblinear', C=1e2, max_iter=200)
        elif model_config["model_name"] == "svm":
            model = SVC(kernel='linear', C=1.0)
        elif model_config["model_name"] == "xgb":
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=500, max_depth=5, reg_alpha=.1)
        else:
            raise ValueError("Model not supported")

        # fit model
        model.fit(X_train, train_labels)

        if evaluate_model:
            # target_names = [criteria, 'Non-'+criteria]
            target_name = le.inverse_transform(model.classes_)
            predictions = TextClassifier.predict_sdr_labels(model, model_config, test_data)
            print(classification_report(test_labels, predictions[0], target_names=target_name))
            return model, model_config, test_data, test_labels

        return model, model_config

    @staticmethod
    def export_model(model, model_config, model_name, out_folder):
        print("Exporting model to given folder: ", out_folder)
        model_config_path = path.join(out_folder, model_name + ".config")
        model_path= path.join(out_folder, model_name + ".model")
        pickle.dump(model_config, open(model_config_path, 'wb'))
        pickle.dump(model, open(model_path, 'wb'))


    @staticmethod
    def get_classifier(criteria, df_all, text_field, label_field, label_dict, model_type="lg", oversampling=1, preprocessor=None, holdout_testsize=0.1, stop_words=[]):
        info = TextClassifier.train_text_classifier(criteria, df_all, text_field, label_field, model_type=model_type, holdout_testsize=holdout_testsize,
                                                    stop_words=stop_words, over_sample=oversampling)
        if holdout_testsize > 0:
            model, model_config, test_data, test_labels = info
            return model, model_config, test_data, test_labels
        else:
            model, model_config = info
            return model, model_config

    @staticmethod
    def evaluate_model(model, model_config, test_df, text_field, label_field, out_csv_path=None ):
        test_data = [x for x in TextClassifier.df2text_sdr(test_df, text_field)]
        predictions, probabilities = TextClassifier.predict_sdr_labels(model, model_config, test_data)
        gold_labels = test_df[label_field]
        label_dict_rev = {y:x for x,y in model_config["label_dict"].items()}
        prediction_labels = [label_dict_rev[x] for x in predictions]
        if out_csv_path:
            test_df["Pred_Label"] = predictions
            test_df["Pred_Probabs"] = probabilities
            df_1 = test_df[["Label", "Pred_Label", "Pred_Probabs", "Text"]]
            df_1.to_csv(out_csv_path)

        print(classification_report(gold_labels, prediction_labels))


    @staticmethod
    def release_model(event_type, training_path, target_name, model_type= "lg"):
        # event_type == "depressurization":
        text_field = "Text"
        label_field = "Label"
        criteria= event_type
        model_type =model_type

        # label_dict = {"Yes": 1, "No": 0}
        label_dict = target_name

        # Training example prior to SME feedback
        # work_dir = r".\experiments\depressurization\data_sets"
        # df_train_prep = pd.read_csv(path.join(work_dir, "train-0.7_prep.csv"))
        # df_test_prep = pd.read_csv(path.join(work_dir, "test-0.3_prep.csv"))
        # df_all = pd.concat([df_train_prep, df_test_prep])
        # text_field = "Text_punct_prep"

        # Traininig examples after SME feedback
        # df = pd.read_csv(r".\depressurization\SDR_Depressurization_Gold_label_with_SME_Feedback.csv")
        df = pd.read_csv(training_path, encoding='latin')
        df_all = pd.DataFrame({text_field:df["Text"], label_field:df["Label"].str.strip()})
        combine_maybe_to_yes = True
        print("Classes Distribution:")
        print(df_all[label_field].value_counts())
        if combine_maybe_to_yes:
            df_all[label_field].loc[df_all[label_field] == 'Maybe'] = "Yes"
            # df_maybe = df_all[df_all[label_field] == "Maybe"]
            # df_pos = pd.concat([df_pos, df_maybe])
            # df_pos[label_field] = "Yes"
        print("------------------")
        print(f"Training a model using all {event_type} records.")
        model, model_config, _, _ = TextClassifier.get_classifier(criteria, df_all, text_field, label_field,
                                                                  label_dict, model_type=model_type)
        # test_df_maybe = df_all[df_all[label_field] == "Maybe"]
        # test_df_maybe[label_field] = "Yes"
        # test_df = test_df_maybe
        # test_data = [x for x in TextClassifier.df2text_sdr(test_df, text_field)]
        # test_labels = [1 if x else 0 for x in test_df[label_field] == "Yes"]
        # predictions, probabilities = TextClassifier.predict_sdr_labels(model, model_config, test_data)
        # print(classification_report(test_labels, predictions, target_names=target_name))

        # test_df = pd.concat([df_pos, df_neg])
        # test_data = [x for x in TextClassifier.df2text_sdr(test_df, text_field)]
        # test_labels = [1 if x else 0 for x in test_df[label_field] == "Yes"]
        # predictions, probabilities = TextClassifier.predict_sdr_labels(model, model_config, test_data)
        # print(classification_report(test_labels, predictions, target_names=target_name))

        return model, model_config


if __name__ == "__main__":
    #### Model Config Path ########################################
    event_type = "degraded_controllability"
    target_name = ["degraded-controllability", "non-degraded"]
    file_path = "./data/Degraded_Controllability.csv"
    # event_type = "corrosion_limit"
    # target_name = ["beyond-limit", "no-corrosion", "no-limit", "within-limit", "within-beyond-limit"]
    model, model_config = TextClassifier.release_model(event_type,file_path, target_name, model_type='xgb')
    model_dump_path = r"./model/"
    TextClassifier.export_model(model, model_config, "sdr-degraded-controllability", model_dump_path)
    # TextClassifier.export_model(model, model_config, "sdr-degraded-controllability", model_dump_path)
    #### Model Config Path ########################################
    exit("Finish Training")


    input("wait here..")
    work_dir = r".\experiments\depressurization\data_sets"
    data = {
        "boeing_annotated": "boeing_annotated.csv",
        "faa_annotated": "faa_annotated.csv",
        "train_seventy":"train-0.7.csv",
        "test_thirty": "test-0.3.csv",
        "train_seventy_prep":"train-0.7_prep.csv",
        "test_thirty_prep": "test-0.3_prep.csv",
        }
    text_field = "Text"
    label_field = "Label"
    criteria = "depressurization"

    print("Printing Stats: ")
    for name,file in data.items():
       data_path = path.join(work_dir, file)
       df = pd.read_csv(data_path)
       print("Data: ", name, "-", len(df))
       print(df.Label.value_counts())

    df_faa = pd.read_csv(path.join(work_dir, data["faa_annotated"]))
    df_boeing = pd.read_csv(path.join(work_dir, data["boeing_annotated"]))
    df_train = pd.read_csv(path.join(work_dir, data["train_seventy"]))
    df_test = pd.read_csv(path.join(work_dir, data["test_thirty"]))

    df_train_prep = pd.read_csv(path.join(work_dir, data["train_seventy_prep"]))
    df_test_prep = pd.read_csv(path.join(work_dir, data["test_thirty_prep"]))


    # Case 1: Train on FAA labeled and test on Boeing labeled
    print("------------------")
    print("Train on FAA labeled and test on Boeing labeled")
    model, model_config = TextClassifier.get_classifier(criteria, df_faa, text_field, label_field, holdout_testsize=0)
    TextClassifier.evaluate_model(model, model_config, df_boeing, text_field, label_field)

    # Case 2: Train on Boeing labeled and test on FAA labeled
    print("------------------")
    print("Train on Boeing labeled and test on FAA labeled")
    model, model_config = TextClassifier.get_classifier(criteria, df_boeing, text_field, label_field, holdout_testsize=0)
    TextClassifier.evaluate_model(model, model_config, df_faa, text_field, label_field)

    # Case 3: Train on Train and test on Test
    print("------------------")
    print("Train on 70% and testing on 30%")
    model, model_config = TextClassifier.get_classifier(criteria, df_train, text_field, label_field, holdout_testsize=0)
    TextClassifier.evaluate_model(model, model_config, df_test, text_field, label_field)

    # Case 4: Train on PREP Train and test on Test
    print("------------------")
    print("Train on 70% and testing on 30% using preprocessed text by MAD")
    text_field = "Text_mad_prep"
    model, model_config = TextClassifier.get_classifier(criteria, df_train_prep, text_field, label_field, holdout_testsize=0)
    TextClassifier.evaluate_model(model, model_config, df_test_prep, text_field, label_field)

    # Case 4: Train on PREP Train and test on Test
    print("------------------")
    print("Train on 70% and testing on 30% using preprocessed text by Removing punctuations")
    text_field = "Text_punct_prep"
    model, model_config = TextClassifier.get_classifier(criteria, df_train_prep, text_field, label_field, holdout_testsize=0)
    TextClassifier.evaluate_model(model, model_config, df_test_prep, text_field, label_field)