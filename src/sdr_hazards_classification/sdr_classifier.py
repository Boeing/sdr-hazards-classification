'''
    Boeing Proprietary.
    Developed by Nobal Niraula, Boeing Research & Technology
    Developed by Hai Nguyen, Enterprise Safety
'''
from dataclasses import dataclass, field

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import logging
from os import path
import pickle
import traceback as tb
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from prep_utils import PreprocessingUtils, DEPRESSURIZATION, DEGRADED_CONTROLLABILITY, CORROSION_LIMIT, FIRE, RTO, PDA, \
    ENGINE, FUEL, RUNWAY_EXCURSION, FLIGHT_CREW, EMERGENCY_EQUIPMENT, STRUCTURE, VIBRATION, GENERAL_EQUIPMENT
from vectorizers import Vectorizers

prep_util = PreprocessingUtils()

@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    """
    event_type: str = field(
        default=DEPRESSURIZATION, metadata={"help": "The hazard type that will be classified"},
    )
    target_name: dict = field(default=False, metadata={"help": "The categories of the hazard event"})
    file_path: str = field(default=False, metadata={"help": "File path of training data"})

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
        print(f"Training {model_type} classifier ...")
        all_data = TextClassifier.df2text_sdr(all_df, text_field)
        # all_df = pd.concat([positive_df, negative_df])

        if evaluate_model:
            print(f"Evaluation is enabled with holdout test size {holdout_testsize}.")
            logging.info(msg=f'Evaluation is enabled with holdout test size {holdout_testsize}')
            train_df, test_df, y_train, y_test = train_test_split(all_df, le.transform(all_df[label_field]), test_size=0.1, stratify=all_df[label_field], random_state=42)
            print("Train size:")
            print(len(train_df))
            print("Test size:")
            print(len(test_df))
            test_data = [x for x in TextClassifier.df2text_sdr(test_df, text_field)]
            test_labels = y_test
            final_df = train_df
        else:
            final_df = all_df

        train_data = TextClassifier.df2text_sdr(final_df, text_field)
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

        sklearn_model = None
        if model_config["model_name"] == "lg":
            sklearn_model = LogisticRegression(solver='liblinear', C=1e2, max_iter=200)
        elif model_config["model_name"] == "svm":
            sklearn_model = SVC(kernel='linear', C=1.0)
        elif model_config["model_name"] == "xgb":
            from xgboost import XGBClassifier
            sklearn_model = XGBClassifier(n_estimators=1000, max_depth=10, reg_alpha=.1)
        else:
            model_name = model_config["model_name"]
            raise ValueError(f"Model '{model_name}' not supported")

        # fit model
        print(f"Let's do CV for {type(sklearn_model).__name__}")
        scores = cross_val_score(sklearn_model, X_train, train_labels, scoring= 'accuracy', cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        sklearn_model.fit(X_train, train_labels)

        if evaluate_model:
            # target_names = [criteria, 'Non-'+criteria]
            target_name = le.inverse_transform(sklearn_model.classes_)
            predictions = TextClassifier.predict_sdr_labels(sklearn_model, model_config, test_data)
            print(classification_report(test_labels, predictions[0], target_names=target_name))
            logging.info(msg=f'Performance:\n {classification_report(test_labels, predictions[0], target_names=target_name)}')
            return sklearn_model, model_config, test_data, test_labels

        return sklearn_model, model_config

    @staticmethod
    def export_model(model, model_config, model_name, out_folder):
        print("Exporting model to given folder: ", out_folder)
        logging.info(msg=f'Exporting {model_name} to given folder {out_folder}')
        model_config_path = path.join(out_folder, model_name + ".config")
        model_path= path.join(out_folder, model_name + ".model")
        pickle.dump(model_config, open(model_config_path, 'wb'))
        pickle.dump(model, open(model_path, 'wb'))
        logging.info(msg=f'Complet with {model_name} !\n')

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
    def release_model( **kwargs):
        try:
            # event_type == "depressurization":
            text_field = "Text"
            label_field = "Label"
            criteria= kwargs['event_type']
            model_type =kwargs['model_type']

            # label_dict = {"Yes": 1, "No": 0}
            label_dict = kwargs['target_name']

            # Traininig examples after SME feedback
            df = pd.read_csv(kwargs['training_path'], encoding='latin')
            df_all = pd.DataFrame({text_field:df["Text"], label_field:df["Label"].str.strip()})
            combine_maybe_to_yes = True
            print("Classes Distribution:")
            print(df_all[label_field].value_counts())
            logging.info(msg=f'Classes Distribution\n: {df_all[label_field].value_counts()}')
            if combine_maybe_to_yes:
                df_all[label_field].loc[df_all[label_field] == 'Maybe'] = "Yes"
            print("------------------")
            print(f"Training a model using all {criteria} records.")
            logging.info(msg=f"Training a model using all {criteria} records.")

            if df_all.Label.value_counts().count() < 2:
                df_non_hazards = pd.read_csv(kwargs['non_hazards_path'], encoding='latin')
                df_all = pd.concat([df_all, df_non_hazards])

            model, model_config, _, _ = TextClassifier.get_classifier(criteria, df_all, text_field, label_field,
                                                                      label_dict, model_type=model_type)

        except Exception as err:
            error_message = ''.join(tb.format_exception(None, err, err.__traceback__))
            logging.error(msg=error_message)
            print(error_message)
            exit(f"Critical Error encounter")

        return model, model_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train SDR classifier.')
    parser.add_argument('--event_type', '-e', metavar='N', type=str, default= "depressurization",
                        help='aviation hazard type (i.e.: "depressurization", "degraded-controllability", "corrosion-limit")')
    parser.add_argument('--target_name', '-t', metavar='N', type=str, default= "depressurization; non-depressurization",
                        help='target names for the hazard separated by semicolon (i.e.: "depressurization; non-depressurization")')
    parser.add_argument('--model_type', '-m', metavar='N', type=str, default= "lg",
                        help='modeling method (i.e.: "lg", "xgb", "svm")')
    parser.add_argument('--file_path', '-f', metavar='N', type=str, default= "./data/Depressurization.csv",
                        help='path to training dataset (i.e.: "./data/Depressurization.csv")')

    args = parser.parse_args()
    logging.info("Start logging for sdr-hazards-classification")

    if args.event_type in DEPRESSURIZATION:
        training_args = {'event_type': DEPRESSURIZATION,
                    'model_type': args.model_type,
                   'target_name': [DEPRESSURIZATION, "non-" + DEPRESSURIZATION],
                   'training_path': "./data/Depressurization.csv"}
    elif args.event_type in ENGINE:
        training_args = {'event_type': ENGINE,
                    'model_type': args.model_type,
                   'target_name': [ENGINE, "non-" + ENGINE],
                   'training_path': "./data/Engine.csv"}
    elif args.event_type in STRUCTURE:
        training_args = {'event_type': STRUCTURE,
                    'model_type': args.model_type,
                   'target_name': [STRUCTURE, "non-" + STRUCTURE],
                   'training_path': "./data/Structure.csv"}
    elif args.event_type in VIBRATION:
        training_args = {'event_type': VIBRATION,
                    'model_type': args.model_type,
                   'target_name': [VIBRATION, "non-" + VIBRATION],
                   'training_path': "./data/Abnormal_Vibration.csv"}
    elif args.event_type in GENERAL_EQUIPMENT:
        training_args = {'event_type': GENERAL_EQUIPMENT,
                    'model_type': args.model_type,
                   'target_name': [GENERAL_EQUIPMENT, "non-" + GENERAL_EQUIPMENT],
                   'training_path': "./data/General_Equipment.csv"}
    elif args.event_type in FUEL:
        training_args = {'event_type': FUEL,
                    'model_type': args.model_type,
                   'target_name': [FUEL, "non-" + FUEL],
                   'training_path': "./data/Fuel.csv"}
    elif args.event_type in RUNWAY_EXCURSION:
        training_args = {'event_type': RUNWAY_EXCURSION,
                    'model_type': args.model_type,
                   'target_name': [RUNWAY_EXCURSION, "non-" + RUNWAY_EXCURSION],
                   'training_path': "./data/Runway_Excursion.csv"}
    elif args.event_type in FLIGHT_CREW:
        training_args = {'event_type': FLIGHT_CREW,
                    'model_type': args.model_type,
                   'target_name': [FLIGHT_CREW, "non-" + FLIGHT_CREW],
                   'training_path': "./data/Flight_Crew.csv"}
    elif args.event_type in EMERGENCY_EQUIPMENT:
        training_args = {'event_type': EMERGENCY_EQUIPMENT,
                    'model_type': args.model_type,
                   'target_name': [EMERGENCY_EQUIPMENT, "non-" + EMERGENCY_EQUIPMENT],
                   'training_path': "./data/Emergency_Equipment.csv"}
    elif args.event_type in FIRE:
        training_args = {'event_type': FIRE,
                    'model_type': args.model_type,
                   'target_name': [FIRE, "non-" + FIRE],
                   'training_path': "./data/Fire.csv"}
    elif args.event_type in PDA:
        training_args = {'event_type': PDA,
                    'model_type': args.model_type,
                   'target_name': [PDA, "non-" + PDA],
                   'training_path': "./data/Parts_Departing_Aircraft.csv"}
    elif args.event_type in RTO:
        training_args = {'event_type': RTO,
                    'model_type': args.model_type,
                   'target_name': [RTO, "non-" + RTO],
                   'training_path': "./data/Reject_To_Takeoff.csv"}
    elif args.event_type in DEGRADED_CONTROLLABILITY:
        training_args = {'event_type': DEGRADED_CONTROLLABILITY,
                    'model_type': args.model_type,
                   'target_name': [DEGRADED_CONTROLLABILITY, "non-" + DEGRADED_CONTROLLABILITY],
                   'training_path': "./data/Degraded_Controllability.csv"}
    elif args.event_type in CORROSION_LIMIT:
        training_args = {
            'event_type': CORROSION_LIMIT,
            'model_type': args.model_type,
            'target_name': ["beyond-limit", "no-corrosion", "no-limit", "within-limit", "within-beyond-limit"],
            'training_path': "./data/Corrosion_Limit.csv"}
    else:
        training_args = {'event_type': args.event_type,
                    'model_type': args.model_type,
                   'target_name': args.target_name.split(";"),
                   'training_path': args.file_path}

    #### Model Config Path ########################################
    training_args['non_hazards_path'] = './data/Non_Hazards.csv'
    logging.info(msg=f'Start training for {training_args}')

    model, model_config = TextClassifier.release_model(**training_args)

    model_dump_path = r"./model/"
    model_name = "".join(['sdr-', training_args['event_type']])
    TextClassifier.export_model(model, model_config, model_name, model_dump_path)
    #### Model Config Path ########################################
    exit("Finish Training")
