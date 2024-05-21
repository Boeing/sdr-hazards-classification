'''
    Developed by
    Nobal Niraula, Boeing Research & Technology
    Daniel Whyatt, Boeing Research & Technology
    Hai Nguyen, Boeing Enterprise Safety Data Analytics
'''

import pickle
import os

import numpy as np

from .prep_utils import PreprocessingUtils, DEPRESSURIZATION, DEGRADED_CONTROLLABILITY, CORROSION_LIMIT, FIRE, \
    PDA, RTO, ENGINE, FUEL, RUNWAY_EXCURSION, FLIGHT_CREW, EMERGENCY_EQUIPMENT, STRUCTURE, VIBRATION, GENERAL_EQUIPMENT
from .vectorizers import Vectorizers
import pandas as pd


class SdrInferenceAPI:
    def __init__(self, event_type="depressurization"):

        if event_type in DEGRADED_CONTROLLABILITY:
            model_type = f"sdr-{DEGRADED_CONTROLLABILITY}.model"
            model_config = f"sdr-{DEGRADED_CONTROLLABILITY}.config"
        elif event_type in CORROSION_LIMIT:
            model_type = f"sdr-{CORROSION_LIMIT}.model"
            model_config = f"sdr-{CORROSION_LIMIT}.config"
        elif event_type in DEPRESSURIZATION:
            model_type = f"sdr-{DEPRESSURIZATION}.model"
            model_config = f"sdr-{DEPRESSURIZATION}.config"
        elif event_type in FIRE:
            model_type = f"sdr-{FIRE}.model"
            model_config = f"sdr-{FIRE}.config"
        elif event_type in RTO:
            model_type = f"sdr-{RTO}.model"
            model_config = f"sdr-{RTO}.config"
        elif event_type in PDA:
            model_type = f"sdr-{PDA}.model"
            model_config = f"sdr-{PDA}.config"
        elif event_type in PDA:
            model_type = f"sdr-{PDA}.model"
            model_config = f"sdr-{PDA}.config"
        elif event_type in ENGINE:
            model_type = f"sdr-{ENGINE}.model"
            model_config = f"sdr-{ENGINE}.config"
        elif event_type in FUEL:
            model_type = f"sdr-{FUEL}.model"
            model_config = f"sdr-{FUEL}.config"
        elif event_type in RUNWAY_EXCURSION:
            model_type = f"sdr-{RUNWAY_EXCURSION}.model"
            model_config = f"sdr-{RUNWAY_EXCURSION}.config"
        elif event_type in FLIGHT_CREW:
            model_type = f"sdr-{FLIGHT_CREW}.model"
            model_config = f"sdr-{FLIGHT_CREW}.config"
        elif event_type in EMERGENCY_EQUIPMENT:
            model_type = f"sdr-{EMERGENCY_EQUIPMENT}.model"
            model_config = f"sdr-{EMERGENCY_EQUIPMENT}.config"
        elif event_type in STRUCTURE:
            model_type = f"sdr-{STRUCTURE}.model"
            model_config = f"sdr-{STRUCTURE}.config"
        elif event_type in VIBRATION:
            model_type = f"sdr-{VIBRATION}.model"
            model_config = f"sdr-{VIBRATION}.config"
        elif event_type in GENERAL_EQUIPMENT:
            model_type = f"sdr-{GENERAL_EQUIPMENT}.model"
            model_config = f"sdr-{GENERAL_EQUIPMENT}.config"
        else: assert "Event type not supported!"

        this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
        self.dir_path = this_dir
        model_path = os.path.join(self.dir_path, 'model', model_type)
        model_config_path = os.path.join(self.dir_path, 'model', model_config)
        self.model_path = model_path
        self.mode_config_path = model_config_path
        print(f"Loading the {event_type} model")

        # load the model from disk
        self.model = pickle.load(open(self.model_path, 'rb'))
        self.configs = pickle.load(open(self.mode_config_path, 'rb'))
        self.prep_util = PreprocessingUtils()

    def get_predictions(self, record_list):
        '''
        Function to invoke the prediction method on a model and return predictions and prediction probabilities
        :param record_list: A list of string containing the records
        :return:prediction, prediction_probabilities
        '''
        prep_records = self.prep_util.preprocess_records(self.configs, record_list)
        vectorizers = self.configs["vectorizers"]
        records_transformed = Vectorizers.transform_with_vectorizers(prep_records, vectorizers)
        predictions = self.model.predict(records_transformed)
        label_dict = self.configs["label_dict"]
        predictions = [label_dict[i] for i in predictions]
        z = self.model.predict_proba(records_transformed)
        logit = np.argmax(z, axis=1)
        predictions_probabilities = [x[i] for i,x in zip(logit,z)]
        return predictions, predictions_probabilities

    def test_sdr_depressurization_predictions(self):
        '''
        Function to test the prediction of Depressurizatio event
        :return: None
        '''
        sample_record = """
        RAPID DEPRESSURIZATION AT FL 390 PRESSURE CONTROLLER BLANKED. MASKS DEPLOYED. NO AUTO FAIL LIGHT. DIVERTED AND DECLAREDAN EMERGENCY.   
        R & R THE CABIN PRESSURE SELECTOR PANEL IAW AMM. Nature of Condition:  WARNING INDICATION Precautionary Procedure:  
        UNSCHED LANDING Part Name: SELECTOR PANEL Part Condition: FAILED
        """
        pred, probs =  self.get_predictions([sample_record])
        print(sample_record)
        print("Prediction:", pred, probs)

        # Predict on existing records
        print("Batch prediction test")
        # df = pd.read_csv(r"data/SDR_Example.csv")
        example_path = os.path.join(self.dir_path, 'data', 'SDR_Example.csv')
        df = pd.read_csv(example_path)
        records = df["Text"]
        # golds = [1 if x == 'Yes' else 0 for x in df["Label"]]
        golds = df["Label"]
        pred, probs = self.get_predictions(records)
        matches = [1 if x==y else 0 for x,y in zip(golds, pred)]
        print(sum(matches), " matches out of ", len(matches))

    def test_sdr_degraded_controllability(self):
        """
        Function to test the prediction of Degraded Controllability event
        :return: None
        """
        sample_record = ["""air turn back declared emergency, stabilizer out of trim light illuminated with autopilot engaged autopilot was not trimming. 
        right & right stabilizer trim motor per amm.""", """toilet service panel door has spots ( 02 ea ) of corrosion at inboard corners . [ 01 ] removed toilet service panel door for ac cess as per 
        ref b737-700 amm 52-49-07-000-801 ; [ 02 ] found exfoli aton at toilet service panel door upper forward and aft corne right and bonding wire 
        hole fastener out of limits as per ref b737-700 srm 52-40-01-1a-4 ; [ 03 ] verified steps 1 and 2ok to install serviceable toilet service panel door . 
        accomplished bonding resistance test . found ok as per ref b737-700 amm 52-49-07-400-801 ."""]

        # pred, probs = self.get_predictions(sample_record)
        # # print(sample_record)
        # # print("Prediction:", pred, probs)

        for s in sample_record:
            pred, probs =  self.get_predictions([s])
            print(s)
            print("Prediction:", pred, probs)

    def test_sdr_corrosion_limit(self):
        """
        Function to test the prediction of Degraded Controllability event
        :return: None
        """
        sample_record = """upper trailing edge panel attach chord at ws.434 22 inches aft of rear spar has corrosion around nutplate .
         blended corrosion on upper cord on aileron rib at ws 434 to a depth of 0.17 `` , max depth allowed . out of limits . 
         mechanically measured tag # 2932. installed new upper chord on aileron rib at ws 434 iaw b767srm 51-40-02 par 4 and 7 boeing dwg 113t1510
        """
        pred, probs =  self.get_predictions([sample_record])
        print(sample_record)
        print("Prediction:", pred, probs)

if __name__ == "__main__":

   #Load the trained model
   model_api = SdrInferenceAPI(event_type=CORROSION_LIMIT)
   # model_api.test_sdr_degraded_controllability()
   model_api.test_sdr_corrosion_limit()
   # model_api.test_sdr_depressurization_predictions()

   fire_model = SdrInferenceAPI(event_type=FIRE)
   text = """strong odor of smoke in the flight deck. troubleshooting showed main deck cargo pdu 9r motor burned, tripping the circuit-breaker. and 10r overheated. both pdu ' s removed and cover plate installed"""
   print(fire_model.get_predictions([text]))


   fuel_model = SdrInferenceAPI(event_type=FUEL)
   text = """aircraft was grounded: after return to gate for pax disturbance, fuel was streaming out of right wing tank surge valve, 
    right center boost pump was on and after turning off right center pump leak stopped. replaced nr 2 fueling shutoff valve , 
    per amm 28-21-51 (aala202009109005).  replaced nr 2 fuel float vent valve, per amm 28-13-11 (aala202009169012).  
    repla ced right float switch, per amm 28-21-71 (aala202009169013). 
    accomplished a satisfactory operational check of all repla ced items, per amm 28-26-00-650-802 and amm 28-21-00-700-801."""
   print(fuel_model.get_predictions([text]))


   engine_model = SdrInferenceAPI(event_type=ENGINE)
   text = """iffo & div - flight crew reported hearing loud rumbles from the left engine and then the engine flamed out. 
    flight crew  diverted to stl.  maintenance checked left engine chip detector, no defects, boroscope inspection per amm 72-00 no dama ge. 
    removed and replaced number nr 1 engine fuel pump, hmu and eec per amm 73-11 and 73-21.  engine operational check o k."""
   print(engine_model.get_predictions([text]))