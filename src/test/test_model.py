
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sdr_classifier import sdr_api

my_model = sdr_api.SDRAPI()
my_model.test_sdr_depressurization_predictions()

def run_prediction(record):

    pred, probs = my_model.get_depressurization_predictions(record)
    return pred, probs

def test_prediction_No():

    expect_result = np.array([0, 0])

    record = ["""RETURNED TO DEPARTURE DUE TO THE LEFT ENGINE OIL QUANTITY WAS DECREASING. REPLACED THE ENGINE. 
        Nature of Condition:  WARNING INDICATION, FLUID LOSS Precautionary Procedure:  UNSCHED LANDING Part Name: OIL SYSTEM Part Condition: LEAKING""",
              """PAX OXYGEN MASKS AT ROW 11ABC DEPLOYED DURING CLIMB.  RETURNED TO LAND.  ACFT GROUNDED.   REPACKED PSU THE OXYGEN MASKSIAW.AMM 35-22-31-000-803-001. 
              Nature of Condition:  OTHER Precautionary Procedure:  UNSCHED LANDING Part Name: OXYGEN MASK Part Condition: UNWANTED DEPLOY"""]

    pred, probs = run_prediction(record)

    assert_array_equal(pred, expect_result)
    assert np.all(probs < 0.4)

def test_prediction_Yes():

    expect_result = np.array([1, 1])

    record = ["""CABIN ALTITUDE EICAS WARNING AT FL380. COMPLIED WITH QRH,  DESCENDED TO FINAL ALTITUDE FL250. RECOVERED CABIN PRESSURE IN AUTO 2 AT CABIN ALTITUDE 7500.  NORMAL DESCENT AND PRESSURE UNTIL LFPG.   REPLACED THE NR 1 CPC IAW MM 21-31-02-4, TESTED OK.  COMPLYED WITH CABIN PRESSURE DECAY CHECK IAW MM 05-51-24-2, TEST PASSED. Nature of Condition:  
    WARNING INDICATION Precautionary Procedure:  O2 MASK DEPLOYED Part Name: CONTROLLER Part Condition: MALFUNCTIONED""",
                  """LOST CABIN PRESSUURIZATION AT FLIGHT LEVEL 30000, CABIN ALTITUDE WARNING HORN SOUNDED AT 10000 FEET CABIN PRESSURE. UNAB LE TO CONTROL CABIN PRESSURE 
                  WITH OUTFLOW VALVE CLOSED."""]

    pred, probs = run_prediction(record)

    assert_array_equal(pred, expect_result)
    assert np.all(probs > 0.9)

def test_prediction_Maybe():

    expect_result = np.array([1, 1])

    record = ["""PASSING THROUGH FL 360 DURING CLIMB OUT, CABIN ALTITUDE WARNING HORN SOUNDED. CABIN ALTITUDE WAS 5500 FT, CABIN DIFFERENTIAL PRESSURE APPROACHED 9. DIVERTED.  R & R THE NR 2 CABIN PRESSURE SWITCH IAW MM. 
    Nature of Condition:  WARNING INDICATION Precautionary Procedure:  UNSCHED LANDING Part Name: SWITCH Part Condition: FAULTED""",
              """AIRCRAFT WAS GROUNDED: 10 MINUTES AFTER SECURING LEFT PACK PER BLEED TRIP, QRH PROCEDURE, CABIN ALTITUDE WARNING SOUNDED . EMERGENCY DECLARED, DONNED THEIR O2 MASKS
               (NO CABIN O2 MASK DEPLOYED) AND DIVERTED TO MIA. FLIGHT LANDED WITHOUT FURTH ER INCIDENT.  MAINTENANCE REPLACED 450 SENSOR IAW AMM 36-11-05 AND REPLACED PRECOOLER CONTROL VALVE PER AMM 36-12-02 AND  PERFORMED SATISFACTORY OPERATIONAL CHECK IAW AMM 71-00-00.  """]

    pred, probs = run_prediction(record)

    assert_array_equal(pred, expect_result)
    assert np.all(probs > 0.8)

if __name__ == "__main__":

    #Run unit tests
    test_prediction_No()
    test_prediction_Yes()
    test_prediction_Maybe()