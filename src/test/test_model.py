
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sdr_hazards_classification.sdr_api import SdrInferenceAPI
# my_model = sdr_api.SdrInferenceAPI()
# my_model.test_sdr_depressurization_predictions()

class TestSdrModel():
    # Constructor
    @pytest.fixture
    def model_fixture(self):
        def _load_model(event_type="depressurization"):
            model = SdrInferenceAPI(event_type)
            yield model

        yield _load_model

    @pytest.fixture
    def load_depressurization_model(self, event_type="depressurization"):
        yield SdrInferenceAPI(event_type)

    @pytest.fixture
    def load_controll_model(self, event_type="degraded-controllability"):
        yield SdrInferenceAPI(event_type)

    @pytest.fixture
    def load_corrosion_model(self, event_type="corrosion-limit"):
        yield SdrInferenceAPI(event_type)

    @pytest.fixture
    def load_fire_model(self, event_type="fire"):
        yield SdrInferenceAPI(event_type)

    @pytest.fixture
    def load_pda_model(self, event_type="pda"):
        yield SdrInferenceAPI(event_type)

    @pytest.fixture
    def load_rto_model(self, event_type="rto"):
        yield SdrInferenceAPI(event_type)

    def test_depressurization_prediction_No(self, load_depressurization_model):

        expect_result = np.array(["No", "No"])

        record = ["""RETURNED TO DEPARTURE DUE TO THE LEFT ENGINE OIL QUANTITY WAS DECREASING. REPLACED THE ENGINE. 
            Nature of Condition:  WARNING INDICATION, FLUID LOSS Precautionary Procedure:  UNSCHED LANDING Part Name: OIL SYSTEM Part Condition: LEAKING""",
                  """PAX OXYGEN MASKS AT ROW 11ABC DEPLOYED DURING CLIMB.  RETURNED TO LAND.  ACFT GROUNDED.   REPACKED PSU THE OXYGEN MASKSIAW.AMM 35-22-31-000-803-001. 
                  Nature of Condition:  OTHER Precautionary Procedure:  UNSCHED LANDING Part Name: OXYGEN MASK Part Condition: UNWANTED DEPLOY"""]

        pred, probs = load_depressurization_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs) > 0.9)

    def test_depressurization_prediction_Yes(self, load_depressurization_model):

        expect_result = np.array(["Yes", "Yes"])

        record = ["""CABIN ALTITUDE EICAS WARNING AT FL380. COMPLIED WITH QRH,  DESCENDED TO FINAL ALTITUDE FL250. RECOVERED CABIN PRESSURE IN AUTO 2 AT CABIN ALTITUDE 7500.  NORMAL DESCENT AND PRESSURE UNTIL LFPG.   REPLACED THE NR 1 CPC IAW MM 21-31-02-4, TESTED OK.  COMPLYED WITH CABIN PRESSURE DECAY CHECK IAW MM 05-51-24-2, TEST PASSED. Nature of Condition:  
        WARNING INDICATION Precautionary Procedure:  O2 MASK DEPLOYED Part Name: CONTROLLER Part Condition: MALFUNCTIONED""",
                      """LOST CABIN PRESSUURIZATION AT FLIGHT LEVEL 30000, CABIN ALTITUDE WARNING HORN SOUNDED AT 10000 FEET CABIN PRESSURE. UNAB LE TO CONTROL CABIN PRESSURE 
                      WITH OUTFLOW VALVE CLOSED."""]

        pred, probs = load_depressurization_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs) > 0.9)

    def test_depressurization_prediction_Maybe(self, load_depressurization_model):

        expect_result = np.array(["Yes", "Yes"])

        record = ["""PASSING THROUGH FL 360 DURING CLIMB OUT, CABIN ALTITUDE WARNING HORN SOUNDED. CABIN ALTITUDE WAS 5500 FT, CABIN DIFFERENTIAL PRESSURE APPROACHED 9. DIVERTED.  R & R THE NR 2 CABIN PRESSURE SWITCH IAW MM. 
        Nature of Condition:  WARNING INDICATION Precautionary Procedure:  UNSCHED LANDING Part Name: SWITCH Part Condition: FAULTED""",
                  """AIRCRAFT WAS GROUNDED: 10 MINUTES AFTER SECURING LEFT PACK PER BLEED TRIP, QRH PROCEDURE, CABIN ALTITUDE WARNING SOUNDED . EMERGENCY DECLARED, DONNED THEIR O2 MASKS
                   (NO CABIN O2 MASK DEPLOYED) AND DIVERTED TO MIA. FLIGHT LANDED WITHOUT FURTH ER INCIDENT.  MAINTENANCE REPLACED 450 SENSOR IAW AMM 36-11-05 AND REPLACED PRECOOLER CONTROL VALVE PER AMM 36-12-02 AND  PERFORMED SATISFACTORY OPERATIONAL CHECK IAW AMM 71-00-00.  """]

        pred, probs = load_depressurization_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs)> 0.8)

    def test_controllability_prediction_No(self, load_controll_model):

        expect_result = np.array(["No", "No"])

        record = ["""RETURNED TO DEPARTURE DUE TO THE LEFT ENGINE OIL QUANTITY WAS DECREASING. REPLACED THE ENGINE. 
            Nature of Condition:  WARNING INDICATION, FLUID LOSS Precautionary Procedure:  UNSCHED LANDING Part Name: OIL SYSTEM Part Condition: LEAKING""",
                  """uncommanded electrical seat movement in the flight deck, flight officer. a flight crew seat power adjustment.
                  troubleshooting first officer `s uncommanded seat movement per fim 25-10-fo-23a. found no defects noted at this time."""]

        pred, probs = load_controll_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs) > 0.9)

    def test_controllability_prediction_Yes(self, load_controll_model):
        expect_result = np.array(["Yes", "Yes"])

        record = ["""CABIN ALTITUDE EICAS WARNING AT FL380. COMPLIED WITH QRH,  DESCENDED TO FINAL ALTITUDE FL250. RECOVERED CABIN PRESSURE IN AUTO 2 AT CABIN ALTITUDE 7500.  NORMAL DESCENT AND PRESSURE UNTIL LFPG.   REPLACED THE NR 1 CPC IAW MM 21-31-02-4, TESTED OK.  COMPLYED WITH CABIN PRESSURE DECAY CHECK IAW MM 05-51-24-2, TEST PASSED. Nature of Condition:  
        WARNING INDICATION Precautionary Procedure:  O2 MASK DEPLOYED Part Name: CONTROLLER Part Condition: MALFUNCTIONED""",
                  """aircraft was not grounded : during climb, autopilot engaged for approximately one second then began to roll left, 
                  followed by autopilot disconnect and aural warning on two attempts. replaced autopilot aileron switch iaw amm 22-11-12, operation check good."""]

        pred, probs = load_controll_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs) > 0.9)

    def test_fire(self, load_fire_model):

        expect_result = np.array(["No", "Yes", "No", "Yes"])

        record = ["""RETURNED TO DEPARTURE DUE TO THE LEFT ENGINE OIL QUANTITY WAS DECREASING. REPLACED THE ENGINE. 
            Nature of Condition:  WARNING INDICATION, FLUID LOSS Precautionary Procedure:  UNSCHED LANDING Part Name: OIL SYSTEM Part Condition: LEAKING""",
                  """burning electrical odor noticed in cabin from row 28 to back of aircraft. no signs of smoke noticed. odor approximately 1 hr out from landing. ran nr 1 and 2 engines independently with packs running. no smoke or any burning odor. checked reading lights iaw mcc for shorting and arcing. no signs of shorting and arcing. checked ovens and coffeemakers, no evidence of burning or arcing noted. nature of condition : smoke fumes odors sparks precautionary procedure : other part name : unknown part condition : odor.""",
                  """CABIN ALTITUDE EICAS WARNING AT FL380. COMPLIED WITH QRH,  DESCENDED TO FINAL ALTITUDE FL250. RECOVERED CABIN PRESSURE IN AUTO 2 AT CABIN ALTITUDE 7500.  NORMAL DESCENT AND PRESSURE UNTIL LFPG.   REPLACED THE NR 1 CPC IAW MM 21-31-02-4, TESTED OK.  COMPLYED WITH CABIN PRESSURE DECAY CHECK IAW MM 05-51-24-2, TEST PASSED. Nature of Condition:  
                WARNING INDICATION Precautionary Procedure:  O2 MASK DEPLOYED Part Name: CONTROLLER Part Condition: MALFUNCTIONED""",
                  """on climbout the cockpit filled with smoke. the aircraft declared an emergency and returned to the airport. the smoke was observed coming from first officer ' s side. after aircraft returned to gate, mechanic found p6-12 circuit-breaker with evidence of being overheated and terminal connector burned. nature of condition : smoke fumes odors sparks precautionary procedure : unsched landing part name : circuit-breaker part condition : failed"""
                  ]

        pred, probs = load_fire_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs) > 0.9)

    def test_pda(self, load_pda_model):
        expect_result = np.array(["No", "Yes", "No", "Yes"])

        record = ["""CABIN ALTITUDE EICAS WARNING AT FL380. COMPLIED WITH QRH,  DESCENDED TO FINAL ALTITUDE FL250. RECOVERED CABIN PRESSURE IN AUTO 2 AT CABIN ALTITUDE 7500.  NORMAL DESCENT AND PRESSURE UNTIL LFPG.   REPLACED THE NR 1 CPC IAW MM 21-31-02-4, TESTED OK.  COMPLYED WITH CABIN PRESSURE DECAY CHECK IAW MM 05-51-24-2, TEST PASSED. Nature of Condition:  
        WARNING INDICATION Precautionary Procedure:  O2 MASK DEPLOYED Part Name: CONTROLLER Part Condition: MALFUNCTIONED""",
        """right engine pylon aft outboard side panel assembly is missing panel number 446al. the panel possibly departed during op eration of the aircraft. the airplane was removed from service ( grounded ) and a serviceable panel 446al was installed. the cause of the missing panel is unknown.""",
                  """aircraft was not grounded : during climb, autopilot engaged for approximately one second then began to roll left, 
                  followed by autopilot disconnect and aural warning on two attempts. replaced autopilot aileron switch iaw amm 22-11-12, operation check good.""",
                  """( log 5515918 parts departing airplane ) during pre-flight discovered large panel missing forward of right main wheel well. maintenance reinstalled body panel 192br per amm 53-51."""]

        pred, probs = load_pda_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs) > 0.9)

    def test_rto(self, load_rto_model):
        expect_result = np.array(["No", "Yes", "No", "Yes"])

        record = ["""forward cargo bay, bs 478 - 500, stringer 26l found cracked. removed the damaged stringer section and fabricated a repair iaw srm 53-00-03, fig 201, repair 2. installed iaw srm 51-40-02. nature of condition : other precautionary procedure : none part name : stringer part condition : cracked""",
        """no airspeed on captains indicator, aborted takeoff and returned to blocks. inspected pitot tube and removed debris, op ' s and leak check good iaw amm 34-11-00. nature of condition : other precautionary procedure : aborted takeoff part name : pitot tube part condition : obstructed""",
                  """during inspection, found a crack on the floor support between bs 277 and 294.5, rbl 1. maintenance replaced the floor support iaw srm 51-40-02.""",
                  """during takeoff roll, the left thrust reverser warning light illuminated. aborted takeoff at approximately 40 knots. removed and replaced the engine accessory unit iaw amm 75-34-06. accomplished built in test equipment check iaw fim 78-31-00. system checked normal. nature of condition : warning indication precautionary procedure : aborted takeoff part name : accessory unit part condition : faulty"""]

        pred, probs = load_rto_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs) > 0.9)

    def test_corrosion_limit(self, load_corrosion_model):
        expect_result = np.array(["Beyond Limit", "No Corrosion", "No Limit", "Within Limit", "Within_Beyond_Limit"])

        record = ["""while accomplishing special non-rountine maintenance . repair found cabin floor seat track at sta 5-113 x= 27.5 corrod ed . 
        corrosion on seat track flange at cabin sta 5-113 x= 27.5 evaluated found exfoliated and exceed limits iaw srm 53 - 00-52-1a figure 101 allowable damage 1. 
        corroded seat track segment removed iaw srm 51-31-01-0g and repaired by sectio nal seat track replacement iaw srm 53-00-52-2r figure 201 repair 1. 
        repair classified as major per gmm 6 - 2-100. eo # 1 - 5100-7-5001 b7 part 1 .""",
          """aircraft in base maintenance: main cabin floor beam seat track upper splice plate has over sized/elongated holes located  
          at sta 887 right butt line 24.75. removed and replaced upper splice plate per srm 51-40-02.""",
        """wing center section upper surface at sta 955 , rbl 86 found corrosion , corrective action : repaired the corrosion on the center 
        wing upper skin panel and rear spar upper chord at sta 955 ; rbl 86 as per pri 603514-14 rev 0 and pri 603514-14 ( 252 3687 )""",
        """a22-18 ; aft pit sta 727i+10-827 ; lbl22 ; wl157 ; z-channel has corrosion adjacent to 21 ea fastener holes upper and a coup 
        leading edge on lower mrk ' d with black b . [ 01 ] removed corrosion on z - channel iaw srm 51-10-02-0g-0 item 15 ; 
        [ 02 ] evaluated allo table limits : .""",
        """aft cargo right aft transition panel is corroded . grounded - yes . performed surface blend removed corrosion , 
        sta 727j+8 - 727k+12 found to be out of limits repaired section , all other areas found to be within limits re-stored 
        finish iaw b737-9 00 srm 53-00-53-2r-5 srm 53-00-53-1a-2w 4 `` o ."""]

        pred, probs = load_corrosion_model.get_predictions(record)

        assert_array_equal(pred, expect_result)
        assert np.all(np.array(probs) > 0.9)

if __name__ == "__main__":

    #Run unit tests
    unit_test = TestSdrModel()

    # my_test.test_depressurization_prediction_No()
    # my_test.test_depressurization_prediction_Yes()
    # my_test.test_depressurization_prediction_Maybe()
    #
    # my_test.load_model(event_type="degraded-controllability")
    #
    # my_test.test_controllability_prediction_No()
    # my_test.test_controllability_prediction_Yes()
    # my_test.test_depressurization_prediction_Maybe()
