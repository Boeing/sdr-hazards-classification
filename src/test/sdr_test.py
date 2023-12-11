
from sdr_classifier import sdr_api
import pandas as pd
import pyodbc
import numpy as np
import urllib
import sqlalchemy as sa

my_model = sdr_api.SDRAPI()

my_model.test_sdr_depressurization_predictions()

# df = pd.read_csv('../sdr_gold_labels.csv')
# records = df["NormalizedEventText"]

records = ["""RETURNED TO DEPARTURE DUE TO THE LEFT ENGINE OIL QUANTITY WAS DECREASING. REPLACED THE ENGINE. 
Nature of Condition:  WARNING INDICATION, FLUID LOSS Precautionary Procedure:  UNSCHED LANDING Part Name: OIL SYSTEM Part Condition: LEAKING"""]

pred, probs = my_model.get_depressurization_predictions(records)


print("Done")

def SDR_Prediction():
    """

    :return:
    """

    SDR_SQL_CON = 'driver={{SQL Server}}; server={0}; database=BCAB_Data_Science; {1}'
    connection_string = SDR_SQL_CON.format('SQL-S-16D105S.NOS.BOEING.COM', 'Trusted_Connection=Yes')

    sdr_cnxn = pyodbc.connect(connection_string)
    sdr_df = pd.read_sql("select DataSourceId,[NormalizedEventText] from [dbo].[COSP_ML_SDR] where AircraftModel not like 'A%' and EventDate >= '2017-01-01' order by EventDate desc", sdr_cnxn)

    my_model = sdr_api.BoeingSDRAPI()
    records = sdr_df["NormalizedEventText"]
    pred, probs = my_model.get_depressurization_predictions(records)

    sdr_df['Prediction'] = pred
    sdr_df['Prob'] = np.round(probs, 4)
    sdr_df['Classifier'] = 'Depressurization'

    sdr_df = sdr_df[(sdr_df["Prediction"]==1)]

    ASIS_SQL_CON = 'DRIVER=ODBC Driver 17 for SQL Server; server={0}; database=ASIS_UAT; {1}'
    connection_string = ASIS_SQL_CON.format('agl-nws-14D101', 'Trusted_Connection=Yes')
    asis_cnxn = pyodbc.connect(connection_string)

    print(sdr_df.head(2))
    print(f'{len(sdr_df)} rows inserted to the [dbo].[tblSDRDepressurization] table')
    # insert_query = f"INSERT INTO [dbo].[tblSDRDepressurization] VALUES (?,?,?,?,?)"
    # cursor = asis_cnxn.cursor()
    # cursor.fast_executemany = True
    # cursor.executemany(insert_query, sdr_df.values.tolist())
    # # print(f'{len(sdr_pd)} rows inserted to the [dbo].[tblSDRDepressurization] table')
    # cursor.commit()
    # cursor.close()
    # asis_cnxn.close()

    params = urllib.parse.quote_plus(r'DRIVER=ODBC Driver 17 for SQL Server;SERVER=agl-nws-14D101;DATABASE=ASIS_UAT;Trusted_Connection=yes')

    conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
    engine = sa.create_engine(conn_str, fast_executemany=True)
    print("loaded")
    # sdr_pd.to_sql(name='tblSDRDepressurization',schema= 'dbo', con=engine, if_exists='replace',index=False, chunksize = 1000, method = 'multi')

    sdr_df.to_sql("tblSDRDepressurization", engine, schema="dbo", if_exists="append", index=False)

    print('Done')


if __name__ == "__main__":
    print("Test")
    SDR_Prediction()