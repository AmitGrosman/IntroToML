from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


def prepare_data(training_data, new_data):
    test = new_data.copy()
    train = training_data.copy()

    # ----- blood_type -> blood types groups -----
    # group_1_blood_type
    train.insert(len(train.columns) - 2, "group_1_blood_type", train['blood_type'].isin(['A+', 'A-']))
    train['group_1_blood_type'] = train['group_1_blood_type'].replace({True: 1, False: 0})
    test.insert(len(test.columns) - 2, "group_1_blood_type", test['blood_type'].isin(['A+', 'A-']))
    test['group_1_blood_type'] = test['group_1_blood_type'].replace({True: 1, False: 0})

    # group_2_blood_type
    train.insert(len(train.columns) - 2, "group_2_blood_type", train['blood_type'].isin(['AB+', 'AB-', 'B+', 'B-']))
    train['group_2_blood_type'] = train['group_2_blood_type'].replace({True: 1, False: 0})
    test.insert(len(test.columns) - 2, "group_2_blood_type", test['blood_type'].isin(['AB+', 'AB-', 'B+', 'B-']))
    test['group_2_blood_type'] = test['group_2_blood_type'].replace({True: 1, False: 0})

    # group_3_blood_type
    train.insert(len(train.columns) - 2, "group_3_blood_type", train['blood_type'].isin(['O+', 'O-']))
    train['group_3_blood_type'] = train['group_3_blood_type'].replace({True: 1, False: 0})
    test.insert(len(test.columns) - 2, "group_3_blood_type", test['blood_type'].isin(['O+', 'O-']))
    test['group_3_blood_type'] = test['group_3_blood_type'].replace({True: 1, False: 0})

    # drop blood_type
    train = train.drop('blood_type', axis=1)
    test = test.drop('blood_type', axis=1)

    # ----- symptoms OHE -----
    train['symptoms'] = train['symptoms'].fillna("")
    test['symptoms'] = test['symptoms'].fillna("")

    # transform symptoms feature into 5 binary features
    train.insert(len(train.columns) - 2, "sore_throat", train['symptoms'].str.contains("sore_throat"))
    train['sore_throat'] = train['sore_throat'].replace({True: 1, False: 0})
    train.insert(len(train.columns) - 2, "cough", train['symptoms'].str.contains("cough"))
    train['cough'] = train['cough'].replace({True: 1, False: 0})
    train.insert(len(train.columns) - 2, "shortness_of_breath", train['symptoms'].str.contains("shortness_of_breath"))
    train['shortness_of_breath'] = train['shortness_of_breath'].replace({True: 1, False: 0})
    train.insert(len(train.columns) - 2, "fever", train['symptoms'].str.contains("fever"))
    train['fever'] = train['fever'].replace({True: 1, False: 0})
    train.insert(len(train.columns) - 2, "smell_loss", train['symptoms'].str.contains("smell_loss"))
    train['smell_loss'] = train['smell_loss'].replace({True: 1, False: 0})
    train = train.drop('symptoms', axis=1)

    # same transformation for test data
    test.insert(len(test.columns) - 2, "sore_throat", test['symptoms'].str.contains("sore_throat"))
    test['sore_throat'] = test['sore_throat'].replace({True: 1, False: 0})
    test.insert(len(test.columns) - 2, "cough", test['symptoms'].str.contains("cough"))
    test['cough'] = test['cough'].replace({True: 1, False: 0})
    test.insert(len(test.columns) - 2, "shortness_of_breath", test['symptoms'].str.contains("shortness_of_breath"))
    test['shortness_of_breath'] = test['shortness_of_breath'].replace({True: 1, False: 0})
    test.insert(len(test.columns) - 2, "fever", test['symptoms'].str.contains("fever"))
    test['fever'] = test['fever'].replace({True: 1, False: 0})
    test.insert(len(test.columns) - 2, "smell_loss", test['symptoms'].str.contains("smell_loss"))
    test['smell_loss'] = test['smell_loss'].replace({True: 1, False: 0})
    test = test.drop('symptoms', axis=1)

    # ----- sex -> 0 for M, 1 for F -----
    train['sex'] = train['sex'].replace({'M': 0, 'F': 1})
    test['sex'] = test['sex'].replace({'M': 0, 'F': 1})

    # ----- current_location -> coordiante_X, coordiante_Y -----
    train['current_location'] = train['current_location'].str.replace(r"[() ']", '')
    train[['coordinate_X', 'coordinate_Y']] = train['current_location'].str.split(',', expand=True)
    train['coordinate_X'] = train['coordinate_X'].apply(float)
    train['coordinate_Y'] = train['coordinate_Y'].apply(float)
    train = train.drop('current_location', axis=1)

    test['current_location'] = test['current_location'].str.replace(r"[() ']", '')
    test[['coordinate_X', 'coordinate_Y']] = test['current_location'].str.split(',', expand=True)
    test['coordinate_X'] = test['coordinate_X'].apply(float)
    test['coordinate_Y'] = test['coordinate_Y'].apply(float)
    test = test.drop('current_location', axis=1)

    # ----- pcr_date -> time since epoch (timestamp) -----
    train['pcr_date_timestamp'] = pd.to_datetime(train['pcr_date'], format="%Y-%m-%d")
    train['pcr_date_timestamp'] = train['pcr_date_timestamp'].apply(pd.Timestamp.timestamp)
    train['pcr_date_timestamp'] = train['pcr_date_timestamp'].astype(int)
    train = train.drop('pcr_date', axis=1)

    test['pcr_date_timestamp'] = pd.to_datetime(test['pcr_date'], format="%Y-%m-%d")
    test['pcr_date_timestamp'] = test['pcr_date_timestamp'].apply(pd.Timestamp.timestamp)
    test['pcr_date_timestamp'] = test['pcr_date_timestamp'].astype(int)
    test = test.drop('pcr_date', axis=1)

    # ----- Scaling -----
    test_normalized = test.copy()

    # not normalizing patient_id
    standard_cols = ['age','weight','num_of_siblings','happiness_score','household_income',
                 'conversations_per_day','sugar_levels','sport_activity','pcr_date_timestamp']

    minmax_cols = ['sex','group_1_blood_type','group_2_blood_type','group_3_blood_type','sore_throat','cough',
               'shortness_of_breath','fever','smell_loss','coordinate_X','coordinate_Y'] + train.filter(like='PCR').columns.tolist()

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

    standard_scaler.fit(train[standard_cols])
    test_normalized[standard_cols] = standard_scaler.transform(test[standard_cols])

    minmax_scaler.fit(train[minmax_cols])
    test_normalized[minmax_cols] = minmax_scaler.transform(test[minmax_cols])

    return test_normalized


if __name__ == "__main__":
    df = pd.read_csv("./virus_data.csv")
    train_size = 0.8
    test_size = 1 - train_size
    train_df, test_df = train_test_split(df, test_size=test_size, train_size=train_size,
                                         random_state=51 + 14, shuffle=True)

    # Prepare training set according to itself
    train_df_prepared = prepare_data(train_df, train_df)
    # Prepare test set according to the raw training set
    test_df_prepared = prepare_data(train_df, test_df)

    # save CSV
    train_df_prepared.to_csv("/content/train_preprocessed.csv")
    test_df_prepared.to_csv("/content/test_preprocessed")
