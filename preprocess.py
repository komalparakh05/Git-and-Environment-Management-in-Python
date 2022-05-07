import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load


def preprocess(data: pd.DataFrame, data_context: str) -> pd.DataFrame:
    # encoding and arranging data
    data = data[['Id', 'LotArea', 'LotShape', 'RoofStyle',
                 'KitchenQual', 'OverallQual', 'Neighborhood',
                 'GrLivArea', 'GarageCars', 'CentralAir']]
    if data_context == 'train':
        Onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        Onehot_encoder = Onehot_encoder.fit(data[['LotShape',
                                                  'Neighborhood',
                                                  'RoofStyle',
                                                  'CentralAir']])
        dump(Onehot_encoder, r"../models/onehot_encoder.joblib")

    Onehot_encoder = load(r"../models/onehot_encoder.joblib")
    encoded_data = pd.DataFrame(Onehot_encoder.transform
                                (data[['LotShape', 'Neighborhood',
                                       'RoofStyle', 'CentralAir']]),
                                columns=Onehot_encoder.
                                get_feature_names_out())

    encoded_data = pd.concat([data.reset_index(drop=True),
                              encoded_data.reset_index(drop=True)], axis=1)
    l1 = {}
    l1["kitchen_qual"] = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan])
    l1["KQcode"] = np.array([5, 4, 3, 2, 1, 0])
    kitchen_qual = pd.DataFrame(l1)
    encoded_data = pd.merge(left=encoded_data, right=kitchen_qual,
                            how='left', left_on='KitchenQual',
                            right_on='kitchen_qual')
    encoded_data = encoded_data.drop(["KitchenQual", "kitchen_qual"], axis=1)
    encoded_data["GarageCars"][encoded_data.isna().any(axis=1)] = 0.0
    encoded_data["KQcode"][encoded_data.isna().any(axis=1)] = 0
    encoded_data = encoded_data.drop(['LotShape',
                                      'Neighborhood', 'RoofStyle',
                                      'CentralAir'], axis=1)
    return encoded_data



