import warnings
warnings.simplefilter("ignore")

import pandas as pd
import os
import sys
import numpy as np
import pickle
from xgboost import XGBClassifier
import xgboost as xgb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inp", help=""" Input metrics path """)
parser.add_argument("--model_path", help="""Path to where the QC model is saved """)
parser.add_argument("--out", help=""" QC file """)
args = parser.parse_args()



# Read the metrics dataframe
df = pd.read_csv(args.inp + "/metrics.csv")
X = df.drop(['subjectID','JHU3_Body_MaxThickness',
                     'JHU3_Body_MeanThickness', 'JHU3_Body_MinThickness', 'JHU3_Body_StdThickness'], axis = 1)


# Selected features
feats = ['Total_Area','Total_Curve','Total_MeanCurve', 'Total_StdCurve','Total_MaxCurve', 'Total_MeanThickness','Total_StdThickness','Total_MaxThickness','Total_Perimeter','Total_EuclideanDist','Total_MedialCurveLength',
'Ratio_MedialCurve_MaxEuclideanDist','Witelson5_Genu_Area','Witelson5_Genu_Curve','Witelson5_Genu_MeanCurve','Witelson5_Genu_StdCurve',
'Witelson5_Genu_MaxCurve','Witelson5_Genu_MinCurve','Witelson5_Genu_MeanThickness','Witelson5_Genu_StdThickness','Witelson5_Genu_MaxThickness',
'Witelson5_Genu_MinThickness','Witelson5_AnteriorBody_Curve','Witelson5_AnteriorBody_MeanCurve','Witelson5_AnteriorBody_StdCurve', 
'Witelson5_AnteriorBody_MaxCurve','Witelson5_AnteriorBody_MinCurve','Witelson5_AnteriorBody_MeanThickness','Witelson5_AnteriorBody_StdThickness',
'Witelson5_AnteriorBody_MaxThickness','Witelson5_AnteriorBody_MinThickness','Witelson5_PosteriorBody_Area','Witelson5_PosteriorBody_Curve',
'Witelson5_PosteriorBody_MeanCurve','Witelson5_PosteriorBody_StdCurve','Witelson5_PosteriorBody_MaxCurve','Witelson5_PosteriorBody_MinCurve',
'Witelson5_PosteriorBody_MeanThickness','Witelson5_PosteriorBody_StdThickness','Witelson5_PosteriorBody_MaxThickness','Witelson5_PosteriorBody_MinThickness',
'Witelson5_Isthmus_Area','Witelson5_Isthmus_Curve','Witelson5_Isthmus_MeanCurve','Witelson5_Isthmus_StdCurve','Witelson5_Isthmus_MaxCurve', 
'Witelson5_Isthmus_MinCurve','Witelson5_Isthmus_MeanThickness','Witelson5_Isthmus_StdThickness','Witelson5_Isthmus_MaxThickness','Witelson5_Isthmus_MinThickness','Witelson5_Splenium_Area','Witelson5_Splenium_Curve','Witelson5_Splenium_MeanCurve','Witelson5_Splenium_StdCurve','Witelson5_Splenium_MaxCurve',
'Witelson5_Splenium_MinCurve','Witelson5_Splenium_MeanThickness','Witelson5_Splenium_StdThickness','Witelson5_Splenium_MaxThickness','JHU3_Genu_Curve',
'JHU3_Genu_MeanCurve','JHU3_Genu_StdCurve','JHU3_Genu_MaxCurve','JHU3_Genu_MinCurve','JHU3_Genu_MeanThickness','JHU3_Genu_StdThickness','JHU3_Genu_MaxThickness',
'JHU3_Body_Curve','JHU3_Body_MeanCurve','JHU3_Body_StdCurve','JHU3_Splenium_Curve','JHU3_Splenium_MeanCurve','JHU3_Splenium_StdCurve','JHU3_Splenium_MaxCurve',
'JHU3_Splenium_MinCurve','JHU3_Splenium_MeanThickness','JHU3_Splenium_StdThickness','JHU3_Splenium_MaxThickness','JHU3_Splenium_MinThickness']
X_new = X[feats]


# Load the model 
filename = args.model_path + '/model_qc.sav'
xgb = pickle.load(open(filename, 'rb'))

# Predict the labels
y_pred = xgb.predict(X_new)

df_new = df.copy()
df_new["QC_label"] = y_pred
df_new.to_csv(args.out + "/metrics_qc.csv", index=False)

print("Done with QC!")




