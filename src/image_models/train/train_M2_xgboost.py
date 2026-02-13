import xgboost as xgb
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# chemins
BASE_DIR = r"C:\Users\amisf\Desktop\datascientest_projet"
OUT_DIR = os.path.join(BASE_DIR, "implementation", "outputs")

def train_m2():
    print("lancement train m2 xgboost")
    
    # je charge les features numpy extraites
    x_path = os.path.join(OUT_DIR, 'train_features_resnet50_augmented.npy')
    y_path = os.path.join(OUT_DIR, 'train_labels_augmented.npy')
    
    if not os.path.exists(x_path):
        print("features introuvables")
        return

    x_data = np.load(x_path)
    y_data = np.load(y_path)
    
    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_data)
    
    # je sauvegarde l encodeur pour la predict
    joblib.dump(le, os.path.join(OUT_DIR, "M2_encoder.pkl"))
    
    # split
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_enc, test_size=0.2)
    
    # config gpu hist
    params = {
        'objective': 'multi:softmax',
        'num_class': len(le.classes_),
        'n_estimators': 3000,
        'max_depth': 8,
        'learning_rate': 0.05,
        'tree_method': 'hist',
        'device': 'cuda',
        'early_stopping_rounds': 50
    }
    
    # fit
    model = xgb.XGBClassifier(**params)
    model.fit(
        x_train, y_train, 
        eval_set=[(x_val, y_val)], 
        verbose=100
    )
    
    # save format json xgboost
    model.save_model(os.path.join(OUT_DIR, "M2_IMAGE_Classic_XGBoost.json"))
    print("modele m2 sauvegarde")

if __name__ == "__main__":
    train_m2()