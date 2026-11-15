import os
import shutil
import kagglehub

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Telecharge dans:", path)

os.makedirs("data/raw", exist_ok=True)
shutil.copy(os.path.join(path, "creditcard.csv"), "data/raw/creditcard.csv")
print("Copie dans data/raw/creditcard.csv")