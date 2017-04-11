import os
import pandas as pd

# machine learning
from sklearn.ensemble import RandomForestClassifier

#To set the working directory
os.chdir("/Users/steven/Documents/dataMining/Kaggle/digitRecognizer")
cwd = os.getcwd()

# get titanic & test csv files as a DataFrame
digit_train_df = pd.read_csv("input/train.csv")
digit_test_df    = pd.read_csv("input/test.csv")

# preview the data
digit_train_df.head()
digit_test_df.head()

#defining the training data set
X_train = digit_train_df.drop("label",axis=1)
Y_train = digit_train_df["label"]
X_test  = digit_test_df

random_forest = RandomForestClassifier(n_estimators=10)

random_forest.fit(X_train, Y_train)

random_forest.decision_path(X_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

submission = pd.DataFrame({
        "ImageId": X_test.index +1,
        "Label": Y_pred
    })
submission.to_csv('RF_10.csv', index=False)
