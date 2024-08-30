import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier

# get data
# TODO: connect this pipeline with the segmentation one

# labeling?
X = df.drop(['class'], axis=1)

y = df['class']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# encode categorical variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=[])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

# instantiate the classifier
rfc = RandomForestClassifier(random_state=0)

# fit the model
rfc.fit(X_train, y_train)

# Predict the Test set results
y_pred = rfc.predict(X_test)
