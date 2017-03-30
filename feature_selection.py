import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

dataset = pd.read_csv("LargeTrain.csv")

forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

data = dataset
label = dataset.pop('Class')

forest.fit(data, label)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(data.shape[1]):
      print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

      plt.figure()
      plt.title("Feature importances")
      plt.bar(range(data.shape[1]), importances[indices],
                 color="r", yerr=std[indices], align="center")
      plt.xticks(range(data.shape[1]), indices)
      plt.xlim([-1, data.shape[1]])
      plt.show()
