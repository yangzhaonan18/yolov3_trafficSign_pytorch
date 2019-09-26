import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from operator import itemgetter

def print_accuracy(targets, preds):
    seaborn.set(font_scale=0.5)
    f, ax = plt.subplots(figsize=(14, 10))
    cm  = confusion_matrix(targets, preds)
    # cm_df = pd.DataFrame(cm, columns= np.unique(targets), index =  np.unique(targets))
    info = []
    for i in range(len(cm)):
        acc = cm[i][i] / (sum(cm[i] + 1e-5))
        info.append([cm[i][i], sum(cm[i]), acc])

    info = sorted(info, key=itemgetter(1), reverse=True)
    for i in  range(len(cm)):
        # print(info[i])
        a, b, c = info[i][0], info[i][1], info[i][2]
        print(f"{i:3d},cm[i][i]/sum(cm[i]) = {a:3d} / {b:3d} = acc = {int(100 * c):2d}%")


    cm_df = pd.DataFrame(cm)


    # cm_df.index.name = "True Label"
    # cm_df.columns.name = "Predicted Label"
    # print(cm_df)


    # seaborn.heatmap(cm_df, annot=True,  cmap="RdBu",linewidths = 0.005, fmt="d")  #
    # plt.savefig("25.png", dpi = 400, bbox_inches="tight")