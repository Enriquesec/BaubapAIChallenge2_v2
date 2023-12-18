from sklearn.metrics import classification_report, brier_score_loss, confusion_matrix, accuracy_score


# Create a function that return the metrics we need to compare if the model is a good one.
# The function returns 'brier_score', 'confusion matrix', 'accuracy negative and positive', 'accuracy'
def metricas_modelo(y_real, prob_pred):
    y_pred = list(map(lambda x: 1 if x==True else 0, prob_pred>0.5))
    brier = round(brier_score_loss(y_real,prob_pred)*100,2)
    cm = confusion_matrix(y_real,y_pred)
    acc_neg = round(100*cm[0,0]/(cm[0,0]+cm[0,1]),2)
    acc_pos = round(100*cm[1,1]/(cm[1,0]+cm[1,1]),2)
    acc = round(100*accuracy_score(y_real, y_pred),2)
    print(f' brier: {brier},\n confusion_m: \n{cm},\n acc_neg: {acc_neg},\n acc_pos: {acc_pos},\n accuracy: {acc}')
