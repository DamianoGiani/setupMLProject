import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score)


def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]
    return np.array(scores).T


def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)



# %% Constants
score_fun = {'Precision': precision_score,
             'Recall': recall_score, 'Specificity': specificity_score,
             'F1 score': f1_score}
diagnosis = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
nclasses = len(diagnosis)
predictor_names = ['DNN', 'cardio.', 'emerg.', 'stud.']

# %% Read datasets
# Get two annotators
y_cardiologist1 = pd.read_csv('./data/annotations/cardiologist1.csv').values
y_cardiologist2 = pd.read_csv('./data/annotations/cardiologist2.csv').values
# Get true values
y_true = pd.read_csv('./data/annotations/gold_standard.csv').values
# Get residents and students performance
y_cardio = pd.read_csv('./data/annotations/cardiology_residents.csv').values
y_emerg = pd.read_csv('./data/annotations/emergency_residents.csv').values
y_student = pd.read_csv('./data/annotations/medical_students.csv').values
# get y_score for different models
y_score_list = np.load('outputs/dnn_output.npy')


# %% Get average model model
# Get micro average precision
micro_avg_precision = average_precision_score(y_true[:, :nclasses], y_score_list[:, :nclasses], average='micro')
# get ordered index

print('Micro average precision')
print((micro_avg_precision))
# get 6th best model (immediatly above median) out 10 different models
y_score_best = y_score_list
# Get threshold that yield the best precision recall using "get_optimal_precision_recall" on validation set
#   (we rounded it up to three decimal cases to make it easier to read...)
_, _, threshold = get_optimal_precision_recall(y_true,y_score_list)
threshold=[round(elem,3) for elem in threshold]
print(threshold)
#threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174]) #loro top
mask = y_score_best > threshold
# Get neural network prediction
# This data was also saved in './data/annotations/dnn.csv'
y_neuralnet = np.zeros_like(y_score_best)
y_neuralnet[mask] = 1


# %% Generate table with scores for the average model (Table 2)
scores_list = []
for y_pred in [y_neuralnet, y_cardio, y_emerg, y_student]:
    # Compute scores
    scores = get_scores(y_true, y_pred, score_fun)
    # Put them into a data frame
    scores_df = pd.DataFrame(scores, index=diagnosis, columns=score_fun.keys())
    # Append
    scores_list.append(scores_df)
# Concatenate dataframes
scores_all_df = pd.concat(scores_list, axis=1, keys=['DNN', 'cardio.', 'emerg.', 'stud.'])
# Change multiindex levels
scores_all_df = scores_all_df.swaplevel(0, 1, axis=1)
scores_all_df = scores_all_df.reindex(level=0, columns=score_fun.keys())
# Save results
scores_all_df.to_excel("outputs/results/tables/scores.xlsx", float_format='%.3f')
scores_all_df.to_csv("outputs/results/tables/scores.csv", float_format='%.3f')