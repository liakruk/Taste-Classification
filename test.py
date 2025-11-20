from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow.compat.v1 import ConfigProto,InteractiveSession

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction=0.5
session = InteractiveSession(config=config)

violations_list = ['False','True',]

test_dataset = pd.read_csv('test.csv')
x_test = test_dataset.iloc[:, :-1].values
y_test_ = test_dataset.iloc[:, -1].values
y_test = []

for i in range(len(y_test_)):
	if y_test_[i] == True:
		y_test.append(1)
	else:
		y_test.append(0)


# Recreate the exact same model, including its weights and the optimizer
dnn_model = tf.keras.models.load_model('dnn_model.h5')
dnn_model.summary()

predictions = []
i = 0
length = len(y_test)
P = dnn_model(x_test)

# pred = np.argmax(P, axis = 1)

temp = P.numpy().tolist()
pred = []
for i in temp:
    pred.append(i[1] > 0.95)

#temp = P.numpy().flatten()
#pred = temp > 0.5    

predictions = pred
print()

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
i = 0
for violation in violations_list:
	print(i,violation)
	i+=1
print(classification_report(y_test, predictions, labels=range(len(violations_list)), target_names=violations_list))
cm = confusion_matrix(y_test, predictions)
print(cm)

print()

print('************* IARA++ *************')
for i in cm:
	for j in i:
		print(100*j/sum(i), end = '\t')
	print()
print('Acc: %.2f %c'%(100*accuracy_score(y_test, predictions),'%'))
print('AUC-ROC: ', roc_auc_score(y_test, P[:,1]))

x = [0,0.5,1]
fp, tp, thesholds = roc_curve(y_test, P[:,1])
plt.plot(fp, tp, 'blue')
plt.plot(x, x, 'black', linestyle='dashed')

plt.title('ROC Curve (AUC: %.4f)'%(roc_auc_score(y_test, P[:,1])))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()

print()
print('************* Original *************')
cm = [[67.64, 4.28],[6.99, 21.09]]
	
for i in cm:
	for j in i:
		print(100*j/sum(i), end = '\t')
	print()
print('Acc: %.2f %c'%(100*(67.64+21.09)/(67.64+4.28+6.99+21.09),'%'))
print()