# firstmachine
import sys
print('Python: {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(scipy._version_))
import numpy
print('Numpy: {}'.format(numpy._version_))
import matplotlib
print('Matplotlib: {}'.format(matplotlib._version_))
import pandas
print('Pandas: {}'.format(pandas._version_))
import sklearn
print('Sklearn: {}'.format(sklearn._version_))

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKfold
from sklearn.metrices import classification_report
from sklearn.metrices import confusion_matrix
from sklearn.metrices import accuracy_score
from sklearn.linear_model import LogisticRegresssion
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbours import KNeighboursClassifier
from sklearn.discrimant analysis import LinearDiscrimantAnalysis
from sklearn.naivebayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

