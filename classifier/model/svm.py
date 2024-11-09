import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC

svm = SVC(kernel="poly", C=0.1, gamma=1)
