#merging multiple spreadsheets 

import pandas as pd
d1 = pd.read_csv('1.csv')
d2 = pd.read_csv('2.csv')
d3 = pd.read_csv('3.csv')
d4 = pd.read_csv('4.csv')
d5 = pd.read_csv('5.csv')

result = pd.concat([d1, d2, d3, d4, d5], axis = 1, ignore_index=False)
display(result)

    #1. Univariate Selection
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv("D://Blogs//train.csv")

X = result.iloc[:,:-1] #independent columns
y = result.iloc[:,-1]  #target column i.e price range
    
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

    #2. Feature Importance
import pandas as pd
import numpy as np
data = pd.read_csv("D://Blogs//train.csv")

X = result.iloc[:,:-1] #independent columns
y = result.iloc[:,-1]  #target column i.e price range

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

    #3. Correlation Matrix with Heatmap
import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_csv("D://Blogs//train.csv")

X = result.iloc[:,:-1] #independent columns
y = result.iloc[:,-1]  #target column i.e price range

#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

    #4. Feature Selection Using Correlation Matrix
cor = df.corr()
cor_target = abs(cor["havarth3"])
relevant_features = cor_target[cor_target > 0.2]
relevant_features.index

    #5. Wrapper Method
import statsmodels.api as sm
X_new = sm.add_constant(X)
model = sm.OLS(y, X_new).fit()
model.pvalues

selected_features = list(X.columns)
pmax = 1
while (len(selected_features)>0):
    p= []
    X_new = X[selected_features]
    X_new = sm.add_constant(X_new)
    model = sm.OLS(y,X_new).fit()
    p = pd.Series(model.pvalues.values[1:],index = selected_features)      
    pmax = max(p)
    feature_pmax = p.idxmax()
    if(pmax>0.05):
        selected_features.remove(feature_pmax)
    else:
        break
selected_features

    #6. Recursive feature elimination with cross-validation
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
X, y = make_classification(
    n_samples=1000,
    n_features=25,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=8,
    n_clusters_per_class=1,
    random_state=0,
)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring shows the proportion of correct classifications

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(
    estimator=svc,
    step=1,
    cv=StratifiedKFold(2),
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
    rfecv.grid_scores_,
)
plt.show()
    
















    
