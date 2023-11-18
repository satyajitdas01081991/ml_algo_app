
import streamlit as st
import plotly_express as px
import pandas as pd

import streamlit as st 
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
 

import sys
sys.tracebacklimit=0


 

global features
global unique_target



st.title( 'Find the Best ML Algorithm for your dataset.' )

try:
	uploaded_file = st.sidebar.file_uploader(label="Upload your CSV file.", type=['csv'])  
	
	
except Exception as e:
	print(e)
	st.write("Please upload file to the application.")  	
	



print(type(uploaded_file))






print(type(uploaded_file))

df=pd.read_csv(uploaded_file) 
# st.write("Please upload file to the application.")  
	
		
st.title('Dataset Description')
check_box = st.sidebar.checkbox(label="Display dataset") 
if check_box:
	st.write(df) 







options = st.sidebar.multiselect(
    'Please select the actions ',
    ['Dataset Description', 'Data Visualization','Data Preprocessing'],)


#st.write('You selected:', options)




 
try:
 features = list(df.select_dtypes(['float', 'int']).columns)
 #st.write("Features :",features)

 
 
 target = list(df.select_dtypes(['object']).columns)  
# st.write(target) 
# st.write("Output Label:",target)
 
 target.append(None) 
 
 #unique_target =target.unique()  
 
 #st.write(target)
 #print(target)
 #st.write('Shape of dataset:', features.shape)
 #st.write('number of classes:', len(np.unique(target)))
 
except Exception as e:
 print(e)
 st.write("Please upload file to the application.")  
 



features = list(df.select_dtypes(['float', 'int']))

X=df[features].to_numpy() 

print("Value of X",X)


print("Shape of X ",X.shape)



target = list(df.select_dtypes(['object']).columns)


y=df[target].to_numpy() 

label=np.unique(y)

print(y)

st.write("Input DataFrame shape",X.shape) 
st.write("Features",df.columns) 
st.write("Output Target",label.shape)
st.write("Output Target",label)




 
 
#st.write('Shape of dataset:', features.shape)
#st.write('number of classes:', len(np.unique(target)))

 
 
 # 1) Data Pre-Processing 


# Add a sidebar
st.sidebar.subheader("Preprocessing the data")

feature_selection = st.sidebar.multiselect(label="Features to plot", options=features) 



target_dropdown = st.sidebar.selectbox(label="Target_Variable",options=target) 




st.write("Seleted_Features",feature_selection)   
st.write("Seleted_target : ",target_dropdown)   



# Time series Line plot of the data 



#st.write("Seleted_Features",feature_selection)  


# 2) Data Visualization    

st.title('Data Visualization')


df1 = df[df['species']==target_dropdown]
df_features = df[feature_selection]  


plotly_figure = px.line(data_frame=df_features,
                        x=df_features.index,y=feature_selection,
                        title=('Individual Data and values Plots')
                        ) 
                        
st.plotly_chart(plotly_figure)  


                


# Add a select widget to the side bar

chart_select = st.sidebar.selectbox( label="Select the chart type", options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot'] )  


if chart_select == 'Scatterplots':
    st.sidebar.subheader("Scatterplot Settings")
    try:
        x_values = st.sidebar.selectbox('X axis', options=features)
        y_values = st.sidebar.selectbox('Y axis', options=features)
        color_value = st.sidebar.selectbox("Color", options=target)
        plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
        # display the chart
        st.plotly_chart(plot)
    except Exception as e:
        print(e)

if chart_select == 'Lineplots':
    st.sidebar.subheader("Line Plot Settings")
    try:
        x_values = st.sidebar.selectbox('X axis', options=features)
        y_values = st.sidebar.selectbox('Y axis', options=features)
        color_value = st.sidebar.selectbox("Color", options=target)
        plot = px.line(data_frame=df, x=x_values, y=y_values, color=color_value)
        st.plotly_chart(plot)
    except Exception as e:
        print(e)

if chart_select == 'Histogram':
    st.sidebar.subheader("Histogram Settings")
    try:
        x = st.sidebar.selectbox('Feature', options=features)
        bin_size = st.sidebar.slider("Number of Bins", min_value=10,
                                     max_value=100, value=40)
        color_value = st.sidebar.selectbox("Color", options=target)
        plot = px.histogram(x=x, data_frame=df, color=color_value)
        st.plotly_chart(plot)
    except Exception as e:
        print(e)

if chart_select == 'Boxplot':
    st.sidebar.subheader("Boxplot Settings")
    try:
        y = st.sidebar.selectbox("Y axis", options=features)
        x = st.sidebar.selectbox("X axis", options=features)
        color_value = st.sidebar.selectbox("Color", options=target)
        plot = px.box(data_frame=df, y=y, x=x, color=color_value)
        st.plotly_chart(plot)
    except Exception as e:
        print(e)
        
        


# 3) Select the Machine Learning Algorithm for the dataset.







data_dim = st.sidebar.radio('Select the type of Learning',('Supervised Learning','Unsupervised Learning')) 



if data_dim == 'Supervised Learning': 
	st.title("Training with Classifier Algo")
	st.text("Select the Supervised ML Algo") 
	X=df.iloc[:,0:len(df.columns)-1].to_numpy() 
	st.write("Your features/Input data is",X)
	y=df.iloc[:,-1].to_numpy() 
	st.write("Your Labeled data is",y) 
	
	
	
	
	
if data_dim == 'Unsupervised Learning':
	st.title("Training with Clustering Algo")

	st.text("Select the Unsupervised ML Algo")
	X=df.iloc[:,0:len(df.columns)-1].to_numpy() 
	st.write("Your features/Input data is",X) 

	



st.title('Training with Regression Algo')



classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)



# Function to select the parameters for classifers

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params 
    

params = add_parameter_ui(classifier_name)

st.write(params)


# Apply the  Classifier for the elected parameters

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)





clustering_name = st.sidebar.selectbox(
    'Select Clustering Algorithm',
    ('K-means', 'DBSCAN', 'Gaussian Mixture Model','Agglomerative Hierarchcal Clustering','BIRCH algorithm','Mini-batch k-means clustering','Mean shift clustering','Optics clustering','Spectral clustering','Gaussian mixture clustering','') ) 



# Separating the Features and target variables from dataset for training 


features = list(df.select_dtypes(['float', 'int']))

X=df[features].to_numpy() 

target = list(df.select_dtypes(['object']).columns)   

y=df[target].to_numpy()


# Splitting the data training and testing 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1234)  


clf.fit(X_train,y_train)

y_pred=clf.predict(X_test) 

acc = accuracy_score(y_test, y_pred) 


st.write(f'Classifier = {clf}')
st.write(f'Accuracy =', acc) 



# Function to select the parameters for Clustering  

#st.title('Training with Clustering Algo')

# Clustering Algorithm :- K-means clustering

from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.cluster import Birch




def add_parameter_clust(clf_name):
    params1 = dict()
    
    if clf_name == 'K-means':
        par1 = st.sidebar.slider('clusters', 1,5)
        params1['par1'] = par1
        
    elif clf_name == 'DBSCAN':
        par1 = st.sidebar.slider('eps', 0.01,0.90)
        params1['par1'] = par1
        par2 = st.sidebar.slider('min_samples',5,len(X)) 
        
        params1['par2'] = par2
    elif clf_name == 'BIRCH':
    	par1 = st.sidebar.slider('threshold', 0.01,0.90)
    	params1['par1'] = par1
    	par2 = st.sidebar.slider('par1',5,len(X)) 
    	params1['par2'] = par2
 
    return params1 
    

params1 = add_parameter_clust(clustering_name)

#st.write(params1)


# Apply the  Clustering for the elected parameters

def get_cluster(clf_name, params):
    clu = None
    
    if clf_name == 'K-means':
        clu = KMeans(params1['par1']) 
        print(params["par1"])
        #st.write(clu)
        
    elif clf_name == 'DBSCAN':
    	clu = DBSCAN(params['par1'],params['par2'])  
    	
    elif clf_name == 'BIRCH':
    	clu = Birch(params['par1'],params['par2'])  
    
    return clu
    	
        

clu = get_cluster(clustering_name, params1)

# Fitting a Clustering Algorithm 



y_kmeans=clu.fit_predict(X) 

print(y_kmeans)






# st.title("Visualising the clusters") 


#Visualising the clusters




















# 3d scatterplot using matplotlib

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(clu.cluster_centers_[:, 0], clu.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.show()
















# retrieve unique clusters
#clusters = unique(y_test)


# create scatter plot for samples from each cluster


#for cluster in clusters:
 # get row indexes for samples with this cluster 
 
# row_ix = where(yhat == cluster)
 # create scatter of these samples
# pyplot.scatter(X_train[row_ix, 0], X_train[row_ix, 1])
# show the plot
# pyplot.show()







regressor_name = st.sidebar.selectbox(
    'Select Regressor',
    ('Linear Regression','Decision Tree','Support Vector Regression','Lasso Regression','Random Forest') 
    
)  






































 
 
 
 
 
