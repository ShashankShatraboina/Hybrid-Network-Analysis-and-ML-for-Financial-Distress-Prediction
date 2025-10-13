# Libraries
from django.shortcuts import render,redirect
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from .models import User




################ Home #################
def home(request):
	return render(request,'home1.html')
def login(request):
	return render(request,'loginform.html')
def loginCheck(request):
	if request.method=="POST":
		print('printtttttttttttttttttttttttttttttttt')
		username= request.POST.get('username')
		password= request.POST.get('email')
		try:
			user_object = User.objects.get(firstname=username,password=password)
			print(user_object)
		except:
			#user_object = None
			print('hello')
		if user_object is not None:
			print('hiiiiiiii')
			request.session['useremail'] = user_object.email
			return redirect('home')
			print('hiiiiiiii')
	return render(request,'home.html')	
def logout(request):
	return render(request,'index.html')	
def reg(request):
	return render(request,'register.html')

######## SVM ######
def save(request):
	if request.method == 'POST':
		print('printtttttttttttttttttttttttttttttttt')
		print('checkkkkkkkkkkkkkkkkk')
		username= request.POST.get('username')
		password= request.POST.get('password')
		address= request.POST.get('address')
		email= request.POST.get('email')
		age= request.POST.get('age')
		gender= request.POST.get('gender')
		phone= request.POST.get('phone')
		user=User()
		user.firstname= request.POST.get('username')
		user.password= request.POST.get('password')
		user.address= request.POST.get('address')
		user.email= request.POST.get('email')
		user.age= request.POST.get('age')
		user.gender= request.POST.get('gender')
		user.phone= request.POST.get('phone')
		user.save()		
		return render(request,'loginform.html')
	return render(request,'loginform.html')	

######## SVM ######
def nvb(request):
	return render(request,'pacweb1.html')
def pac(request):
	if request.method == 'POST':
		if request.method == 'POST':
			headline1= request.POST.get('headline1')
			headline2= request.POST.get('headline2')
			headline3= request.POST.get('headline3')
			headline4= request.POST.get('headline4')
			headline5= request.POST.get('headline5')
			headline6= request.POST.get('headline6')
			headline7= request.POST.get('headline7')
			headline8= request.POST.get('headline8')
			headline9= request.POST.get('headline9')
			headline10= request.POST.get('headline10')
			headline11= request.POST.get('headline11')
			headline12= request.POST.get('headline12')

			print(headline1)
			
			
			headline1= int(headline1)
			headline2 = int(headline2)
			headline3 = int(headline3)
			headline4 = int(headline4)	
			headline5 = int(headline5)	
			headline6 = int(headline6)	
			headline7 = int(headline7)	
			headline8 = int(headline8)	
			headline9 = int(headline9)	
			headline10 = int(headline10)	
			headline11 = int(headline11)	
			headline12 = int(headline12)	

			from django.shortcuts import render
			from django.http import HttpResponse
			import pandas as pd
			import numpy as np
			import matplotlib.pyplot as plt
			from sklearn.model_selection import train_test_split
			from sklearn.feature_extraction.text import TfidfVectorizer
			import itertools
			from sklearn import metrics
			import os
			import seaborn as sns
			from sklearn.model_selection import train_test_split
			from sklearn.metrics import confusion_matrix
			df = pd.read_csv('C:/Users/Bhanu Prakash Bandi/OneDrive/Desktop/PRACTICE PROJECTS/financial_distress/fd.csv')
			df1=df.fillna(0)

			dfle = df1.copy()
			dfle
			dfle
			print(dfle)
			X = dfle[['A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24']].values
			X
			y = dfle[['LABEL']]		
			y

			#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
			#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
			#train_test separation
			X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
			atest=[[headline1,headline2,headline3,headline4,headline5,headline6,headline7,headline8,headline9,headline10,headline11,headline12]]
			#train_test separation
			X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
			linear_clf = RandomForestClassifier(n_estimators = 100, max_depth =4, random_state=42, min_samples_split = 5, oob_score = True, n_jobs = -1, max_features = "auto",criterion = 'entropy', max_leaf_nodes = 30,class_weight='balanced_subsample',min_samples_leaf = 10)
			linear_clf.fit(X_train, y_train)
			pred = linear_clf.predict(X_test)
			print('=====================================================================')
			pred1 = linear_clf.predict(atest)
			#pred2 = linear_clf.predict(atest1)
			print(pred1)
			result=''
			if pred1==0:
				result='not a financial distress'
			else:
				result='financial distress'
			print(linear_clf.score(X_train, y_train))
			d={'predictedvalue':result,'accuracy':linear_clf.score(X_train, y_train)}				 
	return render(request,'result.html',d)
def svm(request):	
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	df = pd.read_csv('C:/Users/Bhanu Prakash Bandi/OneDrive/Desktop/PRACTICE PROJECTS/financial_distress/fd.csv')
	df1=df.fillna(0)

	dfle = df1.copy()
	dfle
	dfle
	print(dfle)
	X = dfle[['A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24']].values
	X
	y = dfle[['LABEL']]		
	y

	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	linear_clf = KNeighborsClassifier()
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)		
def dec(request):	
	return render(request,'result.html')
def randomf(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	df = pd.read_csv('C:/Users/Bhanu Prakash Bandi/OneDrive/Desktop/PRACTICE PROJECTS/financial_distress/fd.csv')
	df1=df.fillna(0)

	dfle = df1.copy()
	dfle
	dfle
	print(dfle)
	X = dfle[['A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24']].values
	X
	y = dfle[['LABEL']]		
	y

	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	linear_clf = DecisionTreeClassifier()
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def mnb(request):
	return render(request,'acc1.html',d)
def graph(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	df = pd.read_csv('C:/Users/Bhanu Prakash Bandi/OneDrive/Desktop/PRACTICE PROJECTS/financial_distress/fd.csv')
	df1=df.fillna(0)

	dfle = df1.copy()
	dfle
	dfle
	print(dfle)
	X = dfle[['A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24']].values
	X
	y = dfle[['LABEL']]		
	y

	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	linear_clf = PassiveAggressiveClassifier()
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def accuracy(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	df = pd.read_csv('C:/Users/Bhanu Prakash Bandi/OneDrive/Desktop/PRACTICE PROJECTS/financial_distress/fd.csv')
	df1=df.fillna(0)

	dfle = df1.copy()
	dfle
	dfle
	print(dfle)
	X = dfle[['A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24']].values
	X
	y = dfle[['LABEL']]		
	y

	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	linear_clf = LogisticRegression()
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def accuracy1(request):
	return render(request,'index.html')	
			