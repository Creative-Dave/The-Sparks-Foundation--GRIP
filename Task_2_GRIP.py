#!/usr/bin/env python
# coding: utf-8

# ## Import the necessary libraries
# 

# In[86]:




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in the data

# In[44]:


_data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
_data


# In[45]:


_data.shape


# Plotting the data to have a view

# In[46]:


_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[47]:


X = _data.iloc[:, :-1].values  
y = _data.iloc[:, 1].values 


# In[48]:


X


# In[72]:


linearmodel.intercept_


# In[82]:


linearmodel.coef_


# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[51]:


from sklearn.linear_model import LinearRegression  
linearmodel = LinearRegression()  
linearmodel.fit(X_train, y_train) 


# In[52]:


# Plotting the regression line
line = linearmodel.coef_*X+linearmodel.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[53]:


print(X_test) # Testing data - In Hours
y_pred = linearmodel.predict(X_test) # Predicting the scores


# In[54]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ### Testing with 'outside' data

# In[64]:


# You can also test with your own data
hours = 9.25
own_pred = linearmodel.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### To calculate the different Evaluation metrics to test our algorithm

# In[65]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import  mean_squared_error


# In[69]:


print('Mean Absolute Error:',mean_absolute_error(y_test, y_pred)) 


# In[71]:


print('Mean Squared Error:',mean_squared_error(y_test, y_pred))


# In[84]:


print('R2 Score:', r2_score(y_test, y_pred))


# In[89]:


print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

