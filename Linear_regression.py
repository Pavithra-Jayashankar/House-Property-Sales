import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
os.chdir("D:\Imarticus\Python_program")
os.getcwd()

test_raw = pd.read_csv("Property_Price_Test.csv")
train_raw = pd.read_csv("Property_Price_Train.csv")

test_raw['Sale_Price'] = 0
test_raw['Source'] = "Test"
train_raw['Source'] = "Train"

full_raw = pd.concat([test_raw,train_raw],axis=0)
full_raw.shape

full_raw.drop(['Id'], axis=1, inplace = True) #or full_raw = full_raw.drop(['Id'], axis = 1)
full_raw.isnull().sum()

#####  missing value imputation ######  & 

full_raw['Zoning_Class'].mode()[0]
full_raw['Lot_Extent'].median()

subset_condition = full_raw['Source'] == "Train"
half_rows = 0.5*full_raw[subset_condition].shape[0]

for col in full_raw.columns:
    total_na = full_raw.loc[subset_condition,col].isnull().sum()
    if(total_na < half_rows):
        if(full_raw[col].dtype == "object"):
            temp_mode = full_raw.loc[subset_condition,col].mode()[0]
            full_raw[col] = full_raw[col].fillna(temp_mode)
        else:
            temp_median = full_raw.loc[subset_condition,col].median()
            full_raw[col].fillna(temp_median, inplace = True)
    else:
        full_raw.drop([col], axis = 1, inplace = True)
            
sum(full_raw.isnull().sum())
full_raw.shape

###### Corelation checl #####

import seaborn as sns

corrDf = full_raw.corr()
sns.heatmap(corrDf,
            xticklabels = corrDf.columns,
            yticklabels = corrDf.columns, cmap = 'gist_earth_r')

#### Dummy variable ########

full_raw1 = pd.get_dummies(full_raw,drop_first = True)
full_raw1.shape

######### Intercept column ############

full_raw1['Intercept'] = 1 # By defaulth in python intercept column in not included therefore we are including manually to get the coefficient 
full_raw1.shape

###### Sampling #######

trainset = full_raw1[full_raw1['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
finaltest = full_raw1[full_raw1['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()

trainset.shape
finaltest.shape

from sklearn.model_selection import train_test_split
Train, Test = train_test_split(trainset,train_size = 0.8, random_state = 123) 

Train.shape
Test.shape

Train_x = Train.drop(['Sale_Price'], axis = 1).copy()
Train_y = Train['Sale_Price'].copy()
Test_x = Test.drop(['Sale_Price'], axis = 1).copy()
Test_y = Test['Sale_Price'].copy()

Train_x.shape
Train_y.shape
Test_x.shape
Test_y.shape

###### VIF check ###### 


from statsmodels.stats.outliers_influence import variance_inflation_factor

Max_VIF = 10
Train_X_Copy = Train_x.copy()
counter = 1
High_VIF_Column_Names = []


while (Max_VIF >= 10):
    
    print(counter)
    
    VIF_Df = pd.DataFrame()   
    VIF_Df['VIF'] = [variance_inflation_factor(Train_X_Copy.values, i) for i in range(Train_X_Copy.shape[1])]  
    VIF_Df['Column_Name'] = Train_X_Copy.columns
    
    Max_VIF = max(VIF_Df['VIF'])
    Temp_Column_Name = VIF_Df.loc[VIF_Df['VIF'] == Max_VIF, 'Column_Name']
    print(Temp_Column_Name, ": ", Max_VIF)
    
    if (Max_VIF >= 10): # This condition will ensure that ONLY columns having VIF lower than 10 are NOT dropped
        print(Temp_Column_Name, Max_VIF)
        Train_X_Copy = Train_X_Copy.drop(Temp_Column_Name, axis = 1)    
        High_VIF_Column_Names.extend(Temp_Column_Name)
    
    counter = counter + 1

High_VIF_Column_Names.remove('Intercept')

Train_x = Train_x.drop(High_VIF_Column_Names, axis = 1)
Test_x = Test_x.drop(High_VIF_Column_Names, axis = 1)

from statsmodels.api import OLS
M1_ModelDef = OLS(Train_y, Train_x) #Or OLS(Train_y, Train_x)
M1_ModelBuild = M1_ModelDef.fit()
M1_ModelBuild.summary()


Max_Pvalue = 0.1
Train_x_Copy= Train_x.copy()
counter = 1
High_Pvalue_Column_Names = []

while(Max_Pvalue >= 0.1):
    
    print(counter)
    
    Pvalue_Df = pd.DataFrame()
    Model = OLS(Train_y,Train_x_Copy).fit()
    Pvalue_Df['P_Value']= Model.pvalues
    Pvalue_Df['Column_Names'] = Train_x_Copy.columns
    
    Max_Pvalue = max(Pvalue_Df['P_Value'])
    Temp_Column_Names = Pvalue_Df.loc[Pvalue_Df['P_Value'] == Max_Pvalue, 'Column_Names']
    print(Max_Pvalue)
    
    if(Max_Pvalue >= 0.1):
        Train_x_Copy = Train_x_Copy.drop(Temp_Column_Names, axis =1)
        High_Pvalue_Column_Names.extend(Temp_Column_Names)
        
        counter = counter+1
    
Model.summary()

Test_Pred = M1_ModelBuild.predict(Test_x)

import numpy as np

np.sqrt(np.mean((Test_y - Test_Pred)**2))

(np.mean(np.abs(((Test_y - Test_Pred)/Test_y))))*100
    