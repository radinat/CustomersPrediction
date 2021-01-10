# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:36:32 2021

@author: user
"""
# Raiffeisen Bank Case Study
# Path-2-Digital
# MOBILE BANKING

# DEMOGRAPHICS 

# Load libraries
import pandas as pd
import seaborn as sns

# Set output display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format

# Import Data 
# Table with target variable 
mb = pd.read_csv("MB_base_flag.csv")
mb = mb.iloc[:,:3]
# Demographics
demog = pd.read_csv("demographics.csv")

# Combine with mb
mb_demog = pd.merge(mb, demog, on = ['client_id', 'month'])

mb_demog.head(100)
mb_demog.describe()
mb_demog.dtypes

# Create category column for nationality codes
mb_demog['nationality_iso_code'].unique()
mb_demog['nat_cat'] = 'x'
dgf = mb_demog['nationality_iso_code'].value_counts()
dgf = dgf.to_frame()
dgf.columns = ['Frequency']
natlist = dgf[dgf['Frequency'] > 5000]
nat_list = natlist.index.tolist()
def func_nat(x):
    if x in nat_list:
        return x
    else:
        return 'OTHERS'
mb_demog['nat_cat'] = mb_demog['nationality_iso_code'].apply( \
        func_nat).str.upper()
mb_demog['nat_cat'] = mb_demog.nat_cat.astype('category')
mb_demog.drop(['nationality_iso_code'], axis = 1, inplace = True) 


# Separate codes from towns and add them to new columns in mb_demog
# 1) Codes
city_codes = [] #create empty list
word = ''
for i in range(0, len(mb_demog['city_town'])):
    word = '' # create new empty word on every iteration
    city = mb_demog['city_town'][i]
    for x in city: # check each character in the string
        if x.isdigit():
            word = word + x
    if word != '':
        city_codes.append(word)
    else: city_codes.append(None)

mb_demog['city_codes'] = city_codes

# 2) Towns
city_names = []
word = ''
for i in range(0, len(mb_demog['city_town'])):
    word = '' # create new empty word on every iteration
    city = mb_demog['city_town'][i]
    for x in city: # check each character in the string
        if not x.isdigit():
            word = word + x
    if word != '':
        city_names.append(word)
    else: city_names.append(None)

mb_demog['city_names'] = city_names

# Remove white spaces
mb_demog['city_names'] = mb_demog['city_names'].str.strip()
mb_demog['city_codes'] = mb_demog['city_codes'].str.strip()

# Add missing names 
missing_names = pd.DataFrame(mb_demog[mb_demog['city_names'].isnull()][ \
                                    'city_codes'].unique()) 
missing_names['names'] = ''
name = ''
for i in missing_names.index:
    code = missing_names[i].values[0]
    for z in range(0, len(mb_demog)):
        print(z)
        if (mb_demog['city_codes'][z] == code):        
            name = mb_demog['city_names'][z]            
            break;
    missing_names['names'][i] = name # add name where missing

# Add the names to mb_demog 
for i in mb_demog[mb_demog['city_names'].isnull()].index:
    code = mb_demog['city_codes'][i]
    print(code)
    mb_demog['city_names'][i] = missing_names[missing_names[0] == code][ \
            'names'][0]
 

# Create new category column for cities
mb_demog['city_cat'] = ''
mb_demog['city_names'] = mb_demog['city_names'].str.upper() # uppercase to all
dgc = mb_demog['city_names'].value_counts()
dgc = dgc.to_frame()
dgc.columns = ['Frequency']
citylist = dgc[dgc['Frequency'] > 100000] 
city_list = citylist.index.tolist()
def func_city(x):
    if x in city_list:
        return x
    else:
        return 'OTHERS'

mb_demog['city_cat'] = mb_demog['city_names'].apply(func_city)
mb_demog['city_cat'] = mb_demog.city_cat.astype('category')
mb_demog.drop(['city_town', 'postal_code', 'city_names', 'city_codes'], 
              axis = 1, inplace = True) 

# Category for profit centres
mb_demog['profit_centre_code'].value_counts()
mb_demog['profit_centre_code'] = mb_demog.profit_centre_code.astype('category')

# Age
mb_demog['age'].value_counts()
mb_demog['age'].isna().sum()
mb_demog.dropna(inplace = True)

# Retain only clients with age higher than 14 years
mb_demog = mb_demog[mb_demog['age'] >= 14]

# Boxplot
sns.boxplot(x = mb_demog['FLAG_new_active_user_MB'], y = mb_demog['age'])
# Histogram
sns.histplot(mb_demog, x = 'age', bins = 87)

# Number of active/non-active users by age
age_flag_table = pd.crosstab(mb_demog.age, mb.FLAG_new_active_user_MB)
age_flag_table.reset_index(level = 0, inplace = True)
age_flag_table.columns = ['age', 'flag0', 'flag1']
sns.barplot(x = 'age', y = 'flag1', data = age_flag_table)

# Transform client_since into months_since_client
# Convert client_since into a datetime variable
mb_demog['client_since'] = pd.to_datetime(mb_demog['client_since'],
            format = '%Y-%m-%d')
# Create date from the month variable
mb_demog['date'] = (mb_demog.month.astype(str) + '01')
mb_demog['date'] = pd.to_datetime(mb_demog['date'], format = '%Y%m%d')
# Calculate months difference
mb_demog['months_since_client'] = 12*(mb_demog.date.dt.year - 
            mb_demog.client_since.dt.year) + (mb_demog.date.dt.month - 
                                             mb_demog.client_since.dt.month)
# Remove unnecessary columns
mb_demog = mb_demog.drop(['client_since', 'date'], axis = 1)

# Save table
mb_demog.to_csv(r'C:\Users\user\Desktop\UNI\MASTER\First Semester'
                    r'\Business Intelligence\Case Study Raiffeisen'
                    r'\Raiffeisen files\mb_demog.csv', index = False)

