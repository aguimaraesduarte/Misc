import pandas as pd
import numpy as np
import sys
from sklearn import linear_model, metrics, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor

inp = sys.argv[1]
out = sys.argv[2]

# files
file_life_exp = "life expectancy by country and year.csv"
file_gdp = "GDP by country and year.csv"
file_hiv = "HIV rates by country and year.csv"
file_income = "Income per capita by country and year.csv"
file_internet = "Internet user rates by country and year.csv"

# missing chunks of data
years_1950_1960 = ["{}".format(i) for i in range(1950,1961)]
years_2010_2016 = ["{}".format(i) for i in range(2011,2017)]
pre = pd.DataFrame(np.nan, index=range(202), columns=years_1950_1960)
pos = pd.DataFrame(np.nan, index=range(202), columns=years_2010_2016)

# read in data
life_exp = pd.read_csv(file_life_exp, header=0)
gdp = pd.read_csv(file_gdp, header=0)
hiv = pd.read_csv(file_hiv, header=0)
income = pd.read_csv(file_income, header=0)
internet = pd.read_csv(file_internet, header=0)

# country names
countries = life_exp['Country Name']

# add columns for 1950-1960 and 2010-2016
life_exp = pd.concat([pre, life_exp[life_exp.columns[1:]], pos], axis=1)
gdp = pd.concat([pre, gdp[gdp.columns[1:]], pos], axis=1)
hiv = pd.concat([pre, hiv[hiv.columns[1:]], pos], axis=1)
income = pd.concat([pre, income[income.columns[1:]], pos], axis=1)
internet = pd.concat([pre, internet[internet.columns[1:]], pos], axis=1)

# inpute missing data
life_exp = life_exp.interpolate(method='linear', axis=1).bfill(axis=1).ffill(axis=1).fillna(0)
gdp = gdp.fillna(0) #fill missing gdp with 0
hiv = hiv.interpolate(method='linear', axis=1).ffill(axis=1).fillna(0)
income = income.interpolate(method='linear', axis=1).bfill(axis=1).ffill(axis=1).fillna(0)
internet = internet.interpolate(method='linear', axis=1).ffill(axis=1).fillna(0)

# encode country labels
le = preprocessing.LabelEncoder()
le.fit(countries)
countries_encoded = le.transform(countries)
countries_encoded = pd.DataFrame(countries_encoded, columns=["Country Label"])

# put country names (encoded) back in
life_exp = pd.concat([countries_encoded, life_exp], axis=1)
gdp = pd.concat([countries_encoded, gdp], axis=1)
hiv = pd.concat([countries_encoded, hiv], axis=1)
income = pd.concat([countries_encoded, income], axis=1)
internet = pd.concat([countries_encoded, internet], axis=1)

# melt tables
life_exp = pd.melt(life_exp, id_vars='Country Label', var_name='Year', value_name='Life Expectancy')
gdp = pd.melt(gdp, id_vars='Country Label', var_name='Year', value_name='GDP Growth')
hiv = pd.melt(hiv, id_vars='Country Label', var_name='Year', value_name='HIV Rate')
income = pd.melt(income, id_vars='Country Label', var_name='Year', value_name='Income')
internet = pd.melt(internet, id_vars='Country Label', var_name='Year', value_name='Internet Use')

# make Year columns int64
life_exp["Year"] = pd.to_numeric(life_exp["Year"])
gdp["Year"] = pd.to_numeric(gdp["Year"])
hiv["Year"] = pd.to_numeric(hiv["Year"])
income["Year"] = pd.to_numeric(income["Year"])
internet["Year"] = pd.to_numeric(internet["Year"])

# create X, Y
X = life_exp[life_exp.columns[:2]]
X = gdp
X = pd.merge(gdp, hiv, "inner", ["Country Label","Year"])
X = pd.merge(X, income, "inner", ["Country Label","Year"])
X = pd.merge(X, internet, "inner", ["Country Label","Year"])
Y = life_exp["Life Expectancy"]

"""
# linear regression
regressor = linear_model.LinearRegression()
regressor.fit(X, Y)
cv = cross_val_score(regressor, X, Y, cv=10,
	                     scoring=metrics.make_scorer(metrics.mean_squared_error))
print "Cross-val: ", cv.mean()

# ridge regression
regressor = linear_model.Ridge(alpha = 1.0)
regressor.fit(X, Y)
cv = cross_val_score(regressor, X, Y, cv=10,
	                     scoring=metrics.make_scorer(metrics.mean_squared_error))
print "Cross-val: ", cv.mean()

# lasso regression
regressor = linear_model.Lasso(alpha = 1.0)
regressor.fit(X, Y)
cv = cross_val_score(regressor, X, Y, cv=10,
	                     scoring=metrics.make_scorer(metrics.mean_squared_error))
print "Cross-val: ", cv.mean()

# decision tree
regressor = DecisionTreeRegressor(max_depth=30)
regressor.fit(X, Y)
cv = cross_val_score(regressor, X, Y, cv=10,
	                     scoring=metrics.make_scorer(metrics.mean_squared_error))
print "Cross-val: ", cv.mean()
"""

# random forest regression
estimators = 50
regressor = RandomForestRegressor(n_estimators=estimators)
regressor.fit(X, Y)
"""
# read input file
test_df = pd.read_csv(inp, header=None, names=["Country Name","Year","GDP"])
test_countries = test_df["Country Name"]
test_countries_encoded = le.transform(test_countries)
test_countries_encoded = pd.DataFrame(test_countries_encoded, columns=["Country Label"])
test_X = pd.concat([test_countries_encoded, test_df["Year"]], axis=1)

# construct missing columns for test X
n = test_df.shape[0]
v_countries = []
v_years = []
#v_gdp = []
#v_hiv = []
#v_income = []
#v_internet = []
for i in range(n):
	country_encoded = test_X.iloc[i][0]
	year = test_X.iloc[i][1]
	v_countries.append(test_X.iloc[i][0])
	v_years.append(test_X.iloc[i][1])
	#row = X[(X["Country Label"]==country_encoded) & (X["Year"]==year)]
	#v_gdp.append(row["GDP Growth"].values[0])
	#v_hiv.append(row["HIV Rate"].values[0])
	#v_income.append(row["Income"].values[0])
	#v_internet.append(row["Internet Use"].values[0])
test_X = pd.DataFrame([v_countries, v_years]).transpose()
#test_X = pd.DataFrame([v_countries, v_years, v_gdp, v_hiv, v_income, v_internet]).transpose()
test_X.columns = X.columns
test_X["Year"] = test_X["Year"].astype(int)

# create predictions
hypotheses = regressor.predict(test_X)
pd.DataFrame(hypotheses, columns=["Prediction"]).to_csv(out, index=False, header=False)
"""
# cross-validation
cv = cross_val_score(regressor, X, Y, cv=10,
                     scoring=metrics.make_scorer(metrics.mean_squared_error))
print "Cross-val: ", cv.mean()

# all 4: 29.73868771
# only gdp: 17.9670245628
# nothing: 4.10251222792
# linear/lasso/ridge: 117.126319116
# decision tree: 6.57634318966
