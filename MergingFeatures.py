import collections
import gdown
import pandas as pd
import os

# load data
gdown.download(id='1GDoSIIggxk8GlaZVVigIgZqpi1iV9MXq')
importlist = [('RawData', 'county_complete.csv', 0), ('RawData', 'County_Transportation_Profiles.csv', 0),
              ('RawData', 'state_fips.csv', 0), ('RawData', 'age_above_65_state.csv', 0),
              ('RawData', 'health_insurance.csv', 0), ('RawData', 'poverty_rate.csv', 0),
              ('RawData', 'State_population.csv', 0), ('RawData', 'countypres_2000-2020.csv', 0),
              ('PreprocessedData', 'centrality.csv', 0), ('PreprocessedData', 'betweeness.csv', 0),
              ('PreprocessedData', 'pagerank.csv', 0), ('PreprocessedData', 'degree_commuting_between_states.csv', 0),
              ('RawData', 'tas_timeseries_monthly_cru_1901-2020_USA.csv', 2),
              ('RawData', 'tas_timeseries_annual_cru_1901-2020_USA.csv', 1),
              ('', 'vaccine_per_county.csv', 0), ('PreprocessedData', 'measures.csv', 0)]
df_list = []
for importdata in importlist:
    directory = os.path.join(importdata[0], importdata[1])
    df_list.append(pd.read_csv(directory, header=importdata[2], encoding='utf-8'))
df_list.append(pd.read_excel(os.path.join('RawData', 'LND01.xls'), header=0, usecols="A,B,X"))
df_list.append(pd.read_excel(os.path.join('RawData', 'lifeexpectancy_county.xlsx'), header=1, nrows=3194))

periods = [pd.Timestamp("2020-06-20"), pd.Timestamp("2020-12-20"), pd.Timestamp("2021-06-20"),
           pd.Timestamp("2021-12-20"), pd.Timestamp("2022-04-11")]


# define functions for merging
def prepareMerging(nr, keepcols, renaming):
    df = df_list[nr][keepcols].copy()
    df.rename(columns=renaming, inplace=True)
    return df


def doMerging(alldata, mergedf, onvar):
    alldata = alldata.merge(mergedf, how='left', on=onvar, indicator=True)
    print(f"For {mergedf.columns}: {collections.Counter(alldata['_merge'].tolist())}")
    if 'left_only' in alldata['_merge'].tolist():
        print(alldata[alldata['_merge'] == 'left_only'])
    alldata.drop(columns="_merge", inplace=True)
    return alldata


# social demographic data
### select 2019 data (most recent)
alldata = pd.concat([df_list[0].loc[:, ["fips", "state", "name"]],
                     df_list[0].loc[:, df_list[0].columns.str.endswith('2019')]], axis=1)

### select among 2019 features
keepcols = ["fips", "state", "name",
            "pop_2019", "median_age_2019", "housing_one_unit_structures_2019",
            "households_speak_limited_english_2019", "median_household_income_2019", "median_individual_income_2019",
            "unemployment_rate_2019", "uninsured_2019", "household_has_smartphone_2019", "persons_per_household_2019",
            "hs_grad_2019", "bachelors_2019",
            "white_2019", "black_2019", "native_2019", "asian_2019", "hispanic_2019"]
alldata = alldata[keepcols]
# poverty variables excluded because they have a lot of missings


# land area
### keep county data (drop state & US-wide data)
landdata = df_list[16][(df_list[16]["STCOU"] >= 1000) & (~df_list[16]["STCOU"].astype(str).str.endswith("000"))].copy()

### select features (LND110210D in column X refers to most recent 2010 data)
renaming = {"Areaname": "County", "STCOU": "fips", "LND110210D": "Land area"}
landdata = landdata.rename(columns=renaming).drop(columns="County")

alldata = doMerging(alldata, landdata, "fips")

### compute population density
alldata["pop_density"] = alldata['pop_2019'] / alldata['Land area']
alldata['pop_density'] = alldata['pop_density'].fillna(alldata.groupby('state')['pop_density'].transform('mean'))


# life expectancy
### keep county data (drop state & US-wide data)
lifeexpdata = df_list[17][(df_list[17]["FIPS"] >= 1000) & (df_list[17]["FIPS"].notna())].copy()

### select and preprocess features (most recent data on life expectancy)
lifeexpdata["Life expectancy"] = lifeexpdata["Life expectancy, 2014*"].apply(lambda x: float(x[0:5]))
lifeexpdata["fips"] = lifeexpdata["FIPS"].astype('int')

keepcols = ["fips", "Life expectancy"]
alldata = doMerging(alldata, lifeexpdata[keepcols], "fips")


# county transportation profile
keepcols = ["County FIPS", "Primary and Commercial Airports"]
renaming = {"County FIPS": "fips", "Primary and Commercial Airports": "Airports"}
transportdata = prepareMerging(1, keepcols, renaming)
alldata = doMerging(alldata, transportdata, "fips")
alldata['Airports'] = alldata['Airports'].fillna(alldata.groupby('state')['Airports'].transform('mean'))


# state FIPS
keepcols = ['Name', 'Postal Code', 'fips']
renaming = {'Name': 'state', 'Postal Code': 'state abbreviation', 'fips': 'state fips'}
statefipsdata = prepareMerging(2, keepcols, renaming)
alldata = doMerging(alldata, statefipsdata, "state")
# DC missing


# age above 65: percentage of state population
keepcols = ["State", "Population Ages 65+ (percent of state population)"]
renaming = {'State': 'state', 'Population Ages 65+ (percent of state population)': 'Percent >65yrs per state'}
agedata = prepareMerging(3, keepcols, renaming)
alldata = doMerging(alldata, agedata, "state")
# DC missing


# health insurance: group percentages of state population
keepcols = ["Location", "Uninsured"]
renaming = {"Location": "state", "Uninsured": "Insured per state: uninsured"}
insurancedata = prepareMerging(4, keepcols, renaming)
alldata = doMerging(alldata, insurancedata, "state")


# poverty rate: percentage per state
keepcols = ['State', 'PovertyRate']
renaming = {'State': 'state'}
povertydata = prepareMerging(5, keepcols, renaming)
alldata = doMerging(alldata, povertydata, "state")
# DC missing


# state population: growthSince2010, percent, density
keepcols = ['State', 'Pop', 'growthSince2010', 'Percent', 'density']
renaming = {'State': 'state', 'Pop': 'state population', 'growthSince2010': 'State population growth since 2010',
            'Percent': 'State population percent', 'density': 'State population density'}
populationdata = prepareMerging(6, keepcols, renaming)
alldata = doMerging(alldata, populationdata, 'state')


# voting
def convertFIPS(fipscol, namecol):
    if pd.isna(fipscol):
        try:
            result = alldata[alldata['name'].str.lower() == namecol.lower()]['fips'].tolist()[0]
            print(f"isna: return {result} for {namecol}")
            return result
        except:
            print(f"isna: no return for {namecol}")
            return pd.NA
    else:
        return int(fipscol)


### select voting data for 2020 presidential elections
votingdata = df_list[7].loc[(df_list[7]['year'] == 2020) & (df_list[7]['office'] == "US PRESIDENT")].copy()

### compute voting percentages (instead of absolute counts per party)
votingdata['percent_votes'] = votingdata['candidatevotes'] / votingdata['totalvotes']

### convert FIPS format to common FIPS format for merging
### search for FIPS in alldata (based on county name) if FIPS is missing in votingdata
votingdata['fips'] = votingdata.apply(lambda x: convertFIPS(x['county_fips'], x['county_name']), axis=1)

### reshape data from long to wide format
keepcols = ['fips', 'party', 'percent_votes']
votingdata = pd.pivot_table(votingdata[keepcols], index='fips', columns='party', values='percent_votes')

renaming = {'DEMOCRAT': 'Vote Democrat', 'REPUBLICAN': 'Vote Republican'}
keepcols = ['Vote Democrat', 'Vote Republican']
votingdata = votingdata.rename(columns=renaming)[keepcols]
alldata = doMerging(alldata, votingdata, 'fips')

### impute missing values with state-wide mean
for col in keepcols:
    alldata[col] = alldata[col].fillna(alldata.groupby('state')[col].transform('mean'))


# graph centrality
keepcols = ['geoid', 'centralScore']
renaming = {'geoid': 'fips', 'centralScore': 'centrality score'}
centralitydata = prepareMerging(8, keepcols, renaming)
alldata = doMerging(alldata, centralitydata, "fips")


# graph betweenness
keepcols = ['geoid', 'betweenScore']
renaming = {'geoid': 'fips', 'betweenScore': 'betweenness score'}
betweendata = prepareMerging(9, keepcols, renaming)
alldata = doMerging(alldata, betweendata, "fips")


# graph PageRank
keepcols = ['geoid', 'pageRank']
renaming = {'geoid': 'fips', 'pageRank': 'pagerank score'}
pagerankdata = prepareMerging(10, keepcols, renaming)
alldata = doMerging(alldata, pagerankdata, "fips")


# commuting between states
keepcols = ['Place', 'Degree_Centrality_com']
renaming = {'Place': 'state', 'Degree_Centrality_com': 'degree centrality state'}
comstatedata = prepareMerging(11, keepcols, renaming)
alldata = doMerging(alldata, comstatedata, "state")


# mean temperature
### average per period (based on most recent 2020 data)
seasondata = df_list[12][df_list[12]['Unnamed: 0'] == 2020].copy()
seasondata['first'] = seasondata[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']].mean(axis=1)
seasondata['second'] = seasondata[['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].mean(axis=1)

for i in range(1, 6):
    newcol = 'Mean_temperature_period' + str(i)
    if (i % 2) == 0:
        alldata[newcol] = seasondata['first'].values[0]
    else:
        alldata[newcol] = seasondata['second'].values[0]

### per state (based on most recent 2020 data)
renaming = {119: 'Mean_temperature_perstate', 'index': 'state'}
weatherdata = df_list[13][df_list[13]['Unnamed: 0'] == 2020].transpose()\
    .drop(index=['Unnamed: 0', 'United States']).reset_index().rename(columns=renaming)
alldata = doMerging(alldata, weatherdata, 'state')


# vaccination data
vaccinedata = df_list[14].copy()
vaccinedata['datetime'] = pd.to_datetime(vaccinedata['Date'], format="%Y-%m-%d")

### average over the whole time
vaccinedata_agg = vaccinedata.groupby(['State', 'Recip_County']).agg(
    {'Series_Complete_Yes': 'mean', 'Booster_Doses': 'mean'})
renaming = {'Series_Complete_Yes': 'Complete_all', 'Booster_Doses': 'Booster_all', 'State': 'state',
            'Recip_County': 'name'}
vaccinedata_agg = vaccinedata_agg.reset_index().rename(columns=renaming)

### average per period
vaccinedata_periods = []
vaccinedata_periods_agg = []
for i in range(0, 5):
    if i == 0:
        vaccinedata_periods.append(vaccinedata[vaccinedata['datetime'] <= periods[i]])
    else:
        vaccinedata_periods.append(vaccinedata[(vaccinedata['datetime'] <= periods[i]) &
                                               (vaccinedata['datetime'] > periods[i-1])])
    vaccinedata_periods_agg.append(vaccinedata_periods[i].groupby(['State', 'Recip_County'])\
        .agg({'Series_Complete_Yes': 'mean', 'Booster_Doses': 'mean'}))
    renaming = {'Series_Complete_Yes': f'Complete_period{i+1}', 'Booster_Doses': f'Booster_period{i+1}',
                'State': 'state', 'Recip_County': 'name'}
    vaccinedata_periods_agg[i] = vaccinedata_periods_agg[i].reset_index().rename(columns=renaming)

### merge with average over whole period
for i in range(0, 5):
    vaccinedata_agg = doMerging(vaccinedata_agg, vaccinedata_periods_agg[i], ['state', 'name'])

alldata = doMerging(alldata, vaccinedata_agg, ['state', 'name'])

### impute missing values with state-wide mean
### compute vaccination rate by county population
vaccine_col = [col for col in alldata if col.startswith(('Complete', 'Booster'))]
for col in vaccine_col:
    alldata[col] = alldata[col].fillna(alldata.groupby('state')[col].transform('mean'))
    newcol = str(col) + "_rate"
    alldata[newcol] = alldata[col] / alldata['pop_2019']

### no vaccination data for period1 available
for col in ['Complete_period1', 'Booster_period1', 'Complete_period1_rate', 'Booster_period1_rate']:
    alldata[col].fillna(0, inplace=True)


# Covid measures
policydata = df_list[15].copy()
policydata['datetime'] = pd.to_datetime(policydata['Date'], format="%Y-%m-%d")
policydata = policydata[policydata['datetime'] <= pd.Timestamp("2022-04-11")]

### define function to compute weighted mean per period, depending on how long measures were in place
### define function to retrieve last change if no policy change in period 5
def weightedMeans(x, endofPeriod):
    x['duration'] = x['datetime'].diff().dt.days.shift(-1, axis=0)
    x['duration'].fillna((endofPeriod - x['datetime']).dt.days, inplace=True)
    d = {'Vaccination': sum(x['duration'] * x['Vaccination']) / x['duration'].sum(),
         'Masks': sum(x['duration'] * x['Masks']) / x['duration'].sum(),
         'Close_schools': sum(x['duration'] * x['Close_schools']) / x['duration'].sum()}
    return pd.Series(d, index=['Vaccination', 'Masks', 'Close_schools'])


def fillNoChange(col, fipscol, originalcol):
    if pd.isna(col):
        group_max = policydata[policydata['fips'] == fipscol]['datetime'].max()
        prior_change = policydata.loc[(policydata['fips'] == fipscol) & (policydata['datetime'] == group_max),
                                      originalcol].values[0]
        return prior_change
    else:
        return col


### weighted average over whole time
policydata_agg = policydata.groupby('fips').apply(weightedMeans, periods[4])
renaming = {'Vaccination': 'Vaccination_all', 'Masks': 'Masks_all', 'Close_schools': 'Close_schools_all',
            'fips': 'state fips'}
policydata_agg = policydata_agg.reset_index().rename(columns=renaming)

### weighted average per period
policydata_periods = []
policydata_periods_agg = []
for i in range(0, 5):
    if i == 0:
        policydata_periods.append(policydata[policydata['datetime'] <= periods[i]])
    else:
        policydata_periods.append(policydata[(policydata['datetime'] <= periods[i]) &
                                             (policydata['datetime'] > periods[i - 1])])
    policydata_periods_agg.append(policydata_periods[i].groupby('fips').apply(weightedMeans, periods[i]))
    renaming = {'Vaccination': f'Vaccination_period{i + 1}', 'Masks': f'Masks_period{i + 1}',
                'Close_schools': f'Close_schools_period{i + 1}', 'fips': 'state fips'}
    policydata_periods_agg[i] = policydata_periods_agg[i].reset_index().rename(columns=renaming)

### merge periods
for i in range(0, 5):
    policydata_agg = doMerging(policydata_agg, policydata_periods_agg[i], 'state fips')

### fill in where no changes occurred in the last period
for col in policydata_agg:
    if col.startswith('Vaccination'):
        originalcol = 'Vaccination'
    elif col.startswith('Masks'):
        originalcol = 'Masks'
    else:
        originalcol = 'Close_schools'
    policydata_agg[col] = policydata_agg.apply(lambda x: fillNoChange(x[col], x['state fips'], originalcol), axis=1)

alldata = doMerging(alldata, policydata_agg, 'state fips')
# missing for DC


# export
for col in alldata.columns:
    print(f"missings for {col}: {sum(alldata[col].isna())}")

alldata.to_csv(os.path.join("PreprocessedData", "mergedata.csv"), sep=",", header=True, index=False)



