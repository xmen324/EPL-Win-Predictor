import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


##########################################################################################################


# Gets cumulative match results arranged by teams and matchweek
# Gets cumulative match results arranged by teams and matchweek
def cum_results(ds):
    # Create a dictionary with team names as keys
    unique_teams = list(np.sort(ds['HomeTeam'].unique()))
    matchweeks = int(len(ds)/10)

    wins_dict = {}
    draws_dict = {}
    loss_dict = {}
    points_dict = {}

    for i in unique_teams:
        wins_dict[i] = []
        draws_dict[i] = []
        loss_dict[i] = []
        points_dict[i] = []

    # Create new columns for home wins and away wins for each fixture
    ds['HomeWins'] = np.where(ds['FTR'] == 'H', 1, 0)
    ds['AwayWins'] = np.where(ds['FTR'] == 'A', 1, 0)

    # Create new columns for home draws and away draws for each fixture
    ds['HomeDraws'] = np.where(ds['FTR'] == 'D', 1, 0)
    ds['AwayDraws'] = np.where(ds['FTR'] == 'D', 1, 0)

    # Create new columns for home losses and away losses for each fixture
    ds['HomeLosses'] = np.where(ds['FTR'] == 'A', 1, 0)
    ds['AwayLosses'] = np.where(ds['FTR'] == 'H', 1, 0)

    # Create new columns for homepoints and awaypoints for each fixture
    ds['HomePoints'] = np.where(ds['FTR'] == 'H', 3, np.where(ds['FTR'] == 'A', 0, 1))
    ds['AwayPoints'] = np.where(ds['FTR'] == 'A', 3, np.where(ds['FTR'] == 'H', 0, 1))
    
    # the value corresponding to keys is a list containing the match location.
    for i in range(len(ds)):
        HW = ds.iloc[i]['HomeWins']
        AW = ds.iloc[i]['AwayWins']
        HD = ds.iloc[i]['HomeDraws']
        AD = ds.iloc[i]['AwayDraws']
        HL = ds.iloc[i]['HomeLosses']
        AL = ds.iloc[i]['AwayLosses']
        HP = ds.iloc[i]['HomePoints']
        AP = ds.iloc[i]['AwayPoints']

        wins_dict[ds.iloc[i].HomeTeam].append(HW)
        wins_dict[ds.iloc[i].AwayTeam].append(AW)
        draws_dict[ds.iloc[i].HomeTeam].append(HD)
        draws_dict[ds.iloc[i].AwayTeam].append(AD)
        loss_dict[ds.iloc[i].HomeTeam].append(HL)
        loss_dict[ds.iloc[i].AwayTeam].append(AL)
        points_dict[ds.iloc[i].HomeTeam].append(HP)
        points_dict[ds.iloc[i].AwayTeam].append(AP)

    # ds.drop(['HomeWins','AwayWins','HomeDraws','AwayDraws','HomeLosses','AwayLosses','HomePoints','AwayPoints'], axis=1, inplace=True)
    
    # Create a dataframe for league points where rows are teams and cols are matchweek.
    Wins = pd.DataFrame(data=wins_dict, index = [i for i in range(1,matchweeks+1)]).T
    Draws = pd.DataFrame(data=draws_dict, index = [i for i in range(1,matchweeks+1)]).T
    Loss = pd.DataFrame(data=loss_dict, index = [i for i in range(1,matchweeks+1)]).T
    Points = pd.DataFrame(data=points_dict, index = [i for i in range(1,matchweeks+1)]).T
    PrevResult = pd.DataFrame(data=points_dict, index = [i for i in range(1,matchweeks+1)]).T
    Form5M = pd.DataFrame(data=points_dict, index = [i for i in range(1,matchweeks+1)]).T

    # print(Points.head())

    Wins[0] = 0
    Draws[0] = 0
    Loss[0] = 0
    Points[0] = 0
    PrevResult[0] = 0
    Form5M[0] = 0

    # Calculate previous result and 5-match form
    for i in range(2,matchweeks+1):
        PrevResult[i] = Points[i-1]
        if i<6:
            Form5M[i] = 0
            for j in range(1,i):
                Form5M[i] = Form5M[i] + Points[j]
        else:
            Form5M[i] = Points[i-1] + Points[i-2] + Points[i-3] + Points[i-4] + Points[i-5]

    # Aggregate results upto each matchweek
    for i in range(2,matchweeks+1):
        Wins[i] = Wins[i] + Wins[i-1]
        Draws[i] = Draws[i] + Draws[i-1]
        Loss[i] = Loss[i] + Loss[i-1]
        Points[i] = Points[i] + Points[i-1]

    return Wins, Draws, Loss, Points, PrevResult, Form5M



##########################################################################################################


# Gets the cumulative goals scored, conceded and difference arranged by teams and matchweek
def cum_goalstats(ds):

    unique_teams = list(np.sort(ds['HomeTeam'].unique()))
    matchweeks = int(len(ds)/10)

    # Create dictionaries with team names as keys
    gs_dict = {}
    gc_dict = {}
    gd_dict = {}
    sf_dict = {}
    stf_dict = {}
    sc_dict = {}
    stc_dict = {}

    for i in unique_teams:
        gs_dict[i] = []
        gc_dict[i] = []
        gd_dict[i] = []
        sf_dict[i] = []
        stf_dict[i] = []
        sc_dict[i] = []
        stc_dict[i] = []

    for i in range(len(ds)):
        HTGS = ds.iloc[i]['FTHG']
        ATGS = ds.iloc[i]['FTAG']
        HTGC = ds.iloc[i]['FTAG']
        ATGC = ds.iloc[i]['FTHG']
        HTSF = ds.iloc[i]['HS']
        ATSF = ds.iloc[i]['AS']
        HTSTF = ds.iloc[i]['HST']
        ATSTF = ds.iloc[i]['AST']
        HTSC = ds.iloc[i]['AS']
        ATSC = ds.iloc[i]['HS']
        HTSTC = ds.iloc[i]['AST']
        ATSTC = ds.iloc[i]['HST']

        gs_dict[ds.iloc[i].HomeTeam].append(HTGS)
        gs_dict[ds.iloc[i].AwayTeam].append(ATGS)
        gc_dict[ds.iloc[i].HomeTeam].append(HTGC)
        gc_dict[ds.iloc[i].AwayTeam].append(ATGC)
        gd_dict[ds.iloc[i].HomeTeam].append(HTGS - HTGC)
        gd_dict[ds.iloc[i].AwayTeam].append(ATGS - ATGC)
        sf_dict[ds.iloc[i].HomeTeam].append(HTSF)
        sf_dict[ds.iloc[i].AwayTeam].append(ATSF)
        stf_dict[ds.iloc[i].HomeTeam].append(HTSTF)
        stf_dict[ds.iloc[i].AwayTeam].append(ATSTF)
        sc_dict[ds.iloc[i].HomeTeam].append(HTSC)
        sc_dict[ds.iloc[i].AwayTeam].append(ATSC)
        stc_dict[ds.iloc[i].HomeTeam].append(HTSTC)
        stc_dict[ds.iloc[i].AwayTeam].append(ATSTC)
        
    
    # Create dataframes where rows are teams and cols are matchweek.
    GoalsScored = pd.DataFrame(data=gs_dict, index = [i for i in range(1,matchweeks+1)]).T
    GoalsConceded = pd.DataFrame(data=gc_dict, index = [i for i in range(1,matchweeks+1)]).T
    GoalDifference = pd.DataFrame(data=gd_dict, index = [i for i in range(1,matchweeks+1)]).T
    ShotsFor = pd.DataFrame(data=sf_dict, index = [i for i in range(1,matchweeks+1)]).T
    ShotsTargetFor = pd.DataFrame(data=stf_dict, index = [i for i in range(1,matchweeks+1)]).T
    ShotsConceded = pd.DataFrame(data=sc_dict, index = [i for i in range(1,matchweeks+1)]).T
    ShotsTargetConceded = pd.DataFrame(data=stc_dict, index = [i for i in range(1,matchweeks+1)]).T
    GoalsScored[0] = 0
    GoalsConceded[0] = 0
    GoalDifference[0] = 0
    ShotsFor[0] = 0
    ShotsTargetFor[0] = 0
    ShotsConceded[0] = 0
    ShotsTargetConceded[0] = 0

    # Aggregate to get uptil that point
    for i in range(2,matchweeks+1):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
        GoalDifference[i] = GoalDifference[i] + GoalDifference[i-1]
        ShotsFor[i] = ShotsFor[i] + ShotsFor[i-1]
        ShotsTargetFor[i] = ShotsTargetFor[i] + ShotsTargetFor[i-1]
        ShotsConceded[i] = ShotsConceded[i] + ShotsConceded[i-1]
        ShotsTargetConceded[i] = ShotsTargetConceded[i] + ShotsTargetConceded[i-1]

    return GoalsScored, GoalsConceded, GoalDifference, ShotsFor, ShotsTargetFor, ShotsConceded, ShotsTargetConceded



##########################################################################################################


# Get the league position of each team at each matchweek
def get_league_pos(ds, p, gd, gs):
    unique_teams = list(np.sort(ds['HomeTeam'].unique()))
    matchweeks = int(len(ds)/10)
    alph_dict = dict(zip(unique_teams, range(20,0,-1)))
    alph = pd.DataFrame(data=alph_dict, index=[0]).T

    league_pos = pd.DataFrame(index=unique_teams, columns=[i for i in range(1,39)])

    # Rank teams by points, then goal difference, then goals scored, then alphabetically
    # Hack used: using weighted sum of criteria
    for i in range(1,matchweeks+1):
        league_pos[i] = 5000*p[i] + 100*gd[i] + 20*gs[i] + alph[0]

    # print(league_table[1])
    
    # Rank table values in decreasing order from 1 to 20
    league_pos[0] = 0
    for i in range(1,matchweeks+1):
        league_pos[i] = league_pos[i].rank(method='min', ascending=False).astype(int)
    
    return league_pos



##########################################################################################################


# Put together the previous functions to calculate all the stats
def get_stats(ds):
    GS, GC, GD, SF, STF, SC, STC = cum_goalstats(ds)
    W, D, L, P, PR, F5 = cum_results(ds)
    POS = get_league_pos(ds, P, GD, GS)

    j = 0
    MW = []
    
    HW = []
    AW = []
    HD = []
    AD = []
    HL = []
    AL = []
    HP = []
    AP = []

    HPR = []
    APR = []
    HF5 = []
    AF5 = []

    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []
    HTGD = []
    ATGD = []

    HTSF = []
    ATSF = []
    HTSTF = []
    ATSTF = []
    HTSC = []
    ATSC = []
    HTSTC = []
    ATSTC = []

    HPOS = []
    APOS = []

    HPR = []
    APR = []
    HF5 = []
    AF5 = []

    for i in range(len(ds)):
        ht = ds.iloc[i].HomeTeam
        at = ds.iloc[i].AwayTeam

        MW.append(j+1)

        HW.append(W.loc[ht][j])
        AW.append(W.loc[at][j])
        HD.append(D.loc[ht][j])
        AD.append(D.loc[at][j])
        HL.append(L.loc[ht][j])
        AL.append(L.loc[at][j])
        HP.append(P.loc[ht][j])
        AP.append(P.loc[at][j])

        HPR.append(PR.loc[ht][j])
        APR.append(PR.loc[at][j])
        HF5.append(F5.loc[ht][j])
        AF5.append(F5.loc[at][j])

        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        HTGD.append(GD.loc[ht][j])
        ATGD.append(GD.loc[at][j])

        HTSF.append(SF.loc[ht][j])
        ATSF.append(SF.loc[at][j])
        HTSTF.append(STF.loc[ht][j])
        ATSTF.append(STF.loc[at][j])
        HTSC.append(SC.loc[ht][j])
        ATSC.append(SC.loc[at][j])
        HTSTC.append(STC.loc[ht][j])
        ATSTC.append(STC.loc[at][j])

        HPOS.append(POS.loc[ht][j])
        APOS.append(POS.loc[at][j]) 
        
        if ((i + 1)% 10) == 0:
            j = j + 1
        
    ds['MW'] = MW

    ds['HP'] = HP
    ds['AP'] = AP
    ds['Pdiff'] = ds['HP'] - ds['AP']

    ds['HPOS'] = HPOS
    ds['APOS'] = APOS
    ds['POSdiff'] = ds['HPOS'] - ds['APOS']

    ds['HW'] = HW
    ds['AW'] = AW
    ds['HD'] = HD
    ds['AD'] = AD
    ds['HL'] = HL
    ds['AL'] = AL

    ds['HTGS'] = HTGS
    ds['ATGS'] = ATGS
    ds['HTGC'] = HTGC
    ds['ATGC'] = ATGC
    ds['HTGD'] = HTGD
    ds['ATGD'] = ATGD

    ds['HTSF'] = HTSF
    ds['ATSF'] = ATSF
    ds['HTSTF'] = HTSTF
    ds['ATSTF'] = ATSTF
    ds['HTSC'] = HTSC
    ds['ATSC'] = ATSC
    ds['HTSTC'] = HTSTC
    ds['ATSTC'] = ATSTC

    ds['HPR'] = HPR
    ds['APR'] = APR
    ds['HF5'] = HF5
    ds['AF5'] = AF5

    ds['HTHGS'] = ds.groupby(['HomeTeam'])['FTHG'].cumsum() - ds['FTHG']
    ds['ATAGS'] = ds.groupby(['AwayTeam'])['FTAG'].cumsum() - ds['FTAG']
    ds['HTHGC'] = ds.groupby(['HomeTeam'])['FTAG'].cumsum() - ds['FTAG']
    ds['ATAGC'] = ds.groupby(['AwayTeam'])['FTHG'].cumsum() - ds['FTHG']
    ds['HTHSF'] = ds.groupby(['HomeTeam'])['HS'].cumsum() - ds['HS']
    ds['ATASF'] = ds.groupby(['AwayTeam'])['AS'].cumsum() - ds['AS']
    ds['HTHSC'] = ds.groupby(['HomeTeam'])['AS'].cumsum() - ds['AS']
    ds['ATASC'] = ds.groupby(['AwayTeam'])['HS'].cumsum() - ds['HS']
    ds['HTHSTF'] = ds.groupby(['HomeTeam'])['HST'].cumsum() - ds['HST']
    ds['ATASTF'] = ds.groupby(['AwayTeam'])['AST'].cumsum() - ds['AST']
    ds['HTHSTC'] = ds.groupby(['HomeTeam'])['AST'].cumsum() - ds['AST']
    ds['ATASTC'] = ds.groupby(['AwayTeam'])['HST'].cumsum() - ds['HST']

    ds['HTHP'] = ds.groupby(['HomeTeam'])['HomePoints'].cumsum() - ds['HomePoints']
    ds['ATAP'] = ds.groupby(['AwayTeam'])['AwayPoints'].cumsum() - ds['AwayPoints']
    ds['HTHW'] = ds.groupby(['HomeTeam'])['HomeWins'].cumsum() - ds['HomeWins']
    ds['ATAW'] = ds.groupby(['AwayTeam'])['AwayWins'].cumsum() - ds['AwayWins']
    ds['HTHD'] = ds.groupby(['HomeTeam'])['HomeDraws'].cumsum() - ds['HomeDraws']
    ds['ATAD'] = ds.groupby(['AwayTeam'])['AwayDraws'].cumsum() - ds['AwayDraws']
    ds['HTHL'] = ds.groupby(['HomeTeam'])['HomeLosses'].cumsum() - ds['HomeLosses']
    ds['ATAL'] = ds.groupby(['AwayTeam'])['AwayLosses'].cumsum() - ds['AwayLosses']

    ds['avg_HGPG'] = (ds['FTHG'].cumsum() - ds['FTHG'])/(ds.index)
    ds['avg_AGPG'] = (ds['FTAG'].cumsum() - ds['FTAG'])/(ds.index)

    ds.drop(['AwayWins','HomeDraws','AwayDraws','HomeLosses','AwayLosses','HomePoints','AwayPoints'], axis=1, inplace=True)
    
    return ds



##########################################################################################################


# Normalize cumulative stats by Matchweek
def norm_mw(ds):
    cols = ['HP', 'AP', 'HW', 'AW', 'HD', 'AD', 'HL', 'AL', 'HTGS', 
            'ATGS', 'HTGC', 'ATGC', 'HTGD', 'ATGD', 'HTSF', 'ATSF', 
            'HTSTF', 'ATSTF', 'HTSC', 'ATSC', 'HTSTC', 'ATSTC']
    
    ha_cols = ['HTHGS', 'ATAGS', 'HTHGC', 'ATAGC', 'HTHSF', 'ATASF', 
               'HTHSC', 'ATASC', 'HTHSTF', 'ATASTF', 'HTHSTC', 'ATASTC', 
               'HTHP', 'ATAP', 'HTHW', 'ATAW', 'HTHD', 'ATAD', 'HTHL', 'ATAL']

    ds['MW'] = ds['MW'].astype(float)
    for col in cols:
        ds[col] /= (ds['MW']-1)

    for col in ha_cols:
        ds[col] /= (0.5*(ds['MW']-1))
    ds['MW'] = ds['MW'].astype(int)



##########################################################################################################
    

# Single function to convert any raw data to engineered data
def engg(ds):
    req_cols = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HS','AS','HST','AST']
    ds = ds[req_cols]
    stats_ds = get_stats(ds)
    engg_ds = stats_ds.iloc[50:] # Optional
    norm_mw(engg_ds) # Optional
    engg_ds.drop(req_cols, axis=1, inplace=True)
    return engg_ds



##########################################################################################################

# Uncomment if first time running the code

# # Load up and prepare the data
# raw_data = []
# raw_data.append(pd.read_csv('data/epl1314.csv'))
# raw_data.append(pd.read_csv('data/epl1415.csv'))
# raw_data.append(pd.read_csv('data/epl1516.csv'))
# raw_data.append(pd.read_csv('data/epl1617.csv'))
# raw_data.append(pd.read_csv('data/epl1718.csv'))
# raw_data.append(pd.read_csv('data/epl1819.csv'))
# raw_data.append(pd.read_csv('data/epl1920.csv'))
# raw_data.append(pd.read_csv('data/epl2021.csv'))
# raw_data.append(pd.read_csv('data/epl2122.csv'))
# raw_data.append(pd.read_csv('data/epl2223.csv'))

# data = []
# for i in range(10):
#     data.append(engg(raw_data[i]))

# # Save the data
# dataset = pd.concat(data)

# dataset.to_csv('engg_data/epl_engg_dataset.csv') # Uncomment only to save the dataset


dataset = pd.read_csv('engg_data/epl_engg_dataset.csv', index_col=0)
X_train = dataset.drop(['HomeWins'], axis=1)
y_train = dataset['HomeWins']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# for i in range(10):
#     # data[i].to_csv('engg_data/epl' + str("{:02d}".format(i)) + str("{:02d}".format(i+1)) + '.csv', index=False)

current_season = pd.read_csv('data/epl2324.csv', index_col=0)
fixtures = current_season.iloc[-10:, [0,1,2]]
fixtures.reset_index(drop=True, inplace=True)
fixtures.index += 1

X_sample = current_season.drop(['HomeTeam','AwayTeam'], axis=1)
X_sample = X_sample.iloc[-10:, :]

##########################################################################################################
    

# ONLY FOR DEVELOPMENT PURPOSES - NOT RELEVANT TO APP
    
# # Various strategies to split the data into train and test sets

# # Random split
# def random_split(data, train_split=0.8):
#     dataset = pd.concat(data)
#     X = dataset.drop(['HomeWins'], axis=1)
#     y = dataset['HomeWins']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, stratify=y, random_state=324)

#     # Standardize the data
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test


# # Chronological split - train on first n years (eg. 8), test on last 10-n years (eg. 2)
# def chrono_split(data, train_yrs=8):
#     X_train_set = []
#     X_test_set = []
#     y_train_set = []
#     y_test_set = []
#     for i in range(train_yrs):
#         X = data[i].drop(['HomeWins'], axis=1)
#         y = data[i]['HomeWins']
#         X_train_set.append(X)
#         y_train_set.append(y)
        
#     for i in range(train_yrs, 10):
#         X = data[i].drop(['HomeWins'], axis=1)
#         y = data[i]['HomeWins']
#         X_test_set.append(X)
#         y_test_set.append(y)

#     X_train = pd.concat(X_train_set)
#     y_train = pd.concat(y_train_set)
#     X_test = pd.concat(X_test_set)
#     y_test = pd.concat(y_test_set)

#     # Standardize the data
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test


# # Seasonal split - train upto a certain matchweek (eg. MW33) across all seasons, 
# #                  test on remaining (death) matches across all seasons
# def seasonal_split(data, mw=33):
#     X_train_set = []
#     X_test_set = []
#     y_train_set = []
#     y_test_set = []
#     for i in range(10):
#         X = data[i].drop(['HomeWins'], axis=1)
#         y = data[i]['HomeWins']
#         X_train_set.append(X.iloc[:(mw-6)*10, :])
#         y_train_set.append(y.iloc[:(mw-6)*10])
#         X_test_set.append(X.iloc[(mw-6)*10:, :])
#         y_test_set.append(y.iloc[(mw-6)*10:])

#     X_train = pd.concat(X_train_set)
#     X_test = pd.concat(X_test_set)
#     y_train = pd.concat(y_train_set)
#     y_test = pd.concat(y_test_set)

#     # Standardize the data
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test



# # Prediction models

# # Logistic Regression
# def log_reg(sample):
#     # Create the logistic regression model
#     model_lr = LogisticRegression()

#     # Train the model
#     model_lr.fit(X_train, y_train)

#     # Make predictions on the test set
#     y_pred_lr = model_lr.predict(sample)

#     return y_pred_lr


# # Support Vector Machine - RBF Kernel
# def svm_rbf(sample, c=0.6, deg=2):
#     # Create the SVM model
#     model_svm = SVC(kernel='rbf', C=c, degree=deg, random_state=42)

#     # Train the model
#     model_svm.fit(X_train, y_train)

#     # Make predictions on the test set
#     y_pred_svm = model_svm.predict(sample)

#     return y_pred_svm


# # Random Forest
# def randfor(sample, est=70, dep=5):
#     # Create the Random Forest model
#     model_rf = RandomForestClassifier(n_estimators=est, max_depth=dep, random_state=42, n_jobs=-1)

#     # Train the model
#     model_rf.fit(X_train, y_train)

#     # Make predictions on the test set
#     y_pred_rf = model_rf.predict(sample)

#     return y_pred_rf


# # XGBoost
# def xgboost(X_train, X_test, y_train, est=10, dep=2):
#     # Create the XGBoost model
#     model_xgb = XGBClassifier(n_estimators=est, max_depth=dep, random_state=42, n_jobs=-1)

#     # Train the model
#     model_xgb.fit(X_train, y_train)

#     # Make predictions on the test set
#     y_pred_xgb = model_xgb.predict(X_test)

#     return y_pred_xgb


# # Ensemble of all models
# def ensemble(X_train, X_test, y_train, c=0.6, deg=2, rf_est=70, rf_dep=5, xgb_est=10, xgb_dep=2):
#     y_pred_lr = log_reg(X_train, X_test, y_train)
#     y_pred_svm = svm_rbf(X_train, X_test, y_train, c, deg)
#     y_pred_rf = randfor(X_train, X_test, y_train, rf_est, rf_dep)
#     y_pred_xgb = xgboost(X_train, X_test, y_train, xgb_est, xgb_dep)

#     # Combine the predictions ('Hard' voting)
#     y_pred_maj = scipy.stats.mode([y_pred_lr, y_pred_svm, y_pred_rf, y_pred_xgb], axis=0)[0]

#     return y_pred_maj



##########################################################################################################

## ML Prediction

# Hyperparams
c=0.4
deg=2
rf_est=85
rf_dep=4
xgb_est=10
xgb_dep=2

## Prediction models

# Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Support Vector Machine - RBF Kernel
model_svm = SVC(kernel='rbf', C=c, degree=deg, random_state=42)
model_svm.fit(X_train, y_train)

# Random Forest
model_rf = RandomForestClassifier(n_estimators=rf_est, max_depth=rf_dep, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

# XGBoost
model_xgb = XGBClassifier(n_estimators=xgb_est, max_depth=xgb_dep, random_state=42, n_jobs=-1)
model_xgb.fit(X_train, y_train)


# Prediction functions

def log_reg(sample):
    # Make predictions on the test set
    pred_lr = model_lr.predict(sample)
    return pred_lr

def svm_rbf(sample, c=c, deg=deg):
    # Make predictions on the test set
    pred_svm = model_svm.predict(sample)
    return pred_svm

def randfor(sample, est=rf_est, dep=rf_dep):
    # Make predictions on the test set
    pred_rf = model_rf.predict(sample)
    return pred_rf

def xgboost(sample, est=xgb_est, dep=xgb_dep):
    # Make predictions on the test set
    pred_xgb = model_xgb.predict(sample)
    return pred_xgb

# Ensemble of all models
def ensemble(sample):
    pred_lr = log_reg(sample)
    pred_svm = svm_rbf(sample, c, deg)
    pred_rf = randfor(sample, rf_est, rf_dep)
    pred_xgb = xgboost(sample, xgb_est, xgb_dep)

    # Combine the predictions (Majority voting)
    pred_maj = stats.mode([pred_lr, pred_svm, pred_rf, pred_xgb], axis=0)[0]

    return pred_maj



##########################################################################################################


# Web app code

# Web app code

st.set_page_config(page_icon="img/crystal ball 2.png", page_title="The Pitch Prophecy", layout="centered")

st.write("""
         # ⚽ The Pitch Prophecy 🪄
         AI powered ✨ **English Premier League** win predictor! 🎯
         """)
st.write('---')

# Sidebar

st.sidebar.image("img/crystal ball 2.png")
st.sidebar.markdown("<h2 style='text-align: center;'>The Pitch Prophecy</h2>", unsafe_allow_html=True)
# st.sidebar.header('The Pitch Prophecy')
# cols = st.sidebar.columns(2)
# cols[0].sidebar.header('The Pitch Prophecy')
# cols[1].link_button('About', 'https://www.football-data.co.uk/')
st.sidebar.markdown('---')

st.sidebar.header('See Also')
st.sidebar.markdown(
    """
- [EPL Viz](https://epl-viz.streamlit.app/) 🕵🏼 \
(Visualizing 24yrs of EPL)
- [The xG Philosophy](https://xg-philosophy.streamlit.app/) 🧙🏼‍♂️ \
(EPL xG Projector)
"""
)
st.sidebar.markdown('---')

cols = st.sidebar.columns(2)
cols[0].link_button('GitHub Repo', 'https://github.com/saranggalada/EPL-Win-Predictor')
cols[1].link_button('Data Source', 'https://www.football-data.co.uk/')
st.sidebar.markdown("---\n*Copyright (c) 2024: Sarang Galada*")
# st.sidebar.link_button('Author', 'https://www.linkedin.com/in/saranggalada')

# st.sidebar.header('Menu')

# season = st.sidebar.selectbox('EPL Season', ('2023-24 season','2022-23 season','2021-22 season','2020-21 season','2019-20 season','2018-19 season','2017-18 season','2016-17 season','2015-16 season','2014-15 season','2013-14 season'))

# seasons = ['2013-14 season', '2014-15 season', '2015-16 season', '2016-17 season', '2017-18 season', '2018-19 season', '2019-20 season', '2020-21 season', '2021-22 season', '2022-23 season', '2023-24 season'] 
# sample_data = data[seasons.index(season)]
# unique_teams = list(np.sort(sample_data['HomeTeam'].unique()))

# hometeam = st.sidebar.selectbox('Home Team', tuple(unique_teams))
# awayteam = st.sidebar.selectbox('Away Team', tuple(unique_teams))

# split = st.radio('Training Mode', ('Chronological', 'Seasonal', 'Random'), horizontal=True)

modeltype = st.selectbox('Prediction Model', ('Voting Ensemble', 'Logistic Regression', 'SVM (RBF kernel)', 'Random Forest', 'XGBoost'))

# if split == 'Chronological':
#     X_train, X_test, y_train, y_test = chrono_split(data)
#     c=0.4
#     deg=2
#     rf_est=85
#     rf_dep=4
#     xgb_est=10
#     xgb_dep=2

# elif split == 'Seasonal':
#     X_train, X_test, y_train, y_test = seasonal_split(data)
#     c=0.59
#     deg=2
#     rf_est=74
#     rf_dep=5
#     xgb_est=10
#     xgb_dep=2


# else:
#     X_train, X_test, y_train, y_test = random_split(data)
#     c=0.74
#     deg=2
#     rf_est=51
#     rf_dep=6
#     xgb_est=10
#     xgb_dep=2

        

if modeltype == 'Voting Ensemble':
    pred = ensemble(X_sample)
elif modeltype == 'Logistic Regression':
    pred = log_reg(X_sample)
elif modeltype == 'SVM (RBF kernel)':
    pred = svm_rbf(X_sample)
elif modeltype == 'Random Forest':
    pred = randfor(X_sample)
else:
    pred = xgboost(X_sample)


for i in range(10):
    if pred[i] == 1:
        fixtures.loc[i+1, 'Predicted Winner'] = fixtures.loc[i+1, 'HomeTeam']
    else:
        fixtures.loc[i+1, 'Predicted Winner'] = fixtures.loc[i+1, 'AwayTeam']

fixtures = fixtures[['MW', 'HomeTeam', 'AwayTeam', 'Predicted Winner']]

st.write('### 🔴 Live Predictions')
st.dataframe(fixtures)
# st.dataframe(
#     fixtures.style.apply(
#         lambda row: ["background-color: LightGreen;" if x == row['Predicted Winner'] else "" for x in row],
#         axis=1
#     )    
# )

# def highlight(x):
#     c = f"background-color:red" 
#     #condition
#     m = x["Predicted Winner"]
#     # DataFrame of styles
#     df1 = pd.DataFrame('', index=x.index, columns=x.columns)
#     # set columns by condition
#     df1.loc[m, 'HomeTeam'] = c
#     return df1

# st.dataframe(fixtures.style.apply(highlight, axis=None))

# outcome = [' Wins!', ' doesn\'t Win :(']
# msg = hometeam + outcome[pred]