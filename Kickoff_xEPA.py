import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import joblib
from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupKFold, cross_val_predict, cross_val_score
from sklearn import metrics
from functools import partial
import pyarrow
from pygam import LinearGAM, s, l

plt.rcParams['figure.facecolor'] = 'white'

pff = pd.read_csv('PFFScoutingData.csv')
epa = pd.read_csv('epa.csv')
epa = epa.rename(columns={'old_game_id':'gameId', 'play_id':'playId'})
plays = pd.read_csv('plays.csv')
pl = pd.read_csv('players.csv')

df = epa.merge(plays, on=['gameId', 'playId'], how='left')

kickoff = df.loc[df.specialTeamsPlayType == 'Kickoff']

kickoff = kickoff.merge(pff[['gameId', 'playId', 'hangTime', 'kickType']])
kickoff = kickoff.loc[kickoff.kickType.isin(['D', 'F', 'P', 'B'])]

tr18 = pd.read_csv('tracking2018.csv')
tr19 = pd.read_csv('tracking2019.csv')
tr20 = pd.read_csv('tracking2020.csv')
tr = pd.concat([tr18, tr19, tr20])

events = ['ball_snap', 'kickoff', 'punt', 'touchback', 'kick_received', 'punt_received', 'fair_catch', 
          'punt_land', 'out_of_bounds', 'punt_downed', 'kickoff_land', 'punt_muffed', 'onside_kick', 
          'kick_recovered', 'punt_blocked', 'free_kick']
ft = tr.loc[tr.nflId.isna() & tr.event.isin(events)].copy()
ft['x'] = np.where(ft.playDirection == 'right', ft.x, 120-ft.x)
ft['y'] = abs(ft.y-26.67)
ft['time'] = pd.to_datetime(ft.time)

kt = ft.merge(kickoff[['gameId', 'playId']])

cols = ['time', 'x', 'y', 'event', 'gameId', 'playId']
kte = kt.loc[kt.event.isin(['kickoff']), cols].merge(
kt.loc[kt.event.isin(['kickoff_land']), cols], how='left', on=['gameId', 'playId'], suffixes=('', '_land')
).merge(
kt.loc[kt.event.isin(['kick_received']), cols], how='left', on=['gameId', 'playId'], suffixes=('', '_rec')
).merge(
kt.loc[kt.event.isin(['touchback']), cols], how='left', on=['gameId', 'playId'], suffixes=('', '_tb')
).merge(
kt.loc[kt.event.isin(['out_of_bounds']), cols], how='left', on=['gameId', 'playId'], suffixes=('', '_oob')
)
        
kte['land_diff'] = ((kte.time_land-kte.time)/np.timedelta64(1, 's'))
kte['rec_diff'] = ((kte.time_rec-kte.time)/np.timedelta64(1, 's'))
kte['tb_diff'] = ((kte.time_tb-kte.time)/np.timedelta64(1, 's'))
kte['oob_diff'] = ((kte.time_oob-kte.time)/np.timedelta64(1, 's'))

kte['time_diff'] = np.fmin(
    np.fmin(
        np.fmin(
            kte.land_diff, kte.rec_diff), 
        kte.tb_diff), 
    kte.oob_diff)

kte['diff_type'] = np.where(kte.land_diff == kte.time_diff, 'land', 
                            np.where(kte.rec_diff == kte.time_diff, 'rec',
                                     np.where(kte.tb_diff == kte.time_diff, 'tb',
                                              np.where(kte.oob_diff == kte.time_diff, 'oob', 
                                                       np.nan))))

kte['x_end'] = np.where(kte.diff_type == 'land', kte.x_land, 
                        np.where(kte.diff_type == 'rec', kte.x_rec, 
                                 np.where(kte.diff_type == 'tb', kte.x_tb, 
                                         np.where(kte.diff_type == 'oob', kte.x_oob,
                                                 np.nan))))

kte['y_end'] = np.where(kte.diff_type == 'land', kte.y_land, 
                        np.where(kte.diff_type == 'rec', kte.y_rec, 
                                 np.where(kte.diff_type == 'tb', kte.y_tb, 
                                         np.where(kte.diff_type == 'oob', kte.y_oob,
                                                 np.nan))))

kte['time_diff'] = np.clip(kte.time_diff, 2.5, 5)

k = kte[['gameId', 'playId', 'diff_type', 'time_diff', 'x_end', 'y_end']].copy()

k = k.merge(kickoff)

k['x_end'] = np.clip(k.x_end, 0, 125)
k['x_end_net'] = k.x_end - k.yardline_100
k['hangTime'] = k.hangTime.fillna(k.time_diff)
k['kickLength'] = k.kickLength.fillna(k.x_end_net)
k['kickLength'] = np.where(k.diff_type == 'oob', k.x_end_net, k.kickLength)
k['time_diff'] = k.time_diff.fillna(k.hangTime)
k['y_end_out'] = np.clip(k.y_end, 26.67, 35)
k['y_end_clip'] = np.clip(k.y_end, 0, 26.67)
k['x_end_clip'] = np.clip(k.x_end, 0, 115)
k['x_end_ez'] = np.clip(k.x_end, 110, 125)

k_adj_yard = k.groupby(['yardline_100']).epa_new.mean().reset_index()
k = k.merge(k_adj_yard, on='yardline_100', suffixes=('', '_adj_yard'))
k['epa_new'] -= k.epa_new_adj_yard

predictors = ['time_diff', 'x_end_clip', 'x_end_ez', 'y_end_clip', 'y_end_out', 
              'yardline_100', 'x_end_net', 'hangTime']
target = 'epa_new'
other = ['hashed_game_id', 'weight']

group_kfold = GroupKFold(n_splits=5)

k = k.dropna(subset=predictors + [target])
X = k[predictors].dropna()
y = k[target]
grp = k.hashed_game_id
w = k.weight
for train_index, val_index in group_kfold.split(X, y, grp):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    grp_train, grp_val = grp.iloc[train_index], grp.iloc[val_index]
    w_train, w_val = w.iloc[train_index], w.iloc[val_index]

kick_gam = LinearGAM(s(0, constraints = 'monotonic_dec')+\
                     s(1)+\
                     s(2, constraints = 'monotonic_inc')+\
                     s(3)+\
                     s(4, constraints = 'monotonic_inc')+\
                     s(5, constraints = 'monotonic_dec')+\
                     s(6, constraints = 'monotonic_dec')+\
                     s(7, constraints = 'monotonic_dec'), n_splines=8, lam=.2)
    
kick_gam.fit(k[predictors], k[target], weights=k.weight)

pdd = k[predictors].copy()
keep_cols = ['x_end_clip', 'x_end_ez', 'yardline_100', 'x_end_net']#, 'y_end_clip', 'y_end_out']
temp_gam = {}
for col in [col for col in pdd.columns if col not in keep_cols]:
    temp_gam[col] = LinearGAM(s(0)+s(1)+s(2),#+s(3)+s(4),
                         n_splines=5, lam=1)
    temp_gam[col].fit(pdd[keep_cols], pdd[col])
    pdd[col] = temp_gam[col].predict(pdd[keep_cols])
    
kpl = k.merge(pl, left_on='kickerId', right_on='nflId')

print(kpl.groupby('kickerId').filter(lambda x: len(x) > 50).groupby('displayName').x_epa_kick.mean()\
.reset_index().sort_values(by='displayName'))

