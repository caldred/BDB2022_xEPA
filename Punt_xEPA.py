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
punt = df.loc[df.specialTeamsPlayType == 'Punt']

punt = punt.merge(pff[['gameId', 'playId', 'snapDetail', 'operationTime', 'hangTime', 'kickType']])
punt = punt.loc[punt.kickType.isin(['N', 'A'])].reset_index(drop=True)

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

pt = ft.merge(punt[['gameId', 'playId']])
punt_events = ['punt_muffed', 'punt_land', 'fair_catch', 'punt_received', 
               'out_of_bounds', 'touchback']
cols = ['time', 'x', 'y', 'event', 'gameId', 'playId']
pef = {}
for event in punt_events:
    pef[event] = pt.loc[pt.event.isin(['punt']), cols].merge(
    pt.loc[pt.event.isin([event]), cols], how='left', on=['gameId', 'playId'], suffixes=('', '_end'))
    pef[event]['gameId'] = pef[event]['gameId'].astype('int')
    pef[event]['playId'] = pef[event]['playId'].astype('int')
    
pte = pef['punt_muffed'].combine_first(pef['punt_land']).combine_first(pef['fair_catch']).combine_first(
    pef['punt_received']).combine_first(pef['out_of_bounds']).combine_first(pef['touchback'])

pte['time_diff'] = ((pte.time_end-pte.time)/np.timedelta64(1, 's'))
pte = pte.loc[pte.time_diff < 6]
snap = ft.loc[ft.event=='ball_snap', ['gameId', 'playId', 'time']]
pte = pte.merge(snap, on=['gameId', 'playId'], suffixes=('','_snap'))
pte['punt_time'] = (pte.time-pte.time_snap)/np.timedelta64(1, 's')

p = pte.merge(punt)

p['x_dist'] = -p.x-p.yardline_100+110

p = pd.get_dummies(p, columns=['snapDetail'])
p = p.drop(columns='snapDetail_OK')

predictors = ['x_dist', 'y', 'yardline_100', 'operationTime', 'punt_time', 'ydstogo',
              'snapDetail_<', 'snapDetail_>', 'snapDetail_H', 'snapDetail_L']
target = 'epa_new'
other = ['hashed_game_id', 'weight']
p = p.loc[(abs(p.operationTime-2.1) < 0.5) &
          (abs(p.punt_time-2.1) < 0.5) &
          (abs(p.x_dist-10) < 5)].dropna(subset=predictors)
p['y'] = np.clip(p.y, 0, 5)
punt_gam1 = LinearGAM(s(0, constraints='monotonic_dec', n_splines=7, lam=1)+\
                      s(1, constraints='monotonic_inc', n_splines=7, lam=1)+\
                      s(2, constraints='monotonic_inc', n_splines=7, lam=1)+\
                      s(3, constraints='monotonic_inc', n_splines=7, lam=1)+\
                      s(4, constraints='monotonic_dec', n_splines=7, lam=1)+\
                      s(5, constraints='monotonic_inc', n_splines=7, lam=1)+\
                      l(6)+l(7)+l(8)+l(9))
punt_gam1.fit(p[predictors], p[target])

p['pre_xepa'] = punt_gam1.predict(p[predictors])

plt.scatter(p.yardline_100, p.ydstogo, c=np.clip(p.pre_xepa, -1.8, 0.1), s=100)
plt.colorbar()
plt.xlabel('Yardline', fontsize=11)
plt.ylabel('Yards To Go', fontsize=11)
plt.title('Pre-Punt EPA', fontsize=16)
plt.xlim(40, 100)
plt.ylim(0, 20)
plt.show()

p['x_diff'] = p.yardline_100+p.x_end-110
p['x_end_rz'] = np.where(p.x_end < 100, np.clip(p.x_end, 95, 100), np.clip(p.x_end, 100, 115))
p['x_end_ez'] = np.where(p.x_end < 110, 105, np.clip(p.x_end, 110, 125))
p['x_end_clip'] = np.clip(p.x_end, 0, 105)
p['y_end_in'] = np.clip(p.y_end, 0, 28)
p['y_end_out'] = np.clip(p.y_end, 23, 40)

predictors = ['x_end_clip', 'x_end_rz', 'x_end_ez', 'y_end_in', 'y_end_out', 
              'yardline_100', 'x_diff', 'hangTime', 'time_diff']
target = 'epa_new'
other = ['hashed_game_id', 'weight']

group_kfold = GroupKFold(n_splits=5)

p = p.dropna(subset=predictors + [target])
X = p[predictors]
y = p[target]
grp = p.hashed_game_id
w = p.weight
for train_index, val_index in group_kfold.split(X, y, grp):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    grp_train, grp_val = grp.iloc[train_index], grp.iloc[val_index]
    w_train, w_val = w.iloc[train_index], w.iloc[val_index]
    
punt_gam = LinearGAM(s(0, constraints = 'monotonic_dec')+\
                     s(1, constraints = 'monotonic_dec')+\
                     s(2, constraints = 'monotonic_dec')+\
                     s(3, constraints = 'monotonic_inc')+\
                     s(4, constraints = 'monotonic_dec')+\
                     s(5, constraints = 'monotonic_dec')+\
                     s(6, constraints = 'monotonic_inc')+\
                     s(7, constraints = 'monotonic_inc')+\
                     s(8, constraints = 'monotonic_inc'), n_splines=7, lam=1)
    
punt_gam.fit(X, y)

p['x_epa_punt'] = punt_gam.predict(X)
p['net_epa_punt'] = p.x_epa_punt-p.pre_xepa

ppl = p.merge(pl, left_on='kickerId', right_on='nflId')
ppl.groupby('kickerId').filter(lambda x: len(x) > 50).groupby('displayName').net_epa_punt.sum()\
.reset_index().sort_values(by='displayName')

pdd = p[predictors].copy()
pdd['yardline_100'] = 65
keep_cols = ['x_diff', 'y_end_in', 'y_end_out', 'yardline_100']
for col in [col for col in pdd.columns if col not in keep_cols]:
    temp_gam = LinearGAM(s(0, constraints='monotonic_inc')+\
                         s(1, constraints='monotonic_inc')+\
                         s(2, constraints='monotonic_inc')+\
                         s(3, constraints='monotonic_dec'), n_splines=5, lam=1)
    temp_gam.fit(pdd[keep_cols], pdd[col])
    pdd[col] = temp_gam.predict(pdd[keep_cols])
pdd['x_end'] = (110-pdd.yardline_100)+pdd.x_diff
pdd['x_end_rz'] = np.where(pdd.x_end < 100, np.clip(pdd.x_end, 95, 100), np.clip(pdd.x_end, 100, 115))
pdd['x_end_ez'] = np.where(pdd.x_end < 110, 105, np.clip(pdd.x_end, 110, 125))
pdd['x_end_clip'] = np.clip(pdd.x_end, 0, 105)
pdd['x_punt_epa'] = punt_gam.predict(pdd[predictors])

plt.scatter(pdd.x_diff, pdd.y_end_in+pdd.y_end_out, c=np.clip(pdd.x_punt_epa, -.7, -.15), s=300)
plt.scatter(pdd.x_diff, -(pdd.y_end_in+pdd.y_end_out-53.3), c=np.clip(pdd.x_punt_epa, -.7, -.15), s=300)
plt.colorbar()
plt.scatter(pdd.x_diff, np.sqrt(59**2-pdd.x_diff**2-20*pdd.x_diff-100)+26.67, c='k', alpha=1, s=4, 
            label='49-yard punt equivalent')
plt.scatter(pdd.x_diff, -np.sqrt(59**2-pdd.x_diff**2-20*pdd.x_diff-100)+26.67, c='k', alpha=1, s=4)
plt.legend()
plt.xlim(40, 50)
plt.ylim(-4, 57.3)
plt.title('Post-Punt EPA from Own 35', fontsize=16)
plt.xlabel('Punt Length', fontsize=11)
plt.ylabel('Punt Y-coordinate', fontsize=11)
plt.show()

pdd['yardline_100'] = 55
keep_cols = ['x_diff', 'y_end_in', 'y_end_out', 'yardline_100']
for col in [col for col in pdd.columns if col not in keep_cols]:
    temp_gam = LinearGAM(s(0, constraints='monotonic_inc')+\
                         s(1, constraints='monotonic_inc')+\
                         s(2, constraints='monotonic_inc')+\
                         s(3, constraints='monotonic_dec'), n_splines=5, lam=1)
    temp_gam.fit(pdd[keep_cols], pdd[col])
    pdd[col] = temp_gam.predict(pdd[keep_cols])
pdd['x_end'] = (110-pdd.yardline_100)+pdd.x_diff
pdd['x_end_rz'] = np.where(pdd.x_end < 100, np.clip(pdd.x_end, 95, 100), np.clip(pdd.x_end, 100, 115))
pdd['x_end_ez'] = np.where(pdd.x_end < 110, 105, np.clip(pdd.x_end, 110, 125))
pdd['x_end_clip'] = np.clip(pdd.x_end, 0, 105)
pdd['x_punt_epa'] = punt_gam.predict(pdd[predictors])

plt.scatter(pdd.x_diff, pdd.y_end_in+pdd.y_end_out, c=np.clip(pdd.x_punt_epa, -.7, 0), s=200)
plt.scatter(pdd.x_diff, -(pdd.y_end_in+pdd.y_end_out-53.3), c=np.clip(pdd.x_punt_epa, -.7, 0), s=200)
plt.colorbar()
plt.scatter(pdd.x_diff, np.sqrt(59**2-pdd.x_diff**2-20*pdd.x_diff-100)+26.67, c='k', alpha=1, s=4, 
            label='49-yard punt equivalent')
plt.scatter(pdd.x_diff, -np.sqrt(59**2-pdd.x_diff**2-20*pdd.x_diff-100)+26.67, c='k', alpha=1, s=4)
plt.legend()
plt.xlim(40, 50)
plt.ylim(-4, 57.3)
plt.title('Post-Punt EPA from Own 45', fontsize=16)
plt.xlabel('Punt Length', fontsize=11)
plt.ylabel('Punt Y-coordinate', fontsize=11)
plt.show()

pdd['yardline_100'] = 45
keep_cols = ['x_diff', 'y_end_in', 'y_end_out', 'yardline_100']
for col in [col for col in pdd.columns if col not in keep_cols]:
    temp_gam = LinearGAM(s(0, constraints='monotonic_inc')+\
                         s(1, constraints='monotonic_inc')+\
                         s(2, constraints='monotonic_inc')+\
                         s(3, constraints='monotonic_dec'), n_splines=5, lam=1)
    temp_gam.fit(pdd[keep_cols], pdd[col])
    pdd[col] = temp_gam.predict(pdd[keep_cols])
pdd['x_end'] = (110-pdd.yardline_100)+pdd.x_diff
pdd['x_end_rz'] = np.where(pdd.x_end < 100, np.clip(pdd.x_end, 95, 100), np.clip(pdd.x_end, 100, 115))
pdd['x_end_ez'] = np.where(pdd.x_end < 110, 105, np.clip(pdd.x_end, 110, 125))
pdd['x_end_clip'] = np.clip(pdd.x_end, 0, 105)
pdd['x_punt_epa'] = punt_gam.predict(pdd[predictors])

plt.scatter(pdd.x_diff, pdd.y_end_in+pdd.y_end_out, c=np.clip(pdd.x_punt_epa, -1.7, -1), s=300)
plt.scatter(pdd.x_diff, -(pdd.y_end_in+pdd.y_end_out-53.3), c=np.clip(pdd.x_punt_epa, -1.7, -1), s=300)
plt.colorbar()
plt.scatter(pdd.x_diff, np.sqrt(56**2-pdd.x_diff**2-20*pdd.x_diff-100)+26.67, c='k', alpha=1, s=4, 
            label='46-yard punt equivalent')
plt.scatter(pdd.x_diff, -np.sqrt(56**2-pdd.x_diff**2-20*pdd.x_diff-100)+26.67, c='k', alpha=1, s=4)
plt.legend()
plt.xlim(37, 47)
plt.ylim(-4, 57.3)
plt.title('Post-Punt EPA from Opp 45', fontsize=16)
plt.xlabel('Punt Length', fontsize=11)
plt.ylabel('Punt Y-coordinate', fontsize=11)
plt.show()




