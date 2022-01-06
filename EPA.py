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
import seaborn as sns

plt.rcParams['figure.facecolor'] = 'white'

YEARS = range(2016, 2022)

df = pd.DataFrame()

for i in YEARS:
    print(i)
    #low_memory=False eliminates a warning
    i_data = pd.read_csv('https://github.com/nflverse/nflfastR-data/blob/master/data/' \
                         'play_by_play_' + str(i) + '.csv.gz?raw=True',
                         compression='gzip', low_memory=False)

    #sort=True eliminates a warning and alphabetically sorts columns
    df = df.append(i_data)

#Give each row a unique index
df.reset_index(drop=True, inplace=True)

all_cols = [
    'season', 'old_game_id', 'play_id', 'game_half', 'qtr', 'fixed_drive', 'fixed_drive_result', 
    'order_sequence', 'posteam', 'defteam', 'home_team', 'away_team', 'posteam_score', 'defteam_score', 
    'score_differential', 'score_differential_post', 'yardline_100', 'down', 'ydstogo', 
    'half_seconds_remaining', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 
    'extra_point_result', 'two_point_conv_result', 'wp', 'epa', 'desc']

df = df[all_cols]

dr = df.groupby(['old_game_id','game_half','fixed_drive']).fixed_drive_result.agg(func=pd.Series.mode, 
                                                                                  dropna=False).reset_index()

dr1 = dr.copy()
dr1['fixed_drive'] -= 1
dr2 = dr.copy()
dr2['fixed_drive'] -= 2
dr3 = dr.copy()
dr3['fixed_drive'] -= 3
dr4 = dr.copy()
dr4['fixed_drive'] -= 4
dr5 = dr.copy()
dr5['fixed_drive'] -= 5

dr = dr.merge(dr1, on=['old_game_id', 'game_half', 'fixed_drive'], suffixes = ('', '_1'), how='left')
dr = dr.merge(dr2, on=['old_game_id', 'game_half', 'fixed_drive'], suffixes = ('', '_2'), how='left')
dr = dr.merge(dr3, on=['old_game_id', 'game_half', 'fixed_drive'], suffixes = ('', '_3'), how='left')
dr = dr.merge(dr4, on=['old_game_id', 'game_half', 'fixed_drive'], suffixes = ('', '_4'), how='left')
dr = dr.merge(dr5, on=['old_game_id', 'game_half', 'fixed_drive'], suffixes = ('', '_5'), how='left')

off = df.groupby(['season', 'posteam']).epa.mean().reset_index()
dff = df.groupby(['season', 'defteam']).epa.mean().reset_index()

df = df.merge(dr)
df = df.merge(off, on=['season', 'posteam'], suffixes=['','_off'])
df = df.merge(dff, on=['season', 'defteam'], suffixes=['','_def'])
df = df.merge(off, left_on=['season', 'defteam'], right_on=['season', 'posteam'], suffixes=['','_off2'])
df = df.merge(dff, left_on=['season', 'posteam'], right_on=['season', 'defteam'], suffixes=['','_def2'])

df = df.loc[~df.yardline_100.isna() &
               df.extra_point_result.isna() &
               df.two_point_conv_result.isna()]
df = df.sort_values(by=['old_game_id', 'game_half', 'half_seconds_remaining', 
                        'posteam_score', 'defteam_score', 'play_id'],
                    ascending = [True, True, False, True, True, True])

df['next_points'] = 0
df['finished'] = 0
end_results = ['Touchdown', 'Field goal', 'Opp touchdown', 'End of half']

df['next_points'] += np.where(df.fixed_drive_result == 'Touchdown', 7, 0)
df['next_points'] += np.where(df.fixed_drive_result == 'Field goal', 3, 0)
df['next_points'] += np.where(df.fixed_drive_result == 'Opp touchdown', -7, 0)
df['next_points'] += np.where(df.fixed_drive_result == 'Safety', -2, 0)
df['finished'] = np.where(df.fixed_drive_result.isin(end_results), 1, 0)

for i in range(5):
    df['next_points'] += np.where((df[f'fixed_drive_result_{i+1}'] == 'Touchdown') &
                                 (df.finished == 0), 7*(2*(i%2)-1), 0)
    df['next_points'] += np.where((df[f'fixed_drive_result_{i+1}'] == 'Field goal') &
                                 (df.finished == 0), 3*(2*(i%2)-1), 0)
    df['next_points'] += np.where((df[f'fixed_drive_result_{i+1}'] == 'Opp touchdown') &
                                 (df.finished == 0), -7*(2*(i%2)-1), 0)
    df['next_points'] += np.where((df[f'fixed_drive_result_{i+1}'] == 'Safety') &
                                 (df.finished == 0), -2*(2*(i%2)-1), 0)
    df['finished'] = np.where(df[f'fixed_drive_result_{i+1}'].isin(end_results) &
                              (df.finished == 0), i+2, df.finished)

df['garbage_time'] = np.where((df.wp < 0.01) | (df.wp > 0.99), 1, 0)

df['weight'] = 1#(3-(df.finished-1)//2)/3
df['ydstogo'] = np.where(df.down.isna(), np.nan, df.ydstogo)
df['yardline_100'] = np.where(df.down.isna(), (df.yardline_100-35)/2+75, df.yardline_100)
df['log_half_min_remaining'] = np.round(np.log(df.half_seconds_remaining/34.466+1))
df['hashed_game_id'] = df.old_game_id.apply(lambda x: hash(str(x)))

df = df.sort_values(by=['old_game_id', 'game_half', 'fixed_drive', 'order_sequence',
                        'half_seconds_remaining', 'play_id'],
                    ascending = [True, True, True, True, False, True])
df = df.reset_index(drop=True).reset_index()

predictors = ['yardline_100', 'down', 'ydstogo', 'log_half_min_remaining', 'epa_off', 'epa_def', 
              'posteam_timeouts_remaining', 'defteam_timeouts_remaining']
target = 'next_points'
other = ['index', 'play_id', 'old_game_id', 'hashed_game_id', 'game_half', 
         'half_seconds_remaining', 'fixed_drive_result', 'weight']

dt = df.loc[(df.finished > 0) &
            (df.garbage_time == 0), 
            other+predictors+[target]].reset_index(drop=True)

def model_eval(params, model, X, y, groups, w, scorer):
    if 'max_depth' in params:
        params['max_depth'] = int(params['max_depth'])
    if 'min_child_weight' in params:
        params['min_child_weight'] = int(params['min_child_weight'])
    clf = model(**params)
    group_kfold = GroupKFold(n_splits=5)
    score = -cross_val_score(clf, X, y, scoring=scorer, n_jobs=-1, groups=groups, 
                             fit_params={'sample_weight':w}, cv=group_kfold).mean()
    return {'loss': score,
            'status': STATUS_OK}

def tune_model(model, param_dist, scorer, loss_fn, proba,
               X_train, X_val, y_train, y_val, grp_train, grp_val, w_train, w_val, max_evals=100):
    trials = Trials()
    
    objective = partial(model_eval, 
                        model=model, 
                        X=X_train, 
                        y=y_train,
                        groups=grp_train,
                        w=w_train,
                        scorer=scorer)
    
    best = fmin(fn=objective, space=param_dist, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    
    model_params = param_dist.copy()
    
    for item in best:
        model_params[item] = best[item]
        
    if 'max_depth' in model_params:
        model_params['max_depth'] = int(model_params['max_depth'])
    if 'min_child_weight' in model_params:
        model_params['min_child_weight'] = int(model_params['min_child_weight'])
    if 'learning_rate' in model_params:
        model_params['learning_rate'] *= 0.1
    if 'n_estimators' in model_params:
        model_params['n_estimators'] *= 10
        print('n_estimators', model_params['n_estimators'])
        
    for item in best:
        print(item, model_params[item])

    mod = model(**model_params)

    mod.fit(X_train, y_train, sample_weight=w_train)
    
    if proba:
        print(loss_fn.__name__, loss_fn(y_val, mod.predict_proba(X_val)))
    else:
        print(loss_fn.__name__, loss_fn(y_val, mod.predict(X_val)))      
    
    return mod

group_kfold = GroupKFold(n_splits=5)

X = dt[predictors]
y = dt[target]
grp = dt.hashed_game_id
w = dt.weight
for train_index, val_index in group_kfold.split(X, y, grp):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    grp_train, grp_val = grp.iloc[train_index], grp.iloc[val_index]
    w_train, w_val = w.iloc[train_index], w.iloc[val_index]

monotone_constraints = (-1, -1, -1, 0, 1, 1, 1, -1)

xgb_params = {
    'learning_rate':    hp.lognormal('learning_rate', -2.75, 0.5),
    'max_depth':        hp.quniform('max_depth', 4, 10, 1),
    'min_child_weight': hp.qloguniform('min_child_weight', 2, 6, 1),
    'gamma':            hp.lognormal('gamma', -1, 2),
    'colsample_bytree': 1,
    'colsample_bylevel':1,
    'colsample_bynode': 1,
    'subsample':        hp.uniform('subsample', 0.5, 1),
    'reg_alpha':        hp.lognormal('reg_alpha', -1, 2),
    'reg_lambda':       hp.lognormal('reg_lambda', -1, 2),
    'n_estimators':     100,
    'sampling_method':  'uniform',
    'tree_method':      'hist',
    'gpu_id':           0,
    'objective':        'reg:squarederror',
    'nthread':          1,
    'importance_type':  'total_gain',
    'use_label_encoder': False,
    'validate_parameters': True,
    'verbosity': 0,
    'monotone_constraints': monotone_constraints
}

best = tune_model(xgb.XGBRegressor, xgb_params, 'neg_mean_squared_error', metrics.mean_squared_error, False,
                  X_train, X_val, y_train, y_val, grp_train, grp_val, w_train, w_val, max_evals=100)

group_kfold = GroupKFold(n_splits=5)

X = dt[predictors].copy()
y = dt[target]
grp = dt.hashed_game_id
w = dt.weight

best.fit(X, y, sample_weight = w)
joblib.dump(best, 'best.dat')

i = 0
cv_models = {}
cv_res = []
for train_index, val_index in group_kfold.split(X, y, grp):
    grp_train, grp_val = grp.iloc[train_index], grp.iloc[val_index]
    train = dt.loc[dt.hashed_game_id.isin(grp_train)]
    deploy = df.loc[df.hashed_game_id.isin(grp_val)]
    
    cv_models[i] = best.fit(train[predictors], train[target], sample_weight = train.weight)
    
    deploy = deploy.sort_values(by=['index'])
    
    deploy['epa_off'] = 0
    deploy['epa_def'] = 0
    deploy['expected_points'] = cv_models[i].predict(deploy[predictors])
    
    deploy_next = deploy.copy()
    deploy_next['index'] -= 1
    deploy_next['log_half_min_remaining'] = deploy_next.log_half_min_remaining.shift(1)
    deploy_next['expected_points'] = cv_models[i].predict(deploy_next[predictors])
    
    deploy = deploy.merge(deploy_next[['index', 'posteam', 'old_game_id', 'game_half', 
                                       'log_half_min_remaining', 'expected_points']],
                 on=['index', 'old_game_id', 'game_half'], how='left', suffixes = ('', '_next'))
    
    cv_res.append(deploy.copy())
    
    i += 1
    
ep = pd.concat(cv_res)

ep['ep'] = ep.expected_points
ep['ep_next'] = ep.expected_points_next

ep['score_change'] = ep.score_differential_post-ep.score_differential
ep['score_change'] = np.where(ep.score_change == 6, 7, 
                  np.where(ep.score_change == -6, -7, ep.score_change))
ep['epa_new'] = np.where(ep.ep_next.isna(), ep.score_change-ep.ep,
                     np.where(ep.score_change == -2, ep.score_change-ep.ep-ep.ep_next,
                              np.where(ep.score_change != 0, ep.score_change-ep.ep,
                                       np.where(ep.posteam == ep.posteam_next, 
                                                ep.ep_next-ep.ep, -ep.ep_next-ep.ep))))

ep.to_csv('epa.csv', index=False)

X_simul = X_val.copy()
X_simul['epa_off'] = 0
X_simul['epa_def'] = 0
X_simul['pred'] = best.predict(X_simul[predictors])
X_simul['pred_diff'] = X_simul['pred'] - X_val['pred']
X_simul['yardline_group'] = round(X_simul.yardline_100,-1)

plt.plot(X_val.loc[~X_val.down.isna()].groupby('yardline_group').avg_epa_diff.mean(), linewidth=4)
plt.xlabel('Yardline', fontsize=11)
plt.ylabel('EPA/play difference', fontsize=11)
plt.title('Average Team Strength by Field Position', fontsize=18)
plt.show()

plt.plot(X_simul.loc[~X_simul.down.isna()].groupby('yardline_group').pred_diff.mean(), linewidth=4)
plt.xlabel('Yardline', fontsize=11)
plt.ylabel(r'$\Delta$ Expected Points', fontsize=11)
plt.grid(axis='x', which='both')
plt.title('Impact of Team Strength Adjustment', fontsize=18)
plt.show()