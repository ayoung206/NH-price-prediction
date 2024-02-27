# %%
# -*- coding: utf-8 -*-
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import argparse
import datetime
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from glob import glob
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import scikeras
from scikeras.wrappers import KerasRegressor
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import GRU, Dense
import tensorflow as tf
from impala.dbapi import connect
from impala.util import as_pandas
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import time
import pickle

pd.set_option('display.max_columns', None)

# %% [markdown]
# ## Step0. 데이터 수집 및 마트 생성

# %%
## 사용변수 ##

# 1) 공판장
# 경락단가, 상장수수료, 운송비, 하역비, kg당 판매금액 (평균)
# 중량 (합계)

# 2) 기상청

# 3) 재고
# nfbpsb.tmp_stpl_trn_inf_03
# 차변수량, 매출수량, 재고수량 (합계)
# 차변금액, 매출금액, 재고금액 (평균)

# 4) 매입
# 매입수량 (합계)
# 매입금액 (평균)

# 5) 소매
# 소매가격 (평균)

# 6) 도매시장 경락가격
# kg당 경락단가 (평균)

# 7) 물가지수 (월별)
# 품목별 생산자물가지수, 품목성질별 생산자물가지수, 품목별 소비자물가지수, 품목성질별 소비자물가지수

# 8) 동향지수 (월별)
# 현재생활형편CSI, 생활형편전망CSI, 향후경기전망CSI, 소비지출전망CSI, 외식비 지출전망CSI, 물가수준전망(1년후)CSI

# 9) 수출입 (월별)
# 수입금액, 수출금액 (평균)
# 수입중량, 수출중량 (합계)

# 10) 하나로마트
# 순매출금액 (합계)

# %% [markdown]
# ## Step1. Data Read

# %%
def select_df(df): #쿼리
    sql = f"""SELECT * FROM """+str(df)

    con = connect(host = 'nhbpunloap01.nhbpad.nonghyup.com', port = '21050', 
                  kerberos_service_name = 'impala', use_ssl=False, auth_mechanism = 'GSSAPI')

    impala_cursor = con.cursor() 
    impala_cursor.execute(sql) #쿼리 실행 
    df = as_pandas(impala_cursor) #pandas 데이터 프레임으로 변경
    
    #byte 타입 변경
    for col in df.columns:
        if str(df[col][0])[0] == 'b':
            df[col] = df[col].str.decode('utf-8')
    impala_cursor.close()
    con.close()
    
    return df

#훈련 데이터 전체의 시작일 부터 마지막 일까지의 주차 생성
def week_label(df):
    
    df['week_mark'] = np.nan
    df['yyyy'] = df['yyyy'].astype('int')
    year_range = int(max(df['yyyy'])) - int(min(df['yyyy']))+1
    
    for i in range(0,year_range*53):
        min_mark = (datetime.strptime(min(df['bas_dt']), '%Y%m%d') + relativedelta(weeks=i)).strftime('%Y%m%d')
        max_mark = (datetime.strptime(min(df['bas_dt']), '%Y%m%d') + relativedelta(weeks=i+1)).strftime('%Y%m%d')
        
        df.loc[(df['bas_dt']>=min_mark)&(df['bas_dt']<max_mark), 'week_mark'] = i
    df['week_mark'] = df['week_mark'].astype(int)
    df_prp = df.copy()
    
    return df_prp

# %% [markdown]
# ### 데이터 불러오기 & Join

# %%
# 설명 변수
wmc = select_df('nfbpsb.tmp_wmc_trn_inf') #공판장
weather = select_df('nfbpsb.tmp_weather_trn_inf') #기상청
weather = weather.fillna(0)
stpl = select_df('nfbpsb.tmp_stpl_trn_inf') #재고
byng = select_df('nfbpsb.tmp_byng_trn_inf') #매입
retail = select_df('nfbpsb.tmp_retail_trn_inf') #소매
actopr = select_df('nfbpsb.tmp_actopr_trn_inf') #도매시장 경락가격
prsix = select_df('nfbpsb.tmp_prsix_trn_inf') #물가지수(월별)
tnix = select_df('nfbpsb.tmp_tnix_trn_inf') #동향지수(월별)
imxp = select_df('nfbpsb.tmp_imxp_trn_inf') #수출입(월별) (풋고추 없음)
nacf_rtl = select_df('nfbpsb.tmp_nacf_rtl_trn_inf') #하나로마트

# 타겟 변수
sl = select_df('nfbpsb.tmp_sl_trn_inf') #가락시장 가격

# %%
# 일별 테이블 한번에 붙이기
datasets = [wmc, stpl, byng, retail, actopr, nacf_rtl]
df = reduce(lambda left, right: pd.merge(left, right, on = ['frpd_latcnm', 'bas_dt', 'bas_week'],
                                         how = 'outer'), datasets)
# 기상 테이블 join
df = pd.merge(df, weather, on = ['bas_dt', 'bas_week'], how = 'outer')

# 월별 테이블 붙이기 (월별테이블은 월 말에 적재되는 이유로 전월 데이터만 사용 예정)
df['bas_dt'] = pd.to_datetime(df['bas_dt'])
df['bas_ym'] = df['bas_dt'].dt.to_period('M')
df['bas_ym'] = df['bas_ym'].dt.strftime('%Y%m')
df['bas_dt'] = df['bas_dt'].dt.strftime('%Y%m%d')
datasets = [df, prsix, imxp]
df = reduce(lambda left, right: pd.merge(left, right, on = ['frpd_latcnm', 'bas_ym'],
                                         how = 'outer'), datasets)
df = pd.merge(df, tnix, on = 'bas_ym', how = 'outer')

# 가락시장 가격 테이블에 left join
df = pd.merge(garak_price_avg, df, on = ['frpd_latcnm', 'bas_dt'], how = 'left')

# 요일, 년도, 월, 일, 주차 생성
df['bas_dt'] = pd.to_datetime(df['bas_dt'])
df['weekday'] = df['bas_dt'].dt.day_name()
df['yyyy'] = df['bas_dt'].dt.year
df['mm'] = df['bas_dt'].dt.month
df['dd'] = df['bas_dt'].dt.day
df['week'] = df['bas_week'].str[4:].astype(int)
df['bas_dt'] = df['bas_dt'].dt.strftime('%Y%m%d')

drop_idx = df.loc[df['weekday']=='Sunday'].index #일요일 삭제
df.drop(drop_idx, inplace=True)

# 요일 컬럼 더미변수화
weekday_dummies = pd.get_dummies(df['weekday'])
df = pd.concat([df, weekday_dummies], axis = 1)

# 훈련 데이터 전체의 시작일 부터 마지막 일까지의 주차 생성
df = week_label(df)

df = df.sort_values(['frpd_latcnm', 'bas_dt']).reset_index(drop = True)

# %%
# 주차별 표준편차 계산
def price_std_vy_frpd(datasets, frpd_latc_c):
    df = datasets.loc[datasets['frpd_latc_c'] == frpd_latc_c].drop(['frpd_latc_c'], axis = 1)
    std = pd.DataFrame(df.groupby('week')['gk_price'].std(ddof = 0)).reset_index()
    return std

# %% [markdown]
# ### 품목별 DataSet 생성

# %%
def df_preprocess(datasets, frpd_latcnm):
    
    df = datasets.loc[datasets['frpd_latcnm'] == frpd_latcnm].drop(['frpd_latcnm'], axis = 1)
    
    # 1) 가격변수/물량변수/기상변수(강수계속시간, 일강수량, 안개계속시간) 
    # 주평균 -> 월평균
    # 날짜 제외한 모든 변수 데이터 타입 변경
    cols_date = ['frpd_latcnm', 'bas_dt', 'bas_ym', 'bas_week', 'yyyy', 'mm', 'dd', 'week', 'weekday', 'week_mark',
                 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    cols_all = [col for col in list(df.columns) if col not in cols_date]
    for col in cols_all:
        df[col] = df[col].astype(float)
    col_am_mqt = ['gk_price', 'gk_mqt', 'acto_upr', 'lstg_fee', 'trpcs', 'stvcs', 'wt', 
                  'sel_am', 'db_wt', 'db_am', 'sl_wt', 'slam', 'stpl_wt', 'stpl_am',
                  'byng_mqt', 'byam', 'price', 'tot_qty', 'whsl_acto_trqt', 'nslam']
    col_weather = ['sumrndur_south', 'sumrndur_mid', 'sumrn_south', 'sumrn_mid', 'sumfogdur_south', 'sumfogdur_mid']
    column_to_fill = col_am_mqt + col_weather

    # 주 평균으로 채우기
    weekly_avg = df.groupby(['bas_week'])[column_to_fill].transform('mean')
    df[column_to_fill].fillna(weekly_avg, inplace = True)

    # 월 평균으로 채우기 (주 평균으로 null값 채워지지 않을경우 대비)
    monthly_avg = df.groupby(['bas_ym'])[column_to_fill].transform('mean')
    df[column_to_fill].fillna(monthly_avg, inplace = True)

    # 2) 나머지 변수 -> ffill/bfill
    df = df.fillna(method = 'ffill').fillna(method = 'bfill')
    
    return df

# %%
df_potato = df_preprocess(df, '감자')
df_leak = df_preprocess(df, '대파')
df_radish = df_preprocess(df, '무')
df_cabbage = df_preprocess(df, '배추')
df_apple = df_preprocess(df, '사과')
df_pepper = df_preprocess(df, '풋고추')

# %% [markdown]
# ## Step2. 파생변수 생성

# %% [markdown]
# ### 시차변수 생성

# %%
# 시차변수 생성
def lag_variable(df, y_col, col_type):
    
    if col_type == 'target':
        # 1일 ~ 24일 후 값
        for day in range(1, 25):
            df[y_col+'_'+str(day)+'day_after'] = df[y_col].shift(-day)
    
    if col_type == 'day':
        # 1일 전 값
        day_ls = [1]
        for day in day_ls:
            df[y_col+'_'+str(day)+'day_ago'] = df[y_col].shift(day)

        # 1주, 4주 전 값
        week_ls = [1,4]
        for week in week_ls:
            df[y_col+'_'+str(week)+'wk_ago'] = df.apply(lambda row: df[(df['week_mark'] == row['week_mark'] - week) 
                                                                  & (df['weekday'] == row['weekday'])][y_col].mean(), axis = 1)

        # 지난 3/5/7/15일간 평균
        day_ls = [3,5,7,15]
        for day in day_ls:
            df[y_col+'_'+str(day)+'day_avg'] = df[y_col].rolling(window=day).mean()

        # 1년 전 동일 월 평균
        year_ls = [1]
        for year in year_ls:
            df[y_col+'_'+str(year)+'ym_avg'] = df.apply(lambda row: df[(df['yyyy'] == row['yyyy'] - year) 
                                                                  & (df['mm'] == row['mm'])][y_col].mean(), axis = 1)

        # 1년 전 동일 주차 평균
        year_ls = [1]
        for year in year_ls:
            df[y_col+'_'+str(year)+'yw_avg'] = df.apply(lambda row: df[(df['week_mark'] == row['week_mark'] - 52) 
                                                                      ][y_col].mean(), axis = 1)
            
    if col_type == 'month':
        # 1월 전
        month_ls = [1]
        for month in month_ls:
            df[y_col+'_'+str(month)+'month_ago'] = df.apply(lambda row: df[(df['bas_ym'] == (pd.to_datetime(row['bas_dt']) - relativedelta(months = month)).strftime('%Y%m'))
                                                                          ][y_col].mean(), axis = 1)
        
        # 1년 전 동일 월 평균
        year_ls = [1]
        for year in year_ls:
            df[y_col+'_'+str(year)+'ym_avg'] = df.apply(lambda row: df[(df['yyyy'] == row['yyyy'] - year) 
                                                                  & (df['mm'] == row['mm'])][y_col].mean(), axis = 1)
    
    df.reset_index(drop = True, inplace=True)
    
    return df

# %%
df_all = {'df_potato':df_potato, 'df_leak':df_leak, 'df_radish':df_radish,
          'df_cabbage':df_cabbage, 'df_apple':df_apple, 'df_pepper':df_pepper}
df_all_preprocessed = []
for key, value in df_all.items():
    start_time = time.time()
    # 시차변수 생성을 위한 dataframe
    df_lag = value.copy()

    # 시차변수 생성 (일/월)
    # 일별컬럼
    for col in cols_day:
        df_lag = lag_variable(df_lag, col, 'day')
    
    # 월별컬럼
    for col in cols_month:
        df_lag = lag_variable(df_lag, col, 'month')
        
    # 타겟 변수
    df_lag = lag_variable(df_lag, 'gk_price', 'target')
    
    # 전년동월/주차 시차변수를 생성할 수 없는 값들을 drop
    # 2016년~ 데이터를 생성한 뒤 2017년 데이터부터 사용하는 방법
    df_lag = df_lag.loc[df_lag['yyyy'] != 2016].reset_index(drop = True)
    
    # 당월변수 삭제 (월별테이블은 월 말에 적재되는 이유로 전월 데이터만 사용 예정)
    df_lag.drop(cols_month, axis = 1, inplace = True)
    
    # null값 전처리
    df_lag = df_lag.fillna(method = 'ffill').fillna(method = 'bfill')

    df_all_preprocessed.append(df_lag)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{key} 실행시간: {execution_time}초")
    
df_potato = df_all_preprocessed[0]
df_leak = df_all_preprocessed[1]
df_radish = df_all_preprocessed[2]
df_cabbage = df_all_preprocessed[3]
df_apple = df_all_preprocessed[4]
df_pepper = df_all_preprocessed[5]

# %% [markdown]
# ## Step3. Variable Selection

# %% [markdown]
# ### 변수 중요도 분석

# %%
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def analyze_regression(X, y):
    # 회귀분석 수행
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    p_values = model.pvalues[1:]
    coef_abs = np.abs(model.params[1:])
    return p_values, coef_abs

# %%
df_all = {'price_potato':df_potato, 'price_leak':df_leak, 'price_radish':df_radish,
          'price_cabbage':df_cabbage, 'price_apple':df_apple, 'price_pepper':df_pepper}

importance_xgb_all = []
importance_reg_all = []

for key, value in df_all.items():
    df_tmp = value.copy()
    
    df_tmp = df_tmp.drop(cols_pk, axis = 1)
    
    start_time = time.time()

    X = df_tmp.drop(cols_target, axis = 1)
    Y = df_tmp['gk_price_1day_after']

    # XGBoost Regressor
    model = XGBRegressor(objective = 'reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X, Y)

    importances_xgb = model.feature_importances_

    importance_xgb_df = pd.DataFrame({'Feature':X.columns, 'Importance':importances_xgb})
    importance_xgb_df = importance_xgb_df.sort_values(by = 'Importance', ascending = False).reset_index(drop=True)
    
    importance_xgb_all.append(importance_xgb_df)
    
    # Linear Regression
    p_values, coef_abs = analyze_regression(X, Y)
    
    variable_importance = dict(zip(X.columns, coef_abs))

    importance_reg_df = pd.DataFrame({
        'Columns': list(variable_importance.keys()),
        'p_value': p_values,
        'Importance': coef_abs
    })
    importance_reg_df = importance_reg_df[importance_reg_df['p_value'] < 0.05] # 유의한 변수만
    importance_reg_df = importance_reg_df.sort_values(by = 'Importance', ascending = False).reset_index(drop=True)
    
    importance_reg_all.append(importance_reg_df)

    end_time = time.time()
    execution_time = end_time - start_time

# %% [markdown]
# ## Step4. Model 학습

# %%
def random_sample_same_week(df, column_name):
    unique_weeks = df[column_name].unique()
    sampled_rows = []
    
    for week in unique_weeks:
        week_rows = df[df[column_name] == week]
        sampled_row = week_rows.sample(n = 1, random_state = 42)
        sampled_rows.append(sampled_row)
    
    result_df = pd.concat(sampled_rows)
    return result_df

# dataset for GRU
def make_dataset(data, label, window_size):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

# %%
df_all = {'price_potato':df_potato, 'price_leak':df_leak, 'price_radish':df_radish,
          'price_cabbage':df_cabbage, 'price_apple':df_apple, 'price_pepper':df_pepper}
# 성능지표 저장
eval_df_all = []
# 변수 중요도 저장
importance_df_all = []
# 결과 저장
results_all = []

for key, value in df_all.items():
    start_time = time.time()
    
    if key == 'price_potato': frpd_latcnm = '감자'
    if key == 'price_leak': frpd_latcnm = '대파'
    if key == 'price_radish': frpd_latcnm = '무'
    if key == 'price_cabbage': frpd_latcnm = '배추'
    if key == 'price_apple': frpd_latcnm = '사과'
    if key == 'price_pepper': frpd_latcnm = '풋고추'
    
    df_tmp = value.copy()
    
    ##############################################################################
    ## Variable Selection
    df_tmp = df_tmp.drop(cols_pk, axis = 1)
    X = df_tmp.drop(cols_target, axis = 1)
    Y = df_tmp['gk_price_1day_after']

    model = XGBRegressor(objective = 'reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X, Y)

    importances_xgb = model.feature_importances_

    importance_xgb_df = pd.DataFrame({'Feature':X.columns, 'Importance':importances_xgb})
    importance_xgb_df = importance_xgb_df.sort_values(by = 'Importance', ascending = False).reset_index(drop=True)
    
    ##############################################################################
    df_tmp = value.copy()
    
    # TOP100 변수 선택
    cols_selected = list(importance_xgb_df['Feature'].head(100)) #변수 개수 지정
    cols_model = cols_pk + cols_target + cols_selected
    df_tmp = df_tmp[cols_model]
    
    # 주별로 하나의 값만 random sampling
    test_sampled = random_sample_same_week(test_tmp, 'bas_week')
    # 나머지 row들은 학습에 사용
    remaining_indices = list(set(test_tmp.index) - set(test_sampled.index))
    remaining_test = test_tmp.loc[remaining_indices].reset_index(drop = True)
    train_tmp = pd.concat([train_tmp, remaining_test], ignore_index = True)

    test_tmp = test_sampled.copy()

    train_data = train_tmp.drop(cols_pk, axis = 1)
    test_data = test_tmp.drop(cols_pk, axis = 1)
    
    ## cols_target
    # 훈련 데이터셋 생성
    X_train = train_data.drop(cols_target, axis = 1)
    y_train = train_data[cols_target]

    # 테스트 데이터셋 생성
    X_test = test_data.drop(cols_target, axis = 1)
    y_test = test_data[cols_target]
        
    ##############################################################################
    ### XGBoost
    print('XGBoost Model Training...')

    # TimeSeriesSplit 객체 생성
    tscv = TimeSeriesSplit(n_splits = 5)

    # hyperparameter 그리드 정의
    param_grid = {
        'estimator__n_estimators': [100, 200, 300],
        'estimator__max_depth': [3, 4, 5],
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__colsample_bytree':[0.3, 0.7, 1.0]
        }

    # Grid Search 객체 생성
    grid_search = GridSearchCV(
        MultiOutputRegressor(XGBRegressor(objective = 'reg:squarederror')), 
        param_grid = param_grid, 
        scoring = 'neg_mean_absolute_error', # 평가지표
        cv = tscv # TimeSeriesSplit
        )

    # 최적의 하이퍼파라미터
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    ##############################################################################
    ### GRU
    print('GRU Model Training...')

    # Scaling
    scaler = MinMaxscaler()
    x_train_GRU = pd.DataFrame(scaler.fit_transform(x_train_GRU))
    y_train_GRU = pd. DataFrame(scaler.fit_transform(y_train_GRU))
    x_test_GRU = pd.DataFrame(scaler.fit_transform(x_test_GRU))
    y_test_GRU = pd. DataFrame(scaler.fit_transform(y_test_GRU))

    look_back = 6 # 입력 시퀀스 길이
    Target_horizon = 24 # 모델의 출력 차원

    x_train_window, y_train_window = make_dataset(x_train_GRU, y_train_GRU, look_back)
    x_test_wIndow, y_test_window = make_dataset(x_test_GRU, y_test_GRU, look_back)

    def create_model(look_back, target_horizon):
        model = Sequential()
        model.add(GRU(units = 128, activation = 'relu', input_shape = (look_back, 100, return_sequences = False)))
        model.add(Dense(target_horizon))
        model.compile(optimizer = 'adam', loss = 'mse')
        return model
    
    # 모델 래퍼 생성
    model = KerasRegressor(build_fn = create_model, look_back = look_back, target_horizon = target_horizon, verbose = 2)

    # hyperparameter 그리드 정의
    param_grid = {'epochs': [10,20,30],
                'batch_size': [16,32,64]
                }

    # 시계열 교차 검증
    tscv = TimeSeriesSplit(n_splits = 6)

    grid_search = GridSearchCV(
        estimator = model, 
        param_grid = param_grid,
        cv = tscv # TimeSeriesSplit
        )

    grid_result = grid_search.fit(x_train_window, y_train_window, callbacks = [early_stop])
    best_model = grid_search.best_estimator_

    # 예측
    test_preds = best_model.predict(X_test)

    ##############################################################################
    # Test 성능
    mse = mean_squared_error(y_test, test_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, test_preds)
    mape = mean_absolute_percentage_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)
    
    print(f"{key} Test 성능")
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("R-Squared:", r2)
    
    ## Test 성능 저장
    # 운영 시 -> '기준일자', '기준주차' 컬럼 필요
    eval_df = pd.DataFrame({'Frpd_latcnm': [frpd_latcnm],
                            'Model_Name':['Price Prediction - XGBRegressor'],
                            'Model_Object' :[best_model],
                            'MSE':[mse],
                            'RMSE':[rmse],
                            'MAE':[mae],
                            'MAPE':[mape]})
    
    eval_df_all.append(eval_df)
    
    ##############################################################################
    # Train 성능
    train_preds = best_model.predict(X_train)

    mse = mean_squared_error(y_train, train_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, train_preds)
    mape = mean_absolute_percentage_error(y_train, train_preds)
    r2 = r2_score(y_train, train_preds)

    print(f"{key} Train 성능")
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("R-Squared:", r2)
    
    ##############################################################################
    ## 변수 중요도 저장
    importance_data  = []
    feature_names = list(X_train.columns)
    for i, model in enumerate(best_model.estimators_):
        importance = model.feature_importances_
        ranked_features = np.argsort(importance)[::-1] #중요도 내림차순 정렬 후 인덱스 반환
        for rank, j in enumerate(ranked_features):
            importance_data.append([frpd_latcnm, 'Price Prediction - XGBRegressor',
                                    f"Model {i+1}", rank+1, feature_names[j], importance[j]])

    columns = ['Frpd_latcnm', 'Model_Name', 'Model_Num', 'Rank', 'Feature', 'Importance']
    importance_df = pd.DataFrame(importance_data, columns = columns)
    importance_df_all.append(importance_df)
    
    ##############################################################################
    ## 시각화를 위한 실제값/예측값 저장    
    # 실제값
    y_averaged_columns = [y_test.iloc[:, i:i+6].mean(axis=1) for i in range(0, len(y_test.columns), 6)]
    y_test_week = pd.concat(y_averaged_columns, axis=1).reset_index(drop = True)
    new_column = {0: '실제값 - 1주차', 1: '실제값 - 2주차', 2: '실제값 - 3주차', 3: '실제값 - 4주차'}
    y_test_week.rename(columns = new_column, inplace = True)
    
    # 예측값
    test_preds = pd.DataFrame(test_preds)
    pred_averaged_columns = [test_preds.iloc[:, i:i+6].mean(axis=1) for i in range(0, len(test_preds.columns), 6)]
    pred_week = pd.concat(pred_averaged_columns, axis=1).reset_index(drop = True)
    new_column = {0: '예측값 - 1주차', 1: '예측값 - 2주차', 2: '예측값 - 3주차', 3: '예측값 - 4주차'}
    pred_week.rename(columns = new_column, inplace = True)
    
    date_df = test_tmp[['bas_dt', 'bas_ym', 'bas_week']].reset_index(drop = True)
    results = pd.concat([date_df, y_test_week.iloc[:,0], pred_week.iloc[:,0], y_test_week.iloc[:,1], pred_week.iloc[:,1], 
                         y_test_week.iloc[:,2], pred_week.iloc[:,2], y_test_week.iloc[:,3], pred_week.iloc[:,3]], axis = 1)
    
    results_all.append(results)
    
    ##############################################################################
    ## 모델 저장
    model_name = f"{key}.pkl"
    with open(model_name, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"{key} model saved.")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{key} 실행시간: {execution_time}초")
    print('==='*30)

# %%
result = {'Potato':results_potato, 'Leak':results_leak, 'Radish':results_radish,
          'Cabbage':results_cabbage, 'Apple':results_apple, 'Pepper':results_pepper}

for key, value in result.items():
    for i in range(1, 5):
        plt.figure(figsize=(20, 6))
        plt.plot(value[f"실제값 - {i}주차"], label = 'Original Data', color = 'blue')
        plt.plot(value[f"예측값 - {i}주차"], label = 'Predicted Data', color = 'red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(range(0, 52), value['bas_dt'])
        plt.title(f"{key} Week {i} Prediction")
        plt.legend()
        plt.show()


