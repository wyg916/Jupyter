# -*- coding: utf-8 -*-
"""
日本房价预测项目 · 综合优化 v3-fix3
----------------------------------------------------------------
• 修复 SimpleImputer 关键字参数兼容性
• 新增可视化：MLP_loss曲线、PCA散点图、相关热力图
"""

import os, time, datetime, json, warnings, joblib, pickle
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose      import ColumnTransformer
from sklearn.pipeline     import Pipeline
from sklearn.impute       import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection  import KFold, RandomizedSearchCV
from sklearn.linear_model     import RidgeCV
from sklearn.metrics          import (r2_score, mean_squared_error,
                                      accuracy_score, confusion_matrix,
                                      roc_auc_score, RocCurveDisplay)
from sklearn.decomposition    import PCA
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models

# ---------- util ----------
ts  = lambda : datetime.datetime.now().strftime("%H:%M:%S")
tic = time.time
def toc(s,m): print(f"[{ts()}] {m} 用时 {time.time()-s:.2f}s")

def safe_save_parquet(df,path):
    try: df.to_parquet(path,index=False)
    except: df.to_csv(path.replace(".parquet",".csv"),index=False)

def safe_save_hdf(df,path):
    try: df.to_hdf(path,key="data",mode="w")
    except: pickle.dump(df,open(path.replace(".h5",".pkl"),"wb"))

# ---------- paths ----------
BASE = r"C:\Users\Administrator\Desktop\kaggle\kg-20250705-220-python"
TX,LOC = os.path.join(BASE,"All_prefectures_buildings_with_migration.xlsx"), os.path.join(BASE,"Location_Data.xlsx")
RES = os.path.join(BASE,"结果"); os.makedirs(RES,exist_ok=True)

# ---------- 中文显示 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- meta ----------
json.dump({"采集时间":ts(),
           "数据源":[os.path.basename(TX),os.path.basename(LOC)],
           "代码版本":"v3_fix3"},open(os.path.join(RES,"meta_info.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)

# ---------- read ----------
t0=tic(); tx=pd.read_excel(TX,engine="openpyxl"); loc=pd.read_excel(LOC,engine="openpyxl"); toc(t0,"读取完成")
KEY="City,Town,Ward,Village code"
df=tx.merge(loc,on=KEY,how="left"); df=df[df["Year"]>=2022].copy()

# ---------- clean ----------
for col in df.select_dtypes(include=[np.number]).columns:
    q1,q3=df[col].quantile([0.25,0.75]); iqr=q3-q1
    df=df[(df[col]>=q1-1.5*iqr)&(df[col]<=q3+1.5*iqr)]
df=df.drop_duplicates()
safe_save_parquet(df,os.path.join(RES,"raw_clean.parquet")); safe_save_hdf(df,os.path.join(RES,"raw_clean.h5"))
pd.DataFrame({"缺失率":df.isna().mean(),"唯一值":df.nunique()}).to_excel(os.path.join(RES,"质量报告.xlsx"))

train_df, test_df = df[df["Year"]<2024].copy(), df[df["Year"]>=2024].copy()

# ---------- feature engineer ----------
def engineer(d,med=None):
    d=d.copy()
    d["Quarter"].fillna(1,inplace=True)
    d["month"]=d["Quarter"].astype(int)*3
    d["date"]=pd.to_datetime(dict(year=d.Year,month=d.month,day=1),errors="coerce")
    d["Year_f"]=d.date.dt.year; d["Weekday"]=d.date.dt.weekday
    d["ConstructionYear"].fillna(d["Year"].median(),inplace=True)
    d["Age"]=(d["Year"]-d["ConstructionYear"]).clip(lower=0)
    for c in ["Area","AverageTimeToStation","Frontage","TotalFloorArea"]:
        d[c].fillna(0,inplace=True); d[f"log_{c}"]=np.log1p(d[c])
    d["FAR_logArea"]=d["FloorAreaRatio"]*d["log_Area"]
    if med is None: med=d.groupby(KEY)["TotalTransactionValue"].median()
    d["MedianRegionPrice"]=d[KEY].map(med).fillna(med.median())
    d.drop(columns=[c for c in ["date","month","Year","Quarter",KEY] if c in d.columns],inplace=True)
    d.fillna("Unknown",inplace=True)
    return d,med
train_df,med=engineer(train_df); test_df,_=engineer(test_df,med)
full=pd.concat([train_df,test_df],ignore_index=True)

# ---------- preprocess + SelectK ----------
num=full.select_dtypes(include=[np.number]).columns.tolist(); num.remove("TotalTransactionValue")
cat=[c for c in full.columns if c not in num+["TotalTransactionValue"]]
cat_keep=[c for c in cat if full[c].nunique()<=50]
pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),
                         ("sc",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="Unknown")),
                         ("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat_keep)],
        remainder="drop",n_jobs=1)
n=len(train_df)
X_tr,X_te = full.drop(columns=["TotalTransactionValue"]).iloc[:n], full.drop(columns=["TotalTransactionValue"]).iloc[n:]
y_tr,y_te = full["TotalTransactionValue"].values[:n], full["TotalTransactionValue"].values[n:]
Xt_tr,Xt_te = pre.fit_transform(X_tr), pre.transform(X_te)
k=min(200,Xt_tr.shape[1]); skb=SelectKBest(f_regression,k=k); Xt_tr, Xt_te = skb.fit_transform(Xt_tr,np.log1p(y_tr)), skb.transform(Xt_te)

# ---------- 相关热力图 ----------
corr=np.corrcoef(Xt_tr.T)
plt.figure(figsize=(6,5)); sns.heatmap(corr,cmap="RdBu_r",vmin=-1,vmax=1); plt.title("相关热力图")
plt.tight_layout(); plt.savefig(os.path.join(RES,"相关热力图.png"),dpi=300); plt.close()

# ---------- Baseline Ridge ----------
r=RidgeCV(alphas=np.logspace(-2,2,10),cv=5).fit(Xt_tr,np.log1p(y_tr))
r2_r=r2_score(y_te,np.expm1(r.predict(Xt_te))); mse_r=mean_squared_error(y_te,np.expm1(r.predict(Xt_te)))

# ---------- MLP ----------
def build(d):
    m=models.Sequential([layers.Dense(256,activation='relu',input_dim=d),
                         layers.Dropout(0.3),
                         layers.Dense(128,activation='relu'),
                         layers.Dropout(0.3),
                         layers.Dense(64,activation='relu'),
                         layers.Dense(1)])
    m.compile(optimizer='adam',loss='mse'); return m
hist=[]
for tr,val in KFold(5,shuffle=True,random_state=42).split(Xt_tr):
    net=build(Xt_tr.shape[1])
    h=net.fit(Xt_tr[tr],np.log1p(y_tr[tr]),
              validation_data=(Xt_tr[val],np.log1p(y_tr[val])),
              epochs=30,batch_size=512,verbose=0,
              callbacks=[callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
    hist.append(pd.DataFrame(h.history))
hist[0].to_excel(os.path.join(RES,"MLP_loss曲线.xlsx"),index=False)
plt.figure(figsize=(6,4))
plt.plot(hist[0]["loss"],label="loss"); plt.plot(hist[0]["val_loss"],label="val_loss"); plt.legend()
plt.title("MLP 训练曲线"); plt.tight_layout(); plt.savefig(os.path.join(RES,"MLP_loss曲线.png"),dpi=300); plt.close()
net=build(Xt_tr.shape[1]); net.fit(Xt_tr,np.log1p(y_tr),epochs=20,batch_size=512,verbose=0)
y_pred_m=np.expm1(net.predict(Xt_te,batch_size=512).flatten())
r2_m=r2_score(y_te,y_pred_m); mse_m=mean_squared_error(y_te,y_pred_m)

# ---------- PCA ----------
pca=PCA(n_components=2,random_state=42); Z=pca.fit_transform(Xt_tr)
plt.figure(figsize=(6,6))
plt.scatter(Z[:,0],Z[:,1],c=np.log1p(y_tr),s=8,cmap="viridis"); plt.colorbar(label="log房价")
plt.title("PCA散点图"); plt.tight_layout(); plt.savefig(os.path.join(RES,"PCA散点图.png"),dpi=300); plt.close()

# ---------- XGB 随机搜索 ----------
grid={"learning_rate":[0.03,0.05,0.07],"max_depth":[6,8,10],
      "subsample":[0.7,0.8,0.9],"colsample_bytree":[0.7,0.8,0.9],
      "n_estimators":[800,1000,1200]}
base=xgb.XGBRegressor(objective="reg:squarederror",tree_method="gpu_hist",
                      predictor="gpu_predictor",random_state=42)
search=RandomizedSearchCV(base,grid,n_iter=15,cv=3,scoring="r2",random_state=42,verbose=0)
search.fit(Xt_tr,np.log1p(y_tr)); cv=pd.DataFrame(search.cv_results_)
pivot=cv.pivot_table(index="param_max_depth",columns="param_n_estimators",
                     values="mean_test_score",aggfunc="mean")
sns.heatmap(pivot,annot=True,cmap="YlGnBu"); plt.title("XGB 参数热力图")
plt.tight_layout(); plt.savefig(os.path.join(RES,"XGB参数热力图.png"),dpi=300); plt.close()
best=search.best_estimator_
y_pred_x=np.expm1(best.predict(Xt_te)); r2_x=r2_score(y_te,y_pred_x); mse_x=mean_squared_error(y_te,y_pred_x)

# ---------- 分类 ----------
thr=np.percentile(y_tr,75)
y_bin_tr=(y_tr>=thr).astype(int); y_bin_te=(y_te>=thr).astype(int)
clf=xgb.XGBClassifier(tree_method="gpu_hist",predictor="gpu_predictor",
                      objective="binary:logistic",eval_metric="auc",
                      n_estimators=600,learning_rate=0.05,max_depth=6,
                      subsample=0.8,colsample_bytree=0.8,random_state=42)
clf.fit(Xt_tr,y_bin_tr,verbose=False)
acc=accuracy_score(y_bin_te,clf.predict(Xt_te))
auc=roc_auc_score(y_bin_te,clf.predict_proba(Xt_te)[:,1])
cm_df=pd.DataFrame(confusion_matrix(y_bin_te,clf.predict(Xt_te)),
                   index=["真实低","真实高"],columns=["预测低","预测高"])
RocCurveDisplay.from_predictions(y_bin_te,clf.predict_proba(Xt_te)[:,1])
plt.plot([0,1],[0,1],'k--'); plt.title("ROC曲线")
plt.tight_layout(); plt.savefig(os.path.join(RES,"ROC曲线.png"),dpi=300); plt.close()

plt.figure(figsize=(6,6))
plt.scatter(y_te,y_pred_x,alpha=0.3); plt.plot([y_te.min(),y_te.max()],[y_te.min(),y_te.max()],'r--')
plt.xlabel("真实房价"); plt.ylabel("预测房价"); plt.title("XGB 真实 vs 预测")
plt.tight_layout(); plt.savefig(os.path.join(RES,"预测散点.png"),dpi=300); plt.close()

# ---------- Excel 汇总 ----------
with pd.ExcelWriter(os.path.join(RES,"模型综合结果.xlsx"),engine="xlsxwriter") as w:
    pd.DataFrame({"模型":["Ridge","MLP","XGB"],
                  "R2":[r2_r,r2_m,r2_x],"MSE":[mse_r,mse_m,mse_x]})\
      .to_excel(w,sheet_name="回归指标",index=False)
    pd.DataFrame({"模型":["XGB_分类"],"准确率":[acc],"AUC":[auc]})\
      .to_excel(w,sheet_name="分类指标",index=False)
    pd.DataFrame({"缺失率":df.isna().mean(),"唯一值":df.nunique()}).to_excel(w,sheet_name="数据质量")
    cm_df.to_excel(w,sheet_name="混淆矩阵")
    pivot.to_excel(w,sheet_name="XGB参数网格")

# ---------- console ----------
print("\n===== 回归结果 =====")
print(f"Ridge  R2={r2_r:.3f}"); print(f"MLP    R2={r2_m:.3f}"); print(f"XGB    R2={r2_x:.3f}")
print("\n===== 分类结果 ====="); print(f"Acc={acc:.3f}  AUC={auc:.3f}")

# ---------- save models ----------
joblib.dump(best,os.path.join(RES,"best_xgb.pkl"))
joblib.dump(clf,os.path.join(RES,"xgb_clf.pkl"))


# [22:07:08] 读取完成 用时 53.18s
# 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 

# ===== 回归结果 =====
# Ridge  R2=0.684
# MLP    R2=-2.609
# XGB    R2=0.787

# ===== 分类结果 =====
# Acc=0.879  AUC=0.942
