import json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
df = pd.read_csv("used_mobile_data.csv")
info = {"rows_raw": len(df), "missing": df.isna().sum()[df.isna().sum()>0].to_dict(), "duplicates": int(df.duplicated().sum())}

# ---- Preprocessing ----
df = df.drop_duplicates().reset_index(drop=True)
info["rows_clean"] = len(df)
df["phone_age"] = 2025 - df["release_year"]          # feature engineering
df = df.drop(columns=["release_year"])
target = "used_price"
y = np.log1p(df[target])                              # log target: prices are right-skewed
X = df.drop(columns=[target])
cat = ["brand","os","condition"]
num = [c for c in X.columns if c not in cat]

# EDA plots
fig, ax = plt.subplots(1,2,figsize=(10,4))
sns.histplot(df[target], bins=40, kde=True, ax=ax[0]); ax[0].set_title("Used price distribution")
sns.histplot(y, bins=40, kde=True, ax=ax[1], color="green"); ax[1].set_title("Log(price) distribution")
plt.tight_layout(); plt.savefig("plots/01_price_distribution.png", dpi=130); plt.close()
plt.figure(figsize=(9,7)); sns.heatmap(df[num+[target]].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size":7})
plt.title("Correlation heatmap"); plt.tight_layout(); plt.savefig("plots/02_correlation.png", dpi=130); plt.close()
plt.figure(figsize=(9,4)); order=df.groupby("brand")[target].median().sort_values().index
sns.boxplot(data=df, x="brand", y=target, order=order); plt.xticks(rotation=45); plt.title("Used price by brand")
plt.tight_layout(); plt.savefig("plots/03_brand_price.png", dpi=130); plt.close()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
info["train"], info["test"] = len(Xtr), len(Xte)

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
          "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
res, fitted = {}, {}
for name, m in models.items():
    p = Pipeline([("pre", pre), ("model", m)]).fit(Xtr, ytr)
    pred = np.expm1(p.predict(Xte)); true = np.expm1(yte)
    cv = cross_val_score(Pipeline([("pre", pre), ("model", m)]), Xtr, ytr, cv=5, scoring="r2").mean()
    res[name] = dict(MAE=mean_absolute_error(true,pred), RMSE=mean_squared_error(true,pred)**0.5,
                     R2=r2_score(true,pred), R2_log=r2_score(yte,p.predict(Xte)), CV_R2=cv)
    fitted[name] = (p, pred)
    print(name, {k: round(v,3) for k,v in res[name].items()})
best = max(res, key=lambda k: res[k]["R2"]); info["best"] = best
bp, bpred = fitted[best]; btrue = np.expm1(yte)

# Result plots
r = pd.DataFrame(res).T
fig, ax = plt.subplots(1,2,figsize=(10,4))
r["R2"].plot.bar(ax=ax[0], color=["#999","#4c8","#48c"]); ax[0].set_title("R² score (test)"); ax[0].set_ylim(0,1); ax[0].tick_params(axis="x", rotation=15)
r["MAE"].plot.bar(ax=ax[1], color=["#999","#4c8","#48c"]); ax[1].set_title("MAE (₹, test)"); ax[1].tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig("plots/04_model_comparison.png", dpi=130); plt.close()
plt.figure(figsize=(5.5,5)); plt.scatter(btrue,bpred,alpha=.5,s=15)
lim=[btrue.min(),btrue.max()]; plt.plot(lim,lim,"r--"); plt.xlabel("Actual price (₹)"); plt.ylabel("Predicted price (₹)")
plt.title(f"Actual vs Predicted ({best})"); plt.tight_layout(); plt.savefig("plots/05_actual_vs_pred.png", dpi=130); plt.close()
resid = btrue-bpred
plt.figure(figsize=(6,4)); sns.histplot(resid, bins=40, kde=True); plt.title("Residuals (Actual − Predicted)"); plt.xlabel("Error (₹)")
plt.tight_layout(); plt.savefig("plots/06_residuals.png", dpi=130); plt.close()
rf = fitted["Random Forest"][0]
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values().tail(10)
imp.index = [i.split("__")[1] for i in imp.index]
plt.figure(figsize=(7,4.5)); imp.plot.barh(color="#48c"); plt.title("Top 10 feature importances (Random Forest)")
plt.tight_layout(); plt.savefig("plots/07_feature_importance.png", dpi=130); plt.close()

joblib.dump(bp, "price_model.pkl")
info["results"] = res; info["top_features"] = imp.sort_values(ascending=False).round(3).to_dict()
info["median_price"] = float(df[target].median()); info["mean_price"] = float(df[target].mean())
json.dump(info, open("results.json","w"), indent=1, default=float)
print(json.dumps(info["top_features"]), best)
