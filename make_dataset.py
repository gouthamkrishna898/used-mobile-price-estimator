"""Generates a synthetic dataset with the same structure as the public
'Used Device Data' (ReCell) dataset. Replace used_mobile_data.csv with the real
Kaggle CSV (adjust column names) if you want to use real-world data."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 1500
brands = {"Samsung":1.0,"Apple":1.7,"Xiaomi":0.75,"Realme":0.7,"OnePlus":0.95,"Oppo":0.75,"Vivo":0.75,"Motorola":0.7,"Nokia":0.6,"Google":1.1}
brand = rng.choice(list(brands), n, p=[.2,.12,.17,.12,.07,.1,.1,.05,.04,.03])
year = rng.integers(2016, 2025, n)
ram = rng.choice([2,3,4,6,8,12,16], n, p=[.05,.1,.25,.25,.22,.09,.04])
rom = rng.choice([32,64,128,256,512], n, p=[.1,.25,.35,.22,.08])
screen = np.round(rng.normal(6.3,0.35,n).clip(5.0,7.2),1)
battery = (rng.normal(4300,600,n).clip(2500,6000)//100*100).astype(int)
main_cam = rng.choice([8,12,13,16,24,32,48,50,64,108], n)
selfie = rng.choice([5,8,12,16,20,32], n)
g5 = ((year>=2020)&(rng.random(n)<0.7)).astype(int)
os_ = np.where(brand=="Apple","iOS","Android")
condition = rng.choice(["Used","Refurbished"], n, p=[.6,.4])
age = 2025-year
days_used = np.clip(rng.normal(age*250+100,150,n),30,2500).astype(int)
base = 6000 + ram*1500 + rom*28 + (screen-5)*2500 + (battery-2500)*1.2 + main_cam*60 + selfie*40 + g5*5000
new_price = base*np.array([brands[b] for b in brand])*(1+0.04*(year-2016)) * rng.normal(1,0.05,n)
dep = np.exp(-0.17*age - days_used/6000)
cond_f = np.where(condition=="Refurbished",1.08,1.0)
used = (new_price*dep*cond_f*rng.normal(1,0.06,n)).clip(1000)
df = pd.DataFrame(dict(brand=brand, os=os_, screen_size=screen, ram_gb=ram, storage_gb=rom,
    battery_mah=battery, main_camera_mp=main_cam, selfie_camera_mp=selfie, has_5g=g5,
    release_year=year, days_used=days_used, condition=condition,
    new_price=new_price.round(0), used_price=used.round(0)))
for c in ["screen_size","battery_mah","main_camera_mp"]:
    df.loc[rng.choice(n,25,replace=False), c] = np.nan   # missing values for preprocessing demo
df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)  # a few duplicates
df.to_csv("used_mobile_data.csv", index=False)
print(df.shape)
