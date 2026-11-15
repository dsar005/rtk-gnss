"""Generate illustrative data only. Replace these files with your real datasets."""
from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(7)
out=Path("data/sample"); out.mkdir(parents=True,exist_ok=True)
settings={"Medium_Urban_Canyon":1.8,"Deep_Urban_Canyon":3.5,
          "Harsh_Urban_Canyon":5.5,"Tunnel":7.0}
for name,noise in settings.items():
    n=2400
    t=np.arange(n)*0.1
    gt_x=0.5*t+2*np.sin(t/12); gt_y=0.2*t+np.cos(t/9); gt_z=3+0.1*np.sin(t/5)
    err=rng.normal(0,noise,(n,3))
    # Tunnel has a larger GNSS degradation profile, intentionally.
    gx,gy,gz=gt_x+err[:,0],gt_y+err[:,1],gt_z+err[:,2]
    speed=np.sqrt(np.gradient(gt_x)**2+np.gradient(gt_y)**2)*10
    df=pd.DataFrame({
      "timestamp":t,"trajectory_id":np.repeat(np.arange(4),n//4),
      "gnss_x":gx,"gnss_y":gy,"gnss_z":gz,"gnss_speed":speed+rng.normal(0,.2,n),
      "gnss_cn0":np.clip(42-noise*2+rng.normal(0,3,n),5,50),
      "gnss_satellites":np.clip((16-noise+rng.normal(0,2,n)).round(),3,24),
      "gnss_hdop":np.clip(.7+noise*.25+rng.normal(0,.25,n),.3,8),
      "accel_x":np.gradient(np.gradient(gt_x)) + rng.normal(0,.08,n),
      "accel_y":np.gradient(np.gradient(gt_y)) + rng.normal(0,.08,n),
      "accel_z":rng.normal(0,.1,n),
      "gyro_x":rng.normal(0,.02,n),"gyro_y":rng.normal(0,.02,n),
      "gyro_z":rng.normal(0,.03,n),
      "gt_x":gt_x,"gt_y":gt_y,"gt_z":gt_z
    })
    df.to_csv(out/f"{name}.csv",index=False)
print("Created four illustrative client CSV files in data/sample/.")
