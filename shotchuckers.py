# shotchuckers.py — Find NBA "ShotChuckers" using volume/usage/assist%/TS%

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguedashplayerstats
from sklearn.preprocessing import StandardScaler

SEASON = "2023-24"  # use completed season first
MIN_GP = 20
MIN_MPG = 18

def fetch(season, per_mode, measure):
    return leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed=per_mode,      # "PerGame" or "Per100Possessions"
        measure_type_detailed_defense=measure,
        league_id_nullable="00"
    ).get_data_frames()[0]

# 1) Get PerGame for filters
pergame = fetch(SEASON, "PerGame", "Base")[["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","GP","MIN"]].copy()
pergame["TOTAL_MIN"] = pergame["GP"] * pergame["MIN"]

# 2) Get Per100 Base + Advanced
base = fetch(SEASON, "Per100Possessions", "Base")[["PLAYER_ID","FGA"]]
adv  = fetch(SEASON, "Per100Possessions", "Advanced")[["PLAYER_ID","USG_PCT","AST_PCT","TS_PCT"]]

# 3) Merge
df = pergame.merge(base, on="PLAYER_ID", how="left").merge(adv, on="PLAYER_ID", how="left")

# 4) Filter rotation players
df = df[(df["GP"] >= MIN_GP) & (df["MIN"] >= MIN_MPG)].copy().reset_index(drop=True)

# 5) Features for ShotChucker score
features = ["FGA","USG_PCT","AST_PCT","TS_PCT"]

# Clean + z-score
df[features] = df[features].astype(float).replace([np.inf,-np.inf], np.nan).fillna(df[features].median())
scaler = StandardScaler()
Z = pd.DataFrame(scaler.fit_transform(df[features]), columns=[f"z_{f}" for f in features])

# 6) ShotChucker Score
df["ShotChuckerScore"] = Z["z_FGA"] + Z["z_USG_PCT"] - Z["z_AST_PCT"] - Z["z_TS_PCT"]

# 7) Rank
df = df.sort_values("ShotChuckerScore", ascending=False).reset_index(drop=True)

print(f"Top NBA ShotChuckers ({SEASON}) — GP ≥ {MIN_GP}, MPG ≥ {MIN_MPG}\n")
print(df[["PLAYER_NAME","TEAM_ABBREVIATION","GP","MIN","FGA","USG_PCT","AST_PCT","TS_PCT","ShotChuckerScore"]].head(15).to_string(index=False))
