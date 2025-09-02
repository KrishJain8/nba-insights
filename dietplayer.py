# dietplayer.py — "Diet" versions of NBA stars (projection first, then shots with caching)

import os, json, time, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from nba_api.stats.endpoints import leaguedashplayerstats, shotchartdetail
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
import requests

SEASON = "2023-24"   # start with a completed season
MIN_GP  = 15
MIN_MPG = 10

# how many projection-nearest players to fetch shots for
TOP_K_PRE_SHOTS = 50

# requests / nba_api safety
REQUEST_TIMEOUT = 10.0     # seconds per call
RETRY_TIMES     = 2
RETRY_SLEEP     = 1.0      # seconds between retries
POLITE_SLEEP    = 0.6      # between successful NBA calls to avoid rate limits

# cache shots here
CACHE_DIR = Path("shots_cache")
CACHE_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn").lower().strip()

def fetch_leaguedash(season: str, per_mode: str, measure: str) -> pd.DataFrame:
    return leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed=per_mode,              # "PerGame" | "Per100Possessions"
        measure_type_detailed_defense=measure,   # "Base" | "Advanced" | "Usage"
        league_id_nullable="00",
        timeout=REQUEST_TIMEOUT,
    ).get_data_frames()[0]

def shots_cache_key(player_id: int, season: str) -> Path:
    return CACHE_DIR / f"{player_id}_{season.replace('/','-')}.json"

def read_shots_cache(player_id: int, season: str) -> Optional[Dict[str, float]]:
    fp = shots_cache_key(player_id, season)
    if fp.exists():
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: float(v) for k, v in data.items()}
        except Exception:
            return None
    return None

def write_shots_cache(player_id: int, season: str, dist: Dict[str, float]) -> None:
    fp = shots_cache_key(player_id, season)
    try:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(dist, f)
    except Exception:
        pass

def fetch_shot_distribution(player_id: int, season: str) -> Optional[Dict[str, float]]:
    """
    Return dict {SHOT_ZONE_BASIC: pct_of_attempts} with caching + retries.
    """
    # cache first
    cached = read_shots_cache(player_id, season)
    if cached is not None:
        return cached

    last_err = None
    for attempt in range(RETRY_TIMES + 1):
        try:
            shots = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_type_all_star="Regular Season",
                season_nullable=season,
                context_measure_simple="FGA",
                timeout=REQUEST_TIMEOUT,
            ).get_data_frames()[0]

            if shots is None or shots.empty:
                return None

            counts = shots.groupby("SHOT_ZONE_BASIC")["SHOT_ATTEMPTED_FLAG"].count()
            dist = (counts / counts.sum()).to_dict()
            write_shots_cache(player_id, season, dist)
            time.sleep(POLITE_SLEEP)
            return dist
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP)
    # give up for this player
    return None

# -----------------------------
# Build master table (Per100 + PerGame filter)
# -----------------------------
def build_table(season: str, min_gp: int, min_mpg: int) -> Tuple[pd.DataFrame, List[str]]:
    # pull per100, but only keep identifiers + needed features (avoid GP/MIN collisions)
    base_all = fetch_leaguedash(season, "Per100Possessions", "Base")
    adv_all  = fetch_leaguedash(season, "Per100Possessions", "Advanced")
    pergame  = fetch_leaguedash(season, "PerGame", "Base")[["PLAYER_ID", "GP", "MIN"]]

    base_feats_wanted = ["PTS","AST","REB","STL","BLK","TOV","FG3A","FTA"]
    adv_feats_wanted  = ["TS_PCT","EFG_PCT","USG_PCT","AST_PCT","REB_PCT","OREB_PCT","DREB_PCT"]

    base_cols = ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION"] + [c for c in base_feats_wanted if c in base_all.columns]
    adv_cols  = ["PLAYER_ID"] + [c for c in adv_feats_wanted if c in adv_all.columns]

    base = base_all[base_cols].copy()
    adv  = adv_all[adv_cols].copy()

    df = base.merge(adv, on="PLAYER_ID", how="left").merge(pergame, on="PLAYER_ID", how="left")

    # filters
    df = df[(df["GP"] >= min_gp) & (df["MIN"] >= min_mpg)].copy().reset_index(drop=True)

    # final features available
    features = [c for c in (base_feats_wanted + adv_feats_wanted) if c in df.columns]
    if len(features) < 8:
        raise SystemExit(f"Too few features available ({len(features)}). Present: {features}")

    # clean + zscore
    df[features] = (
        df[features]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df[features].median())
    )
    X = StandardScaler().fit_transform(df[features])
    df["VECTOR"] = list(X)

    # normalized name for accent-insensitive matching
    df["NAME_NORM"] = df["PLAYER_NAME"].apply(normalize_name)
    return df, features

# -----------------------------
# Diet ranking
# -----------------------------
def rank_diet_versions(
    df: pd.DataFrame,
    star_query: str,
    season: str,
    top_k_pre: int = TOP_K_PRE_SHOTS,
    top_n: int = 10,
    weights: Optional[np.ndarray] = None,
) -> Tuple[str, pd.DataFrame]:
    """
    1) Projection-only ranking over whole pool (fast).
    2) Take top_k_pre and fetch shots for them + star; compute Wasserstein; rerank.
    """
    # locate star (exact normalized else contains)
    q = normalize_name(star_query)
    hits = df.index[df["NAME_NORM"] == q]
    if len(hits) == 0:
        hits = df.index[df["NAME_NORM"].str.contains(q)]
    if len(hits) == 0:
        raise SystemExit(f'Star "{star_query}" not found.')

    i_star = int(hits[0])
    star_name = df.loc[i_star, "PLAYER_NAME"]
    star_vec  = np.array(df.loc[i_star, "VECTOR"])

    # optional feature weights
    if weights is not None:
        if len(weights) != len(df.iloc[0]["VECTOR"]):
            raise ValueError("weights length must match vector length")
        star_vec = star_vec * weights

    # 1) projection similarity across all players
    proj_rows = []
    for i, row in df.iterrows():
        if i == i_star: 
            continue
        v = np.array(row["VECTOR"])
        if weights is not None:
            v = v * weights
        denom = np.linalg.norm(v) * np.linalg.norm(star_vec)
        proj = (v @ star_vec) / denom if denom > 0 else 0.0
        proj_rows.append((i, row["PLAYER_NAME"], row["TEAM_ABBREVIATION"], proj))

    proj_df = pd.DataFrame(proj_rows, columns=["idx","PLAYER_NAME","TEAM","projection_similarity"])
    proj_df = proj_df.sort_values("projection_similarity", ascending=False).reset_index(drop=True)

    # 2) limit to top_k_pre and compute Wasserstein on shot zones
    candidate_idxs = proj_df["idx"].head(top_k_pre).tolist()

    # fetch star shots once (cached)
    star_shot = fetch_shot_distribution(int(df.loc[i_star, "PLAYER_ID"]), season)

    w_rows = []
    for i in candidate_idxs:
        player_id = int(df.loc[i, "PLAYER_ID"])
        wdist = np.nan
        if star_shot is not None:
            p_shot = fetch_shot_distribution(player_id, season)
            if p_shot is not None:
                cats = sorted(set(star_shot.keys()) | set(p_shot.keys()))
                s = np.array([star_shot.get(c,0.0) for c in cats], dtype=float)
                p = np.array([p_shot.get(c,0.0) for c in cats], dtype=float)
                wdist = wasserstein_distance(p, s)
        w_rows.append((i, wdist))

    w_df = pd.DataFrame(w_rows, columns=["idx","shot_distance"])
    out = proj_df.merge(w_df, on="idx", how="left")

    # rank: high projection, low shot distance (NaN last)
    fill_val = out["shot_distance"].max() if out["shot_distance"].notna().any() else 1.0
    out["shot_distance_filled"] = out["shot_distance"].fillna(fill_val)
    out = out.sort_values(["projection_similarity","shot_distance_filled"], ascending=[False,True])

    return star_name, out[["PLAYER_NAME","TEAM","projection_similarity","shot_distance"]].head(top_n)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Building table (Per100 Base+Advanced) ...")
    df, features = build_table(SEASON, MIN_GP, MIN_MPG)
    print(f"Built table with {len(df)} players, {len(features)} features\n")

    # Example: give the star name at prompt
    try:
        star_in = input("Enter a star player's name (accents optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        star_in = "Nikola Jokic"

    # Optional weights example (same length as vector). Uncomment to emphasize passing/rebounding:
    # weights = np.ones(len(features))
    # for nm in ["AST","AST_PCT","REB_PCT","OREB_PCT","DREB_PCT"]:
    #     if nm in features:
    #         weights[features.index(nm)] = 1.3
    # star_name, diet = rank_diet_versions(df, star_in, season=SEASON, top_k_pre=TOP_K_PRE_SHOTS, top_n=10, weights=weights)

    star_name, diet = rank_diet_versions(df, star_in, season=SEASON, top_k_pre=TOP_K_PRE_SHOTS, top_n=10, weights=None)
    print(f"\nDiet versions of {star_name}:")
    print(diet.to_string(index=False))
