import os
import json
import time
import math
import re
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import itertools
import concurrent.futures
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import logging
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 環境設定・定数
# =============================================================================
logger = logging.getLogger("V5_Daily_EVSniper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(sh)

JST = timezone(timedelta(hours=9), 'JST')
TODAY_OBJ = datetime.now(JST)

GCP_SA_CREDENTIALS = os.environ.get("GCP_SA_CREDENTIALS")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

TIDE_CSV_NAME = "Tide_Master_2020_2026.csv"
# 🚀 P3, P4を完全排除。証明された市場のみに絞る
PROJECT_IDS = ['P0_SG', 'P1_G1_Elite', 'P2_Ladies']

MAX_WORKERS = 3  

# 🚀 V5 共通カテゴリ定義
CATEGORIES_DEF_S1 = {
    'PlaceID': list(range(1, 25)), 'RaceNum': list(range(1, 13)), 
    'Course_Type': [1, 2, 3, 4, 5], 'Weather_Code': [0, 1, 2, 3, 4, 5, 6], 'Tide_Trend': [-1, 0, 1]
}
CATEGORIES_DEF_S2 = CATEGORIES_DEF_S1.copy(); CATEGORIES_DEF_S2['Boat_Number'] = [1, 2, 3, 4, 5, 6]
CATEGORIES_DEF_S3 = CATEGORIES_DEF_S1.copy()
# 今回は1頭の20通りのみ
CATEGORIES_DEF_S3['Bet_Ticket'] = [f"1-{p[0]}-{p[1]}" for p in itertools.permutations(range(2, 7), 2)]

# 🏆 【V5 中穴EVスナイパー】 OOSでROI 100%超えが証明された聖杯条件
HOLY_GRAIL = {
    'P0_SG':       {'S1_th': 0.50, 'Min_Odd': 20.0, 'Max_Odd': 30.0, 'EV_th': 7.0},
    'P1_G1_Elite': {'S1_th': 0.60, 'Min_Odd': 10.0, 'Max_Odd': 50.0, 'EV_th': 20.0},
    'P2_Ladies':   {'S1_th': 0.70, 'Min_Odd': 25.0, 'Max_Odd': 50.0, 'EV_th': 20.0}
}

JCD_MAP = {f"{i:02d}": name for i, name in enumerate(["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"], 1)}
PLACE_COORDS = {1: {"lat": 36.39, "lon": 139.30}, 2: {"lat": 35.82, "lon": 139.66}, 3: {"lat": 35.69, "lon": 139.86}, 4: {"lat": 35.58, "lon": 139.73}, 5: {"lat": 35.65, "lon": 139.51}, 6: {"lat": 34.69, "lon": 137.56}, 7: {"lat": 34.82, "lon": 137.21}, 8: {"lat": 34.88, "lon": 136.82}, 9: {"lat": 34.68, "lon": 136.51}, 10: {"lat": 36.21, "lon": 136.16}, 11: {"lat": 35.01, "lon": 135.85}, 12: {"lat": 34.60, "lon": 135.47}, 13: {"lat": 34.71, "lon": 135.38}, 14: {"lat": 34.20, "lon": 134.60}, 15: {"lat": 34.30, "lon": 133.79}, 16: {"lat": 34.46, "lon": 133.81}, 17: {"lat": 34.29, "lon": 132.30}, 18: {"lat": 34.03, "lon": 131.81}, 19: {"lat": 33.99, "lon": 130.98}, 20: {"lat": 33.89, "lon": 130.75}, 21: {"lat": 33.88, "lon": 130.66}, 22: {"lat": 33.59, "lon": 130.39}, 23: {"lat": 33.43, "lon": 129.98}, 24: {"lat": 32.89, "lon": 129.96}}
TRACK_ANGLES = {1: 163.6, 2: 101.6, 3: 17.8, 4: 355.6, 5: 273.4, 6: 187.0, 7: 243.7, 8: 271.3, 9: 282.1, 10: 152.9, 11: 192.5, 12: 186.1, 13: 250.5, 14: 109.5, 15: 333.3, 16: 181.1, 17: 228.7, 18: 299.0, 19: 222.8, 20: 244.1, 21: 90.9, 22: 68.1, 23: 212.4, 24: 50.6}
COURSE_TYPE_MAP = {24: 1, 18: 1, 21: 1, 19: 1, 13: 1, 10: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 1: 2, 12: 3, 15: 3, 16: 3, 17: 3, 20: 3, 23: 3, 2: 4, 4: 4, 14: 4, 11: 4, 22: 4, 3: 5}

# =============================================================================
# 2. Google Drive & API連携
# =============================================================================
def get_drive_service():
    if not GCP_SA_CREDENTIALS: return None
    creds_dict = json.loads(GCP_SA_CREDENTIALS)
    creds = service_account.Credentials.from_service_account_info(creds_dict)
    return build('drive', 'v3', credentials=creds)

def download_latest_file_by_name(service, file_name, save_dir="."):
    query = f"name='{file_name}' and trashed=false"
    res = service.files().list(q=query, orderBy="createdTime desc", fields="files(id, name)").execute()
    if not res.get('files'): return False
    file_id = res['files'][0]['id']
    req = service.files().get_media(fileId=file_id)
    with io.FileIO(os.path.join(save_dir, file_name), 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done: _, done = downloader.next_chunk()
    return True

def prepare_ai_models(service):
    logger.info("🤖 V5 AIモデル一式（.pkl）をダウンロードします...")
    os.makedirs("Models_Stage1_V5", exist_ok=True)
    os.makedirs("Models_Stage2_V5", exist_ok=True)
    os.makedirs("Models_Stage3_V5", exist_ok=True)
    
    download_latest_file_by_name(service, TIDE_CSV_NAME)
    for pid in PROJECT_IDS:
        download_latest_file_by_name(service, f"LGBM_Stage1_V5_Ensemble_{pid}.pkl", "Models_Stage1_V5")
        download_latest_file_by_name(service, f"LGBM_Stage1_V5_Features_{pid}.pkl", "Models_Stage1_V5")
        download_latest_file_by_name(service, f"LGBM_Stage2_Ranker_V5_{pid}.pkl", "Models_Stage2_V5")
        download_latest_file_by_name(service, f"LGBM_Stage2_Ranker_V5_Features_{pid}.pkl", "Models_Stage2_V5")
        download_latest_file_by_name(service, f"LGBM_Stage3_Odds_V5_{pid}.pkl", "Models_Stage3_V5")
        download_latest_file_by_name(service, f"LGBM_Stage3_Odds_V5_Features_{pid}.pkl", "Models_Stage3_V5")

def send_line_broadcast(msg):
    if not LINE_CHANNEL_ACCESS_TOKEN: return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    try:
        resp = requests.post(url, headers=headers, json={"messages": [{"type": "text", "text": msg}]})
        if resp.status_code != 200: logger.error(f"❌ LINE Error: {resp.text}")
    except Exception as e:
        logger.error(f"❌ LINE Exception: {e}")

WEATHER_CACHE = {}
def fetch_weather(place_id, target_time_str):
    try:
        hour = target_time_str.split(':')[0]
        cache_key = f"{place_id}_{hour}"
        if cache_key in WEATHER_CACHE: return WEATHER_CACHE[cache_key]

        hour_int, minute_int = map(int, target_time_str.split(':'))
        now = TODAY_OBJ.replace(hour=hour_int, minute=minute_int, second=0, microsecond=0)
        coords = PLACE_COORDS[place_id]
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={coords['lat']}&lon={coords['lon']}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        closest = min(res.get('list', []), key=lambda x: abs(x['dt'] - int(now.timestamp())))
        main = closest['weather'][0].get('main', 'Clear')
        ws = float(closest['wind'].get('speed', 0.0))
        wd = float(closest['wind'].get('deg', 0.0))
        wc = 1 if main == 'Clear' else 2 if main == 'Clouds' else 3
        WEATHER_CACHE[cache_key] = (ws, wd, wc)
        return ws, wd, wc
    except: return 0.0, 0.0, 1

# =============================================================================
# 3. スクレイピング処理
# =============================================================================
def clean_str(s): return s.replace('\u3000', '').strip() if s else ""
def clean_rank_value(val):
    if not val: return None
    v = str(val).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    return float(v) if v in ['1', '2', '3', '4', '5', '6'] else None

def fetch_soup(url, retries=3):
    for _ in range(retries):
        try:
            time.sleep(1.0) 
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
        except: time.sleep(2)
    return None

def extract_additional_data(soup_list):
    data = {}
    unique_past_dates = set()
    for boat_idx, tbody in enumerate(soup_list.find_all('tbody', class_='is-fs12')[:6], 1):
        trs = tbody.find_all('tr')
        if len(trs) < 4: continue
        tds_course, tds_st, tds_rank = trs[1].find_all('td'), trs[2].find_all('td'), trs[3].find_all('td')
        for i in range(min(len(tds_course), 14)):
            if 9 + i < len(trs[0].find_all('td')):
                data[f"Boat{boat_idx}_Past_{i+1}_Course"] = clean_str(tds_course[i].get_text(strip=True))
                data[f"Boat{boat_idx}_Past_{i+1}_ST"] = clean_str(tds_st[i].get_text(strip=True))
                data[f"Boat{boat_idx}_Past_{i+1}_Rank"] = clean_rank_value(tds_rank[i].get_text(strip=True))
        for td in tds_rank:
            a = td.find('a')
            if a and 'href' in a.attrs:
                m = re.search(r'hd=(\d{8})', a['href'])
                if m: unique_past_dates.add(m.group(1))
    data["Tournament_Day"] = str(len(unique_past_dates) + 1)
    return data

def parse_today_race(task_tuple):
    date_str, jcd, r = task_tuple
    soup = fetch_soup(f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={r}&jcd={jcd}&hd={date_str}")
    if not soup or "データがありません" in soup.text: return None

    sched = "12:00"
    try:
        time_td = soup.find('td', string=re.compile('締切予定時刻'))
        if time_td: sched = time_td.find_parent('tr').find_all('td')[r].text.strip()
    except: pass

    title_class = " ".join(soup.find('div', class_='heading2_title').get('class')) if soup.find('div', class_='heading2_title') else ""
    r_name = clean_str(soup.find('h2').text) if soup.find('h2') else ""
    
    flags = {'Is_SG': 1 if 'is-SG' in title_class else 0, 'Is_G1': 1 if 'is-G1' in title_class else 0, 'Is_Lady': 1 if 'is-G3' in title_class and 'オールレディース' in r_name else 0}
    
    # 🚀 V5: 対象外プロジェクトは容赦なくスキップ（処理の超高速化）
    pid = "P3_General_Std"
    if flags['Is_SG']: pid = "P0_SG"
    elif flags['Is_G1']: pid = "P1_G1_Elite"
    elif flags['Is_Lady']: pid = "P2_Ladies"
    
    if pid not in PROJECT_IDS: return None

    r_data = {}
    for i, tbody in enumerate(soup.find_all('tbody', class_=lambda x: x and 'is-fs12' in x)[:6], 1):
        try:
            tds = tbody.find_all('td')
            r_data[f"R{i}_Weight"] = re.split(r'\s+', tds[2].find_all('div')[2].text.strip())[1].split('/')[1].replace('kg', '')
            r_data[f"R{i}_F_Count"] = list(tds[3].stripped_strings)[0].replace('F', '')
            r_data[f"R{i}_Avg_ST"] = list(tds[3].stripped_strings)[2]
            w_nat, w_loc = list(tds[4].stripped_strings)[0], list(tds[5].stripped_strings)[0]
            r_data[f"R{i}_WinRate_National"], r_data[f"R{i}_WinRate_Local"] = w_nat, w_loc if w_loc != "0.00" else w_nat
            r_data[f"R{i}_Motor_2Ren"] = list(tds[6].stripped_strings)[1]
        except: pass

    row = {'Date': date_str, 'PlaceID': jcd, 'RaceNum': r, 'Scheduled_Time': sched, 'Project_ID': pid}
    row.update(r_data)
    row.update(extract_additional_data(soup))
    return row

def scrape_today(today_obj):
    logger.info("🚤 SG/G1/Ladies の本日のレース情報を取得中...")
    tasks = [(today_obj.strftime('%Y%m%d'), f"{j:02d}", r) for j in range(1, 25) for r in range(1, 13)]
    res = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as e:
        for f in concurrent.futures.as_completed({e.submit(parse_today_race, t): t for t in tasks}):
            try:
                if f.result(): res.append(f.result())
            except: pass
    return pd.DataFrame(res)

# =============================================================================
# 4. 特徴量生成（V5仕様に完全準拠）
# =============================================================================
def safe_float(val, default=0.0):
    if pd.isna(val) or val == "" or val is None: return default
    try: return float(val)
    except ValueError:
        s = str(val).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        clean_val = re.sub(r'[^\d.-]', '', s)
        if clean_val in ('', '-', '.', '-.'): return default
        try: return float(clean_val)
        except ValueError: return default

def transform_for_inference_v5(df_raw, df_tide):
    features_list = []
    WEIGHTS_14 = np.array([max(0.2, (15 - i) * 0.1) for i in range(1, 15)][::-1])
    
    for _, row in df_raw.iterrows():
        pid = int(safe_float(row.get('PlaceID')))
        rnum = int(safe_float(row.get('RaceNum')))
        date_int = int(row.get('Date', 0))
        race_id = f"{date_int}_{pid}_{rnum}"
        project_id = row.get('Project_ID')

        sched = str(row.get('Scheduled_Time', '12:00'))
        ws, wd, wc = fetch_weather(pid, sched)
        ta = TRACK_ANGLES.get(pid, 0.0)
        tw = round(ws * math.cos(math.radians((wd + 180) - ta)), 2)
        cw = round(ws * math.sin(math.radians((wd + 180) - ta)), 2)
        
        hr = int(sched.split(':')[0]) if ':' in sched else 12
        trow = df_tide[(df_tide['DateInt'] == date_int) & (df_tide['PlaceID'] == pid) & (df_tide['Hour'] == hr)]
        tl = float(trow['Tide_Level_cm'].iloc[0]) if not trow.empty else 0.0
        tt = int(trow['Tide_Trend'].iloc[0]) if not trow.empty else 0
        tournament_day = safe_float(row.get('Tournament_Day'), 1.0)
        
        boats_data = {}
        for b in range(1, 7):
            win_nat = safe_float(row.get(f"R{b}_WinRate_National"))
            win_local = safe_float(row.get(f"R{b}_WinRate_Local"), win_nat) 
            motor_2ren = safe_float(row.get(f"R{b}_Motor_2Ren"))
            boat_2ren = 30.0 # 本番APIにボート勝率はないため学習時のデフォルト値
            avg_st = safe_float(row.get(f"R{b}_Avg_ST"), 0.17)
            f_count = safe_float(row.get(f"R{b}_F_Count"), 0) 
            weight = safe_float(row.get(f"R{b}_Weight"), 51.0)
            
            # V5 過去14走の集計
            sts = []
            ranks = []
            for i in range(1, 15):
                pst = safe_float(row.get(f"Boat{b}_Past_{i}_ST"))
                prk = safe_float(row.get(f"Boat{b}_Past_{i}_Rank"))
                if pst > 0: sts.append(pst)
                if prk > 0: ranks.append({1:10, 2:8, 3:6, 4:4, 5:2, 6:1}.get(prk, 0))
            
            true_recent_st = np.mean(sts) if sts else avg_st
            true_momentum = np.sum(np.array(ranks) * WEIGHTS_14[-len(ranks):]) if ranks else 0.0
            
            raw_power = win_nat + (win_local * 0.5) + (motor_2ren / 10.0) + (true_momentum / 10.0 * 0.3) - (true_recent_st * 10) - (f_count * 0.5)
            
            boats_data[b] = {
                'win_nat': win_nat, 'win_local': win_local, 'motor': motor_2ren, 'boat': boat_2ren,
                'avg_st': avg_st, 'f_count': f_count, 'weight': weight,
                'recent_avg_st': true_recent_st, 'momentum': true_momentum, 'power': raw_power
            }

        b1_power = boats_data[1]['power']
        b1_advantage = b1_power - max([boats_data[b]['power'] for b in range(2, 7)]) 
        wall_st = (boats_data[2]['recent_avg_st'] + boats_data[3]['recent_avg_st']) / 2.0 
        dash_threat = max([boats_data[b]['power'] for b in range(4, 7)]) 
        
        feat = {
            'Race_ID': race_id, 'DateInt': date_int, 'PlaceID': pid, 'RaceNum': rnum, 'Project_ID_Calc': project_id,
            'Scheduled_Time': sched,
            'Course_Type': COURSE_TYPE_MAP.get(pid, 3), 'Month_Cos': math.cos(2 * math.pi * int(str(date_int)[4:6]) / 12.0),
            'Weather_Code': wc, 'Wind_Speed': ws, 'Tailwind_Comp': tw, 'Crosswind_Comp': cw,
            'Tide_Level_cm': tl, 'Tide_Trend': tt, 'Tournament_Day': tournament_day, 
            'B1_Advantage': b1_advantage, 'Wall_ST': wall_st, 'Dash_Threat': dash_threat,
            'B1_vs_B2_Power_Diff': b1_power - boats_data[2]['power'],
            'B1_vs_B3_Power_Diff': b1_power - boats_data[3]['power'],
            'B1_vs_B2_ST_Diff': boats_data[1]['recent_avg_st'] - boats_data[2]['recent_avg_st']
        }
        
        for b in range(1, 7):
            inside_b = b - 1 if b > 1 else 1
            outside_b = b + 1 if b < 6 else 6
            
            feat[f'B{b}_WinRate_Nat'] = boats_data[b]['win_nat']
            feat[f'B{b}_WinRate_Local'] = boats_data[b]['win_local']
            feat[f'B{b}_Motor_2Ren'] = boats_data[b]['motor']
            feat[f'B{b}_Boat_2Ren'] = boats_data[b]['boat']
            feat[f'B{b}_F_Count'] = boats_data[b]['f_count']
            feat[f'B{b}_Weight'] = boats_data[b]['weight']
            feat[f'B{b}_Recent_ST'] = boats_data[b]['recent_avg_st']
            feat[f'B{b}_Momentum'] = boats_data[b]['momentum']
            feat[f'B{b}_Inside_F_Count'] = boats_data[inside_b]['f_count']
            feat[f'B{b}_Outside_F_Count'] = boats_data[outside_b]['f_count']
            feat[f'B{b}_ST_Advantage_Inside'] = boats_data[inside_b]['recent_avg_st'] - boats_data[b]['recent_avg_st']
            feat[f'B{b}_ST_Advantage_Outside'] = boats_data[outside_b]['recent_avg_st'] - boats_data[b]['recent_avg_st']
            feat[f'B{b}_WinRate_Diff_Inside'] = boats_data[b]['win_nat'] - boats_data[inside_b]['win_nat']
            feat[f'B{b}_WinRate_Diff_Outside'] = boats_data[b]['win_nat'] - boats_data[outside_b]['win_nat']
            feat[f'B{b}_Motor_Diff_Inside'] = boats_data[b]['motor'] - boats_data[inside_b]['motor']
            feat[f'B{b}_Motor_Diff_Outside'] = boats_data[b]['motor'] - boats_data[outside_b]['motor']

        features_list.append(feat)

    return pd.DataFrame(features_list)

# =============================================================================
# 5. AI推論 ＆ LINE通知（V5 EV Sniper）
# =============================================================================
def run_v5_inference_and_notify(df_wide):
    buys = []
    
    for pid in PROJECT_IDS:
        df_w = df_wide[df_wide['Project_ID_Calc'] == pid].copy()
        if df_w.empty: continue
            
        logger.info(f"🧠 {pid} の推論を開始します... (対象: {len(df_w)}レース)")
        try:
            # --- Stage 1: 鉄板確率算出 ---
            with open(f"Models_Stage1_V5/LGBM_Stage1_V5_Ensemble_{pid}.pkl", 'rb') as f: s1_models = pickle.load(f)
            with open(f"Models_Stage1_V5/LGBM_Stage1_V5_Features_{pid}.pkl", 'rb') as f: s1_features = pickle.load(f)
            
            for col, cats in CATEGORIES_DEF_S1.items():
                if col in df_w.columns: df_w[col] = pd.Categorical(df_w[col], categories=cats, ordered=False)
                    
            preds = np.zeros((len(df_w), 4))
            for m in s1_models: preds += m.predict(df_w[s1_features]) / len(s1_models)
            df_w['Prob_Class0'] = preds[:, 0] # 1着確率のみ保持
            
            # 足切り（この段階で確率が満たないレースは計算を打ち切って超高速化）
            th_s1 = HOLY_GRAIL[pid]['S1_th']
            df_w = df_w[df_w['Prob_Class0'] >= th_s1].copy()
            if df_w.empty: continue

            # --- Stage 2: Ranker ---
            base_cols = ['Race_ID', 'DateInt', 'PlaceID', 'RaceNum', 'Course_Type', 'Month_Cos', 'Weather_Code', 'Wind_Speed', 'Tailwind_Comp', 'Crosswind_Comp', 'Tide_Level_cm', 'Tide_Trend', 'Tournament_Day', 'B1_Advantage', 'Wall_ST', 'Dash_Threat', 'B1_vs_B2_Power_Diff', 'B1_vs_B3_Power_Diff', 'B1_vs_B2_ST_Diff', 'Prob_Class0']
            boat_specific_cols = ['WinRate_Nat', 'WinRate_Local', 'Motor_2Ren', 'Boat_2Ren', 'F_Count', 'Weight', 'Recent_ST', 'Momentum', 'Inside_F_Count', 'Outside_F_Count', 'ST_Advantage_Inside', 'ST_Advantage_Outside', 'WinRate_Diff_Inside', 'WinRate_Diff_Outside', 'Motor_Diff_Inside', 'Motor_Diff_Outside']
            
            long_dfs = []
            for b in range(1, 7):
                extract_cols = [c for c in base_cols + [f'B{b}_{c}' for c in boat_specific_cols] if c in df_w.columns]
                temp = df_w[extract_cols].copy()
                temp.rename(columns={f'B{b}_{c}': c for c in boat_specific_cols}, inplace=True)
                temp['Boat_Number'] = b
                long_dfs.append(temp)
            df_long = pd.concat(long_dfs, ignore_index=True).sort_values(['Race_ID', 'Boat_Number']).reset_index(drop=True)
            
            with open(f"Models_Stage2_V5/LGBM_Stage2_Ranker_V5_{pid}.pkl", 'rb') as f: s2_model = pickle.load(f)
            with open(f"Models_Stage2_V5/LGBM_Stage2_Ranker_V5_Features_{pid}.pkl", 'rb') as f: s2_features = pickle.load(f)
            
            for col, cats in CATEGORIES_DEF_S2.items():
                if col in df_long.columns: df_long[col] = pd.Categorical(df_long[col], categories=cats, ordered=False)
                    
            scores = s2_model.predict(df_long[s2_features])
            df_long['Ranker_Score'] = MinMaxScaler().fit_transform(scores.reshape(-1, 1))
            
            # --- Stage 3: Odds & EV Calculation ---
            bets = []
            for race_id in df_w['Race_ID'].unique():
                for p in itertools.permutations(range(2, 7), 2): # 1アタマの20通りのみ
                    bets.append({'Race_ID': race_id, 'Bet_Ticket': f"1-{p[0]}-{p[1]}"})
            df_bets = pd.DataFrame(bets)
            df_s3 = pd.merge(df_bets, df_w, on='Race_ID', how='inner')
            
            with open(f"Models_Stage3_V5/LGBM_Stage3_Odds_V5_{pid}.pkl", 'rb') as f: s3_model = pickle.load(f)
            with open(f"Models_Stage3_V5/LGBM_Stage3_Odds_V5_Features_{pid}.pkl", 'rb') as f: s3_features = pickle.load(f)
            for col, cats in CATEGORIES_DEF_S3.items():
                if col in df_s3.columns: df_s3[col] = pd.Categorical(df_s3[col], categories=cats, ordered=False)
                    
            df_bets['Predicted_Odds'] = np.expm1(s3_model.predict(df_s3[s3_features]))
            
            # Ticket_EV計算
            score_dict = df_long.set_index(['Race_ID', 'Boat_Number'])['Ranker_Score'].to_dict()
            prob0_dict = df_long[df_long['Boat_Number'] == 1].set_index('Race_ID')['Prob_Class0'].to_dict()
            
            def calc_ev(row):
                rid, tkt = row['Race_ID'], row['Bet_Ticket']
                b2, b3 = int(tkt[2]), int(tkt[4])
                ts = prob0_dict.get(rid, 0.0) * (score_dict.get((rid, b2), 0.0) + score_dict.get((rid, b3), 0.0))
                return ts * row['Predicted_Odds']
                
            df_bets['Ticket_EV'] = df_bets.apply(calc_ev, axis=1)
            
            # 🎯 聖杯フィルター適用（単点スナイプ抽出）
            min_o, max_o, ev_th = HOLY_GRAIL[pid]['Min_Odd'], HOLY_GRAIL[pid]['Max_Odd'], HOLY_GRAIL[pid]['EV_th']
            df_hit = df_bets[(df_bets['Predicted_Odds'] >= min_o) & (df_bets['Predicted_Odds'] <= max_o) & (df_bets['Ticket_EV'] >= ev_th)]
            
            for _, r in df_hit.iterrows():
                plid = int(r['Race_ID'].split('_')[1])
                rnum = int(r['Race_ID'].split('_')[2])
                # 重複防止とデータ整理
                buys.append({
                    'pid': pid, 'plid': plid, 'rnum': rnum, 
                    'time': df_w[df_w['Race_ID']==r['Race_ID']]['Scheduled_Time'].iloc[0],
                    'ticket': r['Bet_Ticket'], 'odds': r['Predicted_Odds'], 'ev': r['Ticket_EV']
                })
        except Exception as e: 
            logger.error(f"AI Error ({pid}): {e}")

    # 📊 LINE通知
    if not buys:
        msg = f"🤖 【V5 中穴EVスナイパー】\n📅 {TODAY_OBJ.strftime('%Y年%m月%d日')}\n本日は「全天候型・中穴バリュー投資」の厳しい基準をクリアするレースがありませんでした🍵\n（資金防衛のための見送りです）"
        send_line_broadcast(msg)
    else:
        # レースごとにグループ化
        race_groups = {}
        for b in buys:
            k = (b['plid'], b['rnum'], b['pid'], b['time'])
            race_groups.setdefault(k, []).append(b)
            
        msg = f"🤖 【V5 中穴EVスナイパー】\n📅 {TODAY_OBJ.strftime('%Y年%m月%d日')}\n🎯 本日の単点スナイプ: {len(race_groups)}レース\n"
        for (plid, rnum, pid, t), tkts in sorted(race_groups.items(), key=lambda x: x[0][3]): # 時間順
            place_name = JCD_MAP.get(f"{plid:02d}", "不明")
            grade = "SG" if pid == "P0_SG" else "G1" if pid == "P1_G1_Elite" else "Ladies(G3)"
            msg += f"\n🚤 {place_name} {rnum}R ({t} 締切)\n👑 {grade}特化エッジ\n"
            # EV順に並び替え
            for tk in sorted(tkts, key=lambda x: x['ev'], reverse=True):
                msg += f" 🔥 {tk['ticket']} (予想オッズ: {tk['odds']:.1f}倍 | EV: {tk['ev']:.1f})\n"
        send_line_broadcast(msg)
        logger.info(f"買い目送信完了: {len(race_groups)}レース")

def main():
    logger.info("V5 System Start (EV Sniper Edition)")
    srv = get_drive_service()
    if srv: prepare_ai_models(srv)
    
    dtide = pd.read_csv(TIDE_CSV_NAME) if os.path.exists(TIDE_CSV_NAME) else pd.DataFrame(columns=['DateInt', 'PlaceID', 'Hour', 'Tide_Level_cm', 'Tide_Trend'])
    df = scrape_today(TODAY_OBJ)
    
    if df.empty:
        logger.info("データ取得不可（SG/G1/Ladies開催なし、またはメンテ）")
        return
        
    df_wide = transform_for_inference_v5(df, dtide)
    if not df_wide.empty:
        run_v5_inference_and_notify(df_wide)
        
    logger.info("Daily Job Completed.")

if __name__ == "__main__": 
    main()
