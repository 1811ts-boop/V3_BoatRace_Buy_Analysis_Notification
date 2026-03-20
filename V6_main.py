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
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 環境設定・定数
# =============================================================================
logger = logging.getLogger("V6_DailyRun")
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
PROJECT_IDS = ['P0_SG', 'P1_G1_Elite', 'P2_Ladies', 'P3_General_Std', 'P4_Planning']
MAX_WORKERS = 3  

CATEGORIES_DEF_S1 = {
    'PlaceID': list(range(1, 25)), 'RaceNum': list(range(1, 13)), 
    'Course_Type': [1, 2, 3, 4, 5], 'Weather_Code': [0, 1, 2, 3, 4, 5, 6], 'Tide_Trend': [-1, 0, 1]
}
CATEGORIES_DEF_S2 = CATEGORIES_DEF_S1.copy()
CATEGORIES_DEF_S2['Boat_Number'] = [1, 2, 3, 4, 5, 6]
CATEGORIES_DEF_S3 = CATEGORIES_DEF_S1.copy()
CATEGORIES_DEF_S3['Bet_Ticket'] = [f"{p[0]}-{p[1]}-{p[2]}" for p in itertools.permutations(range(1, 7), 3)]

# 🏆 真・聖杯ターゲット（テスト期間ROI 100%超え・幻排除の完全版）
TARGET_STRATEGIES = [
    {
        "pid": "P2_Ladies", 
        "s1_min": 0.34, 
        "odds_min": 110.0, 
        "form": "1st_is_top2", 
        "desc": "P2 レディース (波乱34%超/オッズ110倍+/1着上位2艇)"
    },
    {
        "pid": "P3_General_Std", 
        "s1_min": 0.29, 
        "odds_min": 110.0, 
        "form": "top_3_box", 
        "desc": "P3 一般戦A (波乱29%超/オッズ110倍+/上位3艇BOX)"
    },
    {
        "pid": "P3_General_Std", 
        "s1_min": 0.34, 
        "odds_min": 110.0, 
        "form": "top_3_box", 
        "desc": "P3 一般戦B (波乱34%超/オッズ110倍+/上位3艇BOX)"
    },
    {
        "pid": "P4_Planning", 
        "s1_min": 0.34, 
        "odds_min": 110.0, 
        "form": "1st_is_top2", 
        "desc": "P4 企画レースA (波乱34%超/オッズ110倍+/1着上位2艇)"
    },
    {
        "pid": "P4_Planning", 
        "s1_min": 0.34, 
        "odds_min": 120.0, 
        "form": "1st_is_top2", 
        "desc": "P4 企画レースB (波乱34%超/オッズ120倍+/1着上位2艇)"
    }
]

JCD_MAP = {f"{i:02d}": name for i, name in enumerate(["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"], 1)}
PLACE_COORDS = {1: {"lat": 36.39, "lon": 139.30}, 2: {"lat": 35.82, "lon": 139.66}, 3: {"lat": 35.69, "lon": 139.86}, 4: {"lat": 35.58, "lon": 139.73}, 5: {"lat": 35.65, "lon": 139.51}, 6: {"lat": 34.69, "lon": 137.56}, 7: {"lat": 34.82, "lon": 137.21}, 8: {"lat": 34.88, "lon": 136.82}, 9: {"lat": 34.68, "lon": 136.51}, 10: {"lat": 36.21, "lon": 136.16}, 11: {"lat": 35.01, "lon": 135.85}, 12: {"lat": 34.60, "lon": 135.47}, 13: {"lat": 34.71, "lon": 135.38}, 14: {"lat": 34.20, "lon": 134.60}, 15: {"lat": 34.30, "lon": 133.79}, 16: {"lat": 34.46, "lon": 133.81}, 17: {"lat": 34.29, "lon": 132.30}, 18: {"lat": 34.03, "lon": 131.81}, 19: {"lat": 33.99, "lon": 130.98}, 20: {"lat": 33.89, "lon": 130.75}, 21: {"lat": 33.88, "lon": 130.66}, 22: {"lat": 33.59, "lon": 130.39}, 23: {"lat": 33.43, "lon": 129.98}, 24: {"lat": 32.89, "lon": 129.96}}
TRACK_ANGLES = {1: 163.6, 2: 101.6, 3: 17.8, 4: 355.6, 5: 273.4, 6: 187.0, 7: 243.7, 8: 271.3, 9: 282.1, 10: 152.9, 11: 192.5, 12: 186.1, 13: 250.5, 14: 109.5, 15: 333.3, 16: 181.1, 17: 228.7, 18: 299.0, 19: 222.8, 20: 244.1, 21: 90.9, 22: 68.1, 23: 212.4, 24: 50.6}
COURSE_TYPE_MAP = {24: 1, 18: 1, 21: 1, 19: 1, 13: 1, 10: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 1: 2, 12: 3, 15: 3, 16: 3, 17: 3, 20: 3, 23: 3, 2: 4, 4: 4, 14: 4, 11: 4, 22: 4, 3: 5}

# =============================================================================
# 2. Google Drive & API連携 (インフラ対応)
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
    logger.info("🤖 V6 AIモデル（.pkl）の最新版をダウンロードします...")
    os.makedirs("Models_Stage1_V6", exist_ok=True)
    os.makedirs("Models_Stage2_Ranker_V6", exist_ok=True)
    os.makedirs("Models_Stage3_Odds_V6", exist_ok=True)
    
    download_latest_file_by_name(service, TIDE_CSV_NAME)
    
    for pid in PROJECT_IDS:
        download_latest_file_by_name(service, f"LGBM_Stage1_V6_Ensemble_{pid}.pkl", "Models_Stage1_V6")
        download_latest_file_by_name(service, f"LGBM_Stage1_V6_Features_{pid}.pkl", "Models_Stage1_V6")
        download_latest_file_by_name(service, f"LGBM_Stage2_Ranker_V6_{pid}.pkl", "Models_Stage2_Ranker_V6")
        download_latest_file_by_name(service, f"LGBM_Stage2_Ranker_V6_Features_{pid}.pkl", "Models_Stage2_Ranker_V6")
        download_latest_file_by_name(service, f"LGBM_Stage3_Odds_V6_{pid}.pkl", "Models_Stage3_Odds_V6")
        download_latest_file_by_name(service, f"LGBM_Stage3_Odds_V6_Features_{pid}.pkl", "Models_Stage3_Odds_V6")

def send_line_broadcast(msg):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        logger.warning("⚠️ LINEトークンが環境変数に設定されていません。")
        return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    try:
        resp = requests.post(url, headers=headers, json={"messages": [{"type": "text", "text": msg}]})
        if resp.status_code != 200:
            logger.error(f"❌ LINE API Error: Status {resp.status_code}, Response: {resp.text}")
        else:
            logger.info("✅ LINEへのメッセージ送信に成功しました。")
    except Exception as e:
        logger.error(f"❌ LINEリクエスト送信中に例外が発生しました: {e}")

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
    
    flags = {'Is_SG': 1 if 'is-SG' in title_class else 0, 'Is_G1': 1 if 'is-G1' in title_class else 0, 'Is_Rookie': 1 if any(w in r_name for w in ['ルーキー', 'ヤング', '若手']) else 0}
    flags['Is_Lady'] = 1 if 'is-lady' in title_class or 'ヴィーナス' in r_name or 'オールレディース' in r_name else 0
    flags['Is_General'] = 1 if sum([flags['Is_SG'], flags['Is_G1'], 1 if 'is-G2' in title_class else 0, 1 if 'is-G3' in title_class else 0]) == 0 else 0
    
    pid = "P3_General_Std"
    if flags['Is_SG']: pid = "P0_SG"
    elif flags['Is_G1'] and not flags['Is_Rookie']: pid = "P1_G1_Elite"
    elif flags['Is_Lady']: pid = "P2_Ladies"
    elif flags['Is_General'] and (r == 1 or "進入固定" in r_name or "シード" in r_name): pid = "P4_Planning"

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
            r_data[f"R{i}_Boat_2Ren"] = list(tds[7].stripped_strings)[1] 
        except: pass

    row = {'Date': date_str, 'PlaceID': jcd, 'RaceNum': r, 'Scheduled_Time': sched, 'Project_ID': pid}
    row.update(r_data)
    row.update(extract_additional_data(soup))
    return row

def scrape_today(today_obj):
    tasks = [(today_obj.strftime('%Y%m%d'), f"{j:02d}", r) for j in range(1, 25) for r in range(1, 13)]
    res = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as e:
        for f in concurrent.futures.as_completed({e.submit(parse_today_race, t): t for t in tasks}):
            try:
                if f.result(): res.append(f.result())
            except: pass
    return pd.DataFrame(res)

# =============================================================================
# 4. 特徴量生成 
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

def transform_for_inference_v6(df_raw, df_tide):
    features_list = []
    error_count = 0
    
    WEIGHTS_14 = np.array([max(0.2, (15 - i) * 0.1) for i in range(1, 15)][::-1])
    
    for _, row in df_raw.iterrows():
        pid, rnum, dt = int(safe_float(row.get('PlaceID'))), int(safe_float(row.get('RaceNum'))), int(row.get('Date', 0))
        race_id = f"{dt}_{pid}_{rnum}"
        
        win_nat_sum = sum(safe_float(row.get(f"R{b}_WinRate_National")) for b in range(1, 7))
        if win_nat_sum < 15.0: 
            error_count += 1
            logger.warning(f"⚠️ {pid}場 {rnum}R: 勝率データ異常。推論をスキップします。")
            continue

        sched = str(row.get('Scheduled_Time', '12:00'))
        ws, wd, wc = fetch_weather(pid, sched)
        ta = TRACK_ANGLES.get(pid, 0.0)
        tw = round(ws * math.cos(math.radians((wd + 180) - ta)), 2)
        cw = round(ws * math.sin(math.radians((wd + 180) - ta)), 2)
        
        hr = int(sched.split(':')[0]) if ':' in sched else 12
        trow = df_tide[(df_tide['DateInt'] == dt) & (df_tide['PlaceID'] == pid) & (df_tide['Hour'] == hr)]
        tl = float(trow['Tide_Level_cm'].iloc[0]) if not trow.empty else 0.0
        tt = int(trow['Tide_Trend'].iloc[0]) if not trow.empty else 0
        tournament_day = safe_float(row.get('Tournament_Day'), 1.0)
        
        boats_data = {}
        for b in range(1, 7):
            win_nat = safe_float(row.get(f"R{b}_WinRate_National"))
            win_local = safe_float(row.get(f"R{b}_WinRate_Local"), win_nat) 
            motor_2ren = safe_float(row.get(f"R{b}_Motor_2Ren"))
            boat_2ren = safe_float(row.get(f"R{b}_Boat_2Ren"), 30.0)
            avg_st = safe_float(row.get(f"R{b}_Avg_ST"), 0.17)
            f_count = safe_float(row.get(f"R{b}_F_Count"), 0) 
            weight = safe_float(row.get(f"R{b}_Weight"), 51.0)
            
            st_list, rank_pt_list = [], []
            for i in range(1, 15):
                prk = safe_float(row.get(f"Boat{b}_Past_{i}_Rank"))
                pst = safe_float(row.get(f"Boat{b}_Past_{i}_ST"))
                rank_pt = {1:10, 2:8, 3:6, 4:4, 5:2, 6:1}.get(prk, 0)
                st_list.append(pst if pst > 0 else np.nan)
                rank_pt_list.append(rank_pt)
                
            st_arr = np.array(st_list)
            st_arr = st_arr[~np.isnan(st_arr)]
            true_recent_st = np.mean(st_arr) if len(st_arr) > 0 else avg_st
            
            rank_arr = np.array(rank_pt_list)
            true_momentum = np.sum(rank_arr * WEIGHTS_14[-len(rank_arr):]) if len(rank_arr) > 0 else 0.0
            
            raw_power = win_nat + (win_local * 0.5) + (motor_2ren / 10.0) + (true_momentum / 10.0 * 0.3) - (true_recent_st * 10) - (f_count * 0.5)
            
            boats_data[b] = {
                'win_nat': win_nat, 'win_local': win_local, 'motor': motor_2ren, 'boat': boat_2ren,
                'avg_st': avg_st, 'f_count': f_count, 'weight': weight,
                'recent_avg_st': true_recent_st, 'momentum': true_momentum, 'power': raw_power
            }

        b1_power = boats_data[1]['power']
        rival_powers = [boats_data[b]['power'] for b in range(2, 7)]
        dash_powers = [boats_data[b]['power'] for b in range(4, 7)]
        
        b1_advantage = b1_power - max(rival_powers) 
        wall_st = (boats_data[2]['recent_avg_st'] + boats_data[3]['recent_avg_st']) / 2.0 
        dash_threat = max(dash_powers) 
        
        feat = {
            'Race_ID': race_id, 'DateInt': dt, 'PlaceID': pid, 'RaceNum': rnum, 'Project_ID_Calc': row.get('Project_ID'),
            'Course_Type': COURSE_TYPE_MAP.get(pid, 3), 'Month_Cos': math.cos(2 * math.pi * int(str(dt)[4:6]) / 12.0),
            'Weather_Code': wc, 'Tailwind_Comp': tw, 'Crosswind_Comp': cw,
            'Tide_Level_cm': tl, 'Tide_Trend': tt, 'Tournament_Day': tournament_day, 
            'B1_Advantage': b1_advantage, 'Wall_ST': wall_st, 'Dash_Threat': dash_threat
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
            
    if error_count > 0:
        send_line_broadcast(f"⚠️【警告】スクレイピングデータ異常（{error_count}レース）。誤推論を防ぐためスキップしました。")
        
    return pd.DataFrame(features_list)

# =============================================================================
# 5. AI推論 ＆ LINE通知 
# =============================================================================
def run_ai_and_notify_v6(df_wide):
    buys = []
    debug_logs = {}  # 📊 デバッグログ用辞書
    
    for pid in PROJECT_IDS:
        df_pid = df_wide[df_wide['Project_ID_Calc'] == pid].copy()
        if df_pid.empty: continue
        
        try:
            # -----------------------------------
            # Stage 1: 波乱確率 (Prob_Class3)
            # -----------------------------------
            with open(os.path.join("Models_Stage1_V6", f"LGBM_Stage1_V6_Ensemble_{pid}.pkl"), 'rb') as f: s1_models = pickle.load(f)
            with open(os.path.join("Models_Stage1_V6", f"LGBM_Stage1_V6_Features_{pid}.pkl"), 'rb') as f: s1_features = pickle.load(f)
            
            for col, cats in CATEGORIES_DEF_S1.items():
                if col in df_pid.columns:
                    df_pid[col] = df_pid[col].fillna(cats[0]).astype(int)
                    df_pid[col] = pd.Categorical(df_pid[col], categories=cats, ordered=False)
            
            preds = np.zeros((len(df_pid), 4))
            for model in s1_models: 
                preds += model.predict(df_pid[s1_features]) / len(s1_models)
            
            for i in range(4): df_pid[f'Prob_Class{i}'] = preds[:, i]
            
            # 📊 ログ収集用: このプロジェクトで処理された全レース情報を保存
            pid_targets = [t for t in TARGET_STRATEGIES if t['pid'] == pid]
            for _, r in df_pid.iterrows():
                plid = int(r['PlaceID'])
                rnum = int(r['RaceNum'])
                prob = float(r['Prob_Class3'])
                if plid not in debug_logs: debug_logs[plid] = []
                debug_logs[plid].append({'rnum': rnum, 'pid': pid, 'prob': prob, 'pid_targets': pid_targets})
            
            # -----------------------------------
            # Stage 2: Ranker推論 (Wide -> Long)
            # -----------------------------------
            base_cols = ['Race_ID', 'DateInt', 'PlaceID', 'RaceNum', 'Course_Type', 'Month_Cos', 'Weather_Code', 'Tailwind_Comp', 'Crosswind_Comp', 'Tide_Level_cm', 'Tide_Trend', 'Tournament_Day', 'B1_Advantage', 'Wall_ST', 'Dash_Threat', 'Prob_Class0', 'Prob_Class1', 'Prob_Class2', 'Prob_Class3']
            boat_specific_cols = ['WinRate_Nat', 'WinRate_Local', 'Motor_2Ren', 'Boat_2Ren', 'F_Count', 'Weight', 'Recent_ST', 'Momentum', 'Inside_F_Count', 'Outside_F_Count', 'ST_Advantage_Inside', 'ST_Advantage_Outside', 'WinRate_Diff_Inside', 'WinRate_Diff_Outside', 'Motor_Diff_Inside', 'Motor_Diff_Outside']
            
            long_dfs = []
            for b in range(1, 7):
                extract_cols = [c for c in base_cols + [f'B{b}_{c}' for c in boat_specific_cols] if c in df_pid.columns]
                temp = df_pid[extract_cols].copy()
                temp.rename(columns={f'B{b}_{c}': c for c in boat_specific_cols}, inplace=True)
                temp['Boat_Number'] = b
                long_dfs.append(temp)
                
            df_long = pd.concat(long_dfs, ignore_index=True)
            df_long['PlaceID_Sort'] = pd.to_numeric(df_long['PlaceID'], errors='coerce').fillna(0).astype(int)
            df_long['RaceNum_Sort'] = pd.to_numeric(df_long['RaceNum'], errors='coerce').fillna(0).astype(int)
            df_long = df_long.sort_values(['DateInt', 'PlaceID_Sort', 'RaceNum_Sort', 'Boat_Number']).reset_index(drop=True)
            
            with open(os.path.join("Models_Stage2_Ranker_V6", f"LGBM_Stage2_Ranker_V6_{pid}.pkl"), 'rb') as f: s2_model = pickle.load(f)
            with open(os.path.join("Models_Stage2_Ranker_V6", f"LGBM_Stage2_Ranker_V6_Features_{pid}.pkl"), 'rb') as f: s2_features = pickle.load(f)
            
            for col, cats in CATEGORIES_DEF_S2.items():
                if col in df_long.columns:
                    df_long[col] = df_long[col].fillna(cats[0]).astype(int)
                    df_long[col] = pd.Categorical(df_long[col], categories=cats, ordered=False)
                    
            df_long['Ranker_Score'] = s2_model.predict(df_long[s2_features])
            df_long['AI_Rank'] = df_long.groupby('Race_ID')['Ranker_Score'].rank(ascending=False, method='min')
            
            # -----------------------------------
            # Stage 3: 疑似オッズ予測とフォーメーション生成
            # -----------------------------------
            bets = []
            for race_id, group in df_long.groupby('Race_ID'):
                sorted_boats = group.sort_values('AI_Rank')['Boat_Number'].values
                top4 = sorted_boats[:4]; top3 = sorted_boats[:3]; top2 = sorted_boats[:2]
                
                for p in itertools.permutations(top4, 3):
                    is_top3 = all(x in top3 for x in p)
                    is_1st_top2 = (p[0] in top2) and (p[1] in top4) and (p[2] in top4)
                    bets.append({
                        'Race_ID': race_id, 'Bet_Ticket': f"{p[0]}-{p[1]}-{p[2]}",
                        'Is_top_4_box': True, 'Is_top_3_box': is_top3, 'Is_1st_is_top2': is_1st_top2
                    })
                    
            df_bets = pd.DataFrame(bets)
            df_s3 = pd.merge(df_bets, df_pid, on='Race_ID', how='inner')
            
            with open(os.path.join("Models_Stage3_Odds_V6", f"LGBM_Stage3_Odds_V6_{pid}.pkl"), 'rb') as f: s3_model = pickle.load(f)
            with open(os.path.join("Models_Stage3_Odds_V6", f"LGBM_Stage3_Odds_V6_Features_{pid}.pkl"), 'rb') as f: s3_features = pickle.load(f)
            
            for col, cats in CATEGORIES_DEF_S3.items():
                if col in df_s3.columns:
                    df_s3[col] = df_s3[col].fillna(cats[0]).astype(int) if col != 'Bet_Ticket' else df_s3[col]
                    df_s3[col] = pd.Categorical(df_s3[col], categories=cats, ordered=False)
                    
            df_bets['Predicted_Odds'] = np.expm1(s3_model.predict(df_s3[s3_features]))
            df_cache = pd.merge(df_bets, df_pid[['Race_ID', 'PlaceID', 'RaceNum', 'Prob_Class3']], on='Race_ID', how='left')
            
            # -----------------------------------
            # 抽出ロジック（真・聖杯ターゲット）
            # -----------------------------------
            for target in TARGET_STRATEGIES:
                if target['pid'] != pid: continue
                
                df_hit = df_cache[
                    (df_cache['Prob_Class3'] >= target['s1_min']) &
                    (df_cache[f"Is_{target['form']}"] == True) &
                    (df_cache['Predicted_Odds'] >= target['odds_min'])
                ]
                
                if not df_hit.empty:
                    for race_id, grp in df_hit.groupby('Race_ID'):
                        place_id = int(grp['PlaceID'].iloc[0])
                        race_num = int(grp['RaceNum'].iloc[0])
                        tickets = grp.sort_values('Predicted_Odds', ascending=False)['Bet_Ticket'].tolist()
                        
                        buys.append({
                            'p': place_id, 'r': race_num, 
                            'desc': target['desc'], 'tickets': tickets
                        })
                        
        except Exception as e: 
            logger.error(f"AI Error ({pid}): {e}")

    # -----------------------------------
    # 📊 デバッグログ出力
    # -----------------------------------
    logger.info("📊 === AI推論結果の詳細レポート ===")
    for plid in sorted(debug_logs.keys()):
        place_name = JCD_MAP.get(f"{plid:02d}", "不明")
        races = sorted(debug_logs[plid], key=lambda x: x['rnum'])
        pid_groups = {}
        for r in races:
            pid_groups.setdefault(r['pid'], []).append(r)
            
        for p_id, p_races in pid_groups.items():
            targets = p_races[0]['pid_targets']
            target_str = "ターゲット条件: " + " / ".join([t['desc'] for t in targets]) if targets else "ターゲット条件: 設定なし（見送り対象）"
            logger.info(f"🚤 {place_name} ({p_id}) - 全{len(p_races)}レース分析完了 | {target_str}")
            
            for r in p_races:
                rnum = r['rnum']
                prob = r['prob']
                # 抽出された買い目の中に、このレースが含まれているかチェック
                race_buys = [b for b in buys if b['p'] == plid and b['r'] == rnum and b.get('desc') in [t['desc'] for t in targets]]
                match_mark = "✅ 条件クリア（抽出済）" if race_buys else "❌ スルー"
                
                logger.info(f"    {rnum:>2}R: 波乱予測({prob*100:.1f}%) -> {match_mark}")
    logger.info("======================================")

    # -----------------------------------
    # LINE通知フォーマット
    # -----------------------------------
    if not buys:
        msg = f"🤖 【V6 真・聖杯AI】\n📅 {TODAY_OBJ.strftime('%Y年%m月%d日')}\n本日はウォーク・フォワード・テストをクリアした「聖杯条件」に合致するレースがありませんでした🙅‍♂️"
        logger.info("本日は条件合致レースがありませんでした。LINEに通知します。")
        send_line_broadcast(msg)
    else:
        msg = f"🤖 【V6 真・聖杯AI】\n📅 {TODAY_OBJ.strftime('%Y年%m月%d日')}\n✅ 合致：{len(buys)}レース\n"
        # 開催地 > レース番号 順にソート
        for b in sorted(buys, key=lambda x: (x['p'], x['r'])):
            place_name = JCD_MAP.get(f"{b['p']:02d}", "不明")
            ticket_str = "\n".join([f"🎯 {t}" for t in b['tickets']])
            msg += f"\n🚤 {place_name} {b['r']}R\n【{b['desc']}】\n{ticket_str}\n"
            
        send_line_broadcast(msg)
        logger.info(f"買い目送信完了: {len(buys)}レース")
        
def main():
    logger.info("V6 System Start (True Alpha Edition)")
    srv = get_drive_service()
    if srv: prepare_ai_models(srv)
    
    dtide = pd.read_csv(TIDE_CSV_NAME) if os.path.exists(TIDE_CSV_NAME) else pd.DataFrame(columns=['DateInt', 'PlaceID', 'Hour', 'Tide_Level_cm', 'Tide_Trend'])
    df = scrape_today(TODAY_OBJ)
    if df.empty:
        logger.info("データ取得不可（開催なし、またはメンテ）")
        return
        
    df_wide = transform_for_inference_v6(df, dtide)
    if not df_wide.empty:
        run_ai_and_notify_v6(df_wide)
        
    logger.info("Daily Job Completed.")

if __name__ == "__main__": 
    main()
