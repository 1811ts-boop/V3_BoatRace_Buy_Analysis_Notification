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
import gc
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 環境設定・定数
# =============================================================================
logger = logging.getLogger("V9_DailyRun_LambdaRank")
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
MASTER_CSV_NAME = "BoatRace_Master_Updated_with_2Tan_2Fuku.csv"
PORTFOLIO_2TAN_CSV = "V10_Portfolio_OOS_Detailed_2Tan_Results.csv"
PORTFOLIO_2FUKU_CSV = "V10_Portfolio_OOS_Detailed_2Fuku_Results.csv"

PROJECT_IDS = ['P0_SG', 'P1_G1_Elite', 'P2_Ladies', 'P3_General_Std', 'P4_Planning']
MAX_WORKERS = 3  

CATEGORIES_DEF_S1 = {'Course_Type': [1, 2, 3, 4, 5], 'Weather_Code': [1, 2, 3], 'Tide_Trend': [-1, 0, 1]}
CATEGORIES_DEF_S2 = {'PlaceID': list(range(1, 25)), 'Course_Type': [1, 2, 3, 4, 5], 'Weather_Code': [0, 1, 2, 3, 4, 5, 6], 'Tide_Trend': [-1, 0, 1], 'Boat_Number': [1, 2, 3, 4, 5, 6]}

JCD_MAP = {f"{i:02d}": name for i, name in enumerate(["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"], 1)}
PLACE_COORDS = {1: {"lat": 36.39, "lon": 139.30}, 2: {"lat": 35.82, "lon": 139.66}, 3: {"lat": 35.69, "lon": 139.86}, 4: {"lat": 35.58, "lon": 139.73}, 5: {"lat": 35.65, "lon": 139.51}, 6: {"lat": 34.69, "lon": 137.56}, 7: {"lat": 34.82, "lon": 137.21}, 8: {"lat": 34.88, "lon": 136.82}, 9: {"lat": 34.68, "lon": 136.51}, 10: {"lat": 36.21, "lon": 136.16}, 11: {"lat": 35.01, "lon": 135.85}, 12: {"lat": 34.60, "lon": 135.47}, 13: {"lat": 34.71, "lon": 135.38}, 14: {"lat": 34.20, "lon": 134.60}, 15: {"lat": 34.30, "lon": 133.79}, 16: {"lat": 34.46, "lon": 133.81}, 17: {"lat": 34.29, "lon": 132.30}, 18: {"lat": 34.03, "lon": 131.81}, 19: {"lat": 33.99, "lon": 130.98}, 20: {"lat": 33.89, "lon": 130.75}, 21: {"lat": 33.88, "lon": 130.66}, 22: {"lat": 33.59, "lon": 130.39}, 23: {"lat": 33.43, "lon": 129.98}, 24: {"lat": 32.89, "lon": 129.96}}
TRACK_ANGLES = {1: 163.6, 2: 101.6, 3: 17.8, 4: 355.6, 5: 273.4, 6: 187.0, 7: 243.7, 8: 271.3, 9: 282.1, 10: 152.9, 11: 192.5, 12: 186.1, 13: 250.5, 14: 109.5, 15: 333.3, 16: 181.1, 17: 228.7, 18: 299.0, 19: 222.8, 20: 244.1, 21: 90.9, 22: 68.1, 23: 212.4, 24: 50.6}
COURSE_TYPE_MAP = {24: 1, 18: 1, 21: 1, 19: 1, 13: 1, 10: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 1: 2, 12: 3, 15: 3, 16: 3, 17: 3, 20: 3, 23: 3, 2: 4, 4: 4, 14: 4, 11: 4, 22: 4, 3: 5}

# =============================================================================
# 2. Google Drive & API連携 (V7継承)
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
    logger.info("☁️ V10専用 AIモデル・マスターデータ・ポートフォリオをダウンロードします...")
    os.makedirs("Models_Stage1_V9", exist_ok=True)
    os.makedirs("Models_Stage2_V10", exist_ok=True)
    
    download_latest_file_by_name(service, TIDE_CSV_NAME)
    logger.info("   - マスターデータ(500MB超)をダウンロード中...少々お待ちください")
    download_latest_file_by_name(service, PORTFOLIO_2TAN_CSV)
    download_latest_file_by_name(service, PORTFOLIO_2FUKU_CSV)
    
    for pid in PROJECT_IDS:
        download_latest_file_by_name(service, f"LGBM_Stage1_V9_{pid}.pkl", "Models_Stage1_V9")
        download_latest_file_by_name(service, f"LGBM_V10_Model1_1st_{pid}.pkl", "Models_Stage2_V10")
        download_latest_file_by_name(service, f"LGBM_V10_Model2_2nd_{pid}.pkl", "Models_Stage2_V10")

def send_line_broadcast(msg):
    if not LINE_CHANNEL_ACCESS_TOKEN: return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    try:
        resp = requests.post(url, headers=headers, json={"messages": [{"type": "text", "text": msg}]})
        if resp.status_code != 200: logger.error(f"❌ LINE API Error: {resp.text}")
    except Exception as e:
        logger.error(f"❌ LINEリクエスト送信エラー: {e}")

WEATHER_CACHE = {}
def fetch_weather(place_id, target_time_str):
    if not OPENWEATHER_API_KEY: return 0.0, 0.0, 1 # フェイルセーフ
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
# 3. V9 最新ハードウェア辞書の動的生成
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

def get_rank_point(rank_val):
    r = safe_float(rank_val, 99)
    return {1:10, 2:8, 3:6, 4:4, 5:2, 6:1}.get(r, 0.0)

def build_latest_hardware_dict():
    logger.info("⚙️ ダウンロードしたマスターCSVからV9用『最新ハードウェア辞書』を構築中...")
    if not os.path.exists(MASTER_CSV_NAME):
        logger.warning(f"⚠️ マスターデータが見つかりません。モーター予測値はデフォルト(0)で進行します。")
        return {}, {}
        
    cols = ['Date', 'PlaceID'] + [f'R{b}_Motor_No' for b in range(1,7)] + [f'R{b}_Boat_No' for b in range(1,7)] + \
           [f'R{b}_WinRate_National' for b in range(1,7)] + [f'Result_Boat{b}_Rank' for b in range(1,7)]
    
    try:
        df = pd.read_csv(MASTER_CSV_NAME, usecols=lambda c: c in cols, dtype=str)
    except Exception as e:
        logger.error(f"マスターデータ読み込みエラー: {e}")
        return {}, {}
        
    df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')
    df['PlaceID'] = pd.to_numeric(df['PlaceID'], errors='coerce').fillna(0).astype(int)
    
    records = []
    for b in range(1, 7):
        tmp = df[['Date_Parsed', 'PlaceID']].copy()
        tmp['Motor_No'] = df[f'R{b}_Motor_No'].apply(safe_float)
        tmp['Boat_No'] = df.get(f'R{b}_Boat_No', pd.Series([0]*len(df))).apply(safe_float) 
        tmp['Driver_WinRate'] = df[f'R{b}_WinRate_National'].apply(safe_float)
        tmp['Rank_Point'] = df[f'Result_Boat{b}_Rank'].apply(get_rank_point)
        records.append(tmp)

    df_hw = pd.concat(records, ignore_index=True).dropna(subset=['Date_Parsed'])
    
    df_hw = df_hw.sort_values(by=['PlaceID', 'Motor_No', 'Date_Parsed'])
    df_hw['M_Time_Gap'] = df_hw.groupby(['PlaceID', 'Motor_No'])['Date_Parsed'].diff().dt.days
    # 💡 apply()を使わず、boolean系列のcumsumで直接計算し、インデックスエラーを回避
    df_hw['is_M_reset'] = df_hw['M_Time_Gap'] > 60
    df_hw['Motor_Generation_ID'] = df_hw.groupby(['PlaceID', 'Motor_No'])['is_M_reset'].cumsum()
    
    df_hw = df_hw.sort_values(by=['PlaceID', 'Boat_No', 'Date_Parsed'])
    df_hw['B_Time_Gap'] = df_hw.groupby(['PlaceID', 'Boat_No'])['Date_Parsed'].diff().dt.days
    # 💡 ボートも同様に修正
    df_hw['is_B_reset'] = df_hw['B_Time_Gap'] > 60
    df_hw['Boat_Generation_ID'] = df_hw.groupby(['PlaceID', 'Boat_No'])['is_B_reset'].cumsum()

    df_hw = df_hw.sort_values(by='Date_Parsed').reset_index(drop=True)
    grp_m = df_hw.groupby(['PlaceID', 'Motor_No', 'Motor_Generation_ID'])
    grp_b = df_hw.groupby(['PlaceID', 'Boat_No', 'Boat_Generation_ID'])

    df_hw['Motor_Runs'] = grp_m.cumcount() + 1
    df_hw['Boat_Deterioration_Idx'] = grp_b.cumcount() + 1
    df_hw['True_Motor_Score'] = (grp_m['Rank_Point'].cumsum() - grp_m['Driver_WinRate'].cumsum()) / df_hw['Motor_Runs']
    df_hw['True_Boat_Score'] = (grp_b['Rank_Point'].cumsum() - grp_b['Driver_WinRate'].cumsum()) / df_hw['Boat_Deterioration_Idx']

    latest_m = df_hw.groupby(['PlaceID', 'Motor_No']).last().reset_index()
    latest_b = df_hw.groupby(['PlaceID', 'Boat_No']).last().reset_index()
    
    dict_motor = {(int(r.PlaceID), int(r.Motor_No)): r.True_Motor_Score for r in latest_m.itertuples()}
    dict_boat = {(int(r.PlaceID), int(r.Boat_No)): {'score': r.True_Boat_Score, 'idx': r.Boat_Deterioration_Idx} for r in latest_b.itertuples()}
    
    del df, df_hw, latest_m, latest_b
    gc.collect()
    logger.info("✅ ハードウェア辞書の生成完了")
    return dict_motor, dict_boat

# =============================================================================
# 4. スクレイピング処理
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
    if not soup or "データがありません" in soup.text or "中止" in soup.text: return None

    sched = "12:00"
    try:
        time_td = soup.find('td', string=re.compile('締切予定時刻'))
        if time_td: sched = time_td.find_parent('tr').find_all('td')[r].text.strip()
    except: pass

    title_class = " ".join(soup.find('div', class_='heading2_title').get('class')) if soup.find('div', class_='heading2_title') else ""
    r_name = clean_str(soup.find('h2').text) if soup.find('h2') else ""
    
    flags = {'Is_SG': 1 if 'is-SG' in title_class else 0, 'Is_G1': 1 if 'is-G1' in title_class else 0, 'Is_Rookie': 1 if any(w in r_name for w in ['ルーキー', 'ヤング', '若手']) else 0}
    flags['Is_General'] = 1 if sum([flags['Is_SG'], flags['Is_G1'], 1 if 'is-G2' in title_class else 0, 1 if 'is-G3' in title_class else 0]) == 0 else 0
    
    pid = "P3_General_Std"
    if flags['Is_SG']: pid = "P0_SG"
    elif flags['Is_G1'] and not flags['Is_Rookie']: pid = "P1_G1_Elite"
    elif flags['Is_General'] and (r == 1 or "進入固定" in r_name or "シード" in r_name): pid = "P4_Planning"
    if 'ヴィーナス' in r_name or 'オールレディース' in r_name or '女子' in r_name or 'レディース' in r_name: pid = "P2_Ladies"

    r_data = {}
    for i, tbody in enumerate(soup.find_all('tbody', class_=lambda x: x and 'is-fs12' in x)[:6], 1):
        try:
            tds = tbody.find_all('td')
            r_data[f"R{i}_Weight"] = re.split(r'\s+', tds[2].find_all('div')[2].text.strip())[1].split('/')[1].replace('kg', '')
            r_data[f"R{i}_F_Count"] = list(tds[3].stripped_strings)[0].replace('F', '')
            r_data[f"R{i}_Avg_ST"] = list(tds[3].stripped_strings)[2]
            w_nat, w_loc = list(tds[4].stripped_strings)[0], list(tds[5].stripped_strings)[0]
            r_data[f"R{i}_WinRate_National"], r_data[f"R{i}_WinRate_Local"] = w_nat, w_loc if w_loc != "0.00" else w_nat
            
            m_strings = list(tds[6].stripped_strings)
            b_strings = list(tds[7].stripped_strings)
            r_data[f"R{i}_Motor_No"] = m_strings[0] if len(m_strings) > 0 else 0
            r_data[f"R{i}_Motor_2Ren"] = m_strings[1] if len(m_strings) > 1 else 0
            r_data[f"R{i}_Boat_No"] = b_strings[0] if len(b_strings) > 0 else 0
        except: pass

    row = {'Date': date_str, 'PlaceID': jcd, 'RaceNum': r, 'Scheduled_Time': sched, 'Project_ID': pid}
    row.update(r_data)
    row.update(extract_additional_data(soup))
    return row

def scrape_today(today_obj):
    logger.info("🚤 本日の全国レース情報を取得中...")
    tasks = [(today_obj.strftime('%Y%m%d'), f"{j:02d}", r) for j in range(1, 25) for r in range(1, 13)]
    res = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as e:
        for f in concurrent.futures.as_completed({e.submit(parse_today_race, t): t for t in tasks}):
            try:
                if f.result(): res.append(f.result())
            except: pass
    return pd.DataFrame(res)

# =============================================================================
# 5. V9 特徴量パイプライン
# =============================================================================
def get_rank_point_s1(rank_val):
    if pd.isna(rank_val) or rank_val == "" or rank_val is None: return -5.0
    r = safe_float(rank_val, 99)
    return {1:10, 2:8, 3:6, 4:4, 5:2, 6:1}.get(r, -5.0)

def transform_for_v9_inference(df_raw, df_tide, dict_motor, dict_boat):
    fs1, fs2 = [], []
    error_count = 0 
    
    for _, row in df_raw.iterrows():
        pid = int(safe_float(row.get('PlaceID')))
        rnum = int(safe_float(row.get('RaceNum')))
        dt = int(row.get('Date', 0))
        proj_id = row.get('Project_ID')
        
        # 🛡️ V7/V8継承: バリデーションゲート
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
        
        bdata = {}
        for b in range(1, 7):
            wn = safe_float(row.get(f"R{b}_WinRate_National"))
            wl = safe_float(row.get(f"R{b}_WinRate_Local"), wn)
            m2 = safe_float(row.get(f"R{b}_Motor_2Ren"))
            ast = safe_float(row.get(f"R{b}_Avg_ST"), 0.17)
            wt = safe_float(row.get(f"R{b}_Weight"), 51.0)
            fc = safe_float(row.get(f"R{b}_F_Count"))
            
            ss, vs, mm = 0, 0, 0
            cr, c_rank = 0, 0
            mm_s1 = 0
            for i in range(1, 15):
                pst, prk, pc = safe_float(row.get(f"Boat{b}_Past_{i}_ST")), safe_float(row.get(f"Boat{b}_Past_{i}_Rank")), safe_float(row.get(f"Boat{b}_Past_{i}_Course"))
                w = max(0.2, (15 - i) * 0.1)
                mm_s1 += get_rank_point_s1(prk) * w
                if prk > 0: mm += get_rank_point(prk) * w
                if pst > 0: ss += pst; vs += 1
                if pc == b and prk > 0: cr += 1; c_rank += prk
            
            rst = (ss / vs) if vs > 0 else ast
            avg_mm_s1 = mm_s1 / 14.0
            rpwr = wn + (wl * 0.5) + (m2 / 10.0) + (avg_mm_s1 * 0.3) - (rst * 10)
            
            m_no = int(safe_float(row.get(f"R{b}_Motor_No")))
            b_no = int(safe_float(row.get(f"R{b}_Boat_No")))
            
            bdata[b] = {
                'pwr': rpwr, 'st': rst, 'wn': wn, 'wl': wl, 'm2': m2, 'fc': fc, 'wt': wt, 
                'mm': mm, 'cr': cr, 'crk': (c_rank / cr) if cr > 0 else 3.5,
                't_motor': dict_motor.get((pid, m_no), 0.0),
                't_boat': dict_boat.get((pid, b_no), {}).get('score', 0.0),
                'b_idx': dict_boat.get((pid, b_no), {}).get('idx', 0)
            }
            
        fs1.append({
            'Race_ID': f"{dt}_{pid}_{rnum}", 'Project_ID': proj_id, 'PlaceID': pid, 'Scheduled_Time': sched,
            'Course_Type': COURSE_TYPE_MAP.get(pid, 3), 'Month_Cos': math.cos(2 * math.pi * int(str(dt)[4:6]) / 12.0),
            'Weather_Code': wc, 'Tailwind_Comp': tw, 'Crosswind_Comp': cw, 'Tide_Level_cm': tl, 'Tide_Trend': tt,
            'B1_Power': bdata[1]['pwr'], 'B1_Local_WinRate': bdata[1]['wl'], 'B1_ST': bdata[1]['st'],
            'B1_Advantage': bdata[1]['pwr'] - max([bdata[x]['pwr'] for x in range(2, 7)]),
            'Wall_ST': (bdata[2]['st'] + bdata[3]['st']) / 2.0, 'Dash_Threat': max([bdata[x]['pwr'] for x in range(4, 7)])
        })
        
        for b in range(1, 7):
            ib, ob = max(1, b - 1), min(6, b + 1)
            fs2.append({
                'Race_ID': f"{dt}_{pid}_{rnum}", 'Project_ID': proj_id, 'Boat_Number': b, 'PlaceID': pid,
                'Scheduled_Time': sched, 'Course_Type': COURSE_TYPE_MAP.get(pid, 3), 'Weather_Code': wc,
                'Tailwind_Comp': tw, 'Crosswind_Comp': cw, 'Tide_Level_cm': tl, 'Tide_Trend': tt,
                'Month_Cos': math.cos(2 * math.pi * int(str(dt)[4:6]) / 12.0), 'Tournament_Day': safe_float(row.get('Tournament_Day'), 1.0),
                'Boat_WinRate': bdata[b]['wn'], 'Boat_WinRate_Local': bdata[b]['wl'], 'Boat_Motor': bdata[b]['m2'],
                'Boat_Weight': bdata[b]['wt'], 'F_Count': bdata[b]['fc'], 'Recent_Avg_ST': bdata[b]['st'],
                'Momentum_Score': bdata[b]['mm'], 'Target_Course_Runs': bdata[b]['cr'], 'Target_Course_AvgRank': bdata[b]['crk'],
                'Inside_F_Count': bdata[ib]['fc'], 'ST_Advantage_Inside': bdata[ib]['st'] - bdata[b]['st'],
                'ST_Advantage_Outside': bdata[ob]['st'] - bdata[b]['st'], 'WinRate_Diff_Inside': bdata[b]['wn'] - bdata[ib]['wn'],
                'Motor_Diff_Inside': bdata[b]['m2'] - bdata[ib]['m2'],
                'True_Motor_Score': bdata[b]['t_motor'], 'True_Boat_Score': bdata[b]['t_boat'], 'Boat_Deterioration_Idx': bdata[b]['b_idx']
            })

    if error_count > 0:
        send_line_broadcast(f"⚠️【警告】スクレイピングデータ異常（{error_count}レース）。誤推論を防ぐためスキップしました。")

    return pd.DataFrame(fs1), pd.DataFrame(fs2)

# =============================================================================
# 6. V10 ハイブリッド推論 ＆ LINE通知
# =============================================================================
def get_rough_cat(p): return "超堅め(0-20%)" if p < 0.2 else "やや堅め(20-40%)" if p < 0.4 else "普通(40-60%)" if p < 0.6 else "やや荒れ(60-80%)" if p < 0.8 else "大荒れ(80-100%)"

def run_v10_inference_and_notify(df_s1, df_s2):
    current_month = TODAY_OBJ.month
    
    if not os.path.exists(PORTFOLIO_2TAN_CSV) or not os.path.exists(PORTFOLIO_2FUKU_CSV):
        send_line_broadcast("❌ V10ポートフォリオファイルが読み込めませんでした。稼働を停止します。")
        return
        
    df_port_2t = pd.read_csv(PORTFOLIO_2TAN_CSV)
    df_port_2f = pd.read_csv(PORTFOLIO_2FUKU_CSV)
    
    # --- 💡 動的フィルタリング（黄金条件の自動抽出） ---
    def parse_plus_yrs(val):
        try:
            p = str(val).split('/')
            return int(p[0]), int(p[1]) if len(p)==2 else 0
        except:
            return 0, 0
            
    # 2連単の黄金条件抽出
    df_port_2t[['Plus', 'Active']] = pd.DataFrame(df_port_2t['OOS_プラス年数(24~26年)'].apply(parse_plus_yrs).tolist(), index=df_port_2t.index)
    df_port_2t = df_port_2t[(df_port_2t['OOS(未知)_統合レース数'] >= 15) & (df_port_2t['OOS(未知)_統合ROI'] >= 105) & (df_port_2t['Active'] >= 2) & (df_port_2t['Plus'] >= 2)]
    
    # 2連複の黄金条件抽出
    df_port_2f[['Plus', 'Active']] = pd.DataFrame(df_port_2f['OOS_2連複_プラス年数(24~26年)'].apply(parse_plus_yrs).tolist(), index=df_port_2f.index)
    df_port_2f = df_port_2f[(df_port_2f['OOS(未知)_統合レース数'] >= 15) & (df_port_2f['OOS(未知)_統合2連複_ROI'] >= 105) & (df_port_2f['Active'] >= 2) & (df_port_2f['Plus'] >= 2)]
    
    # 検索高速化のための辞書作成 (抽出後のデータで作成)
    valid_conditions_2t = set(zip(df_port_2t['Month'], df_port_2t['Project_ID'], df_port_2t['場名'], df_port_2t['Rough_Category']))
    valid_conditions_2f = set(zip(df_port_2f['Month'], df_port_2f['Project_ID'], df_port_2f['場名'], df_port_2f['Rough_Category']))
    
    buys_2t = []
    buys_2f = []
    debug_logs = {}

    for pid in PROJECT_IDS:
        ds1 = df_s1[df_s1['Project_ID'] == pid].copy()
        if ds1.empty: continue
            
        try:
            # --- 🛡️ Stage 1 (荒れ度予測) ---
            m1_path = f"Models_Stage1_V9/LGBM_Stage1_V9_{pid}.pkl"
            if not os.path.exists(m1_path): continue
            with open(m1_path, 'rb') as f: stage1_model = pickle.load(f)
            
            X1 = ds1[stage1_model.feature_name()].copy()
            for col in X1.columns: X1[col] = pd.to_numeric(X1[col], errors='coerce').fillna(0.0)
            
            for c, cats in CATEGORIES_DEF_S1.items():
                if c in X1.columns: X1[c] = pd.Categorical(X1[c].fillna(cats[0]).astype(int), categories=cats, ordered=False).codes
            ds1['Stage1_Rough_Prob'] = stage1_model.predict(X1.astype(float).values)
            
            # --- ⚡ Stage 2 準備 (V10) ---
            ds2 = df_s2[df_s2['Project_ID'] == pid].merge(ds1[['Race_ID', 'Stage1_Rough_Prob']], on='Race_ID', how='inner')
            
            for c, cats in CATEGORIES_DEF_S2.items():
                if c in ds2.columns:
                    ds2[c] = pd.Categorical(ds2[c].fillna(cats[0]).astype(int), categories=cats, ordered=False)

            m1_1st_path = f"Models_Stage2_V10/LGBM_V10_Model1_1st_{pid}.pkl"
            m2_2nd_path = f"Models_Stage2_V10/LGBM_V10_Model2_2nd_{pid}.pkl"
            if not (os.path.exists(m1_1st_path) and os.path.exists(m2_2nd_path)): continue
            
            with open(m1_1st_path, 'rb') as f: model1_1st = pickle.load(f)
            with open(m2_2nd_path, 'rb') as f: model2_2nd = pickle.load(f)
            
            # --- ⚡ Model 1 (1着予測) ---
            features_m1 = model1_1st.feature_name()
            X2_m1 = ds2[features_m1].copy()
            for col in X2_m1.columns:
                if col not in CATEGORIES_DEF_S2:
                    X2_m1[col] = pd.to_numeric(X2_m1[col], errors='coerce').fillna(0.0)
            
            ds2['P1_Raw'] = model1_1st.predict_proba(X2_m1)[:, 1]
            ds2['P1_Sum'] = ds2.groupby('Race_ID')['P1_Raw'].transform('sum')
            ds2['P1_Norm'] = np.where(ds2['P1_Sum'] > 0, ds2['P1_Raw'] / ds2['P1_Sum'], 1.0 / 6.0)
            
            # --- ⚡ Model 2 (2着予測: 仮想データ生成) ---
            df_m2_list = []
            for win_b in range(1, 7):
                temp_df = ds2[ds2['Boat_Number'] != win_b].copy()
                temp_df['Win_Boat'] = win_b
                df_m2_list.append(temp_df)
                
            df_m2_all = pd.concat(df_m2_list, ignore_index=True)
            df_m2_all['Win_Boat_Cat'] = pd.Categorical(df_m2_all['Win_Boat'], categories=[1,2,3,4,5,6], ordered=False)
            df_m2_all['Win_Boat'] = df_m2_all['Win_Boat_Cat'] 
            
            features_m2 = model2_2nd.feature_name()
            X2_m2 = df_m2_all[features_m2].copy()
            for col in X2_m2.columns:
                if col not in CATEGORIES_DEF_S2 and col != 'Win_Boat':
                    X2_m2[col] = pd.to_numeric(X2_m2[col], errors='coerce').fillna(0.0)
            
            df_m2_all['P2_Raw'] = model2_2nd.predict_proba(X2_m2)[:, 1]
            df_m2_all['Win_Boat'] = df_m2_all['Win_Boat'].astype(int) 
            df_m2_all['Boat_Number'] = df_m2_all['Boat_Number'].astype(int)
            
            df_m2_all['P2_Sum'] = df_m2_all.groupby(['Race_ID', 'Win_Boat'])['P2_Raw'].transform('sum')
            df_m2_all['P2_Norm'] = np.where(df_m2_all['P2_Sum'] > 0, df_m2_all['P2_Raw'] / df_m2_all['P2_Sum'], 1.0 / 5.0)

            # --- ⚡ 2連単・2連複の確率合成 ---
            p1_subset = ds2[['Race_ID', 'Boat_Number', 'P1_Norm']].rename(columns={'Boat_Number': 'Win_Boat', 'P1_Norm': 'P1_Win_Norm'})
            p1_subset['Win_Boat'] = p1_subset['Win_Boat'].astype(int)
            
            df_prob = df_m2_all.merge(p1_subset, on=['Race_ID', 'Win_Boat'], how='left')
            df_prob['P_2tan'] = df_prob['P1_Win_Norm'] * df_prob['P2_Norm']
            best_2tan_df = df_prob.sort_values(['Race_ID', 'P_2tan'], ascending=[True, False]).drop_duplicates('Race_ID')
            
            # 【復活】2連複の合成
            df_prob['B1_fuku'] = df_prob[['Win_Boat', 'Boat_Number']].min(axis=1)
            df_prob['B2_fuku'] = df_prob[['Win_Boat', 'Boat_Number']].max(axis=1)
            df_fuku = df_prob.groupby(['Race_ID', 'B1_fuku', 'B2_fuku'])['P_2tan'].sum().reset_index(name='P_2fuku')
            best_2fuku_df = df_fuku.sort_values(['Race_ID', 'P_2fuku'], ascending=[True, False]).drop_duplicates('Race_ID')
            
            # --- 🎯 ポートフォリオ照合と買い目抽出 ---
            race_info = ds2[['Race_ID', 'PlaceID', 'Scheduled_Time', 'Stage1_Rough_Prob']].drop_duplicates('Race_ID')
            
            for _, r_info in race_info.iterrows():
                rid = r_info['Race_ID']
                plid = int(r_info['PlaceID'])
                place_name = JCD_MAP.get(f"{plid:02d}", "不明")
                cat = get_rough_cat(r_info['Stage1_Rough_Prob'])
                rnum = int(rid.split('_')[2])
                sched_time = r_info['Scheduled_Time']
                
                # 2連単・2連複それぞれの合致確認
                is_hit_2t = (current_month, pid, place_name, cat) in valid_conditions_2t
                is_hit_2f = (current_month, pid, place_name, cat) in valid_conditions_2f
                is_hit = is_hit_2t or is_hit_2f
                
                reason = []
                if is_hit_2t: reason.append("2単")
                if is_hit_2f: reason.append("2複")
                reason_str = f"✅ 合致: {','.join(reason)}" if is_hit else "❌ 当場・当条件の勝負指定なし"
                
                if plid not in debug_logs: debug_logs[plid] = []
                debug_logs[plid].append({'rnum': rnum, 'pid': pid, 'cat': cat, 'is_hit': is_hit, 'reason': reason_str})
                
                if is_hit:
                    grade = "SG" if pid == "P0_SG" else "G1" if pid == "P1_G1_Elite" else "女子" if pid == "P2_Ladies" else "一般" if pid == "P3_General_Std" else "企画"
                    
                    if is_hit_2t:
                        b_2t = best_2tan_df[best_2tan_df['Race_ID'] == rid].iloc[0]
                        t_2t = f"{int(b_2t['Win_Boat'])}-{int(b_2t['Boat_Number'])}"
                        buys_2t.append({'time': sched_time, 'p': plid, 'place': place_name, 'r': rnum, 'grade': grade, 'cat': cat, 'ticket': t_2t, 'prob': b_2t['P_2tan']})
                    
                    if is_hit_2f:
                        b_2f = best_2fuku_df[best_2fuku_df['Race_ID'] == rid].iloc[0]
                        t_2f = f"{int(b_2f['B1_fuku'])}={int(b_2f['B2_fuku'])}"
                        buys_2f.append({'time': sched_time, 'p': plid, 'place': place_name, 'r': rnum, 'grade': grade, 'cat': cat, 'ticket': t_2f, 'prob': b_2f['P_2fuku']})
                    
        except Exception as e: 
            logger.error(f"AI Error ({pid}): {e}")

    # 📊 判定プロセスの詳細コンソール出力
    logger.info("📊 === V10 AI推論 1レースごとの判定レポート ===")
    for plid in sorted(debug_logs.keys()):
        place_name = JCD_MAP.get(f"{plid:02d}", "不明")
        races = sorted(debug_logs[plid], key=lambda x: x['rnum'])
        pid_groups = {}
        for r in races: pid_groups.setdefault(r['pid'], []).append(r)
            
        for p_id, p_races in pid_groups.items():
            logger.info(f"🚤 {place_name} ({p_id}) - {len(p_races)}レース分析")
            for r in p_races:
                match_mark = "✅ 買い" if r['is_hit'] else "❌ 見送"
                logger.info(f"   {r['rnum']:>2}R: [{r['cat']}] -> {match_mark} (理由: {r['reason']})")
    logger.info("======================================")

    # LINE通知の組み立て
    msg = f"🤖 V10 System (Conditional Prob)\n📅 {TODAY_OBJ.strftime('%Y年%m月%d日')}\n"
    
    if not buys_2t and not buys_2f:
        msg += "\n本日は勝負条件に合致するレースがありません。\n資金を温存します。"
        send_line_broadcast(msg)
        return

    buys_all = []
    for b in buys_2t:
        b['type'] = '🎯2連単'
        buys_all.append(b)
    for b in buys_2f:
        b['type'] = '🛡️2連複'
        buys_all.append(b)

    buys_all = sorted(buys_all, key=lambda x: (x['p'], x['r'], x['type']))

    msg += f"\n■ 本日の厳選勝負レース (計{len(buys_all)}件)\n"
    prev_race_key = ""
    for b in buys_all:
        current_race_key = f"{b['p']}_{b['r']}"
        if current_race_key != prev_race_key:
            msg += f"\n[{b['time']}] {b['place']} {b['r']}R\n"
            msg += f" ├ {b['grade']} / {b['cat']}\n"
            prev_race_key = current_race_key
        msg += f" └ {b['type']}: {b['ticket']} (推定勝率: {b['prob']*100:.1f}%)\n"

    send_line_broadcast(msg.strip())
    logger.info(f"V10買い目送信完了: 買い目計{len(buys_all)}件")

def main():
    logger.info("🚀 V10 System Start (Conditional Probability Edition)")
    
    # 💡 GitHub Actions環境の場合のみ、Driveから必須ファイルを一式ダウンロード
    if os.environ.get("GITHUB_ACTIONS") == "true":
        srv = get_drive_service()
        if srv: prepare_ai_models(srv)
        else: logger.error("⚠️ GCP認証に失敗したためダウンロードをスキップします")
    
    dict_motor, dict_boat = build_latest_hardware_dict()
    dtide = pd.read_csv(TIDE_CSV_NAME) if os.path.exists(TIDE_CSV_NAME) else pd.DataFrame(columns=['DateInt', 'PlaceID', 'Hour', 'Tide_Level_cm', 'Tide_Trend'])
    
    df = scrape_today(TODAY_OBJ)
    if df.empty:
        logger.info("データ取得不可（開催なし、またはメンテ）")
        return
        
    s1, s2 = transform_for_v9_inference(df, dtide, dict_motor, dict_boat)
    if not s1.empty and not s2.empty:
        run_v10_inference_and_notify(s1, s2) # V10用に変更
        
    logger.info("Daily Job Completed.")

if __name__ == "__main__": 
    main()
