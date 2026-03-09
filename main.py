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

# =============================================================================
# 1. 環境設定・定数
# =============================================================================
logger = logging.getLogger("V3_DailyRun")
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
PROJECT_IDS = ['P0_SG', 'P1_G1_Elite', 'P3_General_Std', 'P4_Planning']

# ★罠解決：IPバン（DDoS判定）を防ぐための安全なスレッド数
MAX_WORKERS = 3  

# 🚀 【最終決定版】全システム共通・絶対不変のカテゴリ定義
CATEGORIES_DEF = {
    'PlaceID': list(range(1, 25)),
    'Course_Type': [1, 2, 3, 4, 5],
    'Weather_Code': [0, 1, 2, 3, 4, 5, 6],
    'Tide_Trend': [-1, 0, 1], # 修正済：下げ、止まり、上げ
    'Boat_Number': [1, 2, 3, 4, 5, 6]
}

# 🏆 新・聖杯カレンダー（V4ハイブリッド・最終決定版）
PORTFOLIO = {
    1: [("P4_Planning", 3, "普通(40-60%)"), ("P4_Planning", 15, "普通(40-60%)"), ("P4_Planning", 8, "やや堅め(20-40%)"), ("P3_General_Std", 23, "やや荒れ(60-80%)"), ("P3_General_Std", 13, "やや堅め(20-40%)")],
    2: [("P3_General_Std", 15, "普通(40-60%)"), ("P3_General_Std", 22, "やや堅め(20-40%)"), ("P4_Planning", 1, "普通(40-60%)"), ("P4_Planning", 5, "普通(40-60%)")],
    3: [("P4_Planning", 3, "普通(40-60%)"), ("P4_Planning", 24, "普通(40-60%)"), ("P4_Planning", 16, "やや堅め(20-40%)"), ("P3_General_Std", 16, "普通(40-60%)"), ("P4_Planning", 4, "やや荒れ(60-80%)"), ("P3_General_Std", 12, "超堅め(0-20%)"), ("P3_General_Std", 1, "普通(40-60%)"), ("P3_General_Std", 6, "超堅め(0-20%)")],
    4: [], # 魔の月は全休（資金防衛）
    5: [("P3_General_Std", 17, "やや荒れ(60-80%)")],
    6: [("P3_General_Std", 8, "超堅め(0-20%)"), ("P3_General_Std", 7, "普通(40-60%)"), ("P3_General_Std", 23, "やや荒れ(60-80%)")],
    7: [("P4_Planning", 19, "普通(40-60%)"), ("P4_Planning", 8, "やや堅め(20-40%)"), ("P4_Planning", 18, "やや堅め(20-40%)")],
    8: [("P4_Planning", 5, "やや堅め(20-40%)"), ("P3_General_Std", 17, "超堅め(0-20%)"), ("P4_Planning", 12, "普通(40-60%)"), ("P3_General_Std", 24, "やや荒れ(60-80%)"), ("P3_General_Std", 12, "超堅め(0-20%)")],
    9: [("P4_Planning", 23, "やや堅め(20-40%)"), ("P3_General_Std", 23, "普通(40-60%)")],
    10: [("P3_General_Std", 4, "やや荒れ(60-80%)"), ("P3_General_Std", 17, "やや荒れ(60-80%)")],
    11: [("P4_Planning", 7, "普通(40-60%)"), ("P3_General_Std", 12, "普通(40-60%)")],
    12: [("P3_General_Std", 21, "やや荒れ(60-80%)"), ("P4_Planning", 13, "やや堅め(20-40%)")]
}

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
    # ★罠解決：同名ファイル複数存在時の旧モデル取得バグを排除 (orderBy追加)
    query = f"name='{file_name}' and trashed=false"
    res = service.files().list(q=query, orderBy="createdTime desc", fields="files(id, name)").execute()
    if not res.get('files'): return False
    
    file_id = res['files'][0]['id']
    req = service.files().get_media(fileId=file_id)
    
    # ★罠解決：FileIOのメモリリーク（リソース枯渇）を自動解放で防ぐ
    with io.FileIO(os.path.join(save_dir, file_name), 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done: _, done = downloader.next_chunk()
    return True

def prepare_ai_models(service):
    logger.info("🤖 AIモデル（.pkl）の最新版をダウンロードします...")
    os.makedirs("Models_Stage1_V2", exist_ok=True)
    os.makedirs("Models_Stage2_V3", exist_ok=True)
    
    download_latest_file_by_name(service, TIDE_CSV_NAME)
    
    for pid in PROJECT_IDS:
        download_latest_file_by_name(service, f"LGBM_Stage1_V2_{pid}.pkl", "Models_Stage1_V2")
        download_latest_file_by_name(service, f"LGBM_Stage2_1st_V3_{pid}.pkl", "Models_Stage2_V3")
        download_latest_file_by_name(service, f"LGBM_Stage2_2nd_V3_{pid}.pkl", "Models_Stage2_V3")
        download_latest_file_by_name(service, f"LGBM_Stage2_3rd_V3_{pid}.pkl", "Models_Stage2_V3")

def send_line_broadcast(msg):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        logger.warning("⚠️ LINEトークンが環境変数に設定されていません。")
        return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    
    # ▼ ここから下を丸ごと置き換えます ▼
    try:
        # 古い1行をここで実行し、結果を「resp」という変数で受け取るようにしています
        resp = requests.post(url, headers=headers, json={"messages": [{"type": "text", "text": msg}]})
        
        # ステータスコードが200（成功）以外なら警告、200なら成功をログ出力
        if resp.status_code != 200:
            logger.error(f"❌ LINE API Error: Status {resp.status_code}, Response: {resp.text}")
        else:
            logger.info("✅ LINEへのメッセージ送信に成功しました。")
    except Exception as e:
        logger.error(f"❌ LINEリクエスト送信中に例外が発生しました: {e}")

# ★罠解決：OpenWeather APIのレートリミット（429エラー）超過を防ぐキャッシュ辞書
WEATHER_CACHE = {}

def fetch_weather(place_id, target_time_str):
    try:
        hour = target_time_str.split(':')[0]
        cache_key = f"{place_id}_{hour}" # 競艇場×時間単位でキャッシュ
        
        if cache_key in WEATHER_CACHE:
            return WEATHER_CACHE[cache_key]

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
            time.sleep(1.0) # IPバン回避用インターバル
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
    flags['Is_General'] = 1 if sum([flags['Is_SG'], flags['Is_G1'], 1 if 'is-G2' in title_class else 0, 1 if 'is-G3' in title_class else 0]) == 0 else 0
    
    pid = "P3_General_Std"
    if flags['Is_SG']: pid = "P0_SG"
    elif flags['Is_G1'] and not flags['Is_Rookie']: pid = "P1_G1_Elite"
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
# 4. 特徴量生成 ＆ バリデーションゲート
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

def get_rank_point_s1(rank_val):
    if pd.isna(rank_val) or rank_val == "" or rank_val is None: return -5.0
    r = safe_float(rank_val, 99)
    return {1:10, 2:8, 3:6, 4:4, 5:2, 6:1}.get(r, -5.0)

def get_rank_point_s2(v): 
    r = safe_float(v, 99)
    return {1:10, 2:8, 3:6, 4:4, 5:2, 6:1}.get(r, 0.0)

def transform_for_inference_v3(df_raw, df_tide):
    fs1, fs2 = [], []
    error_count = 0
    
    for _, row in df_raw.iterrows():
        pid, rnum, dt = int(safe_float(row.get('PlaceID'))), int(safe_float(row.get('RaceNum'))), int(row.get('Date', 0))
        
        # 🛡️ バリデーション
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
            
            ss, vs, mm_s1, mm_s2, cr, c_rank = 0, 0, 0, 0, 0, 0
            for i in range(1, 15):
                raw_pst = row.get(f"Boat{b}_Past_{i}_ST")
                raw_prk = row.get(f"Boat{b}_Past_{i}_Rank")
                raw_pc = row.get(f"Boat{b}_Past_{i}_Course")

                pst = safe_float(raw_pst)
                prk = safe_float(raw_prk)
                pc = safe_float(raw_pc)
                
                w = max(0.2, (15 - i) * 0.1)
                
                # ★修正：Stage1とStage2で異なるロジックを完全再現
                mm_s1 += get_rank_point_s1(raw_prk) * w
                if prk > 0: mm_s2 += get_rank_point_s2(raw_prk) * w

                if pst > 0: ss += pst; vs += 1
                if pc == b and prk > 0: cr += 1; c_rank += prk
            
            rst = (ss / vs) if vs > 0 else ast
            avg_mm_s1 = mm_s1 / 14.0
            rpwr = wn + (wl * 0.5) + (m2 / 10.0) + (avg_mm_s1 * 0.3) - (rst * 10)
            
            bdata[b] = {
                'pwr': rpwr, 'st': rst, 'wn': wn, 'wl': wl, 'm2': m2, 
                'fc': safe_float(row.get(f"R{b}_F_Count")), 
                'wt': safe_float(row.get(f"R{b}_Weight"), 51.0), 
                'mm_s2': mm_s2, 'cr': cr, 'crk': (c_rank / cr) if cr > 0 else 3.5
            }
            
        fs1.append({
            'Race_ID': f"{dt}_{pid}_{rnum}", 'Project_ID_Calc': row.get('Project_ID'), 'PlaceID': pid,
            'Course_Type': COURSE_TYPE_MAP.get(pid, 3), 'Month_Cos': math.cos(2 * math.pi * int(str(dt)[4:6]) / 12.0),
            'Weather_Code': wc, 'Tailwind_Comp': tw, 'Crosswind_Comp': cw, 'Tide_Level_cm': tl, 'Tide_Trend': tt,
            'B1_Power': bdata[1]['pwr'], 'B1_Local_WinRate': bdata[1]['wl'], 'B1_ST': bdata[1]['st'],
            'B1_Advantage': bdata[1]['pwr'] - max([bdata[x]['pwr'] for x in range(2, 7)]),
            'Wall_ST': (bdata[2]['st'] + bdata[3]['st']) / 2.0, 'Dash_Threat': max([bdata[x]['pwr'] for x in range(4, 7)])
        })
        
        for b in range(1, 7):
            ib, ob = max(1, b - 1), min(6, b + 1)
            fs2.append({
                'Race_ID': f"{dt}_{pid}_{rnum}", 'Project_ID_Calc': row.get('Project_ID'), 'Boat_Number': b,
                'PlaceID': pid, 'Course_Type': COURSE_TYPE_MAP.get(pid, 3), 'Weather_Code': wc,
                'Tailwind_Comp': tw, 'Crosswind_Comp': cw, 'Tide_Level_cm': tl, 'Tide_Trend': tt,
                'Month_Cos': math.cos(2 * math.pi * int(str(dt)[4:6]) / 12.0), 'Tournament_Day': safe_float(row.get('Tournament_Day'), 1.0),
                'Boat_WinRate': bdata[b]['wn'], 'Boat_WinRate_Local': bdata[b]['wl'], 'Boat_Motor': bdata[b]['m2'],
                'Boat_Weight': bdata[b]['wt'], 'F_Count': bdata[b]['fc'], 'Recent_Avg_ST': bdata[b]['st'],
                'Momentum_Score': bdata[b]['mm_s2'], 'Target_Course_Runs': bdata[b]['cr'], 'Target_Course_AvgRank': bdata[b]['crk'],
                'Inside_F_Count': bdata[ib]['fc'], 'ST_Advantage_Inside': bdata[ib]['st'] - bdata[b]['st'],
                'ST_Advantage_Outside': bdata[ob]['st'] - bdata[b]['st'], 'WinRate_Diff_Inside': bdata[b]['wn'] - bdata[ib]['wn'],
                'Motor_Diff_Inside': bdata[b]['m2'] - bdata[ib]['m2']
            })
            
    if error_count > 0:
        send_line_broadcast(f"⚠️【警告】スクレイピングデータ異常（{error_count}レース）。誤推論を防ぐためスキップしました。")
        
    return pd.DataFrame(fs1), pd.DataFrame(fs2)

# =============================================================================
# 5. AI推論 ＆ LINE通知
# =============================================================================
def get_rough_cat(p): return "超堅め(0-20%)" if p < 0.2 else "やや堅め(20-40%)" if p < 0.4 else "普通(40-60%)" if p < 0.6 else "やや荒れ(60-80%)" if p < 0.8 else "大荒れ(80-100%)"

def run_ai_and_notify_v3(df_s1, df_s2):
    t_cond = PORTFOLIO.get(TODAY_OBJ.month, [])
    if not t_cond:
        # 休みの月もLINEに通知する
        msg = f"🤖 【V3 真・聖杯AI】\n📅 {TODAY_OBJ.strftime('%Y年%m月%d日')}\n今月（{TODAY_OBJ.month}月）は魔の月のため稼働全休です🍵"
        logger.info("本日は稼働対象月ではありません。LINEに全休通知を送ります。")
        send_line_broadcast(msg)
        return

    CATEGORIES_DEF_S1 = {
        'Course_Type': [1, 2, 3, 4, 5],
        'Weather_Code': [1, 2, 3], 
        'Tide_Trend': [-1, 0, 1]
    }
    
    buys = []
    for pid in PROJECT_IDS:
        ds1 = df_s1[df_s1['Project_ID_Calc'] == pid].copy()
        if ds1.empty: continue
        try:
            with open(f"Models_Stage1_V2/LGBM_Stage1_V2_{pid}.pkl", 'rb') as f: m1 = pickle.load(f)
            
            # 🛡️ 究極対策1: カテゴリ型を「LightGBMが内部で使う数値（.codes）」に直接変換し、型チェックを無効化
            X1 = ds1.drop(columns=['Race_ID', 'Project_ID_Calc'])[m1.feature_name()].copy()
            for c, cats in CATEGORIES_DEF_S1.items():
                if c in X1.columns:
                    # pd.Categoricalの末尾に .codes を付けて純粋な整数配列にする
                    X1[c] = pd.Categorical(X1[c].fillna(cats[0]).astype(int), categories=cats, ordered=False).codes
            
            # データフレーム全体をただのfloat型として渡し、AIの型チェックを完全にスルーさせる
            X1 = X1.astype(float)
            ds1['Stage1_Rough_Prob'] = m1.predict(X1)
            
            ds2 = df_s2[df_s2['Project_ID_Calc'] == pid].merge(ds1[['Race_ID', 'Stage1_Rough_Prob']], on='Race_ID', how='inner')
            with open(f"Models_Stage2_V3/LGBM_Stage2_1st_V3_{pid}.pkl", 'rb') as f: m2_1 = pickle.load(f)
            with open(f"Models_Stage2_V3/LGBM_Stage2_2nd_V3_{pid}.pkl", 'rb') as f: m2_2 = pickle.load(f)
            with open(f"Models_Stage2_V3/LGBM_Stage2_3rd_V3_{pid}.pkl", 'rb') as f: m2_3 = pickle.load(f)
            
            # 🛡️ 究極対策2: Stage 2も同様に内部コード化
            X2 = ds2.drop(columns=['Race_ID', 'Project_ID_Calc'])[m2_1.feature_name()].copy()
            for c, cats in CATEGORIES_DEF.items():
                if c in X2.columns:
                    X2[c] = pd.Categorical(X2[c].fillna(cats[0]).astype(int), categories=cats, ordered=False).codes
                    
            X2 = X2.astype(float)
            ds2['P1'], ds2['P2'], ds2['P3'] = m2_1.predict(X2), m2_2.predict(X2), m2_3.predict(X2)
            
            for rid, grp in ds2.groupby('Race_ID', sort=False):
                if len(grp) != 6: continue
                plid = int(grp['PlaceID'].iloc[0])
                cat = get_rough_cat(grp['Stage1_Rough_Prob'].iloc[0])
                if not any(pid == tp and plid == tpl and cat == tc for tp, tpl, tc in t_cond): continue
                
                p1, p2, p3 = {r['Boat_Number']: r['P1'] for _, r in grp.iterrows()}, {r['Boat_Number']: r['P2'] for _, r in grp.iterrows()}, {r['Boat_Number']: r['P3'] for _, r in grp.iterrows()}
                sc = {f"{p[0]}-{p[1]}-{p[2]}": p1[p[0]] * p2[p[1]] * p3[p[2]] for p in itertools.permutations([1,2,3,4,5,6], 3)}
                buys.append({'p': plid, 'r': int(rid.split('_')[2]), 'c': cat, 'b': [x[0] for x in sorted(sc.items(), key=lambda x: x[1], reverse=True)[:4]]})
        except Exception as e: 
            logger.error(f"AI Error ({pid}): {e}")

    if not buys:
        # 合致レースがない場合もLINEに通知する
        msg = f"🤖 【V3 真・聖杯AI】\n📅 {TODAY_OBJ.strftime('%Y年%m月%d日')}\n本日は「新・聖杯カレンダー」の条件に合致する堅守レースがありませんでした🙅‍♂️"
        logger.info("本日は条件合致レースがありませんでした。LINEに通知します。")
        send_line_broadcast(msg)
    else:
        msg = f"🤖 【V3 真・聖杯AI】\n📅 {TODAY_OBJ.strftime('%Y年%m月%d日')}\n✅ 合致：{len(buys)}レース\n"
        for b in sorted(buys, key=lambda x: (x['p'], x['r'])):
            place_name = JCD_MAP.get(f"{b['p']:02d}", "不明")
            msg += f"\n🚤 {place_name} {b['r']}R\n【{b['c']}】\n◎ {b['b'][0]}\n○ {b['b'][1]}\n▲ {b['b'][2]}\n△ {b['b'][3]}\n"
        send_line_broadcast(msg)
        logger.info(f"買い目送信完了: {len(buys)}レース")
        
def main():
    logger.info("V3 System Start (Robust Edition)")
    srv = get_drive_service()
    if srv: prepare_ai_models(srv)
    
    dtide = pd.read_csv(TIDE_CSV_NAME) if os.path.exists(TIDE_CSV_NAME) else pd.DataFrame(columns=['DateInt', 'PlaceID', 'Hour', 'Tide_Level_cm', 'Tide_Trend'])
    df = scrape_today(TODAY_OBJ)
    if df.empty:
        logger.info("データ取得不可（開催なし、またはメンテ）")
        return
        
    s1, s2 = transform_for_inference_v3(df, dtide)
    if not s1.empty and not s2.empty:
        run_ai_and_notify_v3(s1, s2)
        
    logger.info("Daily Job Completed.")

if __name__ == "__main__": 
    main()
