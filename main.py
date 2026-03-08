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

GCP_SA_CREDENTIALS = os.environ.get("GCP_SA_CREDENTIALS")
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

TIDE_CSV_NAME = "Tide_Master_2020_2026.csv"

# 🎯 P2(女子戦)はノイズとなるため除外
PROJECT_IDS = ['P0_SG', 'P1_G1_Elite', 'P3_General_Std', 'P4_Planning']
MAX_WORKERS = 3

# 🏆 V3専用 究極の聖杯カレンダーポートフォリオ
PORTFOLIO = {
    1: [("P3_General_Std", 12, "やや堅め(20-40%)")],
    2: [("P4_Planning", 4, "普通(40-60%)"), ("P4_Planning", 13, "普通(40-60%)"), ("P3_General_Std", 22, "やや堅め(20-40%)"), ("P1_G1_Elite", 15, "普通(40-60%)")],
    3: [("P3_General_Std", 4, "やや荒れ(60-80%)")],
    4: [("P3_General_Std", 12, "やや堅め(20-40%)"), ("P3_General_Std", 13, "やや堅め(20-40%)")],
    5: [("P3_General_Std", 19, "普通(40-60%)"), ("P3_General_Std", 3, "やや荒れ(60-80%)")],
    6: [("P3_General_Std", 7, "やや荒れ(60-80%)")],
    7: [("P4_Planning", 18, "やや堅め(20-40%)")],
    8: [("P3_General_Std", 14, "普通(40-60%)")],
    9: [("P3_General_Std", 3, "普通(40-60%)"), ("P3_General_Std", 23, "普通(40-60%)")],
    10: [("P3_General_Std", 4, "やや荒れ(60-80%)")],
    11: [("P3_General_Std", 2, "普通(40-60%)")],
    12: [("P4_Planning", 19, "普通(40-60%)")]
}

JCD_MAP = {f"{i:02d}": name for i, name in enumerate([
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
    "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
    "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"
], 1)}

PLACE_COORDS = {
    1: {"lat": 36.39, "lon": 139.30}, 2: {"lat": 35.82, "lon": 139.66}, 3: {"lat": 35.69, "lon": 139.86}, 
    4: {"lat": 35.58, "lon": 139.73}, 5: {"lat": 35.65, "lon": 139.51}, 6: {"lat": 34.69, "lon": 137.56}, 
    7: {"lat": 34.82, "lon": 137.21}, 8: {"lat": 34.88, "lon": 136.82}, 9: {"lat": 34.68, "lon": 136.51}, 
    10: {"lat": 36.21, "lon": 136.16}, 11: {"lat": 35.01, "lon": 135.85}, 12: {"lat": 34.60, "lon": 135.47}, 
    13: {"lat": 34.71, "lon": 135.38}, 14: {"lat": 34.20, "lon": 134.60}, 15: {"lat": 34.30, "lon": 133.79}, 
    16: {"lat": 34.46, "lon": 133.81}, 17: {"lat": 34.29, "lon": 132.30}, 18: {"lat": 34.03, "lon": 131.81}, 
    19: {"lat": 33.99, "lon": 130.98}, 20: {"lat": 33.89, "lon": 130.75}, 21: {"lat": 33.88, "lon": 130.66}, 
    22: {"lat": 33.59, "lon": 130.39}, 23: {"lat": 33.43, "lon": 129.98}, 24: {"lat": 32.89, "lon": 129.96}
}
TRACK_ANGLES = {
    1: 163.6, 2: 101.6, 3: 17.8, 4: 355.6, 5: 273.4, 6: 187.0, 7: 243.7, 8: 271.3, 
    9: 282.1, 10: 152.9, 11: 192.5, 12: 186.1, 13: 250.5, 14: 109.5, 15: 333.3, 16: 181.1, 
    17: 228.7, 18: 299.0, 19: 222.8, 20: 244.1, 21: 90.9, 22: 68.1, 23: 212.4, 24: 50.6
}
COURSE_TYPE_MAP = {
    24: 1, 18: 1, 21: 1, 19: 1, 13: 1, 10: 1,
    5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 1: 2,
    12: 3, 15: 3, 16: 3, 17: 3, 20: 3, 23: 3,
    2: 4, 4: 4, 14: 4, 11: 4, 22: 4,
    3: 5
}

# =============================================================================
# 2. 通信API群
# =============================================================================
def get_drive_service():
    if not GCP_SA_CREDENTIALS: raise ValueError("GCP_SA_CREDENTIALS is not set")
    creds_dict = json.loads(GCP_SA_CREDENTIALS)
    creds = service_account.Credentials.from_service_account_info(creds_dict)
    return build('drive', 'v3', credentials=creds)

def download_file(service, file_name):
    query = f"name='{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id, name)").execute()
    files = res.get('files', [])
    if not files: return None
    file_id = files[0]['id']
    req = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done: status, done = downloader.next_chunk()
    return file_id

def send_line_broadcast(msg):
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    data = {"messages": [{"type": "text", "text": msg}]}
    requests.post(url, headers=headers, json=data)

def fetch_weather(place_id, target_time_str):
    try:
        hour, minute = map(int, target_time_str.split(':'))
        now = datetime.now(JST)
        target_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        target_unix = int(target_dt.timestamp())
        
        coords = PLACE_COORDS[place_id]
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={coords['lat']}&lon={coords['lon']}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        
        closest = min(res.get('list', []), key=lambda x: abs(x['dt'] - target_unix))
        w_speed = closest['wind'].get('speed', 0.0)
        w_deg = closest['wind'].get('deg', 0.0)
        main = closest['weather'][0].get('main', 'Clear')
        w_code = 1 if main == 'Clear' else 2 if main == 'Clouds' else 3
        return float(w_speed), float(w_deg), w_code
    except:
        return 0.0, 0.0, 1

# =============================================================================
# 3. 今日の出走表スクレイピング
# =============================================================================
def clean_str(s): return s.replace('\u3000', '').strip() if s else ""
def clean_rank_value(val):
    if not val: return None
    v = str(val).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    if v in ['1', '2', '3', '4', '5', '6']: return float(v)
    return None

def fetch_soup(url, retries=3):
    for _ in range(retries):
        try:
            time.sleep(0.5)
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
        except: time.sleep(1)
    return None

def extract_additional_data(soup_list):
    data = {}
    tbodies = soup_list.find_all('tbody', class_='is-fs12')
    unique_past_dates = set()
    for boat_idx, tbody in enumerate(tbodies[:6], 1):
        trs = tbody.find_all('tr')
        if len(trs) < 4: continue
        start_col_idx = 9
        tds_race_no = trs[0].find_all('td')
        tds_course = trs[1].find_all('td')
        tds_st = trs[2].find_all('td')
        tds_rank = trs[3].find_all('td')
        loop_range = min(len(tds_course), 14)
        for i in range(loop_range):
            if start_col_idx + i < len(tds_race_no):
                prefix = f"Boat{boat_idx}_Past_{i+1}"
                data[f"{prefix}_Course"] = clean_str(tds_course[i].get_text(strip=True))
                data[f"{prefix}_ST"] = clean_str(tds_st[i].get_text(strip=True))
                data[f"{prefix}_Rank"] = clean_rank_value(tds_rank[i].get_text(strip=True))
        for td in tds_rank:
            a_tag = td.find('a')
            if a_tag and 'href' in a_tag.attrs:
                match = re.search(r'hd=(\d{8})', a_tag['href'])
                if match: unique_past_dates.add(match.group(1))
    data["Tournament_Day"] = str(len(unique_past_dates) + 1)
    return data

def parse_today_race(task_tuple):
    date_str, jcd, r = task_tuple
    url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={r}&jcd={jcd}&hd={date_str}"
    soup_list = fetch_soup(url_list)
    if not soup_list or "データがありません" in soup_list.text: return None

    scheduled_time = "12:00"
    try:
        time_td = soup_list.find('td', string=re.compile('締切予定時刻'))
        if time_td:
            tds = time_td.find_parent('tr').find_all('td')
            if len(tds) > r: scheduled_time = tds[r].text.strip()
    except: pass

    heading = soup_list.find('div', class_='heading2_title')
    race_name = clean_str(heading.find('h2').text) if heading else ""
    title_class = " ".join(heading.get('class')) if heading else ""
    
    flags = {
        'Is_SG': 1 if 'is-SG' in title_class else 0,
        'Is_G1': 1 if 'is-G1' in title_class else 0,
        'Is_General': 0,
        'Is_Rookie': 1 if 'ルーキー' in race_name or 'ヤング' in race_name or '若手' in race_name else 0,
    }
    if sum([flags['Is_SG'], flags['Is_G1'], 1 if 'is-G2' in title_class else 0, 1 if 'is-G3' in title_class else 0]) == 0: 
        flags['Is_General'] = 1
    
    project_id = "P3_General_Std"
    if flags['Is_SG'] == 1: project_id = "P0_SG"
    elif flags['Is_G1'] == 1 and flags['Is_Rookie'] == 0: project_id = "P1_G1_Elite"
    elif flags['Is_General'] == 1 and (r == 1 or "進入固定" in race_name or "シード" in race_name): project_id = "P4_Planning"

    racer_data = {}
    tbodies = soup_list.find_all('tbody', class_=lambda x: x and 'is-fs12' in x)
    for i, tbody in enumerate(tbodies[:6], 1):
        pfx = f"R{i}_"
        try:
            tds = tbody.find_all('td')
            parts = re.split(r'\s+', tds[2].find_all('div')[2].text.strip())
            racer_data[pfx+'Weight'] = parts[1].split('/')[1].replace('kg', '')
            lines = list(tds[3].stripped_strings)
            racer_data[pfx+'F_Count'] = lines[0].replace('F', '')
            racer_data[pfx+'Avg_ST'] = lines[2]
            
            win_nat = list(tds[4].stripped_strings)[0]
            win_local = list(tds[5].stripped_strings)[0]
            racer_data[pfx+'WinRate_National'] = win_nat
            racer_data[pfx+'WinRate_Local'] = win_local if win_local != "0.00" else win_nat
            racer_data[pfx+'Motor_2Ren'] = list(tds[6].stripped_strings)[1]
        except: pass

    row = {
        'Date': date_str, 'PlaceID': jcd, 'RaceNum': r,
        'Scheduled_Time': scheduled_time, 'Project_ID': project_id
    }
    row.update(racer_data)
    row.update(extract_additional_data(soup_list))
    return row

def scrape_today(today_obj):
    tasks = []
    d_str = today_obj.strftime('%Y%m%d')
    for j in range(1, 25):
        for r in range(1, 13):
            tasks.append((d_str, f"{j:02d}", r))
            
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parse_today_race, t): t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
                if res: results.append(res)
            except: pass
    return pd.DataFrame(results)

# =============================================================================
# 4. 特徴量変換ロジック（V3専用 インメモリ生成）
# =============================================================================
def get_rank_point(rank_val):
    if pd.isna(rank_val) or rank_val == "" or rank_val is None: return 0.0
    scores = {1:10, 2:8, 3:6, 4:4, 5:2, 6:1}
    return scores.get(float(rank_val), 0.0)

def safe_float(val, default=0.0):
    if pd.isna(val) or val == "" or val is None: return default
    try: return float(val)
    except ValueError:
        clean = re.sub(r'[^\d.-]', '', str(val))
        return float(clean) if clean not in ('', '-', '.') else default

def transform_for_inference_v3(df_raw, df_tide):
    features_s1 = []
    features_s2_v3 = []
    
    for _, row in df_raw.iterrows():
        place_id = int(safe_float(row.get('PlaceID', 0)))
        race_num = int(safe_float(row.get('RaceNum', 0)))
        date_int = int(row.get('Date', 0))
        race_id = f"{date_int}_{place_id}_{race_num}"
        scheduled_time = str(row.get('Scheduled_Time', '12:00'))
        
        w_speed, w_deg, w_code = fetch_weather(place_id, scheduled_time)
        t_angle = TRACK_ANGLES.get(place_id, 0.0)
        angle_diff = math.radians((w_deg + 180.0) - t_angle)
        t_wind = round(w_speed * math.cos(angle_diff), 2)
        c_wind = round(w_speed * math.sin(angle_diff), 2)
        
        hour = int(scheduled_time.split(':')[0]) if ':' in scheduled_time else 12
        tide_row = df_tide[(df_tide['DateInt'] == date_int) & (df_tide['PlaceID'] == place_id) & (df_tide['Hour'] == hour)]
        tide_level = float(tide_row['Tide_Level_cm'].iloc[0]) if not tide_row.empty else 0.0
        tide_trend = int(tide_row['Tide_Trend'].iloc[0]) if not tide_row.empty else 0
        
        month_cos = math.cos(2 * math.pi * int(str(date_int)[4:6]) / 12.0)
        c_type = COURSE_TYPE_MAP.get(place_id, 3)
        tournament_day = safe_float(row.get('Tournament_Day'), 1.0)
        
        boats_data = {}
        for b in range(1, 7):
            win_nat = safe_float(row.get(f"R{b}_WinRate_National"))
            win_local = safe_float(row.get(f"R{b}_WinRate_Local"), win_nat)
            motor_2ren = safe_float(row.get(f"R{b}_Motor_2Ren"))
            avg_st = safe_float(row.get(f"R{b}_Avg_ST"), 0.17)
            f_count = safe_float(row.get(f"R{b}_F_Count"))
            weight = safe_float(row.get(f"R{b}_Weight"), 51.0)
            
            st_sum = 0; valid_st = 0; momentum = 0
            course_runs = 0; course_ranks = 0
            
            for i in range(1, 15):
                p_st = safe_float(row.get(f"Boat{b}_Past_{i}_ST"), 0)
                p_rank = safe_float(row.get(f"Boat{b}_Past_{i}_Rank"), 0)
                p_course = safe_float(row.get(f"Boat{b}_Past_{i}_Course"), 0)
                
                if p_st > 0: st_sum += p_st; valid_st += 1
                if p_rank > 0: momentum += get_rank_point(p_rank) * max(0.2, (15 - i) * 0.1)
                if p_course == b and p_rank > 0: course_runs += 1; course_ranks += p_rank
                    
            r_st = (st_sum / valid_st) if valid_st > 0 else avg_st
            r_c_rank = (course_ranks / course_runs) if course_runs > 0 else 3.5
            
            raw_pwr = win_nat + (win_local * 0.5) + (motor_2ren / 10.0) + ((momentum / 14.0) * 0.3) - (r_st * 10)
            
            boats_data[b] = {
                'power': raw_pwr, 'st': r_st, 'win_nat': win_nat, 'win_local': win_local, 
                'motor': motor_2ren, 'avg_st': avg_st, 'f_count': f_count, 'weight': weight,
                'recent_avg_st': r_st, 'momentum': momentum,
                'course_runs': course_runs, 'course_avg_rank': r_c_rank
            }
            
        rivals = [boats_data[b]['power'] for b in range(2, 7)]
        dash = [boats_data[b]['power'] for b in range(4, 7)]
        feat_s1 = {
            'Race_ID': race_id, 'Project_ID_Calc': row.get('Project_ID'), 'PlaceID': place_id,
            'Course_Type': c_type, 'Month_Cos': month_cos, 'Weather_Code': w_code,
            'Tailwind_Comp': t_wind, 'Crosswind_Comp': c_wind, 
            'Tide_Level_cm': tide_level, 'Tide_Trend': tide_trend,
            'B1_Power': boats_data[1]['power'], 'B1_Local_WinRate': boats_data[1]['win_local'],
            'B1_ST': boats_data[1]['st'], 'B1_Advantage': boats_data[1]['power'] - max(rivals),
            'Wall_ST': (boats_data[2]['st'] + boats_data[3]['st']) / 2.0,
            'Dash_Threat': max(dash)
        }
        features_s1.append(feat_s1)
        
        for b in range(1, 7):
            inside_b = b - 1 if b > 1 else 1
            outside_b = b + 1 if b < 6 else 6
            
            feat_s2 = {
                'Race_ID': race_id, 'Project_ID_Calc': row.get('Project_ID'), 'Boat_Number': b,
                'PlaceID': place_id, 'Course_Type': c_type, 'Weather_Code': w_code,
                'Tailwind_Comp': t_wind, 'Crosswind_Comp': c_wind, 
                'Tide_Level_cm': tide_level, 'Tide_Trend': tide_trend,
                'Month_Cos': month_cos, 'Tournament_Day': tournament_day,
                
                'Boat_WinRate': boats_data[b]['win_nat'], 'Boat_WinRate_Local': boats_data[b]['win_local'],
                'Boat_Motor': boats_data[b]['motor'], 'Boat_Weight': boats_data[b]['weight'], 'F_Count': boats_data[b]['f_count'],
                'Recent_Avg_ST': boats_data[b]['recent_avg_st'], 'Momentum_Score': boats_data[b]['momentum'],
                'Target_Course_Runs': boats_data[b]['course_runs'], 'Target_Course_AvgRank': boats_data[b]['course_avg_rank'],
                
                'Inside_F_Count': boats_data[inside_b]['f_count'],
                'ST_Advantage_Inside': boats_data[inside_b]['recent_avg_st'] - boats_data[b]['recent_avg_st'],
                'ST_Advantage_Outside': boats_data[outside_b]['recent_avg_st'] - boats_data[b]['recent_avg_st'],
                'WinRate_Diff_Inside': boats_data[b]['win_nat'] - boats_data[inside_b]['win_nat'],
                'Motor_Diff_Inside': boats_data[b]['motor'] - boats_data[inside_b]['motor']
            }
            features_s2_v3.append(feat_s2)

    return pd.DataFrame(features_s1), pd.DataFrame(features_s2_v3)

# =============================================================================
# 5. AI推論＆通知フロー（V3専用）
# =============================================================================
def get_rough_category(prob):
    if prob < 0.20: return "超堅め(0-20%)"
    elif 0.20 <= prob < 0.40: return "やや堅め(20-40%)"
    elif 0.40 <= prob < 0.60: return "普通(40-60%)"
    elif 0.60 <= prob < 0.80: return "やや荒れ(60-80%)"
    else: return "大荒れ(80-100%)"

def run_ai_and_notify_v3(df_s1, df_s2):
    today = datetime.now(JST)
    month = today.month
    date_str = today.strftime('%Y年%m月%d日')
    
    target_conditions = PORTFOLIO.get(month, [])
    if not target_conditions:
        send_line_broadcast(f"🤖【V3AI予測】 {date_str}\n本日はV3カレンダーの稼働対象月ではありません。")
        return

    buy_list = []
    
    # 🚨【修正箇所】学習時と完全に一致するカテゴリ定義を強制適用
    cat_definitions = {
        'PlaceID': list(range(1, 25)), 
        'Course_Type': list(range(1, 6)),
        'Weather_Code': list(range(1, 4)), # [1, 2, 3] 
        'Tide_Trend': [-1, 0, 1],          # [-1, 0, 1]
        'Boat_Number': list(range(1, 7))
    }
    
    for col, cats in cat_definitions.items():
        if col in df_s1.columns: df_s1[col] = pd.Categorical(df_s1[col].fillna(cats[0]).astype(int), categories=cats)
        if col in df_s2.columns: df_s2[col] = pd.Categorical(df_s2[col].fillna(cats[0]).astype(int), categories=cats)
    
    s1_drop = ['Race_ID', 'Project_ID_Calc']
    s2_drop = ['Race_ID', 'Project_ID_Calc']

    for pid in PROJECT_IDS:
        df_pid_s1 = df_s1[df_s1['Project_ID_Calc'] == pid].copy()
        if df_pid_s1.empty: continue
            
        try:
            m1_path = f"Models_Stage1_V2/LGBM_Stage1_V2_{pid}.pkl"
            with open(m1_path, 'rb') as f: model_s1 = pickle.load(f)
            df_pid_s1['Stage1_Rough_Prob'] = model_s1.predict(df_pid_s1.drop(columns=s1_drop))
            
            df_pid_s2 = df_s2[df_s2['Project_ID_Calc'] == pid].merge(df_pid_s1[['Race_ID', 'Stage1_Rough_Prob']], on='Race_ID', how='inner')
            
            with open(f"Models_Stage2_V3/LGBM_Stage2_1st_V3_{pid}.pkl", 'rb') as f: m2_1 = pickle.load(f)
            with open(f"Models_Stage2_V3/LGBM_Stage2_2nd_V3_{pid}.pkl", 'rb') as f: m2_2 = pickle.load(f)
            with open(f"Models_Stage2_V3/LGBM_Stage2_3rd_V3_{pid}.pkl", 'rb') as f: m2_3 = pickle.load(f)
            
            X_s2 = df_pid_s2.drop(columns=s2_drop)
            df_pid_s2['Prob_1st'] = m2_1.predict(X_s2)
            df_pid_s2['Prob_2nd'] = m2_2.predict(X_s2)
            df_pid_s2['Prob_3rd'] = m2_3.predict(X_s2)
            
            for race_id, group in df_pid_s2.groupby('Race_ID', sort=False):
                if len(group) != 6: continue
                place_id = int(group['PlaceID'].iloc[0])
                r_prob = group['Stage1_Rough_Prob'].iloc[0]
                cat = get_rough_category(r_prob)
                
                # 🛡️ 究極の聖杯ポートフォリオとの照合
                if not any(pid == tp and place_id == tpl and cat == tc for tp, tpl, tc in target_conditions):
                    continue
                    
                p1 = {int(r['Boat_Number']): float(r['Prob_1st']) for _, r in group.iterrows()}
                p2 = {int(r['Boat_Number']): float(r['Prob_2nd']) for _, r in group.iterrows()}
                p3 = {int(r['Boat_Number']): float(r['Prob_3rd']) for _, r in group.iterrows()}
                
                scores = {f"{p[0]}-{p[1]}-{p[2]}": p1[p[0]] * p2[p[1]] * p3[p[2]] for p in itertools.permutations([1,2,3,4,5,6], 3)}
                top4 = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]]
                race_num = int(race_id.split('_')[2])
                
                buy_list.append({'place_id': place_id, 'race_num': race_num, 'rough_cat': cat, 'buy_pattern': top4})
        except Exception as e:
            logger.warning(f"AI推論エラー({pid}): {e}")

    if not buy_list:
        send_line_broadcast(f"🤖【V3AI予測】 {date_str}\nV3条件に合致する激アツレースは本日はありませんでした。")
        return
        
    buy_list = sorted(buy_list, key=lambda x: (x['place_id'], x['race_num']))
    msg = f"🤖 【V3】本日の展開予測AI\n📅 {date_str}\n✅ 合致：{len(buy_list)}レース\n"
    
    for b in buy_list:
        p_name = JCD_MAP.get(f"{b['place_id']:02d}", "不明")
        msg += f"\n🚤 {p_name} {b['race_num']}R\n"
        msg += f"【{b['rough_cat']}】\n"
        msg += f"◎ {b['buy_pattern'][0]}\n○ {b['buy_pattern'][1]}\n▲ {b['buy_pattern'][2]}\n△ {b['buy_pattern'][3]}\n"
        
    send_line_broadcast(msg)

# =============================================================================
# 6. メインフロー（実行統括）
# =============================================================================
def main():
    logger.info("=== V3 日次自動運用システム 起動 ===")
    today = datetime.now(JST)
    
    drive_service = get_drive_service()
    download_file(drive_service, TIDE_CSV_NAME)
    
    if os.path.exists(TIDE_CSV_NAME):
        df_tide = pd.read_csv(TIDE_CSV_NAME)
    else:
        logger.warning("潮位データが取得できませんでした。デフォルト値で進行します。")
        df_tide = pd.DataFrame(columns=['DateInt', 'PlaceID', 'Hour', 'Tide_Level_cm', 'Tide_Trend'])

    logger.info(f"🔍 本日の出走表({today.strftime('%Y%m%d')})をスクレイピング中...")
    df_today = scrape_today(today)
    
    if df_today.empty:
        send_line_broadcast(f"🤖【V3AI予測】 {today.strftime('%m月%d日')}\n本日の出走データが取得できませんでした。")
        return
        
    logger.info("🧠 インメモリでのV3特徴量生成を開始...")
    df_s1, df_s2 = transform_for_inference_v3(df_today, df_tide)
    
    logger.info("🧠 V3 AI推論を実行中...")
    run_ai_and_notify_v3(df_s1, df_s2)
    
    logger.info("🎉 V3 全タスク正常完了")

if __name__ == "__main__":
    main()
