import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os
import io
import logging
import concurrent.futures
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

# Google Drive API関連
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# =============================================================================
# 設定
# =============================================================================
MAX_WORKERS = 3
TIMEOUT_SEC = 20
CHUNK_SIZE = 1000

# GitHub Actionsの環境変数から情報を取得
CREDENTIALS_FILE = 'credentials.json'
TARGET_FILE_ID = os.environ.get('DRIVE_FILE_ID')

# ロガー設定
logger = logging.getLogger("BoatRaceScraper")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(sh)

# =============================================================================
# ヘルパー関数群（変更なし）
# =============================================================================
JCD_MAP = {
    "01":"桐生", "02":"戸田", "03":"江戸川", "04":"平和島", "05":"多摩川",
    "06":"浜名湖", "07":"蒲郡", "08":"常滑", "09":"津", "10":"三国",
    "11":"びわこ", "12":"住之江", "13":"尼崎", "14":"鳴門", "15":"丸亀",
    "16":"児島", "17":"宮島", "18":"徳山", "19":"下関", "20":"若松",
    "21":"芦屋", "22":"福岡", "23":"唐津", "24":"大村"
}
zenkaku_map = str.maketrans('０１２３４５６７８９', '0123456789')

def clean_str(s): return s.replace('\u3000', '').strip() if s else ""
def get_float(s):
    if not s: return 0.0
    try:
        match = re.search(r'(\d+\.\d+|\d+)', s)
        return float(match.group(1)) if match else 0.0
    except: return 0.0

def clean_rank_value(val):
    if not val: return None
    val_str = str(val).strip().translate(zenkaku_map)
    if val_str in ['1', '2', '3', '4', '5', '6']: return float(val_str)
    return None

def fetch_soup(url, retries=3):
    for _ in range(retries):
        try:
            time.sleep(0.5)
            resp = requests.get(url, timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
        except: time.sleep(1)
    return None

def extract_additional_data(soup_list, target_date_str):
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
                data[f"{prefix}_RaceNo"] = clean_str(tds_race_no[start_col_idx + i].get_text(strip=True))
                data[f"{prefix}_Course"] = clean_str(tds_course[i].get_text(strip=True))
                data[f"{prefix}_ST"] = clean_str(tds_st[i].get_text(strip=True))
                data[f"{prefix}_Rank"] = clean_rank_value(tds_rank[i].get_text(strip=True))
        for td in tds_rank:
            a_tag = td.find('a')
            if a_tag and 'href' in a_tag.attrs:
                match = re.search(r'hd=(\d{8})', a_tag['href'])
                if match and match.group(1) < target_date_str: unique_past_dates.add(match.group(1))
    data["Tournament_Day"] = str(len(unique_past_dates) + 1)
    return data

def parse_single_race(task_tuple):
    date_str, jcd, r = task_tuple
    jcd_str = str(jcd).zfill(2)
    
    url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={r}&jcd={jcd_str}&hd={date_str}"
    soup_list = fetch_soup(url_list)
    if not soup_list or "データがありません" in soup_list.text: return None
    
    url_result = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={r}&jcd={jcd_str}&hd={date_str}"
    soup_result = fetch_soup(url_result)

    heading = soup_list.find('div', class_='heading2_title')
    race_name = clean_str(heading.find('h2').text) if heading else ""
    title_class = " ".join(heading.get('class')) if heading else ""
    flags = {
        'Is_SG': 1 if 'is-SG' in title_class else 0, 'Is_G1': 1 if 'is-G1' in title_class else 0,
        'Is_G2': 1 if 'is-G2' in title_class else 0, 'Is_G3': 1 if 'is-G3' in title_class else 0,
        'Is_General': 0,
        'Is_Lady': 1 if any(w in race_name for w in ['レディース', '女子', 'ヴィーナス', 'オール女子']) else 0,
        'Is_Venus': 1 if 'ヴィーナス' in race_name else 0,
        'Is_Rookie': 1 if 'ルーキー' in race_name or 'ヤング' in race_name or '若手' in race_name else 0,
        'Is_Master': 1 if 'マスターズ' in race_name or '名人' in race_name else 0
    }
    if sum([flags['Is_SG'], flags['Is_G1'], flags['Is_G2'], flags['Is_G3']]) == 0: flags['Is_General'] = 1
    
    race_type = "一般"
    if '準優勝戦' in race_name: race_type = "準優勝戦"
    elif '優勝戦' in race_name: race_type = "優勝戦"
    elif 'ドリーム' in race_name: race_type = "ドリーム"
    elif '特選' in race_name: race_type = "特選"
    
    project_id = "P3_General_Std"
    if flags['Is_Lady'] == 1: project_id = "P2_Ladies"
    elif flags['Is_SG'] == 1: project_id = "P0_SG"
    elif flags['Is_G1'] == 1 and flags['Is_Rookie'] == 0: project_id = "P1_G1_Elite"
    elif flags['Is_General'] == 1 and (r == 1 or "進入固定" in race_name or "シード" in race_name): project_id = "P4_Planning"

    racer_data = {}
    tbodies = soup_list.find_all('tbody', class_=lambda x: x and 'is-fs12' in x)
    for i, tbody in enumerate(tbodies[:6], 1):
        pfx = f"R{i}_"
        try:
            tds = tbody.find_all('td')
            divs = tds[2].find_all('div')
            racer_data[pfx+'Toban'] = divs[0].text.strip().split('/')[0].strip()
            racer_data[pfx+'Class'] = divs[0].text.strip().split('/')[1].strip()
            racer_data[pfx+'Name'] = clean_str(divs[1].text)
            parts = re.split(r'\s+', divs[2].text.strip())
            racer_data[pfx+'Branch'] = parts[0].split('/')[0]
            racer_data[pfx+'Age'] = parts[1].split('/')[0].replace('歳', '')
            racer_data[pfx+'Weight'] = parts[1].split('/')[1].replace('kg', '')
            racer_data[pfx+'Gender'] = 'F' if 'fa-venus' in str(tds[2]) or 'is-lady' in str(tds[2]) else 'M'
            lines = list(tds[3].stripped_strings)
            racer_data[pfx+'F_Count'] = lines[0].replace('F', '')
            racer_data[pfx+'L_Count'] = lines[1].replace('L', '')
            racer_data[pfx+'Avg_ST'] = lines[2]
            for idx, key in enumerate(['National', 'Local']):
                lines = list(tds[4+idx].stripped_strings)
                racer_data[f"{pfx}WinRate_{key}"] = lines[0]
                racer_data[f"{pfx}2Ren_{key}"] = lines[1]
                racer_data[f"{pfx}3Ren_{key}"] = lines[2]
            for idx, key in enumerate(['Motor', 'Boat']):
                lines = list(tds[6+idx].stripped_strings)
                racer_data[f"{pfx}{key}_No"] = lines[0]
                racer_data[f"{pfx}{key}_2Ren"] = lines[1]
                racer_data[f"{pfx}{key}_3Ren"] = lines[2]
        except: pass

    result_data = {
        'Result_Bet': "", 'Payout': "", 'Kimarite': "", 'Kimarite_Code': "", 
        'Weather': "", 'WindDirection': "", 'WindSpeed': "", 'WaveHeight': "", 'WaterTemp': "",
        'Result_2Tan_Bet': "", 'Payout_2Tan': "", 'Result_2Fuku_Bet': "", 'Payout_2Fuku': ""
    }
    
    if soup_result:
        try:
            td_3ren = soup_result.find('td', string=re.compile(r'3連単'))
            if td_3ren:
                row_val = td_3ren.find_parent('tr')
                payout_span = row_val.find('span', class_='is-payout1')
                if payout_span: result_data['Payout'] = re.sub(r'[^\d]', '', payout_span.text)
                if not result_data.get('Payout') or result_data['Payout'] == '0': return None
                number_div = row_val.find('div', class_='numberSet1_row')
                if number_div: result_data['Result_Bet'] = "-".join([n.text.strip() for n in number_div.find_all('span', class_='numberSet1_number')])
        except: pass
        try:
            td_2tan = soup_result.find('td', string=re.compile(r'2連単'))
            if td_2tan:
                row_val = td_2tan.find_parent('tr')
                payout_span = row_val.find('span', class_='is-payout1')
                if payout_span: result_data['Payout_2Tan'] = re.sub(r'[^\d]', '', payout_span.text)
                number_div = row_val.find('div', class_='numberSet1_row')
                if number_div: result_data['Result_2Tan_Bet'] = "".join([n.text.strip() for n in number_div.find_all('span')])
        except: pass
        try:
            td_2fuku = soup_result.find('td', string=re.compile(r'2連複'))
            if td_2fuku:
                row_val = td_2fuku.find_parent('tr')
                payout_span = row_val.find('span', class_='is-payout1')
                if payout_span: result_data['Payout_2Fuku'] = re.sub(r'[^\d]', '', payout_span.text)
                number_div = row_val.find('div', class_='numberSet1_row')
                if number_div: result_data['Result_2Fuku_Bet'] = "".join([n.text.strip() for n in number_div.find_all('span')])
        except: pass

        try:
            for tbl in soup_result.find_all('table'):
                if '決まり手' in tbl.text:
                    kimarite_td = tbl.find('tbody').find('td')
                    if kimarite_td:
                        val = kimarite_td.text.strip()
                        result_data['Kimarite'] = val
                        result_data['Kimarite_Code'] = {'逃げ':1, '差し':2, 'まくり':3, 'まくり差し':4, '抜き':5, '恵まれ':6}.get(val, 0)
                    break
        except: pass
        try:
            w_block = soup_result.find('div', class_='weather1_body')
            if w_block:
                for cls_name, key in [('is-wind', 'WindSpeed'), ('is-wave', 'WaveHeight'), ('is-waterTemperature', 'WaterTemp')]:
                    unit = w_block.find('div', class_=cls_name)
                    if unit and unit.find('span', class_='weather1_bodyUnitLabelData'): result_data[key] = get_float(unit.find('span', class_='weather1_bodyUnitLabelData').text)
                weather_unit = w_block.find('div', class_='is-weather')
                if weather_unit: result_data['Weather'] = weather_unit.find('span', class_='weather1_bodyUnitLabelTitle').text.strip()
                wind_dir_p = w_block.find('p', class_='weather1_bodyUnitImage')
                if wind_dir_p:
                    for c in wind_dir_p.get('class'):
                        if 'is-direction' in c: result_data['WindDirection'] = c.replace('is-direction', '')
        except: pass
        tables = soup_result.find_all('table')
        for tbl in tables:
            headers = [th.get_text(strip=True) for th in tbl.find_all('th')]
            if '着' in headers and 'レースタイム' in headers:
                for tbody in tbl.find_all('tbody'):
                    row_tr = tbody.find('tr')
                    if row_tr and len(row_tr.find_all('td')) >= 4:
                        cells = row_tr.find_all('td')
                        b_txt = cells[1].get_text(strip=True)
                        if b_txt.isdigit():
                            result_data[f"Result_Boat{b_txt}_Rank"] = clean_rank_value(cells[0].get_text(strip=True))
                            result_data[f"Result_Boat{b_txt}_Time"] = cells[3].get_text(strip=True).replace("\u3000", "").strip()
        for st_div in soup_result.find_all('div', class_=re.compile('table1_boatImage1')):
            num_span = st_div.find('span', class_=re.compile('table1_boatImage1Number'))
            if num_span:
                b_num = num_span.get_text(strip=True)
                time_span = st_div.find('span', class_='table1_boatImage1TimeInner')
                if time_span:
                    match = re.search(r'(F?\.?\d+)', time_span.get_text(strip=True))
                    result_data[f"Result_Boat{b_num}_ST"] = match.group(1) if match else time_span.get_text(strip=True)
                    
    additional_data = extract_additional_data(soup_list, date_str)
    
    row = {'Date': date_str, 'PlaceID': jcd_str, 'PlaceName': JCD_MAP.get(jcd_str, ""), 'RaceNum': r, 'RaceName': race_name, 'RaceInfo_URL': url_list, 'RaceResult_URL': url_result, 'Project_ID': project_id, 'Race_Type': race_type}
    row.update(result_data)
    row.update(flags)
    row.update(racer_data)
    row.update(additional_data)
    return row

# =============================================================================
# Google Drive API 操作関数
# =============================================================================
def get_drive_service():
    scopes = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
    return build('drive', 'v3', credentials=creds)

def download_csv(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh, dtype=str)

def upload_csv(service, df, file_id):
    temp_filename = 'temp_upload.csv'
    df.to_csv(temp_filename, index=False, encoding='utf-8')
    media = MediaFileUpload(temp_filename, mimetype='text/csv', resumable=True)
    service.files().update(fileId=file_id, media_body=media).execute()
    os.remove(temp_filename)

# =============================================================================
# メイン処理
# =============================================================================
def main():
    if not TARGET_FILE_ID:
        print("エラー: DRIVE_FILE_IDが設定されていません。")
        return
        
    print("Google Driveからマスターデータをダウンロードしています...")
    service = get_drive_service()
    try:
        df = download_csv(service, TARGET_FILE_ID)
    except Exception as e:
        print(f"ファイルのダウンロードに失敗しました: {e}")
        return

    max_date_str = str(df['Date'].max())
    latest_date = datetime.strptime(max_date_str, '%Y%m%d')
    start_date = latest_date + timedelta(days=1)

    jst = pytz.timezone('Asia/Tokyo')
    now_jst = datetime.now(jst)
    end_date = (now_jst - timedelta(days=1)).replace(tzinfo=None)

    if start_date > end_date:
        print(f"すでに最新のデータ（{max_date_str}）まで取得済みです。更新は行いません。")
        return

    print(f"今回取得する期間: {start_date.strftime('%Y%m%d')} 〜 {end_date.strftime('%Y%m%d')}")

    tasks = []
    d = start_date
    while d <= end_date:
        d_str = d.strftime('%Y%m%d')
        for j in range(1, 25):
            for r in range(1, 13):
                tasks.append((d_str, j, r))
        d += timedelta(days=1)

    print(f"=== スクレイピング開始 (予定タスク数: {len(tasks)}) ===")
    new_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parse_single_race, t): t for t in tasks}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            try:
                res = f.result()
                if res: new_results.append(res)
            except Exception: pass

    if new_results:
        print(f"計 {len(new_results)} 件の新規データを結合し、Google Driveへアップロードします...")
        new_df = pd.DataFrame(new_results)
        for col in df.columns:
            if col not in new_df.columns: new_df[col] = ""
        new_df = new_df[df.columns]
        
        df = pd.concat([df, new_df], ignore_index=True)
        upload_csv(service, df, TARGET_FILE_ID)
        print("Google Driveのマスターデータ更新が完了しました！")
    else:
        print("新規データは見つかりませんでした。")

if __name__ == "__main__":
    main()
