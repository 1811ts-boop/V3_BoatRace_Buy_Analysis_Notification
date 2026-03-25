import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os
import io
import json
import logging
import concurrent.futures
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 設定・環境変数
# =============================================================================
MAX_WORKERS = 3
TIMEOUT_SEC = 20

# GitHub Actionsの環境変数から取得（V9推論コードと統一）
GCP_SA_CREDENTIALS = os.environ.get("GCP_SA_CREDENTIALS")
TARGET_FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')
FILE_NAME = "BoatRace_Master_Updated_with_2Tan_2Fuku.csv"

# ロガー設定
logger = logging.getLogger("V9_Master_Updater")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(sh)

# =============================================================================
# 2. ヘルパー関数群（V9の推論ロジックと完全同期）
# =============================================================================
JCD_MAP = {f"{i:02d}": name for i, name in enumerate(["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"], 1)}

def clean_str(s): return s.replace('\u3000', '').strip() if s else ""

# 💡 修正1: フライング等に対応したV9完全互換の float 変換
def safe_float(val, default=0.0):
    if pd.isna(val) or val == "" or val is None: return default
    try: return float(val)
    except ValueError:
        s = str(val).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        clean_val = re.sub(r'[^\d.-]', '', s)
        if clean_val in ('', '-', '.', '-.'): return default
        try: return float(clean_val)
        except ValueError: return default

def clean_rank_value(val):
    if not val: return None
    v = str(val).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    return float(v) if v in ['1', '2', '3', '4', '5', '6'] else None

def fetch_soup(url, retries=3):
    for _ in range(retries):
        try:
            time.sleep(1.0)
            resp = requests.get(url, timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
        except: time.sleep(2)
    return None

# 💡 修正2: V9推論コードと全く同じ節間（Tournament_Day）算出ロジック
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

# =============================================================================
# 3. コア・パーサー
# =============================================================================
def parse_single_race(task_tuple):
    date_str, jcd, r = task_tuple
    jcd_str = str(jcd).zfill(2)
    
    url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={r}&jcd={jcd_str}&hd={date_str}"
    soup_list = fetch_soup(url_list)
    if not soup_list or "データがありません" in soup_list.text or "中止" in soup_list.text: return None
    
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
        'Is_Rookie': 1 if any(w in race_name for w in ['ルーキー', 'ヤング', '若手']) else 0,
        'Is_Master': 1 if any(w in race_name for w in ['マスターズ', '名人']) else 0
    }
    if sum([flags['Is_SG'], flags['Is_G1'], flags['Is_G2'], flags['Is_G3']]) == 0: flags['Is_General'] = 1
    
    race_type = "一般"
    if '準優勝戦' in race_name: race_type = "準優勝戦"
    elif '優勝戦' in race_name: race_type = "優勝戦"
    elif 'ドリーム' in race_name: race_type = "ドリーム"
    elif '特選' in race_name: race_type = "特選"
    
    project_id = "P3_General_Std"
    if flags['Is_SG']: project_id = "P0_SG"
    elif flags['Is_G1'] and not flags['Is_Rookie']: project_id = "P1_G1_Elite"
    elif flags['Is_General'] and (r == 1 or "進入固定" in race_name or "シード" in race_name): project_id = "P4_Planning"
    if flags['Is_Lady']: project_id = "P2_Ladies"

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
                if len(lines) > 2: racer_data[f"{pfx}{key}_3Ren"] = lines[2]
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
                if number_div: result_data['Result_2Tan_Bet'] = "-".join([n.text.strip() for n in number_div.find_all('span', class_='numberSet1_number')])
        except: pass
        try:
            td_2fuku = soup_result.find('td', string=re.compile(r'2連複'))
            if td_2fuku:
                row_val = td_2fuku.find_parent('tr')
                payout_span = row_val.find('span', class_='is-payout1')
                if payout_span: result_data['Payout_2Fuku'] = re.sub(r'[^\d]', '', payout_span.text)
                number_div = row_val.find('div', class_='numberSet1_row')
                if number_div: result_data['Result_2Fuku_Bet'] = "=".join([n.text.strip() for n in number_div.find_all('span', class_='numberSet1_number')])
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
                    if unit and unit.find('span', class_='weather1_bodyUnitLabelData'): result_data[key] = safe_float(unit.find('span', class_='weather1_bodyUnitLabelData').text)
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
                    
    additional_data = extract_additional_data(soup_list)
    
    row = {'Date': date_str, 'PlaceID': jcd_str, 'PlaceName': JCD_MAP.get(jcd_str, ""), 'RaceNum': str(r), 'RaceName': race_name, 'RaceInfo_URL': url_list, 'RaceResult_URL': url_result, 'Project_ID': project_id, 'Race_Type': race_type}
    row.update(result_data)
    row.update(flags)
    row.update(racer_data)
    row.update(additional_data)
    return row

# =============================================================================
# 4. Google Drive API 操作関数（堅牢なアトミック更新版）
# =============================================================================
def get_drive_service():
    # 💡 修正3: 環境変数からJSONを読み込む方式に統一
    if not GCP_SA_CREDENTIALS: return None
    creds_dict = json.loads(GCP_SA_CREDENTIALS)
    creds = service_account.Credentials.from_service_account_info(creds_dict)
    return build('drive', 'v3', credentials=creds)

def get_master_file_id(service, folder_id, file_name):
    query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    if items: return items[0]['id']
    return None

def download_csv(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh, dtype=str)

def safe_upload_csv(service, df, folder_id, final_file_name, old_file_id):
    local_tmp = 'temp_upload.csv'
    df.to_csv(local_tmp, index=False, encoding='utf-8-sig') # 💡 utf-8-sigで文字化け防止
    
    temp_name = f"temp_{final_file_name}"
    file_metadata = {'name': temp_name, 'parents': [folder_id]}
    media = MediaFileUpload(local_tmp, mimetype='text/csv', resumable=True)
    
    logger.info(f"新データ（{temp_name}）をアップロード中...")
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    new_file_id = uploaded_file.get('id')
    logger.info("アップロード完了！安全なすり替え処理を実行します...")
    
    if old_file_id:
        backup_name = f"backup_{final_file_name}"
        logger.info(f"既存のファイルを退避中 ({backup_name})...")
        service.files().update(fileId=old_file_id, body={'name': backup_name}).execute()
            
    logger.info(f"新ファイル名を正式名称（{final_file_name}）に変更中...")
    service.files().update(fileId=new_file_id, body={'name': final_file_name}).execute()
    
    if old_file_id:
        try:
            logger.info("退避した古いマスターファイルを削除しています...")
            service.files().delete(fileId=old_file_id).execute()
        except Exception as e:
            logger.warning(f"古いファイルの削除に失敗しましたが、更新自体は成功しています: {e}")
            
    os.remove(local_tmp)
    logger.info("完全なアトミック更新が完了しました！")

# =============================================================================
# メイン処理
# =============================================================================
def main():
    if not TARGET_FOLDER_ID:
        logger.error("エラー: GDRIVE_FOLDER_ID が設定されていません。")
        return
        
    service = get_drive_service()
    if not service:
        logger.error("GCP認証に失敗しました。")
        return
        
    logger.info(f"フォルダから '{FILE_NAME}' を探しています...")
    old_file_id = get_master_file_id(service, TARGET_FOLDER_ID, FILE_NAME)
    
    if not old_file_id:
        logger.error("エラー: フォルダ内に指定されたマスターファイルが見つかりません。")
        return
        
    logger.info("マスターデータをダウンロードしています...")
    try:
        df = download_csv(service, old_file_id)
    except Exception as e:
        logger.error(f"ファイルのダウンロードに失敗しました: {e}")
        return

    max_date_str = str(df['Date'].max())
    latest_date = datetime.strptime(max_date_str, '%Y%m%d')
    start_date = latest_date + timedelta(days=1)

    jst = pytz.timezone('Asia/Tokyo')
    now_jst = datetime.now(jst)
    # 昨日までのデータを取得
    end_date = (now_jst - timedelta(days=1)).replace(tzinfo=None)

    if start_date > end_date:
        logger.info(f"すでに最新のデータ（{max_date_str}）まで取得済みです。更新は行いません。")
        return

    logger.info(f"今回取得する期間: {start_date.strftime('%Y%m%d')} 〜 {end_date.strftime('%Y%m%d')}")

    tasks = []
    d = start_date
    while d <= end_date:
        d_str = d.strftime('%Y%m%d')
        for j in range(1, 25):
            for r in range(1, 13):
                tasks.append((d_str, j, r))
        d += timedelta(days=1)

    logger.info(f"=== スクレイピング開始 (予定タスク数: {len(tasks)}) ===")
    new_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parse_single_race, t): t for t in tasks}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            try:
                res = f.result()
                if res: new_results.append(res)
            except Exception as e:
                logger.error(f"タスク処理中にエラーが発生しました: {e}")

    if new_results:
        logger.info(f"計 {len(new_results)} 件の新規データを取得しました。結合処理を行います...")
        new_df = pd.DataFrame(new_results, dtype=str)
        
        # カラム順を既存マスターに揃える（欠損カラムは空文字で埋める）
        for col in df.columns:
            if col not in new_df.columns: new_df[col] = ""
        new_df = new_df[df.columns]
        
        df = pd.concat([df, new_df], ignore_index=True)
        
        # アトミックアップロード
        safe_upload_csv(service, df, TARGET_FOLDER_ID, FILE_NAME, old_file_id)
    else:
        logger.info("新規データは見つかりませんでした（中止など）。")

if __name__ == "__main__":
    main()
