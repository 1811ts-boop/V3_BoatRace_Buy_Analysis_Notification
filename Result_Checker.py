import os
import json
import logging
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# =============================================================================
# 1. 環境設定・定数
# =============================================================================
logger = logging.getLogger("Result_Checker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(sh)

GCP_SA_CREDENTIALS = os.environ.get("GCP_SA_CREDENTIALS")
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID")
MASTER_CSV_NAME = "BoatRace_Master_Updated_with_2Tan_2Fuku.csv"
SHEET_RANGE = 'Sheet1!A:R' # 💡シート名に合わせて変更してください（例：シート1!A:R）

# 場名からPlaceIDへの変換辞書
JCD_MAP = {name: i for i, name in enumerate(["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"], 1)}

# =============================================================================
# 2. Google API連携
# =============================================================================
def get_gcp_credentials():
    if not GCP_SA_CREDENTIALS: return None
    creds_dict = json.loads(GCP_SA_CREDENTIALS)
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    return service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)

def download_master_csv(creds):
    logger.info("☁️ Google Driveから最新のマスターデータをダウンロードします...")
    service = build('drive', 'v3', credentials=creds)
    query = f"name='{MASTER_CSV_NAME}' and trashed=false"
    
    # 💡【重要修正】作成日時ではなく「直前にScraperが上書きしたファイル(modifiedTime)」を掴むように変更
    res = service.files().list(q=query, orderBy="modifiedTime desc", fields="files(id, name)").execute()
    
    if not res.get('files'):
        logger.error("❌ マスターデータが見つかりません。")
        return False
        
    file_id = res['files'][0]['id']
    req = service.files().get_media(fileId=file_id)
    with io.FileIO(MASTER_CSV_NAME, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done: _, done = downloader.next_chunk()
    return True

# =============================================================================
# 3. メイン処理：答え合わせとスプレッドシート更新
# =============================================================================
def main():
    logger.info("🚀 スプレッドシート結果自動チェック処理を開始します...")
    
    creds = get_gcp_credentials()
    if not creds or not SPREADSHEET_ID:
        logger.error("❌ 認証情報またはSPREADSHEET_IDが設定されていません。")
        return

    if not download_master_csv(creds): return
    
    logger.info("📊 マスターデータを読み込んでいます...")
    cols = ['Date', 'PlaceID', 'RaceNum', 'Payout_2Tan', 'Result_2Tan_Bet', 'Payout_2Fuku', 'Result_2Fuku_Bet']
    df_master = pd.read_csv(MASTER_CSV_NAME, usecols=lambda c: c in cols, dtype=str)
    
    # 💡【重要修正】日付形式のブレを完全に吸収して 20260402 形式に強制統一
    df_master['DateInt'] = df_master['Date'].astype(str).str.replace('-', '').str.replace('/', '').str.replace('.', '').str.slice(0, 8)
    df_master['PlaceID'] = pd.to_numeric(df_master['PlaceID'], errors='coerce').fillna(0).astype(int)
    df_master['RaceNum'] = pd.to_numeric(df_master['RaceNum'], errors='coerce').fillna(0).astype(int)

    sheets_service = build('sheets', 'v4', credentials=creds)
    sheet_data = sheets_service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=SHEET_RANGE
    ).execute().get('values', [])
    
    if not sheet_data: return

    batch_update_data = []
    updated_count = 0

    for i, row in enumerate(sheet_data):
        if i == 0: continue 
        row += [""] * (18 - len(row))
        
        date_str = str(row[0]).strip()
        place_name = str(row[3]).strip()
        race_str = str(row[4]).strip()
        ticket_type = str(row[6]).strip()
        ticket = str(row[7]).strip()
        result_cell = str(row[11]).strip()
        
        if result_cell != "" or not date_str or not place_name or not race_str:
            continue
            
        date_int = date_str.replace('/', '').replace('-', '')
        place_id = JCD_MAP.get(place_name, 0)
        try: race_num = int(race_str)
        except: continue
        
        match = df_master[(df_master['DateInt'] == date_int) & 
                          (df_master['PlaceID'] == place_id) & 
                          (df_master['RaceNum'] == race_num)]
                          
        if match.empty:
            # 💡【追加】なぜスキップしたのかをログに表示する機能
            logger.info(f"⏩ スキップ: {date_str} {place_name} {race_str}R の結果は、ダウンロードしたマスターデータ内に存在しませんでした。")
            continue

        actual = match.iloc[0]
        
        if "単" in ticket_type:
            act_ticket = str(actual.get('Result_2Tan_Bet', '')).strip()
            payout_str = str(actual.get('Payout_2Tan', '0')).replace(',', '')
        else:
            act_ticket = str(actual.get('Result_2Fuku_Bet', '')).strip()
            payout_str = str(actual.get('Payout_2Fuku', '0')).replace(',', '')

        try: act_payout_100 = float(payout_str) if payout_str and payout_str != 'nan' else 0
        except: act_payout_100 = 0

        try: basic_bet = int(row[12]) if row[12] else 100
        except: basic_bet = 100
        try: quant_bet = int(row[15]) if row[15] else 100
        except: quant_bet = 100

        if act_ticket == "" or act_ticket == "nan":
            result_mark = "⚠️返還等"
            basic_payout = basic_bet
            quant_payout = quant_bet
            basic_profit = 0
            quant_profit = 0
        elif ticket == act_ticket:
            result_mark = "🎯的中"
            basic_payout = int((act_payout_100 / 100) * basic_bet)
            quant_payout = int((act_payout_100 / 100) * quant_bet)
            basic_profit = basic_payout - basic_bet
            quant_profit = quant_payout - quant_bet
        else:
            result_mark = "❌不的中"
            basic_payout = 0
            quant_payout = 0
            basic_profit = -basic_bet
            quant_profit = -quant_bet

        update_values = [result_mark, basic_bet, basic_payout, basic_profit, quant_bet, quant_payout, quant_profit]
        # 💡 シート名に合わせて変更してください（例：シート1!L...）
        batch_update_data.append({'range': f'Sheet1!L{i+1}:R{i+1}', 'values': [update_values]})
        updated_count += 1

    if batch_update_data:
        logger.info(f"📤 {updated_count}件のレース結果をスプレッドシートに書き込みます...")
        body = {'valueInputOption': 'USER_ENTERED', 'data': batch_update_data}
        sheets_service.spreadsheets().values().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()
        logger.info("✅ 書き込みが完了しました！")
    else:
        logger.info("✅ 新しく更新すべきレース結果はありませんでした。")

if __name__ == "__main__":
    main()
