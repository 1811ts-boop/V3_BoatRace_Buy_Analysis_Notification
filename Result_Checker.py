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
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID") # 💡 環境変数か、ここに直接ID文字列を入れてください
MASTER_CSV_NAME = "BoatRace_Master_Updated_with_2Tan_2Fuku.csv"
SHEET_RANGE = 'シート1!A:R' # シート名が違う場合は変更してください

# 場名からPlaceIDへの変換辞書
JCD_MAP = {name: i for i, name in enumerate(["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"], 1)}

# =============================================================================
# 2. Google API連携
# =============================================================================
def get_gcp_credentials():
    if not GCP_SA_CREDENTIALS: return None
    creds_dict = json.loads(GCP_SA_CREDENTIALS)
    return service_account.Credentials.from_service_account_info(creds_dict)

def download_master_csv(creds):
    logger.info("☁️ Google Driveから最新のマスターデータをダウンロードします...")
    service = build('drive', 'v3', credentials=creds)
    query = f"name='{MASTER_CSV_NAME}' and trashed=false"
    res = service.files().list(q=query, orderBy="createdTime desc", fields="files(id, name)").execute()
    
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

    # 1. マスターデータの準備
    if not download_master_csv(creds): return
    
    logger.info("📊 マスターデータを読み込んでいます...")
    cols = ['Date', 'PlaceID', 'RaceNum', 'Payout_2Tan', 'Result_2Tan_Bet', 'Payout_2Fuku', 'Result_2Fuku_Bet']
    df_master = pd.read_csv(MASTER_CSV_NAME, usecols=lambda c: c in cols, dtype=str)
    
    # 検索しやすいようにキー（YYYYMMDD, PlaceID, RaceNum）を作成
    df_master['DateInt'] = pd.to_datetime(df_master['Date'], errors='coerce').dt.strftime('%Y%m%d').fillna('0')
    df_master['PlaceID'] = pd.to_numeric(df_master['PlaceID'], errors='coerce').fillna(0).astype(int)
    df_master['RaceNum'] = pd.to_numeric(df_master['RaceNum'], errors='coerce').fillna(0).astype(int)

    # 2. スプレッドシートのデータ取得
    sheets_service = build('sheets', 'v4', credentials=creds)
    sheet_data = sheets_service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=SHEET_RANGE
    ).execute().get('values', [])
    
    if not sheet_data:
        logger.info("📝 スプレッドシートにデータがありません。")
        return

    batch_update_data = []
    updated_count = 0

    # 3. 1行ずつ確認して答え合わせ
    for i, row in enumerate(sheet_data):
        if i == 0: continue # ヘッダー行をスキップ
        
        # 列数が足りない場合は空文字で埋める（L〜R列まで確保）
        row += [""] * (18 - len(row))
        
        date_str = str(row[0]).strip()
        place_name = str(row[3]).strip()
        race_str = str(row[4]).strip()
        ticket_type = str(row[6]).strip() # 2連単 or 2連複
        ticket = str(row[7]).strip()
        result_cell = str(row[11]).strip() # L列 (結果)
        
        # すでに結果が記入されている行、または必須データがない行はスキップ
        if result_cell != "" or not date_str or not place_name or not race_str:
            continue
            
        date_int = date_str.replace('/', '').replace('-', '')
        place_id = JCD_MAP.get(place_name, 0)
        try: race_num = int(race_str)
        except: continue
        
        # マスターデータから該当レースを検索
        match = df_master[(df_master['DateInt'] == date_int) & 
                          (df_master['PlaceID'] == place_id) & 
                          (df_master['RaceNum'] == race_num)]
                          
        if match.empty:
            continue # まだマスターデータに反映されていない場合はパス

        actual = match.iloc[0]
        
        # 券種に応じた正解と配当を取得
        if "単" in ticket_type:
            act_ticket = str(actual.get('Result_2Tan_Bet', '')).strip()
            payout_str = str(actual.get('Payout_2Tan', '0')).replace(',', '')
        else:
            act_ticket = str(actual.get('Result_2Fuku_Bet', '')).strip()
            payout_str = str(actual.get('Payout_2Fuku', '0')).replace(',', '')

        try: act_payout_100 = float(payout_str) if payout_str and payout_str != 'nan' else 0
        except: act_payout_100 = 0

        # === 投資額の取得 ===
        try: basic_bet = int(row[12]) if row[12] else 100
        except: basic_bet = 100
        try: quant_bet = int(row[15]) if row[15] else 100
        except: quant_bet = 100

        # === 的中判定と収支計算 ===
        if act_ticket == "" or act_ticket == "nan":
            # 不成立・返還などの場合
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

        # === スプレッドシート更新用データの作成 (L列〜R列) ===
        update_values = [
            result_mark,       # L列: 結果
            basic_bet,         # M列: 等倍_投資
            basic_payout,      # N列: 等倍_配当
            basic_profit,      # O列: 等倍_収支
            quant_bet,         # P列: クオンツ_投資
            quant_payout,      # Q列: クオンツ_配当
            quant_profit       # R列: クオンツ_収支
        ]
        
        # バッチ更新リストに追加 (1-indexedなので i+1 行目)
        batch_update_data.append({
            'range': f'シート1!L{i+1}:R{i+1}',
            'values': [update_values]
        })
        updated_count += 1

    # 4. スプレッドシートへ一括書き込み（バッチ処理）
    if batch_update_data:
        logger.info(f"📤 {updated_count}件のレース結果をスプレッドシートに書き込みます...")
        body = {
            'valueInputOption': 'USER_ENTERED',
            'data': batch_update_data
        }
        sheets_service.spreadsheets().values().batchUpdate(
            spreadsheetId=SPREADSHEET_ID, body=body
        ).execute()
        logger.info("✅ 書き込みが完了しました！")
    else:
        logger.info("✅ 新しく更新すべきレース結果はありませんでした。")

if __name__ == "__main__":
    main()
