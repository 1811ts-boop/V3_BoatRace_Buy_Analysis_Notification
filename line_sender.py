import os
import json
import logging
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

GCP_SA_CREDENTIALS = os.environ.get("GCP_SA_CREDENTIALS")
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID")
LINE_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")

def main():
    if not all([GCP_SA_CREDENTIALS, SPREADSHEET_ID, LINE_TOKEN]):
        logger.error("必要な環境変数が設定されていません。")
        return

    # Google Sheets API 認証
    creds_dict = json.loads(GCP_SA_CREDENTIALS)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=creds)

    # LINE_Queueシートから溜まったメッセージを取得
    range_name = "LINE_Queue!A:A"
    result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
    rows = result.get('values', [])

    if not rows:
        logger.info("本日の勝負レース（通知対象）はありませんでした。")
        return

    messages = [row[0] for row in rows if row]
    
    # ▼▼ 文字数オーバー対策：自動分割（チャンキング）処理 ▼▼
    MAX_BUBBLE_LENGTH = 4500 # LINE上限は5000だが、余裕を持たせる
    bubbles = []
    current_bubble = ""

    for msg in messages:
        separator = "\n\n━━━━━━━━━━━━━━\n\n" if current_bubble else ""
        # 結合した結果が上限を超える場合は、現在の塊をリストに保存して新しく作り直す
        if len(current_bubble) + len(separator) + len(msg) > MAX_BUBBLE_LENGTH:
            bubbles.append(current_bubble)
            current_bubble = msg
        else:
            current_bubble += separator + msg

    if current_bubble:
        bubbles.append(current_bubble)

    # LINE APIは1回で最大5つの吹き出しまでしか送れないため、5つずつに分割して送信
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_TOKEN}"
    }
    
    all_success = True
    for i in range(0, len(bubbles), 5):
        chunked_bubbles = bubbles[i:i+5]
        data = {
            "messages": [{"type": "text", "text": b} for b in chunked_bubbles]
        }
        
        logger.info(f"LINEへ一括送信を実行します (バッチ {i//5 + 1})")
        res = requests.post("https://api.line.me/v2/bot/message/broadcast", headers=headers, json=data)
        
        if res.status_code != 200:
            logger.error(f"❌ LINE送信エラー: {res.status_code} - {res.text}")
            all_success = False
            break # 1つでも失敗したらそこで止める（シートはクリアしない）

    # 全てのバッチ送信が成功した場合のみ、シートのキューを空にする
    if all_success:
        logger.info("✅ 全てのLINE送信成功！キューをクリアします。")
        service.spreadsheets().values().clear(
            spreadsheetId=SPREADSHEET_ID, range=range_name
        ).execute()

if __name__ == "__main__":
    main()
