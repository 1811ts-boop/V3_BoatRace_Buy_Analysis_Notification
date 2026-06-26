import os
import json
import logging
import requests
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

GCP_SA_CREDENTIALS = os.environ.get("GCP_SA_CREDENTIALS")
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID")
LINE_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")

# 全国ボートレース場コード（JCD）順のリスト（この順番でソートされます）
JCD_LIST = ["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"]
JCD_DICT = {name: i for i, name in enumerate(JCD_LIST)}

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
    
    # ▼▼ メッセージを一度解体してリスト化する ▼▼
    all_bets = []
    for msg in messages:
        # どのAIからのメッセージかを判定
        ai_name = "V9" if "V9" in msg else "V10" if "V10" in msg else "V11" if "V11" in msg else "V12" if "V12" in msg else "不明"
        
        current_place = "不明"
        for line in msg.split('\n'):
            line = line.strip()
            if line.startswith('◎'):
                current_place = line[1:] # 会場名を取得
            elif line.startswith('['):
                # 正規表現で買い目データを抽出 (例: [12:34] 5R 🎯2単: 1-3 💰3倍)
                m = re.match(r'\[(.*?)\]\s+(\d+)R\s+(.*?):\s+(.*?)\s+(.*)', line)
                if m:
                    all_bets.append({
                        'place': current_place,
                        'time': m.group(1),
                        'race': int(m.group(2)),
                        'type': m.group(3),
                        'ticket': m.group(4),
                        'multi_icon': m.group(5),
                        'ai': ai_name
                    })

    if not all_bets:
        logger.info("有効な買い目データがパースできませんでした。")
        return

    # ▼▼ リストを完璧な順番にソートする ▼▼
    # 優先度: 1. 会場順(JCD順) -> 2. レース順 -> 3. 券種(2単が先) -> 4. AI順
    all_bets.sort(key=lambda x: (
        JCD_DICT.get(x['place'], 99), 
        x['race'], 
        0 if '2単' in x['type'] else 1,
        x['ai']
    ))

    # ▼▼ LINE用のテキストを組み立てる ▼▼
    today_str = datetime.now(timezone(timedelta(hours=9), 'JST')).strftime('%Y年%m月%d日')
    header_main = f"🚤 本日のクオンツ厳選勝負レース 🚤\n📅 {today_str} (計 {len(all_bets)} 件)\n"
    header_main += "━━━━━━━━━━━━━━\n"
    header_main += "【凡例】 🔥5倍 / 💰3倍 / 🪙1倍\n"
    header_main += "━━━━━━━━━━━━━━\n"

    combined_msg = header_main
    prev_place = ""
    for b in all_bets:
        if b['place'] != prev_place:
            combined_msg += f"\n◎{b['place']}\n"
            prev_place = b['place']
        
        # 行の最後に (V9) などを追加して出力
        combined_msg += f"[{b['time']}] {b['race']}R {b['type']}: {b['ticket']} {b['multi_icon']} ({b['ai']})\n"

    # ▼▼ 文字数オーバー対策：自動分割（チャンキング）処理 ▼▼
    MAX_BUBBLE_LENGTH = 4500
    bubbles = []
    current_bubble = ""

    lines = combined_msg.strip().split('\n')
    for line in lines:
        if len(current_bubble) + len(line) + 1 > MAX_BUBBLE_LENGTH:
            bubbles.append(current_bubble)
            current_bubble = line
        else:
            current_bubble += ("\n" + line) if current_bubble else line

    if current_bubble:
        bubbles.append(current_bubble)

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
            break

    # 全てのバッチ送信が成功した場合のみ、シートのキューを空にする
    if all_success:
        logger.info("✅ 全てのLINE送信成功！キューをクリアします。")
        service.spreadsheets().values().clear(
            spreadsheetId=SPREADSHEET_ID, range=range_name
        ).execute()

if __name__ == "__main__":
    main()
