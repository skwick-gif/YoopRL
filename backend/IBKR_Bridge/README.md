# 🌉 IBKR Bridge Package - חבילת חיבור ל-Interactive Brokers

חבילה מוכנה לשימוש עם פתרון מלא לחיבור ל-IBKR דרך C# Bridge ו-Python Adapter.

---

## 📦 מה כלול בחבילה?

### 1. **C# Bridge** (תיקייה: `csharp_bridge/`)
- **InterReactBridge** - שרת ASP.NET Core עם REST API ו-SignalR
- מבוסס על ספריית **InterReact** (C# wrapper ל-TWS/Gateway)
- מספק API מלא לחיבור, נתוני חשבון, פוזיציות, ביצוע פקודות ועוד

### 2. **Python Adapter** (תיקייה: `python_adapter/`)
- **interreact_bridge_adapter.py** - מתאם Python לשרת C#
- תומך ב-PyQt6/PyQt5 עם QThread לביצועים מיטביים
- מספק ממשק פשוט וידידותי

### 3. **תיעוד מלא** (קובץ זה)
- הוראות התקנה והרצה
- דוגמאות קוד
- טיפים ו-troubleshooting

---

## 🚀 התקנה והרצה - מדריך מהיר

### שלב 1: הכנת הסביבה

#### דרישות מקדימות:
- **Windows 10/11**
- **.NET 8.0 SDK** - [הורד כאן](https://dotnet.microsoft.com/download/dotnet/8.0)
- **Python 3.10+** (אם משתמשים ב-Python Adapter)
- **IBKR TWS או Gateway** מותקן ופועל

#### התקנת .NET SDK (אם לא מותקן):
```powershell
# בדוק אם יש .NET 8
dotnet --version

# אם אין, הורד והתקן מהקישור למעלה
```

---

### שלב 2: הרצת C# Bridge

#### אופציה א': הרצה ישירה (מהירה)
```powershell
cd IBKR_Bridge_Package\csharp_bridge
dotnet run --project InterReactBridge.csproj
```

#### אופציה ב': Build והרצה
```powershell
cd IBKR_Bridge_Package\csharp_bridge
dotnet build
dotnet run
```

#### בדיקה שהשרת רץ:
פתח דפדפן ונווט ל: `http://localhost:5080/health`

אם הכל תקין, תראה: `{ "status": "Healthy" }`

---

### שלב 3: חיבור ל-IBKR TWS/Gateway

#### ודא ש-TWS/Gateway פועל:
1. פתח את IBKR TWS או Gateway
2. עבור להגדרות API: **Configure → Settings → API → Settings**
3. **אפשר:** 
   - ☑️ Enable ActiveX and Socket Clients
   - ☑️ Read-Only API
   - ☑️ Download open orders on connection
4. **הגדר פורט:** 
   - TWS Paper: `7497`
   - Gateway: `4001` או `4002`
5. **רשום IP:** `127.0.0.1` (localhost)

#### חבר דרך ה-Bridge:
```powershell
# שלח POST request לחיבור
curl -X POST http://localhost:5080/connect `
  -H "Content-Type: application/json" `
  -d '{"host":"127.0.0.1","port":4001,"clientId":1}'
```

או דרך Python:
```python
import requests
response = requests.post("http://localhost:5080/connect", json={
    "host": "127.0.0.1",
    "port": 4001,
    "clientId": 1
})
print(response.json())
```

---

## 🐍 שימוש ב-Python Adapter

### התקנת תלויות:
```bash
pip install requests PyQt6  # או PyQt5
```

### דוגמה בסיסית:

```python
from python_adapter.interreact_bridge_adapter import InterReactBridgeAdapter

# צור מתאם
adapter = InterReactBridgeAdapter(host="localhost", port=5080)

# התחל ניטור (אופציונלי - רק אם משתמשים ב-QApplication)
# adapter.start_monitoring()

# בדוק חיבור
if adapter.is_connected():
    print("✅ מחובר לגשר!")
    
    # קבל נתוני חשבון
    account = adapter.get_account_summary()
    print(f"נזילות נטו: {account.get('NetLiquidation', {}).get('value', 'N/A')}")
    
    # קבל פוזיציות
    positions = adapter.get_portfolio()
    for pos in positions:
        print(f"{pos['symbol']}: {pos['position']} יחידות")
else:
    print("❌ לא מחובר לגשר")
```

### שילוב עם PyQt6:

```python
from PyQt6.QtWidgets import QApplication, QMainWindow
from python_adapter.interreact_bridge_adapter import InterReactBridgeAdapter

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # צור מתאם
        self.adapter = InterReactBridgeAdapter()
        
        # חבר ל-signals
        self.adapter.connection_status_changed.connect(self.on_connection_changed)
        self.adapter.portfolio_updated.connect(self.on_portfolio_update)
        
        # התחל ניטור
        self.adapter.start_monitoring(interval_ms=5000)  # בדיקה כל 5 שניות
    
    def on_connection_changed(self, connected: bool):
        print(f"סטטוס חיבור: {'מחובר' if connected else 'מנותק'}")
    
    def on_portfolio_update(self, portfolio: list):
        print(f"עדכון פורטפוליו: {len(portfolio)} פוזיציות")

if __name__ == "__main__":
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec()
```

---

## 🔌 API Reference - נקודות קצה זמינות

### 1. **בריאות ומערכת**

#### `GET /health`
בדיקת בריאות השרת
```bash
curl http://localhost:5080/health
```
**תגובה:**
```json
{
  "status": "Healthy",
  "timestamp": "2025-10-19T09:00:00Z"
}
```

---

### 2. **חיבור ו-Status**

#### `POST /connect`
חיבור ל-IBKR TWS/Gateway
```bash
curl -X POST http://localhost:5080/connect \
  -H "Content-Type: application/json" \
  -d '{
    "host": "127.0.0.1",
    "port": 4001,
    "clientId": 1
  }'
```
**תגובה:**
```json
{
  "success": true,
  "message": "Connected to IBKR"
}
```

#### `POST /disconnect`
ניתוק מ-IBKR
```bash
curl -X POST http://localhost:5080/disconnect
```

#### `GET /connection-status`
בדיקת סטטוס חיבור
```bash
curl http://localhost:5080/connection-status
```
**תגובה:**
```json
{
  "isConnected": true,
  "host": "127.0.0.1",
  "port": 4001,
  "clientId": 1
}
```

---

### 3. **נתוני חשבון**

#### `GET /account`
קבלת נתוני חשבון מפורטים
```bash
curl http://localhost:5080/account
```
**תגובה:**
```json
{
  "NetLiquidation": {
    "value": "100000.00",
    "currency": "USD",
    "account": "U123456"
  },
  "BuyingPower": {
    "value": "400000.00",
    "currency": "USD",
    "account": "U123456"
  },
  "TotalCashValue": {
    "value": "50000.00",
    "currency": "USD",
    "account": "U123456"
  }
}
```

---

### 4. **פוזיציות (Portfolio)**

#### `GET /portfolio`
קבלת כל הפוזיציות הפתוחות
```bash
curl http://localhost:5080/portfolio
```
**תגובה:**
```json
[
  {
    "symbol": "AAPL",
    "position": 100,
    "averageCost": 150.50,
    "marketPrice": 155.25,
    "marketValue": 15525.00,
    "unrealizedPnl": 475.00,
    "realizedPnl": 0.00,
    "account": "U123456"
  },
  {
    "symbol": "TSLA",
    "position": 50,
    "averageCost": 250.00,
    "marketPrice": 260.00,
    "marketValue": 13000.00,
    "unrealizedPnl": 500.00,
    "realizedPnl": 0.00,
    "account": "U123456"
  }
]
```

---

### 5. **ביצוע פקודות**

#### `POST /place-order`
ביצוע פקודת קנייה/מכירה
```bash
curl -X POST http://localhost:5080/place-order \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 10,
    "orderType": "MKT"
  }'
```

**פרמטרים:**
- `symbol`: סימול המניה (חובה)
- `action`: `BUY` או `SELL` (חובה)
- `quantity`: כמות (חובה)
- `orderType`: סוג פקודה (חובה)
  - `MKT` - Market
  - `LMT` - Limit
  - `STP` - Stop
  - `STP LMT` - Stop Limit
- `limitPrice`: מחיר (רק ל-LMT)
- `stopPrice`: מחיר stop (רק ל-STP)

**דוגמה - Limit Order:**
```json
{
  "symbol": "AAPL",
  "action": "BUY",
  "quantity": 10,
  "orderType": "LMT",
  "limitPrice": 150.00
}
```

**תגובה:**
```json
{
  "success": true,
  "orderId": 12345,
  "message": "Order placed successfully"
}
```

---

## 🔄 SignalR Streaming (Real-time)

ה-Bridge תומך ב-SignalR לעדכונים בזמן אמת:

### Hubs זמינים:

1. **AccountHub** (`/hubs/account`)
   - עדכוני חשבון בזמן אמת

2. **PortfolioHub** (`/hubs/portfolio`)
   - עדכוני פוזיציות בזמן אמת

3. **MarketDataHub** (`/hubs/marketdata`)
   - נתוני שוק חיים (quotes, trades)

### דוגמת שימוש (Python):
```python
# להתקנה: pip install signalrcore
from signalrcore.hub_connection_builder import HubConnectionBuilder

hub = HubConnectionBuilder() \
    .with_url("http://localhost:5080/hubs/portfolio") \
    .build()

def on_portfolio_update(data):
    print(f"Portfolio update: {data}")

hub.on("PortfolioUpdate", on_portfolio_update)
hub.start()
```

---

## ⚙️ הגדרות מתקדמות

### שינוי פורט השרת:

ערוך את `appsettings.json`:
```json
{
  "Urls": "http://localhost:5080",
  "Logging": {
    "LogLevel": {
      "Default": "Information"
    }
  }
}
```

### טיפים לביצועים:

1. **הקטן polling interval** ב-Python Adapter:
   ```python
   adapter.start_monitoring(interval_ms=10000)  # 10 שניות במקום 5
   ```

2. **השתמש ב-cached data** כשאפשר:
   ```python
   portfolio = adapter.get_cached_portfolio()  # לא עושה HTTP request
   ```

3. **השתמש ב-SignalR** למקום שצריך עדכונים תכופים

---

## 🐛 Troubleshooting

### בעיה: "Cannot connect to IBKR"

**פתרון:**
1. ודא ש-TWS/Gateway פועל
2. בדוק שהגדרות API מופעלות
3. ודא שהפורט נכון (7497 או 4001/4002)
4. בדוק firewall

### בעיה: "Bridge not responding"

**פתרון:**
1. בדוק שהשרת רץ: `curl http://localhost:5080/health`
2. בדוק logs: `logs/bridge.log`
3. הפעל מחדש: סגור והרץ `dotnet run`

### בעיה: "Slow performance"

**פתרון:**
1. הגדל polling interval ל-10-15 שניות
2. השתמש ב-cached methods
3. השתמש ב-SignalR במקום polling

### בעיה: "Thread errors in Python"

**פתרון:**
1. ודא שאתה קורא ל-`start_monitoring()` רק אחרי QApplication
2. אל תיצור InterReactBridgeAdapter מחוץ ל-main thread

---

## 📝 דוגמאות נוספות

### 1. מעקב אחרי פוזיציה ספציפית:

```python
def track_position(adapter, symbol):
    portfolio = adapter.get_portfolio()
    for pos in portfolio:
        if pos['symbol'] == symbol:
            print(f"{symbol}:")
            print(f"  כמות: {pos['position']}")
            print(f"  רווח/הפסד: ${pos['unrealizedPnl']:.2f}")
            print(f"  שווי: ${pos['marketValue']:.2f}")
            return pos
    print(f"{symbol} לא נמצא בפורטפוליו")
    return None

# שימוש
track_position(adapter, "AAPL")
```

### 2. ביצוע פקודה עם error handling:

```python
import requests

def place_order_safe(symbol, action, quantity, order_type="MKT", **kwargs):
    try:
        response = requests.post("http://localhost:5080/place-order", 
            json={
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "orderType": order_type,
                **kwargs
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        if result.get('success'):
            print(f"✅ פקודה בוצעה: Order ID {result.get('orderId')}")
            return result
        else:
            print(f"❌ פקודה נכשלה: {result.get('message')}")
            return None
    except Exception as e:
        print(f"❌ שגיאה: {e}")
        return None

# דוגמת שימוש
place_order_safe("AAPL", "BUY", 10, "LMT", limitPrice=150.00)
```

### 3. ניטור חשבון עם alert:

```python
def monitor_account(adapter, min_buying_power=10000):
    account = adapter.get_account_summary()
    buying_power = float(account.get('BuyingPower', {}).get('value', 0))
    
    if buying_power < min_buying_power:
        print(f"⚠️ אזהרה: כוח קנייה נמוך! ${buying_power:.2f}")
        return False
    else:
        print(f"✅ כוח קנייה תקין: ${buying_power:.2f}")
        return True
```

---

## 📞 תמיכה ועזרה

### לוגים:
- **C# Bridge logs**: `csharp_bridge/logs/`
- **Python logs**: השתמש ב-`logging` module

### Debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
adapter = InterReactBridgeAdapter()
```

### נתונים נוספים:
- [InterReact Documentation](https://github.com/barneygale/InterReact)
- [IBKR API Documentation](https://interactivebrokers.github.io/tws-api/)

---

## 🎯 סיכום מהיר

1. **הרץ C# Bridge**: `cd csharp_bridge && dotnet run`
2. **ודא TWS/Gateway פועל** עם API מופעל
3. **חבר**: `POST /connect` עם host, port, clientId
4. **השתמש ב-API**: GET /account, GET /portfolio, POST /place-order
5. **Python Adapter**: `InterReactBridgeAdapter()` → פשוט וקל!

---

## ✅ Checklist להתחלה

- [ ] התקן .NET 8.0 SDK
- [ ] הפעל IBKR TWS/Gateway
- [ ] אפשר API Settings ב-TWS
- [ ] הרץ C# Bridge (`dotnet run`)
- [ ] בדוק health: `curl http://localhost:5080/health`
- [ ] חבר ל-IBKR: `POST /connect`
- [ ] בדוק חיבור: `GET /connection-status`
- [ ] התחל לעבוד! 🚀

---

**בהצלחה! 💪**

אם יש שאלות או בעיות - בדוק את ה-logs או פנה לתמיכה.
