# 🔑 API Keys & Configuration Guide

## סקירה כללית

חבילת הורדת הנתונים משתמשת במספר API חיצוניים לצורך הורדת נתונים, חדשות וסנטימנט.  
חלק מה-Keys כבר מוגדרים בקוד, אך מומלץ להשיג משלך לשימוש ארוך טווח.

---

## 📊 YahooFinance (yfinance)

**מה זה עושה**: מוריד נתוני OHLCV (Open, High, Low, Close, Volume) היסטוריים וחיים.

**API Key נדרש**: ❌ לא (חינמי לחלוטין)

**הגבלות**:
- נתונים עם דיליי של ~15 דקות (לא real-time)
- Rate limiting לא מפורסם, אך בדרך כלל מקבלים throttling אחרי ~2000 requests/hour
- לפעמים Yahoo משנה את ה-API והספרייה צריכה עדכון

**שימוש**:
```python
import yfinance as yf
df = yf.download("IWM", period="5y", progress=False)
```

**Troubleshooting**:
```bash
# אם יש בעיה, עדכן לגרסה אחרונה
pip install --upgrade yfinance
```

---

## ⏱️ Twelve Data (Intraday)

**מה זה עושה**: מושך נתוני intraday (למשל 15 דקות) עבור מניות/ETF, כולל נפח, לשימוש בתהליך SAC+Dsr.

**API Key נדרש**: ✅ כן

**איך להשיג Key**:
1. הירשם בחינם ב-https://twelvedata.com
2. העתק את ה-API Key מה- dashboard
3. הוסף אותו ל-`.env` או למשתני הסביבה:
   ```bash
   TWELVE_DATA_KEY=your_key_here
   ```

**שימוש בקוד**:
- קובץ `backend/data_download/intraday_loader.py` קורא את המפתח אוטומטית (`TWELVE_DATA_KEY` או `TWELVEDATA_API_KEY`).
- אין צורך לערוך קוד נוסף, רק לוודא שהמפתח מוגדר לפני הרצת התהליך.

**הגבלות (Free Tier)**:
- עד 8 קריאות לדקה ו-800 לקריאות ביום.
- מקסימום 5000 נקודות בכל קריאה (מספיק ליום מסחר אחד במרווח 15 דקות).

**Endpoint**:
```
GET https://api.twelvedata.com/time_series
  ?symbol=TQQQ
  &interval=15min
  &start_date=2020-02-03 09:30:00
  &end_date=2020-02-03 15:45:00
  &timezone=America/New_York
  &apikey=YOUR_KEY
```

**Response טיפוסי**:
```json
{
  "meta": { "symbol": "TQQQ", "interval": "15min", ... },
  "values": [
    {"datetime": "2020-02-03 09:30:00", "open": "100.5", "high": "101.2", "low": "99.8", "close": "100.9", "volume": "123456"},
    ...
  ]
}
```

**Troubleshooting**:
- אם מתקבלת הודעת שגיאה עם `status="error"`, בדרך כלל מדובר בחוסר במפתח או חריגה ממגבלת הקריאות.
- ודא שה-timezone מוגדר כ-`America/New_York` כדי למנוע הזחות זמן.

---

## 📰 Alpha Vantage (News Sentiment)

**מה זה עושה**: מספק חדשות כלכליות וסנטימנט מנותח מאמרים.

**API Key**: `MPLWQD0847NN6LEJ` (כבר מוגדר ב-`sentiment_service.py`)

**איך להשיג Key משלך**:
1. גש ל-https://www.alphavantage.co/support/#api-key
2. מלא email ותקבל Key מייד
3. החלף ב-`sentiment_service.py`:
   ```python
   self.alpha_vantage_key = "YOUR_NEW_KEY"
   ```

**הגבלות (Free Tier)**:
- 500 requests ליום
- 5 requests לדקה
- אם חורגים, מקבלים HTTP 429

**Endpoint**:
```
GET https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=IWM&apikey=YOUR_KEY&limit=50
```

**דוגמת Response**:
```json
{
  "feed": [
    {
      "title": "Market Update...",
      "ticker_sentiment": [
        {
          "ticker": "IWM",
          "ticker_sentiment_score": "0.152",
          "relevance_score": "0.8"
        }
      ]
    }
  ]
}
```

**מחיר (אם רוצים יותר)**:
- **Basic**: $49.99/month → 75 requests/minute
- **Pro**: $149.99/month → 150 requests/minute

---

## 📈 Finnhub (Company News & Sentiment)

**מה זה עושה**: חדשות חברות, buzz metrics, סנטימנט מנותח.

**API Key**: `d2k8n4pr01qs23a143l0d2k8n4pr01qs23a143lg` (כבר מוגדר ב-`sentiment_service.py`)

**איך להשיג Key משלך**:
1. גש ל-https://finnhub.io/register
2. הירשם (דורש email)
3. העתק את ה-API Key מה-dashboard
4. החלף ב-`sentiment_service.py`:
   ```python
   self.finnhub_key = "YOUR_NEW_KEY"
   ```

**הגבלות (Free Tier)**:
- 60 requests לדקה
- 30 calls/second
- אם חורגים, מקבלים HTTP 429

**Endpoint**:
```
GET https://finnhub.io/api/v1/news-sentiment?symbol=AAPL&token=YOUR_KEY
```

**דוגמת Response**:
```json
{
  "buzz": {
    "articlesInLastWeek": 120,
    "buzz": 0.85,
    "weeklyAverage": 98.5
  },
  "sentiment": {
    "score": 0.23,
    "bearishPercent": 0.25,
    "bullishPercent": 0.55
  }
}
```

**מחיר (אם רוצים יותר)**:
- **Starter**: $0 (Free)
- **Developer**: $59/month → 300 calls/minute
- **Pro**: $399/month → unlimited

---

## 🌐 Reddit API (r/wallstreetbets)

**מה זה עושה**: סורק פוסטים ב-r/wallstreetbets עם שם המניה ומנתח sentiment.

**API Key נדרש**: ❌ לא (משתמש ב-public JSON endpoint)

**הגבלות**:
- Reddit יכול לחסום אם שולחים יותר מ-60 requests/minute
- צריך User-Agent מתאים:
  ```python
  headers = {"User-Agent": "ppo-trading-sentiment/1.0"}
  ```

**Endpoint**:
```
GET https://www.reddit.com/r/wallstreetbets/search.json?q=IWM&sort=new&limit=10
```

**שימוש מתקדם (אם רוצים API רשמי)**:
1. צור Reddit App: https://www.reddit.com/prefs/apps
2. קבל `client_id` ו-`client_secret`
3. השתמש ב-`praw` library:
   ```python
   import praw
   reddit = praw.Reddit(client_id="...", client_secret="...", user_agent="...")
   ```

---

## 📊 StockTwits API

**מה זה עושה**: הודעות חברתיות על מניות מפלטפורמת StockTwits.

**API Key נדרש**: ❌ לא (endpoint פתוח)

**הגבלות**:
- 200 requests לשעה בלי אימות
- 400 requests לשעה עם אימות

**Endpoint**:
```
GET https://api.stocktwits.com/api/2/streams/symbol/IWM.json
```

**שימוש עם אימות**:
1. גש ל-https://api.stocktwits.com/developers/docs/authentication
2. צור Application ותקבל `access_token`
3. הוסף לheaders:
   ```python
   headers = {"Authorization": f"Bearer {access_token}"}
   ```

---

## 🔍 Google Trends (אינדיקציה לעניין)

**מה זה עושה**: בודק אם יש עניין בחיפוש Google למניה (proxy פשוט).

**API Key נדרש**: ❌ לא

**מימוש נוכחי**: פשוט שולח GET request ל-Google Search ובודק אם status=200.

**שדרוג אפשרי (pytrends)**:
```bash
pip install pytrends
```

```python
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360)
pytrends.build_payload(['IWM'], timeframe='now 7-d')
data = pytrends.interest_over_time()
```

---

## 📧 NewsAPI (אופציונלי - לא מיושם כרגע)

**מה זה עושה**: חדשות כלליות מכל העולם עם סינון לפי מילות מפתח.

**API Key**: אין כרגע (צריך להשיג)

**איך להשיג**:
1. גש ל-https://newsapi.org/register
2. הירשם עם email
3. קבל API Key
4. התקן:
   ```bash
   pip install newsapi-python
   ```
5. שימוש:
   ```python
   from newsapi import NewsApiClient
   newsapi = NewsApiClient(api_key='YOUR_KEY')
   articles = newsapi.get_everything(q='IWM stock', language='en', sort_by='publishedAt')
   ```

**הגבלות (Free Tier)**:
- 100 requests ליום
- נתונים עד 30 יום אחורה
- לא מקבלים content מלא (רק headlines)

**מחיר**:
- **Developer**: $0 (Free) → 100 requests/day
- **Business**: $449/month → 250,000 requests/day

---

## 🔐 ניהול Keys בצורה בטוחה

### אופציה 1: Environment Variables
```bash
# .env file
ALPHA_VANTAGE_KEY=MPLWQD0847NN6LEJ
FINNHUB_KEY=d2k8n4pr01qs23a143l0d2k8n4pr01qs23a143lg
NEWS_API_KEY=your_key_here
```

```python
from dotenv import load_dotenv
import os

load_dotenv()
alpha_key = os.getenv("ALPHA_VANTAGE_KEY")
finnhub_key = os.getenv("FINNHUB_KEY")
```

### אופציה 2: YAML Config
```yaml
# config/api_keys.yaml
apis:
  alpha_vantage: "MPLWQD0847NN6LEJ"
  finnhub: "d2k8n4pr01qs23a143l0d2k8n4pr01qs23a143lg"
  newsapi: null
```

```python
import yaml
with open("config/api_keys.yaml") as f:
    keys = yaml.safe_load(f)
    alpha_key = keys["apis"]["alpha_vantage"]
```

### אופציה 3: AWS Secrets Manager (Production)
```python
import boto3
client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='ppo-trading/api-keys')
keys = json.loads(secret['SecretString'])
```

---

## 📊 מעקב אחרי שימוש ב-API

### דוגמה: Rate Limiting Logger
```python
import time
from functools import wraps

def rate_limit_logger(max_per_minute=60):
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # נקה calls ישנים
            calls[:] = [t for t in calls if now - t < 60]
            
            if len(calls) >= max_per_minute:
                wait = 60 - (now - calls[0])
                print(f"Rate limit reached, waiting {wait:.1f}s...")
                time.sleep(wait)
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit_logger(max_per_minute=5)  # Alpha Vantage limit
def fetch_alpha_vantage(symbol):
    # ...
    pass
```

---

## 🚨 מה לעשות אם Key נחסם

### Alpha Vantage
1. בדוק שלא עברת 500 requests ליום
2. המתן ליום חדש (מתחדש UTC midnight)
3. או שדרג לתוכנית בתשלום

### Finnhub
1. בדוק שלא עברת 60 calls/minute
2. המתן דקה ונסה שוב
3. או שדרג ל-Developer plan

### Reddit
1. הוסף delay בין requests:
   ```python
   time.sleep(1.0)  # 1 שניה בין calls
   ```
2. שנה User-Agent
3. או השתמש ב-PRAW עם OAuth

---

## 📝 Checklist לפני Production

- [ ] השג API Keys משלך (אל תסתמך על הdefault)
- [ ] שמור Keys ב-environment variables (לא בקוד!)
- [ ] הוסף rate limiting logic
- [ ] הגדר monitoring על שימוש
- [ ] הכן fallback אם API נופל (cache ישן / default values)
- [ ] בדוק billing limits אם משתמש בתשלום
- [ ] תעד איפה כל Key נמצא בשימוש

---

## 🔗 קישורים מועילים

- **Alpha Vantage Docs**: https://www.alphavantage.co/documentation/
- **Finnhub Docs**: https://finnhub.io/docs/api
- **NewsAPI Docs**: https://newsapi.org/docs
- **yfinance GitHub**: https://github.com/ranaroussi/yfinance
- **Reddit API Docs**: https://www.reddit.com/dev/api
- **StockTwits API**: https://api.stocktwits.com/developers/docs

---

**עדכון אחרון**: 2025-11-08  
**גרסה**: 1.0
