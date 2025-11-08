# 📝 CHANGELOG - IBKR Bridge Package

---

## [1.0.0] - 2025-01-19

### ✅ Initial Release

#### C# Bridge (InterReactBridge)
- **REST API** עם ASP.NET Core
  - `/health` - בדיקת בריאות
  - `/connect` - חיבור ל-IBKR TWS/Gateway
  - `/disconnect` - ניתוק
  - `/connection-status` - סטטוס חיבור
  - `/account` - נתוני חשבון מפורטים
  - `/portfolio` - פוזיציות פתוחות
  - `/place-order` - ביצוע פקודות

- **SignalR Hubs** לעדכונים בזמן אמת:
  - `AccountHub` - עדכוני חשבון
  - `PortfolioHub` - עדכוני פוזיציות
  - `MarketDataHub` - נתוני שוק

- **תכונות**:
  - תמיכה ב-TWS ו-Gateway
  - Reconnect אוטומטי
  - Swagger UI לתיעוד API
  - Logging מפורט

#### Python Adapter (interreact_bridge_adapter.py)
- **InterReactBridgeAdapter** class:
  - `connect()` - חיבור ל-IBKR
  - `disconnect()` - ניתוק
  - `is_connected()` - בדיקת חיבור
  - `get_account_summary()` - נתוני חשבון
  - `get_portfolio()` - פוזיציות
  - `get_cached_*()` - גישה ל-cache
  
- **PyQt6 Integration**:
  - QObject-based עם Signals
  - `connection_status_changed` signal
  - `account_updated` signal
  - `portfolio_updated` signal
  - QTimer-based monitoring

- **Performance Optimizations**:
  - HTTP timeout מותאם (0.5-2.0s)
  - Optional caching
  - Configurable polling interval

#### תיעוד
- **README.md** - מדריך מקיף
- **CONFIGURATION.md** - הגדרות מפורטות
- **TROUBLESHOOTING.md** - פתרון בעיות
- **EXAMPLES.md** - דוגמאות קוד
- **requirements.txt** - תלויות Python

---

## Known Issues

### ⚠️ Current Limitations

1. **Order Types**: 
   - תמיכה מלאה: MKT, LMT, STP, STP LMT
   - לא נתמך: Bracket orders (צריך הרחבה)

2. **Market Data**:
   - עדכוני מחיר דורשים מנוי IBKR
   - Delayed data זמין ב-Paper Trading

3. **Multi-Account**:
   - תמיכה בסיסית בלבד
   - לא נבדק עם Financial Advisor accounts

### 🔧 Future Enhancements

#### גרסה 1.1 (מתוכנן):
- [ ] Bracket orders support
- [ ] Advanced order types (OCO, OCA)
- [ ] Order status tracking
- [ ] Historical data endpoints
- [ ] Options trading support

#### גרסה 2.0 (עתידי):
- [ ] Docker containerization
- [ ] Authentication & Authorization
- [ ] Multi-user support
- [ ] Database logging
- [ ] Web dashboard

---

## Performance Improvements

### מתוך התפתחות הפרויקט:

**v0.9 → v1.0**:
1. **HTTP Timeout**: הקטנה מ-2s ל-0.5s
   - שיפור של 4x בזמני תגובה
   - UI responsive יותר

2. **Polling Interval**: הגדלה מ-1s ל-10s (Dashboard)
   - הפחתה של 90% ב-overhead
   - CPU usage נמוך יותר

3. **Connection Check**: הסרת בדיקה מיידית ב-startup
   - הפעלה מהירה יותר של האפליקציה
   - פחות load על ה-Bridge

4. **Architecture**: מעבר מ-async threads ל-sync simple
   - פחות complexity
   - פחות bugs (thread deletion errors)
   - קל יותר ל-debug

---

## Migration Guide

### מגרסה קודמת (Internal Project)

אם השתמשת בגרסה הפנימית של הפרויקט:

1. **Import Path השתנה**:
   ```python
   # ישן
   from src.services.interreact_bridge_adapter import InterReactBridgeAdapter
   
   # חדש
   from python_adapter.interreact_bridge_adapter import InterReactBridgeAdapter
   ```

2. **Configuration**:
   - עכשיו `appsettings.json` במקום environment variables
   - תיעוד מלא ב-CONFIGURATION.md

3. **Dependencies**:
   - התקן: `pip install -r requirements.txt`

---

## Credits

### Built With

- **C#**: .NET 8.0
- **Libraries**:
  - ASP.NET Core (Web API)
  - SignalR (Real-time communication)
  - InterReact (IBKR TWS wrapper)

- **Python**: 3.10+
- **Libraries**:
  - requests (HTTP client)
  - PyQt6 (GUI framework)
  - signalrcore (SignalR client)

### Special Thanks

- **IBKR TWS API** - Interactive Brokers
- **InterReact Library** - C# wrapper for TWS API
- **PyQt Project** - Qt bindings for Python

---

## License

This package is provided as-is for educational and personal use.

**Disclaimer**: Trading involves risk. Use at your own risk.

---

## Support

- 📖 **Documentation**: ראה README.md
- 🐛 **Issues**: צור issue עם פרטים מלאים
- 💡 **Features**: הצעות לשיפורים תמיד מתקבלות בברכה

---

## Version History

| גרסה | תאריך | עיקרי השינויים |
|------|-------|----------------|
| 1.0.0 | 2025-01-19 | Initial packaged release |
| 0.9 | 2025-01-15 | Performance optimizations |
| 0.5 | 2025-01-10 | Internal project version |

---

**עודכן לאחרונה**: 2025-01-19
