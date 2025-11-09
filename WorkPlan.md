# תוכנית עבודה מלאה: פיתוח סוכן RL למסחר ב-ETFs ממונפים (TNA, TQQQ, UPRO)

## 1. הצהרת משימה

בניית סוכן למידת חיזוק (RL Agent) אוטונומי המתמחה במסחר **תוך-יומי בלבד** ב-ETFs ממונפים. מטרת הסוכן היא מקסום **רווח מותאם-סיכון** (Risk-Adjusted Return), תוך הימנעות מוחלטת מחשיפה לסיכוני לילה ושחיקת זמן (Time Decay).

---

## 2. ארכיטקטורת המערכת

### 2.1. מודל (Algorithm)
* **בחירה:** **SAC (Soft Actor-Critic)**
* **נימוק:**
    1.  **מרחב פעולה רציף:** מאפשר לסוכן לבחור *כמה* להשקיע (למשל, 30% או 70% מההון).
    2.  **יעילות נתונים (Sample Efficiency):** מודל Off-Policy המשתמש ב-Replay Buffer ללמידה מהירה יותר.
    3.  **אקספלורציה (Entropy):** אופטימיזציית אנטרופיה מובנית למניעת היתקעות בפתרון פשוט.

### 2.2. סביבה (Environment)
* **יחידת זמן (Timeframe):** **נרות של 15 דקות** (intraday).
* **מימוש:** הרחבה של הסביבה הקיימת ל-`IntradayEquityEnv` תואם Gym (API של `gym==0.26.2`), עם wrapper נפרד ל-DSR.
* **כללי סביבה (חובה):**
    1.  **סגירה כפויה:** כל הפוזיציות נסגרות אוטומטית ב-15:45 (שעון ניו יורק) והסביבה מחזירה סטפ סיום.
    2.  **עלויות מסחר:** יש לדמות **עמלות**, **slippage** ועמלות מימון (אם רלוונטי) בכל פעולה.
    3.  **איפוס יומי:** כל אפיזודה מייצגת יום מסחר יחיד; איפוס הסביבה מחזיר נתוני 15 דקות ליום חדש.
    4.  **תאימות נתונים:** בדיקה שהצוהר intraday לא חופף לסופי שבוע/חגים, והתייחסות לנתוני half-day (אם קיימים).

---

## 3. הנדסת פיצ'רים (Feature Engineering) - ה-State Vector

ה-State Vector שיוזן לסוכן בכל צעד זמן (כל 15 דקות) יורכב משלוש קטגוריות נתונים, מסונכרנות לאותה נקודת זמן:

### 3.1. פיצ'רי "אות" (מנכס הבסיס - QQQ/IWM)
* `Base_Trend_Context`: `(QQQ_Price - QQQ_EMA(50)) / QQQ_ATR(14)`
* `Base_Momentum`: `QQQ_MACD_Histogram(12, 26, 9)`
* `Base_Trend_Strength`: `QQQ_ADX(14)`
* `Base_Extremes`: `QQQ_RSI(14)`

### 3.2. פיצ'רי "רכב" (מה-ETF הממונף - TQQQ/TNA)
* `Leveraged_Volatility`: `TQQQ_ATR(14) / TQQQ_Price`
* `Leveraged_Momentum_Short`: `(TQQQ_Price - TQQQ_EMA(10)) / TQQQ_ATR(14)`

### 3.3. פיצ'רי "הקשר" (כללי)
* `Time_Context`: `Time_of_Day` (מנורמל 0.0 עד 1.0, מפתיחה עד סגירה).
* `Position_Context`: `Current_Position_Size` (ערך רציף בין -1.0 ל-+1.0).

---

## 4. פונקציית תגמול (Reward Function)

נשתמש ב-**Differential Sharpe Ratio (DSR)** כתגמול "צפוף" (Dense Reward) בכל צעד זמן, באמצעות wrapper בשם `DSRRewardWrapper` שיושב מעל הסביבה הבסיסית. התגמול מודד את השינוי הרגעי ביחס שארפ, ומנחה את הסוכן למקסם תשואה מותאמת-סיכון תוך שמירה על נתוני ה-PnL ב-`info`.

### 4.1. חישוב תשואה (Return)
בכל צעד `t`, נחשב תחילה את התשואה נטו (Net Return) של התיק, בניכוי עלויות:
* `pnl_t = (current_portfolio_value_t - last_portfolio_value_t-1)`
* `cost_t = (commission + slippage)` (רק אם בוצעה פעולה)
* `net_return_t = (pnl_t - cost_t) / last_portfolio_value_t-1`

### 4.2. חישוב DSR
ה-DSR מסתמך על עדכון רקורסיבי של ממוצע התשואות (A) וממוצע ריבועי התשואות (B) באמצעות מקדם דעיכה (`eta`, למשל `1e-4`). יש לטפל ביציבות נומרית (מניעת חלוקה באפס עם `epsilon=1e-9`) ובתקופת "חימום" (`warmup_steps=200`) שבה התגמול יהיה `net_return_t` בלבד.

---

## 5. תוכנית פיתוח, אימון ואימות

### 5.1. שלב 1: איסוף ועיבוד נתונים
1.  השגת נתונים היסטוריים (נרות 15 דקות) עבור הנכסים הממונפים (TQQQ, TNA, UPRO) ונכסי הבסיס (QQQ, IWM, SPX) לתקופה של 5+ שנים.
2.  יצירת מחסן נתונים ייעודי: `data/intraday/<symbol>/15m/YYYY-MM-DD.csv` עם חותמות זמן TZ-aware (America/New_York).
3.  סינכרון מלא של הנתונים לפי חותמת זמן (inner join על ציר הזמן, טיפול ב-gap days ובחגים).
4.  חישוב *כל* הפיצ'רים (מקטע 3) עבור כל נר והכנסת metadata של session (today_index, bar_index).
5.  בדיקות איכות: וולידציה של נתונים חסרים, בדיקת סטיות בין מקורות, השוואה מול ימי מבחן ידועים.
6.  **משימה עתידית:** הטמעת pipeline אנכית ל-SQLite (טבלאות intraday) עבור נתונים היסטוריים וחדשים – יתבצע לאחר שה-flow הראשוני מושלם.

### 5.2. שלב 2: בניית הסביבה (Gym-Compatible Environment)
1.  הרחבת הבסיס הקיים ל-`IntradayEquityEnv` (10:00-16:00 ET) עם פרמטרים ל-slippage, commission, forced_exit.
2.  מימוש `reset()` שמבצע draw של יום אקראי (או walk-forward window), מאפס את סטטיסטיקות ה-DSR ומחזיר `observation, info` בפורמט Gym.
3.  מימוש `step(action)`:
    * תרגום האקשן הרציף לגודל פוזיציה (כולל clipping ואזור deadband).
    * חישוב עלויות עסקה (`cost_t`), תשואה נקייה (`net_return_t`) ועדכון equity curve.
    * קריאת wrapper `DSRRewardWrapper` כדי לעדכן A/B ולהפיק תגמול; בזמן Warm-up התגמול יהיה `net_return_t` בלבד.
    * אכיפת כלל "סגירה כפויה" ב-15:45 והחזרת `terminated=True`.
4.  בדיקות יחידה: אימות של forced_exit, בדיקות off-by-one, בדיקה שהסביבה תואמת Optuna/test harness הקיים.

### 5.3. שלב 3: אימון וכוונון היפר-פרמטרים
1.  **חומרה:** יש להשתמש ב-GPU (NVIDIA CUDA) לתהליך האימון.
2.  **היפר-פרמטרים התחלתיים (Baseline):**
    * `policy`: `'MlpPolicy'`
    * `learning_rate`: `1e-4`
    * `buffer_size`: `200000`
    * `batch_size`: `256`
    * `ent_coef`: `'auto'`
    * `gamma`: `0.99`
    * `tau`: `0.005`
    * `learning_starts`: `10000`
    * `policy_kwargs`: `dict(net_arch=[128, 128])`
3.  **תהליך אימון:** הרצת `model.learn()` על סביבת האימון למספר גדול של צעדי זמן (למשל, 1,000,000 צעדים) תוך ניטור DSR, PnL ורמת פעילות.
4.  **אופטימיזציה (Tuning):** שימוש ב-Optuna עם guardrails (מינימום טריידים, מגבלת drawdown) והערכה בבדיקות walk-forward.
5.  **קצבי רענון (Cadence):**
    * **עדכון נתונים:** משיכת נתוני 15 דקות פעם ביום בסיום המסחר.
    * **Re-training:** אחת לשבוע/חודש (תלוי ב-variance) או בעת ירידה במדדי live performance.
    * **Backtest regression:** הרצה אוטומטית של סט בדיקות קצר על טווחי זמן מייצגים לאחר כל מודל חדש.

### 5.4. שלב 4: אימות ובדיקה (Backtesting)
לאחר האימון, חובה לבצע אימות קפדני על נתונים שהמודל מעולם לא ראה.

1.  **שיטה 1: חלוקה פשוטה (Train/Test Split)**
    * **אימון:** 2018-2022
    * **מבחן (Out-of-Sample):** 2023-2024
2.  **שיטה 2: אימות צף (Walk-Forward Validation) - מומלץ**
    * **הרצה 1:** אימון על 2018-2020 ➡️ בדיקה על 2021.
    * **הרצה 2:** אימון על 2018-2021 ➡️ בדיקה על 2022.
    * **הרצה 3:** אימון על 2018-2022 ➡️ בדיקה על 2023.
    * **ניתוח:** בחינת הביצועים המצטברים מכל תקופות המבחן.
3.  **מדדי ביצועים (Metrics):**
    * גרף עקומת הון (Equity Curve) ו-DSR לאורך זמן.
    * יחס שארפ (Sharpe Ratio) ויחס סורטינו (Sortino Ratio).
    * ירידת הון מקסימלית (Max Drawdown) ו-Max Intraday Drawdown.
    * ממוצע טריידים ליום, שיעור הצלחה ועוצמת פעילות (Position Turnover).
    * רווח/הפסד שנתי ממוצע, ופיזור תשואות חודשי.

### 5.5. שלב 5: פריסה ל-Live (Deployment) - עתידי
1.  **ארכיטקטורה:** בניית מערכת (Worker) שתחבר את המודל המאומן ל-API של ברוקר (כמו Alpaca).
2.  **הזנת נתונים:** חיבור ל-Data Feed בזמן אמת (Websockets) לקבלת נרות 15 דקות.
3.  **ניהול סיכונים:** מימוש מנגנוני "Mute" או "Kill-Switch" למקרה של התנהגות חריגה או הפסדים גבוהים.
4.  **טיפול בפער מהמציאות:** ניטור ביצועים חיים מול ה-Backtest לזיהוי פערים (ב-Slippage, Latency וכו').
5.  **הקשחת guardrails:** קביעת טריגרים להפסקת מסחר (DSR שלילי רצוף, Drawdown יומי מעל סף, חריגות במספר הטריידים).

---

## 6. אינטגרציה במערכות קיימות

### 6.1. קונפיגורציה ו-Backend
1.  הוספת פרופיל `SAC_INTRADAY_DSR` ב-`backend/config/training_config.py` (בחירת Env חדשה, נתיב נתונים intraday, `reward_mode="dsr"`).
2.  התאמת `train_agent` ו-optuna pipeline לתמוך בפרופיל החדש תוך שמירה על קשר לאחור.
3.  בדיקת REST API ב-`backend/api/main.py` כדי לוודא שהקריאה החדשה יושבת לצד הפרופילים הקיימים.

### 6.2. Frontend ו-UX
1.  הרחבת `TabTraining.jsx` ו-`useTrainingState.js` עם אפשרות בחירה "SAC + DSR (15m Intraday)".
2.  הצגת סט פרמטרים ייחודי (למשל horizon, ניהול סיכונים) ב-UI, תוך הסתרת שדות שאינם רלוונטיים.
3.  וידוא שהגרפים/לוגים באתר תומכים במדדי DSR ונתוני intraday.

### 6.3. בדיקות ותיעוד
1.  כתיבת בדיקות יחידה ל-`IntradayEquityEnv`, ל-`DSRRewardWrapper`, ולמנגנון forced_exit.
2.  כתיבת בדיקות אינטגרציה לאימון קצר (smoke run) והזנת נתונים מלאכותיים.
3.  עדכון `Docs/SETUP_INSTRUCTIONS.md` ו-`TRAINING_IMPLEMENTATION_PLAN.md` עם התוספות החדשות.