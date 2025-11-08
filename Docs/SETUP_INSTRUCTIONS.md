# הוראות התקנה - מודול Training

**תאריך**: 8 נובמבר 2025

---

## 📦 התקנת Python Dependencies

### שלב 1: וודא שיש לך Python 3.9+

```bash
python --version
```

צריך להיות **Python 3.9** או יותר.

---

### שלב 2: התקנת כל התלויות

מתיקיית הפרויקט הראשית (`D:\YoopRL`):

```bash
pip install -r requirements.txt
```

---

### שלב 3: אימות התקנה

בדוק שהספריות החשובות הותקנו:

```bash
python -c "import stable_baselines3; print('SB3:', stable_baselines3.__version__)"
python -c "import gym; print('Gym:', gym.__version__)"
python -c "import gymnasium; print('Gymnasium:', gymnasium.__version__)"
python -c "import scipy; print('Scipy:', scipy.__version__)"
python -c "import schedule; print('Schedule:', schedule.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import flask; print('Flask:', flask.__version__)"
```

---

## 🔍 רשימת תלויות עיקריות

### Web Framework
- `flask==3.0.0` - Backend API server
- `flask-cors==4.0.0` - CORS support לחיבור Frontend

### Data Processing
- `numpy>=1.26.0` - חישובים מתמטיים
- `pandas>=2.1.0` - ניהול נתוני שוק
- `scipy>=1.11.0` - **חדש!** סטטיסטיקה לדיפ דטקשן (KS test, PSI)

### Reinforcement Learning
- `stable-baselines3==2.1.0` - **עיקרי!** PPO + SAC agents
- `gymnasium==0.29.1` - Gym API החדש (תואם SB3)
- `gym==0.26.2` - **חדש!** Gym API הישן (נדרש לקוד שלנו)

### Task Scheduling
- `schedule>=1.2.0` - **חדש!** תזמון אוטומטי לretraining

### Data Sources
- `yfinance>=0.2.28` - הורדת נתוני שוק
- `ta>=0.10.2` - אינדיקטורים טכניים
- `textblob>=0.17.0` - ניתוח סנטימנט
- `aiohttp>=3.8.0` - HTTP async לניוזים

### IBKR Integration
- `requests==2.31.0` - REST API calls

### Utilities
- `colorlog==6.7.0` - Logging צבעוני
- `python-dotenv>=1.0.0` - Environment variables

### Testing
- `pytest==7.4.3` - Unit tests
- `pytest-cov==4.1.0` - Coverage reports

---

## ⚠️ בעיות נפוצות והפתרונות

### בעיה 1: `gym` vs `gymnasium` conflicts

**תסמינים**: שגיאת import או version mismatch

**פתרון**:
```bash
pip uninstall gym gymnasium -y
pip install gym==0.26.2 gymnasium==0.29.1
```

---

### בעיה 2: `scipy` לא מותקנת

**תסמינים**: `ModuleNotFoundError: No module named 'scipy'`

**פתרון**:
```bash
pip install scipy>=1.11.0
```

**נדרש ל**: `backend/utils/state_normalizer.py` (KS test + PSI calculation)

---

### בעיה 3: `schedule` לא מותקנת

**תסמינים**: `ModuleNotFoundError: No module named 'schedule'`

**פתרון**:
```bash
pip install schedule>=1.2.0
```

**נדרש ל**: `backend/training/retraining_scheduler.py`

---

### בעיה 4: `stable-baselines3` לא עובדת

**תסמינים**: שגיאת import או PyTorch missing

**פתרון**: SB3 דורשת PyTorch (לא ברשימה כי זה נטען אוטומטית)

אם לא עובד:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3==2.1.0
```

---

### בעיה 5: PermissionError בהתקנה

**תסמינים**: `PermissionError: [WinError 5] Access is denied`

**פתרון**:
```bash
pip install --user -r requirements.txt
```

---

## ✅ בדיקת תקינות סופית

הרץ סקריפט בדיקה קצר:

```bash
cd D:\YoopRL\backend
python -c "
from environments.stock_env import StockTradingEnv
from agents.ppo_agent import PPOAgent
from utils.state_normalizer import StateNormalizer
from training.retraining_scheduler import RetrainingScheduler
print('✅ כל הספריות עובדות!')
"
```

אם אין שגיאות - **ההתקנה הצליחה!** 🎉

---

## 🚀 הפעלת המערכת

### Backend:
```bash
cd D:\YoopRL\backend
python api/main.py
```

### Frontend:
```bash
cd D:\YoopRL\frontend
npm start
```

---

## 📝 הערות חשובות

1. **Python 3.9-3.11** מומלץ (SB3 לא תומכת ב-3.12 עדיין)
2. **Virtual Environment** מומלץ אבל לא חובה:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **GPU** - לא נדרש! CPU מספיק למודלים שלנו
4. **Windows** - כל הקוד נבדק על Windows 11

---

## 🔗 קישורים שימושיים

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [Schedule Library](https://schedule.readthedocs.io/)
- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

**מוכן להתחיל לאמן! 🚀**
