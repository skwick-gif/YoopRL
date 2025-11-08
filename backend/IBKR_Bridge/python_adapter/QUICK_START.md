# Python Adapter for IBKR Bridge

## Quick Start

### Installation

```bash
# Install dependencies
pip install requests PyQt6
```

### Basic Usage

```python
from python_adapter.interreact_bridge_adapter import InterReactBridgeAdapter

# Create adapter
adapter = InterReactBridgeAdapter(host="localhost", port=5080)

# Check connection
if adapter.is_connected():
    # Get account data
    account = adapter.get_account_summary()
    print(f"Net Liquidation: {account['NetLiquidation']['value']}")
    
    # Get positions
    portfolio = adapter.get_portfolio()
    for pos in portfolio:
        print(f"{pos['symbol']}: {pos['position']} shares")
```

### PyQt6 Integration

```python
from PyQt6.QtWidgets import QApplication, QMainWindow
from python_adapter.interreact_bridge_adapter import InterReactBridgeAdapter

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.adapter = InterReactBridgeAdapter()
        self.adapter.connection_status_changed.connect(self.on_connection)
        self.adapter.start_monitoring(interval_ms=5000)
    
    def on_connection(self, connected: bool):
        print(f"Connection: {'Connected' if connected else 'Disconnected'}")

if __name__ == "__main__":
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec()
```

---

For full documentation and examples, see:
- **README.md** - Main documentation
- **EXAMPLES.md** - Code examples
- **CONFIGURATION.md** - Advanced settings
