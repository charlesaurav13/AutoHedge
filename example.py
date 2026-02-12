from dotenv import load_dotenv
from autohedge.main import AutoHedge

load_dotenv()


# Define the crypto assets to analyze (in USDT pairs)
cryptos = ["BTC", "SOL", "ETH"]

# Initialize the trading system
trading_system = AutoHedge(
    name="crypto-fund",
    description="AI-Powered Crypto Analysis Fund (DeepSeek)",
    assets=cryptos,
    output_type="str",  # "str", "list", or "dict"
)

# Define the analysis task
task = (
    "Analyze BTC/USDT, SOL/USDT, and ETH/USDT for a portfolio with $50,000 USDT allocation. "
    "Provide trading recommendations with entry/exit points, position sizing, "
    "and risk management for each asset. Consider current market conditions, "
    "technical indicators, and crypto-specific risks."
)

# Run the analysis and print results
print(trading_system.run(task=task))
