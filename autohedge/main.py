import os
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from swarms import Agent, Conversation

load_dotenv()

# ── DeepSeek Configuration ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")

# Set environment variables so the swarms Agent picks up DeepSeek as an OpenAI-compatible provider
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
os.environ["OPENAI_API_BASE"] = DEEPSEEK_BASE_URL

# Swarms Agent model string for DeepSeek (OpenAI-compatible)
AGENT_MODEL = f"deepseek/{DEEPSEEK_MODEL}"


# ── Crypto Data Fetcher ─────────────────────────────────────────────────
CRYPTO_YFINANCE_MAP = {
    "BTC": "BTC-USD",
    "BTCUSDT": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHUSDT": "ETH-USD",
    "SOL": "SOL-USD",
    "SOLUSDT": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "MATIC": "MATIC-USD",
    "LINK": "LINK-USD",
}


def get_yf_ticker(symbol: str) -> str:
    """Convert a crypto symbol to its yfinance ticker."""
    sym = symbol.upper().replace("/", "").replace("-", "")
    if sym in CRYPTO_YFINANCE_MAP:
        return CRYPTO_YFINANCE_MAP[sym]
    if "-" in symbol:
        return symbol
    return f"{sym}-USD"


def fetch_crypto_data(symbol: str) -> Dict:
    """
    Fetch comprehensive crypto market data using yfinance.
    Returns a dict with price, volume, RSI, moving averages, etc.
    """
    yf_ticker = get_yf_ticker(symbol)
    logger.info(f"Fetching crypto data for {symbol} (yfinance: {yf_ticker})")

    try:
        ticker = yf.Ticker(yf_ticker)
        info = ticker.info

        history = ticker.history(period="1y")
        if history.empty:
            logger.warning(f"No historical data for {yf_ticker}")
            return {"symbol": symbol, "error": f"No data for {yf_ticker}"}

        current_price = history["Close"].iloc[-1]

        # RSI (14-day)
        delta = history["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        avg_gain = up.rolling(window=14, min_periods=14).mean()
        avg_loss = down.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

        # Moving averages
        ma_7 = float(history["Close"].rolling(window=7).mean().iloc[-1])
        ma_25 = float(history["Close"].rolling(window=25).mean().iloc[-1])
        ma_50 = float(history["Close"].rolling(window=50).mean().iloc[-1])
        ma_99 = float(history["Close"].rolling(window=99).mean().iloc[-1])
        ma_200 = float(history["Close"].rolling(window=200).mean().iloc[-1])

        # Bollinger Bands (20-day)
        bb_mid = history["Close"].rolling(window=20).mean()
        bb_std = history["Close"].rolling(window=20).std()
        bb_upper = float((bb_mid + 2 * bb_std).iloc[-1])
        bb_lower = float((bb_mid - 2 * bb_std).iloc[-1])

        # MACD
        ema_12 = history["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = history["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line

        # Price changes
        price_1d = float(history["Close"].pct_change(1).iloc[-1] * 100)
        price_7d = float(history["Close"].pct_change(7).iloc[-1] * 100) if len(history) > 7 else None
        price_30d = float(history["Close"].pct_change(30).iloc[-1] * 100) if len(history) > 30 else None
        price_90d = float(history["Close"].pct_change(90).iloc[-1] * 100) if len(history) > 90 else None

        # Volatility (30-day annualized)
        daily_returns = history["Close"].pct_change().dropna()
        volatility_30d = float(daily_returns.tail(30).std() * (365 ** 0.5) * 100)

        # Volume analysis
        avg_volume_30d = float(history["Volume"].tail(30).mean())
        current_volume = float(history["Volume"].iloc[-1])

        # ATH / ATL from available data
        ath = float(history["Close"].max())
        atl = float(history["Close"].min())
        pct_from_ath = float((current_price - ath) / ath * 100)

        # Trend determination
        short_trend = "Bullish" if ma_7 > ma_25 else "Bearish"
        mid_trend = "Bullish" if ma_50 > ma_99 else "Bearish"
        long_trend = "Bullish" if ma_50 > ma_200 else "Bearish"

        crypto_data = {
            "symbol": symbol,
            "pair": f"{symbol.upper().replace('USDT','').replace('-USD','')}/USDT",
            "current_price_usd": round(float(current_price), 4),
            "market_cap": info.get("marketCap", "N/A"),
            "volume_24h": current_volume,
            "avg_volume_30d": avg_volume_30d,
            "volume_ratio": round(current_volume / avg_volume_30d, 2) if avg_volume_30d > 0 else "N/A",
            "price_change_1d_pct": round(price_1d, 2),
            "price_change_7d_pct": round(price_7d, 2) if price_7d else "N/A",
            "price_change_30d_pct": round(price_30d, 2) if price_30d else "N/A",
            "price_change_90d_pct": round(price_90d, 2) if price_90d else "N/A",
            "rsi_14": round(current_rsi, 2) if current_rsi else "N/A",
            "ma_7": round(ma_7, 4),
            "ma_25": round(ma_25, 4),
            "ma_50": round(ma_50, 4),
            "ma_99": round(ma_99, 4),
            "ma_200": round(ma_200, 4),
            "bollinger_upper": round(bb_upper, 4),
            "bollinger_lower": round(bb_lower, 4),
            "macd": round(float(macd_line.iloc[-1]), 4),
            "macd_signal": round(float(signal_line.iloc[-1]), 4),
            "macd_histogram": round(float(macd_histogram.iloc[-1]), 4),
            "volatility_30d_annualized_pct": round(volatility_30d, 2),
            "short_term_trend": short_trend,
            "mid_term_trend": mid_trend,
            "long_term_trend": long_trend,
            "ath_1y": round(ath, 4),
            "atl_1y": round(atl, 4),
            "pct_from_ath": round(pct_from_ath, 2),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Successfully fetched data for {symbol}: ${current_price:.2f}")
        return crypto_data

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


def fetch_multi_crypto_data(symbols: List[str], workers: int = 5) -> Dict[str, Dict]:
    """Fetch data for multiple crypto symbols concurrently."""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_crypto_data, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                results[sym] = future.result()
            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")
                results[sym] = {"symbol": sym, "error": str(e)}
    return results


def format_crypto_data(data: Dict) -> str:
    """Format crypto data dict into a readable string for agents."""
    if "error" in data:
        return f"Error fetching data for {data.get('symbol', 'unknown')}: {data['error']}"

    lines = [
        f"{'='*50}",
        f" {data['pair']} Market Data",
        f"{'='*50}",
        f"Price: ${data['current_price_usd']:,.4f}",
        f"Market Cap: {data['market_cap']}",
        f"",
        f"-- Price Changes --",
        f"  1D:  {data['price_change_1d_pct']}%",
        f"  7D:  {data['price_change_7d_pct']}%",
        f"  30D: {data['price_change_30d_pct']}%",
        f"  90D: {data['price_change_90d_pct']}%",
        f"",
        f"-- Technical Indicators --",
        f"  RSI(14): {data['rsi_14']}",
        f"  MACD: {data['macd']} | Signal: {data['macd_signal']} | Hist: {data['macd_histogram']}",
        f"  Bollinger Upper: ${data['bollinger_upper']:,.4f}",
        f"  Bollinger Lower: ${data['bollinger_lower']:,.4f}",
        f"",
        f"-- Moving Averages --",
        f"  MA(7):   ${data['ma_7']:,.4f}",
        f"  MA(25):  ${data['ma_25']:,.4f}",
        f"  MA(50):  ${data['ma_50']:,.4f}",
        f"  MA(99):  ${data['ma_99']:,.4f}",
        f"  MA(200): ${data['ma_200']:,.4f}",
        f"",
        f"-- Trend Analysis --",
        f"  Short-term: {data['short_term_trend']}",
        f"  Mid-term:   {data['mid_term_trend']}",
        f"  Long-term:  {data['long_term_trend']}",
        f"",
        f"-- Volume & Volatility --",
        f"  24h Volume:     {data['volume_24h']:,.0f}",
        f"  30d Avg Volume: {data['avg_volume_30d']:,.0f}",
        f"  Volume Ratio:   {data['volume_ratio']}x",
        f"  30d Vol (Ann.): {data['volatility_30d_annualized_pct']}%",
        f"",
        f"-- Range (1Y) --",
        f"  ATH: ${data['ath_1y']:,.4f}",
        f"  ATL: ${data['atl_1y']:,.4f}",
        f"  From ATH: {data['pct_from_ath']}%",
    ]
    return "\n".join(lines)


# ── Agent Prompts ────────────────────────────────────────────────────────

DIRECTOR_PROMPT = """You are a Crypto Trading Director AI, responsible for orchestrating the cryptocurrency trading process.

Your primary objectives are:
1. Conduct in-depth crypto market analysis to identify opportunities and challenges.
2. Develop comprehensive trading theses for cryptocurrency pairs (priced in USDT).
3. Collaborate with specialized agents to ensure a cohesive strategy.
4. Make informed, data-driven decisions on trade executions.
5. Consider crypto-specific factors: on-chain metrics, sentiment, DeFi activity, regulatory news, whale movements, funding rates, and market dominance.

For each crypto asset under consideration, provide:
- A concise market thesis outlining the overall position and expected trends
- Key technical and on-chain factors influencing price
- A detailed risk assessment highlighting crypto-specific risks (regulation, smart contract risk, liquidity, etc.)
- Trade parameters: entry/exit points, position sizing, and risk management guidelines
- Timeframe: scalp, swing, or position trade recommendation
"""

QUANT_PROMPT = """You are a Crypto Quantitative Analysis AI, providing in-depth numerical analysis for cryptocurrency trading decisions.

Your primary objectives are:
1. **Technical Indicator Analysis**: Evaluate RSI, MACD, Bollinger Bands, moving averages (7/25/50/99/200), volume profiles.
2. **Statistical Pattern Evaluation**: Identify patterns including mean reversion, momentum, volatility cycles, and correlation with BTC dominance.
3. **Risk Metric Calculation**: Calculate VaR, max drawdown, Sharpe ratio, and crypto-specific risk metrics.
4. **Trade Success Probability**: Provide probability scores based on historical patterns and current technical setup.

Include confidence scores (0-1) for each aspect. Output structured analysis including:
- Technical score, volume score, trend strength, volatility assessment
- Key support and resistance levels
- Probability score for trade success
- Recommended timeframe and direction
"""

RISK_PROMPT = """You are a Crypto Risk Assessment AI. Evaluate and mitigate risks for cryptocurrency trades.

Your responsibilities include:
1. **Position Sizing**: Determine optimal position size considering crypto's higher volatility (recommend % of portfolio).
2. **Drawdown Analysis**: Calculate potential max drawdown based on historical volatility and current market conditions.
3. **Crypto-Specific Risks**: Assess smart contract risk, regulatory risk, liquidity risk, counterparty risk.
4. **Correlation Risk**: Evaluate correlation with BTC, overall market beta, and macro correlation.
5. **Portfolio Impact**: Assess how this trade affects overall portfolio risk and diversification.

Provide clear risk metrics:
- Recommended position size (% of portfolio and USDT amount)
- Maximum drawdown risk
- Stop-loss levels
- Risk/Reward ratio
- Overall risk score (1-10, 10 being highest risk)
"""

EXECUTION_PROMPT = """You are a Crypto Trade Execution AI. Generate precise trade execution details for cryptocurrency pairs in USDT.

Output a structured trade order including:
- Trading pair (e.g., BTC/USDT)
- Direction: LONG or SHORT
- Order type (market/limit/stop-limit)
- Entry price or price range
- Position size in USDT and crypto amount
- Stop loss (price and % from entry)
- Take profit targets (TP1, TP2, TP3) with % allocation
- Risk/Reward ratio
- Leverage recommendation (if applicable)
- Time in force
- Special instructions (DCA, trailing stop, etc.)
"""


# ── Pydantic Output Models ──────────────────────────────────────────────

class AutoHedgeOutput(BaseModel):
    id: str = uuid.uuid4().hex
    thesis: Optional[str] = None
    risk_assessment: Optional[str] = None
    order: Optional[str] = None
    decision: Optional[str] = None
    timestamp: str = datetime.now().isoformat()
    current_asset: str


class AutoHedgeOutputMain(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    id: str = uuid.uuid4().hex
    assets: Optional[list] = None
    task: Optional[str] = None
    timestamp: str = datetime.now().isoformat()
    logs: List[AutoHedgeOutput] = None


# ── Agent Classes ────────────────────────────────────────────────────────

def create_deepseek_agent(name: str, prompt: str) -> Agent:
    """Create a Swarms Agent configured to use DeepSeek API."""
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=AGENT_MODEL,
        output_type="str",
        max_loops=1,
        verbose=True,
        context_length=16000,
    )


class RiskManager:
    def __init__(self):
        self.risk_agent = create_deepseek_agent("Risk-Manager", RISK_PROMPT)
        self.agent_name = "Risk-Manager"

    def assess_risk(self, asset: str, thesis: str, quant_analysis: str) -> str:
        prompt = f"""
        Crypto Asset: {asset}
        Trading Thesis: {thesis}
        Quantitative Analysis: {quant_analysis}
        
        Provide comprehensive risk assessment including:
        1. Recommended position size (% of portfolio and USDT amount)
        2. Maximum drawdown risk
        3. Stop-loss levels
        4. Risk/Reward ratio
        5. Crypto-specific risk factors
        6. Overall risk score (1-10)
        """
        return self.risk_agent.run(prompt)


class ExecutionAgent:
    def __init__(self):
        self.execution_agent = create_deepseek_agent("Execution-Agent", EXECUTION_PROMPT)
        self.agent_name = "Execution-Agent"

    def generate_order(self, asset: str, thesis: str, risk_assessment: str) -> str:
        prompt = f"""
        Crypto Asset: {asset}
        Trading Thesis: {thesis}
        Risk Assessment: {risk_assessment}
        
        Generate a detailed trade order for {asset}/USDT including:
        1. Direction (LONG/SHORT)
        2. Order type (market/limit/stop-limit)
        3. Entry price or range
        4. Position size in USDT
        5. Stop loss (price and %)
        6. Take profit targets (TP1, TP2, TP3)
        7. Risk/Reward ratio
        8. Leverage recommendation
        9. Time in force
        10. DCA strategy if applicable
        """
        return self.execution_agent.run(prompt)


class QuantAnalyst:
    """Quantitative Analysis Agent for crypto markets."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.quant_agent = create_deepseek_agent("Quant-Analyst", QUANT_PROMPT)
        self.agent_name = "Quant-Analyst"
        logger.info("Initialized Crypto Quant Analyst (DeepSeek)")

    def analyze(self, asset: str, market_data: str, thesis: str) -> str:
        logger.info(f"Performing quant analysis for {asset}")
        prompt = f"""
        Crypto Asset: {asset}
        Market Data:
        {market_data}
        
        Director's Thesis: {thesis}
        
        Generate comprehensive quantitative analysis including:
        - Technical score (0-1)
        - Volume score (0-1) 
        - Trend strength (0-1)
        - Volatility assessment
        - Probability score for trade success (0-1)
        - Key support and resistance levels in USDT
        - MACD / RSI / Bollinger Band interpretation
        - Recommended direction and timeframe
        """
        return self.quant_agent.run(prompt)


class TradingDirector:
    """Trading Director Agent for crypto markets."""

    def __init__(self, assets: List[str], output_dir: str = "outputs"):
        logger.info("Initializing Crypto Trading Director (DeepSeek)")
        self.director_agent = create_deepseek_agent("Trading-Director", DIRECTOR_PROMPT)
        self.agent_name = "Trading-Director"

    def generate_thesis(self, task: str, asset: str) -> tuple:
        """Generate trading thesis for a crypto asset with live market data."""
        logger.info(f"Generating thesis for {asset}")

        # Fetch live market data
        market_data_raw = fetch_crypto_data(asset)
        market_data_str = format_crypto_data(market_data_raw)

        prompt = f"""
        Task: {task}
        
        Crypto Asset: {asset}/USDT
        
        Live Market Data:
        {market_data_str}
        
        Generate a comprehensive trading thesis for {asset}/USDT based on the above data.
        Include your conviction level, recommended direction, timeframe, and key catalysts.
        """

        thesis = self.director_agent.run(prompt)
        return thesis, market_data_str

    def make_decision(self, order_details: str, thesis: str) -> str:
        return self.director_agent.run(
            f"Based on the thesis: {thesis}\n\n"
            f"And the proposed order + risk assessment:\n{order_details}\n\n"
            f"Should we execute this trade? Provide a clear GO / NO-GO decision with reasoning."
        )


class AutoHedge:
    """
    Main crypto trading system that coordinates all agents.
    Analyzes cryptocurrency pairs against USDT using DeepSeek AI.
    """

    def __init__(
        self,
        assets: List[str] = None,
        stocks: List[str] = None,  # backward compat alias
        name: str = "autohedge-crypto",
        description: str = "AI-powered crypto analysis system",
        output_dir: str = "outputs",
        output_type: str = "str",
        **kwargs,
    ):
        # Support both 'assets' and 'stocks' parameter names
        self.assets = assets or stocks or []
        self.name = name
        self.description = description
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.output_type = output_type

        logger.info(f"Initializing AutoHedge Crypto System")
        logger.info(f"Assets: {self.assets}")
        logger.info(f"Model: DeepSeek ({DEEPSEEK_MODEL})")

        self.director = TradingDirector(self.assets, output_dir)
        self.quant = QuantAnalyst(output_dir)
        self.risk = RiskManager()
        self.execution = ExecutionAgent()
        self.conversation = Conversation(time_enabled=True)

    def run(self, task: str, *args, **kwargs):
        """Execute a complete analysis cycle for all crypto assets."""
        logger.info("Starting crypto analysis cycle")
        self.conversation.add(role="user", content=f"Task: {task}")

        try:
            for asset in self.assets:
                logger.info(f"{'='*50}")
                logger.info(f" Processing {asset}/USDT")
                logger.info(f"{'='*50}")

                # Step 1: Generate thesis with live market data
                logger.info(f"[1/5] Generating thesis for {asset}...")
                thesis, market_data = self.director.generate_thesis(
                    task=task, asset=asset
                )
                self.conversation.add(
                    role="Trading-Director",
                    content=f"Asset: {asset}/USDT\n\nMarket Data:\n{market_data}\n\nThesis:\n{thesis}",
                )

                # Step 2: Quantitative analysis
                logger.info(f"[2/5] Running quant analysis for {asset}...")
                analysis = self.quant.analyze(asset, market_data, thesis)
                self.conversation.add(role="Quant-Analyst", content=analysis)

                # Step 3: Risk assessment
                logger.info(f"[3/5] Assessing risk for {asset}...")
                risk_assessment = self.risk.assess_risk(asset, thesis, analysis)
                self.conversation.add(role="Risk-Manager", content=risk_assessment)

                # Step 4: Generate execution order
                logger.info(f"[4/5] Generating trade order for {asset}...")
                order = self.execution.generate_order(asset, thesis, risk_assessment)
                self.conversation.add(role="Execution-Agent", content=order)

                # Step 5: Final decision
                logger.info(f"[5/5] Making final decision for {asset}...")
                decision = self.director.make_decision(
                    f"{order}\n\nRisk Assessment:\n{risk_assessment}\n\nMarket Data:\n{market_data}",
                    thesis,
                )
                self.conversation.add(
                    role="Trading-Director",
                    content=f"DECISION for {asset}/USDT:\n{decision}",
                )

                logger.info(f"Completed analysis for {asset}/USDT")

            logger.info("All assets analyzed")

            if self.output_type == "list":
                return self.conversation.return_messages_as_list()
            elif self.output_type == "dict":
                return self.conversation.return_messages_as_dictionary()
            elif self.output_type == "str":
                return self.conversation.return_history_as_string()
            else:
                return self.conversation.return_history_as_string()

        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            raise
