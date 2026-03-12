from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict

from langgraph.graph import StateGraph
from loguru import logger

from ai_trader.agents.chart_agent import ChartAgent
from ai_trader.agents.option_chain_agent import OptionChainAgent
from ai_trader.agents.news_agent import NewsAgent
from ai_trader.agents.volatility_agent import VolatilityAgent
from ai_trader.agents.regime_agent import RegimeAgent
from ai_trader.agents.liquidity_agent import LiquidityAgent
from ai_trader.agents.trigger_agent import TradeTriggerAgent, TradeSignal
from ai_trader.agents.risk_agent import RiskManagerAgent, RiskCheckResult


@dataclass
class OrchestratorOutput:
    signal: TradeSignal
    risk: RiskCheckResult
    decision_score: int
    state: Dict[str, Any]


class DecisionEngine:
    """Orchestrates all agents via a simple LangGraph."""

    def __init__(self) -> None:
        self.chart_agent = ChartAgent()
        self.option_agent = OptionChainAgent()
        self.news_agent = NewsAgent()
        self.vol_agent = VolatilityAgent()
        self.regime_agent = RegimeAgent()
        self.liquidity_agent = LiquidityAgent()
        self.trigger_agent = TradeTriggerAgent()
        self.risk_agent = RiskManagerAgent()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        # State is a simple dict passed between nodes
        graph = StateGraph(dict)

        def chart_node(state: Dict[str, Any]) -> Dict[str, Any]:
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    "chart": executor.submit(self.chart_agent.analyze),
                    "option_chain": executor.submit(self.option_agent.analyze),
                    "news": executor.submit(self.news_agent.analyze),
                    "vol": executor.submit(self.vol_agent.analyze),
                    "regime": executor.submit(self.regime_agent.analyze),
                    "liquidity": executor.submit(self.liquidity_agent.analyze),
                }
                for key, future in futures.items():
                    state[key] = future.result()
            return state

        def trigger_node(state: Dict[str, Any]) -> Dict[str, Any]:
            chart = state["chart"]
            option_chain = state["option_chain"]
            news = state["news"]
            vol = state["vol"]
            regime = state["regime"]
            liq = state["liquidity"]

            signal = self.trigger_agent.generate_signal(
                chart=chart,
                option_chain=option_chain,
                news=news,
                vol=vol,
                regime=regime,
                liquidity=liq,
            )
            state["signal"] = signal

            # Compute decision score using spec weights
            score = 0
            if chart.trend in ("bullish", "bearish"):
                score += 2
            if option_chain.bias in ("bullish", "bearish"):
                score += 2
            if news.macro_bias in ("bullish", "bearish"):
                score += 1
            if regime.regime in ("trend_up", "trend_down"):
                score += 2
            if liq.liquidity != "low":
                score += 1

            state["decision_score"] = score
            return state

        graph.add_node("gather", chart_node)
        graph.add_node("trigger", trigger_node)

        graph.set_entry_point("gather")
        graph.add_edge("gather", "trigger")

        return graph.compile()

    def run_once(self) -> OrchestratorOutput:
        state: Dict[str, Any] = {}
        state = self.graph.invoke(state)  # type: ignore[assignment]

        signal: TradeSignal = state["signal"]
        decision_score: int = state["decision_score"]

        # Apply orchestrator rule: require score >= 5
        if decision_score < 5:
            logger.info(f"Decision score {decision_score} < 5, overriding to NONE signal.")
            signal = TradeSignal(
                signal="NONE",
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                target=signal.target,
                confidence=signal.confidence,
                rationale=f"Decision score {decision_score} below threshold.",
            )

        risk = self.risk_agent.check(signal)
        output = OrchestratorOutput(
            signal=signal,
            risk=risk,
            decision_score=decision_score,
            state=state,
        )
        logger.info(f"DecisionEngine output: {output}")
        return output
