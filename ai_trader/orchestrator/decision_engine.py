from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph
from loguru import logger

from ai_trader.agents.chart_agent import ChartAgent
from ai_trader.agents.option_chain_agent import OptionChainAgent
from ai_trader.agents.news_agent import NewsAgent
from ai_trader.agents.news_agent import NewsMacroAnalysis
from ai_trader.agents.volatility_agent import VolatilityAgent
from ai_trader.agents.volatility_agent import VolatilityAnalysis
from ai_trader.agents.regime_agent import RegimeAgent
from ai_trader.agents.regime_agent import RegimeAnalysis
from ai_trader.agents.liquidity_agent import LiquidityAgent
from ai_trader.agents.liquidity_agent import LiquidityAnalysis
from ai_trader.agents.chart_agent import ChartAnalysis
from ai_trader.agents.option_chain_agent import OptionChainAnalysis
from ai_trader.agents.trigger_agent import TradeTriggerAgent, TradeSignal
from ai_trader.agents.risk_agent import RiskManagerAgent, RiskCheckResult
from ai_trader.agents.fii_positioning_agent import FiiPositioningAgent, FiiPositioningAnalysis
from ai_trader.agents.gamma_agent import GammaAgent, GammaAnalysis
from ai_trader.agents.liquidity_sweep_agent import LiquiditySweepAgent, LiquiditySweepAnalysis
from ai_trader.agents.llm_validator_agent import LlmValidationResult, LlmValidatorAgent
from ai_trader.config.settings import settings
from ai_trader.data.market_data_context import MarketDataProvider
from ai_trader.data.nse_option_chain import NseOptionChainClient
from ai_trader.data.nse_session import build_nse_session


@dataclass
class OrchestratorOutput:
    timestamp: str
    signal: TradeSignal
    risk: RiskCheckResult
    llm_validation: LlmValidationResult
    decision_score: int
    state: Dict[str, Any]


def _serialize_value(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value.__dict__)
    return value


class DecisionEngine:
    """Orchestrates all agents via a simple LangGraph."""

    def __init__(self) -> None:
        shared_nse_session = build_nse_session()
        self.chart_agent = ChartAgent()
        self.option_agent = OptionChainAgent(client=NseOptionChainClient(session=shared_nse_session))
        self.news_agent = NewsAgent()
        self.vol_agent = VolatilityAgent(session=shared_nse_session)
        self.regime_agent = RegimeAgent()
        self.liquidity_agent = LiquidityAgent()
        self.fii_agent = FiiPositioningAgent()
        self.gamma_agent = GammaAgent()
        self.liquidity_sweep_agent = LiquiditySweepAgent()
        self.trigger_agent = TradeTriggerAgent()
        self.risk_agent = RiskManagerAgent()
        self.llm_validator = LlmValidatorAgent()
        self.market_data_provider = MarketDataProvider(
            kite_client=self.chart_agent.client,
            option_chain_client=self.option_agent.client,
            vol_agent=self.vol_agent,
        )

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        # State is a simple dict passed between nodes
        graph = StateGraph(dict)

        fallback_factories = {
            "chart": lambda: ChartAnalysis(trend="neutral", structure="unknown", confidence=0.0),
            "option_chain": lambda: OptionChainAnalysis(support=None, resistance=None, pcr=1.0, bias="neutral"),
            "news": lambda: NewsMacroAnalysis(
                macro_bias="neutral",
                risk_level="high",
                data_available=False,
                fallback_used=True,
            ),
            "vol": lambda: VolatilityAnalysis(
                volatility="high",
                expected_range=(0.0, 0.0),
                data_available=False,
                fallback_used=True,
            ),
            "regime": lambda: RegimeAnalysis(regime="range_bound", confidence=0.0),
            "liquidity": lambda: LiquidityAnalysis(liquidity="low", slippage_risk="high"),
            "fii_positioning": lambda: FiiPositioningAnalysis(
                fii_bias="neutral",
                institutional_support=None,
                institutional_resistance=None,
                confidence=0.0,
            ),
            "gamma_analysis": lambda: GammaAnalysis(
                gamma_regime="positive_gamma",
                gamma_flip_level=None,
                expected_move="compression",
            ),
            "liquidity_sweep": lambda: LiquiditySweepAnalysis(
                liquidity_event=False,
                event_type="none",
                confidence=0.0,
            ),
        }

        def gather_node(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self.market_data_provider.build()
            state["market_context"] = context
            state["agent_health"] = {}
            with ThreadPoolExecutor(max_workers=9) as executor:
                futures = {
                    "chart": executor.submit(self.chart_agent.analyze, context),
                    "option_chain": executor.submit(self.option_agent.analyze, context),
                    "news": executor.submit(self.news_agent.analyze),
                    "vol": executor.submit(self.vol_agent.analyze, context.spot_price, context),
                    "regime": executor.submit(self.regime_agent.analyze, context),
                    "liquidity": executor.submit(self.liquidity_agent.analyze, None, None, context),
                    "fii_positioning": executor.submit(self.fii_agent.analyze, context),
                    "gamma_analysis": executor.submit(self.gamma_agent.analyze, context),
                    "liquidity_sweep": executor.submit(self.liquidity_sweep_agent.analyze, context),
                }
                for key, future in futures.items():
                    try:
                        state[key] = future.result()
                        state["agent_health"][key] = {"status": "ok", "fallback": False, "error": None}
                    except Exception as exc:  # noqa: BLE001
                        logger.error(f"{key} agent failed during gather: {exc}")
                        state[key] = fallback_factories[key]()
                        state["agent_health"][key] = {
                            "status": "error",
                            "fallback": True,
                            "error": str(exc),
                        }
            return state

        def trigger_node(state: Dict[str, Any]) -> Dict[str, Any]:
            chart = state["chart"]
            option_chain = state["option_chain"]
            news = state["news"]
            vol = state["vol"]
            regime = state["regime"]
            liq = state["liquidity"]
            fii = state["fii_positioning"]
            gamma = state["gamma_analysis"]
            liquidity_sweep = state["liquidity_sweep"]
            context = state["market_context"]

            signal = self.trigger_agent.generate_signal(
                chart=chart,
                option_chain=option_chain,
                news=news,
                vol=vol,
                regime=regime,
                liquidity=liq,
                fii=fii,
                gamma=gamma,
                liquidity_sweep=liquidity_sweep,
                spot=context.spot_price,
                market_context=context,
            )
            state["signal"] = signal

            score_breakdown = {
                "chart_alignment": 0,
                "option_chain_confirmation": 0,
                "regime_confirmation": 0,
                "fii_positioning": 0,
                "gamma_signal": 0,
                "liquidity_validation": 0,
            }
            if context.quality.price_data_available and chart.trend in ("bullish", "bearish") and chart.confidence >= 0.55:
                score_breakdown["chart_alignment"] = 2
            if context.quality.option_chain_available and option_chain.bias in ("bullish", "bearish"):
                score_breakdown["option_chain_confirmation"] = 2
            if context.quality.price_data_available and regime.regime in ("trend_up", "trend_down") and regime.confidence >= 0.55:
                score_breakdown["regime_confirmation"] = 2
            if (
                context.quality.option_chain_available
                and context.quality.fii_data_available
                and fii.fii_bias in ("bullish", "bearish")
                and fii.confidence >= 0.55
            ):
                score_breakdown["fii_positioning"] = 2
            if context.quality.option_chain_available and gamma.gamma_regime == "negative_gamma":
                score_breakdown["gamma_signal"] = 1
            if (
                context.quality.price_data_available
                and (not liquidity_sweep.liquidity_event or liquidity_sweep.event_type == "stop_hunt")
            ):
                score_breakdown["liquidity_validation"] = 1

            score = sum(score_breakdown.values())
            state["score_breakdown"] = score_breakdown
            news_available = getattr(news, "data_available", True) and not getattr(news, "fallback_used", False)
            state["score_complete"] = (
                context.quality.critical_inputs_available and signal.data_complete and news_available
            )
            state["decision_score"] = score
            return state

        graph.add_node("gather", gather_node)
        graph.add_node("trigger", trigger_node)

        graph.set_entry_point("gather")
        graph.add_edge("gather", "trigger")

        return graph.compile()

    def run_once(self, *, open_trades: int = 0) -> OrchestratorOutput:
        state: Dict[str, Any] = {"open_trades": open_trades}
        state = self.graph.invoke(state)  # type: ignore[assignment]

        signal: TradeSignal = state["signal"]
        decision_score: int = state["decision_score"]
        score_complete = bool(state.get("score_complete", False))

        if not score_complete:
            logger.info("Decision score marked incomplete due to missing critical market data or option pricing.")
            signal = TradeSignal(
                signal="NONE",
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                target=signal.target,
                confidence=0.0,
                rationale="Critical market data incomplete; refusing to score a trade.",
                underlying_spot=signal.underlying_spot,
                option_strike=signal.option_strike,
                option_expiry=signal.option_expiry,
                instrument_symbol=signal.instrument_symbol,
                instrument_key=signal.instrument_key,
                price_source=signal.price_source,
                data_complete=False,
            )
        elif decision_score < settings.orchestrator_min_score:
            logger.info(
                f"Decision score {decision_score} < {settings.orchestrator_min_score}, overriding to NONE signal."
            )
            signal = TradeSignal(
                signal="NONE",
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                target=signal.target,
                confidence=signal.confidence,
                rationale=f"Decision score {decision_score} below threshold.",
                underlying_spot=signal.underlying_spot,
                option_strike=signal.option_strike,
                option_expiry=signal.option_expiry,
                instrument_symbol=signal.instrument_symbol,
                instrument_key=signal.instrument_key,
                price_source=signal.price_source,
                data_complete=signal.data_complete,
            )

        validation_payload = {
            "signal": signal.signal,
            "entry": signal.entry,
            "stop_loss": signal.stop_loss,
            "target": signal.target,
            "confidence": signal.confidence,
            "chart_analysis": getattr(state["chart"], "__dict__", state["chart"]),
            "option_chain_analysis": getattr(state["option_chain"], "__dict__", state["option_chain"]),
            "regime": getattr(state["regime"], "__dict__", state["regime"]),
            "volatility": getattr(state["vol"], "__dict__", state["vol"]),
            "liquidity": getattr(state["liquidity"], "__dict__", state["liquidity"]),
            "fii_positioning": getattr(state["fii_positioning"], "__dict__", state["fii_positioning"]),
            "gamma_analysis": getattr(state["gamma_analysis"], "__dict__", state["gamma_analysis"]),
            "liquidity_sweep": getattr(state["liquidity_sweep"], "__dict__", state["liquidity_sweep"]),
            "decision_score": decision_score,
            "score_complete": score_complete,
            "data_quality": state["market_context"].quality.to_dict(),
        }
        try:
            llm_validation = self.llm_validator.validate(validation_payload)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"LLM validator failed unexpectedly; using deterministic fallback: {exc}")
            llm_validation = LlmValidationResult(
                validation="approved",
                confidence_adjustment=0.0,
                reasoning=f"LLM validator failure; using deterministic signal. {exc}",
                source="deterministic_fallback",
                fallback_used=True,
            )
        if signal.signal != "NONE" and llm_validation.validation == "approved":
            signal.confidence = max(0.0, min(1.0, signal.confidence + llm_validation.confidence_adjustment))
            signal.rationale = f"{signal.rationale} {llm_validation.reasoning}"
        elif signal.signal != "NONE" and llm_validation.validation == "rejected":
            signal = TradeSignal(
                signal="NONE",
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                target=signal.target,
                confidence=max(0.0, signal.confidence + llm_validation.confidence_adjustment),
                rationale=llm_validation.reasoning,
                underlying_spot=signal.underlying_spot,
                option_strike=signal.option_strike,
                option_expiry=signal.option_expiry,
                instrument_symbol=signal.instrument_symbol,
                instrument_key=signal.instrument_key,
                price_source=signal.price_source,
                data_complete=signal.data_complete,
            )

        if signal.signal == "NONE":
            risk = RiskCheckResult(allowed=False, reason="No trade signal after validation.")
        else:
            risk = self.risk_agent.authorize_signal(
                signal,
                open_trades=state.get("open_trades", 0),
                liquidity=state["liquidity"].liquidity,
                volatility=state["vol"].volatility,
                news_risk=state["news"].risk_level,
            )
            if not risk.allowed:
                signal = TradeSignal(
                    signal="NONE",
                    entry=signal.entry,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    confidence=signal.confidence,
                    rationale=risk.reason or "Rejected by risk manager.",
                    underlying_spot=signal.underlying_spot,
                    option_strike=signal.option_strike,
                    option_expiry=signal.option_expiry,
                    instrument_symbol=signal.instrument_symbol,
                    instrument_key=signal.instrument_key,
                    price_source=signal.price_source,
                    data_complete=signal.data_complete,
                )

        output = OrchestratorOutput(
            timestamp=datetime.utcnow().isoformat(),
            signal=signal,
            risk=risk,
            llm_validation=llm_validation,
            decision_score=decision_score,
            state=state,
        )
        logger.info(f"DecisionEngine output: {output}")
        return output
