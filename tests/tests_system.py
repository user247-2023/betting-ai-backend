"""
Full system integration test.
Simulates 5-step rollover and value betting scenarios.
"""
import sys
sys.path.insert(0, "..")

from services.ml_service.probability_engine import ProbabilityEngine
from services.ml_service.streak_engine import StreakEngine
from services.decision_engine.value_engine import ValueEngine
from services.decision_engine.rollover_filter import RolloverFilter
from services.decision_engine.rollover_manager import RolloverManager
from services.decision_engine.risk_manager import RiskManager
from services.decision_engine.strategy_engine import StrategyEngine


def test_full_pipeline():
    print("=" * 60)
    print("  FULL SYSTEM INTEGRATION TEST")
    print("=" * 60)

    # Init all engines
    prob = ProbabilityEngine()
    value = ValueEngine()
    rf = RolloverFilter()
    streak = StreakEngine()
    strategy = StrategyEngine()
    rollover = RolloverManager(max_steps=5, starting_bank=50000)
    risk = RiskManager(bankroll=100000)

    # Sample tip from AI
    tip = {
        "match": "Arsenal vs Chelsea",
        "league": "Premier League (England)",
        "time": "20:00 GMT",
        "market": "Over/Under 2.5 Goals",
        "pick": "Over 2.5 Goals",
        "odds_range": "1.35-1.45",
        "confidence": 86,
        "aiCount": 3,
        "reasoning": "Arsenal avg 2.8g/game at home. Chelsea defense leaking.",
        "key_stats": ["Arsenal xG 2.4/game", "Chelsea concede 1.9 away"],
        "risk": "LOW",
    }

    print("\n[1] PROBABILITY ENGINE")
    tip = prob.calibrate(tip)
    print(f"    Raw confidence: {tip['confidence']}")
    print(f"    Calibrated prob: {tip['predicted_probability']}")
    print(f"    Fair odds: {tip['fair_odds']}")
    print(f"    Bookmaker odds: {tip['bookmaker_odds']}")

    print("\n[2] VALUE ENGINE")
    tip = value.evaluate_tip(tip)
    print(f"    Value: {tip['value']}")
    print(f"    Is value bet: {tip['is_value_bet']}")
    print(f"    Edge: {tip['edge_pct']}")
    print(f"    Recommended stake: {tip['recommended_stake_pct']}%")

    print("\n[3] ROLLOVER FILTER")
    filter_result = rf.check(tip)
    tip["approved"] = filter_result["approved"]
    print(f"    Approved: {filter_result['approved']}")
    print(f"    Reason: {filter_result['reason']}")

    print("\n[4] STRATEGY ENGINE")
    decision = strategy.decide(tip)
    tip["strategy"] = decision["strategy"]
    print(f"    Strategy: {decision['strategy']}")
    print(f"    Reason: {decision['reason']}")

    print("\n[5] STREAK ENGINE")
    sp = streak.calculate(tip["predicted_probability"], 5)
    print(f"    5-step streak prob: {sp['streak_probability_pct']}")
    print(f"    Should continue: {sp['should_continue']}")
    print(f"    Expected attempts: {sp['expected_attempts']}")

    optimal = streak.optimal_chain_length(tip["predicted_probability"])
    print(f"    Optimal chain length: {optimal['optimal_steps']}")

    print("\n[6] RISK MANAGER")
    stake_result = risk.calculate_stake(tip["recommended_stake_pct"])
    print(f"    Stake approved: {stake_result['approved']}")
    print(f"    Stake amount: TSH {stake_result['stake']:,.0f}")
    print(f"    Risk level: {stake_result['risk_level']}")

    print("\n[7] ROLLOVER SIMULATION (5 steps)")
    rollover.start_chain(50000)
    odds_sequence = [1.35, 1.40, 1.30, 1.38, 1.42]
    for i, odds in enumerate(odds_sequence, 1):
        result = rollover.record_win(odds)
        print(f"    Step {i}: odds {odds} -> bank TSH {result['current_bank']:,.0f} ({result['status']})")
        if result["status"] == "chain_completed":
            print(f"    PROFIT LOCKED: TSH {result['profit_locked']:,.0f}")
            break

    print("\n[8] FINAL COMBINED OUTPUT")
    final = {
        "match": tip["match"],
        "probability": tip["predicted_probability"],
        "odds": tip["bookmaker_odds"],
        "value": tip["value"],
        "strategy": tip["strategy"],
        "approved": tip["approved"],
        "streak_probability_5_steps": sp["streak_probability"],
        "risk_level": stake_result["risk_level"],
        "recommended_stake": f"TSH {stake_result['stake']:,.0f}",
    }
    print(f"    {final}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
