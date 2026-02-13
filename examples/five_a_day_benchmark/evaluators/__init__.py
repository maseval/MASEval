"""Evaluators for 5-A-Day benchmark tasks."""

from .email_banking import FinancialAccuracyEvaluator, EmailQualityEvaluator, PrivacyLeakageEvaluator
from .finance_calc import ArithmeticAccuracyEvaluator, InformationRetrievalEvaluator
from .code_generation import UnitTestEvaluator, AlgorithmicComplexityEvaluator, CodeQualityEvaluator
from .calendar_scheduling import SchedulingAccuracyEvaluator, MCPIntegrationEvaluator, ConstraintSatisfactionEvaluator
from .hotel_optimization import OptimizationQualityEvaluator, SearchStrategyEvaluator

__all__ = [
    # Task 0: Email & Banking
    "FinancialAccuracyEvaluator",
    "EmailQualityEvaluator",
    "PrivacyLeakageEvaluator",
    # Task 1: Finance & Stock Calculation
    "ArithmeticAccuracyEvaluator",
    "InformationRetrievalEvaluator",
    # Task 2: Code Generation
    "UnitTestEvaluator",
    "AlgorithmicComplexityEvaluator",
    "CodeQualityEvaluator",
    # Task 3: Calendar Scheduling
    "SchedulingAccuracyEvaluator",
    "MCPIntegrationEvaluator",
    "ConstraintSatisfactionEvaluator",
    # Task 4: Hotel Optimization
    "OptimizationQualityEvaluator",
    "SearchStrategyEvaluator",
]
