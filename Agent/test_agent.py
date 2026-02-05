"""
DeepEval Evaluation for LangGraph Agent.

Evaluates:
  1. Tool Correctness  - Did the agent call the right tools?
  2. Answer Relevancy  - Is the answer relevant to the question?
  3. Output Correctness - Is the answer factually correct?

Usage:
    # Run with DeepEval test runner
    deepeval test run test_agent.py

    # Run standalone (faster, batched)
    python test_agent.py
"""

import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from dotenv import load_dotenv

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from deepeval import assert_test, evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

from agent import run_agent


# =============================================================================
# Test Scenarios
# =============================================================================

TEST_SCENARIOS = [
    {
        "input": "What's the weather like in San Francisco?",
        "expected_output": "San Francisco: 62°F, sunny and clear.",
        "expected_tools": [ToolCall(name="get_weather")],
    },
    {
        "input": "Calculate 15% tip on a $85 bill.",
        "expected_output": "A 15% tip on $85 is $12.75.",
        "expected_tools": [ToolCall(name="calculate")],
    },
    {
        "input": "What is your refund policy?",
        "expected_output": (
            "Our refund policy allows returns within 30 days for a full refund. "
            "Items must be unused and in original packaging."
        ),
        "expected_tools": [ToolCall(name="search_knowledge_base")],
    },
    {
        "input": "How much is express shipping?",
        "expected_output": "Express shipping (2-day) is available for $9.99.",
        "expected_tools": [ToolCall(name="search_knowledge_base")],
    },
]


# =============================================================================
# Test Case Builder
# =============================================================================

# Cache to avoid re-running agent for same inputs
_test_case_cache: dict[str, LLMTestCase] = {}


def build_test_case(scenario: dict) -> LLMTestCase:
    """Run the agent and build a DeepEval test case (cached)."""
    cache_key = scenario["input"]

    if cache_key not in _test_case_cache:
        result = run_agent(scenario["input"])

        tools_called = [
            ToolCall(
                name=tc["name"],
                input_parameters=tc.get("args"),
                output=tc.get("output"),
            )
            for tc in result["tools_called"]
        ]

        _test_case_cache[cache_key] = LLMTestCase(
            input=scenario["input"],
            actual_output=result["output"],
            expected_output=scenario["expected_output"],
            tools_called=tools_called,
            expected_tools=scenario["expected_tools"],
        )

    return _test_case_cache[cache_key]


# =============================================================================
# Metrics
# =============================================================================


def get_metrics():
    """Return all evaluation metrics."""
    return [
        ToolCorrectnessMetric(threshold=0.5, include_reason=True),
        AnswerRelevancyMetric(threshold=0.5, include_reason=True),
        GEval(
            name="Correctness",
            criteria=(
                "Determine whether the actual output is factually correct "
                "based on the expected output."
            ),
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.5,
        ),
    ]


# =============================================================================
# Pytest Tests (for `deepeval test run`)
# =============================================================================


def test_agent_evaluation():
    """
    Single test that evaluates the agent on all scenarios.
    This is efficient for `deepeval test run`.
    """
    print("\nBuilding test cases...")
    test_cases = [build_test_case(s) for s in TEST_SCENARIOS]
    metrics = get_metrics()

    print(f"Evaluating {len(test_cases)} test cases with {len(metrics)} metrics...")
    for tc in test_cases:
        assert_test(tc, metrics)


# =============================================================================
# Standalone Mode
# =============================================================================


def main():
    """Run evaluation in standalone mode (more efficient for development)."""
    print("=" * 70)
    print("AGENT EVALUATION")
    print("=" * 70)

    print("\nBuilding test cases...")
    test_cases = [build_test_case(s) for s in TEST_SCENARIOS]

    print(f"\nEvaluating {len(test_cases)} test cases...")
    results = evaluate(test_cases=test_cases, metrics=get_metrics())

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for i, tc in enumerate(test_cases):
        print(f"\n[{i + 1}] {tc.input}")
        print(f"    Output: {tc.actual_output[:80]}...")
        print(f"    Tools: {[t.name for t in tc.tools_called]}")

    return results


if __name__ == "__main__":
    main()
