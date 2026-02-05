"""
DeepEval Evaluation for RAG Pipeline.

Evaluates:
  1. Answer Relevancy     - Is the answer relevant to the question?
  2. Faithfulness         - Is the answer grounded in the retrieved context?
  3. Contextual Recall    - Did retrieval capture the expected context?
  4. Contextual Precision - Is the retrieved context relevant to the question?
  5. Correctness (GEval)  - Is the answer factually correct?

Usage:
    # Run with DeepEval test runner (uses 5 samples)
    deepeval test run test_rag.py

    # Run standalone - quick mode (5 samples)
    python test_rag.py --quick

    # Run standalone - full evaluation (20 samples)
    python test_rag.py

    # Evaluate a single question
    python test_rag.py "What is the PTO policy?"
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
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from eval_dataset import EVAL_DATASET, get_quick_dataset
from rag_pipeline import RAGPipeline


# =============================================================================
# Pipeline Initialization
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

# Caches for efficiency
_rag_pipeline: RAGPipeline | None = None
_test_case_cache: dict[str, LLMTestCase] = {}


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline (singleton)."""
    global _rag_pipeline
    if _rag_pipeline is None:
        print("Initializing RAG pipeline...")
        _rag_pipeline = RAGPipeline()
        _rag_pipeline.load_documents(str(DATA_DIR))
    return _rag_pipeline


def build_test_case(eval_item: dict) -> LLMTestCase:
    """Build a DeepEval test case (cached to avoid re-querying)."""
    cache_key = eval_item["input"]

    if cache_key not in _test_case_cache:
        rag = get_pipeline()
        result = rag.query(eval_item["input"])

        _test_case_cache[cache_key] = LLMTestCase(
            input=eval_item["input"],
            actual_output=result["output"],
            expected_output=eval_item["expected_output"],
            retrieval_context=result["context"],
            context=[eval_item["expected_context"]],
        )

    return _test_case_cache[cache_key]


# =============================================================================
# Metrics
# =============================================================================


def get_metrics():
    """Return all RAG evaluation metrics."""
    return [
        AnswerRelevancyMetric(threshold=0.5, include_reason=True),
        FaithfulnessMetric(threshold=0.5, include_reason=True),
        ContextualRecallMetric(threshold=0.5, include_reason=True),
        ContextualPrecisionMetric(threshold=0.4, include_reason=True),
        GEval(
            name="Correctness",
            criteria=(
                "Determine whether the actual output is factually correct "
                "and semantically similar to the expected output."
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

QUICK_DATASET = get_quick_dataset(5)


def test_rag_evaluation():
    """
    Single test that evaluates the RAG pipeline on a subset of cases.
    This is efficient for `deepeval test run`.
    """
    print("\nBuilding test cases...")
    test_cases = [build_test_case(item) for item in QUICK_DATASET]
    metrics = get_metrics()

    print(f"Evaluating {len(test_cases)} test cases with {len(metrics)} metrics...")
    for tc in test_cases:
        assert_test(tc, metrics)


# =============================================================================
# Standalone Evaluation
# =============================================================================


def run_evaluation(quick: bool = False):
    """
    Run full evaluation using DeepEval's evaluate() function.
    More efficient than pytest for development and debugging.
    """
    dataset = QUICK_DATASET if quick else EVAL_DATASET

    print("=" * 70)
    print(f"RAG PIPELINE EVALUATION ({'Quick' if quick else 'Full'} - {len(dataset)} samples)")
    print("=" * 70)

    # Build test cases
    print(f"\nBuilding {len(dataset)} test cases...")
    test_cases = []
    for i, item in enumerate(dataset):
        print(f"  [{i + 1}/{len(dataset)}] {item['input'][:50]}...")
        test_cases.append(build_test_case(item))

    # Run evaluation
    metrics = get_metrics()
    print(f"\nRunning {len(metrics)} metrics on {len(test_cases)} test cases...")
    print("This may take a few minutes...\n")

    results = evaluate(test_cases=test_cases, metrics=metrics)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    for i, tc in enumerate(test_cases):
        print(f"\n[{i + 1}] {tc.input[:60]}...")
        print(f"    Answer: {tc.actual_output[:80]}...")

    return results


def run_single_query(question: str):
    """Evaluate a single question for debugging."""
    print(f"\nEvaluating: {question}")
    print("-" * 60)

    rag = get_pipeline()
    result = rag.query(question)

    print(f"Answer: {result['output']}")
    print(f"Sources: {result['sources']}")
    print(f"\nRetrieved context ({len(result['context'])} chunks):")
    for i, ctx in enumerate(result["context"]):
        print(f"  [{i + 1}] {ctx[:100]}...")

    test_case = LLMTestCase(
        input=question,
        actual_output=result["output"],
        retrieval_context=result["context"],
    )

    # Metrics that don't need expected_output
    metrics = [
        AnswerRelevancyMetric(threshold=0.5, include_reason=True),
        FaithfulnessMetric(threshold=0.5, include_reason=True),
    ]

    print("\nRunning metrics...")
    evaluate(test_cases=[test_case], metrics=metrics)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if "--quick" in sys.argv:
        run_evaluation(quick=True)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        question = " ".join(sys.argv[1:])
        run_single_query(question)
    else:
        run_evaluation(quick=False)
