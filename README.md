# DeepEval Examples

A collection of examples demonstrating LLM evaluation using [DeepEval](https://github.com/confident-ai/deepeval) with LangChain and LangGraph.

## Overview

This repository contains two complete examples:

| Example | Description | Key Metrics |
|---------|-------------|-------------|
| **Agent** | LangGraph ReAct agent with tool calling | Tool Correctness, Answer Relevancy, Correctness |
| **RAG** | Retrieval-Augmented Generation pipeline | Faithfulness, Contextual Recall/Precision, Answer Relevancy |

## Project Structure

```
DeepEval_Examples/
├── Agent/                      # LangGraph Agent Example
│   ├── agent.py               # Agent implementation with tools
│   ├── test_agent.py          # DeepEval evaluation tests
│   └── __init__.py
│
├── RAG/                        # RAG Pipeline Example
│   ├── data/                  # Source documents (5 txt files)
│   │   ├── company_handbook.txt
│   │   ├── product_faq.txt
│   │   ├── support_tickets.txt
│   │   ├── api_refrence.txt
│   │   └── training_doc.txt
│   ├── rag_pipeline.py        # RAG implementation with FAISS
│   ├── eval_dataset.py        # 20 Q&A pairs for evaluation
│   ├── test_rag.py            # DeepEval evaluation tests
│   └── __init__.py
│
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd DeepEval_Examples

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Run Examples

#### Agent Example

```bash
cd Agent

# Test the agent
python agent.py

# Run evaluation (standalone - recommended)
python test_agent.py

# Run with DeepEval test runner
deepeval test run test_agent.py
```

#### RAG Example

```bash
cd RAG

# Test the RAG pipeline
python rag_pipeline.py

# Run evaluation - quick mode (5 samples, ~2 min)
python test_rag.py --quick

# Run evaluation - full mode (20 samples, ~10 min)
python test_rag.py

# Run with DeepEval test runner
deepeval test run test_rag.py

# Evaluate a single question
python test_rag.py "What is the PTO policy?"
```

## Evaluation Metrics

### Agent Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Tool Correctness** | Did the agent call the correct tools? | 0.5 |
| **Answer Relevancy** | Is the answer relevant to the question? | 0.5 |
| **Correctness** | Is the answer factually correct? | 0.5 |

### RAG Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Answer Relevancy** | Is the answer relevant to the question? | 0.5 |
| **Faithfulness** | Is the answer grounded in retrieved context? | 0.5 |
| **Contextual Recall** | Did retrieval capture the expected context? | 0.5 |
| **Contextual Precision** | Is the retrieved context relevant? | 0.4 |
| **Correctness** | Is the answer factually correct? | 0.5 |

## Configuration

### RAG Pipeline Parameters

```python
RAGPipeline(
    model_name="gpt-4o-mini",           # LLM for generation
    embedding_model="text-embedding-3-small",  # Embedding model
    chunk_size=500,                      # Characters per chunk
    chunk_overlap=50,                    # Overlap between chunks
    top_k=4,                             # Documents to retrieve
)
```

### Agent Tools

The agent has access to three tools:
- `get_weather`: Returns weather for major cities
- `calculate`: Evaluates math expressions
- `search_knowledge_base`: Searches company policies

## Development

### Adding New Test Cases

**For Agent:**
Edit `Agent/test_agent.py` and add to `TEST_SCENARIOS`:

```python
{
    "input": "Your question here",
    "expected_output": "Expected answer",
    "expected_tools": [ToolCall(name="tool_name")],
}
```

**For RAG:**
Edit `RAG/eval_dataset.py` and add to `EVAL_DATASET`:

```python
{
    "input": "Your question here",
    "expected_output": "Expected answer",
    "expected_context": "Text that should be retrieved",
    "source_file": "source_document.txt",
}
```

### Adjusting Thresholds

Modify the threshold values in `get_metrics()` functions to tune sensitivity:
- **Higher thresholds**: Stricter evaluation, more failures
- **Lower thresholds**: More lenient, fewer failures

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: OPENAI_API_KEY not set
   ```
   Solution: Ensure `.env` file exists with valid API key

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain_text_splitters'
   ```
   Solution: `pip install -r requirements.txt`

3. **Slow Evaluation**
   - Use `--quick` flag for faster iteration
   - Use `python test_*.py` instead of `deepeval test run` for batched execution

## License

MIT License - See LICENSE file for details.

## Resources

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
