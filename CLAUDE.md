# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A VoiceBot project that processes multilingual government schemes data from a CSV file (`schemes_multilingual (1).csv`) and builds a RAG (Retrieval-Augmented Generation) pipeline using LangChain.

## Package Manager

This project uses **uv** (not pip) for dependency management. Always use `uv` commands:

```bash
uv sync                  # Install dependencies
uv run python main.py    # Run the project
uv add <package>         # Add a dependency
```

## Running the Project

```bash
uv run python main.py
```

## Architecture

- **`main.py`** — Entry point. Imports from `VectorStore.pipeline` and sets the data file path.
- **`VectorStore/pipeline.py`** — Core pipeline module. Reads the CSV using `csv.DictReader` and chunks documents using `langchain_text_splitters.RecursiveCharacterTextSplitter`.
- **`VectorStore/__init__.py`** — Empty init (package marker).
- **`schemes_multilingual (1).csv`** — Source data: multilingual government schemes dataset.

## Key Dependencies

- `langchain` >= 1.2.13
- `langchain-text-splitters` >= 1.1.1
- Python 3.14 (see `.python-version`)
