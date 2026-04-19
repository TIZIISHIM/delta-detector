
# Transcript Delta Detector

A tool for detecting and classifying material changes between two earnings call transcripts. Identifies what changed in management language, which topics were affected, and how material the changes are.

## Features

- Change Detection: Identifies added, removed, and modified sentences between two transcripts
- Topic Classification: Automatically classifies changes by topic (revenue, guidance, margins, competition, etc.)
- Materiality Assessment: Scores changes as low, moderate, or high materiality
- Sentiment Integration: Combines with FinBERT to detect sentiment shifts
- Delta Reports: Generates structured JSON reports of all changes

## Installation

```bash
pip install -r requirements.txt
