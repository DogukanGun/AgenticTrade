# Agent Strategy Drift Detection System

A research proof-of-concept for detecting strategy drift in autonomous trading agents using online learning, semantic embeddings, and behavioral monitoring.

## Overview

This system monitors agent behavior in real-time to detect when an agent's strategy deviates from its original mandate. It uses:

- **ADWIN (Adaptive Windowing)** for online drift detection
- **Sentence Transformers** for semantic embedding of agent responses
- **Agent Strategy Index (ASI)** - a composite metric across 4 behavioral dimensions
- **Contract Enforcement** via YAML-defined mandate rules
- **Episodic Memory** for behavioral context tracking
- **Simulated Blockchain Audit** (SQLite-backed) for immutable event logging

## Architecture

```
Agent Turns → BaselineProfiler → MetricsComputer → DriftDetector
                                       ↓
                            ContractEnforcer → AuditLog
                                       ↓
                            EpisodicMemory → API
```

## Metrics (ASI Score Components)

### Group A: Response Consistency (30%)
1. Cosine embedding similarity
2. Levenshtein distance on reasoning paths
3. JS divergence of confidence distributions

### Group B: Tool Usage (25%)
4. Chi-squared tool distribution test
5. Tool sequence similarity
6. KL divergence of tool parameters

### Group C: Inter-Agent Coordination (25%)
7. Consensus rate
8. Handoff efficiency
9. Mutual information (role → action)

### Group D: Behavioral Boundaries (20%)
10. Output length coefficient of variation
11. Error clustering coefficient
12. Human override rate

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## API

Start the server:
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8765
```

Endpoints:
- `POST /session/start` - Initialize a new monitoring session
- `POST /turn` - Submit an agent turn for analysis
- `GET /session/{id}/audit` - Retrieve metric history
- `GET /session/{id}/chain-events` - Get audit log entries
- `GET /health` - Health check

## Testing

```bash
pytest tests/ -v
```

## Configuration

Edit `config.yaml` to adjust:
- Baseline window size and model
- ADWIN sensitivity (delta parameter)
- Metric weights
- Benchmark parameters

Edit `mandate.yaml` to define trading rules.

## Benchmark Results

The benchmark harness generates synthetic sessions with injected drift and measures:
- **Detection lag**: Turns from drift injection to ADWIN alert
- **False positive rate**: Alerts on clean sessions
- **False negative rate**: Missed drift events
- **Mean recovery turns**: Turns to recover after remediation
