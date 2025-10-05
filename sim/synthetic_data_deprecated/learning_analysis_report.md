# Learning ELM System Analysis Report
==================================================

## Performance Comparison

**ELM Only:**
- Initial Performance (Episodes 1-10): 0.0
- Final Performance (Episodes 41-50): 0.0
- Improvement: 0.0 (nan%)
- Learning Trend: stable

**ELM + LLM Guidance:**
- Initial Performance (Episodes 1-10): 666.0
- Final Performance (Episodes 41-50): 660.0
- Improvement: -6.0 (-0.9%)
- Learning Trend: declining
- LLM Guidance Usage: 1011 times

## Learning Statistics

**ELM Only:**
- Total Episodes: 50
- Learning Updates: 50
- Memory Size: 1000
- Average Reward: -151.44

**ELM + LLM:**
- Total Episodes: 50
- Learning Updates: 49
- Memory Size: 1000
- Average Reward: 200.65
- LLM Guidance Rate: 2022.0%

## Comparative Analysis

**Learning Efficiency:**
- ELM+LLM vs ELM Only improvement difference: -6.0
- Final performance difference: 660.0

**Conclusion:** LLM guidance shows significant positive effect on learning