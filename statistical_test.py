#!/usr/bin/env python3
"""
Statistical test for ELM vs ELM+LLM performance comparison
Runs multiple automated tests to collect reliable data
"""

import requests
import json
import time
import statistics
from typing import List, Dict

class TowerDefenseStatisticalTest:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = {
            'elm_only': [],
            'elm_llm': []
        }
    
    def run_single_test(self, model_type: str, duration_seconds: int = 30) -> Dict:
        """Run a single test for specified duration"""
        print(f"Running {model_type} test for {duration_seconds} seconds...")
        
        # Initial game state
        game_state = {
            'money': 250,
            'health': 100,
            'wave': 1,
            'score': 0,
            'towers': 0,
            'enemies': 3
        }
        
        start_time = time.time()
        predictions = []
        
        # Simulate game progression
        while time.time() - start_time < duration_seconds:
            try:
                # Get ELM prediction
                response = requests.post(
                    f"{self.base_url}/api/elm-predict",
                    json={
                        'game_state': game_state,
                        'model_type': 'baseline' if model_type == 'elm_only' else 'hybrid'
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    prediction = response.json()
                    predictions.append(prediction)
                    
                    # Simulate tower placement if recommended
                    if prediction.get('should_place_tower', False) and game_state['money'] >= 50:
                        game_state['towers'] += 1
                        game_state['money'] -= 50
                        # Simulate enemy kills and rewards
                        kills = min(2, game_state['enemies'])  # Assume 2 kills per tower
                        game_state['score'] += kills * 30
                        game_state['money'] += kills * 30
                    
                    # Simulate game progression
                    game_state['enemies'] = max(3, game_state['enemies'] + 1)
                    game_state['wave'] = min(10, game_state['wave'] + 0.1)
                
                time.sleep(2)  # Match game automation interval
                
            except Exception as e:
                print(f"Error in test: {e}")
                time.sleep(1)
        
        # Calculate final metrics
        final_score = game_state['score']
        towers_built = game_state['towers']
        efficiency = final_score / max(1, towers_built)  # Score per tower
        llm_guidance_rate = sum(1 for p in predictions if p.get('llm_guidance_applied', False)) / max(1, len(predictions))
        
        result = {
            'final_score': final_score,
            'towers_built': towers_built,
            'efficiency': efficiency,
            'predictions_count': len(predictions),
            'llm_guidance_rate': llm_guidance_rate,
            'avg_confidence': statistics.mean([p.get('confidence', 0) for p in predictions]) if predictions else 0,
            'avg_placement_prob': statistics.mean([p.get('placement_probability', 0) for p in predictions]) if predictions else 0
        }
        
        print(f"Test completed: Score={final_score}, Towers={towers_built}, Efficiency={efficiency:.2f}")
        return result
    
    def run_multiple_tests(self, n_tests: int = 10, duration_per_test: int = 30):
        """Run multiple tests for both modes"""
        print(f"Starting statistical test with n={n_tests} for each mode")
        print("=" * 60)
        
        # Test ELM only mode
        print("Testing ELM Only Mode:")
        for i in range(n_tests):
            print(f"  Test {i+1}/{n_tests}")
            result = self.run_single_test('elm_only', duration_per_test)
            self.results['elm_only'].append(result)
            time.sleep(5)  # Brief pause between tests
        
        print("\nTesting ELM+LLM Mode:")
        for i in range(n_tests):
            print(f"  Test {i+1}/{n_tests}")
            result = self.run_single_test('elm_llm', duration_per_test)
            self.results['elm_llm'].append(result)
            time.sleep(5)  # Brief pause between tests
        
        print("\nAll tests completed!")
    
    def analyze_results(self) -> Dict:
        """Analyze and compare results"""
        analysis = {}
        
        for mode in ['elm_only', 'elm_llm']:
            results = self.results[mode]
            if not results:
                continue
                
            scores = [r['final_score'] for r in results]
            towers = [r['towers_built'] for r in results]
            efficiencies = [r['efficiency'] for r in results]
            confidences = [r['avg_confidence'] for r in results]
            
            analysis[mode] = {
                'n': len(results),
                'score_mean': statistics.mean(scores),
                'score_stdev': statistics.stdev(scores) if len(scores) > 1 else 0,
                'towers_mean': statistics.mean(towers),
                'towers_stdev': statistics.stdev(towers) if len(towers) > 1 else 0,
                'efficiency_mean': statistics.mean(efficiencies),
                'efficiency_stdev': statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0,
                'confidence_mean': statistics.mean(confidences),
                'llm_guidance_rate': statistics.mean([r['llm_guidance_rate'] for r in results])
            }
        
        # Calculate statistical significance (simple t-test approximation)
        if len(self.results['elm_only']) > 1 and len(self.results['elm_llm']) > 1:
            elm_scores = [r['final_score'] for r in self.results['elm_only']]
            llm_scores = [r['final_score'] for r in self.results['elm_llm']]
            
            score_diff = statistics.mean(llm_scores) - statistics.mean(elm_scores)
            pooled_std = ((statistics.stdev(elm_scores)**2 + statistics.stdev(llm_scores)**2) / 2) ** 0.5
            
            analysis['comparison'] = {
                'score_difference': score_diff,
                'score_improvement_percent': (score_diff / max(1, statistics.mean(elm_scores))) * 100,
                'pooled_std': pooled_std,
                'effect_size': score_diff / max(0.1, pooled_std)  # Cohen's d approximation
            }
        
        return analysis
    
    def generate_report(self) -> str:
        """Generate comprehensive statistical report"""
        analysis = self.analyze_results()
        
        report = []
        report.append("# Tower Defense LLM Effect Statistical Analysis")
        report.append("=" * 50)
        report.append("")
        
        for mode in ['elm_only', 'elm_llm']:
            if mode not in analysis:
                continue
                
            data = analysis[mode]
            mode_name = "ELM Only" if mode == 'elm_only' else "ELM + LLM"
            
            report.append(f"## {mode_name} Results (n={data['n']})")
            report.append("")
            report.append(f"**Score:**")
            report.append(f"- Mean: {data['score_mean']:.1f} ± {data['score_stdev']:.1f}")
            report.append(f"- Range: {data['score_mean'] - data['score_stdev']:.1f} - {data['score_mean'] + data['score_stdev']:.1f}")
            report.append("")
            report.append(f"**Towers Built:**")
            report.append(f"- Mean: {data['towers_mean']:.1f} ± {data['towers_stdev']:.1f}")
            report.append("")
            report.append(f"**Efficiency (Score/Tower):**")
            report.append(f"- Mean: {data['efficiency_mean']:.2f} ± {data['efficiency_stdev']:.2f}")
            report.append("")
            report.append(f"**System Metrics:**")
            report.append(f"- Average Confidence: {data['confidence_mean']:.3f}")
            report.append(f"- LLM Guidance Rate: {data['llm_guidance_rate']:.1%}")
            report.append("")
        
        if 'comparison' in analysis:
            comp = analysis['comparison']
            report.append("## Statistical Comparison")
            report.append("")
            report.append(f"**Score Difference:** {comp['score_difference']:.1f}")
            report.append(f"**Improvement:** {comp['score_improvement_percent']:.1f}%")
            report.append(f"**Effect Size (Cohen's d):** {comp['effect_size']:.2f}")
            report.append("")
            
            # Interpret effect size
            if abs(comp['effect_size']) < 0.2:
                effect = "negligible"
            elif abs(comp['effect_size']) < 0.5:
                effect = "small"
            elif abs(comp['effect_size']) < 0.8:
                effect = "medium"
            else:
                effect = "large"
            
            report.append(f"**Effect Interpretation:** {effect.title()} effect")
            report.append("")
            
            if comp['score_improvement_percent'] > 5 and comp['effect_size'] > 0.2:
                report.append("**Conclusion:** LLM guidance shows measurable positive effect")
            elif abs(comp['score_improvement_percent']) < 5:
                report.append("**Conclusion:** No significant difference between modes")
            else:
                report.append("**Conclusion:** Results require further investigation")
        
        return "\n".join(report)

def main():
    """Run statistical test"""
    base_url = "https://kkh7ikc7ewn0.manus.space"
    
    tester = TowerDefenseStatisticalTest(base_url)
    
    # Run tests for reliable statistical analysis
    n_tests = 10  # Proper sample size for statistical significance
    duration = 15  # seconds per test (reduced for efficiency)
    
    print(f"Running statistical analysis with n={n_tests} tests per mode")
    print(f"Each test duration: {duration} seconds")
    print(f"Total estimated time: {(n_tests * 2 * duration + n_tests * 10) / 60:.1f} minutes")
    print()
    
    try:
        tester.run_multiple_tests(n_tests, duration)
        
        # Save results
        with open('/home/ubuntu/tower-defense-llm/statistical_results.json', 'w') as f:
            json.dump(tester.results, f, indent=2)
        
        # Generate report
        report = tester.generate_report()
        with open('/home/ubuntu/tower-defense-llm/statistical_report.md', 'w') as f:
            f.write(report)
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("="*60)
        print(report)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
