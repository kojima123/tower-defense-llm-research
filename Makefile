# Tower Defense ELM+LLM Research Makefile
# 実測データ専用システムの簡単実行

.PHONY: help install validate run-elm run-llm run-all analyze visualize update-readme clean

# デフォルトターゲット
help:
	@echo "Tower Defense ELM+LLM Research - 実測データ専用システム"
	@echo ""
	@echo "利用可能なコマンド:"
	@echo "  make install        - 依存関係をインストール"
	@echo "  make validate       - データ品質検証 (合成データ検出)"
	@echo "  make run-elm        - ELM単体実験実行"
	@echo "  make run-llm        - ELM+LLM実験実行"
	@echo "  make run-all        - 4条件比較実験実行"
	@echo "  make analyze        - 実測データ分析"
	@echo "  make visualize      - LLM効果可視化"
	@echo "  make cost-analysis  - コスト・遅延分析"
	@echo "  make update-readme  - README自動更新"
	@echo "  make ablation       - 統一アブレーション実験"
	@echo "  make clean          - 一時ファイル削除"
	@echo ""
	@echo "実験パラメータ:"
	@echo "  EPISODES=10         - エピソード数 (デフォルト: 10)"
	@echo "  SEEDS='42 123 456'  - シード集合 (デフォルト: 42 123 456)"
	@echo ""
	@echo "使用例:"
	@echo "  make run-all EPISODES=20"
	@echo "  make run-llm EPISODES=5 SEEDS='42 123'"

# 設定
EPISODES ?= 10
SEEDS ?= 42 123 456
PYTHON = python3

# 依存関係インストール
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt || pip install numpy pandas matplotlib seaborn scipy

# データ品質検証
validate:
	@echo "🔬 Validating data quality (synthetic data detection)..."
	$(PYTHON) validate_real_data.py
	@echo "🔍 Running integrated validation..."
	@$(PYTHON) -c "from logger import validate_experiment_integrity; import json; result = validate_experiment_integrity('.'); print('✅ VALIDATION PASSED' if result['validation_passed'] else '❌ VALIDATION FAILED'); print(f'Data quality score: {result[\"data_quality_score\"]}/100')"

# ELM単体実験
run-elm:
	@echo "🚀 Running ELM-only experiment..."
	@echo "📊 Episodes: $(EPISODES), Seeds: $(SEEDS)"
	$(PYTHON) run_experiment_cli_fixed.py run --teachers elm_only --episodes $(EPISODES) --seeds $(SEEDS)

# ELM+LLM実験
run-llm:
	@echo "🤖 Running ELM+LLM experiment..."
	@echo "📊 Episodes: $(EPISODES), Seeds: $(SEEDS)"
	@echo "⚠️  Requires OPENAI_API_KEY environment variable"
	$(PYTHON) run_experiment_cli_fixed.py run --teachers elm_llm --episodes $(EPISODES) --seeds $(SEEDS)

# 4条件比較実験
run-all:
	@echo "🔬 Running 4-condition comparison experiment..."
	@echo "📊 Episodes: $(EPISODES), Seeds: $(SEEDS)"
	$(PYTHON) run_experiment_cli_fixed.py run --teachers all --episodes $(EPISODES) --seeds $(SEEDS)

# 実測データ分析
analyze:
	@echo "📊 Analyzing real measurement data..."
	$(PYTHON) analyze_real_data.py runs/real/

# LLM効果可視化
visualize:
	@echo "📈 Creating LLM impact visualizations..."
	$(PYTHON) visualize_llm_impact.py

# コスト・遅延分析
cost-analysis:
	@echo "💰 Analyzing cost and latency..."
	$(PYTHON) analyze_cost_latency.py

# README自動更新
update-readme:
	@echo "📝 Updating README from real measurement data..."
	$(PYTHON) auto_update_readme.py

# 統一アブレーション実験
ablation:
	@echo "🧪 Running unified ablation experiment..."
	@echo "📊 Episodes: $(EPISODES), Seeds: $(SEEDS)"
	$(PYTHON) run_unified_ablation.py --episodes $(EPISODES) --seeds $(SEEDS)

# 完全パイプライン
full-pipeline: validate run-all analyze visualize cost-analysis update-readme
	@echo "🎉 Full pipeline completed!"
	@echo "📁 Check generated files:"
	@echo "  - README.md (updated)"
	@echo "  - llm_*.png (visualizations)"
	@echo "  - cost_latency_*.* (cost analysis)"
	@echo "  - runs/real/ (experiment logs)"

# クリーンアップ
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.tmp" -delete
	find . -name "*.temp" -delete
	find . -name "temp_*" -delete
	find . -name "tmp_*" -delete
	rm -rf test_output/ 2>/dev/null || true

# 開発用ターゲット
dev-setup: install validate
	@echo "🔧 Development environment setup complete"
	@echo "✅ Ready for experiments"

# テスト実行
test:
	@echo "🧪 Running quick test experiments..."
	$(PYTHON) run_experiment_cli_fixed.py run --teachers elm_only --episodes 2 --seeds 42 --dry-run
	@echo "✅ Test completed"

# 統計情報表示
stats:
	@echo "📊 Project statistics:"
	@echo "Real data files: $$(find runs/real -name '*.csv' | wc -l)"
	@echo "Total steps logged: $$(find runs/real -name '*.csv' -exec wc -l {} + | tail -1 | awk '{print $$1-NR+1}')"
	@echo "Synthetic data files (isolated): $$(find sim/synthetic_data_deprecated -name '*.json' | wc -l)"
	@echo "Data quality score: $$(python validate_real_data.py 2>/dev/null | grep 'Data quality score' | awk '{print $$4}' || echo 'N/A')"

# ヘルプの詳細版
help-detailed:
	@echo "Tower Defense ELM+LLM Research - 詳細ヘルプ"
	@echo ""
	@echo "🔬 科学的厳密性の保証:"
	@echo "  - 実測データのみ使用 (合成データ完全排除)"
	@echo "  - 固定シード実験による完全再現可能性"
	@echo "  - 統計的妥当性 (scipy.stats使用)"
	@echo "  - 透明性確保 (全実験ログ公開)"
	@echo ""
	@echo "📊 実験条件:"
	@echo "  - elm_only: ELM単体学習"
	@echo "  - rule_teacher: ルールベース教師"
	@echo "  - random_teacher: ランダム教師"
	@echo "  - elm_llm: LLM指導付きELM学習"
	@echo ""
	@echo "🚀 推奨ワークフロー:"
	@echo "  1. make validate      # データ品質確認"
	@echo "  2. make run-all       # 4条件実験実行"
	@echo "  3. make analyze       # 統計分析"
	@echo "  4. make visualize     # 可視化生成"
	@echo "  5. make update-readme # 結果反映"
	@echo ""
	@echo "💡 Tips:"
	@echo "  - OPENAI_API_KEY環境変数を設定してLLM実験を有効化"
	@echo "  - runs/real/ディレクトリに実測ログが保存される"
	@echo "  - sim/synthetic_data_deprecated/に問題ファイルを隔離済み"
