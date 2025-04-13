echo "--------------Starting baseline run----------------"
python src/baseline_run.py
echo "--------------Starting improved run----------------"
python src/improved_run_v3.py
echo "--------------Comparing results----------------"
python src/compare_results.py
echo "All done"