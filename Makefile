.PHONY: install train-dqn train-qrl evaluate plot test clean

install:
	pip install -r requirements.txt

train-dqn:
	python train.py --agent dqn --episodes 500

train-qrl:
	python train.py --agent qrl --episodes 200

evaluate:
	python evaluate.py

plot:
	python plot_results.py

test:
	python -m pytest tests/ -v

clean:
	rm -rf __pycache__ vec_env/__pycache__ agents/__pycache__ tests/__pycache__
	rm -f checkpoints/dqn.pt checkpoints/qrl.npz
	rm -f results/dqn_returns.npy results/qrl_returns.npy results/metrics.csv
	rm -f results/plots/*.png
	rm -f logs/*.log
