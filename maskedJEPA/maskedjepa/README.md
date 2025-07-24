<h2>🧠 VJEPA + Adaptive Masking</h2>

<p>Codebase for <strong>VJEPA</strong> with <strong>Adaptive Masking</strong>.</p>

<h3>⚙️ Setup Instructions</h3>

<pre><code># 1. Create and activate the conda environment
conda create -n vjepa python=3.9.23
conda activate vjepa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package
python setup.py install
</code></pre>

<h3>🚀 Run Evaluation</h3>

<pre><code>python -m evals.main \
  --fname configs/eval/vith16_in1k.yaml \
  --devices cuda:0 cuda:1 cuda:2
</code></pre>
