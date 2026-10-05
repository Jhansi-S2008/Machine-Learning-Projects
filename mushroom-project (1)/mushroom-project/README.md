# Mushroom Edibility Predictor

```bash
pip install -r requirements.txt
python train_model.py   # optional: retrains and rewrites model.pkl
python app.py           # open http://127.0.0.1:5000
```
Uses 6 features (odor, spore print color, gill size, stalk root, stalk surface below ring, habitat) and a Decision Tree with max_depth=3.
