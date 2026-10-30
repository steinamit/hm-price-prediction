"""Builds hm_price_prediction.ipynb from the cell list below."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Predicting H&M product prices from attributes, text and images

**Question.** How well can a product's price be estimated *before it sells*, using only what is known at design time:
its catalogue attributes, its product description, and its photo? And which of the three carries the most signal?

**Data.** [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (Kaggle):
`articles.csv` (product catalogue), `transactions_train.csv` (31.8M purchases, prices are scaled by H&M) and the product images.

**Approach.**
1. Build one price per product: the median transaction price, keeping products with enough sales for a stable estimate.
2. Split train/test **by `product_code`**, so colour variants of the same product never sit on both sides.
3. Compare, on the same test set: a category baseline, tabular models, text models, an image model, and a model that combines all three.

Metrics: MAPE (scale-free, since H&M prices are anonymised), MAE on the scaled price, and R² on log price.
""")

md("## 1. Setup")
code("""
# Settings: change these to match your Drive layout
DATA_DIR   = '/content/drive/MyDrive'          # folder with articles.csv and transactions_train.csv
IMAGE_ROOT = '/content/drive/MyDrive/לימודים ותזה/לבדיקה - משקולות ונתוני אימון ישנים'  # holds images/ or extracted_images*/ folders (Kaggle layout: 010/0108775015.jpg)
MIN_SALES  = 5        # keep products with at least this many transactions
N_PRODUCTS = 20000    # products used for the comparison (image embedding is the slow step)
USE_TEXT_EMB  = True  # sentence-transformer embeddings of the description
USE_IMAGES    = True  # CLIP image embeddings
SEED = 42
OUT_DIR = '/content/drive/MyDrive/hm_price_results'   # results.json and charts are saved here
""")
code("""
import os, json, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
""")

md("## 2. One price per product")
code("""
t0 = time.time()
tx = pd.read_csv(os.path.join(DATA_DIR, 'transactions_train.csv'),
                 usecols=['article_id', 'price'], dtype={'article_id': 'int64', 'price': 'float32'})
print(f'{len(tx):,} transactions loaded in {time.time()-t0:.0f}s')

prices = tx.groupby('article_id')['price'].agg(price='median', n_sales='size').reset_index()
del tx
print(f'{len(prices):,} products with sales')
prices = prices[prices.n_sales >= MIN_SALES]
print(f'{len(prices):,} products with at least {MIN_SALES} sales')
""")
code("""
art = pd.read_csv(os.path.join(DATA_DIR, 'articles.csv'), dtype={'article_id': 'int64'})
df = art.merge(prices, on='article_id', how='inner')
df['detail_desc'] = df['detail_desc'].fillna('')
df['log_price'] = np.log(df['price'])
print(df.shape)
df[['prod_name', 'product_type_name', 'index_group_name', 'price', 'n_sales']].head()
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(df['price'], bins=80, color='#2747D9'); ax[0].set_title('Price (scaled)')
ax[1].hist(df['log_price'], bins=80, color='#2747D9'); ax[1].set_title('log(price)')
plt.tight_layout(); plt.show()

(df.groupby('index_group_name')['price'].median().sort_values()
   .plot.barh(figsize=(6, 3), color='#2747D9', title='Median price by index group'))
plt.show()
""")

md("""
## 3. Sample and split

Images are read from Drive one by one, which is the slow step, so the comparison runs on a random sample of products
(set by `N_PRODUCTS`). When images are on, only products that have a photo are kept, so every model is scored on exactly
the same products.
""")
code("""
# index every product photo under IMAGE_ROOT (the images may be split across several folders)
IMG = {}
if USE_IMAGES and os.path.isdir(IMAGE_ROOT):
    for top in sorted(os.listdir(IMAGE_ROOT)):
        top_dir = os.path.join(IMAGE_ROOT, top)
        if not (top == 'images' or top.startswith('extracted_images')) or not os.path.isdir(top_dir):
            continue
        for sub in os.listdir(top_dir):
            sub_dir = os.path.join(top_dir, sub)
            if os.path.isdir(sub_dir):
                for f in os.listdir(sub_dir):
                    if f.endswith('.jpg'):
                        IMG.setdefault(int(f[:-4]), os.path.join(sub_dir, f))
print(f'{len(IMG):,} product photos indexed')
image_path = IMG.get

has_img = df['article_id'].isin(IMG.keys())
print(f'{has_img.mean():.0%} of priced products have a photo')
if USE_IMAGES and has_img.sum() < 1000:
    print('Too few photos found: check IMAGE_ROOT. Continuing without images.')
    USE_IMAGES = False

data = df.sample(frac=1, random_state=SEED)
if USE_IMAGES:
    data = data[data['article_id'].isin(IMG.keys())].head(N_PRODUCTS)
else:
    data = data.head(N_PRODUCTS)
data = data.reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(data, groups=data['product_code']))
train, test = data.iloc[tr_idx].reset_index(drop=True), data.iloc[te_idx].reset_index(drop=True)
assert not set(train.product_code) & set(test.product_code)
print(f'train {len(train):,} products | test {len(test):,} products | no product_code in both')
""")

md("## 4. Models")
code("""
results, preds = {}, {}
def score(name, log_pred):
    p = np.exp(log_pred); y = test['price'].values
    results[name] = {
        'MAPE_%': float(np.mean(np.abs(p - y) / y) * 100),
        'MAE_x1000': float(mean_absolute_error(y, p) * 1000),
        'R2_log': float(r2_score(test['log_price'], log_pred)),
    }
    preds[name] = log_pred
    print(f"{name:32s} MAPE {results[name]['MAPE_%']:5.1f}%   R2(log) {results[name]['R2_log']:.3f}")
""")

md("### 4.1 Baseline: median price of the product type")
code("""
type_med = train.groupby('product_type_name')['log_price'].median()
base = test['product_type_name'].map(type_med).fillna(train['log_price'].median())
score('Baseline (product type median)', base.values)
""")

md("### 4.2 Tabular: catalogue attributes")
code("""
CAT = ['product_type_name', 'product_group_name', 'graphical_appearance_name', 'colour_group_name',
       'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name', 'index_name',
       'index_group_name', 'section_name', 'garment_group_name']

ohe = OneHotEncoder(handle_unknown='ignore', min_frequency=5)
Xtr_ohe, Xte_ohe = ohe.fit_transform(train[CAT]), ohe.transform(test[CAT])
score('Tabular · Ridge', Ridge(alpha=3.0).fit(Xtr_ohe, train['log_price']).predict(Xte_ohe))

def as_cat(frame):
    out = frame[CAT].copy()
    for c in CAT:
        out[c] = pd.Categorical(out[c], categories=sorted(train[c].unique()))
    return out
LGB = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)
gbm_tab = lgb.LGBMRegressor(**LGB).fit(as_cat(train), train['log_price'])
score('Tabular · LightGBM', gbm_tab.predict(as_cat(test)))
""")

md("### 4.3 Text: product name and description")
code("""
txt = lambda f: (f['prod_name'] + '. ' + f['detail_desc']).str.lower()
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
Xtr_tf, Xte_tf = tfidf.fit_transform(txt(train)), tfidf.transform(txt(test))
score('Text · TF-IDF + Ridge', Ridge(alpha=1.0).fit(Xtr_tf, train['log_price']).predict(Xte_tf))

# words that push the predicted price up or down
r = Ridge(alpha=1.0).fit(Xtr_tf, train['log_price'])
vocab = np.array(tfidf.get_feature_names_out())
order = np.argsort(r.coef_)
print('cheaper:', ', '.join(vocab[order[:15]]))
print('pricier:', ', '.join(vocab[order[-15:]][::-1]))
""")
code("""
if USE_TEXT_EMB:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer('all-MiniLM-L6-v2')
    E_tr = st.encode(txt(train).tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    E_te = st.encode(txt(test).tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    score('Text · MiniLM embeddings + Ridge', Ridge(alpha=1.0).fit(E_tr, train['log_price']).predict(E_te))
""")

md("### 4.4 Image: CLIP embedding of the product photo")
code("""
if USE_IMAGES:
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    clip = SentenceTransformer('clip-ViT-B-32', device='cuda' if torch.cuda.is_available() else 'cpu')

    def embed_images(ids, bs=64):
        out = []
        for i in range(0, len(ids), bs):
            imgs = [Image.open(image_path(a)).convert('RGB') for a in ids[i:i + bs]]
            out.append(clip.encode(imgs, batch_size=bs, normalize_embeddings=True))
            if i % (bs * 50) == 0:
                print(f'{i:,}/{len(ids):,}')
        return np.vstack(out)

    I_tr, I_te = embed_images(train['article_id'].tolist()), embed_images(test['article_id'].tolist())
    score('Image · CLIP + Ridge', Ridge(alpha=1.0).fit(I_tr, train['log_price']).predict(I_te))
""")

md("""
### 4.5 Combined: attributes + text + image

LightGBM on the catalogue attributes plus compressed text and image embeddings (PCA, 32 dimensions each,
fitted on train only).
""")
code("""
def fused(frame, parts):
    return pd.concat([as_cat(frame).reset_index(drop=True)] + parts, axis=1)

parts_tr, parts_te = [], []
if USE_TEXT_EMB:
    p = PCA(32, random_state=SEED).fit(E_tr)
    parts_tr.append(pd.DataFrame(p.transform(E_tr), columns=[f'txt_{i}' for i in range(32)]))
    parts_te.append(pd.DataFrame(p.transform(E_te), columns=[f'txt_{i}' for i in range(32)]))
if USE_IMAGES:
    p = PCA(32, random_state=SEED).fit(I_tr)
    parts_tr.append(pd.DataFrame(p.transform(I_tr), columns=[f'img_{i}' for i in range(32)]))
    parts_te.append(pd.DataFrame(p.transform(I_te), columns=[f'img_{i}' for i in range(32)]))

if parts_tr:
    gbm_all = lgb.LGBMRegressor(**LGB).fit(fused(train, parts_tr), train['log_price'])
    score('Combined · LightGBM', gbm_all.predict(fused(test, parts_te)))
""")

md("## 5. Results")
code("""
res = pd.DataFrame(results).T.sort_values('MAPE_%')
res.style.format({'MAPE_%': '{:.1f}', 'MAE_x1000': '{:.2f}', 'R2_log': '{:.3f}'})
""")
code("""
ax = res['MAPE_%'].sort_values(ascending=False).plot.barh(figsize=(7, 3.5), color='#2747D9')
ax.set_xlabel('MAPE on test products (%), lower is better'); ax.set_title('Price error by model')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'mape_by_model.png'), dpi=150); plt.show()
""")
code("""
best = res.index[0]
err = test.assign(ape=np.abs(np.exp(preds[best]) - test['price']) / test['price'] * 100)
(err.groupby('product_group_name')['ape'].agg(['median', 'size'])
    .query('size >= 30').sort_values('median').round(1).rename(columns={'median': 'median APE %', 'size': 'products'}))
""")
code("""
model = gbm_all if parts_tr else gbm_tab
imp = pd.Series(model.booster_.feature_importance('gain'), index=model.booster_.feature_name())
group = imp.groupby(lambda c: 'text embedding' if c.startswith('txt_') else 'image embedding' if c.startswith('img_') else c).sum()
(group / group.sum() * 100).sort_values().tail(12).plot.barh(figsize=(7, 4), color='#2747D9', title='Share of gain (%)')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'feature_importance.png'), dpi=150); plt.show()
""")
code("""
summary = {'n_products': len(data), 'n_train': len(train), 'n_test': len(test),
           'min_sales': MIN_SALES, 'images': USE_IMAGES, 'text_embeddings': USE_TEXT_EMB, 'results': results}
json.dump(summary, open(os.path.join(OUT_DIR, 'results.json'), 'w'), indent=2)
print(json.dumps(summary, indent=2))
""")

md("""
## 6. Notes

- Prices in this dataset are scaled by H&M, so absolute values are not in a real currency; MAPE and R² are unaffected.
- A product's price is the median over its transactions, which smooths out discounts but does not remove them.
- The split is by `product_code`. A random split would put colour variants of the same item in both train and test and
  make every model look better than it is.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
               'language_info': {'name': 'python'}, 'accelerator': 'GPU',
               'colab': {'provenance': [], 'gpuType': 'T4'}}
nbf.write(nb, 'hm_price_prediction.ipynb')
print('written', len(cells), 'cells')
