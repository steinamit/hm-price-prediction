"""Builds hm_price_prediction.ipynb from the cell list below."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Predicting Fashion Product Prices from Attributes, Text and Images

**Authors:** Amit Stein, Tair Maimon
**Data:** H&M Personalized Fashion Recommendations (Kaggle, 2022)

---

## Abstract

A retailer has to price a new product before it has sold a single unit. This project asks how much of a product's
price can be recovered from what is known at design time: its catalogue attributes, its written description and its
photo. We derive one price per product from 31.8 million H&M transactions, split products into train and test sets so
that colour variants of the same item never appear on both sides, and compare five approaches on the same held-out
products: a category baseline, attribute models, text models, an image model, and a model that combines all three.
Results and their interpretation are in Sections 8 and 9.

## Contents

1. Research questions
2. Data
3. Setup
4. Target: one price per product
5. Sampling and train/test split
6. Evaluation
7. Models
8. Results
9. Discussion and limitations
10. References
""")

md("""
## 1. Research questions

| | Question |
|---|---|
| **RQ1** | How accurately can a product's price be predicted from its catalogue attributes alone? |
| **RQ2** | Does the product text (name and description) carry price information beyond the attributes? |
| **RQ3** | Does the product photo carry price information beyond the attributes and the text? |
| **RQ4** | Does combining all three sources improve on the best single source? |

Every model is compared with a simple baseline, the median price of the product type, so that each gain is measured
against what a merchandiser could do with a lookup table.
""")

md("""
## 2. Data

The data comes from the Kaggle competition
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
It is not redistributed with this notebook; it can be downloaded from Kaggle after accepting the competition rules.

| File | Content | Role in this project |
|---|---|---|
| `transactions_train.csv` | 31.8M purchases, Sep 2018 to Sep 2020, with the price paid | target |
| `articles.csv` | ~105K products with catalogue attributes and a free-text description | features |
| `images/` | one photo per product, stored as `images/010/0108775015.jpg` | image features |

**Catalogue attributes used as features**

| Column | Meaning | Example |
|---|---|---|
| `product_type_name` | product type | Trousers, Sweater |
| `product_group_name` | broad product group | Garment Upper body, Underwear |
| `graphical_appearance_name` | pattern | Solid, Stripe |
| `colour_group_name` | colour | Black, Light Blue |
| `perceived_colour_value_name` / `perceived_colour_master_name` | colour tone and family | Dark / Blue |
| `department_name` | department | Jersey Basic |
| `index_name` / `index_group_name` | target customer segment | Ladieswear, Baby/Children |
| `section_name` | section | Womens Everyday Basics |
| `garment_group_name` | garment group | Knitwear, Jersey Basic |
| `prod_name` + `detail_desc` | product name and description (text) | "Jersey top with narrow shoulder straps." |

Prices in the dataset are scaled by H&M, so they are not in a real currency. This does not affect the comparison:
all metrics below are either relative (MAPE) or computed on log price (R²).
""")

md("""
## 3. Setup

Paths and run settings. `N_PRODUCTS` controls the sample size, because reading product photos from Google Drive is
the slowest step.
""")
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

md("""
## 4. Target: one price per product

A product is sold many times, sometimes at a discount, so its price is defined as the **median** transaction price.
The median is robust to the occasional sale or promotion. Products with fewer than `MIN_SALES` transactions are
dropped, because a price estimated from one or two sales is unreliable.

The model predicts **log(price)**. Prices are right-skewed, and on the log scale an error of 10% counts the same for a
cheap sock and an expensive coat.
""")
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
## 5. Sampling and train/test split

**Sample.** The comparison runs on a random sample of `N_PRODUCTS` products. When images are used, only products that
have a photo are kept, so every model is scored on exactly the same products.

**Split by product, not by row.** H&M sells the same item in several colours. Each colour is a separate `article_id`,
but all colours share one `product_code` and usually one price. A random split would put the black and the white
version of the same shirt on opposite sides, and the model would "predict" a price it has effectively already seen.
We therefore split with `GroupShuffleSplit` on `product_code` (80% train, 20% test), and assert that no product code
appears in both sets.
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

md(r"""
## 6. Evaluation

All models are trained on the same training products and scored on the same test products.

| Metric | Definition | Why |
|---|---|---|
| **MAPE** | $\frac{100}{n}\sum_i \frac{\lvert \hat{p}_i - p_i \rvert}{p_i}$ | average error in percent; unaffected by H&M's price scaling |
| **MAE** | $\frac{1}{n}\sum_i \lvert \hat{p}_i - p_i \rvert$ (shown ×1000) | average error on the scaled price |
| **R² (log)** | $1 - \frac{\sum (\log p_i - \log \hat{p}_i)^2}{\sum (\log p_i - \overline{\log p})^2}$ | share of price variation explained |

Here $p_i$ is the product's median price and $\hat{p}_i = e^{\widehat{\log p_i}}$ the model's prediction.
""")

md("## 7. Models")
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

md("""
### 7.1 Baseline: median price of the product type

Each test product gets the median log price of its product type in the training set (the overall median for an unseen
type). This is what a merchandiser could do with a lookup table, and every other model has to beat it to be useful.
""")
code("""
type_med = train.groupby('product_type_name')['log_price'].median()
base = test['product_type_name'].map(type_med).fillna(train['log_price'].median())
score('Baseline (product type median)', base.values)
""")

md("""
### 7.2 Attributes (RQ1)

Two models on the eleven catalogue attributes:

- **Ridge regression on one-hot encoded attributes.** A linear model: each attribute value adds or subtracts a fixed
  percentage from the price. Rare values (fewer than 5 products) are grouped together.
- **LightGBM with native categorical features.** Gradient-boosted trees that can capture interactions, such as a
  premium that applies to knitwear only in Ladieswear.
""")
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

md("""
### 7.3 Text (RQ2)

The product name and description often name the material and the cut ("cashmere", "wool blend", "padded"), which
the structured attributes do not record.

- **TF-IDF + Ridge.** Word and two-word frequencies, weighted by how distinctive they are. Its coefficients show which
  words push the predicted price up or down.
- **Sentence embeddings + Ridge.** `all-MiniLM-L6-v2` turns each description into a 384-dimensional vector that
  captures meaning rather than exact words.
""")
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

md("""
### 7.4 Image (RQ3)

Each product photo is encoded with **CLIP ViT-B/32**, a model trained to match images with text, into a 512-dimensional
vector, followed by Ridge regression. The photo can show what neither the attributes nor the text record: fabric
texture, finish and overall look.
""")
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
### 7.5 Combined (RQ4)

LightGBM on the catalogue attributes plus the text and image embeddings, each compressed to 32 dimensions with PCA.
The PCA is fitted on the training set only, so no information from the test products leaks into the features.
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

md("""
## 8. Results

### 8.1 All models on the same test products
""")
code("""
res = pd.DataFrame(results).T.sort_values('MAPE_%')
res.style.format({'MAPE_%': '{:.1f}', 'MAE_x1000': '{:.2f}', 'R2_log': '{:.3f}'})
""")
code("""
ax = res['MAPE_%'].sort_values(ascending=False).plot.barh(figsize=(7, 3.5), color='#2747D9')
ax.set_xlabel('MAPE on test products (%), lower is better'); ax.set_title('Price error by model')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'mape_by_model.png'), dpi=150); plt.show()
""")
md("""
### 8.2 Answers to the research questions

Computed directly from the results above, so the text always matches the numbers.
""")
code("""
def mape(name): return results[name]['MAPE_%'] if name in results else None
def best_of(prefix):
    names = [n for n in results if n.startswith(prefix)]
    return min(names, key=mape) if names else None

base_name = 'Baseline (product type median)'
b = mape(base_name)
lines = [f'Baseline (median price of the product type): MAPE {b:.1f}%.']
for rq, prefix, label in [('RQ1', 'Tabular', 'using the attributes'), ('RQ2', 'Text', 'using the text'),
                          ('RQ3', 'Image', 'using the photo'), ('RQ4', 'Combined', 'combining all three')]:
    n = best_of(prefix)
    if n is None:
        lines.append(f'{rq}: not run in this session.')
        continue
    m = mape(n); d = b - m
    vs = f'{abs(d):.1f} points {"better" if d > 0 else "worse"} than the baseline'
    lines.append(f'{rq}: best model {label} is "{n}", MAPE {m:.1f}% ({vs}; R² on log price {results[n]["R2_log"]:.3f}).')
singles = [best_of(p) for p in ('Tabular', 'Text', 'Image') if best_of(p)]
top = min(singles, key=mape)
lines.append(f'Strongest single source: "{top}".')
if best_of('Combined'):
    gain = mape(top) - mape(best_of('Combined'))
    lines.append(f'Combining the sources is {abs(gain):.1f} MAPE points {"better" if gain > 0 else "worse"} than the strongest single source.')
print('\\n'.join(lines))
""")

md("""
### 8.3 Where the best model is wrong
""")
code("""
best = res.index[0]
err = test.assign(ape=np.abs(np.exp(preds[best]) - test['price']) / test['price'] * 100)
(err.groupby('product_group_name')['ape'].agg(['median', 'size'])
    .query('size >= 30').sort_values('median').round(1).rename(columns={'median': 'median APE %', 'size': 'products'}))
""")
md("""
### 8.4 What drives the prediction

Share of the LightGBM model's total gain contributed by each attribute, with the text and image embedding dimensions
grouped together.
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
## 9. Discussion and limitations

**Reading the results.** The gap between the baseline and the attribute models (RQ1) shows how much of the price is
explained by what kind of product it is and who it is for. The gaps between the attribute models and the text and image
models (RQ2, RQ3) show how much extra information sits in the description and the photo. If the combined model (RQ4)
improves only slightly on the best single source, the sources largely overlap: the description and the photo mostly
repeat what the catalogue already says.

**Limitations.**

- **The target is a realised price, not a list price.** The median smooths out discounts but does not remove them;
  products that were heavily promoted will look cheaper than their original price.
- **Prices are scaled.** H&M anonymised prices with an unknown scaling factor, so absolute errors cannot be read in a
  real currency. Relative metrics are unaffected.
- **Sample.** Image embedding is the slow step, so the comparison runs on a sample of products rather than the full
  catalogue. Results on the full catalogue may differ slightly.
- **One split.** Results come from a single grouped train/test split. Repeating the split with different seeds would
  give a range for each metric.
- **Frozen encoders.** The text and image models are used as fixed feature extractors and are not fine-tuned on prices.

**Next steps.** Repeat the evaluation over several seeds and report confidence intervals; fine-tune the image encoder
on price; add time features (season, launch date) to separate a product's intrinsic price from the timing of its sales.

## 10. References

1. H&M Group. *H&M Personalized Fashion Recommendations* (dataset). Kaggle, 2022.
   https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
2. Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML, 2021.
3. Reimers, N. and Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP, 2019.
4. Wang, W. et al. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS, 2020.
5. Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS, 2017.
6. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR 12, 2011.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
               'language_info': {'name': 'python'}, 'accelerator': 'GPU',
               'colab': {'provenance': [], 'gpuType': 'T4'}}
nbf.write(nb, 'hm_price_prediction.ipynb')
print('written', len(cells), 'cells')
