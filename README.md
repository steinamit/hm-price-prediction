# H&M Product Price Prediction

> Can a fashion product's price be estimated from what is known before it sells: its catalogue attributes, its description and its photo? And which of the three carries the most signal?

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B)
![CLIP](https://img.shields.io/badge/CLIP-image%20embeddings-412991)
![Colab](https://img.shields.io/badge/Colab-F9AB00?logo=googlecolab&logoColor=white)

## Highlights

- One price per product, built from 31.8M transactions (median price, products with at least 5 sales).
- Train/test split **by `product_code`**, so colour variants of the same item never appear on both sides.
- Five approaches scored on the same products: a category baseline, catalogue attributes, description text, product photo, and a model that combines all three.
- Explainability: which words push a predicted price up or down, error by product group, and feature importance of the combined model.

## Data

The data comes from the Kaggle competition **[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)** (2022). It is not included in this repository: to run the notebook, accept the competition rules on Kaggle and download it there.

| File | What it holds | Used for |
|---|---|---|
| `articles.csv` | ~105K products: type, group, colour, department, section, name and a free-text description | features |
| `transactions_train.csv` | 31.8M purchases, Sep 2018 – Sep 2020, with the price paid | target (median price per product) |
| `images/` | one photo per product, `images/010/0108775015.jpg` | image features |

Prices in the dataset are scaled by H&M and are not in a real currency, so results are reported as percentage error (MAPE) and R² on log price.

## Approach

| Step | Method |
|---|---|
| Baseline | Median price of the product type |
| Attributes | One-hot + Ridge; LightGBM on the categorical attributes |
| Text | TF-IDF (1–2 grams) + Ridge; MiniLM sentence embeddings + Ridge |
| Image | CLIP ViT-B/32 embeddings + Ridge |
| Combined | LightGBM on attributes + PCA-compressed text and image embeddings (32 dims each, fitted on train only) |

All models predict log price and are scored on the same held-out products.

## Getting started

Open `hm_price_prediction.ipynb` in Google Colab with a GPU runtime, set `DATA_DIR` and `IMAGE_ROOT` in the first cell to where the Kaggle files are, and run all. Results and charts are written to `OUT_DIR`. `N_PRODUCTS` controls the sample size, since reading images is the slow step.

## Structure

| File | Purpose |
|---|---|
| `hm_price_prediction.ipynb` | Full pipeline: price per product, split, models, results |
| `build_nb.py` | Generates the notebook from source |

## Credits

Joint project by Amit Stein and Tair Maimon.
