## 1) A comparison of machine learning methods for predicting the direction of the US stock market on the basis of volatility indices

**Concepts:** Directional forecasting (up/down), volatility indices (term structure + vol-of-vol), binary classification, ensemble methods, regularization, rolling / out-of-sample (OOS) evaluation, metrics (Accuracy / AUC / F1).

**Core question / contribution**  
Do volatility indices contain predictive information about the **future direction** of the U.S. stock market, and **which ML family** extracts it best?

**Data & target (as described in the paper)**  
- Daily U.S. stock index returns + multiple volatility indices (Jan 2011–Jul 2022). :contentReference[oaicite:0]{index=0}  
- Prediction task: market direction classification; models evaluated via Accuracy, AUC, and F-measure. :contentReference[oaicite:1]{index=1}  

**Machine learning methods emphasized**  
- Linear baselines + regularization (e.g., logistic-style baselines; ridge/lasso-style regularization as comparative tools). :contentReference[oaicite:2]{index=2}  
- Tree/ensemble families (e.g., bagging / random forest / boosting-style ensembles) as main competitors for nonlinearities. :contentReference[oaicite:3]{index=3}  

**How this can become a final project (research direction)**  
- Replicate the volatility-index → direction pipeline on a chosen U.S. index/ETF (SPY), with **rolling OOS** evaluation and classification + trading translation.  
- Extend: (i) add regime splits (crisis vs calm), (ii) compare **probability-calibrated** signals vs hard labels, (iii) interpretability via permutation importance / SHAP on ensemble models.

**Full citation**  
Campisi, G., Muzzioli, S., & De Baets, B. (2024). *A comparison of machine learning methods for predicting the direction of the US stock market on the basis of volatility indices.* **International Journal of Forecasting, 40**(3), 869–880. https://doi.org/10.1016/j.ijforecast.2023.07.002 :contentReference[oaicite:4]{index=4}


---

## 2) Automated trading with boosting and expert weighting

**Concepts:** Algorithmic trading, boosting, online learning / expert weighting, ensemble aggregation, risk overlays, long/short portfolio construction, abnormal returns.

**Core question / contribution**  
Can a trading system built from **boosted “experts”** and an **expert-weighting** aggregation scheme deliver outperformance in a realistic multi-asset setting?

**What the methodology looks like (high level)**  
- Learn trading signals using **boosting** (paper centers on boosted decision-rule structures). :contentReference[oaicite:5]{index=5}  
- Combine multiple learned experts using **expert weighting** (online allocation across experts). :contentReference[oaicite:6]{index=6}  

**Machine learning methods emphasized**  
- Boosting-based classifier family (paper frames boosting as the core learning engine). :contentReference[oaicite:7]{index=7}  
- Expert-weighting (online learning aggregation) as the meta-layer over multiple experts. :contentReference[oaicite:8]{index=8}  

**How this can become a final project (research direction)**  
- Build a two-layer system:  
  1) Base learners (e.g., boosting / trees / logistic) trained on a fixed feature set (technical indicators, volatility features).  
  2) A meta “expert weighting” layer that reallocates capital across learners through time (online learning), plus a drawdown-aware risk switch.  
- Report both **ML metrics** and **trading metrics** (Sharpe, alpha/beta, max drawdown) under strict OOS design.

**Full citation**  
Creamer, G., & Freund, Y. (2010). *Automated trading with boosting and expert weighting.* **Quantitative Finance, 10**(4), 401–420. https://doi.org/10.1080/14697680903104113 :contentReference[oaicite:9]{index=9}


---

## 3) Stock index futures price prediction using feature selection and deep learning

**Concepts:** Futures price forecasting, feature selection, deep learning sequence models, technical indicators, benchmarking vs classical ML, OOS testing.

**Core question / contribution**  
Does combining **feature selection** with **deep learning** improve stock index futures price prediction relative to standard ML and alternative deep architectures?

**Data & target (as described in the paper)**  
- Stock index futures dataset (CSI300 index futures context), daily horizon; paper is positioned as “price prediction” with feature engineering + selection + deep models. :contentReference[oaicite:10]{index=10}  

**Machine learning methods emphasized**  
- **Feature selection** stage + a **deep learning** predictor (paper’s main “hybrid” claim). :contentReference[oaicite:11]{index=11}  
- Benchmarking against other ML/DL baselines (the paper explicitly frames comparative evaluation). :contentReference[oaicite:12]{index=12}  

**How this can become a final project (research direction)**  
- Recreate the pipeline on a liquid futures proxy (e.g., ES, NQ) or ETF data if futures data is unavailable:  
  - Stage 1: feature selection (AdaBoost-based, or modern alternatives like SHAP-based selection).  
  - Stage 2: sequence model (LSTM/GRU) vs tabular ensembles (RF/GB).  
- Extension: replace price-level prediction with **return / direction** prediction and evaluate trading outcomes (Sharpe/MDD), aligning it with Paper (1) and (2).

**Full citation**  
Yan, W.-L. (2023). *Stock index futures price prediction using feature selection and deep learning.* **The North American Journal of Economics and Finance, 64**, 101867. https://doi.org/10.1016/j.najef.2022.101867 :contentReference[oaicite:13]{index=13}


---
