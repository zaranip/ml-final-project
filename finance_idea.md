## 1) Text-based PEAD from earnings call language (beyond numeric EPS surprise)
**Paper:** *PEAD.txt: Post-Earnings-Announcement Drift Using Text* (Meursault et al., 2021, Philadelphia Fed Working Paper)  
**Idea:** Use ML/NLP on earnings-call transcripts (or related text) to build a **textual surprise** score that captures information the market underreacts to.  
**Trading rule:** After an earnings announcement, sort firms by the textual surprise score and run **long top quantile / short bottom quantile** (or long-only top) over the post-announcement window to capture drift.

---

## 2) CNN on “visualized earnings history” (learning patterns from charts)
**Paper:** *Visualizing Earnings to Predict Post-Earnings Announcement Drift: A Deep Learning Approach* (Garfinkel, Hribar, Hsiao, working paper)  
**Idea:** Convert a firm’s multi-quarter earnings history (and possibly earnings quality) into **chart-like images**; train a CNN to map “shape patterns” to subsequent drift.  
**Trading rule:** Around earnings announcements, use the CNN score to form **long/short portfolios** (top vs bottom) to exploit conditional PEAD.

---

## 3) “Reviving” PEAD using multi-quarter surprise histories + ML (conditional drift)
**Paper:** *Reviving PEAD with Machine Learning and Historical Earnings Surprises* (Kaczmarek, 2025)  
**Idea:** Instead of only the latest quarter surprise, feed **a long history of surprises (e.g., 8–12 quarters)** into ML to learn when/where PEAD is strongest.  
**Trading rule:** Post-announcement, trade only when the model predicts strong drift (e.g., **long predicted winners / short predicted losers**), optionally with liquidity/size filters.

---

## 4) Jointly predicting “announcement-day reaction” vs “post-announcement drift”
**Paper:** *How Does Financial Machine Learning Predict Returns from Earnings Announcements Data?* (Schnaubelt, 2020)  
**Idea:** Use ML to separately model (i) immediate announcement-day abnormal returns and (ii) delayed post-announcement abnormal returns, using richer event features.  
**Trading rule:** Build a **state-dependent event strategy** (e.g., avoid chasing immediate spikes; instead trade the subset predicted to exhibit delayed underreaction/drift).

---

## 5) Insider trading disclosures (SEC Form 4) as ML event signals
**Paper:** *Insider Purchase Signals in Microcap Equities* (arXiv, 2026-02)  
**Idea:** Treat Form 4 insider purchase filings as events; use gradient boosting / ML to score which filings imply stronger subsequent abnormal returns.  
**Trading rule:** When Form 4 filings are released, trade only **high-score events** (long, or long/short vs matched controls), with strong transaction-cost/liquidity controls.

---

## 6) End-to-end event-driven trading with hierarchical event representations
**Paper:** *Janus-Q: End-to-End Event-Driven Trading via Hierarchical ...* (arXiv, 2026-02)  
**Idea:** Build an event taxonomy and learn a mapping from **event semantics → market impact** using CAR (cumulative abnormal return) as the reward/target.  
**Trading rule:** Generate **event-conditioned trades** (direction/size) using the learned event-to-impact policy; focus on clean event definitions and leakage-safe timing.

---

## 7) Tail-event detection → regime-conditional reversal (trade only in “tail states”)
**Paper:** *Can AI Detect Tail Events? Stock Performance ...* (Jurt, 2026)  
**Idea:** Use AI/ML to detect **tail regimes** (extreme market states) and then apply strategies (e.g., reversal) only when the market is in those states.  
**Trading rule:** Run a **gated strategy**: if the model flags a tail event, execute a predefined reversal (or alternative) trade; otherwise stay in baseline exposure.

---

## 8) A comparison of machine learning methods for predicting the direction of the US stock market on the basis of volatility indices

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

## 9) Automated trading with boosting and expert weighting

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

## 10) Stock index futures price prediction using feature selection and deep learning

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
