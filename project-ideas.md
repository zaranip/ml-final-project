# ML Final Project Ideas

## 1. Music Genre Classification via Boosting Ensembles

**Concepts:** AdaBoost, Gradient Boosting, Weak Classifiers, Multi-class AdaBoost, Feature Importance (Shapley), Gini Index

Extract audio features (MFCCs, spectral centroid, chroma) from the GTZAN dataset and build a multi-class genre classifier. Compare AdaBoost with decision stumps against gradient boosting (XGBoost), analyzing how each method weights weak classifiers and handles the multi-class problem. Use SHAP values to explain which audio features drive genre distinctions.

**References:**
- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119–139.
- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293–302.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

---

## 2. Deepfake Detection from Facial Landmarks using Gradient Boosted Trees

**Concepts:** Gradient Boosting, Loss Functions, Decision Boundaries, Feature Importance, Gini Index

Extract geometric facial landmark features (distances, angles, symmetry measures) from real and AI-generated face images (FaceForensics++ dataset). Train a gradient boosted classifier (XGBoost) to distinguish real from fake using *only* geometric inconsistencies—no raw pixels. Analyze which facial geometry features are most discriminative via SHAP, and visualize the decision boundary in a 2D t-SNE projection of the feature space.

**References:**
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- Rössler, A., Cozzolino, D., Verdoliva, L., et al. (2019). FaceForensics++: Learning to detect manipulated facial images. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 1–11.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.


## 3. LLM Embeddings as Features for Ensemble Classifiers

**Concepts:** Bagging, Random Forest, Gradient Boosting, Feature Importance (Shapley), OOB Error, Decision Boundaries

Use a frozen pretrained sentence encoder (e.g., `sentence-transformers/all-MiniLM-L6-v2`) to embed texts from a benchmark dataset (SST-2, AG News, or IMDb). Train Random Forest and XGBoost classifiers on the resulting embedding vectors rather than raw text. Compare OOB error across ensemble sizes, measure convergence as the number of trees grows, and apply SHAP to identify which embedding dimensions are most predictive. Contrast against a TF-IDF baseline to quantify the representational lift from pretrained embeddings.

**References:**
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3982–3992.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

---

## 4. Token-level SHAP: Explaining What an LLM Actually Reads

**Concepts:** Feature Importance (Shapley), Logistic Regression, ANN, Sigmoid Function, Decision Boundaries

Fine-tune DistilBERT on a text classification task (e.g., toxic comment detection using the Jigsaw dataset). Then apply SHAP's `Explainer` to attribute each prediction to individual input tokens—producing a heatmap over the sentence. Compare SHAP token attributions against simpler gradient-based saliency maps and attention weights. Investigate cases where SHAP and attention disagree, and test whether the model's reasoning is consistent with human intuition about which words signal toxicity.

**References:**
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 3543–3556.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.

---

## 5. Tuning LLM Fine-tuning Hyperparameters with Optuna

**Concepts:** Optimization (Optuna), Hyper-parameter Tuning, Loss Functions, Gradient Descent, Regularization (L2), Mini-batch Gradient Descent

Fine-tune DistilBERT or RoBERTa on a text classification task and use Optuna to search the fine-tuning hyperparameter space: learning rate, weight decay (L2 regularization), batch size, warmup ratio, and number of epochs. Compare Optuna's TPE sampler against random search in terms of final accuracy and trials-to-convergence. Visualize the hyperparameter landscape as contour plots, and analyze which hyperparameters interact most strongly—learning rate × weight decay is often the dominant interaction.

**References:**
- Akiba, T., Sano, S., Yanase, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.
- Mosbach, M., Andriushchenko, M., & Klakow, D. (2021). On the stability of fine-tuning BERT: Misconceptions, explanations, and strong baselines. *International Conference on Learning Representations*.
- Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune BERT for text classification? *China National Conference on Chinese Computational Linguistics*, 194–206.

---

## 6. Logistic Regression Probes on Transformer Hidden Layers

**Concepts:** Logistic Regression, Sigmoid Function, Hidden Layers, Decision Boundaries, Regularization (L1/L2), Gradient Descent

Train a simple L2-regularized logistic regression "probe" on the hidden-state activations extracted from each layer of a pretrained BERT model to classify part-of-speech tags or named entity types. Since BERT has 12 layers, you get 12 decision boundaries to compare—revealing *where* in the network linguistic structure emerges. Use L1 regularization to identify which hidden dimensions encode each linguistic property. This directly connects the theory of decision boundaries and regularization to the internals of a real LLM.

**References:**
- Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 4593–4601.
- Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. *ICLR Workshop*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.