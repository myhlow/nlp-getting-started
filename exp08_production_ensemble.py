import numpy as np
import pandas as pd
import re
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', str(text))
    text = re.sub(r'@\w+', ' @user ', text)
    return text

def format_input(df):
    kws = df['keyword'].fillna('').astype(str).str.replace('%20', ' ')
    cleaned = df['text'].apply(clean_text)
    return np.where(kws != '', 'keyword: ' + kws + ' | tweet: ' + cleaned, cleaned)

def main():
    print("=" * 70)
    print("Experiment 07: Production 4-Way Multi-Modal Ensemble for Top 100")
    print("=" * 70)

    train = pd.read_csv('/root/nlp-getting-started/train.csv')
    test = pd.read_csv('/root/nlp-getting-started/test.csv')
    sample_sub = pd.read_csv('/root/nlp-getting-started/sample_submission.csv')

    # 1. Clean conflicting duplicates in train via majority voting
    vote = train.groupby('text')['target'].agg(lambda s: s.value_counts().index[0]).to_dict()
    n_changed = (train['target'] != train['text'].map(vote)).sum()
    print(f"Ground-Truth Conflict Resolution: Corrected {n_changed} conflicting labels in train.csv")
    y = train['text'].map(vote).values

    # 2. Load cached 384-dimensional dense transformer embeddings
    X_tr_emb = np.load('/root/nlp-getting-started/train_embeddings.npy')
    X_te_emb = np.load('/root/nlp-getting-started/test_embeddings.npy')
    print(f"Loaded Dense Embeddings: Train {X_tr_emb.shape}, Test {X_te_emb.shape}")

    tr_txt = format_input(train)
    te_txt = format_input(test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lr_dense = np.zeros(len(train))
    oof_mlp_dense = np.zeros(len(train))
    oof_lr_sparse = np.zeros(len(train))
    oof_joint = np.zeros(len(train))

    te_lr_dense = np.zeros(len(test))
    te_mlp_dense = np.zeros(len(test))
    te_lr_sparse = np.zeros(len(test))
    te_joint = np.zeros(len(test))

    print("\nTraining 5-Fold Stratified Models across 4 Disparate Inductive Biases...")
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_tr_emb, y)):
        f_t0 = time.time()

        # Model 1: Logistic Regression on Dense Transformer Embeddings (C=2.0)
        clf_dense = LogisticRegression(C=2.0, max_iter=300, random_state=42)
        clf_dense.fit(X_tr_emb[tr_idx], y[tr_idx])
        oof_lr_dense[va_idx] = clf_dense.predict_proba(X_tr_emb[va_idx])[:, 1]
        te_lr_dense += clf_dense.predict_proba(X_te_emb)[:, 1] / 5.0

        # Model 2: Multi-Layer Perceptron on Dense Transformer Embeddings
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.1, max_iter=200, random_state=42+fold, early_stopping=True)
        mlp.fit(X_tr_emb[tr_idx], y[tr_idx])
        oof_mlp_dense[va_idx] = mlp.predict_proba(X_tr_emb[va_idx])[:, 1]
        te_mlp_dense += mlp.predict_proba(X_te_emb)[:, 1] / 5.0

        # Model 3: Sparse Word (1,2) + Char (3,5) TF-IDF FeatureUnion
        union = FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, sublinear_tf=True))
        ])
        X_tr_sp = union.fit_transform(tr_txt[tr_idx])
        X_va_sp = union.transform(tr_txt[va_idx])
        X_te_sp = union.transform(te_txt)

        clf_sp = LogisticRegression(C=1.5, max_iter=300, random_state=42)
        clf_sp.fit(X_tr_sp, y[tr_idx])
        oof_lr_sparse[va_idx] = clf_sp.predict_proba(X_va_sp)[:, 1]
        te_lr_sparse += clf_sp.predict_proba(X_te_sp)[:, 1] / 5.0

        # Model 4: Joint Concatenated Feature Space [Dense (384) + Sparse (38,000+)]
        X_tr_comb = hstack([csr_matrix(X_tr_emb[tr_idx]), X_tr_sp])
        X_va_comb = hstack([csr_matrix(X_tr_emb[va_idx]), X_va_sp])
        X_te_comb = hstack([csr_matrix(X_te_emb), X_te_sp])

        clf_jt = LogisticRegression(C=1.5, max_iter=400, random_state=42)
        clf_jt.fit(X_tr_comb, y[tr_idx])
        oof_joint[va_idx] = clf_jt.predict_proba(X_va_comb)[:, 1]
        te_joint += clf_jt.predict_proba(X_te_comb)[:, 1] / 5.0

        print(f"  Fold {fold+1}/5 completed in {time.time() - f_t0:.2f}s")

    train_time = time.time() - t0
    print(f"\nAll 5 folds trained in {train_time:.2f}s.")

    # Individual Out-Of-Fold Accuracies
    print("\n--- Individual Model Out-Of-Fold Accuracies ---")
    print(f"  1. Dense Logistic Regression: {accuracy_score(y, (oof_lr_dense >= 0.50).astype(int))*100:.2f}% | F1: {f1_score(y, (oof_lr_dense >= 0.50).astype(int)):.4f}")
    print(f"  2. Dense MLP (128, 64):       {accuracy_score(y, (oof_mlp_dense >= 0.50).astype(int))*100:.2f}% | F1: {f1_score(y, (oof_mlp_dense >= 0.50).astype(int)):.4f}")
    print(f"  3. Sparse Word+Char TF-IDF:   {accuracy_score(y, (oof_lr_sparse >= 0.50).astype(int))*100:.2f}% | F1: {f1_score(y, (oof_lr_sparse >= 0.50).astype(int)):.4f}")
    print(f"  4. Joint Dense+Sparse Model:  {accuracy_score(y, (oof_joint >= 0.50).astype(int))*100:.2f}% | F1: {f1_score(y, (oof_joint >= 0.50).astype(int)):.4f}")

    # 4-Way Weighted Ensemble
    w_dense, w_mlp, w_sparse, w_jt = 0.20, 0.10, 0.30, 0.40
    t_opt = 0.54

    blend_oof = w_dense * oof_lr_dense + w_mlp * oof_mlp_dense + w_sparse * oof_lr_sparse + w_jt * oof_joint
    oof_preds = (blend_oof >= t_opt).astype(int)

    final_acc = accuracy_score(y, oof_preds)
    final_f1 = f1_score(y, oof_preds)

    print("\n" + "=" * 70)
    print(f"🏆 4-WAY PRODUCTION ENSEMBLE BENCHMARK:")
    print(f"  Weights: Dense LR={w_dense:.2f}, Dense MLP={w_mlp:.2f}, Sparse TFIDF={w_sparse:.2f}, Joint={w_jt:.2f}")
    print(f"  Calibrated Decision Threshold: {t_opt:.2f}")
    print(f"  5-Fold Stratified OOF Accuracy: {final_acc*100:.2f}%")
    print(f"  5-Fold Stratified OOF F1-Score: {final_f1:.4f}")
    print("=" * 70)

    # Compute Test Predictions
    blend_test = w_dense * te_lr_dense + w_mlp * te_mlp_dense + w_sparse * te_lr_sparse + w_jt * te_joint
    test_preds = (blend_test >= t_opt).astype(int)

    # 5. Exact-Match Duplicate Post-Processing
    test_in_train = test[test['text'].isin(train['text'])]
    override_count = 0
    for idx, row in test_in_train.iterrows():
        train_matches = train[train['text'] == row['text']]
        unique_targets = train_matches['target'].unique()
        if len(unique_targets) == 1:
            target_val = unique_targets[0]
            if test_preds[idx] != target_val:
                test_preds[idx] = target_val
                override_count += 1

    print(f"\nExact-Match Post-Processing: Aligned {override_count} test predictions to unanimous train ground truth.")
    pos_ratio = np.mean(test_preds) * 100
    print(f"Predicted Disaster Ratio on Test Set: {pos_ratio:.2f}% ({np.sum(test_preds)} / {len(test_preds)})")
    print(f"  (Ground truth benchmark: 42.97%)")

    # 6. Save Submission File
    sub = pd.DataFrame({'id': test['id'], 'target': test_preds})
    assert len(sub) == len(sample_sub)
    assert (sub['id'].values == sample_sub['id'].values).all()
    assert not sub['target'].isnull().any()

    out_file = '/root/nlp-getting-started/submission_top100_candidate.csv'
    sub.to_csv(out_file, index=False)
    print(f"\nSaved Candidate Submission to {out_file} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
