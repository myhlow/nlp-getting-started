import pandas as pd
import numpy as np
import re
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score, classification_report

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', str(text))
    text = re.sub(r'@\w+', ' @user ', text)
    return text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_embeddings(texts, model, tokenizer, batch_size=128):
    all_embs = []
    model.eval()
    t0 = time.time()
    n_batches = int(np.ceil(len(texts) / batch_size))
    with torch.no_grad():
        for b in range(n_batches):
            batch_texts = texts[b*batch_size : (b+1)*batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            outputs = model(**inputs)
            embs = mean_pooling(outputs, inputs['attention_mask'])
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
    all_embs = np.vstack(all_embs)
    print(f"  Encoded {len(texts)} samples in {time.time() - t0:.1f}s ({len(texts)/(time.time()-t0):.1f} smp/s) -> Shape: {all_embs.shape}")
    return all_embs

def main():
    print("=" * 70)
    print("Experiment 06: Transformer Dense Embeddings + Hybrid Feature Union")
    print("=" * 70)

    train = pd.read_csv('/root/nlp-getting-started/train.csv')
    test = pd.read_csv('/root/nlp-getting-started/test.csv')
    sample_sub = pd.read_csv('/root/nlp-getting-started/sample_submission.csv')

    # 1. Clean conflicting duplicates in train via majority voting
    vote = train.groupby('text')['target'].agg(lambda s: s.value_counts().index[0]).to_dict()
    n_changed = (train['target'] != train['text'].map(vote)).sum()
    print(f"Ground-Truth Label Conflict Resolution: Corrected {n_changed} conflicting labels in train.csv")
    y_clean = train['text'].map(vote).values

    # 2. Format input text with structured keyword prefix
    def format_input(df):
        kws = df['keyword'].fillna('').astype(str).str.replace('%20', ' ')
        cleaned = df['text'].apply(clean_text)
        return np.where(kws != '', 'keyword: ' + kws + ' | tweet: ' + cleaned, cleaned)

    train_formatted = format_input(train).tolist()
    test_formatted = format_input(test).tolist()

    # 3. Extract 384-dimensional dense semantic embeddings
    print("\nLoading sentence-transformers/all-MiniLM-L6-v2 on CPU...")
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    print("Encoding train tweets...")
    X_train_emb = compute_embeddings(train_formatted, model, tokenizer, batch_size=128)
    print("Encoding test tweets...")
    X_test_emb = compute_embeddings(test_formatted, model, tokenizer, batch_size=128)

    # 4. Stratified 5-Fold Cross-Validation: Model A (Dense Embeddings alone)
    print("\n--- Model A: 384-d Dense Transformer Embeddings + Logistic Regression ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_a = np.zeros(len(train))
    test_a = np.zeros(len(test))

    for trn_idx, val_idx in cv.split(X_train_emb, y_clean):
        clf = LogisticRegression(C=2.0, max_iter=300, random_state=42)
        clf.fit(X_train_emb[trn_idx], y_clean[trn_idx])
        oof_a[val_idx] = clf.predict_proba(X_train_emb[val_idx])[:, 1]
        test_a += clf.predict_proba(X_test_emb)[:, 1] / 5.0

    acc_a_50 = accuracy_score(y_clean, (oof_a >= 0.50).astype(int))
    f1_a_50 = f1_score(y_clean, (oof_a >= 0.50).astype(int))
    print(f"  Model A (Dense alone @ 0.50): OOF Acc: {acc_a_50*100:.2f}% | F1: {f1_a_50:.4f}")

    # 5. Model B: Sparse Word + Char TF-IDF
    print("\n--- Model B: Sparse Word (1,2) + Char (3,5) TF-IDF ---")
    oof_b = np.zeros(len(train))
    test_b = np.zeros(len(test))

    train_arr = np.array(train_formatted)
    test_arr = np.array(test_formatted)

    for trn_idx, val_idx in cv.split(train_arr, y_clean):
        union = FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, sublinear_tf=True))
        ])
        X_tr = union.fit_transform(train_arr[trn_idx])
        X_va = union.transform(train_arr[val_idx])
        X_te = union.transform(test_arr)

        clf = LogisticRegression(C=1.5, max_iter=300, random_state=42)
        clf.fit(X_tr, y_clean[trn_idx])
        oof_b[val_idx] = clf.predict_proba(X_va)[:, 1]
        test_b += clf.predict_proba(X_te)[:, 1] / 5.0

    acc_b_50 = accuracy_score(y_clean, (oof_b >= 0.50).astype(int))
    f1_b_50 = f1_score(y_clean, (oof_b >= 0.50).astype(int))
    print(f"  Model B (Sparse alone @ 0.50): OOF Acc: {acc_b_50*100:.2f}% | F1: {f1_b_50:.4f}")

    # 6. Model C: Hybrid Stacking Blend (Dense Semantic + Sparse Lexical)
    print("\n--- Model C: Hybrid Semantic + Lexical Ensemble ---")
    best_w = 0.5
    best_acc = 0.0
    best_t = 0.5

    for w in np.linspace(0.2, 0.8, 13):
        blend_oof = w * oof_a + (1 - w) * oof_b
        for t in np.linspace(0.40, 0.60, 41):
            acc = accuracy_score(y_clean, (blend_oof >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_w = w
                best_t = t

    blend_oof_final = best_w * oof_a + (1 - best_w) * oof_b
    final_f1 = f1_score(y_clean, (blend_oof_final >= best_t).astype(int))
    print(f"  Optimal Weights: Dense MiniLM = {best_w:.2f}, Sparse TF-IDF = {1-best_w:.2f}")
    print(f"  Optimal Threshold: {best_t:.3f}")
    print(f"  Hybrid 5-Fold OOF Accuracy: {best_acc*100:.2f}% (F1: {final_f1:.4f})")

    # 7. Generate Predictions & Duplicate Post-Processing
    test_blend = best_w * test_a + (1 - best_w) * test_b
    test_preds = (test_blend >= best_t).astype(int)

    # Post-process exact duplicate matches from train
    test_in_train = test[test['text'].isin(train['text'])]
    override_count = 0
    for idx, row in test_in_train.iterrows():
        train_matches = train[train['text'] == row['text']]
        # If train matches are unanimously of one class, align test prediction
        unique_targets = train_matches['target'].unique()
        if len(unique_targets) == 1:
            target_val = unique_targets[0]
            if test_preds[idx] != target_val:
                test_preds[idx] = target_val
                override_count += 1

    print(f"\nExact Duplicate Post-Processing: Aligned {override_count} test predictions to unanimous train ground truth.")
    print(f"Predicted Disaster ratio on test set: {np.mean(test_preds)*100:.2f}% ({np.sum(test_preds)} / {len(test_preds)})")

    sub = pd.DataFrame({'id': test['id'], 'target': test_preds})
    assert len(sub) == len(sample_sub)
    assert (sub['id'].values == sample_sub['id'].values).all()
    out_file = '/root/nlp-getting-started/submission_hybrid_transformer.csv'
    sub.to_csv(out_file, index=False)
    print(f"Saved submission candidate for tomorrow to {out_file} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
