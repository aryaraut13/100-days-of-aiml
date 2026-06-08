import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from model import find_best_k, train_knn, train_nb

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

# ── KNN: find best K ─────────────────────────────────────────
print("[KNN] Searching best K (1-20)...")
result = find_best_k(X_tr_sc, y_train)
best_k = result["best_k"]
print(f"      Best K = {best_k}  |  CV F1 = {result['best_f1']}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
k_vals  = list(result["scores"].keys())
f1_vals = list(result["scores"].values())
plt.plot(k_vals, f1_vals, "o-", color="#5865F2", lw=2)
plt.axvline(best_k, color="#ED4245", linestyle="--", label=f"Best K={best_k}")
plt.xlabel("K (number of neighbors)")
plt.ylabel("CV F1 Score")
plt.title("KNN — Finding Best K")
plt.legend()

# ── KNN final eval ───────────────────────────────────────────
knn = train_knn(X_tr_sc, y_train, best_k)
knn_preds = knn.predict(X_te_sc)
knn_f1    = f1_score(y_test, knn_preds)

# ── Naive Bayes ──────────────────────────────────────────────
print("\n[NB]  Training GaussianNB (no scaling required)...")
nb = train_nb(X_train, y_train)
nb_preds = nb.predict(X_test)
nb_f1    = f1_score(y_test, nb_preds)

# ── Comparison bar ───────────────────────────────────────────
plt.subplot(1, 2, 2)
names = [f"KNN (k={best_k})", "Naive Bayes"]
f1s   = [round(knn_f1, 4), round(nb_f1, 4)]
bars  = plt.bar(names, f1s, color=["#5865F2","#57F287"])
for bar, v in zip(bars, f1s):
    plt.text(bar.get_x()+bar.get_width()/2, v+0.005,
             str(v), ha="center", fontweight="bold")
plt.ylim(0.85, 1.0)
plt.title("KNN vs Naive Bayes — F1 Score")

plt.tight_layout()
plt.savefig("day22_knn_nb.png", dpi=150)

print(f"\n[RESULTS]  KNN F1 = {round(knn_f1,4)}  |  Naive Bayes F1 = {round(nb_f1,4)}")
print("           KNN needs scaling. Naive Bayes does not.")
print("Plot saved -> day22_knn_nb.png")