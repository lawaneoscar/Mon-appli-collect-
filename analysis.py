# analysis.py - Fonctions d'analyse pour WattScope (INF 232 EC2)
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from models import Client, ReleveQuotidien, Appareil

# ============================================================
# 1. RÉGRESSION LINÉAIRE SIMPLE
# ============================================================
def regression_simple(x, y):
    """
    Calcule la droite de régression y = ax + b
    Retourne : pente (a), intercept (b), R², valeurs prédites
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_mean, y_mean = np.mean(x), np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return {
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r2": round(r2, 4),
        "y_pred": y_pred
    }

# ============================================================
# 2. RÉGRESSION LINÉAIRE MULTIPLE
# ============================================================
def regression_multiple(X, y):
    """
    Calcule y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
    Formule : β = (X'X)⁻¹X'y
    Retourne : coefficients β, R², valeurs prédites
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    # Ajouter la colonne de 1 pour β₀
    X_b = np.column_stack([np.ones(len(X)), X])
    
    # β = (X'X)⁻¹X'y
    try:
        beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        y_pred = X_b @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "intercept": round(beta[0], 4),
            "coefficients": [round(b, 4) for b in beta[1:]],
            "r2": round(r2, 4),
            "y_pred": y_pred
        }
    except np.linalg.LinAlgError:
        return None

# ============================================================
# 3. RÉDUCTION DES DIMENSIONS : ACP (Analyse en Composantes Principales)
# ============================================================
def acp_analyse(X, n_components=2):
    """
    Réduit la dimension des données et retourne les composantes principales.
    Implémentation manuelle de l'ACP :
    1. Centrer les données
    2. Calculer la matrice de covariance
    3. Trouver les vecteurs propres (composantes principales)
    """
    X = np.array(X, dtype=float)
    n_samples, n_features = X.shape
    
    # 1. Centrer les données
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Matrice de covariance
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. Valeurs et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Trier par valeur propre décroissante
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 4. Projection sur les n premières composantes
    X_pca = X_centered @ eigenvectors[:, :n_components]
    
    # Variance expliquée
    variance_expliquee = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return {
        "X_pca": X_pca,
        "variance_expliquee": [round(v, 4) for v in variance_expliquee],
        "composantes": eigenvectors[:, :n_components]
    }

# ============================================================
# 4. CLASSIFICATION SUPERVISÉE
# ============================================================
def classification_supervisee(X, y, seuil):
    """
    Classifie les données en deux catégories selon un seuil.
    Retourne les prédictions et la précision.
    """
    predictions = (X >= seuil).astype(int)
    accuracy = np.mean(predictions == y) if len(y) > 0 else 0
    return {
        "predictions": predictions,
        "accuracy": round(float(accuracy), 4),
        "seuil": seuil
    }

# ============================================================
# 5. CLASSIFICATION NON SUPERVISÉE : K-MEANS
# ============================================================
def kmeans_clustering(X, k=3, max_iters=100):
    """
    Implémentation manuelle de l'algorithme K-means.
    Étapes :
    1. Choisir k centroïdes aléatoires
    2. Assigner chaque point au centroïde le plus proche
    3. Recalculer les centroïdes
    4. Répéter jusqu'à convergence
    """
    X = np.array(X, dtype=float)
    n_samples = len(X)
    
    # Initialisation aléatoire des centroïdes
    np.random.seed(42)
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx].copy()
    
    for iteration in range(max_iters):
        # Assignation des points
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.sum((X - centroids[j]) ** 2, axis=1)
        labels = np.argmin(distances, axis=1)
        
        # Recalcul des centroïdes
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            if np.sum(labels == j) > 0:
                new_centroids[j] = np.mean(X[labels == j], axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        # Vérifier convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Calculer l'inertie (somme des distances au carré)
    inertia = 0
    for j in range(k):
        inertia += np.sum((X[labels == j] - centroids[j]) ** 2)
    
    return {
        "labels": labels.tolist(),
        "centroids": centroids.tolist(),
        "inertia": round(float(inertia), 4),
        "iterations": iteration + 1
    }

# ============================================================
# FONCTIONS D'EXTRACTION DE DONNÉES
# ============================================================
def get_releves_data(db: Session):
    """Récupère tous les relevés sous forme de DataFrame"""
    releves = db.query(ReleveQuotidien).all()
    if not releves:
        return pd.DataFrame()
    data = [{
        "foyer_id": r.foyer_id,
        "date": r.date_releve,
        "kWh": r.index_compteur,
        "coupure_min": r.duree_coupure_minutes,
        "temperature": r.temperature_exterieure or 0,
        "cout_fcfa": r.cout_estime_fcfa or 0
    } for r in releves]
    return pd.DataFrame(data)

def get_clients_stats(db: Session):
    """Récupère les statistiques par client pour l'ACP et le clustering"""
    clients = db.query(Client).all()
    data = []
    for c in clients:
        releves = [r.index_compteur for r in c.releves]
        if len(releves) >= 3:
            data.append({
                "client_id": c.id,
                "nom": c.nom_utilisateur,
                "region": c.region,
                "moyenne_kwh": np.mean(releves),
                "ecart_type_kwh": np.std(releves),
                "max_kwh": np.max(releves),
                "min_kwh": np.min(releves),
                "nb_releves": len(releves)
            })
    return pd.DataFrame(data)
