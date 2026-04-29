# app.py - WattScope Complet - INF 232 EC2
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from models import Client, ReleveQuotidien, Appareil, Base
from analysis import (
    regression_simple,
    regression_multiple,
    acp_analyse,
    kmeans_clustering,
    classification_supervisee,
    get_releves_data,
    get_clients_stats
)

# ---------- BASE DE DONNÉES ----------
@st.cache_resource
def get_db():
    engine = create_engine("sqlite:///wattscope.db")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)

st.set_page_config(page_title="⚡ WattScope", page_icon="⚡", layout="wide")
st.title("⚡ WattScope - Analyse de Consommation Électrique")
st.markdown("**TP INF 232 EC2** | *Analyse de données*")

db_session = get_db()

menu = st.sidebar.radio("📋 Menu", [
    "🏠 Accueil & Collecte",
    "📊 Dashboard Client",
    "📈 Analyses Globales (ACP & Clustering)",
    "📥 Export Excel"
])

# ================================================================
# PAGE ACCUEIL & COLLECTE
# ================================================================
if menu == "🏠 Accueil & Collecte":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("➕ Nouveau Client")
        with st.form("ajout_client"):
            nom = st.text_input("Nom complet", placeholder="Ex: Jean-Yaoundé")
            region = st.text_input("Ville / Région", placeholder="Ex: Yaoundé")
            logement = st.selectbox("Type de logement", ["Studio", "Appartement", "Villa"])
            if st.form_submit_button("➕ Ajouter"):
                if nom and region:
                    db = db_session()
                    db.add(Client(nom_utilisateur=nom, region=region, type_logement=logement))
                    db.commit()
                    db.close()
                    st.success(f"✅ '{nom}' enregistré !")
                    st.rerun()
                else:
                    st.error("Remplissez tous les champs")

    with col2:
        st.subheader("📝 Nouveau Relevé")
        db = db_session()
        clients = db.query(Client).all()
        db.close()

        if clients:
            with st.form("ajout_releve"):
                client_choisi = st.selectbox("Client", clients, format_func=lambda c: f"{c.nom_utilisateur} - {c.region}")
                date_rel = st.date_input("Date", datetime.date.today())
                index_kwh = st.number_input("Index compteur (kWh) *", min_value=0.0, step=0.1)
                col_a, col_b = st.columns(2)
                with col_a:
                    unite = st.selectbox("Unité coupure", ["Minutes", "Heures"])
                with col_b:
                    duree = st.number_input("Durée coupure", min_value=0.0, step=0.5, value=0.0)
                temperature = st.number_input("Température (°C)", value=0.0, step=0.1)
                cout = st.number_input("Coût estimé (FCFA)", min_value=0.0, step=50.0, value=0.0)

                if st.form_submit_button("📊 Enregistrer"):
                    duree_minutes = int(duree * 60) if unite == "Heures" else int(duree)
                    temp_val = temperature if temperature != 0.0 else None
                    db = db_session()
                    db.add(ReleveQuotidien(
                        foyer_id=client_choisi.id,
                        date_releve=date_rel,
                        index_compteur=index_kwh,
                        duree_coupure_minutes=duree_minutes,
                        temperature_exterieure=temp_val,
                        cout_estime_fcfa=cout
                    ))
                    db.commit()
                    db.close()
                    st.success(f"✅ Relevé du {date_rel} enregistré !")
                    st.rerun()
        else:
            st.info("Ajoutez d'abord un client")

    # Liste des clients (session gardée ouverte jusqu'à utilisation)
    st.markdown("---")
    st.subheader("👥 Clients enregistrés")
    db = db_session()
    clients = db.query(Client).all()

    if clients:
        data = []
        for c in clients:
            # Charger les données AVANT de fermer la session
            nb_rel = len(c.releves)
            nb_app = len(c.appareils)
            data.append({
                "ID": c.id,
                "Nom": c.nom_utilisateur,
                "Région": c.region,
                "Logement": c.type_logement,
                "Relevés": nb_rel,
                "Appareils": nb_app
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun client")
    db.close()

# ================================================================
# DASHBOARD CLIENT
# ================================================================
elif menu == "📊 Dashboard Client":
    db = db_session()
    clients = db.query(Client).all()

    if not clients:
        st.warning("Aucun client.")
        db.close()
    else:
        client_choisi = st.selectbox("Client", clients, format_func=lambda c: f"{c.nom_utilisateur} - {c.region}")

        if client_choisi:
            # Charger toutes les données AVANT de fermer la session
            releves = list(client_choisi.releves)
            appareils = list(client_choisi.appareils)

            st.subheader(f"📊 {client_choisi.nom_utilisateur}")
            col1, col2, col3, col4 = st.columns(4)
            conso_moy = round(sum(r.index_compteur for r in releves) / len(releves), 2) if releves else 0
            conso_max = max([r.index_compteur for r in releves]) if releves else 0
            coupure_moy = round(sum(r.duree_coupure_minutes for r in releves) / len(releves), 1) if releves else 0

            with col1:
                st.metric("⚡ Moyenne", f"{conso_moy} kWh")
            with col2:
                st.metric("📋 Relevés", len(releves))
            with col3:
                st.metric("🔺 Max", f"{conso_max} kWh")
            with col4:
                st.metric("⏱️ Coupure moy.", f"{coupure_moy} min")

            if releves:
                st.markdown("---")
                st.subheader("📈 Évolution")
                df_rel = pd.DataFrame([{
                    "Date": r.date_releve,
                    "Consommation (kWh)": r.index_compteur
                } for r in releves])
                df_rel = df_rel.sort_values("Date")
                fig = px.line(df_rel, x="Date", y="Consommation (kWh)", markers=True)
                fig.update_traces(line_color="#ffd700")
                st.plotly_chart(fig, use_container_width=True)

                # 1. RÉGRESSION SIMPLE
                st.markdown("---")
                st.subheader("🔬 1. Régression Linéaire Simple")
                temp_data = [(r.temperature_exterieure, r.index_compteur) for r in releves if r.temperature_exterieure is not None]
                if len(temp_data) >= 5:
                    X_temp = np.array([t[0] for t in temp_data])
                    y_conso = np.array([t[1] for t in temp_data])
                    reg = regression_simple(X_temp, y_conso)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Pente", f"{reg['slope']} kWh/°C")
                    with col_b:
                        st.metric("R²", f"{reg['r2']}")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=X_temp, y=y_conso, mode='markers', name='Données'))
                    fig2.add_trace(go.Scatter(x=X_temp, y=reg['y_pred'], mode='lines', name='Régression', line=dict(color='red')))
                    fig2.update_layout(title="Température vs Consommation")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Min 5 relevés avec température")

                # 2. RÉGRESSION MULTIPLE
                st.markdown("---")
                st.subheader("📊 2. Régression Linéaire Multiple")
                if len(releves) >= 5:
                    X_multi = np.array([[r.temperature_exterieure or 0, r.duree_coupure_minutes] for r in releves])
                    y_multi = np.array([r.index_compteur for r in releves])
                    reg_m = regression_multiple(X_multi, y_multi)
                    if reg_m:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Intercept", reg_m['intercept'])
                        with col_b:
                            st.metric("β Temp.", reg_m['coefficients'][0])
                        with col_c:
                            st.metric("β Coupure", reg_m['coefficients'][1])
                        st.metric("R² multiple", reg_m['r2'])
                else:
                    st.info("Min 5 relevés")

                # 4. CLASSIFICATION SUPERVISÉE
                st.markdown("---")
                st.subheader("🏷️ 4. Classification Supervisée")
                if len(releves) >= 5:
                    seuil = st.slider("Seuil (kWh)",
                                      min_value=float(min(r.index_compteur for r in releves)),
                                      max_value=float(max(r.index_compteur for r in releves)),
                                      value=float(np.median([r.index_compteur for r in releves])))
                    consos = np.array([r.index_compteur for r in releves])
                    labels_reels = (consos >= seuil).astype(int)
                    classif = classification_supervisee(consos, labels_reels, seuil)
                    st.metric("Précision", f"{classif['accuracy']:.1%}")
                    nb_haut = np.sum(classif['predictions'])
                    st.write(f"🔴 Haute : {nb_haut} | 🟢 Basse : {len(classif['predictions']) - nb_haut}")
                else:
                    st.info("Min 5 relevés")

            # APPAREILS
            st.markdown("---")
            st.subheader("🔌 Appareils Énergivores")
            col_app1, col_app2 = st.columns([1, 1])
            with col_app1:
                if appareils:
                    app_data = []
                    for a in appareils:
                        conso_kwh = a.puissance_watts * a.heures_utilisation_jour * a.nombre_appareils / 1000
                        app_data.append({"Appareil": a.nom_appareil, "kWh/jour": round(conso_kwh, 2)})
                    df_app = pd.DataFrame(app_data)
                    fig_pie = px.pie(df_app, values="kWh/jour", names="Appareil")
                    st.plotly_chart(fig_pie, use_container_width=True)
            with col_app2:
                with st.form("ajout_appareil"):
                    nom_app = st.text_input("Appareil")
                    watts = st.number_input("Watts", min_value=1, value=100)
                    heures = st.number_input("Heures/jour", min_value=0.1, value=1.0, step=0.5)
                    if st.form_submit_button("➕ Ajouter"):
                        if nom_app:
                            db_add = db_session()
                            db_add.add(Appareil(foyer_id=client_choisi.id, nom_appareil=nom_app, puissance_watts=watts, heures_utilisation_jour=heures, nombre_appareils=1))
                            db_add.commit()
                            db_add.close()
                            st.success(f"✅ '{nom_app}' ajouté !")
                            st.rerun()

            # Tableau relevés
            st.markdown("---")
            st.subheader("📋 Relevés")
            if releves:
                df_rel = pd.DataFrame([{
                    "Date": str(r.date_releve),
                    "kWh": r.index_compteur,
                    "Coupure (min)": r.duree_coupure_minutes,
                    "Température": r.temperature_exterieure or "N/A",
                    "Coût (FCFA)": r.cout_estime_fcfa or "N/A"
                } for r in releves[:20]])
                st.dataframe(df_rel, use_container_width=True, hide_index=True)

    db.close()

# ================================================================
# ANALYSES GLOBALES
# ================================================================
elif menu == "📈 Analyses Globales (ACP & Clustering)":
    st.subheader("📉 3. ACP & 5. K-means Clustering")
    db = db_session()
    df_clients = get_clients_stats(db)
    db.close()

    if len(df_clients) >= 6:
        features = ["moyenne_kwh", "ecart_type_kwh", "max_kwh"]
        X = df_clients[features].values
        acp = acp_analyse(X)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ACP")
            st.write(f"CP1: {acp['variance_expliquee'][0]:.1%} | CP2: {acp['variance_expliquee'][1]:.1%}")
            df_acp = pd.DataFrame({"CP1": acp['X_pca'][:, 0], "CP2": acp['X_pca'][:, 1], "Client": df_clients['nom']})
            fig = px.scatter(df_acp, x="CP1", y="CP2", text="Client", title="Projection ACP")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### K-means")
            k = st.slider("K", 2, 5, 3)
            km = kmeans_clustering(X, k=k)
            df_clients['Cluster'] = [f"Groupe {l+1}" for l in km['labels']]
            fig2 = px.scatter(df_clients, x="moyenne_kwh", y="max_kwh", color="Cluster", text="nom", title="Clustering")
            st.plotly_chart(fig2, use_container_width=True)
            st.metric("Inertie", km['inertia'])
    else:
        st.warning(f"Besoin de 6 clients avec 3+ relevés. Actuellement : {len(df_clients)}")

# ================================================================
# EXPORT EXCEL
# ================================================================
elif menu == "📥 Export Excel":
    st.subheader("📥 Export Excel")
    db = db_session()
    clients = db.query(Client).all()
    if clients:
        client_choisi = st.selectbox("Client", clients, format_func=lambda c: c.nom_utilisateur)
        if st.button("📥 Télécharger"):
            releves = list(client_choisi.releves)
            output = BytesIO()
            df = pd.DataFrame([{"Date": r.date_releve, "kWh": r.index_compteur, "Coupure(min)": r.duree_coupure_minutes} for r in releves])
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name="Relevés", index=False)
            st.download_button("📥 Cliquer", output.getvalue(), f"WattScope_{client_choisi.nom_utilisateur}.xlsx")
    else:
        st.warning("Aucun client")
    db.close()

st.markdown("---")
st.caption("⚡ WattScope | TP INF 232 EC2")
