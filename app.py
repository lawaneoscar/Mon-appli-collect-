# app.py - WattScope Complet - Tous les points INF 232 EC2
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Imports de nos modules
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

# ---------- INTERFACE ----------
st.set_page_config(page_title="⚡ WattScope", page_icon="⚡", layout="wide")
st.title("⚡ WattScope - Analyse de Consommation Électrique")
st.markdown("**TP INF 232 EC2** | *Analyse de données : Régression, ACP, Clustering, Classification*")

db_session = get_db()

# ---------- BARRE LATÉRALE ----------
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
        st.subheader("➕ Nouveau Client / Foyer")
        with st.form("ajout_client"):
            nom = st.text_input("Nom complet ou Identifiant", placeholder="Ex: Jean-Yaoundé")
            region = st.text_input("Ville / Région", placeholder="Ex: Yaoundé")
            logement = st.selectbox("Type de logement", ["Studio", "Appartement", "Villa"])
            if st.form_submit_button("➕ Ajouter le client"):
                if nom and region:
                    db = db_session()
                    db.add(Client(nom_utilisateur=nom, region=region, type_logement=logement))
                    db.commit()
                    db.close()
                    st.success(f"✅ Client '{nom}' enregistré !")
                    st.rerun()
                else:
                    st.error("Veuillez remplir tous les champs")

    with col2:
        st.subheader("📝 Nouveau Relevé Quotidien")
        db = db_session()
        clients = db.query(Client).all()
        db.close()

        if clients:
            with st.form("ajout_releve"):
                client_choisi = st.selectbox("Client / Foyer", clients, format_func=lambda c: f"{c.nom_utilisateur} - {c.region}")
                date_rel = st.date_input("Date du relevé", datetime.date.today())
                index_kwh = st.number_input("Index compteur (kWh) *", min_value=0.0, step=0.1)

                col_a, col_b = st.columns(2)
                with col_a:
                    unite = st.selectbox("Unité coupure", ["Minutes", "Heures"])
                with col_b:
                    duree = st.number_input("Durée coupure", min_value=0.0, step=0.5, value=0.0)

                temperature = st.number_input("Température (°C) - Facultatif", value=0.0, step=0.1)
                cout = st.number_input("Coût estimé (FCFA) - Facultatif", min_value=0.0, step=50.0, value=0.0)

                if st.form_submit_button("📊 Enregistrer le relevé"):
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

    # Liste des clients
    st.markdown("---")
    st.subheader("👥 Clients / Foyers enregistrés")
    db = db_session()
    clients = db.query(Client).all()
    db.close()

    if clients:
        data = []
        for c in clients:
            data.append({
                "ID": c.id,
                "Nom": c.nom_utilisateur,
                "Région": c.region,
                "Logement": c.type_logement,
                "Relevés": len(c.releves),
                "Appareils": len(c.appareils)
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun client enregistré")

# ================================================================
# DASHBOARD CLIENT
# ================================================================
elif menu == "📊 Dashboard Client":
    db = db_session()
    clients = db.query(Client).all()

    if not clients:
        st.warning("Aucun client. Ajoutez-en d'abord.")
        db.close()
    else:
        client_choisi = st.selectbox("Sélectionnez un client", clients, format_func=lambda c: f"{c.nom_utilisateur} - {c.region}")

        if client_choisi:
            releves = list(client_choisi.releves)
            appareils = db.query(Appareil).filter(Appareil.foyer_id == client_choisi.id).all()

            # Stats de base
            st.subheader(f"📊 Statistiques - {client_choisi.nom_utilisateur}")
            col1, col2, col3, col4 = st.columns(4)

            conso_moy = round(sum(r.index_compteur for r in releves) / len(releves), 2) if releves else 0
            conso_max = max([r.index_compteur for r in releves]) if releves else 0
            conso_min = min([r.index_compteur for r in releves]) if releves else 0
            coupure_moy = round(sum(r.duree_coupure_minutes for r in releves) / len(releves), 1) if releves else 0

            with col1:
                st.metric("⚡ Moyenne", f"{conso_moy} kWh")
            with col2:
                st.metric("📋 Total relevés", len(releves))
            with col3:
                st.metric("🔺 Max", f"{conso_max} kWh")
            with col4:
                st.metric("⏱️ Coupure moy.", f"{coupure_moy} min")

            # Graphique évolution
            if releves:
                st.markdown("---")
                st.subheader("📈 Évolution de la consommation")
                df_rel = pd.DataFrame([{
                    "Date": r.date_releve,
                    "Consommation (kWh)": r.index_compteur,
                    "Coupure (min)": r.duree_coupure_minutes
                } for r in releves])
                df_rel = df_rel.sort_values("Date")

                fig = px.line(df_rel, x="Date", y="Consommation (kWh)", markers=True)
                fig.update_traces(line_color="#ffd700", marker_color="#2c5364")
                st.plotly_chart(fig, use_container_width=True)

                # ==============================================
                # 1. RÉGRESSION LINÉAIRE SIMPLE
                # ==============================================
                st.markdown("---")
                st.subheader("🔬 1. Régression Linéaire Simple")
                st.markdown("*Température vs Consommation*")

                temp_data = [(r.temperature_exterieure, r.index_compteur) for r in releves if r.temperature_exterieure is not None]
                if len(temp_data) >= 5:
                    X_temp = np.array([t[0] for t in temp_data])
                    y_conso = np.array([t[1] for t in temp_data])

                    reg_simple = regression_simple(X_temp, y_conso)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Pente (β)", f"{reg_simple['slope']} kWh/°C")
                        st.metric("Intercept (α)", f"{reg_simple['intercept']} kWh")
                    with col_b:
                        st.metric("R²", f"{reg_simple['r2']}")
                        if reg_simple['r2'] > 0.6:
                            st.success("Forte corrélation")
                        elif reg_simple['r2'] > 0.3:
                            st.warning("Corrélation modérée")
                        else:
                            st.info("Faible corrélation")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=X_temp, y=y_conso, mode='markers', name='Données'))
                    fig.add_trace(go.Scatter(x=X_temp, y=reg_simple['y_pred'], mode='lines',
                                             name='Régression', line=dict(color='red')))
                    fig.update_layout(title="Régression simple : Température vs Consommation",
                                      xaxis_title="Température (°C)", yaxis_title="Consommation (kWh)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Pas assez de données (min 5 relevés avec température)")

                # ==============================================
                # 2. RÉGRESSION LINÉAIRE MULTIPLE
                # ==============================================
                st.markdown("---")
                st.subheader("📊 2. Régression Linéaire Multiple")
                st.markdown("*Consommation = f(Température, Coupures)*")

                if len(releves) >= 5:
                    X_multi = np.array([[r.temperature_exterieure or 0, r.duree_coupure_minutes] for r in releves])
                    y_multi = np.array([r.index_compteur for r in releves])

                    reg_multi = regression_multiple(X_multi, y_multi)

                    if reg_multi:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("β₀ (Intercept)", reg_multi['intercept'])
                        with col_b:
                            st.metric("β₁ (Température)", reg_multi['coefficients'][0])
                        with col_c:
                            st.metric("β₂ (Coupures)", reg_multi['coefficients'][1])
                        st.metric("R² multiple", f"{reg_multi['r2']}")

                        fig_multi = go.Figure()
                        fig_multi.add_trace(go.Scatter(x=list(range(len(y_multi))), y=y_multi,
                                                       mode='lines+markers', name='Réel'))
                        fig_multi.add_trace(go.Scatter(x=list(range(len(y_multi))), y=reg_multi['y_pred'],
                                                       mode='lines+markers', name='Prédit'))
                        fig_multi.update_layout(title="Régression multiple : Réel vs Prédit",
                                                xaxis_title="Relevé", yaxis_title="kWh")
                        st.plotly_chart(fig_multi, use_container_width=True)
                else:
                    st.info("Pas assez de données (min 5 relevés)")

                # ==============================================
                # 4. CLASSIFICATION SUPERVISÉE
                # ==============================================
                st.markdown("---")
                st.subheader("🏷️ 4. Classification Supervisée")
                st.markdown("*Classification haute/basse consommation selon un seuil*")

                if len(releves) >= 5:
                    seuil = st.slider("Seuil de consommation (kWh)",
                                      min_value=float(min(r.index_compteur for r in releves)),
                                      max_value=float(max(r.index_compteur for r in releves)),
                                      value=float(np.median([r.index_compteur for r in releves])))

                    consos = np.array([r.index_compteur for r in releves])
                    labels_reels = (consos >= seuil).astype(int)
                    classif_result = classification_supervisee(consos, labels_reels, seuil)

                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        st.metric("Précision", f"{classif_result['accuracy']:.1%}")
                        nb_haut = np.sum(classif_result['predictions'])
                        nb_bas = len(classif_result['predictions']) - nb_haut
                        st.write(f"🔴 Haute conso : {nb_haut} jours")
                        st.write(f"🟢 Basse conso : {nb_bas} jours")
                    with col_c2:
                        df_class = pd.DataFrame({
                            "Date": [r.date_releve for r in releves],
                            "kWh": consos,
                            "Catégorie": ["🔴 Haute" if p == 1 else "🟢 Basse" for p in classif_result['predictions']]
                        })
                        fig_class = px.bar(df_class, x="Date", y="kWh", color="Catégorie",
                                          title="Classification par seuil")
                        st.plotly_chart(fig_class, use_container_width=True)
                else:
                    st.info("Pas assez de données (min 5 relevés)")

            # ==============================================
            # APPAREILS ÉNERGIVORES
            # ==============================================
            st.markdown("---")
            st.subheader("🔌 Analyse des Appareils Énergivores")

            col_app1, col_app2 = st.columns([1, 1])

            with col_app1:
                if appareils:
                    app_data = []
                    for a in appareils:
                        conso_kwh = a.puissance_watts * a.heures_utilisation_jour * a.nombre_appareils / 1000
                        app_data.append({
                            "Appareil": a.nom_appareil,
                            "Qté": a.nombre_appareils,
                            "Watts": a.puissance_watts,
                            "Heures/j": a.heures_utilisation_jour,
                            "kWh/jour": round(conso_kwh, 2)
                        })
                    df_app = pd.DataFrame(app_data).sort_values("kWh/jour", ascending=False)

                    fig_pie = px.pie(df_app, values="kWh/jour", names="Appareil",
                                    title="Consommation par appareil")
                    st.plotly_chart(fig_pie, use_container_width=True)

                    top = df_app.iloc[0]
                    st.error(f"⚠️ **{top['Appareil']}** est le plus énergivore : **{top['kWh/jour']} kWh/jour**")
                else:
                    st.info("Aucun appareil enregistré")

            with col_app2:
                st.markdown("**➕ Ajouter un appareil**")
                with st.form("ajout_appareil"):
                    nom_app = st.text_input("Nom de l'appareil")
                    watts = st.number_input("Puissance (Watts)", min_value=1, value=100)
                    heures = st.number_input("Heures d'utilisation par jour", min_value=0.1, value=1.0, step=0.5)
                    if st.form_submit_button("➕ Ajouter"):
                        if nom_app:
                            db_add = db_session()
                            db_add.add(Appareil(foyer_id=client_choisi.id, nom_appareil=nom_app,
                                               puissance_watts=watts, heures_utilisation_jour=heures,
                                               nombre_appareils=1))
                            db_add.commit()
                            db_add.close()
                            st.success(f"✅ Appareil '{nom_app}' ajouté !")
                            st.rerun()

                if appareils:
                    st.markdown("---")
                    st.dataframe(df_app, use_container_width=True, hide_index=True)

            # Tableau relevés
            st.markdown("---")
            st.subheader("📋 Derniers relevés")
            if releves:
                df_releves = pd.DataFrame([{
                    "Date": str(r.date_releve),
                    "kWh": r.index_compteur,
                    "Coupure (min)": r.duree_coupure_minutes,
                    "Température (°C)": r.temperature_exterieure or "N/A",
                    "Coût (FCFA)": r.cout_estime_fcfa or "N/A"
                } for r in releves[:20]])
                st.dataframe(df_releves, use_container_width=True, hide_index=True)

    db.close()

# ================================================================
# ANALYSES GLOBALES (ACP & CLUSTERING)
# ================================================================
elif menu == "📈 Analyses Globales (ACP & Clustering)":
    st.subheader("📉 3. Réduction de dimension (ACP)")
    st.subheader("👥 5. Classification non supervisée (K-means)")

    db = db_session()
    df_clients = get_clients_stats(db)
    db.close()

    if len(df_clients) >= 6:
        features = ["moyenne_kwh", "ecart_type_kwh", "max_kwh"]
        X_cluster = df_clients[features].values

        # ----- ACP -----
        acp_result = acp_analyse(X_cluster, n_components=2)

        col_acp1, col_acp2 = st.columns(2)

        with col_acp1:
            st.markdown("### Analyse en Composantes Principales (ACP)")
            st.write(f"**Variance expliquée :**")
            st.write(f"- Composante 1 : {acp_result['variance_expliquee'][0]:.1%}")
            st.write(f"- Composante 2 : {acp_result['variance_expliquee'][1]:.1%}")
            st.write(f"- **Total : {(acp_result['variance_expliquee'][0]+acp_result['variance_expliquee'][1]):.1%}**")

            df_acp = pd.DataFrame({
                "CP1": acp_result['X_pca'][:, 0],
                "CP2": acp_result['X_pca'][:, 1],
                "Client": df_clients['nom'],
                "Région": df_clients['region']
            })
            fig_acp = px.scatter(df_acp, x="CP1", y="CP2", text="Client", color="Région",
                                title="Projection ACP des Clients")
            fig_acp.update_traces(textposition='top center')
            st.plotly_chart(fig_acp, use_container_width=True)

        with col_acp2:
            st.markdown("### K-means Clustering")
            k = st.slider("Nombre de clusters (K)", 2, 5, 3)

            kmeans_result = kmeans_clustering(X_cluster, k=k)
            df_clients['Cluster'] = [f"Groupe {l+1}" for l in kmeans_result['labels']]

            fig_kmeans = px.scatter(df_clients, x="moyenne_kwh", y="max_kwh",
                                   color="Cluster", text="nom", size="nb_releves",
                                   title="Clustering K-means des Clients")
            fig_kmeans.update_traces(textposition='top center')
            st.plotly_chart(fig_kmeans, use_container_width=True)

            st.metric("Inertie", f"{kmeans_result['inertia']}")
            st.caption(f"Convergence en {kmeans_result['iterations']} itérations")

        # Résumé des clusters
        st.markdown("---")
        st.subheader("📊 Résumé des Groupes")
        for cluster_name in sorted(df_clients['Cluster'].unique()):
            subset = df_clients[df_clients['Cluster'] == cluster_name]
            with st.expander(f"{cluster_name} ({len(subset)} clients)"):
                st.write(f"**Consommation moyenne :** {subset['moyenne_kwh'].mean():.1f} kWh")
                st.write(f"**Écart-type moyen :** {subset['ecart_type_kwh'].mean():.1f} kWh")
                st.write(f"**Clients :** {', '.join(subset['nom'].tolist())}")
    else:
        st.warning(f"Pas assez de données. Il faut au moins 6 clients avec 3+ relevés chacun. Actuellement : {len(df_clients)} clients.")
        st.info("Ajoutez plus de clients et de relevés pour voir l'ACP et le clustering.")

# ================================================================
# EXPORT EXCEL
# ================================================================
elif menu == "📥 Export Excel":
    st.subheader("📥 Télécharger les données en Excel")

    db = db_session()
    clients = db.query(Client).all()

    if not clients:
        st.warning("Aucun client")
        db.close()
    else:
        client_choisi = st.selectbox("Choisissez un client", clients, format_func=lambda c: c.nom_utilisateur)

        if client_choisi and st.button("📥 Télécharger le fichier Excel"):
            releves = list(client_choisi.releves)
            appareils = db.query(Appareil).filter(Appareil.foyer_id == client_choisi.id).all()

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_rel = pd.DataFrame([{
                    "Date": r.date_releve,
                    "Index compteur (kWh)": r.index_compteur,
                    "Durée coupure (min)": r.duree_coupure_minutes,
                    "Température (°C)": r.temperature_exterieure or 0,
                    "Coût estimé (FCFA)": r.cout_estime_fcfa or 0
                } for r in releves])
                df_rel.to_excel(writer, sheet_name="Relevés", index=False)

                if appareils:
                    df_app = pd.DataFrame([{
                        "Appareil": a.nom_appareil,
                        "Quantité": a.nombre_appareils,
                        "Puissance (W)": a.puissance_watts,
                        "Heures/jour": a.heures_utilisation_jour,
                        "Consommation (kWh/j)": round(a.puissance_watts * a.heures_utilisation_jour * a.nombre_appareils / 1000, 2)
                    } for a in appareils])
                    df_app.to_excel(writer, sheet_name="Appareils", index=False)

            st.download_button(
                label="📥 Cliquer pour télécharger",
                data=output.getvalue(),
                file_name=f"WattScope_{client_choisi.nom_utilisateur}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("✅ Fichier prêt !")
        db.close()

# ---------- PIED DE PAGE ----------
st.markdown("---")
st.caption("⚡ WattScope | TP INF 232 EC2 | Analyse de données | Régression • ACP • Clustering • Classification")
