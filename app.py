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
from models import Client, ReleveQuotidien, Appareil, Base
from analysis import (regression_simple, regression_multiple, acp_analyse,
                      kmeans_clustering, classification_supervisee,
                      get_releves_data, get_clients_stats)

# ---------- BASE DE DONNÉES ----------
@st.cache_resource
def get_db():
    engine = create_engine("sqlite:///wattscope.db")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)

# ---------- INTERFACE ----------
st.set_page_config(page_title="⚡ WattScope", page_icon="⚡", layout="wide")
st.title("⚡ WattScope - Analyse de Consommation Électrique")
st.markdown("**TP INF 232 EC2** | *Collecte et Analyse Descriptive des Données*")
st.markdown("---")

db_session = get_db()

# ---------- BARRE LATÉRALE ----------
menu = st.sidebar.radio("📋 Menu", [
    "🏠 Accueil & Collecte",
    "📊 Dashboard Client",
    "📈 Analyses Avancées",
    "📥 Export Excel"
])

# ================================================================
# PAGE 1 : ACCUEIL & COLLECTE
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

        st.markdown("---")
        st.subheader("👥 Clients enregistrés")
        db = db_session()
        clients = db.query(Client).all()
        db.close()
        if clients:
            data = [{"ID": c.id, "Nom": c.nom_utilisateur, "Région": c.region,
                     "Logement": c.type_logement, "Relevés": len(c.releves)} for c in clients]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        else:
            st.info("Aucun client enregistré")

    with col2:
        st.subheader("📝 Nouveau Relevé Quotidien")
        db = db_session()
        clients = db.query(Client).all()
        db.close()

        if clients:
            with st.form("ajout_releve"):
                client_choisi = st.selectbox("Client / Foyer", clients,
                                             format_func=lambda c: f"{c.nom_utilisateur} - {c.region}")
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
                    db.add(ReleveQuotidien(foyer_id=client_choisi.id, date_releve=date_rel,
                                           index_compteur=index_kwh, duree_coupure_minutes=duree_minutes,
                                           temperature_exterieure=temp_val, cout_estime_fcfa=cout))
                    db.commit()
                    db.close()
                    st.success(f"✅ Relevé du {date_rel} enregistré !")
                    st.rerun()
        else:
            st.info("Ajoutez d'abord un client")

        # Appareils énergivores
        st.markdown("---")
        st.subheader("🔌 Analyse des Appareils Énergivores")
        if clients:
            client_app = st.selectbox("Client pour appareils", clients,
                                      format_func=lambda c: c.nom_utilisateur, key="client_app")
            db = db_session()
            apps = db.query(Appareil).filter(Appareil.foyer_id == client_app.id).all()
            db.close()

            if apps:
                app_data = []
                for a in apps:
                    conso_kwh = a.puissance_watts * a.heures_utilisation_jour * a.nombre_appareils / 1000
                    app_data.append({"Appareil": a.nom_appareil, "kWh/jour": round(conso_kwh, 2)})
                df_app = pd.DataFrame(app_data).sort_values("kWh/jour", ascending=False)
                fig_pie = px.pie(df_app, values="kWh/jour", names="Appareil", title="Consommation par appareil")
                st.plotly_chart(fig_pie, use_container_width=True)
                top = df_app.iloc[0]
                st.error(f"⚠️ **{top['Appareil']}** est le plus énergivore : **{top['kWh/jour']} kWh/jour**")

            with st.form("ajout_appareil"):
                nom_app = st.text_input("Nom de l'appareil")
                watts = st.number_input("Puissance (Watts)", min_value=1, value=100)
                heures = st.number_input("Heures/jour", min_value=0.1, value=1.0, step=0.5)
                if st.form_submit_button("➕ Ajouter l'appareil") and nom_app:
                    db = db_session()
                    db.add(Appareil(foyer_id=client_app.id, nom_appareil=nom_app,
                                    puissance_watts=watts, heures_utilisation_jour=heures, nombre_appareils=1))
                    db.commit()
                    db.close()
                    st.success(f"✅ '{nom_app}' ajouté !")
                    st.rerun()

# ================================================================
# PAGE 2 : DASHBOARD CLIENT
# ================================================================
elif menu == "📊 Dashboard Client":
    db = db_session()
    clients = db.query(Client).all()

    if not clients:
        st.warning("Aucun client. Ajoutez-en d'abord.")
        db.close()
    else:
        client_choisi = st.selectbox("Sélectionnez un client", clients,
                                     format_func=lambda c: f"{c.nom_utilisateur} - {c.region}")

        if client_choisi:
            releves = list(client_choisi.releves)
            appareils = db.query(Appareil).filter(Appareil.foyer_id == client_choisi.id).all()

            # ---- STATISTIQUES DESCRIPTIVES ----
            st.subheader(f"📊 Statistiques - {client_choisi.nom_utilisateur}")
            col1, col2, col3, col4, col5 = st.columns(5)

            conso_moy = round(sum(r.index_compteur for r in releves) / len(releves), 2) if releves else 0
            conso_max = max([r.index_compteur for r in releves]) if releves else 0
            conso_min = min([r.index_compteur for r in releves]) if releves else 0
            ecart_type = round(np.std([r.index_compteur for r in releves]), 2) if releves else 0
            coupure_moy = round(sum(r.duree_coupure_minutes for r in releves) / len(releves), 1) if releves else 0

            with col1: st.metric("⚡ Moyenne", f"{conso_moy} kWh")
            with col2: st.metric("📋 Relevés", len(releves))
            with col3: st.metric("🔺 Max", f"{conso_max} kWh")
            with col4: st.metric("📉 Écart-type", f"{ecart_type} kWh")
            with col5: st.metric("⏱️ Coupure moy.", f"{coupure_moy} min")

            # ---- GRAPHIQUE ÉVOLUTION ----
            if releves:
                st.markdown("---")
                st.subheader("📈 Évolution de la consommation")
                df_rel = pd.DataFrame([{"Date": r.date_releve, "Consommation (kWh)": r.index_compteur}
                                       for r in releves]).sort_values("Date")
                fig = px.line(df_rel, x="Date", y="Consommation (kWh)", markers=True)
                fig.update_traces(line_color="#ffd700", marker_color="#2c5364")
                st.plotly_chart(fig, use_container_width=True)

                # ---- RÉGRESSION LINÉAIRE SIMPLE ----
                st.markdown("---")
                st.subheader("🔬 1. Régression Linéaire Simple : Température vs Consommation")
                temp_data = [(r.temperature_exterieure, r.index_compteur) for r in releves
                             if r.temperature_exterieure is not None]
                if len(temp_data) >= 5:
                    X = [t[0] for t in temp_data]
                    y = [t[1] for t in temp_data]
                    result = regression_simple(X, y)

                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1: st.metric("R²", f"{result['r2']:.4f}")
                    with col_r2: st.metric("Pente", f"{result['slope']:.4f} kWh/°C")
                    with col_r3: st.metric("Intercept", f"{result['intercept']:.4f} kWh")

                    if result['r2'] > 0.6:
                        st.success("✅ Corrélation FORTE entre température et consommation")
                    elif result['r2'] > 0.3:
                        st.warning("⚠️ Corrélation MODÉRÉE")
                    else:
                        st.info("ℹ️ Corrélation FAIBLE")

                    # Graphique de régression
                    fig_reg = px.scatter(x=X, y=y, labels={"x": "Température (°C)", "y": "Consommation (kWh)"},
                                         title="Nuage de points + Droite de régression")
                    x_line = np.linspace(min(X), max(X), 100)
                    y_line = result['slope'] * x_line + result['intercept']
                    fig_reg.add_scatter(x=x_line, y=y_line, mode='lines', name='Régression',
                                        line=dict(color='red'))
                    st.plotly_chart(fig_reg, use_container_width=True)
                else:
                    st.info("Pas assez de données de température (minimum 5 relevés)")

            # ---- TABLEAU RELEVÉS ----
            st.markdown("---")
            st.subheader("📋 Derniers relevés")
            if releves:
                df_rel = pd.DataFrame([{"Date": str(r.date_releve), "kWh": r.index_compteur,
                                         "Coupure": f"{r.duree_coupure_minutes} min",
                                         "Temp.": f"{r.temperature_exterieure}°C" if r.temperature_exterieure else "N/A",
                                         "Coût": f"{r.cout_estime_fcfa} FCFA" if r.cout_estime_fcfa else "N/A"}
                                        for r in releves[:20]])
                st.dataframe(df_rel, use_container_width=True, hide_index=True)

    db.close()

# ================================================================
# PAGE 3 : ANALYSES AVANCÉES (Régression multiple, ACP, K-means)
# ================================================================
elif menu == "📈 Analyses Avancées":
    st.header("📈 Analyses Avancées - Programme INF 232 EC2")

    db = db_session()
    df = get_releves_data(db)
    df_clients = get_clients_stats(db)
    db.close()

    if df.empty or df_clients.empty:
        st.warning("Ajoutez plus de données (clients + relevés) pour voir les analyses.")
    else:
        # ---- 2. RÉGRESSION LINÉAIRE MULTIPLE ----
        st.subheader("🔬 2. Régression Linéaire Multiple")
        st.markdown("*Prédiction de la consommation (kWh) en fonction de : température + durée coupure*")

        # Préparer les données
        df_valid = df[df['temperature'] > 0].copy()
        if len(df_valid) >= 10:
            X_multi = df_valid[['temperature', 'coupure_min']].values
            y_multi = df_valid['kWh'].values
            result_multi = regression_multiple(X_multi, y_multi)

            if result_multi:
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1: st.metric("R² multiple", f"{result_multi['r2']:.4f}")
                with col_r2: st.metric("Coef. Température", f"{result_multi['coefficients'][0]:.4f}")
                with col_r3: st.metric("Coef. Coupure", f"{result_multi['coefficients'][1]:.4f}")

                st.markdown(f"""
                **Équation :** Consommation = {result_multi['intercept']:.2f} 
                + {result_multi['coefficients'][0]:.2f} × Température 
                + {result_multi['coefficients'][1]:.2f} × Coupure
                """)

                # Graphique valeurs réelles vs prédites
                fig_multi = px.scatter(x=y_multi, y=result_multi['y_pred'],
                                       labels={"x": "Valeurs réelles (kWh)", "y": "Valeurs prédites (kWh)"},
                                       title="Régression multiple : Réel vs Prédit")
                fig_multi.add_scatter(x=[y_multi.min(), y_multi.max()],
                                      y=[y_multi.min(), y_multi.max()],
                                      mode='lines', name='Parfait', line=dict(color='red', dash='dash'))
                st.plotly_chart(fig_multi, use_container_width=True)
            else:
                st.error("Erreur dans le calcul de la régression multiple.")
        else:
            st.info("Pas assez de données (minimum 10 relevés avec température)")

        # ---- 3. ACP - Analyse en Composantes Principales ----
        st.markdown("---")
        st.subheader("📉 3. ACP - Réduction des Dimensions")
        st.markdown("*Projection des clients sur 2 axes principaux*")

        if len(df_clients) >= 4:
            X_acp = df_clients[['moyenne_kwh', 'ecart_type_kwh', 'max_kwh', 'min_kwh']].values
            result_acp = acp_analyse(X_acp, n_components=2)

            df_pca = pd.DataFrame({
                "Composante 1": result_acp['X_pca'][:, 0],
                "Composante 2": result_acp['X_pca'][:, 1],
                "Client": df_clients['nom'].values,
                "Région": df_clients['region'].values
            })

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.metric("Variance expliquée (PC1)", f"{result_acp['variance_expliquee'][0]*100:.1f}%")
            with col_v2:
                st.metric("Variance expliquée (PC2)", f"{result_acp['variance_expliquee'][1]*100:.1f}%")

            fig_acp = px.scatter(df_pca, x="Composante 1", y="Composante 2",
                                 text="Client", color="Région",
                                 title="ACP - Projection des clients",
                                 size=[10]*len(df_pca))
            fig_acp.update_traces(textposition='top center')
            st.plotly_chart(fig_acp, use_container_width=True)
            st.info("Chaque point représente un client. Les clients proches ont des profils de consommation similaires.")
        else:
            st.info("Pas assez de clients (minimum 4)")

        # ---- 5. K-MEANS - Classification Non Supervisée ----
        st.markdown("---")
        st.subheader("🔄 5. K-Means - Classification Non Supervisée")
        st.markdown("*Regroupement automatique des clients en 3 clusters*")

        if len(df_clients) >= 6:
            X_km = df_clients[['moyenne_kwh', 'ecart_type_kwh']].values
            k = st.slider("Nombre de clusters (k)", 2, 5, 3)
            result_km = kmeans_clustering(X_km, k=k)

            df_clusters = df_clients.copy()
            df_clusters['Cluster'] = [f"Cluster {l+1}" for l in result_km['labels']]
            df_clusters['X'] = X_km[:, 0]
            df_clusters['Y'] = X_km[:, 1]

            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.metric("Inertie", f"{result_km['inertia']:.2f}")
            with col_k2:
                st.metric("Itérations", result_km['iterations'])

            fig_km = px.scatter(df_clusters, x="X", y="Y", text="nom",
                                color="Cluster", title="K-Means - Clusters de consommation")
            fig_km.update_traces(textposition='top center')

            # Ajouter les centroïdes
            for i, centroid in enumerate(result_km['centroids']):
                fig_km.add_scatter(x=[centroid[0]], y=[centroid[1]],
                                   mode='markers+text', text=[f"C{i+1}"],
                                   textposition='bottom center',
                                   marker=dict(size=20, symbol='x', color='black'),
                                   showlegend=False)

            st.plotly_chart(fig_km, use_container_width=True)

            # Résumé des clusters
            st.markdown("**Résumé des clusters :**")
            for i in range(k):
                cluster_data = df_clusters[df_clusters['Cluster'] == f"Cluster {i+1}"]
                if len(cluster_data) > 0:
                    st.markdown(f"""
                    **Cluster {i+1}** ({len(cluster_data)} clients) : 
                    Conso moyenne = {cluster_data['moyenne_kwh'].mean():.1f} kWh | 
                    Clients : {', '.join(cluster_data['nom'].tolist()[:3])}
                    """)
        else:
            st.info("Pas assez de clients (minimum 6)")

        # ---- 4. CLASSIFICATION SUPERVISÉE ----
        st.markdown("---")
        st.subheader("📊 4. Classification Supervisée")
        st.markdown("*Classification : consommation > seuil = anormale*")

        seuil = st.number_input("Seuil de consommation anormale (kWh)", value=20.0, step=1.0)
        df_clients['anormale'] = (df_clients['moyenne_kwh'] > seuil).astype(int)
        n_anormales = df_clients['anormale'].sum()

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Clients > seuil (anormaux)", f"{n_anormales}/{len(df_clients)}")
        with col_s2:
            st.metric("% Anormaux", f"{n_anormales/len(df_clients)*100:.1f}%")

        fig_class = px.bar(df_clients, x="nom", y="moyenne_kwh",
                           color="anormale",
                           color_discrete_map={0: "green", 1: "red"},
                           labels={"moyenne_kwh": "Consommation moyenne (kWh)", "nom": "Client"},
                           title=f"Classification : Seuil = {seuil} kWh")
        fig_class.add_hline(y=seuil, line_dash="dash", line_color="red",
                            annotation_text=f"Seuil: {seuil} kWh")
        st.plotly_chart(fig_class, use_container_width=True)

# ================================================================
# PAGE 4 : EXPORT EXCEL
# ================================================================
elif menu == "📥 Export Excel":
    st.subheader("📥 Télécharger les données en Excel")
    db = db_session()
    clients = db.query(Client).all()

    if not clients:
        st.warning("Aucun client")
        db.close()
    else:
        client_choisi = st.selectbox("Choisissez un client", clients,
                                     format_func=lambda c: c.nom_utilisateur)
        if client_choisi and st.button("📥 Télécharger le fichier Excel"):
            releves = list(client_choisi.releves)
            appareils = db.query(Appareil).filter(Appareil.foyer_id == client_choisi.id).all()
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_rel = pd.DataFrame([{"Date": r.date_releve, "kWh": r.index_compteur,
                                         "Coupure (min)": r.duree_coupure_minutes,
                                         "Temp. (°C)": r.temperature_exterieure or 0,
                                         "Coût (FCFA)": r.cout_estime_fcfa or 0} for r in releves])
                df_rel.to_excel(writer, sheet_name="Relevés", index=False)
                if appareils:
                    df_app = pd.DataFrame([{"Appareil": a.nom_appareil, "Qté": a.nombre_appareils,
                                             "Watts": a.puissance_watts, "h/j": a.heures_utilisation_jour,
                                             "kWh/j": round(a.puissance_watts*a.heures_utilisation_jour*a.nombre_appareils/1000, 2)}
                                            for a in appareils])
                    df_app.to_excel(writer, sheet_name="Appareils", index=False)
            st.download_button("📥 Cliquer pour télécharger", data=output.getvalue(),
                               file_name=f"WattScope_{client_choisi.nom_utilisateur}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("✅ Fichier prêt !")
        db.close()

# ---------- PIED DE PAGE ----------
st.markdown("---")
st.caption("⚡ WattScope | TP INF 232 EC2 | Analyse de données | Lawane Oscar 24G2206")
