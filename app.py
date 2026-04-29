# app.py - WattScope - Application Streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ---------- BASE DE DONNÉES ----------
Base = declarative_base()

class Client(Base):
    __tablename__ = "foyers"
    id = Column(Integer, primary_key=True)
    nom_utilisateur = Column(String, unique=True, nullable=False)
    region = Column(String, default="Yaoundé")
    type_logement = Column(String, default="Appartement")
    nombre_habitants = Column(Integer, default=1)
    date_inscription = Column(DateTime, default=datetime.datetime.utcnow)
    releves = relationship("ReleveQuotidien", back_populates="foyer")
    appareils = relationship("Appareil", back_populates="foyer")

class ReleveQuotidien(Base):
    __tablename__ = "releves_quotidiens"
    id = Column(Integer, primary_key=True)
    foyer_id = Column(Integer, ForeignKey("foyers.id"), nullable=False)
    date_releve = Column(Date, nullable=False)
    index_compteur = Column(Float, nullable=False)
    duree_coupure_minutes = Column(Integer, default=0)
    temperature_exterieure = Column(Float, nullable=True)
    cout_estime_fcfa = Column(Float, nullable=True)
    foyer = relationship("Client", back_populates="releves")

class Appareil(Base):
    __tablename__ = "appareils"
    id = Column(Integer, primary_key=True)
    foyer_id = Column(Integer, ForeignKey("foyers.id"), nullable=False)
    nom_appareil = Column(String, nullable=False)
    puissance_watts = Column(Integer)
    heures_utilisation_jour = Column(Float)
    nombre_appareils = Column(Integer, default=1)
    foyer = relationship("Client", back_populates="appareils")

@st.cache_resource
def get_db():
    engine = create_engine("sqlite:///wattscope.db")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)

# ---------- INTERFACE ----------
st.set_page_config(page_title="⚡ WattScope", page_icon="⚡", layout="wide")
st.title("⚡ WattScope - Analyse de Consommation Électrique")
st.markdown("**TP INF 232 EC2** | *Suivi et Analyse Descriptive des Données*")

db_session = get_db()

# ---------- BARRE LATÉRALE ----------
menu = st.sidebar.radio("📋 Menu", ["🏠 Accueil & Collecte", "📊 Dashboard Client", "📥 Export Excel"])

# ---------- PAGE ACCUEIL & COLLECTE ----------
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
                "Relevés": len(c.releves)
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun client enregistré")

# ---------- DASHBOARD CLIENT ----------
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
            
            # Stats
            st.subheader(f"📊 Statistiques - {client_choisi.nom_utilisateur}")
            col1, col2, col3, col4 = st.columns(4)
            
            conso_moy = round(sum(r.index_compteur for r in releves) / len(releves), 2) if releves else 0
            conso_max = max([r.index_compteur for r in releves]) if releves else 0
            coupure_moy = round(sum(r.duree_coupure_minutes for r in releves) / len(releves), 1) if releves else 0
            
            with col1:
                st.metric("⚡ kWh/jour (moy.)", f"{conso_moy} kWh")
            with col2:
                st.metric("📋 Total relevés", len(releves))
            with col3:
                st.metric("🔺 Conso max", f"{conso_max} kWh")
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
                
                # Régression température (avec numpy uniquement)
                st.markdown("---")
                st.subheader("🔬 Régression : Température vs Consommation")
                temp_data = [(r.temperature_exterieure, r.index_compteur) for r in releves if r.temperature_exterieure is not None]
                if len(temp_data) >= 5:
                    X = np.array([t[0] for t in temp_data])
                    y = np.array([t[1] for t in temp_data])
                    
                    # Régression linéaire manuelle (formules du cours)
                    X_mean = np.mean(X)
                    y_mean = np.mean(y)
                    slope = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
                    intercept = y_mean - slope * X_mean
                    y_pred = slope * X + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y_mean) ** 2)
                    r2 = 1 - (ss_res / ss_tot)
                    
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.metric("R²", f"{r2:.4f}")
                    with col_r2:
                        st.metric("Coefficient", f"{slope:.4f} kWh/°C")
                    
                    if r2 > 0.6:
                        st.success("Corrélation forte entre température et consommation")
                    elif r2 > 0.3:
                        st.warning("Corrélation modérée")
                    else:
                        st.info("Corrélation faible")
                else:
                    st.info("Pas assez de données de température (min. 5 relevés)")
            
            # Appareils énergivores
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
                    df_app = pd.DataFrame(app_data)
                    df_app = df_app.sort_values("kWh/jour", ascending=False)
                    
                    # Camembert
                    fig_pie = px.pie(df_app, values="kWh/jour", names="Appareil", title="Consommation par appareil")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Top énergivore
                    top = df_app.iloc[0]
                    st.error(f"⚠️ **{top['Appareil']}** est le plus énergivore : **{top['kWh/jour']} kWh/jour**")
                else:
                    st.info("Aucun appareil enregistré")
            
            with col_app2:
                # Ajouter un appareil
                st.markdown("**➕ Ajouter un appareil**")
                with st.form("ajout_appareil"):
                    nom_app = st.text_input("Nom de l'appareil")
                    watts = st.number_input("Puissance (Watts)", min_value=1, value=100)
                    heures = st.number_input("Heures d'utilisation par jour", min_value=0.1, value=1.0, step=0.5)
                    if st.form_submit_button("➕ Ajouter"):
                        if nom_app:
                            db_add = db_session()
                            db_add.add(Appareil(foyer_id=client_choisi.id, nom_appareil=nom_app, puissance_watts=watts, heures_utilisation_jour=heures, nombre_appareils=1))
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

# ---------- EXPORT EXCEL ----------
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
                # Feuille Relevés
                df_rel = pd.DataFrame([{
                    "Date": r.date_releve,
                    "Index compteur (kWh)": r.index_compteur,
                    "Durée coupure (min)": r.duree_coupure_minutes,
                    "Température (°C)": r.temperature_exterieure or 0,
                    "Coût estimé (FCFA)": r.cout_estime_fcfa or 0
                } for r in releves])
                df_rel.to_excel(writer, sheet_name="Relevés", index=False)
                
                # Feuille Appareils
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
st.caption("⚡ WattScope | TP INF 232 EC2 | Analyse de données")
