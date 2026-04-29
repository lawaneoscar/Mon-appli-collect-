from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base
import datetime

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
