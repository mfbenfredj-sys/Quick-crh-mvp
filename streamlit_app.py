# app.py
# Quick CRH — Streamlit + OpenAI (SDK v1)
# Utilise la clé depuis st.secrets["OPENAI_API_KEY"] (fallback ENV local)

import os
from datetime import datetime

import streamlit as st
from openai import OpenAI

# =========================
#        CONFIG UI
# =========================
st.set_page_config(page_title="Quick CRH", layout="wide")
st.title("📝 Assistant CRH — QuickCRH")
st.caption("Génération assistée, mode libre et mode didactique")

MODEL_NAME = "gpt-4o"  # centralise le modèle ici

# =========================
#   CONSTANTES / PATCHS
# =========================
STYLE_GUIDE = """
STYLE GUIDE CRH — à respecter strictement :
- Ton clinique, sobre, phrases courtes, pas de superlatifs ni d’emphase inutile.
- Aucune invention : si une information manque, l’omettre ou écrire « non communiqué ».
- Ordre des sections : Motif → Contexte/ATCD pertinents → HDM → Clinique (entrée) → Examens → Évolution/Actes
  → Diagnostics de sortie (clairs, listés) → Traitement de sortie (adaptations explicites) → Recommandations & Suivi.
- Cohérence stricte : diagnostics de sortie déduits logiquement de HDM + examens + évolution.
- Médico-légal : datation relative ok, pas d’identifiants, pas de jugement de valeur.
- Lisibilité : titres visibles, paragraphes courts, éviter les redondances.
"""

REF_EXAMPLE = """
EXEMPLE DE CRH (générique, concis)
MOTIF
Douleurs thoraciques constrictives évoluant depuis 24 h.

CONTEXTE
ATCD : HTA, dyslipidémie. Non-fumeur. Pas d’allergie connue.

HISTOIRE DE LA MALADIE
Douleur rétrosternale à l’effort, cédant au repos, associée à dyspnée légère. Pas de syncope.

EXAMEN CLINIQUE (ENTRÉE)
TA 138/82 mmHg, FC 84/min, SpO2 98% AA. Auscultation cardio-pulmonaire sans particularité. Pas d’œdèmes.

EXAMENS COMPLÉMENTAIRES
ECG : sous-décalage ST V4–V6. Troponines élevées. ETT : cinétique segmentaire discrètement altérée.

ÉVOLUTION / ACTES
Traitement : AAS, clopidogrel, statine, bêtabloquant. Coroscanner : sténose ADA moyenne. Pas de complication.

DIAGNOSTICS DE SORTIE
- Angor instable sur maladie coronaire.

TRAITEMENT DE SORTIE
AAS 75 mg/j, clopidogrel 75 mg/j (12 mois), atorvastatine 40 mg/j, bisoprolol 2,5 mg/j. AINS arrêt.

RECOMMANDATIONS & SUIVI
Consultation cardio à 2 semaines. Réadaptation cardiaque. Éducation : reconnaissance douleur, appel 15 si récidive.
"""

HAS_QUALITY_BLOCK = """
Conformité HAS – Indicateur « Qualité du document de sortie » (IPAQSS, MCO) :
Le CRH doit contenir au minimum les 12 critères suivants :
(1) Motif d’hospitalisation ; (2) Synthèse médicale du séjour ; (3) Actes techniques, examens complémentaires et biologiques (résultats principaux) ;
(4) Traitements médicamenteux (entrée/séjour/sortie si pertinent) ; (5) Suites à donner / conduite à tenir / suivi ;
(6) Nom et coordonnées du médecin traitant ; (7) Identification du patient ; (8) Dates d’entrée et de sortie ; (9) Date de rédaction ;
(10) Nom et coordonnées du médecin rédacteur ; (11) Nature du document ; (12) Destination du patient à la sortie.
Exigences de forme : rédaction claire et structurée, titres distincts, abréviations limitées et reconnues, données exactes sans invention.
Transmission le jour de la sortie au patient et au médecin destinataire lorsque possible.
"""

# =========================
#      CLE & CLIENT
# =========================
def _read_api_key():
    # Priorité aux secrets Streamlit (déploiement)
    key = st.secrets.get("OPENAI_API_KEY", None)
    # Fallback local pratique si on lance hors cloud
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    return key

def get_openai_client() -> OpenAI | None:
    key = _read_api_key()
    if not key:
        st.error("❌ Aucune clé API trouvée. Renseigne `OPENAI_API_KEY` dans les **Secrets** Streamlit (ou exporte la variable d'environnement en local).")
        return None
    try:
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Erreur d'initialisation OpenAI : {e}")
        return None

def call_llm(prompt: str, temperature: float, max_tokens: int) -> str:
    client = get_openai_client()
    if client is None:
        return ""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Erreur lors de la génération : {e}")
        return ""

def download_button_from_text(text: str, filename_prefix: str = "CRH"):
    if not text:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_txt = f"{filename_prefix}_{ts}.txt"
    st.download_button("💾 Télécharger (.txt)", data=text.encode("utf-8"),
                       file_name=filename_txt, mime="text/plain")

# =========================
#       SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Paramètres")
    # plus de champ de clé ici — on lit st.secrets / ENV
    mode = st.radio("🧭 Mode d’utilisation", ["Champs guidés", "Mode libre", "Mode didactique"], index=0)

    with st.expander("⚙️ Paramètres avancés", expanded=False):
        st.caption("Ces réglages influencent la forme du texte, non le fond médical.")
        temperature = st.slider("🎛️ Créativité (temp.)", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("🧱 Longueur max (tokens)", 500, 4000, 1800, 100)

    st.markdown("---")
    st.header("🎯 Pertinence")
    use_style_guide = st.checkbox("Appliquer le Style Guide", value=True)
    use_example = st.checkbox("Inclure un exemple de référence", value=True)
    example_text = st.text_area("Exemple éditable", REF_EXAMPLE, height=210)
    use_has_quality = st.checkbox("Inclure le bloc Qualité HAS (12 critères)", value=True)

# =========================
#     MODE 1 — GUIDÉ
# =========================
if mode == "Champs guidés":
    st.subheader("📋 Saisie structurée")
    with st.form("guided_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            identite = st.text_input("👤 Identité (Nom, Prénom, Âge, Sexe)")
            ipp = st.text_input("🏷️ IPP / N° dossier")
            service = st.text_input("🏥 Service / UF")
        with col2:
            date_entree = st.date_input("📅 Date d'entrée")
            date_sortie = st.date_input("📅 Date de sortie")
            medecin = st.text_input("👨‍⚕️ Médecin référent")
        with col3:
            diagnostic_principal = st.text_area("🎯 Diagnostic principal")
            diagnostics_associes = st.text_area("➕ Diagnostics associés")
            motif = st.text_area("📝 Motif d'hospitalisation")

        st.markdown("---")
        hdm = st.text_area("📖 Histoire de la maladie (HDM)")
        clinique = st.text_area("🩺 Examen clinique à l'entrée")
        examens = st.text_area("🧪 Examens complémentaires")
        evolution = st.text_area("📈 Évolution / actes réalisés")

        traitement_entree = st.text_area("💊 Traitement habituel à l'entrée")
        traitement_sortie = st.text_area("💊 Traitement de sortie / modifications")

        situation_sociale = st.text_area("🏠 Situation sociale / mode de vie")
        suivi = st.text_area("📞 Recommandations / suivi post-hospitalisation")

        colx, coly = st.columns([1,1])
        with colx:
            style_bref = st.checkbox("🧾 Résumé succinct (≤ 1 page)", value=False)
        with coly:
            anonymiser = st.checkbox("🕶️ Anonymiser noms/identifiants", value=False)

        submitted = st.form_submit_button("🧠 Générer le CRH")

    if submitted:
        preamble_parts = [
            "Tu es un médecin hospitalier expérimenté. Rédige un CRH professionnel, clair et cohérent.",
            "Ne pas inventer : omettre ce qui manque ou indiquer « non communiqué ».",
            "Structure stricte avec titres visibles. Diagnostics de sortie clairement listés en section dédiée."
        ]
        if use_style_guide:
            preamble_parts.append(STYLE_GUIDE)
        if use_has_quality:
            preamble_parts.append(HAS_QUALITY_BLOCK)
        if use_example and example_text.strip():
            preamble_parts.append("RÉFÉRENCE À IMITER (style et structure) ↓\n" + example_text.strip())

        preamble = "\n\n".join(preamble_parts)

        prompt = f"""
{preamble}

{"Contrainte : vise un résumé ≤ 1 page." if style_bref else ""}
{"Anonymiser les identifiants (noms, IPP) s'ils apparaissent." if anonymiser else ""}

=== MÉTADONNÉES
Identification du patient: {identite or "non communiqué"}
IPP: {ipp or "non communiqué"}
Médecin traitant (si connu): {"non communiqué"}
Service: {service or "non communiqué"}
Médecin référent: {medecin or "non communiqué"}
Dates: entrée {date_entree}, sortie {date_sortie}
Date de rédaction: {datetime.now().date().isoformat()}
Nature du document: Compte Rendu d’Hospitalisation
Destination prévue à la sortie: {"non communiqué"}

=== DIAGNOSTICS
Diagnostic principal: {diagnostic_principal or "non communiqué"}
Diagnostics associés: {diagnostics_associes or "non communiqué"}
Motif d’hospitalisation: {motif or "non communiqué"}

=== HISTOIRE ET DONNÉES
HDM: {hdm or "non communiqué"}
Clinique (entrée): {clinique or "non communiqué"}
Examens complémentaires: {examens or "non communiqué"}
Évolution / actes: {evolution or "non communiqué"}

=== TRAITEMENTS
Traitement à l'entrée: {traitement_entree or "non communiqué"}
Traitement de sortie: {traitement_sortie or "non communiqué"}

=== CONTEXTE SOCIAL & SUIVI
Situation sociale: {situation_sociale or "non communiqué"}
Recommandations / suivi: {suivi or "non communiqué"}
"""
        with st.spinner("Génération du CRH en cours…"):
            crh = call_llm(prompt, temperature=temperature, max_tokens=max_tokens)

        st.markdown("### 🧾 CRH généré")
        st.write(crh)
        download_button_from_text(crh, filename_prefix="CRH")

# =========================
#    MODE 2 — LIBRE
# =========================
elif mode == "Mode libre":
    st.subheader("✍️ Rédaction / édition libre")
    instruction = st.selectbox(
        "Que souhaites-tu faire ?",
        [
            "Réécrire le texte pour un style CRH pro",
            "Corriger/compléter en gardant la structure",
            "Synthétiser en une page",
            "Traduire en français simple pour le patient (plain language)",
        ],
        index=0
    )
    texte_source = st.text_area("Colle ton brouillon / texte source", height=300,
                                placeholder="Colle ici ton CRH ou notes brutes…")
    bouton = st.button("✨ Améliorer / Générer")

    if bouton and texte_source.strip():
        preamble = []
        if use_style_guide:
            preamble.append(STYLE_GUIDE)
        if use_has_quality:
            preamble.append(HAS_QUALITY_BLOCK)
        if use_example and example_text.strip():
            preamble.append("RÉFÉRENCE À IMITER ↓\n" + example_text.strip())

        prompt = f"""
Tu es un rédacteur médical hospitalier.
Objectif: {instruction}.
Consignes générales:
- Conserver l'exactitude médicale, ne pas inventer de données.
- Respecter un format CRH lisible (titres, paragraphes courts, transitions sobres).
- Mettre en avant le diagnostic de sortie et la conduite à tenir.

{'\n\n'.join(preamble).strip()}

TEXTE SOURCE
------------
{texte_source}
"""
        with st.spinner("Traitement en cours…"):
            sortie = call_llm(prompt, temperature=temperature, max_tokens=max_tokens)

        st.markdown("### 🧾 Sortie")
        st.write(sortie)
        download_button_from_text(sortie, filename_prefix="CRH_libre")

# =========================
#  MODE 3 — DIDACTIQUE
# =========================
else:
    st.subheader("🎓 Mode didactique (coaching pour internes)")
    st.write("Colle un CRH (ou un cas) et obtiens : checklist, points forts/faibles, axes d’amélioration, version corrigée, questions de réflexion.")
    colA, colB = st.columns([2,1])
    with colA:
        brouillon = st.text_area("Ton CRH / cas clinique", height=320, placeholder="Colle ici ton brouillon de CRH…")
    with colB:
        niveau = st.selectbox("Niveau", ["DFASM1", "DFASM2", "DFASM3 / Interne"], index=2)
        severite = st.select_slider("Sévérité de l’évaluation", options=["Bienveillance", "Standard", "Exigeant"], value="Standard")

    lancer = st.button("🧠 Analyser pédagogiquement")

    if lancer and brouillon.strip():
        preamble = []
        if use_style_guide:
            preamble.append(STYLE_GUIDE)
        if use_has_quality:
            preamble.append(HAS_QUALITY_BLOCK)
        if use_example and example_text.strip():
            preamble.append("RÉFÉRENCE À IMITER ↓\n" + example_text.strip())

        prompt = f"""
Tu es un encadrant hospitalier (chef de clinique). Niveau de l'étudiant: {niveau}. Sévérité: {severite}.
Objectif : retour pédagogique structuré et actionnable.

Livrables :
1) ✅ Checklist CRH vs critères requis (motif, synthèse du séjour, examens/actes, traitements, suites à donner, éléments administratifs).
2) ⭐ Points forts (3–5 bullets).
3) 🧱 Points à améliorer (5–8 bullets concrets).
4) 🛠️ Corrections/Propositions (exemples de reformulation par section).
5) 🧪 Pièges fréquents à éviter pour ce type de cas.
6) 📌 Version CRH corrigée (propre, directement utilisable).
7) ❓ 3–5 questions courtes de réflexion (contrôle des acquis).

{'\n\n'.join(preamble).strip()}

CRH À ÉVALUER
-------------
{brouillon}
"""
        with st.spinner("📚 Analyse pédagogique en cours…"):
            feedback = call_llm(prompt, temperature=temperature, max_tokens=max_tokens)

        st.markdown("### 🧠 Retour pédagogique de l’IA")
        st.write(feedback)
        download_button_from_text(feedback, filename_prefix="CRH_didactique")
