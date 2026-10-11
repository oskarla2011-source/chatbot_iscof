from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Dict

from PIL import Image
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer, util

# ============================================================
# ISCOF — Assistant FAQ formations
# Base de connaissances adaptée à la page Formation et à la FAQ
# fournie en capture. Les informations non confirmées ne sont
# pas inventées : le bot invite alors à contacter ISCOF.
# ============================================================

st.set_page_config(
    page_title="Assistant formations | ISCOF",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -------------------- Identité visuelle --------------------
ORANGE = "#C45A0A"
INK = "#171B20"
MUTED = "#687383"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,500;1,600&display=swap');
    :root {{ --iscof-orange: {ORANGE}; --iscof-ink: {INK}; --iscof-muted: {MUTED}; }}
    html, body, [data-testid="stAppViewContainer"] {{ background:#fffdfb !important; color:var(--iscof-ink); font-family:'DM Sans',sans-serif; }}
    [data-testid="stHeader"] {{ background:transparent; }}
    #MainMenu, footer {{ visibility:hidden; }}
    .block-container {{ max-width:900px; padding-top:1.7rem; padding-bottom:3rem; }}
    .iscof-kicker {{ color:var(--iscof-orange); font-size:.75rem; font-weight:700; letter-spacing:2.2px; text-transform:uppercase; margin:0 0 12px 0; }}
    .iscof-title {{ font-family:'Playfair Display',Georgia,serif; color:var(--iscof-ink); font-size:clamp(2rem,5vw,3.15rem); line-height:1.12; margin:0 0 10px 0; }}
    .iscof-title em {{ color:var(--iscof-orange); font-weight:500; }}
    .iscof-subtitle {{ color:var(--iscof-muted); font-size:1rem; line-height:1.65; margin-bottom:18px; }}
    .iscof-trust {{ border:1px solid #f0d2bd; background:#fffaf6; border-radius:14px; padding:12px 15px; color:#6c4a35; font-size:.86rem; margin:18px 0 20px; }}
    [data-testid="stChatMessage"] {{ border-radius:16px !important; padding:13px 16px !important; margin-bottom:11px !important; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{ background:#fffaf6 !important; border:1px solid #f2e5dc !important; border-left:3px solid var(--iscof-orange) !important; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background:#f5f5f6 !important; border:1px solid #e9ebee !important; }}
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ color:var(--iscof-ink) !important; line-height:1.7; }}

    /* Zone de saisie : champ doux, contraste net et bouton d'envoi distinct */
    [data-testid="stChatInput"] {{
        border:1.5px solid #ead3c4 !important;
        border-radius:18px !important;
        background:linear-gradient(135deg,#ffffff 0%,#fffaf6 100%) !important;
        box-shadow:0 5px 20px rgba(74,43,24,.07) !important;
        padding:5px 7px 5px 11px !important;
        transition:border-color .2s ease, box-shadow .2s ease, transform .2s ease !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color:var(--iscof-orange) !important;
        box-shadow:0 0 0 4px rgba(196,90,10,.10),0 8px 24px rgba(74,43,24,.09) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background:transparent !important;
        color:var(--iscof-ink) !important;
        font-family:'DM Sans',sans-serif !important;
        font-size:.96rem !important;
        line-height:1.5 !important;
        padding:12px 9px !important;
        min-height:48px !important;
        box-shadow:none !important;
        outline:none !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{ color:#9297a0 !important; opacity:1 !important; }}
    [data-testid="stChatInput"] button {{
        background:var(--iscof-orange) !important;
        color:#fff !important;
        border:0 !important;
        border-radius:12px !important;
        min-width:40px !important;
        min-height:40px !important;
        box-shadow:0 3px 9px rgba(196,90,10,.22) !important;
        transition:background .2s ease, transform .2s ease, box-shadow .2s ease !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        background:#a94805 !important;
        transform:translateY(-1px) !important;
        box-shadow:0 5px 13px rgba(196,90,10,.27) !important;
    }}
    [data-testid="stChatInput"] button svg {{ color:white !important; fill:white !important; }}

    /* Trois points animés l'un après l'autre pendant la génération */
    .iscof-thinking {{ display:flex; align-items:center; gap:11px; padding:7px 2px 10px; color:#86624b; font-size:.88rem; }}
    .iscof-dots {{ display:inline-flex; align-items:center; gap:5px; min-width:30px; }}
    .iscof-dot {{ width:7px; height:7px; display:block; border-radius:50%; background:var(--iscof-orange); opacity:.25; transform:translateY(0) scale(.78); animation:iscof-dot-pulse 1.05s infinite ease-in-out; }}
    .iscof-dot:nth-child(2) {{ animation-delay:.17s; }}
    .iscof-dot:nth-child(3) {{ animation-delay:.34s; }}
    @keyframes iscof-dot-pulse {{ 0%,60%,100% {{ opacity:.25; transform:translateY(0) scale(.78); }} 30% {{ opacity:1; transform:translateY(-3px) scale(1); }} }}
    @media (prefers-reduced-motion:reduce) {{ .iscof-dot {{ animation:none; opacity:.8; }} }}
    .faq-label {{ color:var(--iscof-orange); font-size:.72rem; font-weight:700; letter-spacing:1.7px; text-transform:uppercase; margin:20px 0 10px; }}
    .stButton > button {{ border:1px solid #e9c3ab !important; background:white !important; color:#63351d !important; border-radius:12px !important; min-height:46px; font-size:.85rem !important; transition:all .15s ease; }}
    .stButton > button:hover {{ border-color:var(--iscof-orange) !important; background:#fff7f1 !important; color:var(--iscof-orange) !important; }}
    .footer-note {{ border-top:1px solid #eee5df; margin-top:26px; padding-top:15px; color:#78808b; font-size:.78rem; line-height:1.6; }}
    @media(max-width:640px) {{ .block-container {{ padding-left:1rem; padding-right:1rem; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Configuration API --------------------
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("La clé API Google est introuvable. Ajoutez GOOGLE_API_KEY dans les Secrets de Streamlit Cloud ou dans votre fichier .env / environnement.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

@st.cache_resource(show_spinner="Préparation de l’assistant…")
def charger_modele():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

modele = charger_modele()

# Logo ISCOF utilisé comme avatar du chatbot (fichier placé à côté du script).
LOGO_AVATAR_PATH = Path(__file__).with_name("logo_iscof_avatar.png")
try:
    LOGO_AVATAR = Image.open(LOGO_AVATAR_PATH).convert("RGB")
except Exception:
    # Repli visuel si le fichier image n'a pas été copié avec le script.
    LOGO_AVATAR = "🎓"

# -------------------- Base de connaissances --------------------
# Les réponses sont volontairement limitées aux informations connues.
KNOWLEDGE_CHUNKS: List[Dict[str, object]] = [
    {
        "id": "choisir_formation",
        "keywords": ["choisir", "quelle formation", "adaptée", "projet", "niveau", "positionnement", "rendez-vous", "rdv", "débutant"],
        "text": "CHOISIR SA FORMATION : Lors du premier rendez-vous gratuit, ISCOF valide avec la personne la formation adaptée à son projet et à son niveau de départ. Des tests de positionnement sont inclus. Le site propose notamment des parcours RH/paie, commerce, relation client, formateur professionnel d'adultes, anglais TOEIC et bureautique TOSA."
    },
    {
        "id": "reconnaissance",
        "keywords": ["reconnu", "reconnaissance", "titre", "certification", "rncp", "rs", "état", "employeur", "france compétences", "diplôme"],
        "text": "RECONNAISSANCE DES TITRES : D'après la FAQ ISCOF, les Titres Professionnels sont enregistrés au RNCP et les certifications au Répertoire spécifique (RS). Le site indique qu'ils sont reconnus par les employeurs et l'État, et renvoie à la liste officielle France Compétences. Le type et le niveau exacts dépendent de la formation choisie."
    },
    {
        "id": "delai_demarrage",
        "keywords": ["délai", "démarrer", "commencer", "début", "date", "quand", "attendre", "inscription", "financement", "15 jours"],
        "text": "DÉLAI DE DÉMARRAGE : Selon la FAQ de la page Formation, le démarrage a lieu 15 jours après l'accord de financement. Des sessions de démarrage sont organisées régulièrement en présentiel ; en distanciel, le démarrage est annoncé en continu. Le délai réel dépend donc notamment de l'accord de financement."
    },
    {
        "id": "echec_certification",
        "keywords": ["échec", "échoue", "rate", "raté", "repasser", "repassage", "certification", "examen", "tentative", "frais supplémentaires"],
        "text": "EN CAS DE NON-OBTENTION DE LA CERTIFICATION : La FAQ ISCOF indique que la personne peut repasser la certification sans frais supplémentaires lors de la session suivante. ISCOF précise que son objectif est d'aider les stagiaires à valider dès la première tentative."
    },
    {
        "id": "catalogue",
        "keywords": ["catalogue", "formations", "formation disponible", "rh", "paie", "commerce", "relation client", "formateur", "anglais", "toeic", "tosa", "office", "excel", "word"],
        "text": "CATALOGUE DES FORMATIONS : La page ISCOF présente les parcours suivants : Titre Professionnel Assistant(e) RH (niveau 5, Bac+2, environ 8 mois) ; Négociateur Technico-Commercial (Titre Pro RNCP niveau 5, 6 à 9 mois) ; Conseiller Relation Client (niveau 4, 4 à 8 mois) ; Formateur Professionnel d'Adultes (niveau 5, 6 à 10 mois) ; préparation à la certification d'anglais TOEIC (3 à 6 mois) ; Pack Office TOSA Word, Excel, PowerPoint et Outlook (1 à 3 mois). Vérifier la fiche de chaque parcours pour les détails et les conditions en vigueur."
    },
    {
        "id": "formats",
        "keywords": ["distanciel", "distance", "visio", "présentiel", "vincennes", "hybride", "en ligne", "partout en france", "lieu", "adresse"],
        "text": "FORMATS ET LIEU : ISCOF propose des formations en présentiel à Vincennes (94), en 100 % distanciel et, selon le parcours, en format hybride. L'adresse indiquée est 32 rue Céline Robert, 94300 Vincennes. Le format exact dépend de la formation choisie."
    },
    {
        "id": "pedagogie",
        "keywords": ["accompagnement", "méthode", "suivi", "formateur", "plateforme", "addviseo", "rythme", "individualisé", "pédagogie", "tests"],
        "text": "ACCOMPAGNEMENT PÉDAGOGIQUE : ISCOF présente une méthode d'autoformation accompagnée : parcours ajusté au niveau et aux contraintes, référent/formateur unique pendant le parcours, plateforme AddViseo avec exercices, tests et ressources accessibles 24 h/24 et 7 j/7 entre les sessions, évaluations intermédiaires et jury final pour les Titres Professionnels certifiants."
    },
    {
        "id": "financement",
        "keywords": ["financement", "cpf", "opco", "aif", "france travail", "transitions pro", "ptp", "payer", "prix", "tarif", "reste à charge", "gratuit", "prise en charge", "éligible"],
        "text": "FINANCEMENT : La page Formation indique une éligibilité CPF et mentionne les financements CPF, OPCO et AIF selon les parcours et les situations. Le site ISCOF présente également des dispositifs comme Transitions Pro/PTP et des combinaisons de financements. L'éligibilité, le montant pris en charge et le reste à payer dépendent du dossier individuel ; ils doivent être confirmés par ISCOF. Le rendez-vous d'information est gratuit. Aucun prix précis ne doit être annoncé sans devis ou fiche de formation correspondante."
    },
    {
        "id": "rendez_vous_contact",
        "keywords": ["contact", "appeler", "téléphone", "email", "mail", "rendez-vous", "rdv", "conseiller", "programme", "inscrire", "inscription", "coordonnées"],
        "text": "CONTACT ET PROCHAINES ÉTAPES : ISCOF propose un premier rendez-vous d'information gratuit pour parler du projet, choisir une formation et étudier les possibilités de financement. Téléphone : 01 83 64 19 90. Email : contact@iscof.fr. Adresse : 32 rue Céline Robert, 94300 Vincennes. Site : https://iscof.fr/formation/ ."
    },
    {
        "id": "resultats",
        "keywords": ["réussite", "taux", "résultats", "avis", "recommander", "statistiques"],
        "text": "RÉSULTATS AFFICHÉS SUR LA PAGE FORMATION : ISCOF affiche un taux de réussite aux examens de 87 % et indique que 90 % des stagiaires recommanderaient l'organisme. Ces chiffres sont ceux présentés sur le site et ne garantissent pas le résultat individuel d'un futur stagiaire."
    },
]

SYSTEM_PROMPT = """Tu es l'assistant FAQ officiel d'ISCOF, organisme de formation. Tu aides les visiteurs à comprendre le catalogue, les formats, la reconnaissance des certifications, le financement et les prochaines étapes.

RÈGLES STRICTES :
- Réponds en français, de manière claire, rassurante, professionnelle et concise. Réponds dans la langue de l'utilisateur si celui-ci écrit dans une autre langue.
- Utilise uniquement les informations du CONTEXTE ISCOF fourni et les informations déjà dites dans la conversation. N'invente ni tarifs, ni dates de session, ni conditions d'admission, ni garanties de financement, ni résultats individuels.
- Si une information dépend du dossier, du parcours ou n'est pas présente dans le contexte, dis-le clairement puis propose de la vérifier auprès d'ISCOF.
- Pour le choix d'une formation, pose une seule question à la fois et utilise les informations déjà données par l'utilisateur. Tu peux demander son objectif professionnel ou le domaine qui l'intéresse.
- Ne prétends jamais avoir réservé un rendez-vous, envoyé un dossier ou contacté ISCOF.
- Distingue bien un rendez-vous d'information gratuit du coût éventuel d'une formation.
- Pour le financement, ne promets jamais une prise en charge à 100 % : cela dépend de la situation et de l'accord du financeur.
- Pour la reconnaissance, ne laisse pas entendre que toutes les formations sont le même type de titre : le statut dépend du parcours.
- Donne les coordonnées ISCOF quand cela aide : 01 83 64 19 90, contact@iscof.fr, 32 rue Céline Robert, 94300 Vincennes, https://iscof.fr/formation/.
- N'affiche pas de longs extraits du contexte. Réponds directement à la question."""

FAQ_SUGGESTIONS = [
    "Quelle formation choisir selon mon projet ?",
    "Les titres et certifications sont-ils reconnus ?",
    "Comment financer ma formation ?",
    "Quel est le délai pour démarrer ?",
]

@st.cache_data(show_spinner=False)
def calculer_embeddings(textes: tuple):
    return modele.encode(list(textes), convert_to_tensor=True)

CHUNK_TEXTS = tuple(str(c["text"]) for c in KNOWLEDGE_CHUNKS)
CHUNK_EMBEDDINGS = calculer_embeddings(CHUNK_TEXTS)

def selectionner_contexte(question: str, max_chunks: int = 4) -> str:
    q = question.lower()
    scores_kw = [sum(1 for kw in c["keywords"] if str(kw) in q) for c in KNOWLEDGE_CHUNKS]
    q_emb = modele.encode(question, convert_to_tensor=True)
    scores_sem = util.cos_sim(q_emb, CHUNK_EMBEDDINGS)[0].tolist()
    max_kw = max(scores_kw) if max(scores_kw) > 0 else 1
    scored = []
    for i, chunk in enumerate(KNOWLEDGE_CHUNKS):
        score = 0.62 * float(scores_sem[i]) + 0.38 * (scores_kw[i] / max_kw)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(str(c["text"]) for score, c in scored[:max_chunks])

def generer_reponse(question: str, messages: list) -> str:
    contexte = selectionner_contexte(question)
    conversation = []
    for msg in messages[-10:]:
        role = "Visiteur" if msg["role"] == "user" else "Assistant ISCOF"
        conversation.append(f"{role} : {msg['content']}")
    prompt = (
        f"{SYSTEM_PROMPT}\n\nCONTEXTE ISCOF :\n{contexte}\n\n"
        "HISTORIQUE RÉCENT :\n" + "\n".join(conversation) +
        f"\n\nQUESTION À TRAITER : {question}\nRéponse :"
    )
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            answer = getattr(response, "text", None)
            if answer and answer.strip():
                return answer.strip()
            return "Je n'ai pas réussi à formuler une réponse fiable à cette question. Vous pouvez contacter ISCOF au 01 83 64 19 90 ou à contact@iscof.fr."
        except Exception as exc:
            last_error = exc
            err = str(exc).lower()
            if attempt < 2 and ("429" in err or "503" in err or "tempor" in err or "unavailable" in err):
                time.sleep(1.5 ** attempt)
                continue
            break
    return "Le service de réponse est momentanément indisponible. Vous pouvez contacter ISCOF au 01 83 64 19 90 ou à contact@iscof.fr."

# -------------------- En-tête --------------------
st.markdown('<div class="iscof-kicker">— ISCOF · ASSISTANT FORMATIONS</div>', unsafe_allow_html=True)
st.markdown('<h1 class="iscof-title">Votre projet mérite<br><em>la bonne formation.</em></h1>', unsafe_allow_html=True)
st.markdown('<div class="iscof-subtitle">Posez vos questions sur les formations, les certifications, les modalités ou les financements. Je vous aide à trouver les bonnes informations avant votre premier rendez-vous.</div>', unsafe_allow_html=True)
st.markdown('<div class="iscof-trust">🎓 <b>Premier rendez-vous d’information gratuit</b> · Présentiel à Vincennes ou à distance · Réponses fondées sur les informations ISCOF disponibles.</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([5, 1])
with col_a:
    st.markdown('<div class="faq-label">Questions fréquentes</div>', unsafe_allow_html=True)
with col_b:
    if st.button("Effacer", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if not st.session_state.messages:
    cols = st.columns(2)
    for i, faq in enumerate(FAQ_SUGGESTIONS):
        with cols[i % 2]:
            if st.button(faq, key=f"faq_{i}", use_container_width=True):
                st.session_state.pending_question = faq
                st.rerun()

# Historique
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=LOGO_AVATAR if m["role"] == "assistant" else "👤"):
        st.markdown(m["content"])

# Les boutons FAQ et la saisie passent par le même flux, sans double traitement.
question = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Posez votre question sur les formations…")
if user_input:
    question = user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=LOGO_AVATAR):
        loading = st.empty()
        loading.markdown(
            '<div class="iscof-thinking" role="status" aria-live="polite">'
            '<span class="iscof-dots"><span class="iscof-dot"></span>'
            '<span class="iscof-dot"></span><span class="iscof-dot"></span></span>'
            '<span>Je prépare une réponse…</span></div>',
            unsafe_allow_html=True,
        )
        answer = generer_reponse(question, st.session_state.messages[:-1])
        loading.empty()
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

st.markdown(
    '<div class="footer-note">Cet assistant fournit des informations générales et ne remplace pas la validation de votre dossier par un conseiller ISCOF. Pour confirmer votre parcours, les tarifs ou votre financement : <b>01 83 64 19 90</b> · <a href="mailto:contact@iscof.fr">contact@iscof.fr</a> · <a href="https://iscof.fr/formation/" target="_blank">Catalogue des formations</a>.</div>',
    unsafe_allow_html=True,
)
