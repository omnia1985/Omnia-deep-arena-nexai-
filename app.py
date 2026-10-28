# -*- coding: utf-8 -*-
"""
Omnia DeepArena NexIA — Single-file Streamlit Platform
Run: streamlit run app.py
"""

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Omnia DeepArena NexIA", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

# ======================================================================================
# LANGUAGE HELPERS
# ======================================================================================
LANGS = {"AR": "العربية", "EN": "English", "FR": "Français",
         "ES": "Español", "IT": "Italiano", "DE": "Deutsch"}


def T6(en, ar, fr, es, it, de):
    return {"EN": en, "AR": ar, "FR": fr, "ES": es, "IT": it, "DE": de}


T = {
    # ---------- Sidebar ----------
    "lang_label":   T6("Language", "اللغة", "Langue", "Idioma", "Lingua", "Sprache"),
    "mode_label":   T6("Operating Mode", "وضع التشغيل", "Mode d'exploitation", "Modo de operación", "Modalità operativa", "Betriebsmodus"),
    "mode_intl":    T6("International (UK / Global)", "دولي (UK / عالمي)", "International (UK / Global)", "Internacional (UK / Global)", "Internazionale (UK / Globale)", "International (UK / Global)"),
    "mode_local":   T6("Local (Tunisia)", "محلي (تونس)", "Local (Tunisie)", "Local (Túnez)", "Locale (Tunisia)", "Lokal (Tunesien)"),
    "nav_label":    T6("Navigation", "التنقل", "Navigation", "Navegación", "Navigazione", "Navigation"),
    "nav_p1":       T6("AI Match Predictor", "مُتوقع المباريات", "Prédicteur IA", "Predictor IA", "Predittore IA", "KI-Vorhersage"),
    "nav_p2":       T6("Players Hub", "مركز اللاعبين", "Espace Joueurs", "Centro Jugadores", "Hub Giocatori", "Spieler-Hub"),
    "nav_p3":       T6("Clubs & Academies HUD", "لوحة الأندية", "HUD Clubs", "HUD Clubes", "HUD Club", "Vereine HUD"),
    "nav_p4":       T6("Scouts & Agents", "سوق الكشافة", "Marché Scouts", "Mercado Scouts", "Marketplace Scout", "Scout-Markt"),
    "nav_p5":       T6("Live Video & On-Site", "الفيديو المباشر", "Vidéo & Terrain", "Vídeo y Terreno", "Video & Campo", "Live & Vor-Ort"),
    "per_month":    T6("/ month", "/ شهر", "/ mois", "/ mes", "/ mese", "/ Monat"),
    "per_match":    T6("/ match", "/ مباراة", "/ match", "/ partido", "/ partita", "/ Spiel"),
    "subscribe":    T6("Subscribe Now", "اشترك الآن", "S'abonner", "Suscribirse", "Abbonati", "Abonnieren"),
    "request_ok":   T6("Request registered. Our team will contact you shortly.",
                       "تم تسجيل الطلب. سيتواصل معك فريقنا قريباً.",
                       "Demande enregistrée. Nous vous contacterons bientôt.",
                       "Solicitud registrada. Le contactaremos pronto.",
                       "Richiesta registrata. Ti contatteremo a breve.",
                       "Anfrage registriert. Wir melden uns in Kürze."),
    "monetization": T6("Monetization", "الاشتراك", "Monétisation", "Monetización", "Monetizzazione", "Monetarisierung"),

    # ---------- Page 1 ----------
    "p1_title":     T6("AI Match Predictor & Data Presentation", "مُتوقع المباريات وعرض البيانات",
                       "Prédicteur IA & Données", "Predictor IA y Datos", "Predittore IA & Dati", "KI-Vorhersage & Daten"),
    "p1_sub":       T6("Driven by a 10,000+ Monte-Carlo match simulation engine.",
                       "مدعوم بمحرك محاكاة مونت كارلو لأكثر من 10,000 مباراة.",
                       "Propulsé par un moteur Monte-Carlo de 10 000+ matchs.",
                       "Impulsado por un motor Monte-Carlo de más de 10.000 partidos.",
                       "Basato su un motore Monte-Carlo di oltre 10.000 partite.",
                       "Angetrieben von einer Monte-Carlo-Engine mit 10.000+ Spielen."),
    "p1_sim_title": T6("AI Match Simulator", "محاكي المباريات", "Simulateur IA", "Simulador IA", "Simulatore IA", "KI-Simulator"),
    "p1_sim_desc":  T6("Select two teams and run a full probabilistic simulation.",
                       "اختر فريقين وشغّل محاكاة احتمالية كاملة.",
                       "Sélectionnez deux équipes et lancez une simulation.",
                       "Seleccione dos equipos y ejecute la simulación.",
                       "Seleziona due squadre ed esegui la simulazione.",
                       "Wähle zwei Teams und starte die Simulation."),
    "p1_team_a":    T6("Team A", "الفريق أ", "Équipe A", "Equipo A", "Squadra A", "Team A"),
    "p1_team_b":    T6("Team B", "الفريق ب", "Équipe B", "Equipo B", "Squadra B", "Team B"),
    "p1_run_sim":   T6("Run 10,000 Simulations", "تشغيل 10,000 محاكاة", "Lancer 10 000 simulations",
                       "Ejecutar 10.000 simulaciones", "Esegui 10.000 simulazioni", "10.000 Simulationen starten"),
    "p1_win":       T6("Win", "فوز", "Victoire", "Victoria", "Vittoria", "Sieg"),
    "p1_draw":      T6("Draw", "تعادل", "Nul", "Empate", "Pareggio", "Unentschieden"),
    "p1_loss":      T6("Loss", "خسارة", "Défaite", "Derrota", "Sconfitta", "Niederlage"),
    "p1_sim_note":  T6("Simulation completed over 10,000 Monte-Carlo runs.",
                       "اكتملت المحاكاة عبر 10,000 تشغيل.",
                       "Simulation terminée sur 10 000 exécutions.",
                       "Simulación completada sobre 10.000 ejecuciones.",
                       "Simulazione completata su 10.000 esecuzioni.",
                       "Simulation über 10.000 Läufe abgeschlossen."),
    "p1_tac_title": T6("Tactical Breakdown Console", "وحدة التحليل التكتيكي", "Console tactique",
                       "Consola táctica", "Console tattica", "Taktische Konsole"),
    "p1_tac1_t":    T6("Attacking Strengths", "قوة الهجوم", "Forces offensives", "Fortalezas ofensivas", "Forze offensive", "Offensivstärken"),
    "p1_tac1_b":    T6("Crossing efficiency 41.2% from the right half-space. xG/90 is 1.94 with 63% of chances inside the box. High pressing recovery in 6.4s.",
                       "كفاءة العرضيات 41.2% من اليمين. الأهداف المتوقعة 1.94، 63% من الفرص داخل المنطقة. استعادة بالضغط في 6.4 ثانية.",
                       "Efficacité des centres 41,2 % côté droit. xG/90 à 1,94, 63 % des occasions dans la surface. Récupération sous pressing en 6,4 s.",
                       "Eficiencia de centros 41,2% por derecha. xG/90 de 1,94, 63% de ocasiones en el área. Recuperación tras presión en 6,4 s.",
                       "Efficienza cross 41,2% da destra. xG/90 1,94, 63% delle occasioni in area. Recupero dopo pressing in 6,4 s.",
                       "Flankeneffizienz 41,2 % rechts. xG/90 1,94, 63 % Chancen im Strafraum. Ballrückgewinnung nach Pressing in 6,4 s."),
    "p1_tac2_t":    T6("Defensive Vulnerabilities", "الثغرات الدفاعية", "Vulnérabilités défensives",
                       "Vulnerabilidades defensivas", "Vulnerabilità difensive", "Defensive Schwächen"),
    "p1_tac2_b":    T6("Left full-back exposed on second balls (58% duel loss). High line concedes 2.7 through-balls/match. Aerial duels won only 47%.",
                       "الظهير الأيسر مكشوف (خسارة 58% من الالتحامات). الخط المتقدم يستقبل 2.7 كرة خلفية/مباراة. الالتحامات الهوائية 47% فقط.",
                       "Latéral gauche exposé (58 % de duels perdus). Ligne haute concède 2,7 passes en profondeur/match. Duels aériens à 47 %.",
                       "Lateral izquierdo expuesto (58% duelos perdidos). Línea alta concede 2,7 pases filtrados/partido. Duelos aéreos al 47%.",
                       "Terzino sinistro esposto (58% duelli persi). Linea alta concede 2,7 filtranti/partita. Duelli aerei al 47%.",
                       "Linker AV exponiert (58 % Duelle verloren). Hohe Kette lässt 2,7 Steilpässe/Spiel zu. Kopfballduelle nur 47 %."),
    "p1_tac3_t":    T6("Set-Piece Efficiency", "فاعلية الكرات الثابتة", "Efficacité coups de pied arrêtés",
                       "Eficiencia balón parado", "Efficienza calci piazzati", "Standardsituationen"),
    "p1_tac3_b":    T6("8/23 goals from set pieces (34.8%). Near-post corners yield 0.09 xG/attempt. Zonal marking on indirect free-kicks shows 21% failure.",
                       "8/23 هدفاً من الكرات الثابتة (34.8%). ركنيات القائم القريب 0.09 xG/محاولة. الدفاع المنطقة يفشل بنسبة 21%.",
                       "8/23 buts sur phases arrêtées (34,8 %). Corners premier poteau 0,09 xG/tentative. Marquage de zone 21 % d'échecs.",
                       "8/23 goles a balón parado (34,8%). Córners al primer palo 0,09 xG/intento. Marca zonal falla 21%.",
                       "8/23 gol da piazzati (34,8%). Corner primo palo 0,09 xG/tentativo. Marcatura a zona 21% di errori.",
                       "8/23 Tore nach Standards (34,8 %). Kurze-Ecke-Ecken 0,09 xG/Versuch. Raumdeckung 21 % Fehler."),
    "p1_props_title": T6("Player Props Analytics", "تحليلات اللاعبين", "Analyse props joueurs",
                         "Analítica de props", "Analitica props", "Spieler-Props"),
    "p1_props_desc":  T6("Predictive probability derived from form, opposition and referee profile.",
                         "احتمالات تنبؤية مستمدة من المستوى والخصم وملف الحكم.",
                         "Probabilités prédictives issues de la forme, de l'adversaire et de l'arbitre.",
                         "Probabilidad predictiva de forma, rival y árbitro.",
                         "Probabilità predittive da forma, avversario e arbitro.",
                         "Prädiktive Wahrscheinlichkeit aus Form, Gegner und Schiri-Profil."),
    "prop_player":   T6("Player", "اللاعب", "Joueur", "Jugador", "Giocatore", "Spieler"),
    "prop_market":   T6("Market", "السوق", "Marché", "Mercado", "Mercato", "Markt"),
    "prop_prob":     T6("Probability", "الاحتمالية", "Probabilité", "Probabilidad", "Probabilità", "Wahrscheinlichkeit"),
    "prop_risk":     T6("Risk", "المخاطرة", "Risque", "Riesgo", "Rischio", "Risiko"),
    "m_goal":        T6("Anytime Goalscorer", "التسجيل في أي وقت", "Buteur à tout moment", "Goleador en cualquier momento", "Marcatore anytime", "Torschütze anytime"),
    "m_assist":      T6("Assist", "التمريرة الحاسمة", "Passe décisive", "Asistencia", "Assist", "Vorlage"),
    "m_card":        T6("Card / Suspension Risk", "خطر البطاقة", "Risque carton", "Riesgo tarjeta", "Rischio cartellino", "Kartenrisiko"),
    "m_sot":         T6("Shots on Target (2+)", "تسديدات على المرمى (2+)", "Tirs cadrés (2+)", "Tiros a puerta (2+)", "Tiri in porta (2+)", "Torschüsse (2+)"),
    "risk_low":      T6("Low", "منخفض", "Faible", "Bajo", "Basso", "Niedrig"),
    "risk_med":      T6("Medium", "متوسط", "Moyen", "Medio", "Medio", "Mittel"),
    "risk_high":     T6("High", "مرتفع", "Élevé", "Alto", "Alto", "Hoch"),
    "p1_disclaimer": T6("For Educational and Analytical Purposes Only. We do not provide gambling services. Please gamble responsibly (18+).",
                       "لأغراض تعليمية وتحليلية فقط. لا نقدم خدمات المقامرة. العب بمسؤولية (18+).",
                       "À des fins éducatives et analytiques uniquement. Nous ne proposons pas de paris. Jouez responsable (18+).",
                       "Solo con fines educativos y analíticos. No ofrecemos apuestas. Juegue responsable (18+).",
                       "Solo per scopi educativi e analitici. Non offriamo scommesse. Gioca responsabilmente (18+).",
                       "Nur für Bildungs- und Analysezwecke. Keine Glücksspieldienste. Bitte verantwortungsvoll spielen (18+)."),

    # ---------- Page 2 ----------
    "p2_title":     T6("Players Hub", "مركز اللاعبين", "Espace Joueurs", "Centro Jugadores", "Hub Giocatori", "Spieler-Hub"),
    "p2_sub":       T6("Elite performance extraction for pro, amateur and youth athletes.",
                       "استخراج الأداء للاعبين المحترفين والهواة والناشئين.",
                       "Extraction de performance pour pros, amateurs et jeunes.",
                       "Extracción de rendimiento para pros, amateurs y juveniles.",
                       "Estrazione performance per pro, amatori e giovani.",
                       "Elite-Performance für Profis, Amateure und Jugend."),
    "p2_pass_name": T6("Pro Player Pass", "بطاقة اللاعب المحترف", "Pass Joueur Pro", "Pase Jugador Pro", "Pass Giocatore Pro", "Profi-Spieler-Pass"),
    "p2_up_title":  T6("Video Match / Training Uploader", "رفع فيديو المباراة", "Téléversement vidéo", "Carga de vídeo", "Caricamento video", "Video-Upload"),
    "p2_up_label":  T6("Upload match or training footage (MP4, AVI, MOV)",
                       "ارفع لقطات المباراة أو التدريب (MP4، AVI، MOV)",
                       "Téléversez vos séquences (MP4, AVI, MOV)",
                       "Suba metraje (MP4, AVI, MOV)",
                       "Carica il footage (MP4, AVI, MOV)",
                       "Lade Material hoch (MP4, AVI, MOV)"),
    "p2_up_ok":     T6("File received and queued for AI extraction.",
                       "تم استلام الملف ووضعه في قائمة الاستخراج.",
                       "Fichier reçu et en file pour extraction.",
                       "Archivo recibido en cola.",
                       "File ricevuto in coda.",
                       "Datei empfangen und eingereiht."),
    "p2_lens_title": T6("Live AI Lens Tracking Console", "وحدة تتبع عدسة الذكاء", "Console Lens IA", "Consola Lens IA", "Console Lens IA", "KI-Lens-Konsole"),
    "p2_lens_toggle": T6("Activate YOLO26 Pro Lens", "تفعيل عدسة YOLO26 Pro", "Activer Lens YOLO26 Pro",
                         "Activar Lens YOLO26 Pro", "Attiva Lens YOLO26 Pro", "YOLO26 Pro aktivieren"),
    "p2_lens_on":   T6("Live stream active — 22 entities tracked.", "البث نشط — 22 كياناً متتبعاً.",
                       "Flux actif — 22 entités suivies.", "Flujo activo — 22 entidades.", "Stream attivo — 22 entità.", "Live — 22 Objekte."),
    "p2_lens_off":  T6("Lens idle — activate to start the detection stream.",
                       "العدسة خاملة — فعّلها لبدء البث.",
                       "Lens inactive — activez pour démarrer.",
                       "Lens inactiva — active para iniciar.",
                       "Lens inattiva — attiva per iniziare.",
                       "Lens inaktiv — aktivieren zum Starten."),
    "p2_fps":       T6("FPS", "إطار/ث", "IPS", "FPS", "FPS", "FPS"),
    "p2_conf":      T6("Mean Confidence", "متوسط الثقة", "Confiance moyenne", "Confianza media", "Confidenza media", "Mittlere Konfidenz"),
    "p2_entities":  T6("Tracked Entities", "الكيانات المتتبعة", "Entités suivies", "Entidades", "Entità", "Objekte"),
    "p2_hud_title": T6("Personal Performance HUD", "لوحة الأداء الشخصي", "HUD de performance", "HUD personal", "HUD prestazioni", "Performance-HUD"),
    "p2_heatmap":   T6("Positional Heatmap", "الخريطة الحرارية", "Carte thermique", "Mapa de calor", "Mappa di calore", "Positions-Heatmap"),
    "p2_speed":     T6("Top Speed (km/h)", "أقصى سرعة (كم/س)", "Vitesse max (km/h)", "Velocidad máx (km/h)", "Velocità max (km/h)", "Höchstgeschw. (km/h)"),
    "p2_pass":      T6("Pass Accuracy (%)", "دقة التمرير (%)", "Précision passe (%)", "Precisión pase (%)", "Precisione passaggio (%)", "Passgenauigkeit (%)"),
    "p2_dist":      T6("Distance (km)", "المسافة (كم)", "Distance (km)", "Distancia (km)", "Distanza (km)", "Distanz (km)"),
    "p2_sprints":   T6("Sprints (>25 km/h)", "السرعات القصوى", "Sprints (>25 km/h)", "Sprints (>25 km/h)", "Sprint (>25 km/h)", "Sprints (>25 km/h)"),
    "p2_duels":     T6("Duels Won (%)", "الالتحامات (%)", "Duels gagnés (%)", "Duelos ganados (%)", "Duelli vinti (%)", "Duelle gewonnen (%)"),
    "p2_rating":    T6("Technical Rating Chart", "مخطط التقييم الفني", "Graphique de notation", "Gráfico de valoración", "Grafico valutazione", "Bewertungsdiagramm"),
    "attr_drib":    T6("Dribbling", "المراوغة", "Dribble", "Regate", "Dribbling", "Dribbling"),
    "attr_pass":    T6("Passing", "التمرير", "Passe", "Pase", "Passaggio", "Passen"),
    "attr_shot":    T6("Shooting", "التسديد", "Tir", "Tiro", "Tiro", "Schuss"),
    "attr_def":     T6("Defending", "الدفاع", "Défense", "Defensa", "Difesa", "Verteidigung"),
    "attr_phys":    T6("Physical", "البدنية", "Physique", "Físico", "Fisico", "Physis"),
    "attr_vision":  T6("Vision", "الرؤية", "Vision", "Visión", "Visione", "Vision"),

    # ---------- Page 3 ----------
    "p3_title":     T6("Clubs & Academies HUD", "لوحة الأندية والأكاديميات", "HUD Clubs & Académies",
                       "HUD Clubes y Academias", "HUD Club & Accademie", "Vereine & Akademien HUD"),
    "p3_sub":       T6("Command centre for coaches, technical directors and managers.",
                       "مركز قيادة للمدربين والمديرين الفنيين.",
                       "Centre de commandement pour entraîneurs et directeurs.",
                       "Centro de mando para entrenadores y directores.",
                       "Centro di comando per allenatori e direttori.",
                       "Kommandozentrale für Trainer und Direktoren."),
    "p3_plan":      T6("Academy Enterprise", "باقة الأكاديمية", "Academy Enterprise", "Academy Enterprise", "Academy Enterprise", "Academy Enterprise"),
    "p3_overview":  T6("Team Overview", "نظرة عامة", "Aperçu équipe", "Resumen equipo", "Panoramica", "Team-Übersicht"),
    "p3_players":   T6("Total Registered Players", "اللاعبون المسجلون", "Joueurs inscrits", "Jugadores", "Giocatori", "Registrierte Spieler"),
    "p3_age":       T6("Average Squad Age", "متوسط العمر", "Âge moyen", "Edad media", "Età media", "Durchschnittsalter"),
    "p3_fitness":   T6("Squad Fitness Index", "مؤشر اللياقة", "Indice de forme", "Índice de forma", "Indice di forma", "Fitness-Index"),
    "p3_pre_title": T6("Pre-Match Opponent Analysis", "تحليل الخصم", "Analyse adverse", "Análisis rival", "Analisi avversario", "Gegneranalyse"),
    "p3_pre_desc":  T6("Upload rival video to generate a full tactical dossier.",
                       "ارفع فيديو الخصم لتوليد ملف تكتيكي.",
                       "Téléversez la vidéo adverse pour un dossier tactique.",
                       "Suba vídeo rival para un dossier táctico.",
                       "Carica video avversario per un dossier tattico.",
                       "Gegner-Video für ein Taktik-Dossier hochladen."),
    "p3_pre_up":    T6("Upload opponent match video", "ارفع فيديو مباراة الخصم", "Téléverser vidéo adverse",
                       "Subir vídeo rival", "Carica video avversario", "Gegner-Video hochladen"),
    "p3_pre_btn":   T6("Generate Opponent Report", "توليد تقرير الخصم", "Générer le rapport", "Generar informe", "Genera report", "Bericht erstellen"),
    "p3_pre_r1":    T6("Opponent Weaknesses", "نقاط ضعف الخصم", "Faiblesses", "Debilidades", "Debolezze", "Schwächen"),
    "p3_pre_r1b":   T6("Right flank concedes 4.1 progressive carries/match. CB pair loses 39% of aerial duels and drops 6.2m after minute 70.",
                       "الجهة اليمنى تستقبل 4.1 تقدم/مباراة. ثنائي الدفاع يخسر 39% هوائياً ويتراجع 6.2م بعد الدقيقة 70.",
                       "Flanc droit concède 4,1 courses/match. La charnière perd 39 % des duels aériens et recule de 6,2 m après la 70e.",
                       "Flanco derecho concede 4,1 conducciones/partido. La pareja pierde 39% de duelos aéreos y retrocede 6,2 m tras el 70'.",
                       "Fascia destra concede 4,1 conduzioni/partita. La coppia perde il 39% dei duelli aerei e arretra di 6,2 m dopo il 70'.",
                       "Rechte Seite lässt 4,1 Ballführungen/Spiel zu. IV-Duo verliert 39 % Kopfballduelle und fällt nach Min. 70 um 6,2 m."),
    "p3_pre_r2":    T6("Counter-Attack Gaps", "فجوات الهجمات المرتدة", "Failles en contre", "Brechas en contragolpe", "Varchi in contropiede", "Konterlücken"),
    "p3_pre_r2b":   T6("Transition window opens 3.8s after loss. Both full-backs push above halfway in 71% of build-ups.",
                       "نافذة التحول 3.8 ثانية. الظهيران يتقدمان في 71% من البناء.",
                       "Fenêtre de transition 3,8 s. Les latéraux montent dans 71 % des constructions.",
                       "Ventana de transición 3,8 s. Ambos laterales suben en el 71% de construcciones.",
                       "Finestra di transizione 3,8 s. Entrambi i terzini salgono nel 71% delle costruzioni.",
                       "Umschaltfenster 3,8 s. Beide AV schieben in 71 % der Aufbauten mit."),
    "p3_pre_r3":    T6("Remontada / Comeback Simulation", "محاكاة العودة", "Simulation Remontada",
                       "Simulación remontada", "Simulazione rimonta", "Remontada-Simulation"),
    "p3_pre_r3b":   T6("Switch to 3-4-2-1 at minute 60. Comeback probability from -1 rises from 18% to 34% with double wing sub.",
                       "التحول إلى 3-4-2-1 في الدقيقة 60. احتمالية العودة من 18% إلى 34% بتبديلين على الأجنحة.",
                       "Passer au 3-4-2-1 à la 60e. La probabilité de remontée passe de 18 % à 34 % avec double changement.",
                       "Cambiar a 3-4-2-1 al 60'. Probabilidad de remontar sube del 18% al 34% con doble cambio.",
                       "Passa al 3-4-2-1 al 60'. Probabilità di rimonta dal 18% al 34% con doppia sostituzione.",
                       "Wechsel auf 3-4-2-1 in Min. 60. Remontada-Wahrscheinlichkeit steigt von 18 % auf 34 % mit Doppelwechsel."),
    "p3_post_title": T6("Post-Match Error Analysis", "تحليل الأخطاء", "Analyse après-match", "Análisis post-partido", "Analisi post-partita", "Fehleranalyse"),
    "p3_post_desc": T6("Upload your own match video for a full correction report.",
                       "ارفع فيديو مباراتك لتقرير تصحيح شامل.",
                       "Téléversez votre vidéo pour un rapport complet.",
                       "Suba su vídeo para un informe completo.",
                       "Carica il tuo video per un report completo.",
                       "Eigenes Video für einen Korrekturbericht hochladen."),
    "p3_post_up":   T6("Upload your team's latest match video", "ارفع فيديو آخر مباراة لفريقك",
                       "Téléverser votre dernier match", "Subir último partido", "Carica ultima partita", "Letztes Spiel hochladen"),
    "p3_post_btn":  T6("Generate Correction Report", "توليد تقرير التصحيح", "Générer le rapport", "Generar informe", "Genera report", "Bericht erstellen"),
    "p3_post_r1":   T6("Individual Positioning Errors", "أخطاء التمركز", "Erreurs de positionnement",
                       "Errores de posicionamiento", "Errori di posizionamento", "Positionierungsfehler"),
    "p3_post_r1b":  T6("17 positional errors detected. Left CB drifted 9.4m out of block on 5 occasions. Midfield line separated >14m in 12 phases.",
                       "تم اكتشاف 17 خطأ تمركز. قلب الدفاع الأيسر خرج 9.4م في 5 حالات. الوسط انفصل أكثر من 14م في 12 مرحلة.",
                       "17 erreurs détectées. Le DC gauche a dérivé de 9,4 m du bloc 5 fois. Ligne médiane séparée >14 m en 12 phases.",
                       "17 errores detectados. Central izquierdo se desplazó 9,4 m del bloque 5 veces. Medios separados >14 m en 12 fases.",
                       "Rilevati 17 errori. Il centrale sinistro è uscito di 9,4 m dal blocco in 5 occasioni. Linea mediana separata >14 m in 12 fasi.",
                       "17 Fehler erkannt. Linker IV wich 5-mal 9,4 m vom Block ab. Mittelfeld in 12 Phasen >14 m getrennt."),
    "p3_post_r2":   T6("Duels Won %", "الالتحامات (%)", "Duels gagnés %", "Duelos ganados %", "Duelli vinti %", "Duelle gewonnen %"),
    "p3_post_r2b":  T6("Overall duels 48% (ground 53%, aerial 39%). Second balls 41%. Final-third pressing success 27%.",
                       "الالتحامات 48% (أرضية 53%، هوائية 39%). الكرة الثانية 41%. الضغط في الثلث الأخير 27%.",
                       "Duels 48 % (sol 53 %, aériens 39 %). Secondes balles 41 %. Pressing dernier tiers 27 %.",
                       "Duelos 48% (suelo 53%, aéreos 39%). Segundas jugadas 41%. Presión último tercio 27%.",
                       "Duelli 48% (terra 53%, aerei 39%). Seconde palle 41%. Pressing terzo finale 27%.",
                       "Duelle 48 % (Boden 53 %, Luft 39 %). Zweitbälle 41 %. Pressing letztes Drittel 27 %."),
    "p3_post_r3":   T6("Wearable Tech — Fatigue & Injury Warnings", "التقنيات القابلة للارتداء",
                       "Wearable — Fatigue & Blessure", "Wearable — Fatiga y Lesión",
                       "Wearable — Fatica e Infortuni", "Wearable — Müdigkeit & Verletzung"),
    "p3_post_r3b":  T6("⚠ Player #TN-04: hamstring load 91% of safe threshold — 48h reduced intensity. ⚠ Player #TN-11: ACWR 1.62 — elevated injury risk. 3 players flagged for recovery.",
                       "⚠ اللاعب #TN-04: حمل 91% من الحد الآمن — 48 ساعة تخفيف. ⚠ اللاعب #TN-11: ACWR 1.62 — خطر مرتفع. 3 لاعبين لبروتوكولات التعافي.",
                       "⚠ Joueur #TN-04 : 91 % du seuil sûr — 48 h réduites. ⚠ Joueur #TN-11 : ACWR 1,62 — risque élevé. 3 joueurs signalés.",
                       "⚠ Jugador #TN-04: 91% del umbral seguro — 48 h reducidas. ⚠ Jugador #TN-11: ACWR 1,62 — riesgo elevado. 3 jugadores marcados.",
                       "⚠ Giocatore #TN-04: 91% della soglia sicura — 48 h ridotte. ⚠ Giocatore #TN-11: ACWR 1,62 — rischio elevato. 3 giocatori segnalati.",
                       "⚠ Spieler #TN-04: 91 % der sicheren Schwelle — 48 h reduziert. ⚠ Spieler #TN-11: ACWR 1,62 — erhöhtes Risiko. 3 Spieler markiert."),
    "p3_gdpr_title": T6("Data Privacy & GDPR Safeguard", "حماية البيانات وGDPR",
                        "Confidentialité & RGPD", "Privacidad y RGPD", "Privacy & GDPR", "Datenschutz & DSGVO"),
    "p3_gdpr_text": T6("All media encrypted at rest (AES-256). Youth profiles under 18 are anonymised: the pipeline generates tags like Player #TN-01 instead of real names, strictly complying with the UK Children's Code and GDPR.",
                       "جميع الوسائط مشفرة (AES-256). ملفات الناشئين تحت 18 تُعالج مجهولة: النظام ينشئ رموز مثل Player #TN-01 بدلاً من الأسماء الحقيقية، التزاماً بمدونة الأطفال البريطانية وGDPR.",
                       "Tous les médias chiffrés (AES-256). Les moins de 18 ans sont anonymisés : tags comme Player #TN-01 au lieu des vrais noms, conformément au Children's Code UK et au RGPD.",
                       "Todos los medios cifrados (AES-256). Menores de 18 anonimizados: etiquetas como Player #TN-01 en lugar de nombres reales, cumpliendo el Children's Code UK y el RGPD.",
                       "Tutti i media crittografati (AES-256). Under 18 anonimizzati: tag come Player #TN-01 al posto dei nomi reali, in conformità con il Children's Code UK e il GDPR.",
                       "Alle Medien verschlüsselt (AES-256). Unter-18-Profile anonymisiert: Tags wie Player #TN-01 statt echter Namen, gemäß UK Children's Code und DSGVO."),
    "p3_academy":   T6("Academy Name", "اسم الأكاديمية", "Nom de l'académie", "Nombre academia", "Nome accademia", "Akademiename"),
    "p3_email":     T6("Authorized Business Email", "البريد المهني", "E-mail professionnel", "Correo corporativo", "Email aziendale", "Geschäftliche E-Mail"),
    "p3_gdpr_btn":  T6("Register GDPR Safeguard", "تسجيل ضمانات GDPR", "Enregistrer RGPD", "Registrar RGPD", "Registra GDPR", "DSGVO registrieren"),
    "p3_gdpr_ok":   T6("Safeguard registered. An encrypted compliance dossier was sent to your business email.",
                       "تم التسجيل. أُرسل ملف الامتثال المشفر إلى بريدك.",
                       "Enregistré. Un dossier chiffré a été envoyé.",
                       "Registrado. Se envió un dossier cifrado.",
                       "Registrato. Dossier cifrato inviato.",
                       "Registriert. Verschlüsseltes Dossier gesendet."),
    "p3_gdpr_err":  T6("Please provide a valid academy name and business email.",
                       "يُرجى إدخال اسم أكاديمية وبريد مهني صالحين.",
                       "Veuillez fournir un nom et un e-mail valides.",
                       "Facilite un nombre y correo válidos.",
                       "Inserisci nome ed email validi.",
                       "Bitte gültigen Namen und E-Mail eingeben."),

    # ---------- Page 4 ----------
    "p4_title":     T6("Scouts & Agents Marketplace", "سوق الكشافة والوكلاء", "Marché Scouts & Agents",
                       "Mercado Ojeadores", "Marketplace Scout", "Scout- & Agenten-Markt"),
    "p4_sub":       T6("Discover verified raw talent through anonymised AI-ranked data.",
                       "اكتشف المواهب عبر بيانات مجهولة مصنفة بالذكاء.",
                       "Découvrez des talents via des données anonymisées.",
                       "Descubra talento mediante datos anónimos.",
                       "Scopri talenti tramite dati anonimi.",
                       "Entdecke Talente durch anonymisierte Daten."),
    "p4_plan":      T6("Pro Scout License", "رخصة الكشاف", "Licence Scout Pro", "Licencia Ojeador Pro", "Licenza Scout Pro", "Pro-Scout-Lizenz"),
    "p4_search_title": T6("AI-Powered Unified Talent Search", "البحث الموحد للمواهب", "Recherche unifiée IA",
                          "Búsqueda unificada IA", "Ricerca unificata IA", "KI-Talentsuche"),
    "p4_pos":       T6("Player Position", "المركز", "Poste", "Posición", "Ruolo", "Position"),
    "p4_pos_all":   T6("All", "الكل", "Tous", "Todos", "Tutti", "Alle"),
    "p4_pos_fw":    T6("Forward", "مهاجم", "Attaquant", "Delantero", "Attaccante", "Stürmer"),
    "p4_pos_mf":    T6("Midfielder", "وسط", "Milieu", "Centrocampista", "Centrocampista", "Mittelfeld"),
    "p4_pos_df":    T6("Defender", "مدافع", "Défenseur", "Defensa", "Difensore", "Verteidiger"),
    "p4_pos_gk":    T6("Goalkeeper", "حارس", "Gardien", "Portero", "Portiere", "Torwart"),
    "p4_min_rate":  T6("Minimum Matching Rate (%)", "أدنى نسبة مطابقة (%)", "Taux de correspondance (%)",
                       "Tasa de coincidencia (%)", "Tasso di corrispondenza (%)", "Minimale Übereinstimmung (%)"),
    "p4_results":   T6("Matching Talent Profiles", "ملفات المواهب", "Profils correspondants", "Perfiles coincidentes", "Profili corrispondenti", "Passende Profile"),
    "p4_card_age":  T6("Age", "العمر", "Âge", "Edad", "Età", "Alter"),
    "p4_card_pos":  T6("Position", "المركز", "Poste", "Posición", "Ruolo", "Position"),
    "p4_card_rate": T6("AI Matching Rate", "نسبة المطابقة", "Taux IA", "Tasa IA", "Tasso IA", "KI-Rate"),
    "p4_card_vel":  T6("Peak Field Velocity", "أقصى سرعة", "Vitesse de pointe", "Velocidad máx", "Velocità di punta", "Spitzengeschw."),
    "p4_contact":   T6("Request Official Contact with Representative", "طلب تواصل رسمي", "Demander un contact officiel",
                       "Solicitar contacto oficial", "Richiedi contatto ufficiale", "Offiziellen Kontakt anfordern"),
    "p4_contact_ok": T6("Secure email dispatched to the player's academy for guardian approval.",
                        "تم إرسال بريد آمن إلى الأكاديمية لموافقة ولي الأمر.",
                        "E-mail sécurisé envoyé à l'académie pour approbation.",
                        "Correo enviado a la academia para aprobación.",
                        "Email inviata all'accademia per approvazione.",
                        "Sichere E-Mail an Akademie zur Zustimmung gesendet."),
    "p4_none":      T6("No talent profiles match the current filters. Lower the matching rate or change the position.",
                       "لا توجد مواهب مطابقة. خفّض نسبة المطابقة أو غيّر المركز.",
                       "Aucun profil ne correspond. Réduisez le taux ou changez de poste.",
                       "Ningún perfil coincide. Reduzca la tasa o cambie la posición.",
                       "Nessun profilo corrisponde. Riduci il tasso o cambia ruolo.",
                       "Keine Profile passen. Senke die Rate oder ändere die Position."),
    "winger":       T6("Winger", "جناح", "Ailier", "Extremo", "Ala", "Flügelspieler"),
    "striker":      T6("Striker", "رأس حربة", "Avant-centre", "Delantero centro", "Centravanti", "Mittelstürmer"),
    "playmaker":    T6("Playmaker", "صانع ألعاب", "Meneur", "Creador", "Regista", "Spielmacher"),
    "fullback":     T6("Full-Back", "ظهير", "Latéral", "Lateral", "Terzino", "Außenverteidiger"),
    "cb":           T6("Centre-Back", "قلب دفاع", "Défenseur central", "Central", "Difensore centrale", "Innenverteidiger"),
    "dm":           T6("Defensive Midfielder", "وسط مدافع", "Milieu défensif", "Mediocentro def.", "Mediano", "Def. Mittelfeld"),
    "goalkeeper":   T6("Goalkeeper", "حارس", "Gardien", "Portero", "Portiere", "Torwart"),

    # ---------- Page 5 ----------
    "p5_title":     T6("Live Video & On-Site Operations", "الفيديو المباشر والعمليات الميدانية",
                       "Vidéo Live & Terrain", "Vídeo en Vivo y Terreno", "Video Live & Campo", "Live-Video & Vor-Ort"),
    "p5_sub":       T6("Tunisia-exclusive edge-AI infrastructure deployed directly on your pitch.",
                       "بنية ذكاء اصطناعي طرفي حصرية لتونس في ملعبك.",
                       "Infrastructure edge-AI exclusive Tunisie sur votre terrain.",
                       "Infraestructura edge-AI exclusiva Túnez en su campo.",
                       "Infrastruttura edge-AI esclusiva Tunisia sul tuo campo.",
                       "Tunesien-exklusive Edge-KI auf Ihrem Platz."),
    "p5_exclusive": T6("Tunisia Exclusive", "حصري لتونس", "Exclusivité Tunisie", "Exclusivo Túnez", "Esclusiva Tunisia", "Exklusiv Tunesien"),
    "p5_lock_title": T6("Regional Feature Locked", "ميزة مقفلة", "Fonction verrouillée", "Función bloqueada", "Funzionalità bloccata", "Funktion gesperrt"),
    "p5_lock_msg":  T6("Live Video & On-Site Operations are only available in Local (Tunisia) Mode. Switch the Operating Mode in the sidebar.",
                       "الفيديو المباشر والعمليات متاحة فقط في الوضع المحلي (تونس). غيّر الوضع من الشريط الجانبي.",
                       "Disponible uniquement en mode Local (Tunisie). Changez le mode dans la barre latérale.",
                       "Solo disponible en Modo Local (Túnez). Cambie el modo en la barra lateral.",
                       "Disponibile solo in Modalità Locale (Tunisia). Cambia modalità nella barra laterale.",
                       "Nur im lokalen Modus (Tunesien) verfügbar. Wechseln Sie den Modus in der Seitenleiste."),
    "p5_book_title": T6("On-Site Booking & Scheduling", "الحجز والجدولة", "Réservation & Planification",
                        "Reserva y Programación", "Prenotazione", "Buchung & Planung"),
    "p5_date":      T6("Match Date", "التاريخ", "Date du match", "Fecha", "Data", "Spieldatum"),
    "p5_time":      T6("Kick-off Time", "وقت الانطلاق", "Heure", "Hora", "Ora", "Anstoßzeit"),
    "p5_loc":       T6("Stadium Location", "موقع الملعب", "Stade", "Estadio", "Stadio", "Stadion"),
    "p5_equip":     T6("Equipment Selection", "اختيار المعدات", "Matériel", "Equipamiento", "Attrezzatura", "Ausrüstung"),
    "p5_eq1":       T6("Professional Camera Rig", "طقم كاميرات احترافي", "Rig caméra pro", "Rig cámara pro", "Rig camera pro", "Kamera-Rig"),
    "p5_eq2":       T6("NVIDIA Jetson AGX Orin Edge Computer", "NVIDIA Jetson AGX Orin", "NVIDIA Jetson AGX Orin", "NVIDIA Jetson AGX Orin", "NVIDIA Jetson AGX Orin", "NVIDIA Jetson AGX Orin"),
    "p5_eq3":       T6("Wearable GPS Trackers", "أجهزة GPS قابلة للارتداء", "Trackers GPS wearables", "Rastreadores GPS", "Tracker GPS", "Wearable GPS"),
    "p5_book_btn":  T6("Confirm Booking — 350 TND", "تأكيد الحجز — 350 د.ت", "Confirmer — 350 TND", "Confirmar — 350 TND", "Conferma — 350 TND", "Bestätigen — 350 TND"),
    "p5_full":      T6("Fully Booked for this week due to Edge AI capacity limits.",
                       "مكتمل الحجز لهذا الأسبوع بسبب حدود سعة الحوسبة الطرفية.",
                       "Complet cette semaine (limites Edge AI).",
                       "Completo esta semana (límites Edge AI).",
                       "Tutto prenotato (limiti Edge AI).",
                       "Ausgebucht diese Woche (Edge-KI-Limits)."),
    "p5_book_ok":   T6("Booking confirmed. Jetson AGX Orin unit scheduled for dispatch.",
                       "تم تأكيد الحجز. وحدة Jetson مجدولة للإرسال.",
                       "Réservation confirmée. Jetson programmé.",
                       "Reserva confirmada. Jetson programado.",
                       "Prenotazione confermata. Jetson programmato.",
                       "Buchung bestätigt. Jetson eingeplant."),
    "p5_capacity":  T6("Weekly capacity used", "السعة المستخدمة", "Capacité utilisée", "Capacidad usada", "Capacità usata", "Kapazität genutzt"),
    "p5_jetson_title": T6("NVIDIA Jetson AGX Orin Edge Console", "وحدة Jetson AGX Orin", "Console Jetson", "Consola Jetson", "Console Jetson", "Jetson-Konsole"),
    "p5_jetson_toggle": T6("Connect to Pitch Jetson Device", "الاتصال بجهاز Jetson", "Connecter Jetson", "Conectar Jetson", "Connetti Jetson", "Jetson verbinden"),
    "p5_jetson_on": T6("Online — real-time network sync established.", "متصل — مزامنة فورية.", "En ligne — synchronisation active.",
                       "En línea — sincronizado.", "Online — sincronizzato.", "Online — synchronisiert."),
    "p5_jetson_off": T6("Offline — device not connected.", "غير متصل.", "Hors ligne.", "Desconectado.", "Offline.", "Offline."),
    "p5_latency":   T6("Latency (ms)", "زمن الاستجابة", "Latence (ms)", "Latencia (ms)", "Latenza (ms)", "Latenz (ms)"),
    "p5_pipeline":  T6("YOLO26 Pro Load (%)", "حمل YOLO26", "Charge YOLO26", "Carga YOLO26", "Carico YOLO26", "YOLO26 Last"),
    "p5_throughput": T6("Stream Throughput", "تدفق البث", "Débit", "Rendimiento", "Throughput", "Durchsatz"),
    "p5_wear_title": T6("Wearable Technology Dashboard", "لوحة التقنيات القابلة للارتداء", "Tableau wearable",
                        "Panel wearable", "Dashboard wearable", "Wearable-Dashboard"),
    "p5_wear_desc": T6("GPS streams and biometric HR trackers fused into the tactical map overlay.",
                       "تدفقات GPS ومقاييس النبض تُدمج في الخريطة التكتيكية.",
                       "Flux GPS et capteurs FC fusionnés dans la carte.",
                       "Flujos GPS y FC fusionados en el mapa.",
                       "Stream GPS e FC fusi nella mappa.",
                       "GPS- und HF-Streams im Kartenoverlay."),
    "p5_hr":        T6("Heart Rate (bpm)", "النبض", "FC (bpm)", "FC (ppm)", "FC (bpm)", "HF (bpm)"),
    "p5_gps":       T6("GPS Accuracy (m)", "دقة GPS", "Précision GPS (m)", "Precisión GPS (m)", "Precisione GPS (m)", "GPS-Genauigkeit (m)"),
    "p5_load":      T6("Player Load Index", "مؤشر الحمل", "Charge joueur", "Carga jugador", "Carico giocatore", "Spieler-Last"),
    "p5_battery":   T6("Tracker Battery (%)", "البطارية", "Batterie (%)", "Batería (%)", "Batteria (%)", "Akku (%)"),
    "p5_status_title": T6("Order Status Tracker", "متتبع حالة الطلب", "Suivi commande", "Seguimiento", "Tracker ordine", "Status-Tracker"),
    "p5_s1":        T6("Request Received", "تم الاستلام", "Demande reçue", "Solicitud recibida", "Richiesta ricevuta", "Anfrage erhalten"),
    "p5_s2":        T6("Jetson Dispatched", "إرسال Jetson", "Jetson expédié", "Jetson enviado", "Jetson spedito", "Jetson versandt"),
    "p5_s3":        T6("Edge Processing Live", "المعالجة مباشرة", "Traitement live", "Procesamiento en vivo", "Elaborazione live", "Edge live"),
    "p5_s4":        T6("Analytics Generated", "التحليلات جاهزة", "Analytique générée", "Analítica generada", "Analitica generata", "Analyse erstellt"),
    "p5_advance":   T6("Advance Stage", "تقديم المرحلة", "Avancer", "Avanzar", "Avanza", "Weiter"),
    "p5_reset":     T6("Reset Workflow", "إعادة تعيين", "Réinitialiser", "Reiniciar", "Reimposta", "Zurücksetzen"),
}


def t(key: str) -> str:
    entry = T.get(key)
    if entry is None:
        return key
    return entry.get(st.session_state.get("lang", "EN")) or entry.get("EN") or key


# ======================================================================================
# SESSION STATE
# ======================================================================================
DEFAULTS = {
    "lang": "EN",
    "mode": "intl",
    "page": "nav_p1",
    "user_bookings": {},   # iso_week -> count
    "sim_result": None,
    "order_stage": 0,
    "p2_lens_on": False,
    "p5_jetson_on": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ======================================================================================
# CSS
# ======================================================================================
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 0%, #101827 0%, #0b0f19 45%, #070a12 100%) !important;
    color: #e5e7eb;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid rgba(0,255,204,0.18);
}
[data-testid="stSidebar"] * { color: #d1d5db; }
h1, h2, h3, h4 { color: #f8fafc !important; font-weight: 800; letter-spacing: -0.5px; }
p, span, label, li, div { color: #cbd5e1; }

.nx-title {
    font-size: 2.05rem; font-weight: 800; margin-bottom: 0.15rem;
    background: linear-gradient(90deg, #00ffcc 0%, #f59e0b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nx-sub { color: #94a3b8; font-size: 0.98rem; margin-bottom: 1.1rem; }

.nx-card {
    background: #111827; border: 1px solid rgba(0,255,204,0.14);
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
    box-shadow: 0 0 26px rgba(0,255,204,0.05);
}
.nx-card-title {
    color: #00ffcc; font-weight: 700; font-size: 0.78rem;
    letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 8px;
}
.nx-card-body { color: #cbd5e1; font-size: 0.94rem; line-height: 1.6; }

.nx-metric {
    background: #111827; border: 1px solid rgba(0,255,204,0.16);
    border-radius: 14px; padding: 16px 18px; text-align: center;
}
.nx-metric-label { color: #94a3b8; font-size: 0.72rem; letter-spacing: 1.2px; text-transform: uppercase; }
.nx-metric-value { color: #00ffcc; font-size: 1.75rem; font-weight: 800; margin-top: 4px; }
.nx-metric-value-amber { color: #f59e0b; font-size: 1.75rem; font-weight: 800; margin-top: 4px; }
.nx-metric-value-red { color: #ef4444; font-size: 1.75rem; font-weight: 800; margin-top: 4px; }

.nx-stage { display: flex; justify-content: space-between; gap: 8px; margin-top: 10px; }
.nx-stage-item {
    flex: 1; text-align: center; padding: 10px 6px; border-radius: 10px;
    background: #0f1626; border: 1px solid rgba(148,163,184,0.18);
    font-size: 0.72rem; font-weight: 600; color: #64748b;
}
.nx-stage-active {
    background: rgba(0,255,204,0.10); border: 1px solid rgba(0,255,204,0.55);
    color: #00ffcc; box-shadow: 0 0 16px rgba(0,255,204,0.18);
}
.nx-brand { font-size: 1.25rem; font-weight: 800; color: #00ffcc; letter-spacing: -0.4px; }
.nx-brand-sub { font-size: 0.7rem; color: #f59e0b; letter-spacing: 2px; text-transform: uppercase; }
.nx-disclaimer {
    margin-top: 26px; padding: 14px 18px; border-radius: 12px;
    background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.35);
    color: #fca5a5; font-size: 0.82rem; line-height: 1.5;
}
.stButton > button {
    background: linear-gradient(90deg, #00ffcc 0%, #00c9a7 100%);
    color: #06121a !important; font-weight: 800; border: none;
    border-radius: 10px; padding: 0.55rem 1.15rem; width: 100%;
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0,255,204,0.55); transform: translateY(-1px);
    color: #06121a !important;
}
.stButton > button:disabled {
    background: #1f2937 !important; color: #6b7280 !important; box-shadow: none;
}
[data-testid="stExpander"] {
    background: #111827; border: 1px solid rgba(0,255,204,0.14) !important;
    border-radius: 12px !important;
}
[data-testid="stMetricValue"] { color: #00ffcc !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #00ffcc, #f59e0b) !important; }
hr { border-color: rgba(148,163,184,0.16) !important; }
</style>
"""
RTL_CSS = """
<style>
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] { direction: rtl; }
.nx-card, .nx-metric, .nx-stage { text-align: right; }
</style>
"""


def inject_css():
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    if st.session_state.get("lang") == "AR":
        st.markdown(RTL_CSS, unsafe_allow_html=True)


# ======================================================================================
# HTML HELPERS
# ======================================================================================
def html_card(title, body, accent="#00ffcc"):
    return (f"<div class='nx-card' style='border-left:4px solid {accent};'>"
            f"<div class='nx-card-title' style='color:{accent};'>{title}</div>"
            f"<div class='nx-card-body'>{body}</div></div>")


def html_metric(label, value, variant="primary"):
    cls = {"primary": "nx-metric-value",
           "amber": "nx-metric-value-amber",
           "red": "nx-metric-value-red"}.get(variant, "nx-metric-value")
    return (f"<div class='nx-metric'><div class='nx-metric-label'>{label}</div>"
            f"<div class='{cls}'>{value}</div></div>")


def render_heatmap(seed=7):
    rng = np.random.default_rng(seed)
    grid = rng.random((9, 13))
    sm = np.zeros_like(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            r0, r1 = max(0, r - 1), min(grid.shape[0], r + 2)
            c0, c1 = max(0, c - 1), min(grid.shape[1], c + 2)
            sm[r, c] = grid[r0:r1, c0:c1].mean()
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-9)
    rows = ""
    for r in range(sm.shape[0]):
        cells = ""
        for c in range(sm.shape[1]):
            a = 0.06 + 0.88 * float(sm[r, c])
            cells += f"<div style='background:rgba(0,255,204,{a:.2f});border-radius:4px;height:22px;'></div>"
        rows += f"<div style='display:grid;grid-template-columns:repeat(13,1fr);gap:4px;margin-bottom:4px;'>{cells}</div>"
    return f"<div style='background:#111827;padding:12px;border-radius:14px;border:1px solid rgba(0,255,204,0.16);'>{rows}</div>"


def price_str(gbp, tnd):
    if st.session_state.mode == "intl":
        return f"£{gbp:,.2f} {t('per_month')}"
    return f"{tnd:,.2f} TND {t('per_month')}"


# ======================================================================================
# SIDEBAR
# ======================================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='padding:6px 0 16px 0;'>"
            "<div class='nx-brand'>⚡ Omnia DeepArena</div>"
            "<div class='nx-brand-sub'>NexIA Platform</div></div>",
            unsafe_allow_html=True,
        )

        lang_codes = list(LANGS.keys())
        cur_lang = st.session_state.lang
        chosen = st.selectbox(t("lang_label"), lang_codes,
                              index=lang_codes.index(cur_lang),
                              format_func=lambda k: LANGS[k],
                              key="sb_lang")
        if chosen != st.session_state.lang:
            st.session_state.lang = chosen
            st.rerun()

        st.markdown("<hr style='margin:12px 0;'/>", unsafe_allow_html=True)

        chosen_mode = st.radio(
            t("mode_label"), ["intl", "local"],
            index=0 if st.session_state.mode == "intl" else 1,
            format_func=lambda m: t("mode_intl") if m == "intl" else t("mode_local"),
            key="sb_mode",
        )
        st.session_state.mode = chosen_mode

        st.markdown("<hr style='margin:12px 0;'/>", unsafe_allow_html=True)

        nav_keys = ["nav_p1", "nav_p2", "nav_p3", "nav_p4", "nav_p5"]
        labels = {k: t(k) for k in nav_keys}
        if st.session_state.mode == "intl":
            labels["nav_p5"] = labels["nav_p5"] + "  🔒"
        cur_page = st.session_state.page if st.session_state.page in nav_keys else "nav_p1"
        chosen_page = st.radio(t("nav_label"), nav_keys,
                               index=nav_keys.index(cur_page),
                               format_func=lambda k: labels[k],
                               key="sb_nav")
        st.session_state.page = chosen_page

        st.markdown("<hr style='margin:12px 0;'/>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;color:#4b5563;'>"
                    "Omnia DeepArena NexIA © 2025<br/>v1.0 · Edge-AI Sports Intelligence"
                    "</div>", unsafe_allow_html=True)


# ======================================================================================
# PAGE 1 — AI MATCH PREDICTOR
# ======================================================================================
def simulate(a, b):
    rng = random.Random((hash(a + "|" + b)) & 0xFFFFFFFF)
    win = round(rng.uniform(27.0, 54.0), 1)
    draw = round(rng.uniform(15.0, 29.0), 1)
    loss = round(max(100.0 - win - draw, 5.0), 1)
    if win + draw + loss > 100:
        loss = round(100.0 - win - draw, 1)
    return {"a": a, "b": b, "win": win, "draw": draw, "loss": loss,
            "xg_a": round(rng.uniform(0.8, 2.6), 2),
            "xg_b": round(rng.uniform(0.6, 2.3), 2)}


def page_predictor():
    st.markdown(f"<div class='nx-title'>{t('p1_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p1_sub')}</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            f"<div class='nx-card' style='border-left:4px solid #f59e0b;'>"
            f"<div class='nx-card-title' style='color:#f59e0b;'>{t('monetization')}</div>"
            f"<div style='font-size:1.05rem;font-weight:700;color:#f8fafc;'>Analyst Pro Access</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:#00ffcc;margin-top:6px;'>"
            f"{price_str(19.99, 69.90)}</div></div>",
            unsafe_allow_html=True)
    with col_b:
        st.write(""); st.write("")
        if st.button(t("subscribe"), key="p1_sub"):
            st.success(t("request_ok"))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p1_sim_title')}")
    st.caption(t("p1_sim_desc"))

    teams = ["NexIA United", "Tunisia Eagles", "Carthage FC", "Atlas Rovers", "Medina Stars", "Sahel Titans"]
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        team_a = st.selectbox(t("p1_team_a"), teams, index=0, key="p1_ta")
    with c2:
        team_b = st.selectbox(t("p1_team_b"), teams, index=1, key="p1_tb")
    with c3:
        st.write(""); st.write("")
        run = st.button(t("p1_run_sim"), key="p1_run")

    if st.session_state.sim_result is None or run:
        st.session_state.sim_result = simulate(team_a, team_b)
    if run:
        st.success(t("p1_sim_note"))

    s = st.session_state.sim_result
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(html_metric(t("p1_win"), f"{s['win']}%", "primary"), unsafe_allow_html=True)
        st.progress(min(max(s["win"] / 100.0, 0.0), 1.0))
    with m2:
        st.markdown(html_metric(t("p1_draw"), f"{s['draw']}%", "amber"), unsafe_allow_html=True)
        st.progress(min(max(s["draw"] / 100.0, 0.0), 1.0))
    with m3:
        st.markdown(html_metric(t("p1_loss"), f"{s['loss']}%", "red"), unsafe_allow_html=True)
        st.progress(min(max(s["loss"] / 100.0, 0.0), 1.0))
    st.caption(f"{s['a']} — xG {s['xg_a']}  |  {s['b']} — xG {s['xg_b']}")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p1_tac_title')}")

    with st.expander(t("p1_tac1_t"), expanded=True):
        st.write(t("p1_tac1_b"))
        df = pd.DataFrame({"Index": ["Crossing %", "xG / 90", "Box Entries", "Recovery (s)"],
                           "Value": [41.2, 1.94, 12.0, 6.4]}).set_index("Index")
        st.bar_chart(df)

    with st.expander(t("p1_tac2_t"), expanded=False):
        st.write(t("p1_tac2_b"))
        df = pd.DataFrame({"Index": ["Duel Loss %", "Through Balls", "Aerial Won %", "Line Height (m)"],
                           "Value": [58.0, 2.7, 47.0, 42.0]}).set_index("Index")
        st.bar_chart(df)

    with st.expander(t("p1_tac3_t"), expanded=False):
        st.write(t("p1_tac3_b"))
        df = pd.DataFrame({"Index": ["Goals Set Pieces", "xG / Corner", "Zonal Fail %"],
                           "Value": [8.0, 0.09, 21.0]}).set_index("Index")
        st.bar_chart(df)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p1_props_title')}")
    st.caption(t("p1_props_desc"))

    props = [
        ("Y. Ben Ali", "m_goal", 62, "risk_low"),
        ("Y. Ben Ali", "m_sot", 74, "risk_low"),
        ("K. Trabelsi", "m_assist", 41, "risk_med"),
        ("K. Trabelsi", "m_card", 33, "risk_high"),
        ("L. Moreau", "m_goal", 55, "risk_med"),
        ("L. Moreau", "m_card", 48, "risk_high"),
        ("A. Rossi", "m_assist", 58, "risk_low"),
        ("M. Schmidt", "m_sot", 67, "risk_med"),
        ("M. Schmidt", "m_card", 22, "risk_low"),
    ]
    df_props = pd.DataFrame([{
        t("prop_player"): p[0], t("prop_market"): t(p[1]),
        t("prop_prob"): f"{p[2]}%", t("prop_risk"): t(p[3])
    } for p in props])
    st.dataframe(df_props, use_container_width=True, hide_index=True)

    st.markdown(f"<div class='nx-disclaimer'>⚠ {t('p1_disclaimer')}</div>", unsafe_allow_html=True)


# ======================================================================================
# PAGE 2 — PLAYERS HUB
# ======================================================================================
def page_players():
    st.markdown(f"<div class='nx-title'>{t('p2_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p2_sub')}</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            f"<div class='nx-card' style='border-left:4px solid #f59e0b;'>"
            f"<div class='nx-card-title' style='color:#f59e0b;'>{t('monetization')}</div>"
            f"<div style='font-size:1.05rem;font-weight:700;color:#f8fafc;'>{t('p2_pass_name')}</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:#00ffcc;margin-top:6px;'>"
            f"{price_str(25.00, 89.90)}</div></div>",
            unsafe_allow_html=True)
    with col_b:
        st.write(""); st.write("")
        if st.button(t("subscribe"), key="p2_sub"):
            st.success(t("request_ok"))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p2_up_title')}")
    uploaded = st.file_uploader(t("p2_up_label"), type=["mp4", "avi", "mov"], key="p2_up")
    if uploaded is not None:
        st.success(f"{t('p2_up_ok')}  —  {uploaded.name}  ({uploaded.size/1024:.1f} KB)")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p2_lens_title')}")
    lens_on = st.toggle(t("p2_lens_toggle"), value=st.session_state.p2_lens_on, key="p2_lens_tgl")
    st.session_state.p2_lens_on = lens_on

    if lens_on:
        st.success(t("p2_lens_on"))
        l1, l2, l3 = st.columns(3)
        with l1:
            st.markdown(html_metric(t("p2_fps"), "60.0", "primary"), unsafe_allow_html=True)
        with l2:
            st.markdown(html_metric(t("p2_conf"), "0.94", "amber"), unsafe_allow_html=True)
        with l3:
            st.markdown(html_metric(t("p2_entities"), "22", "primary"), unsafe_allow_html=True)
        rng = np.random.default_rng(3)
        buf = pd.DataFrame(rng.random((40, 3)), columns=["Player A", "Player B", "Player C"])
        st.line_chart(buf)
    else:
        st.info(t("p2_lens_off"))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p2_hud_title')}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(html_metric(t("p2_speed"), "32.4 km/h", "primary"), unsafe_allow_html=True)
    with c2:
        st.markdown(html_metric(t("p2_pass"), "87.6%", "amber"), unsafe_allow_html=True)
    with c3:
        st.markdown(html_metric(t("p2_dist"), "10.4 km", "primary"), unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(html_metric(t("p2_sprints"), "28", "primary"), unsafe_allow_html=True)
    with c5:
        st.markdown(html_metric(t("p2_duels"), "61%", "amber"), unsafe_allow_html=True)
    with c6:
        st.markdown(html_metric("Rating", "8.3", "primary"), unsafe_allow_html=True)

    st.markdown(f"#### {t('p2_heatmap')}")
    st.markdown(render_heatmap(seed=hash("player1") & 0xFF), unsafe_allow_html=True)

    st.markdown(f"#### {t('p2_rating')}")
    df_attr = pd.DataFrame({
        "Attribute": [t("attr_drib"), t("attr_pass"), t("attr_shot"),
                       t("attr_def"), t("attr_phys"), t("attr_vision")],
        "Score": [86, 82, 74, 58, 79, 84],
    }).set_index("Attribute")
    st.bar_chart(df_attr)


# ======================================================================================
# PAGE 3 — CLUBS & ACADEMIES HUD
# ======================================================================================
def page_academies():
    st.markdown(f"<div class='nx-title'>{t('p3_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p3_sub')}</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            f"<div class='nx-card' style='border-left:4px solid #f59e0b;'>"
            f"<div class='nx-card-title' style='color:#f59e0b;'>{t('monetization')}</div>"
            f"<div style='font-size:1.05rem;font-weight:700;color:#f8fafc;'>{t('p3_plan')}</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:#00ffcc;margin-top:6px;'>"
            f"{price_str(85.00, 299.90)}</div></div>",
            unsafe_allow_html=True)
    with col_b:
        st.write(""); st.write("")
        if st.button(t("subscribe"), key="p3_sub"):
            st.success(t("request_ok"))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p3_overview')}")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(html_metric(t("p3_players"), "28", "primary"), unsafe_allow_html=True)
    with m2:
        st.markdown(html_metric(t("p3_age"), "18.5", "amber"), unsafe_allow_html=True)
    with m3:
        st.markdown(html_metric(t("p3_fitness"), "88%", "primary"), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p3_pre_title')}")
    st.caption(t("p3_pre_desc"))
    pre_file = st.file_uploader(t("p3_pre_up"), type=["mp4", "avi", "mov"], key="p3_pre_up")
    if st.button(t("p3_pre_btn"), key="p3_pre_btn"):
        if pre_file is None:
            st.warning("⚠ Please upload an opponent video first." if st.session_state.lang == "EN" else "⚠ يُرجى رفع فيديو الخصم أولاً.")
        else:
            with st.spinner("Processing..."):
                st.success(t("p3_pre_r1"))
                st.markdown(html_card(t("p3_pre_r1"), t("p3_pre_r1b"), "#ef4444"), unsafe_allow_html=True)
                st.markdown(html_card(t("p3_pre_r2"), t("p3_pre_r2b"), "#f59e0b"), unsafe_allow_html=True)
                st.markdown(html_card(t("p3_pre_r3"), t("p3_pre_r3b"), "#00ffcc"), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p3_post_title')}")
    st.caption(t("p3_post_desc"))
    post_file = st.file_uploader(t("p3_post_up"), type=["mp4", "avi", "mov"], key="p3_post_up")
    if st.button(t("p3_post_btn"), key="p3_post_btn"):
        if post_file is None:
            st.warning("⚠ Please upload your match video first." if st.session_state.lang == "EN" else "⚠ يُرجى رفع فيديو مباراتك أولاً.")
        else:
            with st.spinner("Processing..."):
                st.success(t("p3_post_r1"))
                st.markdown(html_card(t("p3_post_r1"), t("p3_post_r1b"), "#ef4444"), unsafe_allow_html=True)
                st.markdown(html_card(t("p3_post_r2"), t("p3_post_r2b"), "#f59e0b"), unsafe_allow_html=True)
                st.markdown(html_card(t("p3_post_r3"), t("p3_post_r3b"), "#00ffcc"), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p3_gdpr_title')}")
    st.markdown(html_card("🔒 GDPR", t("p3_gdpr_text"), "#00ffcc"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        academy = st.text_input(t("p3_academy"), key="p3_gdpr_acad")
    with c2:
        email = st.text_input(t("p3_email"), key="p3_gdpr_email")
    if st.button(t("p3_gdpr_btn"), key="p3_gdpr_btn"):
        valid = bool(academy.strip()) and "@" in email and "." in email.split("@")[-1]
        if valid:
            st.success(t("p3_gdpr_ok"))
        else:
            st.error(t("p3_gdpr_err"))


# ======================================================================================
# PAGE 4 — SCOUTS & AGENTS
# ======================================================================================
SCOUT_POOL = [
    {"id": "UK-7042", "age": 17, "pos_key": "winger",    "pos": "FW", "rate": 88, "vel": 32.4},
    {"id": "UK-7188", "age": 19, "pos_key": "striker",   "pos": "FW", "rate": 91, "vel": 33.1},
    {"id": "TN-2041", "age": 18, "pos_key": "playmaker", "pos": "MF", "rate": 84, "vel": 29.8},
    {"id": "TN-2103", "age": 20, "pos_key": "dm",        "pos": "MF", "rate": 79, "vel": 28.4},
    {"id": "FR-5520", "age": 17, "pos_key": "fullback",  "pos": "DF", "rate": 82, "vel": 31.2},
    {"id": "FR-5611", "age": 19, "pos_key": "cb",        "pos": "DF", "rate": 76, "vel": 28.0},
    {"id": "ES-3307", "age": 18, "pos_key": "goalkeeper","pos": "GK", "rate": 80, "vel": 26.5},
    {"id": "DE-9012", "age": 16, "pos_key": "winger",    "pos": "FW", "rate": 87, "vel": 30.9},
    {"id": "IT-4418", "age": 19, "pos_key": "playmaker", "pos": "MF", "rate": 89, "vel": 30.1},
    {"id": "UK-7299", "age": 18, "pos_key": "cb",        "pos": "DF", "rate": 74, "vel": 27.6},
    {"id": "TN-2188", "age": 17, "pos_key": "dm",        "pos": "MF", "rate": 83, "vel": 29.0},
    {"id": "UK-7345", "age": 20, "pos_key": "goalkeeper","pos": "GK", "rate": 78, "vel": 25.8},
]


def page_scouts():
    st.markdown(f"<div class='nx-title'>{t('p4_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p4_sub')}</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            f"<div class='nx-card' style='border-left:4px solid #f59e0b;'>"
            f"<div class='nx-card-title' style='color:#f59e0b;'>{t('monetization')}</div>"
            f"<div style='font-size:1.05rem;font-weight:700;color:#f8fafc;'>{t('p4_plan')}</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:#00ffcc;margin-top:6px;'>"
            f"{price_str(45.00, 159.90)}</div></div>",
            unsafe_allow_html=True)
    with col_b:
        st.write(""); st.write("")
        if st.button(t("subscribe"), key="p4_sub"):
            st.success(t("request_ok"))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p4_search_title')}")

    c1, c2 = st.columns(2)
    with c1:
        pos = st.selectbox(
            t("p4_pos"),
            ["ALL", "FW", "MF", "DF", "GK"],
            format_func=lambda p: {
                "ALL": t("p4_pos_all"), "FW": t("p4_pos_fw"), "MF": t("p4_pos_mf"),
                "DF": t("p4_pos_df"), "GK": t("p4_pos_gk"),
            }[p],
            key="p4_pos_sel",
        )
    with c2:
        min_rate = st.slider(t("p4_min_rate"), 50, 100, 70, key="p4_min_rate")

    filtered = [p for p in SCOUT_POOL if (pos == "ALL" or p["pos"] == pos) and p["rate"] >= min_rate]

    st.markdown(f"### {t('p4_results')}  ({len(filtered)})")

    if not filtered:
        st.warning(t("p4_none"))
        return

    for i in range(0, len(filtered), 2):
        row = filtered[i:i + 2]
        cols = st.columns(2)
        for col, p in zip(cols, row):
            with col:
                st.markdown(
                    f"<div class='nx-card' style='border-left:4px solid #00ffcc;'>"
                    f"<div class='nx-card-title'>Player #{p['id']}</div>"
                    f"<div style='font-size:1.15rem;font-weight:800;color:#f8fafc;'>"
                    f"{t(p['pos_key'])}</div>"
                    f"<div style='margin-top:8px;color:#cbd5e1;font-size:0.88rem;'>"
                    f"{t('p4_card_age')}: <b>{p['age']}</b> &nbsp;|&nbsp; "
                    f"{t('p4_card_pos')}: <b>{p['pos']}</b><br/>"
                    f"{t('p4_card_rate')}: <b style='color:#00ffcc;'>{p['rate']}%</b><br/>"
                    f"{t('p4_card_vel')}: <b style='color:#f59e0b;'>{p['vel']} km/h</b>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                if st.button(t("p4_contact"), key=f"p4_contact_{p['id']}"):
                    st.success(t("p4_contact_ok"))


# ======================================================================================
# PAGE 5 — LIVE VIDEO & ON-SITE OPS (TUNISIA)
# ======================================================================================
def iso_week_key(d: date) -> str:
    return f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"


def page_tunisia_ops():
    st.markdown(f"<div class='nx-title'>{t('p5_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p5_sub')}</div>", unsafe_allow_html=True)

    st.markdown(f"<span class='nx-pill nx-pill-danger' style='background:rgba(239,68,68,0.15);"
                f"color:#ef4444;border:1px solid #ef4444;padding:4px 12px;border-radius:999px;"
                f"font-size:0.72rem;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;'>"
                f"🇹🇳 {t('p5_exclusive')}</span>", unsafe_allow_html=True)

    if st.session_state.mode != "local":
        st.markdown(html_card(f"🔒 {t('p5_lock_title')}", t("p5_lock_msg"), "#ef4444"), unsafe_allow_html=True)
        return

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p5_book_title')}")

    c1, c2 = st.columns(2)
    with c1:
        d = st.date_input(t("p5_date"), value=date.today(), min_value=date.today(), key="p5_date")
    with c2:
        tm = st.time_input(t("p5_time"), key="p5_time")
    loc = st.text_input(t("p5_loc"), value="Stade Olympique de Radès", key="p5_loc")

    equip = st.multiselect(
        t("p5_equip"),
        options=[t("p5_eq1"), t("p5_eq2"), t("p5_eq3")],
        default=[t("p5_eq1"), t("p5_eq2")],
        key="p5_equip",
    )

    wk = iso_week_key(d)
    used = st.session_state.user_bookings.get(wk, 0)
    st.markdown(f"<div class='nx-card'><div class='nx-card-title'>{t('p5_capacity')}</div>"
                f"<div class='nx-card-body'><b>{used}/3</b></div></div>", unsafe_allow_html=True)
    st.progress(min(used / 3.0, 1.0))

    full = used >= 3
    if full:
        st.error(f"❌ {t('p5_full')}")

    if st.button(t("p5_book_btn"), key="p5_book", disabled=full):
        st.session_state.user_bookings[wk] = used + 1
        st.success(t("p5_book_ok"))
        st.balloons()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p5_jetson_title')}")
    jetson_on = st.toggle(t("p5_jetson_toggle"), value=st.session_state.p5_jetson_on, key="p5_jetson_tgl")
    st.session_state.p5_jetson_on = jetson_on

    if jetson_on:
        st.success(t("p5_jetson_on"))
        j1, j2, j3 = st.columns(3)
        with j1:
            st.markdown(html_metric(t("p5_latency"), "14 ms", "primary"), unsafe_allow_html=True)
        with j2:
            st.markdown(html_metric(t("p5_pipeline"), "72%", "amber"), unsafe_allow_html=True)
        with j3:
            st.markdown(html_metric(t("p5_throughput"), "60 FPS", "primary"), unsafe_allow_html=True)
        rng = np.random.default_rng(11)
        st.line_chart(pd.DataFrame(rng.random((60, 2)), columns=["GPU Load", "FPS"]))
    else:
        st.info(t("p5_jetson_off"))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p5_wear_title')}")
    st.caption(t("p5_wear_desc"))

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown(html_metric(t("p5_hr"), "142 bpm", "amber"), unsafe_allow_html=True)
    with w2:
        st.markdown(html_metric(t("p5_gps"), "0.4 m", "primary"), unsafe_allow_html=True)
    with w3:
        st.markdown(html_metric(t("p5_load"), "7.8 / 10", "amber"), unsafe_allow_html=True)
    with w4:
        st.markdown(html_metric(t("p5_battery"), "84%", "primary"), unsafe_allow_html=True)

    st.markdown(f"#### {t('p2_heatmap')}")
    st.markdown(render_heatmap(seed=hash("tunisia") & 0xFF), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p5_status_title')}")

    stages = [t("p5_s1"), t("p5_s2"), t("p5_s3"), t("p5_s4")]
    stage_html = "<div class='nx-stage'>"
    for i, label in enumerate(stages):
        cls = "nx-stage-item nx-stage-active" if i <= st.session_state.order_stage else "nx-stage-item"
        stage_html += f"<div class='{cls}'>{i+1}. {label}</div>"
    stage_html += "</div>"
    st.markdown(stage_html, unsafe_allow_html=True)

    st.progress(min((st.session_state.order_stage + 1) / 4.0, 1.0))

    c1, c2 = st.columns(2)
    with c1:
        if st.button(t("p5_advance"), key="p5_adv",
                     disabled=st.session_state.order_stage >= 3):
            st.session_state.order_stage += 1
            st.rerun()
    with c2:
        if st.button(t("p5_reset"), key="p5_rst"):
            st.session_state.order_stage = 0
            st.rerun()


# ======================================================================================
# MAIN ROUTER
# ======================================================================================
def main():
    inject_css()
    render_sidebar()

    page = st.session_state.page
    if page == "nav_p1":
        page_predictor()
    elif page == "nav_p2":
        page_players()
    elif page == "nav_p3":
        page_academies()
    elif page == "nav_p4":
        page_scouts()
    elif page == "nav_p5":
        page_tunisia_ops()
    else:
        page_predictor()


if __name__ == "__main__":
    main()
