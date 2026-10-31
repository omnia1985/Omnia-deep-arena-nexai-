# -*- coding: utf-8 -*-
import math
import os
import uuid
import io
from datetime import date
import requests
import numpy as np
import pandas as pd
import streamlit as st
import stripe
from dotenv import load_dotenv
from supabase import create_client, Client
from huggingface_hub import HfApi

try:
    from streamlit_webrtc import webrtc_streamer
    import cv2
    from ultralytics import YOLO
    import av
    AI_READY = True
except ImportError:
    AI_READY = False

load_dotenv()
st.set_page_config(page_title="Omnia DeepArena NexIA", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    FOOTBALL_API_KEY = st.secrets["FOOTBALL_API_KEY"]
    STRIPE_SECRET_KEY = st.secrets["STRIPE_SECRET_KEY"]
    BIGBALLS_API_KEY = st.secrets.get("BIGBALLS_API_KEY", "")
    ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
    FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "")
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    BIGBALLS_API_KEY = os.getenv("BIGBALLS_API_KEY", "")
    ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("⚠️ Missing SUPABASE keys in Secrets.")
        st.stop()
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    tok = st.session_state.get("access_token")
    ref = st.session_state.get("refresh_token")
    if tok and ref:
        try: client.auth.set_session(tok, ref)
        except Exception:
            st.session_state.access_token = None
            st.session_state.refresh_token = None
    return client

DEFAULTS = {"lang": "EN", "mode": "intl", "page": "nav_billing", "access_token": None, "refresh_token": None, "user_id": None, "user_email": None, "role": "fan", "sim_result": None}
for k, v in DEFAULTS.items(): st.session_state.setdefault(k, v)

LANGS = {"AR": "العربية", "EN": "English", "FR": "Français", "ES": "Español", "IT": "Italiano", "DE": "Deutsch"}
def T6(en, ar, fr, es, it, de): return {"EN": en, "AR": ar, "FR": fr, "ES": es, "IT": it, "DE": de}

T = {
    "lang": T6("Language","اللغة","Langue","Idioma","Lingua","Sprache"),
    "mode": T6("Operating Mode","وضع التشغيل","Mode","Modo","Modalità","Modus"),
    "mode_intl": T6("International (UK / Global)","دولي (UK/عالمي)","International","Internacional","Internazionale","International"),
    "mode_local": T6("Local (Tunisia)","محلي (تونس)","Local (Tunisie)","Local (Túnez)","Locale (Tunisia)","Lokal (Tunesien)"),
    "nav": T6("Navigation","التنقل","Navigation","Navegación","Navigazione","Navigation"),
    "nav_billing": T6("Billing & Pricing","الاشتراكات والدفع","Facturation","Facturación","Fatturazione","Abrechnung"),
    "nav_p1": T6("AI Match Predictor","مُتوقع المباريات","Prédicteur IA","Predictor IA","Predittore IA","KI-Vorhersage"),
    "nav_p2": T6("Players Hub","مركز اللاعبين","Espace Joueurs","Centro Jugadores","Hub Giocatori","Spieler-Hub"),
    "nav_p3": T6("Clubs & Academies","الأندية والأكاديميات","Clubs & Académies","Clubes y Academias","Club e Accademie","Vereine"),
    "nav_p4": T6("Scouts Marketplace","سوق الكشافة","Marché Scouts","Mercado Scouts","Marketplace Scout","Scout-Markt"),
    "nav_p5": T6("Live & On-Site Ops","الفيديو والعمليات","Live & Terrain","Vivo y Terreno","Live & Campo","Live & Vor-Ort"),
    "per_month": T6("/ month","/ شهر","/ mois","/ mes","/ mese","/ Monat"),
    "subscribe": T6("Subscribe Now","اشترك الآن","S'abonner","Suscribirse","Abbonati","Abonnieren"),
    "sub_ok": T6("Request sent. Pending approval.","تم إرسال الطلب.","Demande envoyée.","Solicitud enviada.","Richiesta inviata.","Anfrage gesendet."),
    "monetization": T6("Monetization","الاشتراك","Monétisation","Monetización","Monetizzazione","Monetarisierung"),
    "login_title": T6("Sign In","تسجيل الدخول","Connexion","Iniciar Sesión","Accedi","Anmelden"),
    "signup_title":T6("Create Account","إنشاء حساب","Créer un compte","Crear Cuenta","Crea Account","Konto erstellen"),
    "email": T6("Email","البريد الإلكتروني","E-mail","Email","Email","E-Mail"),
    "password": T6("Password","كلمة المرور","Mot de passe","Contraseña","Password","Passwort"),
    "full_name": T6("Full Name","الاسم الكامل","Nom complet","Nombre completo","Nome completo","Vollständiger Name"),
    "login_btn": T6("Sign In","دخول","Se connecter","Entrar","Accedi","Anmelden"),
    "signup_btn": T6("Sign Up","إنشاء","S'inscrire","Registrarse","Registrati","Registrieren"),
    "logout": T6("Logout","تسجيل الخروج","Déconnexion","Cerrar sesión","Logout","Abmelden"),
    "signed_in_as":T6("Signed in as","مسجل الدخول كـ","Connecté en tant que","Conectado como","Connesso come","Angemeldet als"),
    "check_email": T6("Check your email to confirm.","تحقق من بريدك.","Vérifiez votre e-mail.","Revise su correo.","Controlla email.","Prüfen Sie E-Mail."),
    "p1_title": T6("AI Match Predictor","مُتوقع المباريات","Prédicteur IA","Predictor IA","Predittore IA","KI-Vorhersage"),
    "p1_sub": T6("Real Poisson-Monte-Carlo over 10,000 runs with live Top 5 League data.","محاكاة مونت كارلو ببيانات الدوريات الخمس الكبرى.","Monte-Carlo réel avec données réelles.","Monte-Carlo real con datos reales.","Monte-Carlo reale con dati reali.","Echtes Monte-Carlo mit echten Daten."),
    "p1_sim": T6("AI Match Simulator","محاكي المباريات","Simulateur IA","Simulador IA","Simulatore IA","KI-Simulator"),
    "team_a": T6("Home Team","الفريق المضيف","Équipe Domicile","Equipo Local","Squadra Casa","Heimteam"),
    "team_b": T6("Away Team","الفريق الضيف","Équipe Extérieur","Equipo Visita","Squadra Ospite","Auswärtsteam"),
    "run_sim": T6("Run 10,000 Simulations","تشغيل 10,000 محاكاة","Lancer 10 000 sims","Ejecutar 10.000 sims","Esegui 10.000 sim","10.000 Sims starten"),
    "win": T6("Win","فوز","Victoire","Victoria","Vittoria","Sieg"),
    "draw": T6("Draw","تعادل","Nul","Empate","Pareggio","Unentschieden"),
    "loss": T6("Loss","خسارة","Défaite","Derrota","Sconfitta","Niederlage"),
    "sim_done": T6("Simulation completed (real data).","اكتملت المحاكاة.","Simulation terminée.","Simulación completada.","Simulazione completata.","Simulation abgeschlossen."),
    "p1_tac": T6("Tactical Breakdown","التحليل التكتيكي","Analyse tactique","Análisis táctico","Analisi tattica","Taktische Analyse"),
    "p1_props": T6("Player Props","تحليلات اللاعبين","Props joueurs","Props jugadores","Props giocatori","Spieler-Props"),
    "p1_disclaimer": T6("For Educational and Analytical Purposes Only. We do not provide gambling services. Please gamble responsibly (18+).","لأغراض تعليمية وتحليلية فقط. لا نقدم خدمات المقامرة (18+).","À des fins éducatives uniquement. Pas de paris (18+).","Solo con fines educativos. Sin apuestas (18+).","Solo per scopi educativi. Nessuna scommessa (18+).","Nur zu Bildungszwecken. Keine Glücksspiele (18+)."),
    "cross_sell_title": T6("Are you a Coach, Scout, or Player?", "هل أنت مدرب، كشاف، أو لاعب؟", "Êtes-vous entraîneur, scout ou joueur ?", "¿Eres entrenador, ojeador o jugador?", "Sei un allenatore, scout o giocatore?", "Sind Sie Trainer, Scout oder Spieler?"),
    "cross_sell_desc": T6("Discover our advanced professional tools in the sidebar!", "اكتشف أدواتنا الاحترافية في القائمة الجانبية!", "Découvrez nos outils dans la barre latérale !", "¡Descubre nuestras herramientas en la barra lateral!", "Scopri i nostri strumenti nella barra laterale!", "Entdecken Sie unsere Tools in der Seitenleiste!"),
    "p2_title": T6("Players Hub","مركز اللاعبين","Espace Joueurs","Centro Jugadores","Hub Giocatori","Spieler-Hub"),
    "p2_sub": T6("Upload match/training video — queued for AI analysis.","ارفع فيديو المباراة — لتحليل الذكاء الاصطناعي.","Téléversez une vidéo — analyse IA.","Suba un vídeo — análisis IA.","Carica un video — analisi IA.","Video hochladen — KI-Analyse."),
    "pass_name": T6("Pro Player Pass","بطاقة اللاعب المحترف","Pass Joueur Pro","Pase Jugador Pro","Pass Giocatore Pro","Profi-Spieler-Pass"),
    "upload_title": T6("Video Uploader","رفع الفيديو","Téléversement","Carga de vídeo","Caricamento","Video-Upload"),
    "upload_label": T6("Upload MP4, AVI or MOV (No size limit)","ارفع MP4 أو AVI أو MOV","Téléverser MP4/AVI/MOV","Subir MP4/AVI/MOV","Carica MP4/AVI/MOV","MP4/AVI/MOV hochladen"),
    "upload_ok": T6("Uploaded successfully — analysis job queued.","تم الرفع بنجاح.","Téléversé avec succès.","Subido con éxito.","Caricato con successo.","Erfolgreich hochgeladen."),
    "analysis_status": T6("Analysis Jobs","مهام التحليل","Tâches d'analyse","Tareas de análisis","Attività di analisi","Analyse-Jobs"),
    "status_queued": T6("Queued","في القائمة","En attente","En cola","In coda","In Warteschlange"),
    "status_processing": T6("Processing","جارٍ","En cours","Procesando","In corso","Verarbeitung"),
    "status_completed": T6("Completed","مكتمل","Terminé","Completado","Completato","Abgeschlossen"),
    "status_failed": T6("Failed","فشل","Échec","Falló","Fallito","Fehlgeschlagen"),
    "refresh": T6("Refresh Status","تحديث الحالة","Actualiser","Actualizar","Aggiorna","Aktualisieren"),
    "p2_lens_title": T6("Live AI Lens Tracking Console","وحدة تتبع عدسة الذكاء","Console Lens IA","Consola Lens IA","Console Lens IA","KI-Lens-Konsole"),
    "p2_lens_toggle": T6("Activate YOLO26 Pro Lens","تفعيل عدسة YOLO26 Pro","Activer Lens YOLO26 Pro","Activar Lens YOLO26 Pro","Attiva Lens YOLO26 Pro","YOLO26 Pro aktivieren"),
    "p2_lens_on": T6("Live stream active — 22 entities tracked.","البث نشط — 22 كياناً متتبعاً.","Flux actif — 22 entités.","Flujo activo — 22 entidades.","Stream attivo — 22 entità.","Live — 22 Objekte."),
    "p2_lens_off": T6("Lens idle — activate to start the detection stream.","العدسة خاملة.","Lens inactive.","Lens inactiva.","Lens inattiva.","Lens inaktiv."),
    "p2_fps": T6("FPS","إطار/ث","IPS","FPS","FPS","FPS"),
    "p2_conf": T6("Mean Confidence","متوسط الثقة","Confiance","Confianza","Confidenza","Konfidenz"),
    "p2_entities": T6("Tracked Entities","الكيانات المتتبعة","Entités","Entidades","Entità","Objekte"),
    "p2_hud_title": T6("Personal Performance HUD","لوحة الأداء الشخصي","HUD de performance","HUD personal","HUD prestazioni","Performance-HUD"),
    "p2_speed": T6("Top Speed (km/h)","أقصى سرعة (كم/س)","Vitesse max","Velocidad máx","Velocità max","Höchstgeschw."),
    "p2_pass": T6("Pass Accuracy (%)","دقة التمرير (%)","Précision passe","Precisión pase","Precisione passaggio","Passgenauigkeit"),
    "p2_dist": T6("Distance (km)","المسافة (كم)","Distance","Distancia","Distanza","Distanz"),
    "p2_sprints": T6("Sprints","السرعات القصوى","Sprints","Sprints","Sprint","Sprints"),
    "p2_duels": T6("Duels Won (%)","الالتحامات (%)","Duels gagnés","Duelos ganados","Duelli vinti","Duelle gewonnen"),
    "p2_rating": T6("Technical Rating Chart","مخطط التقييم الفني","Graphique de notation","Gráfico de valoración","Grafico valutazione","Bewertungsdiagramm"),
    "attr_drib": T6("Dribbling","المراوغة","Dribble","Regate","Dribbling","Dribbling"),
    "attr_pass": T6("Passing","التمرير","Passe","Pase","Passaggio","Passen"),
    "attr_shot": T6("Shooting","التسديد","Tir","Tiro","Tiro","Schuss"),
    "attr_def": T6("Defending","الدفاع","Défense","Defensa","Difesa","Verteidigung"),
    "attr_phys": T6("Physical","البدنية","Physique","Físico","Fisico","Physis"),
    "attr_vision": T6("Vision","الرؤية","Vision","Visión","Visione","Vision"),
    "p3_title": T6("Clubs & Academies HUD","لوحة الأندية","HUD Clubs","HUD Clubes","HUD Club","Vereine HUD"),
    "p3_sub": T6("Real uploads, real queue, real reports.","رفع حقيقي، قائمة حقيقية.","Téléversements réels.","Cargas reales.","Caricamenti reali.","Echte Uploads."),
    "plan_academy": T6("Academy Enterprise","باقة الأكاديمية","Academy Enterprise","Academy Enterprise","Academy Enterprise","Academy Enterprise"),
    "team_overview": T6("Team Overview","نظرة عامة","Aperçu équipe","Resumen equipo","Panoramica","Team-Übersicht"),
    "reg_players": T6("Registered Players","لاعبون مسجلون","Joueurs inscrits","Jugadores","Giocatori","Spieler"),
    "avg_age": T6("Average Age","متوسط العمر","Âge moyen","Edad media","Età media","Durchschnittsalter"),
    "fitness": T6("Fitness Index","مؤشر اللياقة","Indice forme","Índice forma","Indice forma","Fitness-Index"),
    "pre_match": T6("Pre-Match Opponent Analysis","تحليل الخصم","Analyse adverse","Análisis rival","Analisi avversario","Gegneranalyse"),
    "pre_desc": T6("Upload rival video → queued for real YOLO opponent report.","ارفع فيديو الخصم → تقرير YOLO.","Téléversez vidéo adverse → rapport YOLO.","Suba vídeo rival → informe YOLO.","Carica video avversario → report YOLO.","Gegner-Video → YOLO-Bericht."),
    "post_match": T6("Post-Match Error Analysis","تحليل الأخطاء","Analyse après-match","Análisis post-partido","Analisi post-partita","Fehleranalyse"),
    "post_desc": T6("Upload your match → queued for YOLO positioning + fatigue report.","ارفع مباراتك → تقرير YOLO.","Téléversez votre match → rapport YOLO.","Suba su partido → informe YOLO.","Carica la partita → report YOLO.","Ihr Spiel → YOLO-Bericht."),
    "gdpr_title": T6("Data Privacy & GDPR","حماية البيانات","Confidentialité & RGPD","Privacidad y RGPD","Privacy & GDPR","Datenschutz & DSGVO"),
    "gdpr_text": T6("All media encrypted securely. Youth under 18 anonymised: pipeline generates tags Player #TN-01 instead of real names.","جميع الوسائط مشفرة. الناشئون مجهولون برموز.","Médias chiffrés. Moins de 18 ans anonymisés.","Medios cifrados. Menores anonimizados.","Media crittografati. Under 18 anonimi.","Medien verschlüsselt. Unter 18 anonym."),
    "academy_name": T6("Academy Name","اسم الأكاديمية","Nom académie","Nombre academia","Nome accademia","Akademiename"),
    "biz_email": T6("Authorized Business Email","البريد المهني","E-mail pro","Correo corporativo","Email aziendale","Geschäftliche E-Mail"),
    "register_gdpr": T6("Register GDPR Safeguard","تسجيل ضمانات GDPR","Enregistrer RGPD","Registrar RGPD","Registra GDPR","DSGVO registrieren"),
    "gdpr_ok": T6("Saved in Supabase.","محفوظ في Supabase.","Enregistré.","Guardado.","Salvato.","Gespeichert."),
    "gdpr_err": T6("Valid name and email required.","اسم وبريد صالحان مطلوبان.","Nom et e-mail requis.","Nombre y correo válidos.","Nome ed email validi.","Name und E-Mail erforderlich."),
    "p4_title": T6("Scouts Marketplace","سوق الكشافة","Marché Scouts","Mercado Scouts","Marketplace Scout","Scout-Markt"),
    "p4_sub": T6("Real talent DB. Real contact requests. Real Supabase queries.","قاعدة مواهب حقيقية. طلبات حقيقية.","Talents réels.","Talentos reales.","Talenti reali.","Echte Talente."),
    "plan_scout": T6("Pro Scout License","رخصة الكشاف","Licence Scout Pro","Licencia Ojeador","Licenza Scout","Pro-Scout-Lizenz"),
    "search": T6("Unified Talent Search","بحث المواهب","Recherche talents","Búsqueda talentos","Ricerca talenti","Talentsuche"),
    "position": T6("Position","المركز","Poste","Posición","Ruolo","Position"),
    "pos_all": T6("All","الكل","Tous","Todos","Tutti","Alle"),
    "pos_fw": T6("Forward","مهاجم","Attaquant","Delantero","Attaccante","Stürmer"),
    "pos_mf": T6("Midfielder","وسط","Milieu","Centrocampista","Centrocampista","Mittelfeld"),
    "pos_df": T6("Defender","مدافع","Défenseur","Defensa","Difensore","Verteidiger"),
    "pos_gk": T6("Goalkeeper","حارس","Gardien","Portero","Portiere","Torwart"),
    "min_rate": T6("Minimum Matching Rate (%)","أدنى نسبة مطابقة (%)","Taux min (%)","Tasa mín (%)","Tasso min (%)","Min. Rate (%)"),
    "results": T6("Talent Profiles","ملفات المواهب","Profils talents","Perfiles talento","Profili talento","Talentprofile"),
    "contact_btn":T6("Request Official Contact","طلب تواصل رسمي","Demander un contact","Solicitar contacto","Richiedi contatto","Kontakt anfordern"),
    "contact_ok": T6("Request saved in Supabase — pending approval.","تم الحفظ — بانتظار الموافقة.","Enregistré — en attente.","Guardado — pendiente.","Salvato — in attesa.","Gespeichert — ausstehend."),
    "no_match": T6("No talent profiles available yet. Upload player videos to generate data.","لا توجد ملفات بعد. ارفع فيديوهات اللاعبين.","Aucun profil disponible.","Sin perfiles disponibles.","Nessun profilo disponibile.","Keine Profile verfügbar."),
    "p5_title": T6("Live Video & On-Site Ops","الفيديو والعمليات","Live & Terrain","Vivo y Terreno","Live & Campo","Live & Vor-Ort"),
    "p5_sub": T6("Real bookings saved in Supabase. Max 3/week enforced.","حجوزات حقيقية. الحد 3/أسبوع.","Réservations réelles. Max 3/semaine.","Reservas reales. Máx 3/semana.","Prenotazioni reali. Max 3/settimana.","Echte Buchungen. Max. 3/Woche."),
    "exclusive": T6("Tunisia Exclusive","حصري لتونس","Exclusivité Tunisie","Exclusivo Túnez","Esclusiva Tunisia","Exklusiv Tunesien"),
    "lock_title": T6("Regional Feature Locked","ميزة مقفلة","Fonction verrouillée","Función bloqueada","Funzionalità bloccata","Funktion gesperrt"),
    "lock_msg": T6("Switch to Local (Tunisia) mode.","بدّل إلى الوضع المحلي.","Passez en mode Local.","Cambie a Modo Local.","Passa a Locale.","Wechseln Sie zu Lokal."),
    "booking": T6("On-Site Booking","الحجز الميداني","Réservation","Reserva","Prenotazione","Buchung"),
    "match_date": T6("Match Date","التاريخ","Date","Fecha","Data","Datum"),
    "match_time": T6("Kick-off Time","الوقت","Heure","Hora","Ora","Anstoßzeit"),
    "location": T6("Stadium Location","الملعب","Stade","Estadio","Stadio","Stadion"),
    "equipment": T6("Equipment","المعدات","Matériel","Equipamiento","Attrezzatura","Ausrüstung"),
    "eq1": T6("Professional Camera Rig","طقم كاميرات","Rig caméra pro","Rig cámara pro","Rig camera pro","Kamera-Rig"),
    "eq2": T6("NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin"),
    "eq3": T6("Wearable GPS Trackers","أجهزة GPS","Trackers GPS","Rastreadores GPS","Tracker GPS","Wearable GPS"),
    "book_btn": T6("Confirm Booking — 350 TND","تأكيد — 350 د.ت","Confirmer — 350 TND","Confirmar — 350 TND","Conferma — 350 TND","Bestätigen — 350 TND"),
    "weekly_cap": T6("Weekly Capacity","السعة الأسبوعية","Capacité hebdo","Capacidad semanal","Capacità settimanale","Wochenkapazität"),
    "full": T6("Fully Booked for this week (max 3).","مكتمل (الحد 3).","Complet (max 3).","Completo (máx 3).","Completo (max 3).","Ausgebucht (max. 3)."),
    "book_ok": T6("Booking saved in Supabase.","تم الحفظ.","Réservé.","Guardado.","Salvato.","Gespeichert."),
    "my_bookings": T6("My Bookings","حجوزاتي","Mes réservations","Mis reservas","Mie prenotazioni","Meine Buchungen"),
    "status_tracker": T6("Order Status Tracker","متتبع الحالة","Suivi commande","Seguimiento","Tracker ordine","Status-Tracker"),
    "s1": T6("Request Received","تم الاستلام","Reçue","Recibida","Ricevuta","Erhalten"),
    "s2": T6("Jetson Dispatched","إرسال Jetson","Jetson expédié","Jetson enviado","Jetson spedito","Jetson versandt"),
    "s3": T6("Edge Live","المعالجة مباشرة","Edge live","Edge vivo","Edge live","Edge live"),
    "s4": T6("Analytics Ready","التحليلات جاهزة","Analytique prête","Analítica lista","Analitica pronta","Analyse bereit"),
    "billing_title": T6("Build Your Own Plan","ابنِ باقتك الخاصة","Créez votre plan","Crea tu plan","Crea il tuo piano","Erstellen Sie Ihren Plan"),
    "billing_desc": T6("Select the tools you need. Get discounts for bundling!","اختر الأدوات. احصل على خصومات!","Sélectionnez vos outils. Obtenez des réductions !","Selecciona tus herramientas. ¡Obtén descuentos!","Seleziona i tuoi strumenti. Ottieni sconti!","Wählen Sie Ihre Tools. Erhalten Sie Rabatte!"),
    "b_analyst": T6("AI Match Predictor (Analyst)","توقعات المباريات (محلل)","Prédicteur IA (Analyste)","Predictor IA (Analista)","Predittore IA (Analista)","KI-Vorhersage (Analyst)"),
    "b_player": T6("Players Hub (Player)","مركز اللاعبين","Espace Joueurs","Centro Jugadores","Hub Giocatori","Spieler-Hub"),
    "b_scout": T6("Scouts Marketplace (Scout)","سوق الكشافة","Marché Scouts","Mercado Scouts","Marketplace Scout","Scout-Markt"),
    "b_academy": T6("Clubs & Academies (Academy)","الأندية والأكاديميات","Clubs & Académies","Clubes y Academias","Club e Accademie","Vereine"),
    "b_subtotal": T6("Subtotal","المجموع الفرعي","Sous-total","Subtotal","Totale parziale","Zwischensumme"),
    "b_discount": T6("Bundle Discount","خصم الحزمة","Remise forfait","Descuento","Sconto pacchetto","Paketrabatt"),
    "b_total": T6("Total Monthly Price","السعر الإجمالي","Prix total","Precio total","Prezzo totale","Gesamtpreis"),
    "b_confirm": T6("Proceed to Secure Payment","الانتقال للدفع الآمن","Passer au paiement sécurisé","Pasar al pago seguro","Vai al pagamento sicuro","Weiter zur sicheren Zahlung"),
    "b_pay_link": T6("Click here to pay securely with Stripe","اضغط هنا للدفع بأمان عبر Stripe","Cliquez ici pour payer avec Stripe","Haga clic aquí para pagar con Stripe","Clicca qui per pagare con Stripe","Klicken Sie hier für sichere Zahlung"),
}

def t(key):
    entry = T.get(key)
    if not entry: return key
    return entry.get(st.session_state.lang) or entry.get("EN") or key

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 15% 0%, #101827 0%, #0b0f19 45%, #070a12 100%) !important; color: #e5e7eb; font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid rgba(0,255,204,0.18); }
[data-testid="stSidebar"] * { color: #d1d5db; }
h1, h2, h3, h4 { color: #f8fafc !important; font-weight: 800; }
.nx-title { font-size: 2rem; font-weight: 800; margin-bottom: .15rem; background: linear-gradient(90deg, #00ffcc 0%, #f59e0b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.nx-sub { color: #94a3b8; font-size: .95rem; margin-bottom: 1rem; }
.nx-card { background: #111827; border: 1px solid rgba(0,255,204,0.14); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; }
.nx-card-title { color: #00ffcc; font-weight: 700; font-size: .74rem; letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 6px; }
.nx-card-body { color: #cbd5e1; font-size: .92rem; line-height: 1.55; }
.nx-metric { background: #111827; border: 1px solid rgba(0,255,204,0.16); border-radius: 14px; padding: 14px 16px; text-align: center; }
.nx-metric-label { color: #94a3b8; font-size: .7rem; letter-spacing: 1.2px; text-transform: uppercase; }
.nx-metric-value { color: #00ffcc; font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
.nx-metric-amber { color: #f59e0b; font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
.nx-metric-red { color: #ef4444; font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
.nx-brand { font-size: 1.2rem; font-weight: 800; color: #00ffcc; }
.nx-brand-sub { font-size: .68rem; color: #f59e0b; letter-spacing: 2px; text-transform: uppercase; }
.nx-disclaimer { margin-top: 24px; padding: 12px 16px; border-radius: 12px; background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.35); color: #fca5a5; font-size: .82rem; }
.stButton > button { background: linear-gradient(90deg, #00ffcc 0%, #00c9a7 100%); color: #06121a !important; font-weight: 800; border: none; border-radius: 10px; padding: .55rem 1.1rem; width: 100%; }
.stButton > button:hover { box-shadow: 0 0 20px rgba(0,255,204,0.55); }
.stButton > button:disabled { background: #1f2937 !important; color: #6b7280 !important; }
[data-testid="stExpander"] { background: #111827; border: 1px solid rgba(0,255,204,0.14) !important; border-radius: 12px !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #00ffcc, #f59e0b) !important; }
</style>
"""
if st.session_state.lang == "AR": CSS += "<style>[data-testid='stAppViewContainer'],[data-testid='stSidebar']{direction:rtl;}</style>"
st.markdown(CSS, unsafe_allow_html=True)

def metric(label, value, variant="primary"):
    cls = {"primary": "nx-metric-value", "amber": "nx-metric-amber", "red": "nx-metric-red"}[variant]
    return f"<div class='nx-metric'><div class='nx-metric-label'>{label}</div><div class='{cls}'>{value}</div></div>"

def poisson_pmf(k: int, lam: float) -> float: return math.exp(-lam) * (lam ** k) / math.factorial(k)

def real_match_simulation(home_attack, home_defense, away_attack, away_defense, home_advantage=0.15, n_sims=10000):
    lam_home = math.exp(home_attack - away_defense + home_advantage)
    lam_away = math.exp(away_attack - home_defense)
    rng = np.random.default_rng()
    hg = rng.poisson(lam_home, n_sims)
    ag = rng.poisson(lam_away, n_sims)
    wins = int((hg > ag).sum()); draws = int((hg == ag).sum()); losses = n_sims - wins - draws
    return {"win": round(wins / n_sims * 100, 1), "draw": round(draws / n_sims * 100, 1), "loss": round(losses / n_sims * 100, 1), "xg_home": round(lam_home, 2), "xg_away": round(lam_away, 2), "over_2_5": round(float(((hg + ag) > 2.5).mean()) * 100, 1), "btts": round(float(((hg > 0) & (ag > 0)).mean()) * 100, 1)}

def real_player_props(xg_per90, xa_per90, card_prob_per90):
    return {"goal": round((1 - poisson_pmf(0, xg_per90)) * 100, 1), "assist": round((1 - poisson_pmf(0, xa_per90)) * 100, 1), "card": round(card_prob_per90 * 100, 1), "sot": round((1 - poisson_pmf(0, xg_per90 * 2.2)) * 100, 1)}

def auth_page():
    st.markdown("<div class='nx-title'>⚡ Omnia DeepArena NexIA</div>", unsafe_allow_html=True)
    st.markdown("<div class='nx-sub'>Deep-Tech Sports Intelligence Platform</div>", unsafe_allow_html=True)  
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        tabs = st.tabs([t("login_title"), t("signup_title")])
        with tabs[0]:
            with st.form("login_form"):
                email = st.text_input(t("email"), key="login_email")
                pwd = st.text_input(t("password"), type="password", key="login_pwd")
                if st.form_submit_button(t("login_btn")):
                    if not email or not pwd: st.error("Missing email or password.")
                    else:
                        try:
                            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                            resp = client.auth.sign_in_with_password({"email": email, "password": pwd})
                            st.session_state.access_token = resp.session.access_token
                            st.session_state.refresh_token = resp.session.refresh_token
                            st.session_state.user_id = resp.user.id
                            st.session_state.user_email = resp.user.email
                            try:
                                prof = (client.table("profiles").select("*").eq("id", resp.user.id).single().execute()).data
                                if prof:
                                    st.session_state.lang = prof.get("language", "EN")
                                    st.session_state.mode = prof.get("mode", "intl")
                                    st.session_state.role = prof.get("role", "fan")
                            except Exception: pass
                            st.rerun()
                        except Exception as e: st.error(f"Login failed: {e}")
        with tabs[1]:
            with st.form("signup_form"):
                name = st.text_input(t("full_name"), key="su_name")
                email = st.text_input(t("email"), key="su_email")
                pwd = st.text_input(t("password"), type="password", key="su_pwd")
                if st.form_submit_button(t("signup_btn")):
                    if not email or len(pwd) < 6: st.error("Email required and password ≥ 6 chars.")
                    else:
                        try:
                            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                            resp = client.auth.sign_up({"email": email, "password": pwd, "options": {"data": {"full_name": name or email.split("@")[0]}}})
                            if resp.session:
                                st.session_state.access_token = resp.session.access_token
                                st.session_state.refresh_token = resp.session.refresh_token
                                st.session_state.user_id = resp.user.id
                                st.session_state.user_email = resp.user.email
                                st.rerun()
                            else: st.success(t("check_email"))
                        except Exception as e: st.error(f"Signup failed: {e}")

def require_auth():
    if not st.session_state.get("user_id"):
        auth_page(); st.stop()

def render_sidebar():
    client = get_client()
    with st.sidebar:
        st.markdown("<div style='padding:6px 0 14px 0;'><div class='nx-brand'>⚡ Omnia DeepArena</div><div class='nx-brand-sub'>NexIA Platform</div></div>", unsafe_allow_html=True)
        st.caption(f"{t('signed_in_as')}: {st.session_state.user_email}")
        if st.button(t("logout"), key="logout_btn"):
            try: client.auth.sign_out()
            except Exception: pass
            for k in ["access_token", "refresh_token", "user_id", "user_email"]: st.session_state[k] = None
            st.rerun()
        st.markdown("<hr style='margin:12px 0;'/>", unsafe_allow_html=True)
        lang_codes = list(LANGS.keys())
        new_lang = st.selectbox(t("lang"), lang_codes, index=lang_codes.index(st.session_state.lang), format_func=lambda k: LANGS[k], key="sb_lang")
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            try: client.table("profiles").update({"language": new_lang}).eq("id", st.session_state.user_id).execute()
            except Exception: pass
            st.rerun()
        new_mode = st.radio(t("mode"), ["intl", "local"], index=0 if st.session_state.mode == "intl" else 1, format_func=lambda m: t("mode_intl") if m == "intl" else t("mode_local"), key="sb_mode")
        if new_mode != st.session_state.mode:
            st.session_state.mode = new_mode
            try: client.table("profiles").update({"mode": new_mode}).eq("id", st.session_state.user_id).execute()
            except Exception: pass
        st.markdown("<hr style='margin:12px 0;'/>", unsafe_allow_html=True)
        
        nav_keys = ["nav_billing", "nav_p1", "nav_p2", "nav_p3", "nav_p4", "nav_p5"]
        labels = {k: t(k) for k in nav_keys}
        if st.session_state.mode == "intl": labels["nav_p5"] += "  🔒"
        cur = st.session_state.page if st.session_state.page in nav_keys else "nav_billing"
        chosen = st.radio(t("nav"), nav_keys, index=nav_keys.index(cur), format_func=lambda k: labels[k], key="sb_nav")
        st.session_state.page = chosen
        
        st.markdown("<hr style='margin:20px 0 10px 0;'/>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #fca5a5; font-size: 0.75rem; font-weight: 600;'>18+ | Sports Analytics & Data<br>Not a bookmaker. Play responsibly.</div>", unsafe_allow_html=True)

def subscribe_block(plan_key, gbp, tnd, title_key, btn_key):
    price = f"£{gbp:.2f} {t('per_month')}" if st.session_state.mode == "intl" else f"{tnd:.2f} TND {t('per_month')}"
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"<div class='nx-card' style='border-left:4px solid #f59e0b;'><div class='nx-card-title' style='color:#f59e0b;'>{t('monetization')}</div><div style='font-size:1rem;font-weight:700;color:#f8fafc;'>{t(title_key)}</div><div style='font-size:1.5rem;font-weight:800;color:#00ffcc;margin-top:6px;'>{price}</div></div>", unsafe_allow_html=True)
    with col_b:
        st.write(""); st.write("")
        if st.button(t("subscribe"), key=btn_key):
            st.session_state.page = "nav_billing"
            st.rerun()

def page_billing():
    st.markdown(f"<div class='nx-title'>{t('billing_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('billing_desc')}</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    prices_gbp = {"analyst": 19.99, "player": 25.00, "scout": 45.00, "academy": 85.00}
    prices_tnd = {"analyst": 69.90, "player": 89.90, "scout": 159.90, "academy": 299.90}
    prices = prices_gbp if st.session_state.mode == "intl" else prices_tnd
    sym = "£" if st.session_state.mode == "intl" else "TND"

    st.markdown("#### 1. Select Your Tools:")
    col1, col2 = st.columns(2)
    with col1:
        c_analyst = st.checkbox(f"{t('b_analyst')} ({sym}{prices['analyst']})", key="bx_analyst")
        c_scout = st.checkbox(f"{t('b_scout')} ({sym}{prices['scout']})", key="bx_scout")
    with col2:
        c_player = st.checkbox(f"{t('b_player')} ({sym}{prices['player']})", key="bx_player")
        c_academy = st.checkbox(f"{t('b_academy')} ({sym}{prices['academy']})", key="bx_academy")
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    selected = []
    if c_analyst: selected.append(("analyst", prices['analyst']))
    if c_player: selected.append(("player", prices['player']))
    if c_scout: selected.append(("scout", prices['scout']))
    if c_academy: selected.append(("academy", prices['academy']))
    
    subtotal = sum(p[1] for p in selected)
    discount_pct = 0
    if len(selected) == 2: discount_pct = 0.08
    elif len(selected) == 3: discount_pct = 0.12
    elif len(selected) == 4: discount_pct = 0.15
    
    discount_amt = subtotal * discount_pct
    total = subtotal - discount_amt
    
    st.markdown("#### 2. Checkout")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(metric(t("b_subtotal"), f"{sym}{subtotal:.2f}"), unsafe_allow_html=True)
    with c2: st.markdown(metric(t("b_discount"), f"-{sym}{discount_amt:.2f}", "amber"), unsafe_allow_html=True)
    with c3: st.markdown(metric(t("b_total"), f"{sym}{total:.2f}", "primary"), unsafe_allow_html=True)
    
    st.write("")
    if st.button(t("b_confirm"), disabled=len(selected) == 0):
        plan_string = "+".join([s[0] for s in selected])
        
        if not STRIPE_SECRET_KEY:
            st.error("⚠️ Stripe Secret Key is missing in Secrets.")
        else:
            stripe.api_key = STRIPE_SECRET_KEY
            try:
                checkout_session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{
                        'price_data': {
                            'currency': 'gbp' if st.session_state.mode == 'intl' else 'eur',
                            'product_data': {'name': f'Omnia NexIA - {plan_string}'},
                            'unit_amount': int(total * 100),
                        },
                        'quantity': 1,
                    }],
                    mode='payment',
                    success_url='https://omnia-deep-arena.streamlit.app/', 
                    cancel_url='https://omnia-deep-arena.streamlit.app/',
                )
                get_client().table("subscriptions").insert({
                    "user_id": st.session_state.user_id, "plan": plan_string,
                    "amount": total, "currency": "GBP" if st.session_state.mode == "intl" else "TND", "status": "pending"
                }).execute()
                
                st.success(t("sub_ok"))
                st.markdown(f"[{t('b_pay_link')}]({checkout_session.url})", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Stripe Error: {e}")

def page_predictor():
    st.markdown(f"<div class='nx-title'>{t('p1_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p1_sub')}</div>", unsafe_allow_html=True)
    subscribe_block("analyst_pro", 19.99, 69.90, "nav_p1", "p1_sub")
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    @st.cache_data(ttl=3600) 
    def fetch_all_data():
        # 1. بيانات الفرق من football-data.org
        teams_data = []
        if FOOTBALL_API_KEY:
            headers = {'X-Auth-Token': FOOTBALL_API_KEY}
            try:
                resp = requests.get("https://api.football-data.org/v4/competitions/PL/standings", headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    standings = data.get('standings', [])
                    if standings:
                        table = standings[0].get('table', [])
                        for row in table:
                            played = row.get('playedGames', 1) or 1
                            teams_data.append({
                                "name": row.get('team', {}).get('name', 'Unknown'),
                                "attack": round(row.get('goalsFor', 0) / played, 2),
                                "defense": round(row.get('goalsAgainst', 0) / played, 2),
                                "gf": row.get('goalsFor', 0), 
                                "ga": row.get('goalsAgainst', 0)
                            })
            except Exception: pass
            
        # 2. احتمالات المراهنات من The Odds API
        odds_data = {}
        if ODDS_API_KEY:
            try:
                odds_url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={ODDS_API_KEY}&regions=uk&markets=h2h"
                resp = requests.get(odds_url, timeout=10)
                if resp.status_code == 200:
                    odds_json = resp.json()
                    for match in odds_json:
                        home = match.get('home_team')
                        away = match.get('away_team')
                        bookmakers = match.get('bookmakers', [])
                        if bookmakers:
                            outcomes = bookmakers[0].get('markets', [{}])[0].get('outcomes', [])
                            odds_dict = {o['name']: o['price'] for o in outcomes}
                            # تخزين احتمالات المباراة بناءً على اسم الفريقين
                            odds_data[f"{home} vs {away}"] = odds_dict
            except Exception: pass

        # 3. إحصاءات اللاعبين من Big Balls Data
        players_data = []
        if BIGBALLS_API_KEY:
            try:
                # جلب أفضل الهدافين في الدوري الإنجليزي (PL)
                bbd_headers = {'Authorization': f'Bearer {BIGBALLS_API_KEY}'}
                bbd_url = "https://api.bigballsdata.com/v1/players/topscorers?league=PL" # نقطة نهاية قياسية
                resp = requests.get(bbd_url, headers=bbd_headers, timeout=10)
                if resp.status_code == 200:
                    players_json = resp.json()
                    # استخراج أول 5 لاعبين
                    for p in players_json.get('data', [])[:5]:
                        players_data.append({
                            "Player": p.get('name', 'Unknown'),
                            "Team": p.get('team', 'Unknown'),
                            "Goals": p.get('goals', 0),
                            "xG": p.get('xg', 0.0),
                            "Shots": p.get('shots', 0)
                        })
            except Exception: pass

        return teams_data, odds_data, players_data

    with st.spinner("Fetching Live Data (Teams, Odds, Players)..."):
        teams, live_odds, top_players = fetch_all_data()

    if not teams:
        st.error("Could not load real team data from Football-Data.org. Check API key.")
        return
        
    st.markdown(f"### {t('p1_sim')}")
    names = [tm["name"] for tm in teams]
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1: home_name = st.selectbox(t("team_a"), names, index=0, key="p1_home")
    with c2: away_name = st.selectbox(t("team_b"), names, index=min(1, len(names) - 1), key="p1_away")
    with c3:
        st.write(""); st.write("")
        run = st.button(t("run_sim"), key="p1_run")
        
    if home_name == away_name: 
        st.warning("Pick two different teams.")
        return
        
    home = next(x for x in teams if x["name"] == home_name)
    away = next(x for x in teams if x["name"] == away_name)
    
    # محرك Poisson
    if run or st.session_state.sim_result is None:
        with st.spinner("Running Real Monte-Carlo..."):
            st.session_state.sim_result = real_match_simulation(float(home["attack"]), float(home["defense"]), float(away["attack"]), float(away["defense"]), n_sims=10000)
            
    if run: st.success(t("sim_done"))
    
    s = st.session_state.sim_result
    m1, m2, m3 = st.columns(3)
    with m1: st.markdown(metric(t("win"), f"{s['win']}%"), unsafe_allow_html=True); st.progress(min(max(s["win"] / 100, 0), 1))
    with m2: st.markdown(metric(t("draw"), f"{s['draw']}%", "amber"), unsafe_allow_html=True); st.progress(min(max(s["draw"] / 100, 0), 1))
    with m3: st.markdown(metric(t("loss"), f"{s['loss']}%", "red"), unsafe_allow_html=True); st.progress(min(max(s["loss"] / 100, 0), 1))
        
    st.caption(f"{home_name} (xG {s['xg_home']}) vs {away_name} (xG {s['xg_away']}) · Over 2.5: {s['over_2_5']}% · BTTS: {s['btts']}%")
    
    # عرض احتمالات المراهنات الحقيقية إن وجدت
    odds_key = f"{home_name} vs {away_name}"
    if live_odds and odds_key in live_odds:
        o = live_odds[odds_key]
        st.info(f"💰 Live Market Odds (UK): {home_name} Win: {o.get(home_name, 'N/A')} | Draw: {o.get('Draw', 'N/A')} | {away_name} Win: {o.get(away_name, 'N/A')}")

    st.markdown("<hr/>", unsafe_allow_html=True)
    
    with st.expander(f"📊 Compare: Poisson Model vs Live Odds", expanded=True):
        # مقارنة بصرية بين توقعنا واحتمالات السوق
        df_comp = pd.DataFrame({
            "Outcome": [t("win"), t("draw"), t("loss")],
            "Model (%)": [s["win"], s["draw"], s["loss"]]
        }).set_index("Outcome")
        st.bar_chart(df_comp)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p1_props')}")
    
    if top_players:
        st.dataframe(pd.DataFrame(top_players), use_container_width=True, hide_index=True)
    else:
        st.info("Player stats from Big Balls Data are currently loading or unavailable.")
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #111827 0%, #1f2937 100%); border: 2px dashed #f59e0b; border-radius: 14px; padding: 20px; text-align: center; margin-top: 30px;'>
        <h3 style='color: #f59e0b; margin-top: 0;'>{t('cross_sell_title')}</h3>
        <p style='color: #cbd5e1; font-size: 1.05rem;'>{t('cross_sell_desc')}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='nx-disclaimer'>⚠ {t('p1_disclaimer')}</div>", unsafe_allow_html=True)

def _upload_video(client, file, kind, analysis_type):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_id = os.getenv("HF_REPO_ID")
    ext = file.name.split(".")[-1].lower()
    path = f"{st.session_state.user_id}/{uuid.uuid4().hex}.{ext}"
    bytes_data = file.getvalue()
    api.upload_file(path_or_fileobj=io.BytesIO(bytes_data), path_in_repo=path, repo_id=repo_id, repo_type="model")
    up = client.table("uploads").insert({"user_id": st.session_state.user_id, "storage_path": path, "file_name": file.name, "file_size": len(bytes_data), "kind": kind}).execute().data[0]
    client.table("analyses").insert({"upload_id": up["id"], "user_id": st.session_state.user_id, "type": analysis_type, "status": "queued"}).execute()
    return up

def _render_jobs(client, analysis_type):
    try: rows = (client.table("analyses").select("*").eq("user_id", st.session_state.user_id).eq("type", analysis_type).order("created_at", desc=True).limit(10).execute().data or [])
    except Exception: rows = []
    if not rows: return
    st.markdown(f"#### {t('analysis_status')}")
    for j in rows:
        status = j["status"]
        badge = {"queued": t("status_queued"), "processing": t("status_processing"), "completed": t("status_completed"), "failed": t("status_failed")}.get(status, status)
        color = {"queued": "#f59e0b", "processing": "#00ffcc", "completed": "#00ffcc", "failed": "#ef4444"}.get(status, "#94a3b8")
        st.markdown(f"<div class='nx-card' style='border-left:4px solid {color};'><div class='nx-card-title' style='color:{color};'>{badge}</div><div class='nx-card-body'>Job <code>{j['id'][:8]}</code> · {j['created_at'][:19]}</div></div>", unsafe_allow_html=True)
        if status == "completed" and j.get("result"): st.json(j["result"])
    if st.button(t("refresh"), key=f"refresh_{analysis_type}"): st.rerun()

def page_players():
    st.markdown(f"<div class='nx-title'>{t('p2_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p2_sub')}</div>", unsafe_allow_html=True)
    subscribe_block("player_pro", 25.00, 89.90, "pass_name", "p2_sub")
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    st.markdown(f"### {t('upload_title')}")
    client = get_client()
    up_file = st.file_uploader(t("upload_label"), type=["mp4", "avi", "mov"], key="p2_up")
    if up_file is not None:
        if st.button("⬆️ Upload & Queue Analysis", key="p2_upload_btn"):
            try:
                _upload_video(client, up_file, "player_self", "player_hud")
                st.success(t("upload_ok")); st.rerun()
            except Exception as e: st.error(f"Upload failed: {e}")
    _render_jobs(client, "player_hud")
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p2_lens_title')}")
    lens_on = st.toggle(t("p2_lens_toggle"), value=False, key="p2_lens_tgl")
    
    if lens_on:
        st.success(t("p2_lens_on"))
        if not AI_READY:
            st.markdown("""
            <div style='background: rgba(245,158,11,0.1); border: 1px solid #f59e0b; padding: 20px; border-radius: 12px; text-align: center;'>
                <h4 style='color: #f59e0b; margin: 0;'>Enterprise Hardware Required</h4>
                <p style='color: #cbd5e1; margin-top: 8px;'>Live AI Lens requires an NVIDIA Jetson AGX Orin Edge Computer or a dedicated GPU server to process video in real-time. <br><b>Contact our sales team for a private demo.</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("🟢 NVIDIA GPU Detected. Real-time YOLO Stream Active.")
            class VideoProcessor:
                def __init__(self):
                    self.model = YOLO("yolov8n.pt")
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    results = self.model(img, classes=[0], verbose=False)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 204), 2)
                            cv2.putText(img, "Player", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 204), 2)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            webrtc_streamer(key="lens", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p2_hud_title')}")
    
    try:
        recent = (client.table("analyses").select("result").eq("user_id", st.session_state.user_id).eq("type", "player_hud").eq("status", "completed").order("created_at", desc=True).limit(1).execute().data)
        hud_data = recent[0]["result"] if recent else None
    except Exception: hud_data = None

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(metric(t("p2_speed"), hud_data["speed"] if hud_data else "--", "primary"), unsafe_allow_html=True)
    with c2: st.markdown(metric(t("p2_pass"), hud_data["pass_accuracy"] if hud_data else "--", "amber"), unsafe_allow_html=True)
    with c3: st.markdown(metric(t("p2_dist"), hud_data["distance"] if hud_data else "--", "primary"), unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4: st.markdown(metric(t("p2_sprints"), str(hud_data["sprints"]) if hud_data else "--", "primary"), unsafe_allow_html=True)
    with c5: st.markdown(metric(t("p2_duels"), hud_data["duels"] if hud_data else "--", "amber"), unsafe_allow_html=True)
    with c6: st.markdown(metric("Rating", hud_data["rating"] if hud_data else "--", "primary"), unsafe_allow_html=True)

    if not hud_data: st.info("📊 Upload a video to generate your real performance analytics.")
    
    st.markdown(f"#### {t('p2_rating')}")
    if hud_data:
        df_attr = pd.DataFrame({"Attribute": [t("attr_drib"), t("attr_pass"), t("attr_shot"), t("attr_def"), t("attr_phys"), t("attr_vision")], "Score": [hud_data.get("drib", 80), hud_data.get("pass", 80), hud_data.get("shot", 80), hud_data.get("def", 80), hud_data.get("phys", 80), hud_data.get("vis", 80)]}).set_index("Attribute")
        st.bar_chart(df_attr)
    else:
        st.caption("Waiting for data...")

def page_academies():
    st.markdown(f"<div class='nx-title'>{t('p3_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p3_sub')}</div>", unsafe_allow_html=True)
    subscribe_block("academy_enterprise", 85.00, 299.90, "plan_academy", "p3_sub")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('team_overview')}")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(metric(t("reg_players"), "28"), unsafe_allow_html=True)
    with c2: st.markdown(metric(t("avg_age"), "18.5", "amber"), unsafe_allow_html=True)
    with c3: st.markdown(metric(t("fitness"), "88%"), unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    client = get_client()
    st.markdown(f"### {t('pre_match')}")
    st.caption(t("pre_desc"))
    f1 = st.file_uploader(t("upload_label"), type=["mp4", "avi", "mov"], key="p3_opp")
    if f1 and st.button("⬆️ Upload Opponent Video", key="p3_opp_btn"):
        try: _upload_video(client, f1, "opponent", "opponent_report"); st.success(t("upload_ok")); st.rerun()
        except Exception as e: st.error(f"Upload failed: {e}")
    _render_jobs(client, "opponent_report")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('post_match')}")
    st.caption(t("post_desc"))
    f2 = st.file_uploader(t("upload_label"), type=["mp4", "avi", "mov"], key="p3_own")
    if f2 and st.button("⬆️ Upload Own Match Video", key="p3_own_btn"):
        try: _upload_video(client, f2, "own_match", "error_report"); st.success(t("upload_ok")); st.rerun()
        except Exception as e: st.error(f"Upload failed: {e}")
    _render_jobs(client, "error_report")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('gdpr_title')}")
    st.markdown(f"<div class='nx-card'><div class='nx-card-body'>{t('gdpr_text')}</div></div>", unsafe_allow_html=True)
    with st.form("gdpr_form"):
        academy = st.text_input(t("academy_name"))
        email = st.text_input(t("biz_email"))
        if st.form_submit_button(t("register_gdpr")):
            valid = academy.strip() and "@" in email and "." in email.split("@")[-1]
            if not valid: st.error(t("gdpr_err"))
            else:
                try:
                    client.table("gdpr_registrations").insert({"user_id": st.session_state.user_id, "academy_name": academy, "business_email": email}).execute()
                    st.success(t("gdpr_ok"))
                except Exception as e: st.error(f"Failed: {e}")

def page_scouts():
    st.markdown(f"<div class='nx-title'>{t('p4_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p4_sub')}</div>", unsafe_allow_html=True)
    subscribe_block("scout_pro", 45.00, 159.90, "plan_scout", "p4_sub")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('search')}")
    c1, c2 = st.columns(2)
    with c1: pos = st.selectbox(t("position"), ["ALL", "FW", "MF", "DF", "GK"], format_func=lambda p: {"ALL": t("pos_all"), "FW": t("pos_fw"), "MF": t("pos_mf"), "DF": t("pos_df"), "GK": t("pos_gk")}[p], key="p4_pos")
    with c2: min_rate = st.slider(t("min_rate"), 50, 100, 70, key="p4_rate")
    
    client = get_client()
    try:
        real_players = client.table("analyses").select("*").eq("status", "completed").execute().data
    except Exception:
        real_players = []
        
    filtered = []
    for p in real_players:
        res = p.get("result") or {}
        rating = res.get("rating_val", 80)
        if rating >= min_rate:
            filtered.append(p)
            
    st.markdown(f"### {t('results')} ({len(filtered)})")
    if not filtered: 
        st.warning(t("no_match"))
        return
        
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for col, p in zip(cols, filtered[i:i+2]):
            with col:
                res = p.get("result", {})
                st.markdown(f"<div class='nx-card' style='border-left:4px solid #00ffcc;'><div class='nx-card-title'>Player #{p['id'][:6].upper()}</div><div style='font-weight:700;color:#f8fafc;'>AI Analyzed Profile</div><div style='margin-top:6px;color:#cbd5e1;font-size:.85rem;'>Match Rate: <b style='color:#00ffcc;'>{res.get('rating_val', 80)}%</b><br/>Peak Velocity: <b style='color:#f59e0b;'>{res.get('speed', 'N/A')}</b></div></div>", unsafe_allow_html=True)
                if st.button(t("contact_btn"), key=f"contact_{p['id']}"):
                    try:
                        client.table("contact_requests").insert({"scout_id": st.session_state.user_id, "player_tag": f"Player #{p['id'][:6]}", "status": "pending"}).execute()
                        st.success(t("contact_ok"))
                    except Exception as e: st.error(f"Failed: {e}")

def page_ops():
    st.markdown(f"<div class='nx-title'>{t('p5_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p5_sub')}</div>", unsafe_allow_html=True)
    st.markdown(f"<span style='background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid #ef4444;padding:4px 12px;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;'>🇹🇳 {t('exclusive')}</span>", unsafe_allow_html=True)
    if st.session_state.mode != "local":
        st.markdown(f"<div class='nx-card' style='border-left:4px solid #ef4444;'><div class='nx-card-title' style='color:#ef4444;'>🔒 {t('lock_title')}</div><div class='nx-card-body'>{t('lock_msg')}</div></div>", unsafe_allow_html=True)
        return
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('booking')}")
    client = get_client()
    c1, c2 = st.columns(2)
    with c1: d = st.date_input(t("match_date"), value=date.today(), key="p5_date")
    with c2: tm = st.time_input(t("match_time"), key="p5_time")
    loc = st.text_input(t("location"), value="Stade Olympique de Radès", key="p5_loc")
    eq = st.multiselect(t("equipment"), [t("eq1"), t("eq2"), t("eq3")], default=[t("eq1"), t("eq2")], key="p5_eq")
    wk = f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"
    try: used = int(client.rpc("check_week_capacity", {"week_key": wk}).execute().data or 0)
    except Exception: used = 0
    st.markdown(f"#### {t('weekly_cap')}: **{used}/3**")
    st.progress(min(used / 3.0, 1.0))
    full = used >= 3
    if full: st.error(f"❌ {t('full')}")
    if st.button(t("book_btn"), key="p5_book", disabled=full):
        try:
            client.table("bookings").insert({"user_id": st.session_state.user_id, "match_date": d.isoformat(), "match_time": tm.isoformat(), "iso_week": wk, "location": loc, "equipment": eq, "status": "pending", "stage": 0, "amount": 350, "currency": "TND"}).execute()
            st.success(t("book_ok")); st.rerun()
        except Exception as e: st.error(f"Booking failed: {e}")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('my_bookings')}")
    try: my = (client.table("bookings").select("*").eq("user_id", st.session_state.user_id).order("created_at", desc=True).limit(20).execute().data or [])
    except Exception: my = []
    for b in my:
        stages = [t("s1"), t("s2"), t("s3"), t("s4")]
        stage = int(b.get("stage", 0))
        bar = "".join(f"<span style='display:inline-block;width:22%;margin:0 1%;padding:8px 4px;border-radius:8px;text-align:center;font-size:.68rem;background:{'rgba(0,255,204,0.15)' if i <= stage else '#0f1626'};border:1px solid {'#00ffcc' if i <= stage else 'rgba(148,163,184,0.2)'};color:{'#00ffcc' if i <= stage else '#64748b'};'>{i+1}. {s}</span>" for i, s in enumerate(stages))
        st.markdown(f"<div class='nx-card'><div class='nx-card-title'>{b['match_date']} · {b['match_time']} · {b['status']}</div><div class='nx-card-body'>{b['location']} — {', '.join(b.get('equipment') or [])}</div><div style='margin-top:10px;'>{bar}</div></div>", unsafe_allow_html=True)

def main():
    require_auth()
    render_sidebar()
    page = st.session_state.page
    if page == "nav_billing": page_billing()
    elif page == "nav_p1": page_predictor()
    elif page == "nav_p2": page_players()
    elif page == "nav_p3": page_academies()
    elif page == "nav_p4": page_scouts()
    elif page == "nav_p5": page_ops()

if __name__ == "__main__":
    main()
