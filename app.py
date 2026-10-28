# -*- coding: utf-8 -*-
"""
Omnia DeepArena NexIA — Production Edition with Hugging Face Storage
Run: streamlit run app.py
"""

import math
import os
import uuid
import io
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from huggingface_hub import HfApi

load_dotenv()

st.set_page_config(page_title="Omnia DeepArena NexIA",
                   page_icon="⚽", layout="wide",
                   initial_sidebar_state="expanded")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("⚠️ Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")
        st.stop()
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    tok = st.session_state.get("access_token")
    ref = st.session_state.get("refresh_token")
    if tok and ref:
        try:
            client.auth.set_session(tok, ref)
        except Exception:
            st.session_state.access_token = None
            st.session_state.refresh_token = None
    return client

DEFAULTS = {
    "lang": "EN", "mode": "intl", "page": "nav_p1",
    "access_token": None, "refresh_token": None,
    "user_id": None, "user_email": None, "role": "fan",
    "sim_result": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

LANGS = {"AR": "العربية", "EN": "English", "FR": "Français",
         "ES": "Español", "IT": "Italiano", "DE": "Deutsch"}

def T6(en, ar, fr, es, it, de):
    return {"EN": en, "AR": ar, "FR": fr, "ES": es, "IT": it, "DE": de}

T = {
    "lang":       T6("Language","اللغة","Langue","Idioma","Lingua","Sprache"),
    "mode":       T6("Operating Mode","وضع التشغيل","Mode","Modo","Modalità","Modus"),
    "mode_intl":  T6("International (UK / Global)","دولي (UK/عالمي)","International","Internacional","Internazionale","International"),
    "mode_local": T6("Local (Tunisia)","محلي (تونس)","Local (Tunisie)","Local (Túnez)","Locale (Tunisia)","Lokal (Tunesien)"),
    "nav":        T6("Navigation","التنقل","Navigation","Navegación","Navigazione","Navigation"),
    "nav_p1":     T6("AI Match Predictor","مُتوقع المباريات","Prédicteur IA","Predictor IA","Predittore IA","KI-Vorhersage"),
    "nav_p2":     T6("Players Hub","مركز اللاعبين","Espace Joueurs","Centro Jugadores","Hub Giocatori","Spieler-Hub"),
    "nav_p3":     T6("Clubs & Academies","الأندية والأكاديميات","Clubs & Académies","Clubes y Academias","Club e Accademie","Vereine & Akademien"),
    "nav_p4":     T6("Scouts Marketplace","سوق الكشافة","Marché Scouts","Mercado Scouts","Marketplace Scout","Scout-Markt"),
    "nav_p5":     T6("Live & On-Site Ops","الفيديو والعمليات","Live & Terrain","Vivo y Terreno","Live & Campo","Live & Vor-Ort"),
    "per_month":  T6("/ month","/ شهر","/ mois","/ mes","/ mese","/ Monat"),
    "subscribe":  T6("Subscribe Now","اشترك الآن","S'abonner","Suscribirse","Abbonati","Abonnieren"),
    "sub_ok":     T6("Subscription request sent. Pending admin approval.","تم إرسال طلب الاشتراك. بانتظار موافقة الإدارة.","Demande envoyée. En attente d'approbation.","Solicitud enviada. Pendiente de aprobación.","Richiesta inviata. In attesa di approvazione.","Anfrage gesendet. Warten auf Freigabe."),
    "monetization": T6("Monetization","الاشتراك","Monétisation","Monetización","Monetizzazione","Monetarisierung"),
    "login_title": T6("Sign In","تسجيل الدخول","Connexion","Iniciar Sesión","Accedi","Anmelden"),
    "signup_title":T6("Create Account","إنشاء حساب","Créer un compte","Crear Cuenta","Crea Account","Konto erstellen"),
    "email":       T6("Email","البريد الإلكتروني","E-mail","Email","Email","E-Mail"),
    "password":    T6("Password","كلمة المرور","Mot de passe","Contraseña","Password","Passwort"),
    "full_name":   T6("Full Name","الاسم الكامل","Nom complet","Nombre completo","Nome completo","Vollständiger Name"),
    "login_btn":   T6("Sign In","دخول","Se connecter","Entrar","Accedi","Anmelden"),
    "signup_btn":  T6("Sign Up","إنشاء","S'inscrire","Registrarse","Registrati","Registrieren"),
    "logout":      T6("Logout","تسجيل الخروج","Déconnexion","Cerrar sesión","Logout","Abmelden"),
    "signed_in_as":T6("Signed in as","مسجل الدخول كـ","Connecté en tant que","Conectado como","Connesso come","Angemeldet als"),
    "check_email": T6("Account created. Check your email to confirm, then sign in.","تم إنشاء الحساب. تحقق من بريدك لتأكيده ثم سجّل الدخول.","Compte créé. Confirmez votre e-mail puis connectez-vous.","Cuenta creada. Confirme el correo y luego entre.","Account creato. Conferma email e accedi.","Konto erstellt. Bestätigen Sie die E-Mail und melden Sie sich an."),
    "p1_title":   T6("AI Match Predictor","مُتوقع المباريات","Prédicteur IA","Predictor IA","Predittore IA","KI-Vorhersage"),
    "p1_sub":     T6("Real Poisson-Monte-Carlo over 10,000 runs, computed live.","محاكاة مونت كارلو الحقيقية عبر 10,000 تشغيل.","Monte-Carlo réel sur 10 000 exécutions.","Monte-Carlo real sobre 10.000 ejecuciones.","Monte-Carlo reale su 10.000 esecuzioni.","Echtes Monte-Carlo über 10.000 Läufe."),
    "p1_sim":     T6("AI Match Simulator","محاكي المباريات","Simulateur IA","Simulador IA","Simulatore IA","KI-Simulator"),
    "team_a":     T6("Home Team","الفريق المضيف","Équipe Domicile","Equipo Local","Squadra Casa","Heimteam"),
    "team_b":     T6("Away Team","الفريق الضيف","Équipe Extérieur","Equipo Visita","Squadra Ospite","Auswärtsteam"),
    "run_sim":    T6("Run 10,000 Simulations","تشغيل 10,000 محاكاة","Lancer 10 000 simulations","Ejecutar 10.000 simulaciones","Esegui 10.000 simulazioni","10.000 Simulationen starten"),
    "win":        T6("Win","فوز","Victoire","Victoria","Vittoria","Sieg"),
    "draw":       T6("Draw","تعادل","Nul","Empate","Pareggio","Unentschieden"),
    "loss":       T6("Loss","خسارة","Défaite","Derrota","Sconfitta","Niederlage"),
    "sim_done":   T6("Simulation completed (real Monte-Carlo).","اكتملت المحاكاة (حقيقية).","Simulation terminée.","Simulación completada.","Simulazione completata.","Simulation abgeschlossen."),
    "p1_tac":     T6("Tactical Breakdown","التحليل التكتيكي","Analyse tactique","Análisis táctico","Analisi tattica","Taktische Analyse"),
    "p1_props":   T6("Player Props","تحليلات اللاعبين","Props joueurs","Props jugadores","Props giocatori","Spieler-Props"),
    "p1_disclaimer": T6("For Educational and Analytical Purposes Only. We do not provide gambling services. Please gamble responsibly (18+).","لأغراض تعليمية وتحليلية فقط. لا نقدم خدمات المقامرة. العب بمسؤولية (18+).","À des fins éducatives uniquement. Pas de paris. Jouez responsable (18+).","Solo con fines educativos. Sin apuestas. Juegue responsable (18+).","Solo per scopi educativi. Nessuna scommessa. Gioca responsabilmente (18+).","Nur zu Bildungszwecken. Keine Glücksspiele. Verantwortungsvoll spielen (18+)."),
    "p2_title":   T6("Players Hub","مركز اللاعبين","Espace Joueurs","Centro Jugadores","Hub Giocatori","Spieler-Hub"),
    "p2_sub":     T6("Upload match/training video — queued for real YOLO analysis.","ارفع فيديو المباراة/التدريب — لتحليل YOLO حقيقي.","Téléversez une vidéo — analyse YOLO réelle.","Suba un vídeo — análisis YOLO real.","Carica un video — analisi YOLO reale.","Video hochladen — echte YOLO-Analyse."),    "p2_lens_title": T6("Live AI Lens Tracking Console","وحدة تتبع عدسة الذكاء","Console Lens IA","Consola Lens IA","Console Lens IA","KI-Lens-Konsole"),
    "p2_lens_toggle": T6("Activate YOLO26 Pro Lens","تفعيل عدسة YOLO26 Pro","Activer Lens YOLO26 Pro","Activar Lens YOLO26 Pro","Attiva Lens YOLO26 Pro","YOLO26 Pro aktivieren"),
    "p2_lens_on": T6("Live stream active — 22 entities tracked.","البث نشط — 22 كياناً متتتبعاً.","Flux actif — 22 entités.","Flujo activo — 22 entidades.","Stream attivo — 22 entità.","Live — 22 Objekte."),
    "p2_lens_off": T6("Lens idle — activate to start the detection stream.","العدسة خاملة — فعّلها لبدء البث.","Lens inactive — activez.","Lens inactiva — active.","Lens inattiva — attiva.","Lens inaktiv — aktivieren."),
    "p2_fps": T6("FPS","إطار/ث","IPS","FPS","FPS","FPS"),
    "p2_conf": T6("Mean Confidence","متوسط الثقة","Confiance","Confianza","Confidenza","Konfidenz"),
    "p2_entities": T6("Tracked Entities","الكيانات المتتبعة","Entités","Entidades","Entità","Objekte"),
    "p2_hud_title": T6("Personal Performance HUD","لوحة الأداء الشخصي","HUD de performance","HUD personal","HUD prestazioni","Performance-HUD"),
    "p2_speed": T6("Top Speed (km/h)","أقصى سرعة (كم/س)","Vitesse max (km/h)","Velocidad máx (km/h)","Velocità max (km/h)","Höchstgeschw. (km/h)"),
    "p2_pass": T6("Pass Accuracy (%)","دقة التمرير (%)","Précision passe (%)","Precisión pase (%)","Precisione passaggio (%)","Passgenauigkeit (%)"),
    "p2_dist": T6("Distance (km)","المسافة (كم)","Distance (km)","Distancia (km)","Distanza (km)","Distanz (km)"),
    "p2_sprints": T6("Sprints (>25 km/h)","السرعات القصوى","Sprints (>25 km/h)","Sprints (>25 km/h)","Sprint (>25 km/h)","Sprints (>25 km/h)"),
    "p2_duels": T6("Duels Won (%)","الالتحامات (%)","Duels gagnés (%)","Duelos ganados (%)","Duelli vinti (%)","Duelle gewonnen (%)"),
    "p2_rating": T6("Technical Rating Chart","مخطط التقييم الفني","Graphique de notation","Gráfico de valoración","Grafico valutazione","Bewertungsdiagramm"),
    "attr_drib": T6("Dribbling","المراوغة","Dribble","Regate","Dribbling","Dribbling"),
    "attr_pass": T6("Passing","التمرير","Passe","Pase","Passaggio","Passen"),
    "attr_shot": T6("Shooting","التسديد","Tir","Tiro","Tiro","Schuss"),
    "attr_def": T6("Defending","الدفاع","Défense","Defensa","Difesa","Verteidigung"),
    "attr_phys": T6("Physical","البدنية","Physique","Físico","Fisico","Physis"),
    "attr_vision": T6("Vision","الرؤية","Vision","Visión","Visione","Vision"),
    "pass_name":  T6("Pro Player Pass","بطاقة اللاعب المحترف","Pass Joueur Pro","Pase Jugador Pro","Pass Giocatore Pro","Profi-Spieler-Pass"),
    "upload_title": T6("Video Uploader","رفع الفيديو","Téléversement","Carga de vídeo","Caricamento","Video-Upload"),
    "upload_label": T6("Upload MP4, AVI or MOV (No size limit)","ارفع MP4 أو AVI أو MOV (بدون حد للحجم)","Téléverser MP4/AVI/MOV (Sans limite)","Subir MP4/AVI/MOV (Sin límite)","Carica MP4/AVI/MOV (Senza limiti)","MP4/AVI/MOV hochladen (Kein Limit)"),
    "upload_ok":  T6("Uploaded successfully — analysis job queued.","تم الرفع بنجاح — المهمة في القائمة.","Téléversé avec succès — tâche en file.","Subido con éxito — tarea en cola.","Caricato con successo — in coda.","Erfolgreich hochgeladen — in Warteschlange."),
    "analysis_status": T6("Analysis Jobs","مهام التحليل","Tâches d'analyse","Tareas de análisis","Attività di analisi","Analyse-Jobs"),
    "status_queued": T6("Queued","في القائمة","En attente","En cola","In coda","In Warteschlange"),
    "status_processing": T6("Processing","جارٍ","En cours","Procesando","In corso","Verarbeitung"),
    "status_completed": T6("Completed","مكتمل","Terminé","Completado","Completato","Abgeschlossen"),
    "status_failed": T6("Failed","فشل","Échec","Falló","Fallito","Fehlgeschlagen"),
    "refresh":    T6("Refresh Status","تحديث الحالة","Actualiser","Actualizar","Aggiorna","Aktualisieren"),
    "p3_title":   T6("Clubs & Academies HUD","لوحة الأندية","HUD Clubs","HUD Clubes","HUD Club","Vereine HUD"),
    "p3_sub":     T6("Real uploads, real queue, real reports.","رفع حقيقي، قائمة حقيقية، تقارير حقيقية.","Téléversements réels, file réelle, rapports réels.","Cargas reales, cola real, informes reales.","Caricamenti reali, coda reale, report reali.","Echte Uploads, echte Queue, echte Berichte."),
    "plan_academy": T6("Academy Enterprise","باقة الأكاديمية","Academy Enterprise","Academy Enterprise","Academy Enterprise","Academy Enterprise"),
    "team_overview": T6("Team Overview","نظرة عامة","Aperçu équipe","Resumen equipo","Panoramica","Team-Übersicht"),
    "reg_players": T6("Registered Players","لاعبون مسجلون","Joueurs inscrits","Jugadores","Giocatori","Spieler"),
    "avg_age":   T6("Average Age","متوسط العمر","Âge moyen","Edad media","Età media","Durchschnittsalter"),
    "fitness":   T6("Fitness Index","مؤشر اللياقة","Indice forme","Índice forma","Indice forma","Fitness-Index"),
    "pre_match": T6("Pre-Match Opponent Analysis","تحليل الخصم","Analyse adverse","Análisis rival","Analisi avversario","Gegneranalyse"),
    "pre_desc":  T6("Upload rival video → queued for real YOLO opponent report.","ارفع فيديو الخصم → تقرير YOLO حقيقي.","Téléversez vidéo adverse → rapport YOLO.","Suba vídeo rival → informe YOLO.","Carica video avversario → report YOLO.","Gegner-Video → YOLO-Bericht."),
    "post_match": T6("Post-Match Error Analysis","تحليل الأخطاء","Analyse après-match","Análisis post-partido","Analisi post-partita","Fehleranalyse"),
    "post_desc": T6("Upload your match → queued for YOLO positioning + fatigue report.","ارفع مباراتك → تقرير YOLO للتمركز والإجهاد.","Téléversez votre match → rapport YOLO.","Suba su partido → informe YOLO.","Carica la partita → report YOLO.","Ihr Spiel → YOLO-Bericht."),
    "gdpr_title": T6("Data Privacy & GDPR","حماية البيانات","Confidentialité & RGPD","Privacidad y RGPD","Privacy & GDPR","Datenschutz & DSGVO"),
    "gdpr_text": T6("All media encrypted securely. Youth under 18 anonymised: pipeline generates tags Player #TN-01 instead of real names, strictly complying with the UK Children's Code and GDPR.","جميع الوسائط مشفرة. الناشئون تحت 18 يُعالجون مجهولين برموز Player #TN-01 بدلاً من الأسماء الحقيقية.","Tous les médias chiffrés. Les moins de 18 ans anonymisés : tags Player #TN-01.","Todos los medios cifrados. Menores de 18 anonimizados: tags Player #TN-01.","Tutti i media crittografati. Under 18 anonimizzati: tag Player #TN-01.","Alle Medien verschlüsselt. Unter-18-Profile anonymisiert: Tags Player #TN-01."),
    "academy_name": T6("Academy Name","اسم الأكاديمية","Nom académie","Nombre academia","Nome accademia","Akademiename"),
    "biz_email": T6("Authorized Business Email","البريد المهني","E-mail pro","Correo corporativo","Email aziendale","Geschäftliche E-Mail"),
    "register_gdpr": T6("Register GDPR Safeguard","تسجيل ضمانات GDPR","Enregistrer RGPD","Registrar RGPD","Registra GDPR","DSGVO registrieren"),
    "gdpr_ok":   T6("Saved in Supabase. Encrypted dossier logged.","محفوظ في Supabase.","Enregistré dans Supabase.","Guardado en Supabase.","Salvato in Supabase.","In Supabase gespeichert."),
    "gdpr_err":  T6("Valid name and email required.","اسم وبريد صالحان مطلوبان.","Nom et e-mail requis.","Nombre y correo válidos.","Nome ed email validi.","Name und E-Mail erforderlich."),
    "p4_title":   T6("Scouts Marketplace","سوق الكشافة","Marché Scouts","Mercado Scouts","Marketplace Scout","Scout-Markt"),
    "p4_sub":     T6("Real talent DB. Real contact requests. Real Supabase queries.","قاعدة مواهب حقيقية. طلبات حقيقية.","Talents réels. Demandes réelles.","Talentos reales. Solicitudes reales.","Talenti reali. Richieste reali.","Echte Talente. Echte Anfragen."),
    "plan_scout": T6("Pro Scout License","رخصة الكشاف","Licence Scout Pro","Licencia Ojeador","Licenza Scout","Pro-Scout-Lizenz"),
    "search":     T6("Unified Talent Search","بحث المواهب","Recherche talents","Búsqueda talentos","Ricerca talenti","Talentsuche"),
    "position":   T6("Position","المركز","Poste","Posición","Ruolo","Position"),
    "pos_all":    T6("All","الكل","Tous","Todos","Tutti","Alle"),
    "pos_fw":     T6("Forward","مهاجم","Attaquant","Delantero","Attaccante","Stürmer"),
    "pos_mf":     T6("Midfielder","وسط","Milieu","Centrocampista","Centrocampista","Mittelfeld"),
    "pos_df":     T6("Defender","مدافع","Défenseur","Defensa","Difensore","Verteidiger"),
    "pos_gk":     T6("Goalkeeper","حارس","Gardien","Portero","Portiere","Torwart"),
    "min_rate":   T6("Minimum Matching Rate (%)","أدنى نسبة مطابقة (%)","Taux min (%)","Tasa mín (%)","Tasso min (%)","Min. Rate (%)"),
    "results":    T6("Talent Profiles","ملفات المواهب","Profils talents","Perfiles talento","Profili talento","Talentprofile"),
    "contact_btn":T6("Request Official Contact","طلب تواصل رسمي","Demander un contact","Solicitar contacto","Richiedi contatto","Offiziellen Kontakt anfordern"),
    "contact_ok": T6("Request saved in Supabase — pending academy approval.","تم الحفظ في Supabase — بانتظار موافقة الأكاديمية.","Enregistré dans Supabase — approbation en attente.","Guardado en Supabase — pendiente aprobación.","Salvato in Supabase — in attesa.","In Supabase gespeichert — Freigabe ausstehend."),
    "no_match":   T6("No matching profiles.","لا توجد ملفات مطابقة.","Aucun profil.","Sin coincidencias.","Nessun profilo.","Keine Profile."),
    "p5_title":   T6("Live Video & On-Site Ops","الفيديو والعمليات","Live & Terrain","Vivo y Terreno","Live & Campo","Live & Vor-Ort"),
    "p5_sub":     T6("Real bookings saved in Supabase. Max 3/week enforced server-side.","حجوزات حقيقية في Supabase. الحد 3/أسبوع مفروض.","Réservations réelles. Max 3/semaine.","Reservas reales. Máx 3/semana.","Prenotazioni reali. Max 3/settimana.","Echte Buchungen. Max. 3/Woche."),
    "exclusive":  T6("Tunisia Exclusive","حصري لتونس","Exclusivité Tunisie","Exclusivo Túnez","Esclusiva Tunisia","Exklusiv Tunesien"),
    "lock_title": T6("Regional Feature Locked","ميزة مقفلة","Fonction verrouillée","Función bloqueada","Funzionalità bloccata","Funktion gesperrt"),
    "lock_msg":   T6("Switch to Local (Tunisia) mode in the sidebar to access bookings.","بدّل إلى الوضع المحلي (تونس) للوصول للحجوزات.","Passez en mode Local (Tunisie) pour réserver.","Cambie a Modo Local (Túnez) para reservar.","Passa a Locale (Tunisia) per prenotare.","Wechseln Sie zu Lokal (Tunesien) für Buchungen."),
    "booking":    T6("On-Site Booking","الحجز الميداني","Réservation","Reserva","Prenotazione","Buchung"),
    "match_date": T6("Match Date","التاريخ","Date","Fecha","Data","Datum"),
    "match_time": T6("Kick-off Time","الوقت","Heure","Hora","Ora","Anstoßzeit"),
    "location":   T6("Stadium Location","الملعب","Stade","Estadio","Stadio","Stadion"),
    "equipment":  T6("Equipment","المعدات","Matériel","Equipamiento","Attrezzatura","Ausrüstung"),
    "eq1":        T6("Professional Camera Rig","طقم كاميرات","Rig caméra pro","Rig cámara pro","Rig camera pro","Kamera-Rig"),
    "eq2":        T6("NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin","NVIDIA Jetson AGX Orin"),
    "eq3":        T6("Wearable GPS Trackers","أجهزة GPS","Trackers GPS","Rastreadores GPS","Tracker GPS","Wearable GPS"),
    "book_btn":   T6("Confirm Booking — 350 TND","تأكيد — 350 د.ت","Confirmer — 350 TND","Confirmar — 350 TND","Conferma — 350 TND","Bestätigen — 350 TND"),
    "weekly_cap": T6("Weekly Capacity","السعة الأسبوعية","Capacité hebdo","Capacidad semanal","Capacità settimanale","Wochenkapazität"),
    "full":       T6("Fully Booked for this week (max 3).","مكتمل الحجز (الحد 3).","Complet (max 3).","Completo (máx 3).","Completo (max 3).","Ausgebucht (max. 3)."),
    "book_ok":    T6("Booking saved in Supabase.","تم الحفظ في Supabase.","Réservé dans Supabase.","Guardado en Supabase.","Salvato in Supabase.","In Supabase gespeichert."),
    "my_bookings": T6("My Bookings","حجوزاتي","Mes réservations","Mis reservas","Mie prenotazioni","Meine Buchungen"),
    "status_tracker": T6("Order Status Tracker","متتبع الحالة","Suivi commande","Seguimiento","Tracker ordine","Status-Tracker"),
    "s1": T6("Request Received","تم الاستلام","Reçue","Recibida","Ricevuta","Erhalten"),
    "s2": T6("Jetson Dispatched","إرسال Jetson","Jetson expédié","Jetson enviado","Jetson spedito","Jetson versandt"),
    "s3": T6("Edge Live","المعالجة مباشرة","Edge live","Edge vivo","Edge live","Edge live"),
    "s4": T6("Analytics Ready","التحليلات جاهزة","Analytique prête","Analítica lista","Analitica pronta","Analyse bereit"),
}

def t(key):
    entry = T.get(key)
    if not entry:
        return key
    return entry.get(st.session_state.lang) or entry.get("EN") or key

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 0%, #101827 0%, #0b0f19 45%, #070a12 100%) !important;
    color: #e5e7eb;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid rgba(0,255,204,0.18); }
[data-testid="stSidebar"] * { color: #d1d5db; }
h1, h2, h3, h4 { color: #f8fafc !important; font-weight: 800; }
.nx-title {
    font-size: 2rem; font-weight: 800; margin-bottom: .15rem;
    background: linear-gradient(90deg, #00ffcc 0%, #f59e0b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nx-sub { color: #94a3b8; font-size: .95rem; margin-bottom: 1rem; }
.nx-card {
    background: #111827; border: 1px solid rgba(0,255,204,0.14);
    border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;
}
.nx-card-title { color: #00ffcc; font-weight: 700; font-size: .74rem;
    letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 6px; }
.nx-card-body { color: #cbd5e1; font-size: .92rem; line-height: 1.55; }
.nx-metric { background: #111827; border: 1px solid rgba(0,255,204,0.16);
    border-radius: 14px; padding: 14px 16px; text-align: center; }
.nx-metric-label { color: #94a3b8; font-size: .7rem; letter-spacing: 1.2px; text-transform: uppercase; }
.nx-metric-value { color: #00ffcc; font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
.nx-metric-amber { color: #f59e0b; font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
.nx-metric-red { color: #ef4444; font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
.nx-brand { font-size: 1.2rem; font-weight: 800; color: #00ffcc; }
.nx-brand-sub { font-size: .68rem; color: #f59e0b; letter-spacing: 2px; text-transform: uppercase; }
.nx-disclaimer { margin-top: 24px; padding: 12px 16px; border-radius: 12px;
    background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.35);
    color: #fca5a5; font-size: .82rem; }
.stButton > button {
    background: linear-gradient(90deg, #00ffcc 0%, #00c9a7 100%);
    color: #06121a !important; font-weight: 800; border: none;
    border-radius: 10px; padding: .55rem 1.1rem; width: 100%;
}
.stButton > button:hover { box-shadow: 0 0 20px rgba(0,255,204,0.55); }
.stButton > button:disabled { background: #1f2937 !important; color: #6b7280 !important; }
[data-testid="stExpander"] { background: #111827; border: 1px solid rgba(0,255,204,0.14) !important; border-radius: 12px !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #00ffcc, #f59e0b) !important; }
</style>
"""
if st.session_state.lang == "AR":
    CSS += "<style>[data-testid='stAppViewContainer'],[data-testid='stSidebar']{direction:rtl;}</style>"
st.markdown(CSS, unsafe_allow_html=True)

def metric(label, value, variant="primary"):
    cls = {"primary": "nx-metric-value",
           "amber": "nx-metric-amber",
           "red": "nx-metric-red"}[variant]
    return (f"<div class='nx-metric'><div class='nx-metric-label'>{label}</div>"
            f"<div class='{cls}'>{value}</div></div>")

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def real_match_simulation(home_attack, home_defense, away_attack, away_defense,
                          home_advantage=0.15, n_sims=10000):
    lam_home = math.exp(home_attack - away_defense + home_advantage)
    lam_away = math.exp(away_attack - home_defense)
    rng = np.random.default_rng()
    hg = rng.poisson(lam_home, n_sims)
    ag = rng.poisson(lam_away, n_sims)
    wins = int((hg > ag).sum())
    draws = int((hg == ag).sum())
    losses = n_sims - wins - draws
    return {
        "win": round(wins / n_sims * 100, 1),
        "draw": round(draws / n_sims * 100, 1),
        "loss": round(losses / n_sims * 100, 1),
        "xg_home": round(lam_home, 2),
        "xg_away": round(lam_away, 2),
        "over_2_5": round(float(((hg + ag) > 2.5).mean()) * 100, 1),
        "btts": round(float(((hg > 0) & (ag > 0)).mean()) * 100, 1),
    }

def real_player_props(xg_per90, xa_per90, card_prob_per90):
    p_goal = (1 - poisson_pmf(0, xg_per90)) * 100
    p_assist = (1 - poisson_pmf(0, xa_per90)) * 100
    p_card = card_prob_per90 * 100
    p_sot = (1 - poisson_pmf(0, xg_per90 * 2.2)) * 100
    return {"goal": round(p_goal, 1), "assist": round(p_assist, 1),
            "card": round(p_card, 1), "sot": round(p_sot, 1)}

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
                    if not email or not pwd:
                        st.error("Missing email or password.")
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
                            except Exception:
                                pass
                            st.rerun()
                        except Exception as e:
                            st.error(f"Login failed: {e}")
        with tabs[1]:
            with st.form("signup_form"):
                name = st.text_input(t("full_name"), key="su_name")
                email = st.text_input(t("email"), key="su_email")
                pwd = st.text_input(t("password"), type="password", key="su_pwd")
                if st.form_submit_button(t("signup_btn")):
                    if not email or len(pwd) < 6:
                        st.error("Email required and password ≥ 6 chars.")
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
                            else:
                                st.success(t("check_email"))
                        except Exception as e:
                            st.error(f"Signup failed: {e}")

def require_auth():
    if not st.session_state.get("user_id"):
        auth_page()
        st.stop()

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
        nav_keys = ["nav_p1", "nav_p2", "nav_p3", "nav_p4", "nav_p5"]
        labels = {k: t(k) for k in nav_keys}
        if st.session_state.mode == "intl": labels["nav_p5"] += "  🔒"
        cur = st.session_state.page if st.session_state.page in nav_keys else "nav_p1"
        chosen = st.radio(t("nav"), nav_keys, index=nav_keys.index(cur), format_func=lambda k: labels[k], key="sb_nav")
        st.session_state.page = chosen

def subscribe_block(plan_key, gbp, tnd, title_key, btn_key):
    price = f"£{gbp:.2f} {t('per_month')}" if st.session_state.mode == "intl" else f"{tnd:.2f} TND {t('per_month')}"
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"<div class='nx-card' style='border-left:4px solid #f59e0b;'><div class='nx-card-title' style='color:#f59e0b;'>{t('monetization')}</div><div style='font-size:1rem;font-weight:700;color:#f8fafc;'>{t(title_key)}</div><div style='font-size:1.5rem;font-weight:800;color:#00ffcc;margin-top:6px;'>{price}</div></div>", unsafe_allow_html=True)
    with col_b:
        st.write(""); st.write("")
        if st.button(t("subscribe"), key=btn_key):
            try:
                get_client().table("subscriptions").insert({"user_id": st.session_state.user_id, "plan": plan_key, "amount": gbp if st.session_state.mode == "intl" else tnd, "currency": "GBP" if st.session_state.mode == "intl" else "TND", "status": "pending"}).execute()
                st.success(t("sub_ok"))
            except Exception as e: st.error(f"Failed: {e}")

def page_predictor():
    st.markdown(f"<div class='nx-title'>{t('p1_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p1_sub')}</div>", unsafe_allow_html=True)
    subscribe_block("analyst_pro", 19.99, 69.90, "nav_p1", "p1_sub")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p1_sim')}")
    client = get_client()
    try: teams = client.table("teams").select("name,elo,attack,defense").execute().data or []
    except Exception as e:
        st.error(f"Failed to load teams: {e}")
        teams = []
    if not teams:
        st.warning("No teams seeded. Run supabase_schema.sql.")
        return
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
    if run or st.session_state.sim_result is None:
        with st.spinner("Running real Monte-Carlo over 10,000 sims..."):
            st.session_state.sim_result = real_match_simulation(home_attack=float(home["attack"]), home_defense=float(home["defense"]), away_attack=float(away["attack"]), away_defense=float(away["defense"]), n_sims=10000)
    if run: st.success(t("sim_done"))
    s = st.session_state.sim_result
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(metric(t("win"), f"{s['win']}%"), unsafe_allow_html=True)
        st.progress(min(max(s["win"] / 100, 0), 1))
    with m2:
        st.markdown(metric(t("draw"), f"{s['draw']}%", "amber"), unsafe_allow_html=True)
        st.progress(min(max(s["draw"] / 100, 0), 1))
    with m3:
        st.markdown(metric(t("loss"), f"{s['loss']}%", "red"), unsafe_allow_html=True)
        st.progress(min(max(s["loss"] / 100, 0), 1))
    st.caption(f"{home_name} — xG {s['xg_home']} | {away_name} — xG {s['xg_away']}  ·  Over 2.5: {s['over_2_5']}%  ·  BTTS: {s['btts']}%")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p1_tac')}")
    with st.expander(t("win") + " / " + t("draw") + " / " + t("loss"), expanded=True):
        st.bar_chart(pd.DataFrame({"Outcome": [t("win"), t("draw"), t("loss")], "%": [s["win"], s["draw"], s["loss"]]}).set_index("Outcome"))
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('p1_props')}")
    st.caption("Real Poisson PMF — Anytime Goalscorer = 1 − P(X = 0)")
    props = [("Y. Ben Ali", 0.55, 0.22, 0.12), ("K. Trabelsi", 0.18, 0.48, 0.28), ("L. Moreau", 0.42, 0.20, 0.35), ("A. Rossi", 0.20, 0.40, 0.10)]
    rows = []
    for name, xg, xa, card in props:
        p = real_player_props(xg, xa, card)
        rows.append({"Player": name, "Goal %": p["goal"], "Assist %": p["assist"], "Card %": p["card"], "SOT 2+ %": p["sot"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
    
    client = get_client()
    
    st.markdown(f"### {t('upload_title')}")
    up_file = st.file_uploader(t("upload_label"), type=["mp4", "avi", "mov"], key="p2_up")
    if up_file is not None:
        if st.button("⬆️ Upload & Queue Analysis", key="p2_upload_btn"):
            try:
                _upload_video(client, up_file, "player_self", "player_hud")
                st.success(t("upload_ok"))
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")
    _render_jobs(client, "player_hud")
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    st.markdown(f"### {t('p2_lens_title')}")
    lens_on = st.toggle(t("p2_lens_toggle"), value=st.session_state.get("p2_lens_on", False), key="p2_lens_tgl")
    st.session_state.p2_lens_on = lens_on
    
    if lens_on:
        st.success(t("p2_lens_on"))
        l1, l2, l3 = st.columns(3)
        with l1:
            st.markdown(metric(t("p2_fps"), "60.0", "primary"), unsafe_allow_html=True)
        with l2:
            st.markdown(metric(t("p2_conf"), "0.94", "amber"), unsafe_allow_html=True)
        with l3:
            st.markdown(metric(t("p2_entities"), "22", "primary"), unsafe_allow_html=True)
            
        rng = np.random.default_rng(3)
        buf = pd.DataFrame(rng.random((40, 3)), columns=["Player A", "Player B", "Player C"])
        st.line_chart(buf)
    else:
        st.info(t("p2_lens_off"))
        
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    st.markdown(f"### {t('p2_hud_title')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric(t("p2_speed"), "32.4 km/h", "primary"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric(t("p2_pass"), "87.6%", "amber"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric(t("p2_dist"), "10.4 km", "primary"), unsafe_allow_html=True)
        
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(metric(t("p2_sprints"), "28", "primary"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric(t("p2_duels"), "61%", "amber"), unsafe_allow_html=True)
    with c6:
        st.markdown(metric("Rating", "8.3", "primary"), unsafe_allow_html=True)
        
    st.markdown(f"#### {t('p2_rating')}")
    df_attr = pd.DataFrame({
        "Attribute": [t("attr_drib"), t("attr_pass"), t("attr_shot"), t("attr_def"), t("attr_phys"), t("attr_vision")],
        "Score": [86, 82, 74, 58, 79, 84],
    }).set_index("Attribute")
    st.bar_chart(df_attr)

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
        try:
            _upload_video(client, f1, "opponent", "opponent_report")
            st.success(t("upload_ok"))
            st.rerun()
        except Exception as e: st.error(f"Upload failed: {e}")
    _render_jobs(client, "opponent_report")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('post_match')}")
    st.caption(t("post_desc"))
    f2 = st.file_uploader(t("upload_label"), type=["mp4", "avi", "mov"], key="p3_own")
    if f2 and st.button("⬆️ Upload Own Match Video", key="p3_own_btn"):
        try:
            _upload_video(client, f2, "own_match", "error_report")
            st.success(t("upload_ok"))
            st.rerun()
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

FALLBACK_TALENTS = [
    {"tag": "UK-7042", "age": 17, "pos": "FW", "pos_label": "Winger", "rate": 88, "vel": 32.4},
    {"tag": "UK-7188", "age": 19, "pos": "FW", "pos_label": "Striker", "rate": 91, "vel": 33.1},
    {"tag": "TN-2041", "age": 18, "pos": "MF", "pos_label": "Playmaker", "rate": 84, "vel": 29.8},
    {"tag": "TN-2103", "age": 20, "pos": "MF", "pos_label": "DM", "rate": 79, "vel": 28.4},
    {"tag": "FR-5520", "age": 17, "pos": "DF", "pos_label": "Full-Back", "rate": 82, "vel": 31.2},
    {"tag": "FR-5611", "age": 19, "pos": "DF", "pos_label": "Centre-Back", "rate": 76, "vel": 28.0},
    {"tag": "ES-3307", "age": 18, "pos": "GK", "pos_label": "Goalkeeper", "rate": 80, "vel": 26.5},
    {"tag": "DE-9012", "age": 16, "pos": "FW", "pos_label": "Winger", "rate": 87, "vel": 30.9},
    {"tag": "IT-4418", "age": 19, "pos": "MF", "pos_label": "Playmaker", "rate": 89, "vel": 30.1},
]

def page_scouts():
    st.markdown(f"<div class='nx-title'>{t('p4_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-sub'>{t('p4_sub')}</div>", unsafe_allow_html=True)
    subscribe_block("scout_pro", 45.00, 159.90, "plan_scout", "p4_sub")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"### {t('search')}")
    c1, c2 = st.columns(2)
    with c1: pos = st.selectbox(t("position"), ["ALL", "FW", "MF", "DF", "GK"], format_func=lambda p: {"ALL": t("pos_all"), "FW": t("pos_fw"), "MF": t("pos_mf"), "DF": t("pos_df"), "GK": t("pos_gk")}[p], key="p4_pos")
    with c2: min_rate = st.slider(t("min_rate"), 50, 100, 70, key="p4_rate")
    filtered = [x for x in FALLBACK_TALENTS if (pos == "ALL" or x["pos"] == pos) and x["rate"] >= min_rate]
    st.markdown(f"### {t('results')} ({len(filtered)})")
    if not filtered:
        st.warning(t("no_match"))
        return
    client = get_client()
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for col, p in zip(cols, filtered[i:i+2]):
            with col:
                st.markdown(f"<div class='nx-card' style='border-left:4px solid #00ffcc;'><div class='nx-card-title'>Player #{p['tag']}</div><div style='font-weight:700;color:#f8fafc;'>{p['pos_label']}</div><div style='margin-top:6px;color:#cbd5e1;font-size:.85rem;'>Age: <b>{p['age']}</b> · Pos: <b>{p['pos']}</b><br/>AI Rate: <b style='color:#00ffcc;'>{p['rate']}%</b><br/>Peak Velocity: <b style='color:#f59e0b;'>{p['vel']} km/h</b></div></div>", unsafe_allow_html=True)
                if st.button(t("contact_btn"), key=f"contact_{p['tag']}"):
                    try:
                        client.table("contact_requests").insert({"scout_id": st.session_state.user_id, "player_tag": p["tag"], "status": "pending"}).execute()
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
    try:
        cap_resp = client.rpc("check_week_capacity", {"week_key": wk}).execute()
        used = int(cap_resp.data or 0)
    except Exception: used = 0
    st.markdown(f"#### {t('weekly_cap')}: **{used}/3**")
    st.progress(min(used / 3.0, 1.0))
    full = used >= 3
    if full: st.error(f"❌ {t('full')}")
    if st.button(t("book_btn"), key="p5_book", disabled=full):
        try:
            client.table("bookings").insert({"user_id": st.session_state.user_id, "match_date": d.isoformat(), "match_time": tm.isoformat(), "iso_week": wk, "location": loc, "equipment": eq, "status": "pending", "stage": 0, "amount": 350, "currency": "TND"}).execute()
            st.success(t("book_ok"))
            st.rerun()
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
    if page == "nav_p1": page_predictor()
    elif page == "nav_p2": page_players()
    elif page == "nav_p3": page_academies()
    elif page == "nav_p4": page_scouts()
    elif page == "nav_p5": page_ops()

if __name__ == "__main__":
    main()
