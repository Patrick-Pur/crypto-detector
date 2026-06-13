from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# 1. APP-KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Market Signal Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Reduziert die native Streamlit-App-Chrome bereits beim lokalen Start.
# Die projektbezogene .streamlit/config.toml setzt dieselben Werte vor dem Rendern.
try:
    st.set_option("client.toolbarMode", "minimal")
    st.set_option("client.showSidebarNavigation", False)
except Exception:
    # Kompatibilitäts-Fallback für ältere Streamlit-Versionen.
    pass

BTC_TICKER = "BTC-USD"
BTC_DATA_START = "2015-01-01"
BTC_ANALYSIS_START = "2017-01-01"
BTC_CACHE_TTL_SECONDS = 900
BTC_COOLDOWN_DAYS = 30
BTC_HORIZONS = [30, 90, 180]
BTC_FALLBACK_FILE = Path("btc_analysis.csv")

SPCX_TICKER = "SPCX"
SPCX_ASSET_NAME = "SpaceX"
SPCX_IPO_DATE = date(2026, 6, 12)
SPCX_IPO_PRICE_USD = 135.00
SPCX_DATA_START = SPCX_IPO_DATE.isoformat()
SPCX_CACHE_TTL_SECONDS = 300
SPCX_QUOTE_CACHE_TTL_SECONDS = 60
SPCX_MIN_SIGNAL_BARS = 21
SPCX_FALLBACK_FILE = Path("spcx_hourly_regular_fallback.csv")

# Nasdaq-100 Fast Entry: manuell pflegbare Felder nach offizieller Mitteilung.
SPCX_OFFICIAL_FAST_ENTRY_ANNOUNCED = False
SPCX_OFFICIAL_ANNOUNCEMENT_DATE: date | None = None
SPCX_OFFICIAL_EFFECTIVE_DATE: date | None = None
SPCX_TOP40_FULL_MARKET_CAP_CONFIRMED: bool | None = None

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Zentrale Navigation: Neue Aktien und Kryptowährungen künftig hier ergänzen.
# Der angezeigte SpaceX-Ticker lautet SPCX. SPX wäre der S&P-500-Index.
ASSET_NAVIGATION = (
    {
        "category": "Aktien",
        "items": (
            {"page": "spacex", "label": "SpaceX", "ticker": "SPCX", "icon": "🚀"},
        ),
    },
    {
        "category": "Kryptowährungen",
        "items": (
            {"page": "bitcoin", "label": "Bitcoin", "ticker": "BTC", "icon": "₿"},
        ),
    },
)

# Erste Chatbot-Version: bewusst auf feste Beispiel-Fragen begrenzt.
# Weitere Fragen oder ein freies Texteingabefeld können später ergänzt werden,
# ohne die Datenmodelle der Asset-Seiten neu zu strukturieren.
ASSISTANT_DEFAULT_MODEL = "gpt-5.4-mini"
ASSISTANT_MAX_API_CALLS_PER_SESSION = 6
ASSISTANT_QUESTIONS = {
    "bitcoin": (
        {"id": "status", "label": "Wie ist der aktuelle BTC-Status einzuordnen?"},
        {"id": "missing", "label": "Welche Bedingungen fehlen aktuell bis zum EARLY-Signal?"},
        {"id": "scores", "label": "Wie unterscheiden sich Setup- und Bestätigungs-Score?"},
    ),
    "spacex": (
        {"id": "status", "label": "Wie ist das aktuelle SpaceX-Setup einzuordnen?"},
        {"id": "missing", "label": "Welche Kriterien fehlen für ein konstruktiveres Einstiegsfenster?"},
        {"id": "nasdaq", "label": "Wie nah ist SpaceX an einer möglichen Nasdaq-100-Aufnahme?"},
    ),
}

ASSISTANT_SYSTEM_PROMPT = """
Du bist der Market Signal Assistant eines Research-Dashboards.
Antworte auf Deutsch, präzise und verständlich. Verwende ausschließlich den bereitgestellten Dashboard-Kontext.
Erfinde keine Nachrichten, Kurse, Termine oder Indexentscheidungen. Trenne klar zwischen beobachteten Daten,
heuristischen Scores und nicht bestätigten Annahmen. Formuliere keine Anlageempfehlung und kein Kauf- oder
Verkaufssignal. Nenne bei Bedarf explizit die Grenzen der Datenbasis. Antworte in höchstens 180 Wörtern.
""".strip()

# Relevante Nasdaq-Schließtage 2026. Für spätere Jahre ergänzen.
NASDAQ_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
}


# ============================================================
# 2. VISUELLES DESIGN
# ============================================================

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --lab-bg: #07111f;
                --lab-panel: rgba(14, 28, 47, 0.78);
                --lab-panel-strong: rgba(17, 34, 57, 0.94);
                --lab-border: rgba(142, 180, 219, 0.22);
                --lab-text: #f3f7fb;
                --lab-muted: #a9b8c8;
                --lab-accent: #67e8f9;
                --lab-accent-2: #a78bfa;
                --lab-positive: #5ee6a8;
                --lab-warning: #f6c56b;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 8%, rgba(22, 78, 116, 0.35), transparent 30%),
                    radial-gradient(circle at 92% 12%, rgba(96, 66, 160, 0.26), transparent 30%),
                    linear-gradient(145deg, #06101d 0%, #081525 45%, #07111f 100%);
                color: var(--lab-text);
            }

            /* Die native Streamlit-Kopfleiste erzeugt je nach Theme einen dunklen Balken.
               Die App verwendet stattdessen eine eigene, kompakte Navigation. */
            header[data-testid="stHeader"],
            .stAppHeader {
                display: none !important;
                height: 0 !important;
                min-height: 0 !important;
                background: transparent !important;
            }

            div[data-testid="stDecoration"],
            div[data-testid="stToolbar"],
            .stAppToolbar,
            #MainMenu,
            footer {
                display: none !important;
                visibility: hidden !important;
            }

            section[data-testid="stSidebar"],
            div[data-testid="collapsedControl"] {
                display: none !important;
            }

            .block-container {
                max-width: 1500px;
                padding-top: 1.15rem;
                padding-bottom: 3rem;
                padding-left: 5.85rem;
            }

            .lab-nav-rail {
                position: fixed;
                z-index: 1001;
                top: 1rem;
                left: 0.95rem;
                display: flex;
                width: 3.75rem;
                flex-direction: column;
                align-items: center;
                gap: 0.72rem;
                padding: 0.58rem 0.48rem;
                border: 1px solid rgba(142, 180, 219, 0.19);
                border-radius: 18px;
                background: linear-gradient(160deg, rgba(17, 35, 58, 0.94), rgba(9, 22, 39, 0.94));
                box-shadow: 0 16px 35px rgba(0, 0, 0, 0.22);
                backdrop-filter: blur(18px);
            }

            .lab-nav-icon {
                display: flex;
                width: 2.62rem;
                height: 2.62rem;
                align-items: center;
                justify-content: center;
                border: 1px solid rgba(142, 180, 219, 0.14);
                border-radius: 13px;
                background: rgba(13, 29, 49, 0.72);
                color: #c7d8e8;
                text-decoration: none !important;
                transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
            }

            .lab-nav-icon:hover {
                transform: translateY(-1px);
                border-color: rgba(103, 232, 249, 0.42);
                background: rgba(23, 61, 82, 0.84);
                color: #effcff;
            }

            .lab-nav-home-svg {
                width: 1.16rem;
                height: 1.16rem;
                fill: none;
                stroke: currentColor;
                stroke-linecap: round;
                stroke-linejoin: round;
                stroke-width: 1.8;
            }

            .lab-nav-menu-lines {
                display: flex;
                width: 1.08rem;
                flex-direction: column;
                gap: 0.22rem;
            }

            .lab-nav-menu-lines span {
                display: block;
                width: 100%;
                height: 2px;
                border-radius: 99px;
                background: currentColor;
            }

            .lab-nav-drawer {
                position: fixed;
                z-index: 1000;
                top: 1rem;
                left: 5.25rem;
                width: min(20.5rem, calc(100vw - 6.25rem));
                padding: 1.02rem;
                border: 1px solid rgba(142, 180, 219, 0.20);
                border-radius: 20px;
                background: linear-gradient(150deg, rgba(18, 37, 62, 0.98), rgba(9, 22, 39, 0.98));
                box-shadow: 0 24px 55px rgba(0, 0, 0, 0.30);
                backdrop-filter: blur(22px);
            }

            .lab-nav-drawer-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                padding: 0.15rem 0.12rem 0.65rem 0.12rem;
            }

            .lab-nav-drawer-brand {
                color: #eaf8ff;
                font-size: 0.82rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .lab-nav-close {
                display: flex;
                width: 1.9rem;
                height: 1.9rem;
                align-items: center;
                justify-content: center;
                border: 1px solid rgba(142, 180, 219, 0.15);
                border-radius: 10px;
                background: rgba(13, 29, 49, 0.58);
                color: #b8cadb;
                font-size: 1.12rem;
                line-height: 1;
                text-decoration: none !important;
            }

            .lab-nav-category {
                margin: 0.76rem 0.24rem 0.34rem 0.24rem;
                color: #8ea5bb;
                font-size: 0.67rem;
                font-weight: 800;
                letter-spacing: 0.15em;
                text-transform: uppercase;
            }

            .lab-nav-item {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.6rem;
                margin-top: 0.28rem;
                padding: 0.74rem 0.78rem;
                border: 1px solid rgba(142, 180, 219, 0.12);
                border-radius: 13px;
                background: rgba(12, 27, 46, 0.62);
                color: #c5d6e5;
                font-size: 0.88rem;
                font-weight: 680;
                text-decoration: none !important;
                transition: border-color 150ms ease, background 150ms ease;
            }

            .lab-nav-item:hover,
            .lab-nav-item-active {
                border-color: rgba(103, 232, 249, 0.34);
                background: rgba(22, 59, 80, 0.70);
                color: #effcff;
            }

            .lab-nav-symbol {
                display: inline-flex;
                min-width: 2rem;
                justify-content: flex-end;
                color: #9fb5c9;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
            }

            h1, h2, h3, h4, p, span, label {
                letter-spacing: 0.01em;
            }

            .lab-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.35rem 0.72rem;
                border: 1px solid rgba(103, 232, 249, 0.26);
                border-radius: 999px;
                color: #baf5ff;
                background: rgba(8, 44, 61, 0.42);
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }

            .lab-hero {
                position: relative;
                overflow: hidden;
                padding: 3.4rem 3.2rem 3.15rem 3.2rem;
                border: 1px solid var(--lab-border);
                border-radius: 26px;
                background:
                    linear-gradient(135deg, rgba(19, 45, 74, 0.96), rgba(12, 24, 43, 0.93)),
                    radial-gradient(circle at 78% 22%, rgba(103, 232, 249, 0.28), transparent 34%);
                box-shadow: 0 26px 70px rgba(0, 0, 0, 0.24);
            }

            .lab-hero::after {
                content: "";
                position: absolute;
                right: -110px;
                top: -120px;
                width: 370px;
                height: 370px;
                border-radius: 50%;
                border: 1px solid rgba(103, 232, 249, 0.16);
                box-shadow:
                    0 0 0 38px rgba(103, 232, 249, 0.035),
                    0 0 0 80px rgba(167, 139, 250, 0.025);
            }

            .lab-hero h1 {
                max-width: 820px;
                margin: 1.2rem 0 0.85rem 0;
                color: #f8fbff;
                font-size: clamp(2.55rem, 6vw, 5.2rem);
                line-height: 0.98;
                letter-spacing: -0.065em;
            }

            .lab-hero p {
                max-width: 780px;
                margin: 0;
                color: #c2d2e2;
                font-size: 1.1rem;
                line-height: 1.75;
            }

            .lab-section-title {
                margin: 2.25rem 0 0.5rem 0;
                color: #edf7ff;
                font-size: 1.35rem;
                font-weight: 760;
                letter-spacing: -0.025em;
            }

            .lab-card {
                min-height: 100%;
                padding: 1.4rem 1.4rem 1.3rem 1.4rem;
                border: 1px solid var(--lab-border);
                border-radius: 20px;
                background: linear-gradient(145deg, rgba(17, 35, 58, 0.92), rgba(10, 23, 40, 0.86));
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.13);
            }

            .lab-card-label {
                color: #9fb3c8;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.13em;
                text-transform: uppercase;
            }

            .lab-card h3 {
                margin: 0.55rem 0 0.45rem 0;
                color: #f3f8fc;
                font-size: 1.28rem;
                letter-spacing: -0.025em;
            }

            .lab-card p {
                margin: 0;
                color: #adbdcc;
                font-size: 0.93rem;
                line-height: 1.62;
            }

            .lab-chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.48rem;
                margin-top: 1rem;
            }

            .lab-chip {
                display: inline-flex;
                padding: 0.28rem 0.58rem;
                border: 1px solid rgba(130, 165, 200, 0.21);
                border-radius: 999px;
                background: rgba(11, 24, 41, 0.66);
                color: #c9d8e7;
                font-size: 0.72rem;
            }

            .lab-disclaimer {
                margin-top: 1rem;
                padding: 1.05rem 1.15rem;
                border: 1px solid rgba(246, 197, 107, 0.26);
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(70, 51, 23, 0.39), rgba(30, 31, 38, 0.54));
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.10);
            }

            .lab-disclaimer-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1.1rem;
            }

            .lab-disclaimer strong {
                color: #ffe3a7;
                font-size: 0.84rem;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }

            .lab-disclaimer p {
                margin: 0.32rem 0 0 0;
                color: #e0d6c3;
                font-size: 0.82rem;
                line-height: 1.55;
            }

            .lab-assistant-strip {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin: 1rem 0 0.9rem 0;
                padding: 0.86rem 1rem;
                border: 1px solid rgba(103, 232, 249, 0.15);
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(15, 35, 59, 0.86), rgba(10, 24, 42, 0.78));
            }

            .lab-assistant-strip strong {
                color: #eaf8ff;
                font-size: 0.93rem;
            }

            .lab-assistant-strip span {
                color: #9fb4c9;
                font-size: 0.8rem;
            }

            .lab-assistant-answer {
                margin-top: 0.55rem;
                padding: 0.78rem 0.86rem;
                border: 1px solid rgba(142, 180, 219, 0.16);
                border-radius: 13px;
                background: rgba(8, 20, 35, 0.64);
            }

            .lab-assistant-mode {
                display: inline-flex;
                margin-bottom: 0.55rem;
                padding: 0.22rem 0.46rem;
                border: 1px solid rgba(130, 165, 200, 0.18);
                border-radius: 999px;
                color: #aac7df;
                background: rgba(15, 31, 52, 0.68);
                font-size: 0.67rem;
                font-weight: 760;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .lab-asset-header {
                margin-bottom: 1rem;
                padding: 1.35rem 1.5rem;
                border: 1px solid var(--lab-border);
                border-radius: 20px;
                background: linear-gradient(135deg, rgba(17, 36, 59, 0.92), rgba(9, 23, 40, 0.86));
            }

            .lab-asset-header h2 {
                margin: 0.35rem 0 0.25rem 0;
                color: #f6fbff;
                font-size: 2rem;
                letter-spacing: -0.045em;
            }

            .lab-asset-header p {
                margin: 0;
                color: #b5c6d6;
                line-height: 1.6;
            }

            .lab-status {
                margin: 0.8rem 0 1rem 0;
                padding: 0.95rem 1rem;
                border-radius: 15px;
                border: 1px solid rgba(145, 178, 211, 0.22);
                background: rgba(15, 31, 52, 0.76);
                color: #dce9f6;
                line-height: 1.55;
            }

            .lab-status-positive {
                border-color: rgba(94, 230, 168, 0.32);
                background: rgba(24, 74, 59, 0.38);
            }

            .lab-status-warning {
                border-color: rgba(246, 197, 107, 0.34);
                background: rgba(81, 58, 24, 0.40);
            }

            .lab-status-info {
                border-color: rgba(103, 232, 249, 0.27);
                background: rgba(17, 57, 74, 0.38);
            }

            div[data-testid="stMetric"] {
                padding: 0.78rem 0.88rem;
                border: 1px solid rgba(142, 180, 219, 0.16);
                border-radius: 15px;
                background: rgba(15, 31, 52, 0.58);
            }

            div[data-testid="stMetricLabel"] {
                color: #9eb4c9;
            }

            div[data-testid="stMetricValue"] {
                color: #f6fbff;
            }

            div[data-testid="stTabs"] button[data-baseweb="tab"] {
                height: 46px;
                padding-left: 1rem;
                padding-right: 1rem;
                border-radius: 12px 12px 0 0;
                color: #aebfd0;
                font-weight: 650;
            }

            div[data-testid="stTabs"] button[aria-selected="true"] {
                color: #e8fcff;
                background: rgba(31, 76, 104, 0.35);
            }

            /* ---------------------------------------------------------
               Native Streamlit-Chrome vollständig entfernen.
               Die breit gefassten Selektoren decken mehrere Streamlit-
               Versionen ab und beseitigen den schwarzen Balken oben.
               --------------------------------------------------------- */
            header,
            header[data-testid="stHeader"],
            [data-testid="stHeader"],
            [data-testid="stAppHeader"],
            [data-testid="stToolbar"],
            [data-testid="stAppToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stAppHeader,
            .stAppToolbar {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                min-height: 0 !important;
                max-height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            html,
            body,
            [data-testid="stApp"],
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }

            .stMainBlockContainer,
            [data-testid="stMainBlockContainer"],
            .block-container {
                width: 100% !important;
                max-width: none !important;
                margin: 0 !important;
                padding-top: 1rem !important;
                padding-right: 2rem !important;
                padding-bottom: 3rem !important;
                padding-left: 6rem !important;
            }

            /* ---------------------------------------------------------
               Linksbündige Icon-Navigation. Das details-Element öffnet
               das Menü ohne zusätzliche Streamlit-Neuberechnung.
               --------------------------------------------------------- */
            .lab-nav-rail {
                position: fixed;
                z-index: 1002;
                top: 0.9rem;
                left: 0.9rem;
                display: flex;
                width: 3.85rem;
                flex-direction: column;
                align-items: center;
                gap: 0.72rem;
                padding: 0.62rem 0.5rem;
                border: 1px solid rgba(142, 180, 219, 0.19);
                border-radius: 18px;
                background: linear-gradient(160deg, rgba(17, 35, 58, 0.96), rgba(9, 22, 39, 0.96));
                box-shadow: 0 16px 35px rgba(0, 0, 0, 0.22);
                backdrop-filter: blur(18px);
            }

            .lab-nav-details {
                position: relative;
                display: block;
            }

            .lab-nav-details > summary {
                list-style: none;
                cursor: pointer;
            }

            .lab-nav-details > summary::-webkit-details-marker {
                display: none;
            }

            .lab-nav-details[open] > summary {
                border-color: rgba(103, 232, 249, 0.50);
                background: rgba(23, 61, 82, 0.90);
                color: #effcff;
            }

            .lab-nav-drawer {
                position: fixed;
                z-index: 1001;
                top: 0.9rem;
                left: 5.45rem;
                width: min(21.5rem, calc(100vw - 6.4rem));
                padding: 1.05rem;
                border: 1px solid rgba(142, 180, 219, 0.20);
                border-radius: 20px;
                background: linear-gradient(150deg, rgba(18, 37, 62, 0.99), rgba(9, 22, 39, 0.99));
                box-shadow: 0 24px 55px rgba(0, 0, 0, 0.30);
                backdrop-filter: blur(22px);
            }

            @media (max-width: 800px) {
                .block-container {
                    padding-left: 4.95rem;
                    padding-right: 1rem;
                }
                .lab-nav-rail {
                    top: 0.72rem;
                    left: 0.58rem;
                    width: 3.5rem;
                }
                .lab-nav-drawer {
                    top: 0.72rem;
                    left: 4.6rem;
                    width: min(19rem, calc(100vw - 5.35rem));
                }
                .lab-hero {
                    padding: 2.25rem 1.45rem 2.15rem 1.45rem;
                }
                .lab-disclaimer-grid {
                    grid-template-columns: 1fr;
                }
            }

            /* ===== Landing page refresh ===== */
            .lab-home-hero {
                position: relative;
                overflow: hidden;
                margin-bottom: 1rem;
                border: 1px solid rgba(168, 190, 221, 0.24);
                border-radius: 28px;
                background:
                    radial-gradient(circle at 16% 18%, rgba(255,255,255,0.96), rgba(250,251,253,0.90) 34%, rgba(231,239,249,0.78) 74%, rgba(217,229,245,0.70) 100%),
                    linear-gradient(120deg, #fbf9f5 0%, #eef4fb 58%, #d9e7f6 100%);
                box-shadow: 0 28px 70px rgba(0,0,0,0.22);
            }
            .lab-home-hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.9fr);
                gap: 1.4rem;
                min-height: 470px;
                padding: 3.25rem 3rem;
            }
            .lab-home-copy {
                position: relative;
                z-index: 2;
                display: flex;
                flex-direction: column;
                justify-content: center;
                max-width: 690px;
            }
            .lab-home-copy h1 {
                margin: 1.05rem 0 0.95rem;
                color: #142347;
                font-size: clamp(3rem, 5.8vw, 5.1rem);
                line-height: 0.94;
                letter-spacing: -0.072em;
            }
            .lab-home-copy p {
                max-width: 650px;
                margin: 0;
                color: #55647a;
                font-size: 1.06rem;
                line-height: 1.72;
            }
            .lab-home-cta-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.78rem;
                margin-top: 1.45rem;
            }
            .lab-primary-cta, .lab-secondary-cta {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.78rem 1.15rem;
                border-radius: 999px;
                text-decoration: none !important;
                font-size: 0.92rem;
                font-weight: 760;
                transition: transform .16s ease, box-shadow .16s ease;
            }
            .lab-primary-cta {
                color: #f7fbff !important;
                background: linear-gradient(135deg,#2859f6,#6a93ff);
                box-shadow: 0 14px 28px rgba(48,91,244,.25);
            }
            .lab-secondary-cta {
                color: #21345c !important;
                border: 1px solid rgba(40,68,119,.14);
                background: rgba(255,255,255,.74);
                box-shadow: 0 12px 24px rgba(15,31,60,.08);
            }
            .lab-primary-cta:hover, .lab-secondary-cta:hover { transform: translateY(-1px); }
            .lab-kicker {
                color: #26406d;
                border-color: rgba(59,91,154,.16);
                background: rgba(255,255,255,.68);
                box-shadow: 0 10px 26px rgba(15,31,60,.06);
            }
            .lab-home-visual {
                position: relative;
                overflow: hidden;
                min-height: 100%;
                border-radius: 24px;
                background:
                    radial-gradient(circle at 70% 30%, rgba(68,177,255,.34), transparent 22%),
                    radial-gradient(circle at 58% 52%, rgba(105,84,255,.26), transparent 30%),
                    linear-gradient(145deg,#17294c 0%,#0a172a 72%,#081220 100%);
                box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 18px 48px rgba(10,19,37,.24);
            }
            .lab-home-visual::before {
                content: "";
                position: absolute;
                right: -70px;
                top: -75px;
                width: 360px;
                height: 360px;
                border-radius: 50%;
                border: 1px solid rgba(108,185,255,.22);
                box-shadow: 0 0 0 34px rgba(112,192,255,.055),0 0 0 76px rgba(112,192,255,.032);
            }
            .lab-home-wave, .lab-home-wave-2 {
                position: absolute;
                right: -6%;
                top: 20%;
                width: 96%;
                height: 52%;
                border: 2px solid rgba(101,211,255,.36);
                border-radius: 50%;
                transform: rotate(-17deg);
                filter: drop-shadow(0 0 18px rgba(86,198,255,.16));
            }
            .lab-home-wave-2 {
                right: -1%;
                top: 29%;
                width: 86%;
                height: 44%;
                border-color: rgba(157,184,255,.24);
            }
            .lab-home-visual-card {
                position: absolute;
                left: 1.2rem;
                bottom: 1.2rem;
                z-index: 2;
                width: min(250px, calc(100% - 2.4rem));
                padding: 1rem;
                border: 1px solid rgba(154,189,226,.16);
                border-radius: 16px;
                background: rgba(13,27,48,.72);
                backdrop-filter: blur(16px);
            }
            .lab-home-visual-card-label { color:#91a9c8; font-size:.7rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
            .lab-home-visual-card-value { margin-top:.4rem; color:#f2f7ff; font-size:1.65rem; font-weight:780; letter-spacing:-.04em; }
            .lab-home-visual-card-text { margin-top:.34rem; color:#b7c7d9; font-size:.8rem; line-height:1.5; }
            .lab-disclaimer-card {
                padding: 1rem 1.08rem;
                border: 1px solid rgba(246,197,107,.18);
                border-radius: 18px;
                background: linear-gradient(135deg,rgba(50,43,24,.42),rgba(26,29,38,.52));
            }
            .lab-category-card {
                min-height:100%;
                padding:1.35rem;
                border:1px solid rgba(146,180,221,.18);
                border-radius:20px;
                background:linear-gradient(145deg,rgba(13,27,48,.86),rgba(9,20,35,.82));
            }
            .lab-category-card ul { margin:.9rem 0 0; padding:0; list-style:none; }
            .lab-category-card li { display:flex; align-items:center; justify-content:space-between; gap:.8rem; padding:.8rem 0; border-top:1px solid rgba(143,175,213,.12); color:#dce8f5; font-size:.92rem; }
            .lab-category-card li:first-child { border-top:none; padding-top:0; }
            .lab-category-meta { color:#8fa6bd; font-size:.74rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
            @media (max-width: 800px) {
                .lab-home-hero-grid { grid-template-columns:1fr; min-height:auto; padding:2rem 1.35rem; }
                .lab-home-copy h1 { font-size:clamp(2.45rem,12vw,4rem); }
                .lab-home-visual { min-height:300px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()


# ============================================================
# 3. GEMEINSAME HILFSFUNKTIONEN
# ============================================================

def normalize_market_columns(data: pd.DataFrame, force_utc: bool) -> pd.DataFrame:
    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(column).strip() for column in data.columns]
    data.index = pd.to_datetime(data.index)

    if force_utc:
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")
    elif getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    data = data[~data.index.duplicated(keep="last")]
    return data.sort_index()


def format_percent(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{decimals}f} %"


def format_usd(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f} USD"


def format_large_usd(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} Mrd. USD"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} Mio. USD"
    return f"{value:,.0f} USD"


def format_bool(value: bool | None) -> str:
    if value is True:
        return "✅ Erfüllt"
    if value is False:
        return "❌ Nicht erfüllt"
    return "⚪ Noch nicht belastbar prüfbar"


def render_status_box(title: str, description: str, level: str = "info") -> None:
    safe_level = level if level in {"positive", "warning", "info"} else "info"
    st.markdown(
        f"""
        <div class="lab-status lab-status-{safe_level}">
            <strong>{title}</strong><br>{description}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_check(condition: bool) -> str:
    return "✅ Erfüllt" if bool(condition) else "❌ Fehlt"


def get_optional_secret(name: str, default: str | None = None) -> str | None:
    """Liest ein optionales Streamlit-Secret, ohne lokale Starts ohne secrets.toml zu blockieren."""

    try:
        value = st.secrets[name]
    except Exception:
        return default

    return str(value) if value is not None else default


def assistant_openai_available() -> bool:
    return OpenAI is not None and bool(get_optional_secret("OPENAI_API_KEY"))


def build_assistant_cache_key(asset: str, question_id: str, context: dict) -> str:
    serialized = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"{asset}:{question_id}:{digest}"


def build_btc_assistant_context(latest: pd.Series, analysis: pd.DataFrame, latest_date: date) -> dict:
    price = float(latest["Close"])
    rsi = float(latest["RSI"])
    drawdown = float(latest["DRAWDOWN"])
    vol_ratio = float(latest["VOL_RATIO"])
    macd_above_signal = bool(float(latest["MACD"]) > float(latest["MACD_SIGNAL"]))
    price_above_ema20 = bool(price > float(latest["EMA20"]))
    rsi_rising = bool(rsi > float(analysis.iloc[-4]["RSI"]))
    watch_live = bool(latest["WATCH_SIGNAL"])
    early_live = bool(latest["EARLY_SIGNAL"])

    if early_live:
        status = "EARLY"
    elif watch_live:
        status = "WATCH"
    else:
        status = "NEUTRAL"

    return {
        "asset": "Bitcoin",
        "ticker": BTC_TICKER,
        "data_frequency": "vollständige Tageskerzen",
        "latest_date": latest_date.isoformat(),
        "price_usd": round(price, 2),
        "status": status,
        "watch_signal_active": watch_live,
        "early_signal_active": early_live,
        "rsi14": round(rsi, 2),
        "drawdown_from_ath_pct": round(drawdown, 2),
        "volume_ratio": round(vol_ratio, 3),
        "setup_score": int(latest["SETUP_SCORE"]),
        "confirmation_score": int(latest["CONFIRMATION_SCORE"]),
        "conditions": {
            "drawdown_below_minus_40_pct": drawdown < -40,
            "rsi14_below_30": rsi < 30,
            "macd_above_signal": macd_above_signal,
            "volume_at_least_1_3x": vol_ratio >= 1.3,
            "price_above_ema20": price_above_ema20,
            "rsi_rising_vs_3_days_ago": rsi_rising,
        },
        "methodology_note": "WATCH benötigt Drawdown unter -40 % und RSI14 unter 30. EARLY benötigt zusätzlich MACD über Signallinie.",
    }


def build_spcx_assistant_context(
    latest: pd.Series,
    timeline: dict[str, date],
    completed_trade_days: int,
    advt_proxy: float | None,
    advt_completed_days: int,
    market_phase: str,
    current_index_timing_score: int,
    latest_tactical_score: int,
) -> dict:
    rsi_constructive = bool(pd.notna(latest["RSI7"]) and 45 <= float(latest["RSI7"]) <= 68)
    up_volume_constructive = bool(
        pd.notna(latest["UP_DOWN_VOLUME_RATIO"]) and float(latest["UP_DOWN_VOLUME_RATIO"]) >= 1.1
    )

    return {
        "asset": "SpaceX",
        "ticker": SPCX_TICKER,
        "data_frequency": "vollständige reguläre US-Stundenkerzen",
        "last_regular_close_usd": round(float(latest["Close"]), 2),
        "ipo_price_usd": SPCX_IPO_PRICE_USD,
        "ipo_premium_pct": round(float(latest["IPO_PREMIUM"]), 2),
        "drawdown_from_post_ipo_high_pct": round(float(latest["DRAWDOWN_FROM_HIGH"]), 2),
        "distance_to_ipo_avwap_pct": round(float(latest["DISTANCE_TO_AVWAP"]), 2),
        "market_phase": market_phase,
        "hype_risk_score": int(latest["HYPE_RISK_SCORE"]),
        "entry_quality_score": int(latest["ENTRY_QUALITY_SCORE"]),
        "index_timing_score": current_index_timing_score,
        "tactical_score": latest_tactical_score,
        "tactical_entry_watch": bool(latest["TACTICAL_ENTRY_WATCH"]),
        "conditions": {
            "moderate_pullback_from_post_ipo_high": -25 <= float(latest["DRAWDOWN_FROM_HIGH"]) <= -8,
            "price_at_or_above_ipo_price": float(latest["Close"]) >= SPCX_IPO_PRICE_USD,
            "price_above_ipo_anchored_vwap": bool(latest["ABOVE_AVWAP"]),
            "ema9_above_ema21": bool(latest["EMA_BULL_STRUCTURE"]),
            "short_term_higher_low": bool(latest["HIGHER_LOW"]),
            "volatility_cooling": bool(latest["VOLATILITY_COOLING"]),
            "rsi7_constructive_range_45_to_68": rsi_constructive,
            "up_volume_exceeds_down_volume": up_volume_constructive,
        },
        "nasdaq_100_fast_entry": {
            "official_fast_entry_announced": SPCX_OFFICIAL_FAST_ENTRY_ANNOUNCED,
            "top40_full_market_cap_confirmed": SPCX_TOP40_FULL_MARKET_CAP_CONFIRMED,
            "completed_trading_days": completed_trade_days,
            "reference_day_7": timeline["reference"].isoformat(),
            "possible_announcement_day_10": timeline["expected_announcement"].isoformat(),
            "possible_effective_day_15": timeline["expected_effective"].isoformat(),
            "advt_proxy_usd": None if advt_proxy is None else round(float(advt_proxy), 2),
            "advt_completed_days": advt_completed_days,
            "advt_threshold_met": None if advt_proxy is None else bool(advt_proxy >= 5_000_000),
        },
        "methodology_note": "Die Nasdaq-100-Termine sind methodikbasierte Erwartungstermine, solange keine offizielle Mitteilung hinterlegt ist.",
    }


def local_assistant_answer(asset: str, question_id: str, context: dict) -> str:
    """Deterministische Vorschau-Antworten, damit der Chatbot auch ohne API-Key live funktioniert."""

    if asset == "bitcoin":
        conditions = context["conditions"]
        if question_id == "status":
            return (
                f"Der BTC-Status lautet aktuell **{context['status']}**. Der letzte vollständig ausgewertete Schlusskurs liegt bei "
                f"**{context['price_usd']:,.0f} USD**. Der RSI14 beträgt **{context['rsi14']:.1f}**, der Drawdown vom bisherigen ATH "
                f"**{context['drawdown_from_ath_pct']:.1f} %**. Der Setup-Score liegt bei **{context['setup_score']}/100**, "
                f"der Bestätigungs-Score bei **{context['confirmation_score']}/100**. "
                "Das ist eine heuristische Research-Einordnung und kein automatisches Kaufsignal."
            )
        if question_id == "missing":
            required = [
                ("Drawdown unter -40 %", conditions["drawdown_below_minus_40_pct"]),
                ("RSI14 unter 30", conditions["rsi14_below_30"]),
                ("MACD über Signallinie", conditions["macd_above_signal"]),
            ]
            missing = [label for label, fulfilled in required if not fulfilled]
            if not missing:
                return "Die drei Pflichtbedingungen für ein EARLY-Signal sind aktuell erfüllt. Das Modell zeigt damit eine bullische Frühbestätigung, aber keine Anlageempfehlung."
            return "Für ein EARLY-Signal fehlen aktuell noch: **" + ", ".join(missing) + "**. Zusatzsignale wie Volumen, EMA20 und steigender RSI helfen bei der Einordnung, sind aber keine Pflichtbedingungen."
        return (
            f"Der **Setup-Score** von **{context['setup_score']}/100** misst die Intensität der Stress- und Bodenbildungsphase, insbesondere Drawdown, RSI und Volumen. "
            f"Der **Bestätigungs-Score** von **{context['confirmation_score']}/100** prüft, ob Momentum und kurzfristige Struktur bereits nach oben drehen, unter anderem über MACD, EMA20 und RSI-Verlauf. "
            "Beide Scores beantworten daher unterschiedliche Fragen und sollten nicht isoliert gelesen werden."
        )

    conditions = context["conditions"]
    ndx = context["nasdaq_100_fast_entry"]
    if question_id == "status":
        return (
            f"Das SpaceX-Modell ordnet die aktuelle Phase als **{context['market_phase']}** ein. Der letzte reguläre Schlusskurs liegt bei "
            f"**{context['last_regular_close_usd']:,.2f} USD**. Das Hype-Risiko beträgt **{context['hype_risk_score']}/100**, die Einstiegsqualität "
            f"**{context['entry_quality_score']}/100** und der taktische Score **{context['tactical_score']}/100**. "
            "Der taktische Score kombiniert Kursstruktur und Index-Timing, ersetzt aber keine eigenständige Risikoprüfung."
        )
    if question_id == "missing":
        labels = {
            "moderate_pullback_from_post_ipo_high": "moderater Rücksetzer vom Post-IPO-Hoch",
            "price_at_or_above_ipo_price": "Kurs mindestens auf IPO-Niveau",
            "price_above_ipo_anchored_vwap": "Kurs oberhalb des IPO-anchored VWAP",
            "ema9_above_ema21": "EMA9 oberhalb EMA21",
            "short_term_higher_low": "kurzfristig höheres Tief",
            "volatility_cooling": "abkühlende Volatilität",
            "rsi7_constructive_range_45_to_68": "RSI7 im konstruktiven Bereich",
            "up_volume_exceeds_down_volume": "überwiegendes Aufwärtsvolumen",
        }
        missing = [labels[key] for key, fulfilled in conditions.items() if not fulfilled]
        if not missing:
            return "Alle aktuell modellierten Strukturkriterien sind erfüllt. Das erhöht die Einstiegsqualität im Modell, ist aber weiterhin kein automatisches Kaufsignal."
        return "Für ein konstruktiveres Einstiegsfenster fehlen aktuell noch: **" + ", ".join(missing) + "**. Die Kriterien sind transparente heuristische Startwerte und noch nicht statistisch kalibriert."
    official = "ja" if ndx["official_fast_entry_announced"] else "nein"
    top40 = "bestätigt" if ndx["top40_full_market_cap_confirmed"] is True else "noch nicht bestätigt"
    advt = "noch nicht belastbar" if ndx["advt_proxy_usd"] is None else f"ca. {ndx['advt_proxy_usd']:,.0f} USD"
    return (
        f"Aktuell sind **{ndx['completed_trading_days']}** Handelstage seit dem IPO abgeschlossen. Der methodikbasierte 7. Handelstag ist "
        f"**{ndx['reference_day_7']}**, eine mögliche Ankündigung nach dem 10. Handelstag **{ndx['possible_announcement_day_10']}** und eine mögliche Aufnahme nach 15 Handelstagen "
        f"**{ndx['possible_effective_day_15']}**. Offizielle Fast-Entry-Ankündigung hinterlegt: **{official}**. "
        f"Top-40-Full-Market-Cap: **{top40}**. ADVT-Näherung: **{advt}**. Solange Nasdaq keine Mitteilung veröffentlicht hat, bleibt die Aufnahme offen."
    )


def openai_assistant_answer(asset: str, question_label: str, context: dict, fallback_answer: str) -> tuple[str, str]:
    api_key = get_optional_secret("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return fallback_answer, "Preview-Modus ohne API"

    model = get_optional_secret("OPENAI_MODEL", ASSISTANT_DEFAULT_MODEL) or ASSISTANT_DEFAULT_MODEL
    prompt_payload = {
        "asset": asset,
        "selected_question": question_label,
        "dashboard_context": context,
        "deterministic_reference_answer": fallback_answer,
    }

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=ASSISTANT_SYSTEM_PROMPT,
            input=json.dumps(prompt_payload, ensure_ascii=False, default=str),
            max_output_tokens=450,
            store=False,
        )
        answer = str(response.output_text).strip()
        return (answer or fallback_answer), f"KI-Modus · {model}"
    except Exception:
        return fallback_answer, "Fallback nach API-Fehler"


def render_market_assistant(asset: str, context: dict) -> None:
    """Rendert eine begrenzte, erweiterbare Chatbot-Vorschau mit drei Fragen je Asset."""

    questions = ASSISTANT_QUESTIONS[asset]
    state_key = f"assistant_last_answer_{asset}"
    cache_key = "assistant_response_cache"
    count_key = "assistant_api_call_count"

    st.session_state.setdefault(cache_key, {})
    st.session_state.setdefault(count_key, 0)

    st.markdown(
        """
        <div class="lab-assistant-strip">
            <div>
                <strong>💬 Market Signal Assistant</strong><br>
                <span>MVP mit drei kontextbezogenen Fragen. Die Architektur bleibt für freie Texteingaben erweiterbar.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if hasattr(st, "popover"):
        assistant_container = st.popover("💬 Assistent öffnen", use_container_width=True)
    else:
        assistant_container = st.expander("💬 Assistent öffnen")

    with assistant_container:
        st.caption("Wähle eine exemplarische Frage. Die Antwort berücksichtigt die aktuell angezeigten Dashboard-Werte.")

        for question in questions:
            if st.button(question["label"], key=f"assistant_{asset}_{question['id']}", use_container_width=True):
                cache_identifier = build_assistant_cache_key(asset, question["id"], context)
                cached = st.session_state[cache_key].get(cache_identifier)

                if cached is not None:
                    st.session_state[state_key] = cached
                    continue

                fallback_answer = local_assistant_answer(asset, question["id"], context)
                can_call_api = assistant_openai_available() and st.session_state[count_key] < ASSISTANT_MAX_API_CALLS_PER_SESSION

                if can_call_api:
                    with st.spinner("KI-Antwort wird erstellt ..."):
                        answer, mode = openai_assistant_answer(asset, question["label"], context, fallback_answer)
                    st.session_state[count_key] += 1
                elif assistant_openai_available():
                    answer, mode = fallback_answer, "Preview-Modus · Sitzungslimit erreicht"
                else:
                    answer, mode = fallback_answer, "Preview-Modus ohne API"

                response_payload = {
                    "question": question["label"],
                    "answer": answer,
                    "mode": mode,
                }
                st.session_state[cache_key][cache_identifier] = response_payload
                st.session_state[state_key] = response_payload

        latest_response = st.session_state.get(state_key)
        if latest_response:
            st.markdown(f'<span class="lab-assistant-mode">{latest_response["mode"]}</span>', unsafe_allow_html=True)
            st.markdown(f"**{latest_response['question']}**")
            st.markdown(latest_response["answer"])

        if st.button("Antwort zurücksetzen", key=f"assistant_reset_{asset}", use_container_width=True):
            st.session_state.pop(state_key, None)


# ============================================================
# 4. STARTSEITE
# ============================================================

def render_landing_page() -> None:
    bitcoin_href = build_navigation_href("bitcoin")
    spacex_href = build_navigation_href("spacex")

    st.markdown(
        f"""
        <section class="lab-home-hero">
            <div class="lab-home-hero-grid">
                <div class="lab-home-copy">
                    <span class="lab-kicker">◈ Market Signal Lab</span>
                    <h1>Smarter signal<br>selection for<br>markets in motion.</h1>
                    <p>
                        Ein fokussiertes Research-Dashboard für zwei klar getrennte Marktregime:
                        zyklische Bitcoin-Bodenbildung und die emotionale Preisfindung nach dem SpaceX-IPO.
                        Jedes Asset nutzt ein eigenes, marktspezifisches Modell statt einer pauschalen Formel.
                    </p>
                    <div class="lab-home-cta-row">
                        <a class="lab-primary-cta" href="{bitcoin_href}" target="_self">Bitcoin Dashboard öffnen</a>
                        <a class="lab-secondary-cta" href="{spacex_href}" target="_self">SpaceX Watch öffnen</a>
                    </div>
                </div>
                <div class="lab-home-visual" aria-hidden="true">
                    <div class="lab-home-wave"></div>
                    <div class="lab-home-wave-2"></div>
                    <div class="lab-home-visual-card">
                        <div class="lab-home-visual-card-label">Current setup</div>
                        <div class="lab-home-visual-card-value">2 Assets</div>
                        <div class="lab-home-visual-card-text">Heute mit Bitcoin und SpaceX – vorbereitet für weitere Aktien, Kryptowährungen und zusätzliche Signalmodelle.</div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="lab-disclaimer">
            <div class="lab-disclaimer-grid">
                <div class="lab-disclaimer-card">
                    <strong>⚠ Datenabruf</strong>
                    <p>Die Marktdaten werden über Yahoo Finance und das Open-Source-Paket yfinance geladen. Daten können verzögert, unvollständig oder temporär nicht verfügbar sein. Das Dashboard ist kein professioneller Echtzeit-Marktdatenfeed.</p>
                </div>
                <div class="lab-disclaimer-card">
                    <strong>⚠ Research only</strong>
                    <p>Die dargestellten Scores, Signale und Szenarien dienen ausschließlich der eigenen Analyse. Sie sind keine Anlageempfehlung, kein automatisches Handelssignal und kein Ersatz für eine eigenständige Prüfung der zugrunde liegenden Risiken.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lab-section-title">Asset-Kategorien</div>', unsafe_allow_html=True)
    equities_col, crypto_col = st.columns(2)

    with equities_col:
        st.markdown(
            f"""
            <div class="lab-category-card">
                <div class="lab-card-label">Kategorie</div>
                <h3>Aktien</h3>
                <p>Einzelaktien mit eigenen, marktspezifischen Signalmodellen – aktuell mit Fokus auf IPO- und Event-getriebene Setups.</p>
                <ul><li><span>🚀 SpaceX</span><span class="lab-category-meta">SPCX</span></li></ul>
                <div class="lab-chip-row"><a class="lab-secondary-cta" href="{spacex_href}" target="_self">Zum SpaceX-Reiter</a></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with crypto_col:
        st.markdown(
            f"""
            <div class="lab-category-card">
                <div class="lab-card-label">Kategorie</div>
                <h3>Kryptowährungen</h3>
                <p>Liquidere 24/7-Märkte mit stärker zyklischem Charakter – aktuell mit einem reversalen Research-Modell für Bitcoin.</p>
                <ul><li><span>₿ Bitcoin</span><span class="lab-category-meta">BTC</span></li></ul>
                <div class="lab-chip-row"><a class="lab-secondary-cta" href="{bitcoin_href}" target="_self">Zum Bitcoin-Reiter</a></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="lab-section-title">Modelle im Überblick</div>', unsafe_allow_html=True)
    bitcoin_card, spacex_card = st.columns(2)

    with bitcoin_card:
        st.markdown(
            """
            <div class="lab-card">
                <div class="lab-card-label">₿ Bitcoin · Tagesdaten</div>
                <h3>Bullish Reversal Monitor</h3>
                <p>Beobachtet Stressphasen nach starken Drawdowns und prüft, ob sich Momentum und kurzfristige Marktstruktur wieder verbessern.</p>
                <div class="lab-chip-row"><span class="lab-chip">RSI14</span><span class="lab-chip">MACD</span><span class="lab-chip">Drawdown vom ATH</span><span class="lab-chip">30 / 90 / 180 Tage</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with spacex_card:
        st.markdown(
            """
            <div class="lab-card">
                <div class="lab-card-label">🚀 SpaceX · Stundenkerzen</div>
                <h3>IPO Cooldown & Nasdaq-100 Watch</h3>
                <p>Bewertet, ob sich die erste IPO-Euphorie ausreichend abbaut und eine konstruktive Stabilisierung entsteht. Das Nasdaq-100-Fast-Entry-Fenster wird separat berücksichtigt.</p>
                <div class="lab-chip-row"><span class="lab-chip">IPO-anchored VWAP</span><span class="lab-chip">EMA9 / EMA21</span><span class="lab-chip">Hype-Risiko</span><span class="lab-chip">Nasdaq-100 Timing</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="lab-section-title">Modellprinzipien</div>', unsafe_allow_html=True)
    principle_1, principle_2, principle_3 = st.columns(3)
    principle_cards = [
        (principle_1, "01 · Marktgerecht", "Keine universelle Signalformel", "Bitcoin und eine frisch gelistete Einzelaktie werden mit getrennten, zum jeweiligen Marktregime passenden Logiken analysiert."),
        (principle_2, "02 · Transparent", "Erklärbare Heuristiken", "Jeder Score setzt sich aus nachvollziehbaren Einzelbedingungen zusammen. Die Checklisten zeigen sichtbar, welche Kriterien erfüllt sind."),
        (principle_3, "03 · Erweiterbar", "Saubere Basis für Iterationen", "Neue Datenquellen, zusätzliche Signale und eine spätere statistische Kalibrierung können modular ergänzt werden."),
    ]
    for column, label, heading, description in principle_cards:
        with column:
            st.markdown(f'<div class="lab-card"><div class="lab-card-label">{label}</div><h3>{heading}</h3><p>{description}</p></div>', unsafe_allow_html=True)


# ============================================================
# 5. BITCOIN: DATEN UND INDIKATOREN
# ============================================================

def load_btc_fallback_data() -> pd.DataFrame | None:
    if not BTC_FALLBACK_FILE.exists():
        return None

    fallback = pd.read_csv(BTC_FALLBACK_FILE, parse_dates=["Date"], index_col="Date")
    return normalize_market_columns(fallback, force_utc=False)


@st.cache_data(ttl=BTC_CACHE_TTL_SECONDS, show_spinner=False)
def load_btc_market_data() -> tuple[pd.DataFrame, str, str | None]:
    source = "Yahoo Finance via yfinance"
    fallback_reason = None

    try:
        market_data = yf.download(
            BTC_TICKER,
            start=BTC_DATA_START,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        market_data = normalize_market_columns(market_data, force_utc=False)

        if market_data.empty:
            raise RuntimeError("Yahoo Finance hat keine BTC-Daten geliefert.")

    except Exception as error:
        market_data = load_btc_fallback_data()
        if market_data is None:
            raise RuntimeError(
                "Die aktuellen BTC-Daten konnten nicht geladen werden und es ist keine lokale Fallback-Datei vorhanden."
            ) from error
        source = f"Fallback-Datei {BTC_FALLBACK_FILE.name}"
        fallback_reason = str(error)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in market_data.columns]

    if missing_columns:
        raise RuntimeError("Folgende BTC-Kursdaten-Spalten fehlen: " + ", ".join(missing_columns))

    return market_data[required_columns].copy(), source, fallback_reason


def add_btc_indicators(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()

    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA200"] = data["Close"].ewm(span=200, adjust=False).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    relative_strength = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + relative_strength))

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_BULL_CROSS"] = (
        (data["MACD"] > data["MACD_SIGNAL"])
        & (data["MACD"].shift(1) <= data["MACD_SIGNAL"].shift(1))
    )
    data["MACD_BULL_CROSS_RECENT_7D"] = (
        data["MACD_BULL_CROSS"].rolling(7, min_periods=1).max().astype(bool)
    )

    data["VOL_MA20"] = data["Volume"].rolling(20).mean()
    data["VOL_RATIO"] = data["Volume"] / data["VOL_MA20"]
    data["ATH"] = data["Close"].cummax()
    data["DRAWDOWN"] = (data["Close"] / data["ATH"] - 1) * 100

    data["WATCH_SIGNAL"] = (data["RSI"] < 30) & (data["DRAWDOWN"] < -40)
    data["EARLY_SIGNAL"] = data["WATCH_SIGNAL"] & (data["MACD"] > data["MACD_SIGNAL"])

    drawdown_points = np.select(
        [
            data["DRAWDOWN"] <= -60,
            data["DRAWDOWN"] <= -50,
            data["DRAWDOWN"] <= -40,
            data["DRAWDOWN"] <= -30,
        ],
        [45, 35, 25, 10],
        default=0,
    )
    rsi_points = np.select(
        [data["RSI"] < 25, data["RSI"] < 30, data["RSI"] < 35, data["RSI"] < 45],
        [35, 30, 20, 10],
        default=0,
    )
    volume_points = np.select(
        [data["VOL_RATIO"] >= 1.5, data["VOL_RATIO"] >= 1.3, data["VOL_RATIO"] >= 1.0],
        [20, 15, 8],
        default=0,
    )
    data["SETUP_SCORE"] = np.minimum(100, drawdown_points + rsi_points + volume_points).astype(int)

    confirmation_score = np.zeros(len(data), dtype=int)
    confirmation_score += np.where(data["MACD"] > data["MACD_SIGNAL"], 40, 0)
    confirmation_score += np.where(data["MACD_BULL_CROSS_RECENT_7D"], 20, 0)
    confirmation_score += np.where(data["Close"] > data["EMA20"], 15, 0)
    confirmation_score += np.where(data["EMA20"] > data["EMA20"].shift(5), 10, 0)
    confirmation_score += np.where(data["RSI"] > data["RSI"].shift(3), 10, 0)
    confirmation_score += np.where(data["Close"] > data["Close"].shift(3), 5, 0)
    data["CONFIRMATION_SCORE"] = np.minimum(100, confirmation_score).astype(int)

    required_columns = [
        "Open", "High", "Low", "Close", "Volume", "EMA20", "EMA50", "EMA200", "RSI",
        "MACD", "MACD_SIGNAL", "VOL_MA20", "VOL_RATIO", "DRAWDOWN", "SETUP_SCORE",
        "CONFIRMATION_SCORE",
    ]
    data = data.dropna(subset=required_columns)

    today_utc = pd.Timestamp.now(tz="UTC").date()
    if not data.empty and data.index[-1].date() >= today_utc:
        data = data.iloc[:-1].copy()

    return data.loc[BTC_ANALYSIS_START:].copy()


def select_signal_phases(data: pd.DataFrame, signal_column: str, cooldown_days: int) -> pd.DataFrame:
    signal_mask = data[signal_column].fillna(False)
    phase_starts = signal_mask & ~signal_mask.shift(1, fill_value=False)
    candidate_dates = list(data.index[phase_starts])
    selected_dates: list[pd.Timestamp] = []

    for event_date in candidate_dates:
        if not selected_dates or (event_date - selected_dates[-1]).days >= cooldown_days:
            selected_dates.append(event_date)

    return data.loc[selected_dates].copy()


def calculate_btc_event_table(data: pd.DataFrame, signal_phases: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for event_date, row in signal_phases.iterrows():
        position = int(data.index.get_loc(event_date))
        entry_price = float(data.iloc[position]["Close"])
        record: dict[str, object] = {
            "DATUM": event_date,
            "BTC_PREIS": entry_price,
            "DRAWDOWN": float(row["DRAWDOWN"]),
            "RSI": float(row["RSI"]),
            "VOL_RATIO": float(row["VOL_RATIO"]),
            "SETUP_SCORE": int(row["SETUP_SCORE"]),
            "CONFIRMATION_SCORE": int(row["CONFIRMATION_SCORE"]),
        }

        for horizon in BTC_HORIZONS:
            if position + horizon >= len(data):
                record[f"RET_{horizon}D"] = None
                record[f"MAE_{horizon}D"] = None
                record[f"MFE_{horizon}D"] = None
                continue

            future_price = float(data.iloc[position + horizon]["Close"])
            window = data.iloc[position : position + horizon + 1]["Close"]
            record[f"RET_{horizon}D"] = (future_price / entry_price - 1) * 100
            record[f"MAE_{horizon}D"] = (float(window.min()) / entry_price - 1) * 100
            record[f"MFE_{horizon}D"] = (float(window.max()) / entry_price - 1) * 100

        records.append(record)

    return pd.DataFrame(records)


def create_btc_summary_table(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for horizon in BTC_HORIZONS:
        column = f"RET_{horizon}D"
        if events.empty or column not in events.columns:
            continue

        values = pd.to_numeric(events[column], errors="coerce").dropna()
        if values.empty:
            continue

        wins = values[values > 0]
        losses = values[values <= 0]
        profit_factor = np.nan
        if not wins.empty and not losses.empty:
            profit_factor = wins.sum() / abs(losses.sum())

        records.append(
            {
                "Horizont": f"{horizon} Tage",
                "Ereignisse": len(values),
                "Trefferquote": len(wins) / len(values) * 100,
                "Ø Rendite": values.mean(),
                "Median": values.median(),
                "Beste Rendite": values.max(),
                "Schlechteste Rendite": values.min(),
                "Profit Factor": profit_factor,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 6. BITCOIN: CHARTS UND UI
# ============================================================

def create_btc_price_chart(
    chart_data: pd.DataFrame,
    visible_watch_phases: pd.DataFrame,
    visible_early_phases: pd.DataFrame,
    log_scale: bool,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="BTC",
        )
    )

    for column in ["EMA20", "EMA50", "EMA200"]:
        figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data[column], mode="lines", name=column))

    if not visible_watch_phases.empty:
        figure.add_trace(
            go.Scatter(
                x=visible_watch_phases.index,
                y=visible_watch_phases["Close"],
                mode="markers",
                name="WATCH",
                marker={"symbol": "circle", "size": 10},
            )
        )

    if not visible_early_phases.empty:
        figure.add_trace(
            go.Scatter(
                x=visible_early_phases.index,
                y=visible_early_phases["Close"],
                mode="markers",
                name="EARLY",
                marker={"symbol": "triangle-up", "size": 14},
            )
        )

    latest_date = chart_data.index[-1]
    latest_close = float(chart_data.iloc[-1]["Close"])
    figure.add_trace(
        go.Scatter(
            x=[latest_date], y=[latest_close], mode="markers+text", name="Aktueller Stand",
            text=["Aktuell"], textposition="top center", marker={"symbol": "diamond", "size": 12},
        )
    )
    figure.add_vline(x=latest_date, line_dash="dot")
    figure.update_layout(
        title="BTC-Kurs mit EMA-Linien, WATCH- und EARLY-Signalen",
        height=650,
        xaxis_rangeslider_visible=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    if log_scale:
        figure.update_yaxes(type="log")
    return figure


def create_btc_rsi_chart(chart_data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_hrect(y0=0, y1=30, opacity=0.12, line_width=0, annotation_text="Überverkauft")
    figure.add_hrect(y0=70, y1=100, opacity=0.12, line_width=0, annotation_text="Überkauft")
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["RSI"], mode="lines", name="RSI"))
    figure.add_hline(y=30, line_dash="dash", annotation_text="RSI 30")
    figure.add_hline(y=70, line_dash="dash", annotation_text="RSI 70")
    figure.update_layout(title="RSI14", height=330, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    figure.update_yaxes(range=[0, 100])
    return figure


def create_btc_drawdown_chart(chart_data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_hrect(y0=-100, y1=-40, opacity=0.10, line_width=0, annotation_text="WATCH-Zone")
    figure.add_trace(
        go.Scatter(x=chart_data.index, y=chart_data["DRAWDOWN"], mode="lines", fill="tozeroy", name="Drawdown")
    )
    figure.add_hline(y=-40, line_dash="dash", annotation_text="WATCH-Grenze: -40 %")
    figure.update_layout(
        title="Drawdown vom bisherigen All-Time-High",
        height=330,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def create_btc_score_chart(chart_data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SETUP_SCORE"], mode="lines", name="Setup-Score"))
    figure.add_trace(
        go.Scatter(x=chart_data.index, y=chart_data["CONFIRMATION_SCORE"], mode="lines", name="Bestätigungs-Score")
    )
    figure.update_layout(title="Setup- und Bestätigungs-Score", height=330, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    figure.update_yaxes(range=[0, 100])
    return figure


def create_btc_backtest_chart(events: pd.DataFrame, horizon: int) -> go.Figure | None:
    column = f"RET_{horizon}D"
    if events.empty or column not in events.columns:
        return None

    chart_data = events[["DATUM", column]].copy().dropna(subset=[column])
    if chart_data.empty:
        return None

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=chart_data["DATUM"],
            y=chart_data[column],
            text=[f"{value:.1f} %" for value in chart_data[column]],
            textposition="outside",
            name=f"{horizon}-Tage-Rendite",
        )
    )
    figure.add_hline(y=0, line_dash="dash")
    figure.update_layout(
        title=f"Rendite je EARLY-Signal nach {horizon} Tagen",
        height=500,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def render_bitcoin_page() -> None:
    st.markdown(
        """
        <div class="lab-asset-header">
            <span class="lab-kicker">₿ Bitcoin · Daily Reversal Monitor</span>
            <h2>Bullish Reversal Dashboard</h2>
            <p>Stressphase, Bodenbildung und bullische Frühbestätigung auf Basis vollständiger Tageskerzen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls_1, controls_2, controls_3 = st.columns([2.2, 1.3, 1.0])
    with controls_1:
        lookback_option = st.selectbox(
            "Angezeigter Zeitraum",
            ["180 Tage", "365 Tage", "730 Tage", "Gesamter Zeitraum"],
            index=1,
            key="btc_lookback",
        )
    with controls_2:
        log_scale = st.toggle("Logarithmische Preisskala", value=False, key="btc_log_scale")
    with controls_3:
        st.write("")
        if st.button("BTC-Daten aktualisieren", key="refresh_btc", use_container_width=True):
            load_btc_market_data.clear()
            st.rerun()

    with st.spinner("BTC-Daten werden geladen und analysiert ..."):
        try:
            raw_data, data_source, fallback_reason = load_btc_market_data()
            analysis = add_btc_indicators(raw_data)
        except Exception as error:
            st.error(f"BTC-Daten konnten nicht geladen werden: {error}")
            return

    if analysis.empty:
        st.error("Nach der Berechnung stehen keine auswertbaren BTC-Daten zur Verfügung.")
        return

    watch_phases = select_signal_phases(analysis, "WATCH_SIGNAL", BTC_COOLDOWN_DAYS)
    early_phases = select_signal_phases(analysis, "EARLY_SIGNAL", BTC_COOLDOWN_DAYS)
    early_events = calculate_btc_event_table(analysis, early_phases)

    if lookback_option == "180 Tage":
        chart_data = analysis.tail(180)
    elif lookback_option == "365 Tage":
        chart_data = analysis.tail(365)
    elif lookback_option == "730 Tage":
        chart_data = analysis.tail(730)
    else:
        chart_data = analysis.copy()

    visible_watch_phases = watch_phases[watch_phases.index >= chart_data.index.min()]
    visible_early_phases = early_phases[early_phases.index >= chart_data.index.min()]

    latest = analysis.iloc[-1]
    latest_date = analysis.index[-1].date()
    price = float(latest["Close"])
    rsi = float(latest["RSI"])
    drawdown = float(latest["DRAWDOWN"])
    vol_ratio = float(latest["VOL_RATIO"])
    setup_score = int(latest["SETUP_SCORE"])
    confirmation_score = int(latest["CONFIRMATION_SCORE"])
    watch_live = bool(latest["WATCH_SIGNAL"])
    early_live = bool(latest["EARLY_SIGNAL"])

    st.caption(
        f"Letzte vollständig ausgewertete Tageskerze: {latest_date} · Datenquelle: {data_source} · "
        f"Cache: {BTC_CACHE_TTL_SECONDS // 60} Minuten"
    )
    if fallback_reason:
        st.warning("Yahoo-Finance-Daten konnten aktuell nicht geladen werden. Angezeigt wird die lokale BTC-Fallback-Datei.")

    metric_columns = st.columns(6)
    metric_columns[0].metric("BTC-Kurs", f"{price:,.0f} USD")
    metric_columns[1].metric("RSI14", f"{rsi:.1f}")
    metric_columns[2].metric("Drawdown", f"{drawdown:.1f} %")
    metric_columns[3].metric("Volumen-Faktor", f"{vol_ratio:.2f}x")
    metric_columns[4].metric("Setup-Score", f"{setup_score}/100")
    metric_columns[5].metric("Bestätigung", f"{confirmation_score}/100")

    if early_live:
        render_status_box(
            "EARLY · Bullische Frühbestätigung aktiv",
            "Die WATCH-Kriterien sind erfüllt und der MACD liegt oberhalb seiner Signallinie.",
            "positive",
        )
    elif watch_live:
        render_status_box(
            "WATCH · Stressphase aktiv",
            "Der Markt ist kurzfristig überverkauft und liegt mindestens 40 % unter seinem bisherigen Hoch. Die bullische Bestätigung fehlt noch.",
            "warning",
        )
    else:
        render_status_box("NEUTRAL", "Aktuell ist kein qualifiziertes bullisches Frühwarnsignal aktiv.", "info")

    btc_assistant_context = build_btc_assistant_context(latest, analysis, latest_date)
    render_market_assistant("bitcoin", btc_assistant_context)

    st.subheader("Checkliste bis zur bullischen Frühbestätigung")
    checklist = pd.DataFrame(
        [
            {"Bedingung": "Drawdown unter -40 %", "Rolle": "WATCH-Pflichtbedingung", "Status": format_check(drawdown < -40)},
            {"Bedingung": "RSI14 unter 30", "Rolle": "WATCH-Pflichtbedingung", "Status": format_check(rsi < 30)},
            {
                "Bedingung": "MACD über Signallinie",
                "Rolle": "EARLY-Pflichtbedingung",
                "Status": format_check(float(latest["MACD"]) > float(latest["MACD_SIGNAL"])),
            },
            {"Bedingung": "Volumen mindestens 1.3x", "Rolle": "Zusatzinformation", "Status": format_check(vol_ratio >= 1.3)},
            {"Bedingung": "Kurs über EMA20", "Rolle": "Zusatzinformation", "Status": format_check(price > float(latest["EMA20"]))},
            {
                "Bedingung": "RSI steigt gegenüber vor 3 Tagen",
                "Rolle": "Zusatzinformation",
                "Status": format_check(rsi > float(analysis.iloc[-4]["RSI"])),
            },
        ]
    )
    st.dataframe(checklist, use_container_width=True, hide_index=True)

    chart_tab, signals_tab, backtest_tab, methodology_tab = st.tabs(
        ["Charts", "Historische EARLY-Signale", "Backtest", "Methodik"]
    )

    with chart_tab:
        st.plotly_chart(create_btc_price_chart(chart_data, visible_watch_phases, visible_early_phases, log_scale), use_container_width=True)
        st.plotly_chart(create_btc_rsi_chart(chart_data), use_container_width=True)
        st.plotly_chart(create_btc_drawdown_chart(chart_data), use_container_width=True)
        st.plotly_chart(create_btc_score_chart(chart_data), use_container_width=True)

    with signals_tab:
        st.subheader("Historische EARLY-Signalphasen")
        st.caption(f"Eng aufeinanderfolgende Signale werden durch einen Cooldown von {BTC_COOLDOWN_DAYS} Tagen zu einer Phase gebündelt.")
        if early_events.empty:
            st.info("Keine historischen EARLY-Signalphasen vorhanden.")
        else:
            signal_table = early_events.copy()
            signal_table["DATUM"] = signal_table["DATUM"].dt.date
            numeric_columns = ["BTC_PREIS", "DRAWDOWN", "RSI", "VOL_RATIO", "RET_30D", "RET_90D", "RET_180D"]
            for column in numeric_columns:
                if column in signal_table.columns:
                    signal_table[column] = signal_table[column].round(1)
            st.dataframe(signal_table, use_container_width=True, hide_index=True)
            st.download_button(
                label="EARLY-Signale als CSV herunterladen",
                data=early_events.to_csv(index=False),
                file_name="btc_bullish_early_signals.csv",
                mime="text/csv",
                key="download_btc_early",
            )

    with backtest_tab:
        st.subheader("Historische Einordnung der EARLY-Signale")
        st.caption("Die Anzahl historischer Ereignisse bleibt begrenzt. Die Tabelle dient der explorativen Einordnung der Signalphasen.")
        summary = create_btc_summary_table(early_events)
        if summary.empty:
            st.info("Noch keine auswertbaren Ergebnisse vorhanden.")
        else:
            horizon = st.selectbox(
                "Horizont für Detailansicht",
                BTC_HORIZONS,
                index=1,
                format_func=lambda value: f"{value} Tage",
                key="btc_backtest_horizon",
            )
            selected_row = summary[summary["Horizont"] == f"{horizon} Tage"].iloc[0]
            summary_metrics = st.columns(4)
            summary_metrics[0].metric("EARLY-Signalphasen", int(selected_row["Ereignisse"]))
            summary_metrics[1].metric("Trefferquote", format_percent(selected_row["Trefferquote"]))
            summary_metrics[2].metric("Median-Rendite", format_percent(selected_row["Median"]))
            summary_metrics[3].metric("Ø Rendite", format_percent(selected_row["Ø Rendite"]))

            backtest_chart = create_btc_backtest_chart(early_events, horizon)
            if backtest_chart is not None:
                st.plotly_chart(backtest_chart, use_container_width=True)

            formatted_summary = summary.copy()
            for column in ["Trefferquote", "Ø Rendite", "Median", "Beste Rendite", "Schlechteste Rendite"]:
                formatted_summary[column] = formatted_summary[column].map(format_percent)
            formatted_summary["Profit Factor"] = formatted_summary["Profit Factor"].map(
                lambda value: "-" if pd.isna(value) else f"{value:.2f}"
            )
            st.dataframe(formatted_summary, use_container_width=True, hide_index=True)

    with methodology_tab:
        st.subheader("Methodik")
        st.markdown(
            """
            **WATCH** wird aktiv, wenn BTC mindestens 40 % unter seinem bisherigen All-Time-High liegt und der RSI14 unter 30 fällt.

            **EARLY** wird aktiv, wenn zusätzlich der MACD über seiner Signallinie liegt. Erhöhtes Volumen bleibt eine ergänzende Information.

            Der **Setup-Score** beschreibt die Intensität der Stress- und Bodenbildungsphase. Der **Bestätigungs-Score** beschreibt,
            ob Momentum und kurzfristige Marktstruktur bereits nach oben drehen.
            """
        )


# ============================================================
# 7. SPACEX: KALENDER UND DATEN
# ============================================================

def is_nasdaq_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in NASDAQ_MARKET_HOLIDAYS_2026


def trading_days_from(start_day: date, number_of_days: int) -> list[date]:
    days: list[date] = []
    current_day = start_day
    while len(days) < number_of_days:
        if is_nasdaq_trading_day(current_day):
            days.append(current_day)
        current_day += timedelta(days=1)
    return days


def get_spcx_fast_entry_timeline() -> dict[str, date]:
    first_15_days = trading_days_from(SPCX_IPO_DATE, 15)
    return {
        "ipo": SPCX_IPO_DATE,
        "reference": first_15_days[6],
        "expected_announcement": SPCX_OFFICIAL_ANNOUNCEMENT_DATE or first_15_days[9],
        "expected_effective": SPCX_OFFICIAL_EFFECTIVE_DATE or first_15_days[14],
    }


def market_open_timestamp_utc(day: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time(9, 30), tzinfo=NY_TZ)).tz_convert("UTC")


def regular_bar_end_utc(bar_start_utc: pd.Timestamp) -> pd.Timestamp:
    bar_start_et = bar_start_utc.tz_convert(NY_TZ)
    session_close_et = pd.Timestamp(datetime.combine(bar_start_et.date(), time(16, 0), tzinfo=NY_TZ))
    return min(bar_start_et + pd.Timedelta(hours=1), session_close_et).tz_convert("UTC")


def completed_trading_day_count(as_of_et: datetime) -> int:
    completed = 0
    for trading_day in trading_days_from(SPCX_IPO_DATE, 15):
        if trading_day < as_of_et.date():
            completed += 1
        elif trading_day == as_of_et.date() and as_of_et.time() >= time(16, 0):
            completed += 1
    return completed


def format_date(value: date | None) -> str:
    return "-" if value is None else value.strftime("%d.%m.%Y")


def timeline_status(event_day: date, now_et: datetime) -> str:
    if event_day < now_et.date():
        return "✅ Erreicht"
    if event_day == now_et.date():
        return "🟡 Heute"
    return "⏳ Ausstehend"


def filter_regular_session(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()

    timestamps_et = data.index.tz_convert(NY_TZ)
    minutes_after_midnight = timestamps_et.hour * 60 + timestamps_et.minute
    regular_mask = (
        (timestamps_et.weekday < 5)
        & (minutes_after_midnight >= 9 * 60 + 30)
        & (minutes_after_midnight < 16 * 60)
    )
    return data.loc[regular_mask].copy()


def drop_unfinished_regular_bar(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    now_utc = pd.Timestamp.now(tz="UTC")
    latest_bar_start = data.index[-1]
    if now_utc < regular_bar_end_utc(latest_bar_start):
        return data.iloc[:-1].copy()
    return data.copy()


def load_spcx_fallback_data() -> pd.DataFrame | None:
    if not SPCX_FALLBACK_FILE.exists():
        return None

    fallback = pd.read_csv(SPCX_FALLBACK_FILE)
    timestamp_column = next((column for column in ["Datetime", "Date", "Timestamp"] if column in fallback.columns), None)
    if timestamp_column is None:
        raise RuntimeError("Die SpaceX-Fallback-Datei benötigt eine Zeitspalte namens Datetime, Date oder Timestamp.")

    fallback[timestamp_column] = pd.to_datetime(fallback[timestamp_column])
    fallback = fallback.set_index(timestamp_column)
    return normalize_market_columns(fallback, force_utc=True)


@st.cache_data(ttl=SPCX_CACHE_TTL_SECONDS, show_spinner=False)
def load_spcx_regular_market_data() -> tuple[pd.DataFrame, str, str | None]:
    source = f"Yahoo Finance via yfinance ({SPCX_TICKER}, regulärer Handel)"
    fallback_reason = None

    try:
        market_data = yf.download(
            SPCX_TICKER,
            start=SPCX_DATA_START,
            interval="1h",
            prepost=False,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        market_data = normalize_market_columns(market_data, force_utc=True)
        market_data = filter_regular_session(market_data)
        market_data = drop_unfinished_regular_bar(market_data)
        if market_data.empty:
            raise RuntimeError(f"Yahoo Finance hat keine Stundenkerzen für {SPCX_TICKER} geliefert.")

    except Exception as error:
        market_data = load_spcx_fallback_data()
        if market_data is None:
            raise RuntimeError(
                f"Die aktuellen {SPCX_ASSET_NAME}-Daten konnten nicht geladen werden und es ist keine lokale Fallback-Datei vorhanden."
            ) from error
        market_data = filter_regular_session(market_data)
        market_data = drop_unfinished_regular_bar(market_data)
        source = f"Fallback-Datei {SPCX_FALLBACK_FILE.name}"
        fallback_reason = str(error)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in market_data.columns]
    if missing_columns:
        raise RuntimeError("Folgende SpaceX-Kursdaten-Spalten fehlen: " + ", ".join(missing_columns))

    return market_data[required_columns].copy(), source, fallback_reason


def classify_extended_session(timestamp_utc: pd.Timestamp) -> str:
    local_time = timestamp_utc.tz_convert(NY_TZ).time()
    if time(4, 0) <= local_time < time(9, 30):
        return "Vorbörslich"
    if time(9, 30) <= local_time < time(16, 0):
        return "Regulärer Handel"
    if time(16, 0) <= local_time < time(20, 0):
        return "Nachbörslich"
    return "Außerhalb der US-Handelszeiten"


@st.cache_data(ttl=SPCX_QUOTE_CACHE_TTL_SECONDS, show_spinner=False)
def load_spcx_reference_quote() -> dict[str, object] | None:
    try:
        quote_data = yf.Ticker(SPCX_TICKER).history(period="1d", interval="1m", prepost=True, auto_adjust=False)
        if quote_data.empty:
            return None
        quote_data = normalize_market_columns(quote_data, force_utc=True).dropna(subset=["Close"])
        if quote_data.empty:
            return None
        latest_timestamp = quote_data.index[-1]
        return {
            "timestamp": latest_timestamp,
            "price": float(quote_data.iloc[-1]["Close"]),
            "session": classify_extended_session(latest_timestamp),
        }
    except Exception:
        return None


# ============================================================
# 8. SPACEX: INDIKATOREN UND SIGNALLOGIK
# ============================================================

def calculate_rsi(series: pd.Series, period: int = 7) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.mask((average_loss == 0) & (average_gain > 0), 100)


def calculate_spcx_index_timing_score(completed_trade_days: int, as_of_day: date) -> int:
    if SPCX_OFFICIAL_FAST_ENTRY_ANNOUNCED:
        if SPCX_OFFICIAL_EFFECTIVE_DATE is None:
            return 90
        if as_of_day < SPCX_OFFICIAL_EFFECTIVE_DATE:
            return 100
        if as_of_day == SPCX_OFFICIAL_EFFECTIVE_DATE:
            return 80
        return 25

    if completed_trade_days < 7:
        return 20
    if completed_trade_days < 10:
        return 50
    if completed_trade_days < 15:
        return 80
    return 25


def add_spcx_indicators(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()
    data["RETURN_1H"] = data["Close"].pct_change()
    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["RSI7"] = calculate_rsi(data["Close"], period=7)

    data["TYPICAL_PRICE"] = (data["High"] + data["Low"] + data["Close"]) / 3
    data["TRADED_VALUE"] = data["TYPICAL_PRICE"] * data["Volume"]
    cumulative_volume = data["Volume"].cumsum().replace(0, np.nan)
    data["IPO_AVWAP"] = data["TRADED_VALUE"].cumsum() / cumulative_volume
    data["POST_IPO_HIGH"] = data["High"].cummax()
    data["DRAWDOWN_FROM_HIGH"] = (data["Close"] / data["POST_IPO_HIGH"] - 1) * 100
    data["IPO_PREMIUM"] = (data["Close"] / SPCX_IPO_PRICE_USD - 1) * 100
    data["DISTANCE_TO_AVWAP"] = (data["Close"] / data["IPO_AVWAP"] - 1) * 100

    previous_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR10"] = true_range.rolling(10, min_periods=5).mean()
    data["ATR10_PCT"] = data["ATR10"] / data["Close"] * 100
    data["VOLATILITY_COOLING"] = data["ATR10_PCT"] < data["ATR10_PCT"].shift(5)
    data["VOL_MA10"] = data["Volume"].rolling(10, min_periods=5).mean()
    data["VOL_RATIO"] = data["Volume"] / data["VOL_MA10"]

    up_volume = data["Volume"].where(data["RETURN_1H"] > 0, 0.0)
    down_volume = data["Volume"].where(data["RETURN_1H"] < 0, 0.0)
    data["UP_VOLUME_8H"] = up_volume.rolling(8, min_periods=4).sum()
    data["DOWN_VOLUME_8H"] = down_volume.rolling(8, min_periods=4).sum()
    data["UP_DOWN_VOLUME_RATIO"] = data["UP_VOLUME_8H"] / data["DOWN_VOLUME_8H"].replace(0, np.nan)
    data["UP_DOWN_VOLUME_RATIO"] = data["UP_DOWN_VOLUME_RATIO"].replace([np.inf, -np.inf], np.nan)

    current_three_bar_low = data["Low"].rolling(3, min_periods=3).min()
    previous_three_bar_low = current_three_bar_low.shift(3)
    data["HIGHER_LOW"] = current_three_bar_low > previous_three_bar_low
    data["EMA_BULL_STRUCTURE"] = data["EMA9"] > data["EMA21"]
    data["ABOVE_AVWAP"] = data["Close"] >= data["IPO_AVWAP"]

    premium_points = np.select(
        [data["IPO_PREMIUM"] >= 60, data["IPO_PREMIUM"] >= 40, data["IPO_PREMIUM"] >= 25, data["IPO_PREMIUM"] >= 10],
        [30, 24, 16, 8],
        default=0,
    )
    shallow_pullback_points = np.select(
        [data["DRAWDOWN_FROM_HIGH"] > -3, data["DRAWDOWN_FROM_HIGH"] > -7, data["DRAWDOWN_FROM_HIGH"] > -12],
        [20, 12, 5],
        default=0,
    )
    rsi_hype_points = np.select([data["RSI7"] >= 80, data["RSI7"] >= 70, data["RSI7"] >= 60], [20, 14, 7], default=0)
    atr_hype_points = np.select([data["ATR10_PCT"] >= 6, data["ATR10_PCT"] >= 4, data["ATR10_PCT"] >= 2.5], [15, 10, 5], default=0)
    volume_hype_points = np.select([data["VOL_RATIO"] >= 2.0, data["VOL_RATIO"] >= 1.5, data["VOL_RATIO"] >= 1.1], [15, 10, 5], default=0)
    data["HYPE_RISK_SCORE"] = np.minimum(
        100, premium_points + shallow_pullback_points + rsi_hype_points + atr_hype_points + volume_hype_points
    ).astype(int)

    pullback_quality_points = np.select(
        [
            (data["DRAWDOWN_FROM_HIGH"] <= -8) & (data["DRAWDOWN_FROM_HIGH"] >= -25),
            (data["DRAWDOWN_FROM_HIGH"] <= -5) & (data["DRAWDOWN_FROM_HIGH"] >= -30),
            (data["DRAWDOWN_FROM_HIGH"] <= -3) & (data["DRAWDOWN_FROM_HIGH"] >= -35),
        ],
        [20, 12, 6],
        default=0,
    )
    entry_quality_score = np.zeros(len(data), dtype=int)
    entry_quality_score += pullback_quality_points
    entry_quality_score += np.where(data["Close"] >= SPCX_IPO_PRICE_USD, 10, 0)
    entry_quality_score += np.where(data["ABOVE_AVWAP"], 20, 0)
    entry_quality_score += np.where(data["EMA_BULL_STRUCTURE"], 15, 0)
    entry_quality_score += np.where(data["Close"] >= data["EMA9"], 5, 0)
    entry_quality_score += np.where(data["HIGHER_LOW"], 10, 0)
    entry_quality_score += np.where(data["VOLATILITY_COOLING"], 10, 0)
    entry_quality_score += np.where(data["RSI7"].between(45, 68, inclusive="both"), 5, 0)
    entry_quality_score += np.where(data["UP_DOWN_VOLUME_RATIO"] >= 1.1, 5, 0)
    data["ENTRY_QUALITY_SCORE"] = np.minimum(100, entry_quality_score).astype(int)

    data["SIGNAL_READY"] = (
        (np.arange(len(data)) + 1 >= SPCX_MIN_SIGNAL_BARS)
        & data["RSI7"].notna()
        & data["IPO_AVWAP"].notna()
        & data["ATR10_PCT"].notna()
    )

    timing_scores: list[int] = []
    for timestamp_utc in data.index:
        timestamp_et = timestamp_utc.tz_convert(NY_TZ)
        bar_end_et = min(timestamp_et + pd.Timedelta(hours=1), pd.Timestamp(datetime.combine(timestamp_et.date(), time(16, 0), tzinfo=NY_TZ)))
        completed_days = completed_trading_day_count(bar_end_et.to_pydatetime())
        timing_scores.append(calculate_spcx_index_timing_score(completed_days, bar_end_et.date()))

    data["INDEX_TIMING_SCORE"] = timing_scores
    data["TACTICAL_SCORE"] = np.clip(
        (
            data["ENTRY_QUALITY_SCORE"] * 0.75
            + data["INDEX_TIMING_SCORE"] * 0.25
            - np.maximum(0, data["HYPE_RISK_SCORE"] - 55) * 0.35
        ).round(),
        0,
        100,
    ).astype(int)
    data["TACTICAL_ENTRY_WATCH"] = (
        data["SIGNAL_READY"]
        & (data["TACTICAL_SCORE"] >= 75)
        & (data["ENTRY_QUALITY_SCORE"] >= 70)
        & (data["HYPE_RISK_SCORE"] <= 55)
        & data["ABOVE_AVWAP"]
        & data["EMA_BULL_STRUCTURE"]
    )
    return data


def calculate_spcx_advt_proxy(data: pd.DataFrame) -> tuple[float | None, int]:
    if data.empty:
        return None, 0
    now_et = datetime.now(tz=NY_TZ)
    working_data = data.copy()
    working_data["TRADE_DATE_ET"] = working_data.index.tz_convert(NY_TZ).date
    completed_dates = [
        trading_day
        for trading_day in sorted(working_data["TRADE_DATE_ET"].unique())
        if trading_day < now_et.date() or (trading_day == now_et.date() and now_et.time() >= time(16, 0))
    ]
    if not completed_dates:
        return None, 0
    daily_values = working_data[working_data["TRADE_DATE_ET"].isin(completed_dates)].groupby("TRADE_DATE_ET")["TRADED_VALUE"].sum()
    if daily_values.empty:
        return None, 0
    return float(daily_values.mean()), int(len(daily_values))


def classify_spcx_market_phase(latest: pd.Series) -> tuple[str, str, str]:
    if not bool(latest["SIGNAL_READY"]):
        return "PRICE DISCOVERY", "Für eine robuste IPO-Stundenanalyse fehlen noch vollständige reguläre Handelskerzen.", "info"

    hype_score = int(latest["HYPE_RISK_SCORE"])
    entry_score = int(latest["ENTRY_QUALITY_SCORE"])
    tactical_score = int(latest["TACTICAL_SCORE"])
    drawdown = float(latest["DRAWDOWN_FROM_HIGH"])

    if hype_score >= 65:
        return "HYPE", "Die Preisfindung wirkt weiterhin deutlich euphoriegetrieben oder überdehnt.", "warning"

    if (
        tactical_score >= 75
        and entry_score >= 70
        and hype_score <= 55
        and bool(latest["ABOVE_AVWAP"])
        and bool(latest["EMA_BULL_STRUCTURE"])
    ):
        if SPCX_OFFICIAL_FAST_ENTRY_ANNOUNCED:
            return "CONFIRMED WINDOW", "Die Stabilisierungssignale sind konstruktiv und das Fast Entry ist als offiziell angekündigt hinterlegt.", "positive"
        return "ENTRY WATCH", "Mehrere Stabilisierungssignale sind konstruktiv. Die Nasdaq-100-Aufnahme ist noch nicht offiziell bestätigt.", "positive"

    if entry_score >= 55:
        return "STABILISIERUNG", "Die Kursstruktur verbessert sich. Das taktische Fenster ist noch nicht vollständig bestätigt.", "info"

    if drawdown <= -5:
        return "COOLDOWN", "Die erste IPO-Euphorie wird abgebaut. Für ein konstruktiveres Fenster fehlen noch Stabilisierungssignale.", "warning"

    return "PRICE DISCOVERY", "Die Aktie befindet sich weiterhin in der frühen Preisfindungsphase nach dem IPO.", "info"


def score_label(value: int) -> str:
    if value >= 75:
        return "hoch"
    if value >= 50:
        return "mittel"
    return "niedrig"


# ============================================================
# 9. SPACEX: CHARTS UND UI
# ============================================================

def add_spcx_timeline_lines(figure: go.Figure, chart_data: pd.DataFrame, timeline: dict[str, date]) -> None:
    if chart_data.empty:
        return
    first_timestamp = chart_data.index.min()
    last_timestamp = chart_data.index.max()
    events = [
        ("IPO", timeline["ipo"]),
        ("7. Handelstag: Prüfung", timeline["reference"]),
        ("10. Handelstag: mögliche Ankündigung", timeline["expected_announcement"]),
        ("15. Handelstag: mögliche Aufnahme", timeline["expected_effective"]),
    ]
    for label, event_day in events:
        event_timestamp = market_open_timestamp_utc(event_day)
        if first_timestamp <= event_timestamp <= last_timestamp + pd.Timedelta(days=7):
            # Plotly kann bei add_vline(..., annotation_text=...) mit Datetime-Werten
            # je nach Version einen TypeError auslösen. Linie und Beschriftung werden
            # deshalb bewusst getrennt hinzugefügt.
            event_x = event_timestamp.to_pydatetime()
            figure.add_shape(
                type="line",
                x0=event_x,
                x1=event_x,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line={"dash": "dot"},
            )
            figure.add_annotation(
                x=event_x,
                y=1,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                yshift=12,
            )


def create_spcx_price_chart(chart_data: pd.DataFrame, timeline: dict[str, date]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="SPCX",
        )
    )
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["IPO_AVWAP"], mode="lines", name="IPO-anchored VWAP"))
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["EMA9"], mode="lines", name="EMA9"))
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["EMA21"], mode="lines", name="EMA21"))

    entry_watch_data = chart_data[chart_data["TACTICAL_ENTRY_WATCH"]]
    if not entry_watch_data.empty:
        figure.add_trace(
            go.Scatter(
                x=entry_watch_data.index,
                y=entry_watch_data["Close"],
                mode="markers",
                name="ENTRY WATCH",
                marker={"symbol": "triangle-up", "size": 13},
            )
        )

    figure.add_hline(y=SPCX_IPO_PRICE_USD, line_dash="dash", annotation_text=f"IPO-Preis: {SPCX_IPO_PRICE_USD:.0f} USD")
    add_spcx_timeline_lines(figure, chart_data, timeline)
    figure.update_layout(
        title="SPCX: reguläre Stundenkerzen, IPO-anchored VWAP und taktische Einstiegsfenster",
        height=680,
        xaxis_rangeslider_visible=False,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return figure


def create_spcx_score_chart(chart_data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["HYPE_RISK_SCORE"], mode="lines", name="Hype-Risiko"))
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["ENTRY_QUALITY_SCORE"], mode="lines", name="Einstiegsqualität"))
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["TACTICAL_SCORE"], mode="lines", name="Taktischer Score inkl. Index-Timing"))
    figure.update_layout(title="Heuristische Scores für Hype, Stabilisierung und Index-Timing", height=380, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    figure.update_yaxes(range=[0, 100])
    return figure


def create_spcx_structure_chart(chart_data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["DRAWDOWN_FROM_HIGH"], mode="lines", name="Drawdown vom Post-IPO-Hoch"))
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["DISTANCE_TO_AVWAP"], mode="lines", name="Abstand zum IPO-AVWAP"))
    figure.add_hline(y=0, line_dash="dash")
    figure.update_layout(title="Abbau der IPO-Euphorie und Abstand zum anchored VWAP", height=350, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return figure


def create_spcx_momentum_chart(chart_data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["RSI7"], mode="lines", name="RSI7"))
    figure.add_hline(y=70, line_dash="dash", annotation_text="RSI 70")
    figure.add_hline(y=50, line_dash="dot", annotation_text="RSI 50")
    figure.add_hline(y=30, line_dash="dash", annotation_text="RSI 30")
    figure.update_layout(title="Kurzfristiges Momentum: RSI7 auf regulären Stundenkerzen", height=330, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    figure.update_yaxes(range=[0, 100])
    return figure


def render_spacex_page() -> None:
    st.markdown(
        """
        <div class="lab-asset-header">
            <span class="lab-kicker">🚀 SpaceX · IPO Cooldown Monitor</span>
            <h2>SPCX Nasdaq-100 Fast-Entry Watch</h2>
            <p>Stündliche Analyse der IPO-Preisfindung, der Hype-Abkühlung und eines möglichen Nasdaq-100-Fast-Entry-Fensters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls_1, controls_2 = st.columns([2.5, 1.0])
    with controls_1:
        lookback_option = st.selectbox(
            "Angezeigter Zeitraum",
            ["50 Stundenkerzen", "100 Stundenkerzen", "200 Stundenkerzen", "Gesamter Zeitraum"],
            index=3,
            key="spcx_lookback",
        )
    with controls_2:
        st.write("")
        if st.button("SPCX-Daten aktualisieren", key="refresh_spcx", use_container_width=True):
            load_spcx_regular_market_data.clear()
            load_spcx_reference_quote.clear()
            st.rerun()

    with st.spinner("SPCX-Stundenkerzen werden geladen und analysiert ..."):
        try:
            raw_data, data_source, fallback_reason = load_spcx_regular_market_data()
            analysis = add_spcx_indicators(raw_data)
            reference_quote = load_spcx_reference_quote()
        except Exception as error:
            st.error(f"SPCX-Daten konnten nicht geladen werden: {error}")
            return

    if analysis.empty:
        st.error("Nach der Berechnung stehen keine auswertbaren SpaceX-Daten zur Verfügung.")
        return

    if lookback_option == "50 Stundenkerzen":
        chart_data = analysis.tail(50)
    elif lookback_option == "100 Stundenkerzen":
        chart_data = analysis.tail(100)
    elif lookback_option == "200 Stundenkerzen":
        chart_data = analysis.tail(200)
    else:
        chart_data = analysis.copy()

    now_et = datetime.now(tz=NY_TZ)
    timeline = get_spcx_fast_entry_timeline()
    completed_trade_days = completed_trading_day_count(now_et)
    current_index_timing_score = calculate_spcx_index_timing_score(completed_trade_days, now_et.date())
    advt_proxy, advt_completed_days = calculate_spcx_advt_proxy(analysis)
    latest = analysis.iloc[-1].copy()

    # Nach Handelsschluss kann der aktuelle Kalenderstatus bereits weiter sein als der Status
    # der letzten Stundenkerze. Für den aktuellen Status wird der Timing-Anteil daher aktualisiert.
    latest_tactical_score = int(
        np.clip(
            round(
                int(latest["ENTRY_QUALITY_SCORE"]) * 0.75
                + current_index_timing_score * 0.25
                - max(0, int(latest["HYPE_RISK_SCORE"]) - 55) * 0.35
            ),
            0,
            100,
        )
    )
    latest["INDEX_TIMING_SCORE"] = current_index_timing_score
    latest["TACTICAL_SCORE"] = latest_tactical_score
    market_phase, market_phase_description, market_phase_level = classify_spcx_market_phase(latest)

    latest_timestamp_et = analysis.index[-1].tz_convert(NY_TZ)
    st.caption(
        f"Letzte vollständig ausgewertete reguläre Stundenkerze: {latest_timestamp_et.strftime('%d.%m.%Y %H:%M')} ET · "
        f"Datenquelle: {data_source} · Cache: {SPCX_CACHE_TTL_SECONDS // 60} Minuten"
    )
    if fallback_reason:
        st.warning("Yahoo-Finance-Daten konnten aktuell nicht geladen werden. Angezeigt wird die lokale SpaceX-Fallback-Datei.")

    if reference_quote is not None:
        quote_timestamp_et = pd.Timestamp(reference_quote["timestamp"]).tz_convert(NY_TZ)
        st.info(
            f"Referenznotierung inkl. Extended Hours: **{float(reference_quote['price']):,.2f} USD** · "
            f"{reference_quote['session']} · {quote_timestamp_et.strftime('%d.%m.%Y %H:%M')} ET. "
            "Die Notierung wird separat angezeigt und nicht in das reguläre Stundenmodell eingemischt."
        )

    metric_row_1 = st.columns(4)
    metric_row_1[0].metric("Letzter regulärer Schlusskurs", format_usd(float(latest["Close"])))
    metric_row_1[1].metric("Prämie ggü. IPO-Preis", format_percent(float(latest["IPO_PREMIUM"])))
    metric_row_1[2].metric("Drawdown vom Post-IPO-Hoch", format_percent(float(latest["DRAWDOWN_FROM_HIGH"])))
    metric_row_1[3].metric("Abstand zum IPO-AVWAP", format_percent(float(latest["DISTANCE_TO_AVWAP"])))

    metric_row_2 = st.columns(4)
    metric_row_2[0].metric("Hype-Risiko", f"{int(latest['HYPE_RISK_SCORE'])}/100")
    metric_row_2[1].metric("Einstiegsqualität", f"{int(latest['ENTRY_QUALITY_SCORE'])}/100")
    metric_row_2[2].metric("Index-Timing", f"{current_index_timing_score}/100")
    metric_row_2[3].metric("Taktischer Score", f"{latest_tactical_score}/100")

    render_status_box(market_phase, market_phase_description, market_phase_level)

    spcx_assistant_context = build_spcx_assistant_context(
        latest=latest,
        timeline=timeline,
        completed_trade_days=completed_trade_days,
        advt_proxy=advt_proxy,
        advt_completed_days=advt_completed_days,
        market_phase=market_phase,
        current_index_timing_score=current_index_timing_score,
        latest_tactical_score=latest_tactical_score,
    )
    render_market_assistant("spacex", spcx_assistant_context)

    st.subheader("Nasdaq-100 Fast-Entry-Watch")
    if SPCX_OFFICIAL_FAST_ENTRY_ANNOUNCED:
        st.success("Eine offizielle Fast-Entry-Ankündigung ist im Konfigurationsbereich hinterlegt.")
    else:
        st.info("Die Termine sind aktuell methodikbasierte Erwartungstermine. Eine offizielle Fast-Entry-Ankündigung ist noch nicht hinterlegt.")

    fast_entry_metrics = st.columns(4)
    fast_entry_metrics[0].metric("Abgeschlossene Handelstage", completed_trade_days)
    fast_entry_metrics[1].metric("7. Handelstag: Prüfung", format_date(timeline["reference"]))
    fast_entry_metrics[2].metric("10. Handelstag: mögliche Ankündigung", format_date(timeline["expected_announcement"]))
    fast_entry_metrics[3].metric("15. Handelstag: mögliche Aufnahme", format_date(timeline["expected_effective"]))

    fast_entry_table = pd.DataFrame(
        [
            {"Ereignis": "IPO und Handelsstart", "Datum": format_date(timeline["ipo"]), "Status": timeline_status(timeline["ipo"], now_et), "Einordnung": "Start der öffentlichen Preisfindung"},
            {"Ereignis": "IPO Reference Date", "Datum": format_date(timeline["reference"]), "Status": timeline_status(timeline["reference"], now_et), "Einordnung": "Bewertung nach dem 7. Handelstag"},
            {"Ereignis": "Mögliche Fast-Entry-Ankündigung", "Datum": format_date(timeline["expected_announcement"]), "Status": timeline_status(timeline["expected_announcement"], now_et), "Einordnung": "Typischerweise nach Handelsschluss am 10. Handelstag"},
            {"Ereignis": "Mögliche Nasdaq-100-Aufnahme", "Datum": format_date(timeline["expected_effective"]), "Status": timeline_status(timeline["expected_effective"], now_et), "Einordnung": "Typischerweise nach 15 Handelstagen"},
        ]
    )
    st.dataframe(fast_entry_table, use_container_width=True, hide_index=True)

    eligibility_table = pd.DataFrame(
        [
            {"Kriterium": "Nasdaq-Listing", "Status": "✅ Erfüllt", "Hinweis": "SPCX wird seit dem IPO an Nasdaq gehandelt."},
            {"Kriterium": "7. Handelstag erreicht", "Status": format_bool(completed_trade_days >= 7), "Hinweis": f"Methodikbasierter Prüfungstag: {format_date(timeline['reference'])}"},
            {
                "Kriterium": "ADVT mindestens 5 Mio. USD",
                "Status": format_bool(None if advt_proxy is None else advt_proxy >= 5_000_000),
                "Hinweis": f"Näherung aus {advt_completed_days} vollständigen Handelstagen: {format_large_usd(advt_proxy)}",
            },
            {"Kriterium": "Full Market Cap innerhalb Top 40", "Status": format_bool(SPCX_TOP40_FULL_MARKET_CAP_CONFIRMED), "Hinweis": "Manuell nach offizieller Bestätigung pflegen."},
            {"Kriterium": "Fast Entry offiziell angekündigt", "Status": format_bool(SPCX_OFFICIAL_FAST_ENTRY_ANNOUNCED), "Hinweis": "Manuell nach Veröffentlichung durch Nasdaq pflegen."},
        ]
    )
    st.dataframe(eligibility_table, use_container_width=True, hide_index=True)

    st.subheader("Checkliste für ein konstruktiveres Einstiegsfenster")
    entry_checklist = pd.DataFrame(
        [
            {"Signal": "Moderater Rücksetzer vom Post-IPO-Hoch", "Status": format_bool(-25 <= float(latest["DRAWDOWN_FROM_HIGH"]) <= -8), "Aktueller Wert": format_percent(float(latest["DRAWDOWN_FROM_HIGH"]))},
            {"Signal": "Kurs hält mindestens den IPO-Preis", "Status": format_bool(float(latest["Close"]) >= SPCX_IPO_PRICE_USD), "Aktueller Wert": format_usd(float(latest["Close"]))},
            {"Signal": "Kurs liegt über dem IPO-anchored VWAP", "Status": format_bool(bool(latest["ABOVE_AVWAP"])), "Aktueller Wert": format_percent(float(latest["DISTANCE_TO_AVWAP"]))},
            {"Signal": "EMA9 liegt über EMA21", "Status": format_bool(bool(latest["EMA_BULL_STRUCTURE"])), "Aktueller Wert": f"EMA9 {float(latest['EMA9']):,.2f} · EMA21 {float(latest['EMA21']):,.2f}"},
            {"Signal": "Kurzfristig höheres Tief", "Status": format_bool(bool(latest["HIGHER_LOW"])), "Aktueller Wert": "3-Stunden-Struktur"},
            {"Signal": "Volatilität kühlt ab", "Status": format_bool(bool(latest["VOLATILITY_COOLING"])), "Aktueller Wert": format_percent(float(latest["ATR10_PCT"]))},
            {"Signal": "RSI7 liegt im konstruktiven Bereich 45 bis 68", "Status": format_bool(bool(pd.notna(latest["RSI7"]) and 45 <= float(latest["RSI7"]) <= 68)), "Aktueller Wert": "-" if pd.isna(latest["RSI7"]) else f"{float(latest['RSI7']):.1f}"},
            {
                "Signal": "Aufwärtsvolumen überwiegt Abwärtsvolumen",
                "Status": format_bool(bool(pd.notna(latest["UP_DOWN_VOLUME_RATIO"]) and float(latest["UP_DOWN_VOLUME_RATIO"]) >= 1.1)),
                "Aktueller Wert": "-" if pd.isna(latest["UP_DOWN_VOLUME_RATIO"]) else f"{float(latest['UP_DOWN_VOLUME_RATIO']):.2f}x",
            },
        ]
    )
    st.dataframe(entry_checklist, use_container_width=True, hide_index=True)

    chart_tab, scores_tab, data_tab, methodology_tab = st.tabs(["Charts", "Score-Details", "Stundenkerzen", "Methodik"])

    with chart_tab:
        st.plotly_chart(create_spcx_price_chart(chart_data, timeline), use_container_width=True)
        st.plotly_chart(create_spcx_structure_chart(chart_data), use_container_width=True)
        st.plotly_chart(create_spcx_momentum_chart(chart_data), use_container_width=True)
        st.plotly_chart(create_spcx_score_chart(chart_data), use_container_width=True)

    with scores_tab:
        st.subheader("Aktuelle Score-Einordnung")
        score_table = pd.DataFrame(
            [
                {"Score": "Hype-Risiko", "Wert": int(latest["HYPE_RISK_SCORE"]), "Einordnung": score_label(int(latest["HYPE_RISK_SCORE"])), "Bedeutung": "Je höher der Wert, desto stärker sind Euphorie, Überdehnung oder instabile Preisfindung."},
                {"Score": "Einstiegsqualität", "Wert": int(latest["ENTRY_QUALITY_SCORE"]), "Einordnung": score_label(int(latest["ENTRY_QUALITY_SCORE"])), "Bedeutung": "Je höher der Wert, desto mehr Stabilisierungssignale liegen nach einem Rücksetzer vor."},
                {"Score": "Index-Timing", "Wert": current_index_timing_score, "Einordnung": score_label(current_index_timing_score), "Bedeutung": "Nähe zum möglichen Fast-Entry-Fenster; eigenständiger Kalenderfaktor."},
                {"Score": "Taktischer Score", "Wert": latest_tactical_score, "Einordnung": score_label(latest_tactical_score), "Bedeutung": "Gewichtete Gesamtsicht mit Hype-Abschlag. Schwache Kursstruktur wird nicht durch Index-Nähe überdeckt."},
            ]
        )
        st.dataframe(score_table, use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Gewichtung des taktischen Scores**

            - 75 % Einstiegsqualität
            - 25 % Index-Timing
            - zusätzlicher Abschlag, sobald das Hype-Risiko über 55 Punkte steigt

            Ein `ENTRY WATCH` wird nur angezeigt, wenn zusätzlich harte Bedingungen erfüllt sind:
            ausreichende Datenhistorie, Einstiegsqualität von mindestens 70 Punkten, Hype-Risiko von höchstens
            55 Punkten, Kurs oberhalb des IPO-anchored VWAP und EMA9 oberhalb von EMA21.
            """
        )

    with data_tab:
        st.subheader("Reguläre Stundenkerzen und Modellkennzahlen")
        display_columns = [
            "Open", "High", "Low", "Close", "Volume", "IPO_AVWAP", "IPO_PREMIUM", "DRAWDOWN_FROM_HIGH",
            "DISTANCE_TO_AVWAP", "RSI7", "ATR10_PCT", "VOL_RATIO", "UP_DOWN_VOLUME_RATIO", "HYPE_RISK_SCORE",
            "ENTRY_QUALITY_SCORE", "INDEX_TIMING_SCORE", "TACTICAL_SCORE", "TACTICAL_ENTRY_WATCH",
        ]
        display_data = analysis[display_columns].copy()
        display_data.index = display_data.index.tz_convert(NY_TZ)
        display_data.index.name = "Timestamp ET"
        st.dataframe(display_data.tail(100), use_container_width=True)
        st.download_button(
            label="Auswertung als CSV herunterladen",
            data=display_data.to_csv(index=True),
            file_name="spcx_ipo_ndx_watch_hourly.csv",
            mime="text/csv",
            key="download_spcx_data",
        )

    with methodology_tab:
        st.subheader("Methodik")
        st.markdown(
            f"""
            ### Zweck des Modells
            Das Modell bewertet, ob sich die erste emotional geprägte IPO-Preisfindung sichtbar beruhigt und
            ob sich vor einer möglichen Nasdaq-100-Aufnahme ein konstruktiveres taktisches Fenster entwickelt.

            ### Relevante Referenzen
            - IPO-Preis: **{SPCX_IPO_PRICE_USD:.0f} USD**
            - IPO-anchored VWAP: volumengewichteter Durchschnittspreis ab Handelsstart
            - Drawdown vom Post-IPO-Hoch: Maß für den Abbau der ersten Euphorie
            - EMA9 / EMA21, höheres Tief, RSI7 und Aufwärtsvolumen: transparente Stabilisierungssignale
            - ATR10 in Prozent: Maß für die kurzfristige Schwankungsintensität

            ### Nasdaq-100 Fast Entry
            Ein IPO kann beschleunigt aufgenommen werden, wenn seine vollständige Marktkapitalisierung innerhalb
            der Top 40 der aktuellen Indexmitglieder liegt und die übrigen Kriterien erfüllt sind. Die Prüfung erfolgt
            am Ende des 7. Handelstags. Typischerweise folgt eine Ankündigung nach Handelsschluss am 10. Handelstag
            und eine Aufnahme nach 15 Handelstagen.

            ### Modellgrenzen
            Die Score-Grenzen sind transparente heuristische Startwerte und noch nicht statistisch kalibriert.
            Die vollständige Marktkapitalisierung einschließlich nicht börsennotierter Aktienklassen sowie die
            maßgebliche Top-40-Rangfolge lassen sich nicht zuverlässig allein aus yfinance-Daten ableiten.
            """
        )
        st.markdown(
            """
            **Primärquellen**

            - Nasdaq-100 Index Methodology: https://indexes.nasdaq.com/docs/Methodology_NDX.pdf
            - SpaceX IPO Pricing Announcement: https://content.spacex.com/cms-assets/FINAL_Documents%20and%20Updates/SpaceX_PricingAnnouncement.pdf
            """
        )


# ============================================================
# 10. HAUPTNAVIGATION
# ============================================================

def get_query_parameter(name: str, default: str = "") -> str:
    """Liest URL-Parameter in aktuellen und älteren Streamlit-Versionen robust aus."""

    try:
        value = st.query_params.get(name, default)
    except AttributeError:
        value = st.experimental_get_query_params().get(name, [default])

    if isinstance(value, list):
        return str(value[-1]) if value else default

    return str(value)


def build_navigation_href(page: str) -> str:
    return "?" + urlencode({"page": page})


def build_asset_navigation_html(current_page: str) -> str:
    """Erzeugt Kategorien und Assets aus einer zentralen, leicht erweiterbaren Registry."""

    sections: list[str] = []
    for group in ASSET_NAVIGATION:
        sections.append(f'<div class="lab-nav-category">{group["category"]}</div>')
        for item in group["items"]:
            active_class = " lab-nav-item-active" if current_page == item["page"] else ""
            href = build_navigation_href(item["page"])
            sections.append(
                f'<a class="lab-nav-item{active_class}" href="{href}" target="_self">'
                f'<span>{item["icon"]} {item["label"]}</span>'
                f'<span class="lab-nav-symbol">{item["ticker"]}</span>'
                '</a>'
            )
    return "".join(sections)


def render_navigation() -> str:
    """Rendert links oben eine feste Home-Schaltfläche und ein ausklappbares Asset-Menü."""

    valid_pages = {"home"}
    for group in ASSET_NAVIGATION:
        valid_pages.update(item["page"] for item in group["items"])

    requested_page = get_query_parameter("page", "home").lower()
    current_page = requested_page if requested_page in valid_pages else "home"
    home_href = build_navigation_href("home")
    navigation_items = build_asset_navigation_html(current_page)

    navigation_html = (
        f'<nav class="lab-nav-rail" aria-label="Hauptnavigation">'
        f'<a class="lab-nav-icon" href="{home_href}" target="_self" aria-label="Zur Startseite" title="Startseite">'
        '<svg class="lab-nav-home-svg" viewBox="0 0 24 24" aria-hidden="true">'
        '<path d="M3 10.8 12 3l9 7.8"></path><path d="M5.6 9.4V21h12.8V9.4"></path><path d="M9.5 21v-6.4h5V21"></path>'
        '</svg></a>'
        '<details class="lab-nav-details">'
        '<summary class="lab-nav-icon" aria-label="Navigation öffnen oder schließen" title="Navigation">'
        '<span class="lab-nav-menu-lines" aria-hidden="true"><span></span><span></span><span></span></span>'
        '</summary>'
        '<aside class="lab-nav-drawer" aria-label="Asset-Navigation">'
        '<div class="lab-nav-drawer-head"><span class="lab-nav-drawer-brand">◈ Market Signal Lab</span></div>'
        f'{navigation_items}'
        '</aside></details></nav>'
    )

    if hasattr(st, "html"):
        st.html(navigation_html)
    else:
        st.markdown(navigation_html, unsafe_allow_html=True)

    return current_page

active_page = render_navigation()

if active_page == "bitcoin":
    render_bitcoin_page()
elif active_page == "spacex":
    render_spacex_page()
else:
    render_landing_page()
