"""
OneEye Pro v5.1
Advanced forecast dashboard for Le Morne / One Eye – Mauritius

NEU in v5.1 (Genauigkeit + Live-Ansicht):
- Wind & Swell jetzt STÜNDLICH abgerufen (statt nur Tages-Max/-Dominant)
  → Score wird je Stunde berechnet, der Tagesscore ist der Mittelwert der
    surfbaren Tagesstunden (07–19 Uhr). Das eliminiert die größte
    Ungenauigkeit von v5 (ein einzelner Tagesmax-Wert verschleiert, wann am
    Tag tatsächlich gute Bedingungen herrschen).
- Persönlicher Alarm wird jetzt STÜNDLICH geprüft statt nur 1x/Tag
  → zeigt exakte Uhrzeit-Fenster statt nur "Tag ja/nein"
- Neuer Tab "🔴 Live Wind & Tide": gemeinsamer stündlicher Chart,
  Zeitachse von JETZT (links) bis +7 Tage (rechts), zwei Y-Achsen
  (Wind in kn, Tide in m). Cache TTL auf 30 Min gesetzt, sodass die
  Daten bei jedem Reload/Auto-Refresh aktuell sind.
- Confidence-Berechnung berücksichtigt jetzt auch die stündliche
  Streuung innerhalb eines Tages (hohe Varianz = geringere Konfidenz)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

LAT = -20.45
LON = 57.317

st.set_page_config(
    page_title="OneEye Pro v5",
    page_icon="🏄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .epic-badge { font-size: 1.3em; }
    .tide-good  { color: #00cc66; font-weight: bold; }
    .tide-bad   { color: #cc3333; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# API-ABRUF
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800)
def fetch_weather_hourly() -> pd.DataFrame:
    """
    Stündliche Wind-/Druckdaten statt nur Tages-Max-Werte.
    Das ist die Hauptverbesserung für die Vorhersagegenauigkeit in v5.1:
    Tages-Max verschleiert, WANN am Tag der gute Wind tatsächlich weht.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=wind_speed_10m,wind_gusts_10m,wind_direction_10m,pressure_msl"
        "&forecast_days=7&timezone=auto"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame({
        "time":      pd.to_datetime(data["time"]),
        "wind_kmh":  data["wind_speed_10m"],
        "gust_kmh":  data["wind_gusts_10m"],
        "wind_dir":  data["wind_direction_10m"],
        "pressure":  data["pressure_msl"],
    })
    df["wind_kn"] = (df["wind_kmh"] * 0.54).round(1)
    df["gust_kn"] = (df["gust_kmh"] * 0.54).round(1)
    return df

@st.cache_data(ttl=1800)
def fetch_marine_hourly() -> pd.DataFrame:
    """Stündliche Swell-Daten statt nur Tages-Max/-Dominant."""
    url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=wave_height,wave_period,wave_direction"
        "&forecast_days=7&timezone=auto"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()["hourly"]
    return pd.DataFrame({
        "time":      pd.to_datetime(data["time"]),
        "wave_m":    data["wave_height"],
        "period":    data["wave_period"],
        "wave_dir":  data["wave_direction"],
    })

@st.cache_data(ttl=1800)
def fetch_tides() -> pd.DataFrame:
    """
    Stündliche Meereshöhe inkl. Tiden.
    Versucht zuerst meteofrance_ocean (sea_level_height_msl),
    Fallback: astronomische Näherung via Sinuskurve.
    """
    # Versuch 1: meteofrance_ocean mit korrektem Parameternamen
    urls_to_try = [
        (
            f"https://marine-api.open-meteo.com/v1/marine"
            f"?latitude={LAT}&longitude={LON}"
            "&hourly=sea_level_height_msl"
            "&models=meteofrance_ocean"
            "&forecast_days=7&timezone=auto",
            "sea_level_height_msl"
        ),
        # Fallback: anderer Modellname
        (
            f"https://marine-api.open-meteo.com/v1/marine"
            f"?latitude={LAT}&longitude={LON}"
            "&hourly=sea_level_height_msl"
            "&forecast_days=7&timezone=auto",
            "sea_level_height_msl"
        ),
    ]
    for url, field in urls_to_try:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if "hourly" in data and field in data["hourly"]:
                    times  = pd.to_datetime(data["hourly"]["time"])
                    levels = data["hourly"][field]
                    df = pd.DataFrame({"time": times, "level": levels}).dropna()
                    if not df.empty:
                        return df
        except Exception:
            continue

    # Fallback: astronomische Näherung (Mauritius: semidiurnal, ~0.4m Amplitude)
    # Mauritius hat einen kleinen Tidenhub (~0.8m) mit 2 HW/NW pro Tag
    st.caption("⚠️ Tide-API nicht verfügbar – astronomische Näherung wird verwendet.")
    now   = pd.Timestamp.now().normalize()
    times = pd.date_range(now, periods=7*24, freq="h")
    # Semidiurnale Tide: Periode ~12.42h, Amplitude ~0.4m für Mauritius
    t_h   = np.arange(len(times), dtype=float)
    level = (
        0.38 * np.sin(2 * np.pi * t_h / 12.42 + 0.5) +   # M2 Hauptkomponente
        0.12 * np.sin(2 * np.pi * t_h / 12.00 + 1.2) +   # S2
        0.05 * np.sin(2 * np.pi * t_h / 23.93 + 0.3)     # K1 diurnal
    )
    return pd.DataFrame({"time": times, "level": level})

# ═══════════════════════════════════════════════════════════════════════════════
# TIDE-VERARBEITUNG
# ═══════════════════════════════════════════════════════════════════════════════
def parse_tides(df: pd.DataFrame) -> pd.DataFrame:
    """fetch_tides() gibt jetzt direkt ein DataFrame zurück."""
    return df

def find_extrema(df_tide: pd.DataFrame) -> pd.DataFrame:
    """Findet lokale Maxima (High) und Minima (Low)."""
    levels = df_tide["level"].values
    # Fenster = 3h → typisch für Tide-Extrema-Abstand von ~6h
    hi_idx = argrelextrema(levels, np.greater_equal, order=3)[0]
    lo_idx = argrelextrema(levels, np.less_equal,    order=3)[0]

    highs = df_tide.iloc[hi_idx][["time","level"]].copy()
    highs["type"] = "HW"
    lows  = df_tide.iloc[lo_idx][["time","level"]].copy()
    lows["type"]  = "NW"

    return pd.concat([highs, lows]).sort_values("time").reset_index(drop=True)

def tide_status_at(level: float, day_min: float, day_max: float) -> tuple[str, str]:
    """
    Gibt (Status-Label, CSS-Klasse) für aktuellen Pegelstand.
    One Eye bricht optimal bei Mid–High Tide.
    """
    if day_max == day_min:
        return "Unbekannt", ""
    pct = (level - day_min) / (day_max - day_min) * 100
    if pct >= 55:
        return f"🟢 High ({pct:.0f}%)", "tide-good"
    if pct >= 35:
        return f"🟡 Mid ({pct:.0f}%)", ""
    return f"🔴 Low ({pct:.0f}%)", "tide-bad"

def best_tide_windows(df_tide_day: pd.DataFrame) -> list[str]:
    """Gibt Liste der Stunden zurück, die in gutem Tidefenster liegen."""
    mn = df_tide_day["level"].min()
    mx = df_tide_day["level"].max()
    if mx == mn:
        return []
    good = df_tide_day[
        (df_tide_day["level"] - mn) / (mx - mn) >= 0.40
    ]
    windows = []
    if not good.empty:
        start = good["time"].iloc[0].strftime("%H:%M")
        end   = good["time"].iloc[-1].strftime("%H:%M")
        windows.append(f"{start}–{end}")
    return windows

# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════
def wind_score(kn, direction):
    s = 0
    if kn >= 22:    s += 40
    elif kn >= 18:  s += 30
    elif kn >= 15:  s += 20
    elif kn >= 12:  s += 10
    else:           s -= 10
    if   110 <= direction <= 150: s += 20
    elif 100 <= direction <= 160: s += 15
    elif 90  <= direction <= 170: s += 10
    else:                         s -= 5
    label = ("💨 Perfekt" if s >= 50 else "✅ Gut" if s >= 35
             else "🟡 Mäßig" if s >= 20 else "❌ Schwach")
    return int(max(0, min(50, s))), label

def swell_score(h, p, d):
    s = 0
    if   2.0 <= h <= 3.5: s += 25
    elif 1.5 <= h < 2.0:  s += 18
    elif 1.0 <= h < 1.5:  s += 10
    elif h > 3.5:          s += 10
    if   p >= 14: s += 20
    elif p >= 12: s += 12
    elif p >= 10: s +=  6
    if   190 <= d <= 250: s += 10
    elif 160 <= d <= 270: s +=  5
    label = ("🌊 Epic Swell" if s >= 45 else "✅ Gut" if s >= 30
             else "🟡 Schwach" if s >= 15 else "❌ Kein Swell")
    return int(max(0, min(50, s))), label

def pressure_score(p):
    if p >= 1024: return 10
    if p >= 1020: return  8
    if p >= 1016: return  5
    if p >= 1012: return  2
    return 0

# ═══════════════════════════════════════════════════════════════════════════════
# STÜNDLICHE AGGREGATION (v5.1 – Kern der Genauigkeitsverbesserung)
# ═══════════════════════════════════════════════════════════════════════════════
DAYTIME_START, DAYTIME_END = 7, 19  # Tagesstunden, in denen man tatsächlich kitet

def build_hourly_master(df_wind: pd.DataFrame, df_marine: pd.DataFrame,
                         df_tide: pd.DataFrame) -> pd.DataFrame:
    """
    Führt Wind-, Swell- und Tide-Daten auf gemeinsamer Stundenbasis zusammen
    und berechnet je Stunde Wind-/Swell-/Tide-Score sowie den persönlichen
    Alarm. Das ist die neue Datengrundlage für Tagesscores UND den
    Live-Chart.
    """
    df = df_wind.merge(df_marine, on="time", how="inner")
    df = df.merge(df_tide.rename(columns={"level": "tide_m"}), on="time", how="left")
    df["tide_m"] = df["tide_m"].interpolate(limit_direction="both")

    # Rollierende Tide-Steigung (auflaufend/ablaufend) je Stunde
    df["tide_rising"] = df["tide_m"].diff().fillna(0) > 0

    ws_list, wl_list, ss_list, sl_list = [], [], [], []
    for _, r in df.iterrows():
        ws, wl = wind_score(r["wind_kn"], r["wind_dir"])
        ss, sl = swell_score(r["wave_m"], r["period"], r["wave_dir"])
        ws_list.append(ws); wl_list.append(wl)
        ss_list.append(ss); sl_list.append(sl)
    df["wind_score"]  = ws_list
    df["wind_label"]  = wl_list
    df["swell_score"] = ss_list
    df["swell_label"] = sl_list
    df["pressure_score"] = df["pressure"].apply(pressure_score)

    # Tide-Score je Stunde: relative Tagesposition (0–100%) je Kalendertag
    df["date"] = df["time"].dt.date
    def _tide_pct(g):
        mn, mx = g["tide_m"].min(), g["tide_m"].max()
        if mx == mn:
            return pd.Series(0.0, index=g.index)
        return (g["tide_m"] - mn) / (mx - mn) * 100
    df["tide_pct"] = df.groupby("date", group_keys=False).apply(_tide_pct)
    df["tide_score"] = df["tide_pct"].apply(
        lambda p: 5 if p >= 55 else (2 if p >= 35 else -3)
    )

    df["hour_score"] = (df["wind_score"] + df["swell_score"]
                         + df["pressure_score"] + df["tide_score"])

    # Persönlicher Alarm je Stunde (präzises Zeitfenster statt Tagesflag)
    alarm_list = []
    for _, r in df.iterrows():
        alarm_list.append(personal_alarm(
            r["wind_kn"], r["wind_dir"], r["wave_m"], r["period"],
            bool(r["tide_rising"]) or r["tide_pct"] >= 60
        ))
    df["alarm"] = alarm_list
    return df

def daily_from_hourly(df_hourly: pd.DataFrame, season_factor: int) -> pd.DataFrame:
    """
    Aggregiert die stündlichen Scores zu Tageswerten. Statt eines einzelnen
    Tages-Max-Werts (v5) fließt jetzt der Mittelwert der Tagesstunden
    (07–19 Uhr) ein – das spiegelt die tatsächlich kitebaren Bedingungen
    wider, nicht nur den theoretischen Tagesspitzenwert.
    """
    rows = []
    for date_val, g in df_hourly.groupby("date"):
        day_g = g[(g["time"].dt.hour >= DAYTIME_START) &
                  (g["time"].dt.hour <= DAYTIME_END)]
        if day_g.empty:
            day_g = g  # Fallback falls keine Tagesstunden vorhanden

        # Repräsentative Stunde = die mit dem höchsten Stunden-Score
        best_hour = day_g.loc[day_g["hour_score"].idxmax()]

        wind_kn_mean  = round(day_g["wind_kn"].mean(), 1)
        gust_kn_mean  = round(day_g["gust_kn"].mean(), 1)
        wave_m_mean   = round(day_g["wave_m"].mean(), 2)
        period_mean   = round(day_g["period"].mean(), 1)

        total = (day_g["wind_score"].mean() + day_g["swell_score"].mean()
                 + day_g["pressure_score"].mean() + day_g["tide_score"].mean()
                 + season_factor)
        total = max(0, min(100, total))

        # Konfidenz: je geringer die Streuung des Stunden-Scores am Tag,
        # desto verlässlicher die Vorhersage für diesen Tag
        score_std = day_g["hour_score"].std()
        score_std = 0 if pd.isna(score_std) else score_std
        variance_penalty = min(25, score_std * 1.2)

        alarm_hours = day_g[day_g["alarm"] == "🔔 PERSÖNLICHER ALARM"]["time"]
        if not alarm_hours.empty:
            alarm = "🔔 PERSÖNLICHER ALARM"
            alarm_window = f"{alarm_hours.dt.strftime('%H:%M').iloc[0]}–{alarm_hours.dt.strftime('%H:%M').iloc[-1]}"
        else:
            near = day_g[day_g["alarm"].str.startswith("⚡", na=False)]
            alarm = near["alarm"].iloc[0] if not near.empty else ""
            alarm_window = ""

        rows.append({
            "Date": str(date_val),
            "Wind_kn": wind_kn_mean,
            "Gust_kn": gust_kn_mean,
            "WindDir": round(best_hour["wind_dir"]),
            "Pressure": round(day_g["pressure"].mean(), 1),
            "Wave_m": wave_m_mean,
            "Period": period_mean,
            "WaveDir": round(best_hour["wave_dir"]),
            "WindScore": int(round(day_g["wind_score"].mean())),
            "WindLabel": best_hour["wind_label"],
            "SwellScore": int(round(day_g["swell_score"].mean())),
            "SwellLabel": best_hour["swell_label"],
            "TideScore": int(round(day_g["tide_score"].mean())),
            "Score": round(total, 1),
            "Epic": epic_label(best_hour["wind_kn"], best_hour["wave_m"], best_hour["period"]),
            "Alarm": alarm,
            "AlarmWindow": alarm_window,
            "BestHour": best_hour["time"].strftime("%H:%M"),
            "Confidence": max(30, round(95 - variance_penalty)),
        })
    return pd.DataFrame(rows)

def tide_score(df_tide_day: pd.DataFrame) -> int:
    """
    +5  wenn Hochwasser tagsüber (8–18 Uhr) liegt → One Eye gut surfbar
    -3  wenn Niedrigwasser tagsüber dominiert
    """
    if df_tide_day.empty:
        return 0
    daytime = df_tide_day[
        (df_tide_day["time"].dt.hour >= 8) &
        (df_tide_day["time"].dt.hour <= 18)
    ]
    if daytime.empty:
        return 0
    mn = df_tide_day["level"].min()
    mx = df_tide_day["level"].max()
    if mx == mn:
        return 0
    avg_pct = (daytime["level"].mean() - mn) / (mx - mn)
    if avg_pct >= 0.55: return 5
    if avg_pct >= 0.35: return 2
    return -3

def confidence_pct(i):
    return max(30, 100 - i * 10)

def score_color(s):
    if s >= 75: return "#00cc66"
    if s >= 55: return "#ffcc00"
    if s >= 35: return "#ff8800"
    return "#cc3333"

def personal_alarm(kn: float, direction: float, h: float, period: float,
                   tide_rising: bool) -> str:
    """
    Persönliche One-Eye-Formel (konjunktiv – ALLE Bedingungen müssen erfüllt sein):
      Wind:    18–28 kn
      Richtung: ESE–SE (110°–150°)
      Swell:   1.8–3.0 m
      Periode: >14 s
      Tide:    auflaufend oder Hochwasser
    Gibt einen Alarm-String zurück oder "".
    """
    wind_ok    = 18.0 <= kn <= 28.0
    dir_ok     = 110  <= direction <= 150
    swell_ok   = 1.8  <= h  <= 3.0
    period_ok  = period > 14.0
    tide_ok    = tide_rising

    if wind_ok and dir_ok and swell_ok and period_ok and tide_ok:
        return "🔔 PERSÖNLICHER ALARM"

    # Zeige wie viele Bedingungen erfüllt sind (z.B. "4/5")
    met = sum([wind_ok, dir_ok, swell_ok, period_ok, tide_ok])
    if met >= 4:
        missing = []
        if not wind_ok:   missing.append(f"Wind {kn:.0f} kn (18–28)")
        if not dir_ok:    missing.append(f"Dir {direction:.0f}° (110–150)")
        if not swell_ok:  missing.append(f"Swell {h:.1f} m (1.8–3.0)")
        if not period_ok: missing.append(f"Periode {period:.0f}s (>14)")
        if not tide_ok:   missing.append("Tide nicht auflaufend")
        return f"⚡ Fast! Fehlt: {', '.join(missing)}"
    return ""

def is_tide_rising(df_tide_day: pd.DataFrame) -> bool:
    """
    Prüft ob die Tide tagsüber (8–16 Uhr) im Schnitt auflaufend ist
    oder ob der Tagesdurchschnitt >60% des Tagesbereichs liegt (Hochwasser-Nähe).
    """
    if df_tide_day.empty:
        return False
    daytime = df_tide_day[
        (df_tide_day["time"].dt.hour >= 8) &
        (df_tide_day["time"].dt.hour <= 16)
    ].copy()
    if len(daytime) < 2:
        return False
    mn = df_tide_day["level"].min()
    mx = df_tide_day["level"].max()
    if mx == mn:
        return False
    # Steigend wenn letzter Wert > erster Wert im Zeitfenster ODER Pegel hoch
    first = daytime["level"].iloc[0]
    last  = daytime["level"].iloc[-1]
    avg_pct = (daytime["level"].mean() - mn) / (mx - mn)
    return (last > first) or (avg_pct >= 0.60)

def epic_label(kn, h, p):
    """Additiver Score-basierter Epic-Indikator (ergänzt persönlichen Alarm)."""
    if kn >= 20 and h >= 2.0 and p >= 14: return "⭐⭐⭐ ONCE IN A TRIP"
    if kn >= 18 and h >= 1.5 and p >= 13: return "⭐⭐ EPIC+"
    if kn >= 16 and h >= 1.2 and p >= 12: return "⭐ EPIC"
    return ""

# ═══════════════════════════════════════════════════════════════════════════════
# TIDENGRAFIK für einen Tag
# ═══════════════════════════════════════════════════════════════════════════════
def tide_chart(df_day: pd.DataFrame, extrema_day: pd.DataFrame, date_str: str) -> go.Figure:
    mn = df_day["level"].min()
    mx = df_day["level"].max()
    rang = mx - mn if mx != mn else 1

    # Farbige Hintergrundzone: grün = Mid–High (gut für One Eye)
    good_threshold = mn + rang * 0.40

    fig = go.Figure()

    # Gefüllte Fläche unter der Kurve
    fig.add_trace(go.Scatter(
        x=df_day["time"], y=df_day["level"],
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.18)",
        line=dict(color="#4fc3f7", width=2.5),
        name="Pegelstand",
        hovertemplate="%{x|%H:%M} · %{y:.2f} m<extra></extra>",
    ))

    # "Gut für One Eye"-Zone als gestrichelte Linie
    fig.add_hline(
        y=good_threshold,
        line_dash="dot", line_color="#00cc66", line_width=1.5,
        annotation_text="⬆ Gut für One Eye (Mid–High)",
        annotation_position="top left",
        annotation_font_color="#00cc66",
    )

    # Grüner Bereich (Füllung über Schwellenwert) – als separate Fläche
    fig.add_trace(go.Scatter(
        x=df_day["time"],
        y=[max(v, good_threshold) for v in df_day["level"]],
        fill="tonexty",
        fillcolor="rgba(0,204,102,0.15)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # High/Low Marker
    for _, ex in extrema_day.iterrows():
        is_hw = ex["type"] == "HW"
        fig.add_trace(go.Scatter(
            x=[ex["time"]], y=[ex["level"]],
            mode="markers+text",
            marker=dict(
                size=13,
                color="#00cc66" if is_hw else "#cc3333",
                symbol="triangle-up" if is_hw else "triangle-down",
                line=dict(color="white", width=1.5),
            ),
            text=[f"{'HW' if is_hw else 'NW'}<br>{ex['level']:.2f}m"],
            textposition="top center" if is_hw else "bottom center",
            textfont=dict(size=10, color="#00cc66" if is_hw else "#cc3333"),
            showlegend=False,
            hovertemplate=(
                f"{'Hochwasser' if is_hw else 'Niedrigwasser'}<br>"
                f"{ex['time'].strftime('%H:%M')} · {ex['level']:.2f} m<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"Tidegang {date_str}",
        xaxis=dict(
            title="Uhrzeit (Ortszeit Mauritius)",
            tickformat="%H:%M",
        ),
        yaxis=dict(title="Meereshöhe (m, MSL)", tickformat=".2f"),
        height=260,
        margin=dict(l=50, r=20, t=45, b=40),
        showlegend=False,
        hovermode="x unified",
    )
    return fig

def live_wind_tide_chart(df_hourly: pd.DataFrame, df_tide_full: pd.DataFrame) -> go.Figure:
    """
    Gemeinsamer stündlicher Chart: Wind (kn, Balken) + Tide (m, Linie)
    von JETZT (links) bis +7 Tage (rechts), zwei Y-Achsen.
    """
    now = pd.Timestamp.now()
    df_future = df_hourly[df_hourly["time"] >= now - pd.Timedelta(hours=1)].copy()

    fig = go.Figure()

    # Wind als Balken (linke Y-Achse)
    bar_colors = ["#00cc66" if k >= 18 else "#4fc3f7" if k >= 12 else "#90a4ae"
                  for k in df_future["wind_kn"]]
    fig.add_trace(go.Bar(
        x=df_future["time"], y=df_future["wind_kn"],
        name="Wind (kn)",
        marker_color=bar_colors,
        opacity=0.75,
        yaxis="y1",
        hovertemplate="%{x|%a %d.%m %H:%M}<br>Wind: %{y:.1f} kn<extra></extra>",
    ))
    # Böen als dünne Linie obendrüber
    fig.add_trace(go.Scatter(
        x=df_future["time"], y=df_future["gust_kn"],
        name="Böen (kn)", mode="lines",
        line=dict(color="#ff8800", width=1.2, dash="dot"),
        yaxis="y1",
        hovertemplate="%{x|%a %d.%m %H:%M}<br>Böen: %{y:.1f} kn<extra></extra>",
    ))

    # Tide als Linie (rechte Y-Achse)
    fig.add_trace(go.Scatter(
        x=df_future["time"], y=df_future["tide_m"],
        name="Tide (m)", mode="lines",
        line=dict(color="#ab47bc", width=2.5),
        yaxis="y2",
        hovertemplate="%{x|%a %d.%m %H:%M}<br>Tide: %{y:.2f} m<extra></extra>",
    ))

    # Vertikale Linie für "jetzt"
    fig.add_vline(
        x=now, line_dash="dash", line_color="#ffffff",
        annotation_text="Jetzt", annotation_position="top",
    )

    fig.update_layout(
        title="Live Wind & Tide – stündlich, JETZT → +7 Tage",
        xaxis=dict(title="Zeit (Ortszeit Mauritius)", tickformat="%a %H:%M"),
        yaxis=dict(title="Wind (kn)", side="left", range=[0, max(35, df_future["gust_kn"].max() * 1.1)]),
        yaxis2=dict(title="Tide (m)", side="right", overlaying="y",
                     range=[df_future["tide_m"].min() - 0.2, df_future["tide_m"].max() + 0.2]),
        height=440,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=60, b=40),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🌍 Season Factors")
    enso = st.slider("ENSO", -5, 5, 0, help="Positiv = El Niño")
    iod  = st.slider("IOD",  -5, 5, 0, help="Positiver IOD → mehr SW-Swell")
    sam  = st.slider("SAM",  -5, 5, 0, help="Positiver SAM → mehr Swell")
    season_factor = enso + iod + sam

    st.divider()
    st.header("ℹ️ Spot-Info")
    st.markdown("""
    **One Eye** – Weltklasse-Kite-Spot, SW-Küste Mauritius

    **Ideal:**
    - Wind: SE 18–25 kn
    - Swell: SW 1.5–3m, >12s
    - **Tide: Mid–High** ⬅ neu!
    - Druck: >1018 hPa

    ⚠️ Korallenriff – nur Fortgeschrittene!
    """)

    if st.button("🔄 Neu laden"):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# HAUPTBEREICH
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏄 OneEye Pro v5.1 — Le Morne / Mauritius")
st.caption(f"Forecast {LAT}°S, {LON}°E · Live-Update alle 30 Min · {datetime.now().strftime('%d.%m.%Y %H:%M')}")

try:
    w_hourly = fetch_weather_hourly()
    m_hourly = fetch_marine_hourly()
    t        = fetch_tides()

    # ── Tide DataFrame aufbauen ──────────────────────────────────────────────
    df_tide_full = parse_tides(t)   # t ist bereits ein DataFrame
    df_tide_full = df_tide_full.dropna(subset=["level"])
    extrema_all  = find_extrema(df_tide_full)

    # ── Stündliche Master-Tabelle (Wind + Swell + Tide + Scores) ────────────
    df_hourly = build_hourly_master(w_hourly, m_hourly, df_tide_full)

    # ── Tages-Aggregation aus stündlichen Daten (v5.1 Kernverbesserung) ──────
    df = daily_from_hourly(df_hourly, season_factor)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    best_day   = df.loc[df["Score"].idxmax()]
    alarm_days = int((df["Alarm"] == "🔔 PERSÖNLICHER ALARM").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏆 Best Score",     int(df["Score"].max()))
    c2.metric("📊 Ø Score",        round(df["Score"].mean(), 1))
    c3.metric("⭐ Epic Days",      int((df["Epic"] != "").sum()))
    c4.metric("🔔 Persönl. Alarm", alarm_days,
              delta="🔥 GO!" if alarm_days > 0 else None)
    c5.metric("📅 Bester Tag",     best_day["Date"])

    # Prominenter Banner bei persönlichem Alarm
    alarm_rows = df[df["Alarm"] == "🔔 PERSÖNLICHER ALARM"]
    if not alarm_rows.empty:
        for _, ar in alarm_rows.iterrows():
            d, kn, wm, per = ar["Date"], ar["Wind_kn"], ar["Wave_m"], ar["Period"]
            fenster = f" · Fenster: {ar['AlarmWindow']}" if ar["AlarmWindow"] else ""
            st.error(f"🔔 **PERSÖNLICHER ALARM: {d}**{fenster} — Wind {kn} kn · Swell {wm} m @ {per} s · Tide auflaufend/HW ✅")

    st.divider()

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab_live, tab2, tab3 = st.tabs(
        ["📊 Forecast & Score", "🔴 Live Wind & Tide", "🌊 Tidenkalender", "📋 Tagesdetails"]
    )

    # ════════════════════════════════════════
    # TAB 1: Score-Charts
    # ════════════════════════════════════════
    with tab1:
        colors = [score_color(s) for s in df["Score"]]
        fig = go.Figure(go.Bar(
            x=df["Date"], y=df["Score"],
            marker_color=colors,
            text=df["Score"].astype(int),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>",
        ))
        fig.update_layout(
            title="One Eye Score (7 Tage)",
            yaxis=dict(range=[0, 115], title="Score"),
            xaxis_title="Datum", showlegend=False, height=350,
        )
        fig.add_hline(y=75, line_dash="dot", line_color="#00cc66",
                      annotation_text="Epic (75)", annotation_position="right")
        fig.add_hline(y=55, line_dash="dot", line_color="#ffcc00",
                      annotation_text="Gut (55)",  annotation_position="right")
        st.plotly_chart(fig, use_container_width=True, key="score_bar")

        # Stacked Breakdown inkl. Tide-Score
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Wind",  x=df["Date"], y=df["WindScore"],
                              marker_color="#4fc3f7"))
        fig2.add_trace(go.Bar(name="Swell", x=df["Date"], y=df["SwellScore"],
                              marker_color="#26a69a"))
        fig2.add_trace(go.Bar(name="Druck", x=df["Date"],
                              y=[pressure_score(p) for p in df["Pressure"]],
                              marker_color="#ab47bc"))
        fig2.add_trace(go.Bar(name="Tide",  x=df["Date"], y=df["TideScore"],
                              marker_color="#ff8f00"))
        fig2.update_layout(
            barmode="stack",
            title="Score-Breakdown (Wind / Swell / Druck / Tide)",
            yaxis=dict(range=[-5, 115], title="Punkte"),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True, key="score_breakdown")

        # ── Persönlicher Alarm (konjunktiv) ─────────────────────────────
        st.subheader("🔔 Persönlicher Alarm (One-Eye-Formel)")
        st.caption("Wind 18–28 kn · ESE–SE (110–150°) · Swell 1.8–3.0m · Periode >14s · Tide auflaufend/HW")
        palarms = df[df["Alarm"] != ""]
        if not palarms.empty:
            for _, r in palarms.iterrows():
                d, kn, wm, per, al = r["Date"], r["Wind_kn"], r["Wave_m"], r["Period"], r["Alarm"]
                if al == "🔔 PERSÖNLICHER ALARM":
                    st.error(f"**{d}** · {al} · Wind {kn} kn · Swell {wm} m @ {per} s")
                else:
                    st.warning(f"**{d}** · {al}")
        else:
            st.info("Keine Alarm-Bedingungen in den nächsten 7 Tagen.")

        st.divider()

        # ── Score-Alerts ──────────────────────────────────────────────────
        st.subheader("🚨 Score-Alerts (Score ≥ 75)")
        alerts = df[df["Score"] >= 75]
        if len(alerts):
            for _, r in alerts.iterrows():
                d, sc, kn, wm, per, ep = r["Date"], int(r["Score"]), r["Wind_kn"], r["Wave_m"], r["Period"], r["Epic"]
                st.success(f"**{d}** · Score {sc} · Wind {kn} kn · Swell {wm} m @ {per} s · {ep}")
        else:
            st.info("Keine Score-Alerts in den nächsten 7 Tagen (Score < 75)")

    # ════════════════════════════════════════
    # TAB LIVE: Wind & Tide stündlich, jetzt → +7 Tage
    # ════════════════════════════════════════
    with tab_live:
        st.subheader("🔴 Live Wind & Tide – stündlich aktualisiert")
        st.caption(
            "Daten aktualisieren sich automatisch alle 30 Minuten (Cache-TTL). "
            "Zeitachse läuft von **jetzt** (links) bis **+7 Tage** (rechts)."
        )

        now_row = df_hourly.iloc[
            (df_hourly["time"] - pd.Timestamp.now()).abs().idxmin()
        ]
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("💨 Wind jetzt", f"{now_row['wind_kn']:.1f} kn",
                   f"Böen {now_row['gust_kn']:.1f} kn")
        lc2.metric("🧭 Richtung", f"{int(now_row['wind_dir'])}°")
        lc3.metric("🌊 Tide jetzt", f"{now_row['tide_m']:.2f} m",
                   "↗ steigend" if now_row["tide_rising"] else "↘ fallend")
        lc4.metric("📊 Stunden-Score", f"{int(now_row['hour_score'])}")

        st.plotly_chart(
            live_wind_tide_chart(df_hourly, df_tide_full),
            use_container_width=True,
            key="live_wind_tide",
        )

        with st.expander("🕐 Stündliche Rohdaten (nächste 48h)"):
            disp = df_hourly[df_hourly["time"] >= pd.Timestamp.now() - pd.Timedelta(hours=1)].head(48).copy()
            disp["Zeit"] = disp["time"].dt.strftime("%a %d.%m %H:%M")
            disp["Wind (kn)"] = disp["wind_kn"]
            disp["Böen (kn)"] = disp["gust_kn"]
            disp["Tide (m)"] = disp["tide_m"].round(2)
            disp["Score"] = disp["hour_score"].round(0).astype(int)
            st.dataframe(
                disp[["Zeit", "Wind (kn)", "Böen (kn)", "Tide (m)", "Score"]],
                use_container_width=True, hide_index=True,
            )

    # ════════════════════════════════════════
    # TAB 2: TIDENKALENDER
    # ════════════════════════════════════════
    with tab2:
        st.subheader("🌊 Tidenkalender – 7 Tage")
        st.caption(
            "Daten: Open-Meteo Marine API · MeteoFrance SMOC · "
            "8 km Auflösung · Referenz: Global Mean Sea Level (MSL)\n\n"
            "⚠️ Nur zur Orientierung – nicht für Navigation geeignet!"
        )

        # Kompakte 7-Tage-Übersicht (Tabelle der HW/NW-Zeiten)
        st.markdown("#### 📅 High & Low Tide Übersicht")
        summary_rows = []
        for date_str in df["Date"]:
            day_date = pd.Timestamp(date_str).date()
            ex_day = extrema_all[extrema_all["time"].dt.date == day_date].copy()
            for _, ex in ex_day.iterrows():
                summary_rows.append({
                    "Datum": date_str,
                    "Uhrzeit": ex["time"].strftime("%H:%M"),
                    "Typ": "🔼 Hochwasser" if ex["type"] == "HW" else "🔽 Niedrigwasser",
                    "Höhe (m)": round(ex["level"], 2),
                })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        else:
            st.warning("Keine Extremwerte berechnet – Tidendaten möglicherweise nicht verfügbar.")

        st.divider()

        # Tidengrafik: Selector für Tag
        st.markdown("#### 📈 Tidengrafik für einen Tag")
        selected_date = st.selectbox(
            "Tag auswählen:",
            options=df["Date"].tolist(),
            index=0,
        )

        day_date = pd.Timestamp(selected_date).date()
        df_day  = df_tide_full[df_tide_full["time"].dt.date == day_date].copy()
        ex_day  = extrema_all[extrema_all["time"].dt.date == day_date].copy()

        if not df_day.empty:
            # Beste Tide-Fenster
            windows = best_tide_windows(df_day)
            mn = df_day["level"].min()
            mx = df_day["level"].max()

            # Zeiten als saubere Strings extrahieren
            if not ex_day.empty:
                hw_times = ", ".join(
                    ex_day[ex_day["type"]=="HW"]["time"].dt.strftime("%H:%M").tolist()
                ) or "–"
                nw_times = ", ".join(
                    ex_day[ex_day["type"]=="NW"]["time"].dt.strftime("%H:%M").tolist()
                ) or "–"
            else:
                hw_times = "–"
                nw_times = "–"

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🔼 Hochwasser",    f"{mx:.2f} m", hw_times)
            col_b.metric("🔽 Niedrigwasser", f"{mn:.2f} m", nw_times)
            col_c.metric("✅ Gut für One Eye", windows[0] if windows else "Kaum")

            st.plotly_chart(
                tide_chart(df_day, ex_day, selected_date),
                use_container_width=True,
                key=f"tide_selected_{selected_date}",
            )

            # Stündliche Tabelle
            with st.expander("🕐 Stündliche Tide-Daten anzeigen"):
                df_day_disp = df_day.copy()
                df_day_disp["Uhrzeit"]   = df_day_disp["time"].dt.strftime("%H:%M")
                df_day_disp["Höhe (m)"]  = df_day_disp["level"].round(3)
                df_day_disp["Status"]    = df_day_disp["level"].apply(
                    lambda v: tide_status_at(v, mn, mx)[0]
                )
                st.dataframe(
                    df_day_disp[["Uhrzeit","Höhe (m)","Status"]],
                    use_container_width=True, hide_index=True,
                )
        else:
            st.warning("Keine stündlichen Tidendaten für diesen Tag verfügbar.")

        # 7-Tage-Tidegang Übersicht (alle Tage in einem Chart)
        st.divider()
        st.markdown("#### 📊 7-Tage-Tidegang")
        fig_all = go.Figure()
        fig_all.add_trace(go.Scatter(
            x=df_tide_full["time"],
            y=df_tide_full["level"],
            fill="tozeroy",
            fillcolor="rgba(79,195,247,0.15)",
            line=dict(color="#4fc3f7", width=1.5),
            name="Pegelstand",
            hovertemplate="%{x|%d.%m %H:%M} · %{y:.2f} m<extra></extra>",
        ))
        # HW/NW Marker für gesamte Woche
        hw_all = extrema_all[extrema_all["type"]=="HW"]
        nw_all = extrema_all[extrema_all["type"]=="NW"]
        fig_all.add_trace(go.Scatter(
            x=hw_all["time"], y=hw_all["level"],
            mode="markers",
            marker=dict(size=9, color="#00cc66", symbol="triangle-up"),
            name="Hochwasser",
            hovertemplate="%{x|%d.%m %H:%M}<br>HW: %{y:.2f} m<extra></extra>",
        ))
        fig_all.add_trace(go.Scatter(
            x=nw_all["time"], y=nw_all["level"],
            mode="markers",
            marker=dict(size=9, color="#cc3333", symbol="triangle-down"),
            name="Niedrigwasser",
            hovertemplate="%{x|%d.%m %H:%M}<br>NW: %{y:.2f} m<extra></extra>",
        ))
        fig_all.update_layout(
            title="Tidegang 7 Tage (mit HW/NW)",
            xaxis=dict(title="Datum/Uhrzeit", tickformat="%d.%m"),
            yaxis=dict(title="Meereshöhe (m)", tickformat=".2f"),
            height=300,
            hovermode="x unified",
        )
        st.plotly_chart(fig_all, use_container_width=True, key="tide_7day")

    # ════════════════════════════════════════
    # TAB 3: Tagesdetails
    # ════════════════════════════════════════
    with tab3:
        st.subheader("📋 Tagesdetails")
        for _, r in df.iterrows():
            date_str = r["Date"]
            day_date = pd.Timestamp(date_str).date()
            df_day   = df_tide_full[df_tide_full["time"].dt.date == day_date]
            ex_day   = extrema_all[extrema_all["time"].dt.date == day_date]

            with st.expander(
                f"**{date_str}** — Score {int(r['Score'])} {r['Epic']}",
                expanded=(r["Score"] == df["Score"].max())
            ):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("**💨 Wind**")
                    st.write(f"Speed: {r['Wind_kn']} kn")
                    st.write(f"Böen:  {r['Gust_kn']} kn")
                    st.write(f"Richtung: {int(r['WindDir'])}°")
                    st.write(f"Status: {r['WindLabel']}")
                    st.write(f"Score: **{r['WindScore']}/50**")

                with col2:
                    st.markdown("**🌊 Swell**")
                    st.write(f"Höhe: {r['Wave_m']} m")
                    st.write(f"Periode: {r['Period']} s")
                    st.write(f"Richtung: {int(r['WaveDir'])}°")
                    st.write(f"Status: {r['SwellLabel']}")
                    st.write(f"Score: **{r['SwellScore']}/50**")

                with col3:
                    st.markdown("**🌊 Tide**")
                    if not df_day.empty:
                        mn = df_day["level"].min()
                        mx = df_day["level"].max()
                        windows = best_tide_windows(df_day)
                        hw_times = ex_day[ex_day["type"]=="HW"]["time"].apply(
                            lambda t: t.strftime("%H:%M")).tolist()
                        nw_times = ex_day[ex_day["type"]=="NW"]["time"].apply(
                            lambda t: t.strftime("%H:%M")).tolist()
                        st.write(f"HW: {', '.join(hw_times) or '–'} ({mx:.2f} m)")
                        st.write(f"NW: {', '.join(nw_times) or '–'} ({mn:.2f} m)")
                        st.write(f"Gut: **{windows[0] if windows else 'Kaum'}**")
                        st.write(f"Tide-Score: **{r['TideScore']:+d}**")
                    else:
                        st.write("Keine Tide-Daten")

                with col4:
                    st.markdown("**📊 Gesamt**")
                    st.write(f"Druck: {r['Pressure']} hPa")
                    st.write(f"Score: **{int(r['Score'])}/100**")
                    st.write(f"Konfidenz: {r['Confidence']}%")
                    if r["Epic"]:
                        st.markdown(f"<span class='epic-badge'>{r['Epic']}</span>",
                                    unsafe_allow_html=True)
                    al = r["Alarm"]
                    if al == "🔔 PERSÖNLICHER ALARM":
                        fenster = f" ({r['AlarmWindow']})" if r["AlarmWindow"] else ""
                        st.error(al + fenster)
                    elif al:
                        st.warning(al)

                # Mini-Tidegrafik direkt in der Tageskarte
                if not df_day.empty:
                    st.plotly_chart(
                        tide_chart(df_day, ex_day, date_str),
                        use_container_width=True,
                        key=f"tide_detail_{date_str}",
                    )

except requests.exceptions.RequestException as e:
    st.error(f"🌐 API-Fehler: {e}")
    st.info("Prüfe deine Internetverbindung.")
except KeyError as e:
    st.error(f"📦 API-Format geändert: Feld '{e}' fehlt.")
except ImportError:
    st.error("❌ scipy fehlt. Bitte: pip install scipy")
except Exception as e:
    st.error(f"❌ Fehler: {e}")
    st.exception(e)
