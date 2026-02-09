import streamlit as st
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np

# --- KONFIGURATION ---
DB_FILE = "trading_tool_advanced.db"

# --- DATENBANK FUNKTIONEN ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (symbol TEXT PRIMARY KEY, market TEXT, added_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  date TEXT,
                  orb_high REAL,
                  orb_low REAL,
                  orb_range REAL,
                  entry_time TEXT,
                  entry_price REAL,
                  exit_time TEXT,
                  exit_price REAL,
                  direction TEXT,
                  result TEXT,
                  pnl_percent REAL,
                  atr REAL,
                  created_at TEXT)''')
    conn.commit()
    conn.close()

def add_to_watchlist(symbol, market):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO watchlist VALUES (?, ?, ?)", 
                  (symbol.upper(), market, datetime.now().strftime("%Y-%m-%d")))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_watchlist():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM watchlist", conn)
    conn.close()
    return df

def save_backtest_results(results_df, symbol):
    conn = sqlite3.connect(DB_FILE)
    for _, row in results_df.iterrows():
        conn.execute('''INSERT INTO backtest_results 
                     (symbol, date, orb_high, orb_low, orb_range, entry_time, entry_price,
                      exit_time, exit_price, direction, result, pnl_percent, atr, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (symbol, str(row['Datum']), row['ORB High'], row['ORB Low'], 
                      row['ORB Range'], str(row.get('Entry Time', '')), row.get('Entry Price', 0),
                      str(row.get('Exit Time', '')), row.get('Exit Price', 0),
                      row.get('Direction', ''), row['Ergebnis'], row.get('PnL %', 0),
                      row.get('ATR', 0), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# --- TECHNISCHE INDIKATOREN ---
def calculate_atr(df, period=14):
    """Average True Range Berechnung"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

def is_trending_market(df, lookback=20):
    """Prüft ob der Markt in einem Trend ist (EMA20 Steigung)"""
    ema20 = df['Close'].ewm(span=20).mean()
    slope = (ema20.iloc[-1] - ema20.iloc[-lookback]) / lookback
    return abs(slope) > (df['Close'].iloc[-1] * 0.001)  # Mindestens 0.1% Steigung

# --- ERWEITERTE BACKTEST LOGIK ---
def fetch_data(ticker, days=59, interval="5m"):
    """Holt Daten von yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        return df
        
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    return df

def calculate_advanced_orb(df, market_type, chaos_minutes=30, use_filters=True):
    """
    Erweiterte ORB Analyse mit:
    - Intraday Exit bei 1.5:1 RR
    - Entry-Zeitfilter (nur bis 12:00)
    - ATR-Filter für Volatilität
    - Stop-Loss bei halber ORB-Range
    """
    results = []
    
    target_tz = 'Europe/Berlin' if market_type == 'EU' else 'America/New_York'
    df_local = df.copy()
    df_local.index = df_local.index.tz_convert(target_tz)
    
    # ATR für den gesamten Zeitraum berechnen
    df_local['ATR'] = calculate_atr(df_local, period=14)
    avg_atr = df_local['ATR'].rolling(10).mean()
    
    grouped = df_local.groupby(df_local.index.date)
    
    for date, day_data in grouped:
        if len(day_data) < 20:
            continue
            
        # Marktöffnung definieren
        if market_type == 'EU':
            open_time = day_data.index[0].replace(hour=9, minute=0, second=0, microsecond=0)
            entry_cutoff = day_data.index[0].replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            open_time = day_data.index[0].replace(hour=9, minute=30, second=0, microsecond=0)
            entry_cutoff = day_data.index[0].replace(hour=12, minute=0, second=0, microsecond=0)
            
        chaos_end_time = open_time + timedelta(minutes=chaos_minutes)
        
        # 1. Chaos Range (Opening Range)
        chaos_data = day_data[(day_data.index >= open_time) & (day_data.index < chaos_end_time)]
        
        if chaos_data.empty:
            continue
        
        orb_high = chaos_data['High'].max()
        orb_low = chaos_data['Low'].min()
        orb_range = orb_high - orb_low
        
        # ATR für diesen Tag
        day_atr = day_data['ATR'].iloc[0] if not pd.isna(day_data['ATR'].iloc[0]) else 0
        avg_atr_10 = avg_atr.loc[day_data.index[0]] if day_data.index[0] in avg_atr.index else day_atr
        
        # Volatilitäts-Filter: Nur handeln wenn ATR > Durchschnitt
        high_volatility = day_atr > avg_atr_10 if use_filters else True
        
        # 2. Post-Chaos Trading
        post_chaos = day_data[day_data.index >= chaos_end_time]
        
        if post_chaos.empty:
            continue
        
        # Trade-Variablen
        entry_price = None
        entry_time = None
        exit_price = None
        exit_time = None
        direction = None
        result = "Kein Setup"
        pnl_percent = 0
        
        # 3. Long Setup: Close ÜBER ORB High (nicht nur Touch)
        long_breakout = post_chaos[post_chaos['Close'] > orb_high]
        
        if not long_breakout.empty and (not use_filters or long_breakout.index[0] < entry_cutoff):
            direction = "LONG"
            entry_time = long_breakout.index[0]
            entry_price = long_breakout['Close'].iloc[0]
            
            # Stop Loss: Halbe Range unter Entry
            stop_loss = entry_price - (orb_range * 0.5)
            # Take Profit: 1.5x Range über Entry
            take_profit = entry_price + (orb_range * 1.5)
            
            # Intraday Exit prüfen
            remaining_day = post_chaos[post_chaos.index > entry_time]
            
            for idx, row in remaining_day.iterrows():
                # Stop getroffen?
                if row['Low'] <= stop_loss:
                    exit_time = idx
                    exit_price = stop_loss
                    result = "Stop Loss"
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                    break
                # Target getroffen?
                elif row['High'] >= take_profit:
                    exit_time = idx
                    exit_price = take_profit
                    result = "Take Profit"
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                    break
            
            # Kein Exit bis Tagesende
            if exit_time is None:
                exit_time = remaining_day.index[-1]
                exit_price = remaining_day['Close'].iloc[-1]
                result = "EOD Exit"
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        
        # 4. Short Setup: Close UNTER ORB Low
        short_breakout = post_chaos[post_chaos['Close'] < orb_low]
        
        if direction is None and not short_breakout.empty and (not use_filters or short_breakout.index[0] < entry_cutoff):
            direction = "SHORT"
            entry_time = short_breakout.index[0]
            entry_price = short_breakout['Close'].iloc[0]
            
            stop_loss = entry_price + (orb_range * 0.5)
            take_profit = entry_price - (orb_range * 1.5)
            
            remaining_day = post_chaos[post_chaos.index > entry_time]
            
            for idx, row in remaining_day.iterrows():
                if row['High'] >= stop_loss:
                    exit_time = idx
                    exit_price = stop_loss
                    result = "Stop Loss"
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                    break
                elif row['Low'] <= take_profit:
                    exit_time = idx
                    exit_price = take_profit
                    result = "Take Profit"
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                    break
            
            if exit_time is None:
                exit_time = remaining_day.index[-1]
                exit_price = remaining_day['Close'].iloc[-1]
                result = "EOD Exit"
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100
        
        # Nur Trades speichern, nicht "Kein Setup" Tage
        if direction is not None:
            results.append({
                'Datum': date,
                'ORB High': round(orb_high, 2),
                'ORB Low': round(orb_low, 2),
                'ORB Range': round(orb_range, 2),
                'ATR': round(day_atr, 2),
                'High Vol': high_volatility,
                'Direction': direction,
                'Entry Time': entry_time.strftime('%H:%M') if entry_time else '',
                'Entry Price': round(entry_price, 2) if entry_price else 0,
                'Exit Time': exit_time.strftime('%H:%M') if exit_time else '',
                'Exit Price': round(exit_price, 2) if exit_price else 0,
                'Ergebnis': result,
                'PnL %': round(pnl_percent, 2)
            })
        
    return pd.DataFrame(results)

# --- STREAMLIT GUI ---
st.set_page_config(page_title="Advanced ORB Tool", layout="wide")
init_db()

st.title("🎯 Advanced Opening Range Breakout Tool")
st.markdown("**Mit Intraday-Exits, ATR-Filter und Entry-Timing**")

# Sidebar
st.sidebar.header("⚙️ Einstellungen")

with st.sidebar.expander("📊 Wert hinzufügen"):
    new_ticker = st.text_input("Ticker Symbol", help="z.B. ^GDAXI, NVDA, SAP.DE")
    market_select = st.selectbox("Markt", ["EU", "US"])
    if st.button("Zur Watchlist"):
        if new_ticker:
            if add_to_watchlist(new_ticker, market_select):
                st.success(f"✅ {new_ticker} hinzugefügt")
            else:
                st.warning("⚠️ Bereits vorhanden")

st.sidebar.markdown("---")
use_filters = st.sidebar.checkbox("🔬 Filter aktivieren", value=True,
                                   help="ATR-Filter + Entry-Zeitlimit bis 12:00")
chaos_duration = st.sidebar.slider("⏱️ Opening Range (Min)", 15, 90, 30, 15)

# Hauptbereich
tabs = st.tabs(["📈 Backtest", "📋 Watchlist", "📊 Statistiken"])

with tabs[0]:
    watchlist = get_watchlist()
    
    if watchlist.empty:
        st.info("👈 Füge zuerst Werte in der Sidebar hinzu")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_symbol = st.selectbox(
                "Wert auswählen",
                watchlist['symbol'] + " (" + watchlist['market'] + ")",
                help="Wähle einen Wert aus deiner Watchlist"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_backtest = st.button("🚀 Analyse starten", type="primary")
        
        if run_backtest and selected_symbol:
            ticker = selected_symbol.split(" (")[0]
            market = selected_symbol.split(" (")[1].replace(")", "")
            
            with st.spinner(f'🔄 Analysiere {ticker} mit ORB-Strategie...'):
                df = fetch_data(ticker, days=59, interval="5m")
                
                if df.empty:
                    st.error("❌ Keine Daten gefunden. Prüfe das Ticker-Symbol.")
                else:
                    results_df = calculate_advanced_orb(df, market, chaos_duration, use_filters)
                    
                    if results_df.empty:
                        st.warning("⚠️ Keine gültigen Setups im Analysezeitraum gefunden.")
                    else:
                        # Metriken berechnen
                        total_trades = len(results_df)
                        winning_trades = len(results_df[results_df['PnL %'] > 0])
                        losing_trades = len(results_df[results_df['PnL %'] < 0])
                        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                        
                        avg_win = results_df[results_df['PnL %'] > 0]['PnL %'].mean() if winning_trades > 0 else 0
                        avg_loss = results_df[results_df['PnL %'] < 0]['PnL %'].mean() if losing_trades > 0 else 0
                        
                        total_pnl = results_df['PnL %'].sum()
                        
                        profit_factor = (winning_trades * avg_win) / abs(losing_trades * avg_loss) if losing_trades > 0 else 0
                        
                        # KPIs anzeigen
                        st.subheader("📊 Performance Metriken")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        col1.metric("Trades", total_trades)
                        col2.metric("Win Rate", f"{win_rate:.1f}%")
                        col3.metric("Avg Win", f"{avg_win:.2f}%")
                        col4.metric("Avg Loss", f"{avg_loss:.2f}%")
                        col5.metric("Profit Factor", f"{profit_factor:.2f}")
                        
                        st.metric("Gesamt PnL (ohne Hebel)", f"{total_pnl:.2f}%")
                        
                        # Hebel-Simulation für Ko-Trades
                        st.markdown("---")
                        st.subheader("💰 Ko-Zertifikat Simulation")
                        leverage = st.slider("Hebel auswählen", 3, 10, 5)
                        
                        results_df['Ko PnL %'] = results_df['PnL %'] * leverage
                        ko_total = results_df['Ko PnL %'].sum()
                        
                        col1, col2 = st.columns(2)
                        col1.metric(f"Gesamt PnL mit {leverage}x Hebel", f"{ko_total:.2f}%")
                        col2.metric("Max Drawdown", f"{results_df['Ko PnL %'].min():.2f}%")
                        
                        # Detaillierte Tabelle
                        st.markdown("---")
                        st.subheader("📋 Trade History")
                        
                        display_df = results_df.sort_values('Datum', ascending=False)
                        
                        # Farbcodierung
                        def highlight_pnl(val):
                            if isinstance(val, (int, float)):
                                color = 'background-color: #90EE90' if val > 0 else 'background-color: #FFB6C6' if val < 0 else ''
                                return color
                            return ''
                        
                        styled_df = display_df.style.applymap(highlight_pnl, subset=['PnL %', 'Ko PnL %'])
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
                        # Chart
                        st.markdown("---")
                        st.subheader("📈 Equity Curve (kumulativ)")
                        
                        display_df['Cumulative PnL'] = display_df['Ko PnL %'].cumsum()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=display_df['Datum'],
                            y=display_df['Cumulative PnL'],
                            mode='lines+markers',
                            name='Kumulativer PnL',
                            line=dict(color='#00CC96', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"Equity Curve - {ticker} ({leverage}x Hebel)",
                            xaxis_title="Datum",
                            yaxis_title="Kumulativer PnL (%)",
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Speichern
                        if st.button("💾 Ergebnisse in Datenbank speichern"):
                            save_backtest_results(results_df, ticker)
                            st.success("✅ Gespeichert!")

with tabs[1]:
    st.subheader("📋 Deine Watchlist")
    wl = get_watchlist()
    if not wl.empty:
        st.dataframe(wl, use_container_width=True)
    else:
        st.info("Noch keine Werte gespeichert")

with tabs[2]:
    st.subheader("📊 Historische Backtest-Ergebnisse")
    conn = sqlite3.connect(DB_FILE)
    history = pd.read_sql("SELECT * FROM backtest_results ORDER BY date DESC LIMIT 100", conn)
    conn.close()
    
    if not history.empty:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("Noch keine Backtests durchgeführt")

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Tipp:** Aktiviere Filter für realistischere Ergebnisse")
