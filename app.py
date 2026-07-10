"""
LEADER DL Filter — Streamlit 웹앱
실행: py -3.11 -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import warnings, pickle, os, json, time
warnings.filterwarnings('ignore')

st.set_page_config(page_title='LEADER DL Filter', page_icon='📈', layout='wide')

# ── 한글 폰트 ──
import matplotlib.font_manager as fm
def set_korean_font():
    for fp in [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        'C:/Windows/Fonts/malgun.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    ]:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            matplotlib.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
            break
    matplotlib.rcParams['axes.unicode_minus'] = False
set_korean_font()

# ============================================================
# 설정
# ============================================================
SUPABASE_URL = 'https://qdyzkekzjrzaupeplyxv.supabase.co'
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'sb_secret_u1wuZY958fWHkqTX1qDGfw_tCQQYy7A')
USE_SUPABASE = True
LOOKBACK       = 20
FUTURE_DAYS    = 5
RISE_THRESHOLD = 0.03
TRAIN_YEARS    = 5
TEST_RATIO     = 0.2
MODEL_PATH     = 'leader_dl_model.keras'
SCALER_PATH    = 'leader_scalers.pkl'
PORTFOLIO_PATH = 'portfolio.json'
TRAINED_PATH   = 'trained_stocks.json'
KOSPI_PATH     = 'kospi_stocks.json'   # 코스피 전체 종목 캐시
GMAIL_ADDRESS = 'najirong74@gmail.com'
GMAIL_APP_PW  = 'frqa bcfe etch cbdl'  # 여기에 앱 비밀번호 16자리 입력 (예: 'abcd efgh ijkl mnop')

FEATURE_COLS = [
    'body','upper_wick','lower_wick','candle_range',
    'ret_1','ret_3','ret_5','ret_10','ret_20',
    'ma5_ratio','ma10_ratio','ma20_ratio','ma60_ratio',
    'atr14_ratio','rsi14','bb_pos',
    'vol_ratio5','vol_ratio20',
    'macd','macd_signal','macd_hist',
    'turnover_ratio5','gap_ratio',
    'high52w_ratio','low52w_ratio',
    'foreign_net','institution_net'
]

# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.header('⚙️ 설정')

    THRESHOLD_PRED = st.slider(
        '📊 매수 신호 임계값',
        min_value=0.50, max_value=0.95,
        value=st.session_state.get('threshold', 0.70),
        step=0.05, format='%.2f',
        help='높을수록 신호 횟수는 줄지만 정확도가 올라갑니다'
    )
    st.session_state['threshold'] = THRESHOLD_PRED
    st.caption(f'현재 임계값: **{THRESHOLD_PRED*100:.0f}%**')

    st.divider()
    st.subheader('🧪 백테스트 파라미터')
    BT_ATR_MULT = st.slider(
        'ATR 손절 배수', min_value=1.0, max_value=4.0,
        value=st.session_state.get('atr_mult', 2.0), step=0.5,
        help='ATR × 배수 = 손절가'
    )
    st.session_state['atr_mult'] = BT_ATR_MULT

    BT_GAP_SLIP = st.checkbox(
        '갭다운 시가 체결 반영',
        value=st.session_state.get('gap_slip', True),
        help='손절 당일 시가가 손절가보다 낮으면 시가로 체결'
    )
    st.session_state['gap_slip'] = BT_GAP_SLIP

    st.divider()
    if st.button('🔄 모델 재학습', type='secondary', use_container_width=True):
        for f in [MODEL_PATH, SCALER_PATH, TRAINED_PATH]:
            if os.path.exists(f): os.remove(f)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption('📱 모바일 접속')
    try:
        import qrcode
        import io
        qr = qrcode.QRCode(version=1, box_size=4, border=2)
        qr.add_data('https://leader-dl-filter-5986.streamlit.app')
        qr.make(fit=True)
        img = qr.make_image(fill_color='black', back_color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        st.image(buf.getvalue(), width=160)
        st.caption('leader-dl-filter-5986.streamlit.app')
    except:
        st.caption('https://leader-dl-filter-5986.streamlit.app')

# ============================================================
# Supabase
# ============================================================
@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# 코스피 전체 종목 리스트
# ============================================================
@st.cache_data(ttl=60*60*24)
def get_kospi_all_stocks():
    """kospi_stocks.json 파일에서 코스피 전체 종목 로드"""
    if os.path.exists(KOSPI_PATH):
        with open(KOSPI_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 파일 없으면 pykrx 시도
    try:
        from pykrx import stock
        from datetime import datetime, timedelta
        for delta in range(5):
            try_date = (datetime.today() - timedelta(days=delta)).strftime('%Y%m%d')
            tickers = stock.get_market_ticker_list(try_date, market='KOSPI')
            if tickers:
                result = {code: stock.get_market_ticker_name(code) for code in tickers}
                with open(KOSPI_PATH, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                return result
    except:
        pass
    st.warning('kospi_stocks.json 파일이 없습니다.')
    return {}

def search_stock(query):
    """종목명 또는 종목코드로 검색"""
    all_stocks = get_kospi_all_stocks()
    query = query.strip()
    result = {}
    for code, name in all_stocks.items():
        if query in name or query == code:
            result[code] = name
    return result  # {code: name}

# ============================================================
# 포트폴리오 관리
# ============================================================
def load_portfolio():
    if USE_SUPABASE:
        try:
            sb  = get_supabase()
            res = sb.table('portfolio').select('*').execute()
            portfolio = {}
            for row in res.data:
                portfolio[row['code']] = {
                    'name' : row['name'],
                    'price': float(row['price']),
                    'qty'  : int(row['qty']),
                    'total': float(row['total']),
                }
            return portfolio
        except Exception as e:
            st.warning(f'Supabase 로드 실패: {e}')
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_portfolio(portfolio):
    with open(PORTFOLIO_PATH, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)
    if USE_SUPABASE:
        try:
            sb = get_supabase()
            sb.table('portfolio').delete().neq('code','').execute()
            for code, info in portfolio.items():
                sb.table('portfolio').upsert({
                    'code' : code, 'name' : info['name'],
                    'price': info['price'], 'qty'  : info['qty'],
                    'total': info.get('total', info['price']*info['qty']),
                }).execute()
        except Exception as e:
            st.warning(f'Supabase 저장 실패: {e}')

def load_trained_stocks():
    if USE_SUPABASE:
        try:
            sb  = get_supabase()
            res = sb.table('trained_stocks').select('*').execute()
            if res.data:
                return {row['code']: row['name'] for row in res.data}
        except:
            pass
    if os.path.exists(TRAINED_PATH):
        with open(TRAINED_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return get_kospi_all_stocks()

def save_trained_stocks(stocks):
    with open(TRAINED_PATH, 'w', encoding='utf-8') as f:
        json.dump(stocks, f, ensure_ascii=False, indent=2)
    if USE_SUPABASE:
        try:
            sb = get_supabase()
            # 배치 upsert (한 번에 100개씩)
            items = [{'code': c, 'name': n} for c,n in stocks.items()]
            sb.table('trained_stocks').delete().neq('code','').execute()
            for i in range(0, len(items), 100):
                sb.table('trained_stocks').upsert(items[i:i+100]).execute()
        except Exception as e:
            st.warning(f'Supabase 저장 실패: {e}')

# ============================================================
# 핵심 함수
# ============================================================
def add_features(df, code=None):
    d = df.copy()
    # ── 기존 피처 ──
    d['body']         = (d['close'] - d['open']) / d['open']
    d['upper_wick']   = (d['high'] - d[['open','close']].max(axis=1)) / d['open']
    d['lower_wick']   = (d[['open','close']].min(axis=1) - d['low']) / d['open']
    d['candle_range'] = (d['high'] - d['low']) / d['open']
    for n in [1,3,5,10,20]:
        d[f'ret_{n}'] = d['close'].pct_change(n)
    for w in [5,10,20,60]:
        d[f'ma{w}_ratio'] = d['close'] / d['close'].rolling(w).mean() - 1
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - d['close'].shift()).abs(),
        (d['low']  - d['close'].shift()).abs()
    ], axis=1).max(axis=1)
    d['atr14']       = tr.rolling(14).mean()
    d['atr14_ratio'] = d['atr14'] / d['close']
    delta = d['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi14']  = 100 - 100 / (1 + gain / (loss + 1e-9))
    ma20        = d['close'].rolling(20).mean()
    std20       = d['close'].rolling(20).std()
    d['bb_pos'] = (d['close'] - ma20) / (2 * std20 + 1e-9)
    d['vol_ratio5']  = d['volume'] / d['volume'].rolling(5).mean()
    d['vol_ratio20'] = d['volume'] / d['volume'].rolling(20).mean()
    ema12 = d['close'].ewm(span=12).mean()
    ema26 = d['close'].ewm(span=26).mean()
    d['macd']        = (ema12 - ema26) / d['close']
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    d['macd_hist']   = d['macd'] - d['macd_signal']
    # ── 추가 피처 ──
    # 1. 거래대금 비율
    d['turnover']       = d['close'] * d['volume']
    d['turnover_ratio5'] = d['turnover'] / d['turnover'].rolling(5).mean()
    # 2. 갭 비율
    d['gap_ratio'] = (d['open'] - d['close'].shift()) / d['close'].shift()
    # 3. 52주 최고/최저 대비
    d['high52w_ratio'] = d['close'] / d['high'].rolling(252).max()
    d['low52w_ratio']  = d['close'] / d['low'].rolling(252).min()
    # 4. 외국인/기관 순매수 (pykrx)
    if code is not None:
        try:
            from pykrx import stock
            from datetime import datetime, timedelta
            end   = datetime.today().strftime('%Y%m%d')
            start = (datetime.today() - timedelta(days=365*TRAIN_YEARS+30)).strftime('%Y%m%d')
            inv = stock.get_market_trading_value_by_date(start, end, code)
            if not inv.empty and '외국인합계' in inv.columns:
                inv.index = pd.to_datetime(inv.index)
                d['foreign_net'] = inv['외국인합계'].reindex(d.index).fillna(0) / (d['turnover'] + 1)
            else:
                d['foreign_net'] = 0.0
            if not inv.empty and '기관합계' in inv.columns:
                d['institution_net'] = inv['기관합계'].reindex(d.index).fillna(0) / (d['turnover'] + 1)
            else:
                d['institution_net'] = 0.0
        except:
            d['foreign_net']     = 0.0
            d['institution_net'] = 0.0
    else:
        d['foreign_net']     = 0.0
        d['institution_net'] = 0.0
    return d
    d = df.copy()
    d['body']         = (d['close'] - d['open']) / d['open']
    d['upper_wick']   = (d['high'] - d[['open','close']].max(axis=1)) / d['open']
    d['lower_wick']   = (d[['open','close']].min(axis=1) - d['low']) / d['open']
    d['candle_range'] = (d['high'] - d['low']) / d['open']
    for n in [1,3,5,10,20]:
        d[f'ret_{n}'] = d['close'].pct_change(n)
    for w in [5,10,20,60]:
        d[f'ma{w}_ratio'] = d['close'] / d['close'].rolling(w).mean() - 1
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - d['close'].shift()).abs(),
        (d['low']  - d['close'].shift()).abs()
    ], axis=1).max(axis=1)
    d['atr14']       = tr.rolling(14).mean()
    d['atr14_ratio'] = d['atr14'] / d['close']
    delta = d['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi14']  = 100 - 100 / (1 + gain / (loss + 1e-9))
    ma20        = d['close'].rolling(20).mean()
    std20       = d['close'].rolling(20).std()
    d['bb_pos'] = (d['close'] - ma20) / (2 * std20 + 1e-9)
    d['vol_ratio5']  = d['volume'] / d['volume'].rolling(5).mean()
    d['vol_ratio20'] = d['volume'] / d['volume'].rolling(20).mean()
    ema12 = d['close'].ewm(span=12).mean()
    ema26 = d['close'].ewm(span=26).mean()
    d['macd']        = (ema12 - ema26) / d['close']
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    d['macd_hist']   = d['macd'] - d['macd_signal']
    return d

def make_sequences(df, lookback, future_days, threshold):
    d = df[FEATURE_COLS].copy().dropna()
    future_ret = df['close'].pct_change(future_days).shift(-future_days)
    labels = (future_ret > threshold).astype(int).loc[d.index]
    valid  = d.index.intersection(labels.dropna().index)
    d, labels = d.loc[valid], labels.loc[valid]
    X, y, dates = [], [], []
    for i in range(lookback, len(d)):
        X.append(d.iloc[i-lookback:i].values)
        y.append(labels.iloc[i])
        dates.append(d.index[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), dates

@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_raw_data(code):
    from pykrx import stock
    from datetime import datetime, timedelta
    end   = datetime.today().strftime('%Y%m%d')
    start = (datetime.today() - timedelta(days=365*TRAIN_YEARS+30)).strftime('%Y%m%d')
    df = stock.get_market_ohlcv_by_date(start, end, code)
    df = df.iloc[:,:5]
    df.columns = ['open','high','low','close','volume']
    df = df[df['volume'] > 0]
    if len(df) == 0:
        raise ValueError(f'{code} 데이터 없음')
    return df

def train_on_stocks(stocks_dict, progress_bar=None, status_text=None):
    """
    코스피 전체 종목 학습.
    - 에러 종목은 건너뜀
    - 진행률 표시
    """
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    all_X, all_y = [], []
    scalers, featured_data, raw_data = {}, {}, {}
    total = len(stocks_dict)
    success, fail = 0, 0

    for i, (code, name) in enumerate(stocks_dict.items()):
        if progress_bar:
            progress_bar.progress((i+1)/total, text=f'데이터 수집 중... {i+1}/{total} — {name}({code})')
        try:
            df = get_raw_data(code)
            raw_data[code] = df
            df_feat = add_features(df, code)
            featured_data[code] = df_feat
            d = df_feat[FEATURE_COLS].dropna()
            if len(d) < 100:   # 데이터 너무 적으면 건너뜀
                continue
            n_train = int(len(d) * (1 - TEST_RATIO))
            scaler  = StandardScaler()
            d_sc    = d.copy()
            d_sc.iloc[:n_train] = scaler.fit_transform(d.iloc[:n_train])
            d_sc.iloc[n_train:] = scaler.transform(d.iloc[n_train:])
            scalers[code] = scaler
            df_s = df_feat.copy()
            df_s[FEATURE_COLS] = d_sc
            X, y, _ = make_sequences(df_s, LOOKBACK, FUTURE_DAYS, RISE_THRESHOLD)
            if len(X) == 0:
                continue
            all_X.append(X); all_y.append(y)
            success += 1
        except Exception as e:
            fail += 1

    if status_text:
        status_text.info(f'✅ 수집 완료: {success}개 성공 / {fail}개 실패 — 모델 학습 시작...')

    if not all_X:
        raise ValueError('데이터 수집 실패 — 네트워크(KRX 접근)를 확인하세요')

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    split  = int(len(all_X) * (1 - TEST_RATIO))
    X_tr, y_tr = all_X[:split], all_y[:split]
    pw = (1 - all_y.mean()) / all_y.mean()

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(LOOKBACK, len(FEATURE_COLS))),
        BatchNormalization(), Dropout(0.3),
        LSTM(64), BatchNormalization(), Dropout(0.3),
        Dense(32, activation='relu'), Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    # 학습 진행률 표시용 콜백
    train_pb = st.progress(0, text='모델 학습 중... (epoch 0/100)')
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pct = min((epoch+1)/100, 1.0)
            acc = logs.get('val_auc', 0) if logs else 0
            train_pb.progress(pct, text=f'모델 학습 중... (epoch {epoch+1}/100, val_auc={acc:.3f})')

    model.fit(
        X_tr, y_tr,
        validation_split=0.15, epochs=200, batch_size=64,
        class_weight={0:1.0, 1:pw},
        callbacks=[
            EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
            ProgressCallback()
        ], verbose=0
    )
    train_pb.empty()
    return model, scalers, featured_data, raw_data

@st.cache_resource(show_spinner=False)
def load_model_and_data():
    # 모델 파일 없으면 코스피 전체로 학습
    if not os.path.exists(MODEL_PATH):
        trained_stocks = get_kospi_all_stocks()
    else:
        trained_stocks = load_trained_stocks()

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scalers = pickle.load(f)
        featured_data, raw_data = {}, {}
        for code, name in trained_stocks.items():
            try:
                df = get_raw_data(code)
                raw_data[code] = df
                featured_data[code] = add_features(df, code)
                if code not in scalers:
                    from sklearn.preprocessing import StandardScaler
                    d = featured_data[code][FEATURE_COLS].dropna()
                    sc = StandardScaler()
                    sc.fit(d.iloc[:int(len(d)*0.8)])
                    scalers[code] = sc
            except:
                pass
    else:
        # 최초 학습 — 코스피 전체
        kospi_all = get_kospi_all_stocks()
        pb  = st.progress(0, text='코스피 종목 데이터 수집 중...')
        txt = st.empty()
        model, scalers, featured_data, raw_data = train_on_stocks(kospi_all, pb, txt)
        pb.empty(); txt.empty()
        model.save(MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scalers, f)
        # trained_stocks 업데이트 (성공한 종목만)
        save_trained_stocks({c: kospi_all[c] for c in featured_data if c in kospi_all})

    return model, scalers, featured_data, raw_data

def get_predictions(df, code, model, scalers):
    from sklearn.preprocessing import StandardScaler
    d = df[FEATURE_COLS].dropna()
    if code in scalers:
        scaler = scalers[code]
        d_sc   = pd.DataFrame(scaler.transform(d), columns=FEATURE_COLS, index=d.index)
    else:
        scaler = StandardScaler()
        d_sc   = pd.DataFrame(scaler.fit_transform(d), columns=FEATURE_COLS, index=d.index)
        scalers[code] = scaler
    df_s = df.copy(); df_s[FEATURE_COLS] = d_sc
    X, _, dates = make_sequences(df_s, LOOKBACK, FUTURE_DAYS, RISE_THRESHOLD)
    probs = model.predict(X, verbose=0, batch_size=32).flatten()
    return pd.DatetimeIndex(dates), probs, probs >= THRESHOLD_PRED

def calc_turtle(df_raw, atr_mult=2.0):
    tr = pd.concat([
        df_raw['high'] - df_raw['low'],
        (df_raw['high'] - df_raw['close'].shift()).abs(),
        (df_raw['low']  - df_raw['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr14      = tr.rolling(14).mean().iloc[-1]
    close_now  = df_raw['close'].iloc[-1]
    stop_atr   = close_now - atr_mult * atr14
    stop_10low = df_raw['low'].tail(10).min()
    stop_20low = df_raw['low'].tail(20).min()
    unit       = int(10_000_000 * 0.01 / (atr_mult * atr14)) if atr14 > 0 else 0
    return close_now, atr14, stop_atr, stop_10low, stop_20low, unit

# ============================================================
# 차트 함수
# ============================================================
def draw_chart(df, code, name, start_dt, end_dt, model, scalers):
    dates_arr, probs, signals = get_predictions(df, code, model, scalers)
    mask = (dates_arr >= start_dt) & (dates_arr <= end_dt)
    dates_plot, probs_plot, signals_plot = dates_arr[mask], probs[mask], signals[mask]
    if len(dates_plot) == 0:
        st.warning('해당 기간 데이터 없음'); return

    df_range = df.loc[df.index.isin(dates_plot)].copy()
    df_range.index = pd.to_datetime(df_range.index)
    future_rets = {}
    for dt in df_range.index:
        idx = df.index.get_loc(dt)
        if idx + FUTURE_DAYS < len(df):
            fp = df['close'].iloc[idx+FUTURE_DAYS]
            future_rets[dt] = (fp - df['close'].iloc[idx]) / df['close'].iloc[idx] * 100
        else:
            future_rets[dt] = None

    days_range = (end_dt - start_dt).days
    bar_w = 0.4 if days_range <= 14 else (0.6 if days_range <= 60 else 0.7)
    n = len(df_range); xs = list(range(n))
    dt_labels = [d.strftime('%m.%d') if days_range<=90 else d.strftime('%y.%m') for d in df_range.index]
    dt_to_x   = {dt: i for i, dt in enumerate(df_range.index)}
    prob_map  = dict(zip(dates_plot, probs_plot))
    buy_dates = dates_plot[signals_plot]
    buy_xs    = [dt_to_x[dt] for dt in buy_dates if dt in dt_to_x]
    buy_prices_list = [df_range['close'].iloc[dt_to_x[dt]] * 0.982 for dt in buy_dates if dt in dt_to_x]
    prob_xs   = [dt_to_x[dt] for dt in dates_plot if dt in dt_to_x]
    prob_vals = [probs_plot[i] for i, dt in enumerate(dates_plot) if dt in dt_to_x]

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14,8), gridspec_kw={'height_ratios':[3,1.5]})
    plt.subplots_adjust(hspace=0.08)
    fig.suptitle(f'{name} ({code})  |  {start_dt.strftime("%Y.%m.%d")} ~ {end_dt.strftime("%Y.%m.%d")}', fontsize=14, fontweight='bold')
    for i,(dt,row) in enumerate(df_range.iterrows()):
        c = '#E74C3C' if row['close']>=row['open'] else '#3498DB'
        ax1.plot([i,i],[row['low'],row['high']],color=c,linewidth=0.8,alpha=0.8)
        ax1.bar(i,abs(row['close']-row['open']),bottom=min(row['open'],row['close']),color=c,alpha=0.8,width=bar_w)
    ax1.plot(xs,df_range['close'].values,color='#333333',linewidth=1.2,alpha=0.7,label='종가')
    for w,c,ls in [(5,'#FF6B35','-'),(20,'#4ECDC4','-'),(60,'#9B59B6','--')]:
        ma = df['close'].rolling(w).mean().reindex(df_range.index).values
        ax1.plot(xs,ma,color=c,linewidth=1.1,linestyle=ls,label=f'MA{w}',alpha=0.85)
    if buy_xs:
        ax1.scatter(buy_xs,buy_prices_list,marker='^',s=150,color='#E74C3C',zorder=6,
                    label=f'매수신호({len(buy_xs)}회)',edgecolors='white',linewidths=0.8)
        for bx in buy_xs:
            ax1.axvspan(bx-0.5,bx+0.5,alpha=0.1,color='#E74C3C',zorder=0)
    ax1.set_ylabel('주가 (원)',fontsize=11); ax1.legend(loc='upper left',fontsize=9,framealpha=0.8)
    ax1.grid(axis='y',alpha=0.2)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    tick_step = max(1,n//12)
    ax1.set_xticks(xs[::tick_step]); ax1.set_xticklabels(dt_labels[::tick_step],rotation=30,fontsize=9)
    ax1.set_xlim(-0.8,n-0.2)
    vc = ['#E74C3C' if c>=o else '#3498DB' for c,o in zip(df_range['close'],df_range['open'])]
    ax2.bar(xs,df_range['volume'].values,color=vc,width=bar_w,alpha=0.5,label='거래량')
    ax2.set_ylabel('거래량',fontsize=10,color='#555555')
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax2.grid(axis='y',alpha=0.15)
    ax2r = ax2.twinx()
    if prob_xs:
        ax2r.plot(prob_xs,prob_vals,color='#9B59B6',linewidth=1.8,label='매수확률')
        ax2r.fill_between(prob_xs,THRESHOLD_PRED,prob_vals,
                          where=[v>=THRESHOLD_PRED for v in prob_vals],alpha=0.2,color='#E74C3C')
    ax2r.axhline(THRESHOLD_PRED,color='#E74C3C',linestyle='--',linewidth=1.2,label=f'임계값 {THRESHOLD_PRED*100:.0f}%')
    ax2r.set_ylim(0,1); ax2r.set_ylabel('매수확률',fontsize=10,color='#9B59B6')
    ax2r.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{x:.0%}'))
    ax2r.tick_params(axis='y',labelcolor='#9B59B6')
    l1,lb1 = ax2.get_legend_handles_labels(); l2,lb2 = ax2r.get_legend_handles_labels()
    ax2.legend(l1+l2,lb1+lb2,loc='upper left',fontsize=9)
    ax2.set_xticks(xs[::tick_step]); ax2.set_xticklabels(dt_labels[::tick_step],rotation=30,fontsize=9)
    ax2.set_xlim(-0.8,n-0.2)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    if buy_xs:
        rows = []
        for dt in buy_dates:
            if dt not in dt_to_x: continue
            price = df['close'].get(dt,0)
            ret   = future_rets.get(dt)
            rows.append({'날짜':dt.strftime('%Y-%m-%d'),'확률':f'{prob_map.get(dt,0):.1%}',
                         '현재가':f'{int(price):,}원',f'{FUTURE_DAYS}일후':f'{ret:+.1f}%' if ret else '미래없음'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

def draw_turtle_chart(df_raw, code, name, model, scalers):
    from sklearn.preprocessing import StandardScaler
    atr_mult = st.session_state.get('atr_mult', 2.0)
    df_feat = add_features(df_raw)
    d = df_feat[FEATURE_COLS].dropna()
    sc = StandardScaler()
    n_tr = int(len(d)*0.8)
    d_sc = d.copy()
    d_sc.iloc[:n_tr] = sc.fit_transform(d.iloc[:n_tr])
    d_sc.iloc[n_tr:] = sc.transform(d.iloc[n_tr:])
    scalers[code] = sc
    df_s = df_feat.copy(); df_s[FEATURE_COLS] = d_sc
    X, _, dates = make_sequences(df_s, LOOKBACK, FUTURE_DAYS, RISE_THRESHOLD)
    probs   = model.predict(X, verbose=0).flatten()
    signals = probs >= THRESHOLD_PRED
    dates_arr = pd.DatetimeIndex(dates)
    df_20 = df_raw.tail(20).copy(); df_20.index = pd.to_datetime(df_20.index)
    n = len(df_20); xs = list(range(n))
    dt_labels = [d.strftime('%m.%d') for d in df_20.index]
    mask = pd.DatetimeIndex(dates_arr).isin(df_20.index)
    dates_20, probs_20, signals_20 = dates_arr[mask], probs[mask], signals[mask]
    dt_to_x  = {dt:i for i,dt in enumerate(df_20.index)}
    prob_map_20 = dict(zip(dates_20, probs_20))
    buy_xs_20    = [dt_to_x[dt] for dt in dates_20[signals_20] if dt in dt_to_x]
    buy_prices_20= [df_20['close'].iloc[dt_to_x[dt]]*0.982 for dt in dates_20[signals_20] if dt in dt_to_x]
    close_now, atr14, stop_atr, stop_10low, stop_20low, unit = calc_turtle(df_raw, atr_mult)
    latest_prob   = probs[-1]
    latest_signal = '🟢 매수 신호' if latest_prob >= THRESHOLD_PRED else '🔴 비신호'

    # 매수 신호 발생 시 추적 테이블에 자동 저장
    if latest_prob >= THRESHOLD_PRED:
        latest_date  = df_raw.index[-1]
        latest_price = df_raw['close'].iloc[-1]
        save_signal_tracking(code, name, latest_date, latest_price, latest_prob)

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':[3,1.5]})
    plt.subplots_adjust(hspace=0.08)
    fig.suptitle(f'🐢 {name} ({code})  |  최근 20일  |  터틀 트레이딩 분석',fontsize=14,fontweight='bold')
    bar_w=0.5
    for i,(dt,row) in enumerate(df_20.iterrows()):
        c='#E74C3C' if row['close']>=row['open'] else '#3498DB'
        ax1.plot([i,i],[row['low'],row['high']],color=c,linewidth=0.9,alpha=0.85)
        ax1.bar(i,abs(row['close']-row['open']),bottom=min(row['open'],row['close']),color=c,alpha=0.85,width=bar_w)
    ax1.plot(xs,df_20['close'].values,color='#333333',linewidth=1.3,alpha=0.7,label='종가')
    for w,c,ls in [(5,'#FF6B35','-'),(20,'#4ECDC4','-')]:
        ma = df_raw['close'].rolling(w).mean().reindex(df_20.index).values
        ax1.plot(xs,ma,color=c,linewidth=1.1,linestyle=ls,label=f'MA{w}',alpha=0.85)
    if buy_xs_20:
        ax1.scatter(buy_xs_20,buy_prices_20,marker='^',s=160,color='#E74C3C',zorder=6,
                    label=f'매수신호({len(buy_xs_20)}회)',edgecolors='white',linewidths=0.8)
        for bx in buy_xs_20:
            ax1.axvspan(bx-0.5,bx+0.5,alpha=0.1,color='#E74C3C',zorder=0)
    ax1.axhline(stop_atr,  color='#E67E22',linestyle='--',linewidth=1.5,
                label=f'ATR×{atr_mult:.1f} 손절 {int(stop_atr):,}원')
    ax1.axhline(stop_10low,color='#E74C3C',linestyle=':',linewidth=1.8,label=f'10일최저 {int(stop_10low):,}원')
    ax1.axhline(stop_20low,color='#8E44AD',linestyle=':',linewidth=1.5,label=f'20일최저 {int(stop_20low):,}원')
    y_bot = min(df_20['low'].min()*0.97, stop_atr*0.99)
    ax1.fill_between([-0.8,n-0.2],stop_atr,y_bot,alpha=0.05,color='#E74C3C',zorder=0)
    ax1.set_ylabel('주가 (원)',fontsize=11); ax1.legend(loc='upper left',fontsize=8.5,framealpha=0.85)
    ax1.grid(axis='y',alpha=0.2)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax1.set_xticks(xs); ax1.set_xticklabels(dt_labels,rotation=30,fontsize=8.5); ax1.set_xlim(-0.8,n-0.2)
    vc=['#E74C3C' if c>=o else '#3498DB' for c,o in zip(df_20['close'],df_20['open'])]
    ax2.bar(xs,df_20['volume'].values,color=vc,width=bar_w,alpha=0.5,label='거래량')
    ax2.set_ylabel('거래량',fontsize=10,color='#555555')
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax2.grid(axis='y',alpha=0.15)
    ax2r=ax2.twinx()
    prob_xs_20  = [dt_to_x[dt] for dt in dates_20 if dt in dt_to_x]
    prob_vals_20= [prob_map_20[dt] for dt in dates_20 if dt in dt_to_x]
    if prob_xs_20:
        ax2r.plot(prob_xs_20,prob_vals_20,color='#9B59B6',linewidth=1.8,label='매수확률')
        ax2r.fill_between(prob_xs_20,THRESHOLD_PRED,prob_vals_20,
                          where=[v>=THRESHOLD_PRED for v in prob_vals_20],alpha=0.2,color='#E74C3C')
    ax2r.axhline(THRESHOLD_PRED,color='#E74C3C',linestyle='--',linewidth=1.2,label=f'임계값 {THRESHOLD_PRED*100:.0f}%')
    ax2r.set_ylim(0,1); ax2r.set_ylabel('매수확률',fontsize=10,color='#9B59B6')
    ax2r.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{x:.0%}'))
    ax2r.tick_params(axis='y',labelcolor='#9B59B6')
    l1,lb1=ax2.get_legend_handles_labels(); l2,lb2=ax2r.get_legend_handles_labels()
    ax2.legend(l1+l2,lb1+lb2,loc='upper left',fontsize=9)
    ax2.set_xticks(xs); ax2.set_xticklabels(dt_labels,rotation=30,fontsize=8.5); ax2.set_xlim(-0.8,n-0.2)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric('현재가',f'{int(close_now):,}원')
        st.metric('ATR(14)',f'{int(atr14):,}원')
        st.metric('딥러닝 판정',latest_signal,f'{latest_prob:.1%}')
    with c2:
        st.metric(f'① ATR×{atr_mult:.1f} 손절',f'{int(stop_atr):,}원',
                  f'-{(close_now-stop_atr)/close_now*100:.1f}%',delta_color='inverse')
        st.metric('② 10일 최저가 손절',f'{int(stop_10low):,}원',
                  f'-{(close_now-stop_10low)/close_now*100:.1f}%',delta_color='inverse')
        st.metric('③ 20일 최저가 손절',f'{int(stop_20low):,}원',
                  f'-{(close_now-stop_20low)/close_now*100:.1f}%',delta_color='inverse')
    with c3:
        st.metric('추천 매수 수량(1000만원 기준)',f'{unit}주')
        st.metric('매수 금액',f'{int(unit*close_now):,}원')

    return close_now, atr14, stop_atr, stop_10low, stop_20low, latest_prob, latest_signal

def _run_analysis(new_code, new_name, model, scalers, featured_data):
    with st.spinner(f'{new_name} 데이터 수집 중...'):
        try:
            df_raw = get_raw_data(new_code)
            st.success(f'✅ {len(df_raw)}일치 수집 완료')
            draw_turtle_chart(df_raw, new_code, new_name, model, scalers)
            featured_data[new_code] = add_features(df_raw)
        except Exception as e:
            st.error(f'수집 실패: {e}')
def load_top20_from_supabase():
    try:
        sb  = get_supabase()
        res = sb.table('top20_signals').select('*').order('prob', desc=True).limit(20).execute()
        if res.data:
            return pd.DataFrame(res.data)
    except:
        pass
    return pd.DataFrame()

def save_top20_to_supabase(df):
    try:
        sb = get_supabase()
        sb.table('top20_signals').delete().neq('code', '').execute()
        for _, row in df.iterrows():
            sb.table('top20_signals').insert({
                'code': row['code'], 'name': row['name'],
                'prob': float(row['prob']),
                'price': int(row['price']),
                'stop_atr': int(row['stop_atr']),
                'stop_10low': int(row['stop_10low']),
                'stop_20low': int(row['stop_20low']),
                'signal': row['signal'],
                'updated_at': pd.Timestamp.now().isoformat(),
            }).execute()
    except Exception as e:
        st.warning(f'Supabase 저장 실패: {e}')

def save_signal_tracking(code, name, signal_date, signal_price, prob):
    """매수 신호 발생 시 추적 테이블에 저장"""
    try:
        sb = get_supabase()
        # 같은 날짜 같은 종목 중복 방지
        existing = sb.table('signal_tracking').select('id')\
            .eq('code', code).eq('signal_date', str(signal_date)).execute()
        if existing.data:
            return
        sb.table('signal_tracking').insert({
            'code': code, 'name': name,
            'signal_date': str(signal_date),
            'signal_price': int(signal_price),
            'prob': float(prob),
            'checked': False,
        }).execute()
    except Exception as e:
        pass

def update_signal_results():
    """5일 후 실제 수익률 업데이트"""
    try:
        sb  = get_supabase()
        res = sb.table('signal_tracking').select('*')\
            .eq('checked', False).execute()
        if not res.data:
            return 0
        updated = 0
        for row in res.data:
            signal_dt = pd.Timestamp(row['signal_date'])
            # 5거래일 후 확인
            if pd.Timestamp.now() < signal_dt + pd.Timedelta(days=8):
                continue
            try:
                df_raw = get_raw_data(row['code'])
                df_raw.index = pd.to_datetime(df_raw.index)
                after = df_raw[df_raw.index > signal_dt]
                if len(after) < FUTURE_DAYS:
                    continue
                result_price = int(after['close'].iloc[FUTURE_DAYS-1])
                result_date  = after.index[FUTURE_DAYS-1].date()
                actual_ret   = (result_price - row['signal_price']) / row['signal_price'] * 100
                is_correct   = actual_ret >= RISE_THRESHOLD * 100
                sb.table('signal_tracking').update({
                    'result_price': result_price,
                    'result_date': str(result_date),
                    'actual_ret': round(float(actual_ret), 2),
                    'is_correct': bool(is_correct),
                    'checked': True,
                }).eq('id', row['id']).execute()
                updated += 1
            except:
                pass
        return updated
    except:
        return 0
    
def send_signal_email(top20_df):
    """매수 신호 TOP 20을 이메일로 발송"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    if not GMAIL_APP_PW:
        st.warning('Gmail 앱 비밀번호가 설정되지 않았습니다.')
        return

    try:
        body = f'📈 LEADER DL Filter 매수 신호 TOP 20\n'
        body += f'기준 시각: {pd.Timestamp.now().strftime("%Y.%m.%d %H:%M")}\n'
        body += f'임계값: {THRESHOLD_PRED*100:.0f}%\n\n'
        body += '─' * 50 + '\n'

        for i, (_, row) in enumerate(top20_df.iterrows()):
            body += f"{i+1}위 {row['name']} ({row['code']})\n"
            body += f"   매수확률: {float(row['prob']):.1%} | 현재가: {int(row['price']):,}원\n"
            body += f"   ATR손절: {int(row['stop_atr']):,}원 | 신호: {row['signal']}\n\n"

        msg = MIMEMultipart()
        msg['From']    = GMAIL_ADDRESS
        msg['To']      = GMAIL_ADDRESS
        msg['Subject'] = f'[LEADER DL] 매수 신호 TOP 20 — {pd.Timestamp.now().strftime("%Y.%m.%d")}'
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            server.send_message(msg)

        st.success('✅ 이메일 발송 완료!')
    except Exception as e:
        st.error(f'이메일 발송 실패: {e}')
                
# ============================================================
# UI
# ============================================================
st.title('📈 LEADER DL Filter')
st.caption('KOSPI 캔들차트 딥러닝 매수 타이밍 필터 | LSTM 기반')

# 모델 로드 (최초 실행 시 코스피 전체 학습)
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.warning('⏳ 최초 실행입니다. 코스피 전체 종목 데이터 수집 및 학습을 시작합니다. (약 3~5시간 소요)')

with st.spinner('모델 준비 중...'):
    model, scalers, featured_data, raw_data = load_model_and_data()
st.success(f'✅ 모델 준비 완료 | {len(featured_data)}개 종목 로드됨')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '📊 기간별 차트', '🔍 종목 분석', '💼 내 보유 주식',
    '🧪 백테스트', '🤖 자동 성능 점검', '🏆 매수 신호 TOP 20'
])

# ── 탭1: 기간별 차트 ──
with tab1:
    all_stocks = get_kospi_all_stocks()
    c1,c2,c3 = st.columns([3,2,2])
    with c1:
        # 종목 검색
        search_q = st.text_input('종목 검색 (이름 또는 코드)', placeholder='예: 삼성전자 또는 005930', key='tab1_search')
        if search_q:
            found = search_stock(search_q)
            if found:
                opt_labels = [f'{n} ({c})' for c,n in found.items()]
                sel_label  = st.selectbox('종목 선택', options=opt_labels, key='tab1_sel')
                selected   = list(found.keys())[opt_labels.index(sel_label)]
                sel_name   = found[selected]
            else:
                st.warning('검색 결과 없음')
                st.stop()
        else:
            # 학습된 종목 중 선택
            opts = {f'{n} ({c})': c for c,n in all_stocks.items() if c in featured_data}
            if not opts:
                st.warning('학습된 종목이 없습니다. 사이드바에서 🔄 모델 재학습을 눌러주세요.')
                st.stop()
            sel_label = st.selectbox('종목 선택', options=list(opts.keys()), key='tab1_sel_all')
            selected  = opts[sel_label]
            sel_name  = all_stocks.get(selected, selected)
    with c2:
        from datetime import date, timedelta
        if featured_data:
            all_dates = []
            for df in list(featured_data.values())[:10]:  # 일부만 확인
                all_dates.extend(df.index.tolist())
            max_d = max(all_dates).date()
            min_d = min(all_dates).date()
        else:
            max_d = date.today()
            min_d = date(2020,1,1)
        start_d = st.date_input('시작일', value=max_d-timedelta(days=90), min_value=min_d, max_value=max_d)
    with c3:
        end_d = st.date_input('종료일', value=max_d, min_value=min_d, max_value=max_d)

    btn_cols = st.columns(6)
    for i,(label,days) in enumerate([('1주',7),('1개월',30),('3개월',90),('6개월',180),('1년',365),('전체',9999)]):
        if btn_cols[i].button(label, key=f'p{i}'):
            start_d = max_d - timedelta(days=days) if days<9999 else min_d

    if st.button('📊 차트 보기', type='primary'):
        # 학습된 종목이면 바로, 아니면 데이터 수집
        if selected in featured_data:
            draw_chart(featured_data[selected], selected, sel_name,
                       pd.Timestamp(start_d), pd.Timestamp(end_d), model, scalers)
        else:
            with st.spinner(f'{sel_name} 데이터 수집 중...'):
                try:
                    df_raw = get_raw_data(selected)
                    featured_data[selected] = add_features(df_raw)
                    draw_chart(featured_data[selected], selected, sel_name,
                               pd.Timestamp(start_d), pd.Timestamp(end_d), model, scalers)
                except Exception as e:
                    st.error(f'데이터 수집 실패: {e}')

# ── 탭2: 종목 분석 ──
with tab2:
    st.subheader('🔍 종목 분석 → 20일 차트 + 터틀 손절선')

    # TOP 20 빠른 선택
    df_top20_quick = load_top20_from_supabase()
    if not df_top20_quick.empty:
        st.caption('🏆 TOP 20 빠른 선택')
        top20_opts = ['직접 검색'] + [f"{row['name']} ({row['code']}) — {float(row['prob']):.1%}" for _, row in df_top20_quick.iterrows()]
        top20_sel  = st.selectbox('TOP 20에서 선택', options=top20_opts, key='top20_quick')
        if top20_sel != '직접 검색':
            quick_code = top20_sel.split('(')[1].split(')')[0].strip()
            quick_name = top20_sel.split('(')[0].strip()
            if st.button('📊 바로 분석', type='primary', key='top20_analyze'):
                _run_analysis(quick_code, quick_name, model, scalers, featured_data)
        st.divider()

    c1,c2 = st.columns([3,1])
    with c1:
        name_query = st.text_input('종목명 또는 종목코드', placeholder='예: 셀트리온 또는 068270')
    with c2:
        st.write(''); search_btn = st.button('🔍 검색 & 분석', type='primary')

    if search_btn and name_query:
        results = search_stock(name_query)  # {code: name}
        if not results:
            st.warning('검색 결과 없음')
        elif len(results) == 1:
            new_code = list(results.keys())[0]
            new_name = list(results.values())[0]
            st.info(f'✅ {new_name} ({new_code})')
            _run_analysis(new_code, new_name, model, scalers, featured_data)
        else:
            opt_labels = [f'{n} ({c})' for c,n in results.items()]
            sel = st.selectbox('종목 선택', options=opt_labels)
            sel_code = list(results.keys())[opt_labels.index(sel)]
            sel_name = results[sel_code]
            if st.button('📊 분석 시작', type='primary', key='analyze_sel'):
                _run_analysis(sel_code, sel_name, model, scalers, featured_data)

# ── 탭3: 내 보유 주식 ──
with tab3:
    st.subheader('💼 내 보유 주식 관리')
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_portfolio()

    all_stocks = get_kospi_all_stocks()
    with st.expander('➕ 종목 추가 / 수정', expanded=len(st.session_state.portfolio)==0):
        # 종목 검색으로 추가
        add_query = st.text_input('종목명 또는 코드 검색', placeholder='예: 삼성전자', key='add_query')
        add_results = search_stock(add_query) if add_query else {}
        if add_results:
            opt_labels = [f'{n} ({c})' for c,n in add_results.items()]
            add_sel    = st.selectbox('종목 선택', options=opt_labels, key='add_sel')
            add_code   = list(add_results.keys())[opt_labels.index(add_sel)]
            add_name   = add_results[add_code]
        else:
            add_code, add_name = None, None

        c2, c3, c4 = st.columns([2, 2, 1])
        with c2:
            add_total = st.number_input('총매입금액 (원)', min_value=0, step=10000, key='add_total')
        with c3:
            add_qty = st.number_input('수량 (주)', min_value=0, step=1, key='add_qty')
        with c4:
            st.write(''); save_btn = st.button('💾 저장', key='save_stock', type='primary')

        if add_total > 0 and add_qty > 0 and add_code:
            st.caption(f'→ 주당 평균단가: {int(add_total/add_qty):,}원')
        if save_btn:
            if add_code and add_total > 0 and add_qty > 0:
                avg_price = add_total / add_qty
                current_portfolio = load_portfolio()
                current_portfolio[add_code] = {
                    'name': add_name, 'price': avg_price,
                    'qty': add_qty, 'total': add_total
                }
                save_portfolio(current_portfolio)
                st.session_state.portfolio = current_portfolio
                st.success(f'✅ {add_name} 저장 완료!')
                st.rerun()
            else:
                st.error('종목 선택 후 금액과 수량을 입력하세요.')

    portfolio = load_portfolio()
    if portfolio:
        del_options = ['선택 안 함'] + [f"{v['name']} ({k})" for k,v in portfolio.items()]
        c1, c2 = st.columns([4,1])
        with c1:
            del_sel = st.selectbox('삭제할 종목', options=del_options, key='del_stock', label_visibility='collapsed')
        with c2:
            if st.button('🗑️ 삭제', key='del_btn') and del_sel != '선택 안 함':
                code_to_del = del_sel.split('(')[-1].replace(')','').strip()
                portfolio.pop(code_to_del, None)
                save_portfolio(portfolio)
                st.session_state.portfolio = portfolio
                st.rerun()
    else:
        st.info('보유 종목을 추가해주세요. 위 ➕ 버튼을 눌러 종목을 등록하세요.')

    st.divider()
    portfolio = load_portfolio()
    if not portfolio:
        st.info('보유 종목이 없습니다.')
    else:
        if st.button('🔄 현황 새로고침', type='primary'):
            st.rerun()

        summary_rows = []
        detail_data  = {}
        atr_mult = st.session_state.get('atr_mult', 2.0)

        for code, info in portfolio.items():
            name      = info['name']
            buy_price = info['price']
            qty       = info['qty']
            total_buy = info.get('total', buy_price * qty)
            try:
                df_raw    = get_raw_data(code)
                df_feat   = add_features(df_raw)
                close_now, atr14, stop_atr, stop_10low, stop_20low, unit = calc_turtle(df_raw, atr_mult)
                _, probs, signals = get_predictions(df_feat, code, model, scalers)
                latest_prob   = probs[-1]
                latest_signal = '🟢 매수' if latest_prob >= THRESHOLD_PRED else '🔴 비신호'
                eval_amt  = int(close_now * qty)
                pnl       = eval_amt - int(total_buy)
                pnl_pct   = (close_now - buy_price) / buy_price * 100
                stop_hit  = '⚠️ 손절' if close_now <= stop_atr else '✅ 정상'
                detail_data[code] = {
                    'df_feat': df_feat, 'close_now': close_now,
                    'stop_atr': stop_atr, 'stop_10low': stop_10low,
                    'stop_20low': stop_20low, 'atr14': atr14,
                    'latest_prob': latest_prob, 'latest_signal': latest_signal,
                }
                summary_rows.append({
                    '종목': name, '주당단가': f'{int(buy_price):,}원',
                    '수량': f'{qty}주', '총매입': f'{int(total_buy):,}원',
                    '현재가': f'{int(close_now):,}원', '평가금액': f'{eval_amt:,}원',
                    '손익': f'{pnl:+,}원', '수익률': f'{pnl_pct:+.1f}%',
                    f'ATR손절': f'{int(stop_atr):,}원',
                    '10일손절': f'{int(stop_10low):,}원',
                    '20일손절': f'{int(stop_20low):,}원',
                    'DL신호': f'{latest_signal}({latest_prob:.0%})',
                    '상태': stop_hit,
                })
            except Exception as e:
                summary_rows.append({
                    '종목': name, '주당단가': f'{int(buy_price):,}원',
                    '수량': f'{qty}주', '총매입': f'{int(total_buy):,}원',
                    '현재가': '조회실패', '평가금액': '-', '손익': '-', '수익률': '-',
                    'ATR손절': '-', '10일손절': '-', '20일손절': '-',
                    'DL신호': '-', '상태': f'❌{e}',
                })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
        try:
            total_buy_sum  = sum(info.get('total', info['price']*info['qty']) for info in portfolio.values())
            total_eval_sum = sum(detail_data[c]['close_now'] * portfolio[c]['qty'] for c in detail_data)
            total_pnl = total_eval_sum - total_buy_sum
            total_pct = total_pnl / total_buy_sum * 100 if total_buy_sum > 0 else 0
            m1,m2,m3,m4 = st.columns(4)
            m1.metric('총 매입금액', f'{int(total_buy_sum):,}원')
            m2.metric('총 평가금액', f'{int(total_eval_sum):,}원')
            m3.metric('총 손익', f'{int(total_pnl):+,}원')
            m4.metric('총 수익률', f'{total_pct:+.1f}%', delta_color='normal' if total_pct>=0 else 'inverse')
        except:
            pass

        st.divider()
        st.subheader('📊 종목별 상세')
        for code, info in portfolio.items():
            if code not in detail_data: continue
            d = detail_data[code]; name = info['name']
            with st.expander(f"{name} ({code})  |  {d['latest_signal']}  |  현재가 {int(d['close_now']):,}원  |  {'⚠️손절' if d['close_now']<=d['stop_atr'] else '✅정상'}"):
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric('현재가', f"{int(d['close_now']):,}원")
                mc1.metric('DL 확률', f"{d['latest_prob']:.1%}")
                mc2.metric(f'ATR×{atr_mult:.1f} 손절가', f"{int(d['stop_atr']):,}원",
                           f"-{(d['close_now']-d['stop_atr'])/d['close_now']*100:.1f}%", delta_color='inverse')
                mc2.metric('10일 손절가', f"{int(d['stop_10low']):,}원",
                           f"-{(d['close_now']-d['stop_10low'])/d['close_now']*100:.1f}%", delta_color='inverse')
                mc3.metric('20일 손절가', f"{int(d['stop_20low']):,}원",
                           f"-{(d['close_now']-d['stop_20low'])/d['close_now']*100:.1f}%", delta_color='inverse')
                mc3.metric('ATR(14)', f"{int(d['atr14']):,}원")
                if st.button(f'📈 차트 보기', key=f'chart_{code}'):
                    from datetime import date, timedelta
                    max_d = d['df_feat'].index.max().date()
                    draw_chart(d['df_feat'], code, name,
                               pd.Timestamp(max_d - timedelta(days=90)),
                               pd.Timestamp(max_d), model, scalers)

# ── 탭4: 백테스트 ──
with tab4:
    st.subheader('🧪 백테스트 — 매수 후 손절까지 수익률 시뮬레이션')
    atr_mult = st.session_state.get('atr_mult', 2.0)
    gap_slip = st.session_state.get('gap_slip', True)

    c1, c2, c3 = st.columns(3)
    with c1:
        bt_query = st.text_input('종목명 또는 코드', placeholder='예: 삼성전자 또는 005930', key='bt_query')
        if bt_query:
            bt_found = search_stock(bt_query)
            if bt_found:
                bt_opt_labels = [f'{n} ({c})' for c,n in bt_found.items()]
                bt_sel_label  = st.selectbox('종목 선택', options=bt_opt_labels, key='bt_sel')
                bt_code = list(bt_found.keys())[bt_opt_labels.index(bt_sel_label)]
                bt_name = bt_found[bt_code]
            else:
                st.warning('검색 결과 없음'); st.stop()
        else:
            bt_opts = {f'{n} ({c})': c for c,n in get_kospi_all_stocks().items() if c in featured_data}
            if not bt_opts:
                st.warning('학습된 종목이 없습니다.'); st.stop()
            bt_sel_label = st.selectbox('종목 선택', options=list(bt_opts.keys()), key='bt_sel_all')
            bt_code = bt_opts[bt_sel_label]
            bt_name = bt_sel_label.split('(')[0].strip()
    with c2:
        from datetime import date, timedelta
        bt_start = st.date_input('매수일', value=date(2025,1,2), key='bt_start')
    with c3:
        bt_stop_type = st.selectbox('손절 기준',
            [f'ATR×{atr_mult:.1f} 손절', '10일 최저가', '20일 최저가', '기간 만료(보유일 지정)'],
            key='bt_stop')

    hold_days = None
    if '기간 만료' in bt_stop_type:
        hold_days = st.slider('보유 기간 (거래일)', 1, 60, 20)

    st.info(f'📌 임계값: **{THRESHOLD_PRED*100:.0f}%** | ATR 배수: **{atr_mult:.1f}배** | 갭다운 체결: **{"ON" if gap_slip else "OFF"}**')

    if st.button('🧪 백테스트 실행', type='primary'):
        # 데이터 확보
        if bt_code not in featured_data:
            with st.spinner(f'{bt_name} 데이터 수집 중...'):
                try:
                    df_raw = get_raw_data(bt_code)
                    featured_data[bt_code] = add_features(df_raw)
                except Exception as e:
                    st.error(f'데이터 수집 실패: {e}'); st.stop()

        df = featured_data.get(bt_code)
        if df is None:
            st.error('데이터 없음')
        else:
            df_raw = df.copy(); df_raw.index = pd.to_datetime(df_raw.index)
            bt_start_ts = pd.Timestamp(bt_start)
            avail = df_raw.index[df_raw.index >= bt_start_ts]
            if len(avail) == 0:
                st.error('해당 날짜 이후 데이터 없음')
            else:
                entry_dt    = avail[0]
                entry_price = df_raw.loc[entry_dt, 'close']
                entry_idx   = df_raw.index.get_loc(entry_dt)
                df_before   = df_raw.iloc[:entry_idx+1]
                tr = pd.concat([
                    df_before['high'] - df_before['low'],
                    (df_before['high'] - df_before['close'].shift()).abs(),
                    (df_before['low']  - df_before['close'].shift()).abs()
                ], axis=1).max(axis=1)
                atr14_entry = tr.rolling(14).mean().iloc[-1]

                if 'ATR' in bt_stop_type:
                    if pd.isna(atr14_entry) or atr14_entry <= 0:
                        st.error('ATR 계산 불가 — 매수일 이전 데이터가 부족합니다. 더 이후 날짜를 선택해주세요.')
                        st.stop()
                    stop_price = entry_price - atr_mult * atr14_entry
                    stop_label = f'매수가({int(entry_price):,}) - {atr_mult:.1f}×ATR({int(atr14_entry):,})'
                elif '10일' in bt_stop_type:
                    stop_price = df_before['low'].tail(10).min()
                    stop_label = '매수일 기준 과거 10일 최저가'
                elif '20일' in bt_stop_type:
                    stop_price = df_before['low'].tail(20).min()
                    stop_label = '매수일 기준 과거 20일 최저가'
                else:
                    stop_price = None
                    stop_label = f'{hold_days}거래일 보유 후 청산'

                df_after = df_raw.iloc[entry_idx+1:]
                exit_dt, exit_price, exit_reason = None, None, None
                slippage_note = ''

                if hold_days:
                    for idx_i, (dt, row) in enumerate(df_after.iterrows()):
                        if stop_price and row['open'] <= stop_price:
                            exit_dt, exit_price = dt, row['open']
                            exit_reason = '갭다운 시가 손절'
                            slippage_note = f'⚠️ 갭다운: 손절가 {int(stop_price):,}원 → 시가 {int(row["open"]):,}원 체결'
                            break
                        if idx_i >= hold_days - 1:
                            exit_dt, exit_price = dt, row['close']
                            exit_reason = f'{hold_days}거래일 후 청산'; break
                    if exit_dt is None:
                        exit_dt = df_after.index[-1]; exit_price = df_after['close'].iloc[-1]; exit_reason = '데이터 끝'
                else:
                    trailing_stop = stop_price  # 초기 손절가
                    for dt, row in df_after.iterrows():
                        # 트레일링 스탑: 10일/20일 기준이면 매일 손절가 갱신
                        if '10일' in bt_stop_type:
                            idx_now = df_raw.index.get_loc(dt)
                            trailing_stop = df_raw['low'].iloc[max(0, idx_now-10):idx_now].min()
                        elif '20일' in bt_stop_type:
                            idx_now = df_raw.index.get_loc(dt)
                            trailing_stop = df_raw['low'].iloc[max(0, idx_now-20):idx_now].min()
                        # ATR 트레일링: 매일 ATR 재계산 후 손절가 갱신
                        if 'ATR' in bt_stop_type:
                            idx_now = df_raw.index.get_loc(dt)
                            df_slice = df_raw.iloc[max(0, idx_now-30):idx_now]
                            tr_s = pd.concat([
                                df_slice['high'] - df_slice['low'],
                                (df_slice['high'] - df_slice['close'].shift()).abs(),
                                (df_slice['low']  - df_slice['close'].shift()).abs()
                            ], axis=1).max(axis=1)
                            atr_now = tr_s.rolling(14).mean().iloc[-1]
                            new_stop = row['close'] - atr_mult * atr_now
                            # 손절가는 올라가기만 함 (내려가지 않음)
                            trailing_stop = max(trailing_stop, new_stop)

                        # 갭다운 체크
                        if gap_slip and row['open'] <= trailing_stop:
                            exit_dt, exit_price = dt, row['open']
                            exit_reason = '갭다운 시가 손절'
                            slippage_note = f'⚠️ 갭다운: 손절가 {int(trailing_stop):,}원 → 시가 {int(row["open"]):,}원 체결'
                            break
                        # 일중 저가 터치
                        if row['low'] <= trailing_stop:
                            exit_dt, exit_price = dt, trailing_stop
                            exit_reason = f'트레일링 손절 ({bt_stop_type})'
                            break
                    if exit_dt is None:
                        exit_dt = df_after.index[-1]; exit_price = df_after['close'].iloc[-1]
                        exit_reason = '손절 미도달 (현재까지 보유)'
                    stop_price = trailing_stop  # 최종 손절가 표시용

                ret  = (exit_price - entry_price) / entry_price * 100
                hold = len(df_raw.loc[entry_dt:exit_dt]) - 1
                dates_arr, probs, _ = get_predictions(df, bt_code, model, scalers)
                prob_on_entry = next((p for dt, p in zip(dates_arr, probs) if dt == entry_dt), 0.0)

                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('매수일', entry_dt.strftime('%Y.%m.%d'))
                col1.metric('매수가', f'{int(entry_price):,}원')
                col2.metric('청산일', exit_dt.strftime('%Y.%m.%d'))
                col2.metric('청산가', f'{int(exit_price):,}원')
                col3.metric('수익률', f'{ret:+.1f}%', delta_color='normal' if ret>=0 else 'inverse')
                col3.metric('보유기간', f'{hold}거래일')
                col4.metric('청산 이유', exit_reason)
                col4.metric('DL 확률 (매수일)', f'{prob_on_entry:.1%}',
                            '신호 있음' if prob_on_entry>=THRESHOLD_PRED else '신호 없음')
                if stop_price:
                    loss_pct = (stop_price - entry_price) / entry_price * 100
                    st.caption(f'📌 손절 기준: {stop_label}')
                    st.caption(f'📌 이론 손절가: {int(stop_price):,}원  ({loss_pct:.1f}%)')
                    if slippage_note: st.warning(slippage_note)

                chart_start = pd.Timestamp(bt_start) - pd.Timedelta(days=10)
                chart_end   = exit_dt + pd.Timedelta(days=5)
                df_chart = df_raw.loc[(df_raw.index >= chart_start) & (df_raw.index <= chart_end)].copy()
                n = len(df_chart); xs = list(range(n))
                dt_to_x = {dt:i for i,dt in enumerate(df_chart.index)}
                dt_labels = [d.strftime('%m.%d') for d in df_chart.index]
                fig, ax = plt.subplots(figsize=(14,5))
                for i,(dt,row) in enumerate(df_chart.iterrows()):
                    c = '#E74C3C' if row['close']>=row['open'] else '#3498DB'
                    ax.plot([i,i],[row['low'],row['high']],color=c,linewidth=0.8,alpha=0.8)
                    ax.bar(i,abs(row['close']-row['open']),bottom=min(row['open'],row['close']),color=c,alpha=0.8,width=0.6)
                ax.plot(xs, df_chart['close'].values, color='#333333', linewidth=1.1, alpha=0.6, label='종가')
                if entry_dt in dt_to_x:
                    ax.scatter(dt_to_x[entry_dt], entry_price*1.015, marker='v', s=200, color='#27AE60', zorder=7, label=f'매수 {int(entry_price):,}원')
                if exit_dt and exit_dt in dt_to_x:
                    ax.scatter(dt_to_x[exit_dt], exit_price*0.985, marker='^', s=200,
                               color='#E74C3C' if ret<0 else '#27AE60', zorder=7, label=f'청산 {int(exit_price):,}원')
                if stop_price:
                    ax.axhline(stop_price, color='#E74C3C', linestyle='--', linewidth=1.5, label=f'손절가 {int(stop_price):,}원')
                if entry_dt in dt_to_x and exit_dt and exit_dt in dt_to_x:
                    ax.axvspan(dt_to_x[entry_dt], dt_to_x[exit_dt], alpha=0.08, color='#27AE60' if ret>=0 else '#E74C3C')
                ax.set_title(f'{bt_name} 백테스트 | 수익률: {ret:+.1f}% | {exit_reason}', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.2)
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
                tick_step = max(1, n//12)
                ax.set_xticks(xs[::tick_step]); ax.set_xticklabels(dt_labels[::tick_step], rotation=30, fontsize=9)
                ax.set_xlim(-0.8, n-0.2)
                plt.tight_layout(); st.pyplot(fig); plt.close()

# ── 탭5: 자동 성능 점검 ──
with tab5:
    st.subheader('🤖 자동 성능 점검 & 모델 개선 시스템')
    st.info('''
    - **정확도 점검**: 과거 신호가 실제로 맞았는지 검증
    - **신호 품질 분석**: 확률별 실제 수익률 분포
    - **재학습 권고**: 정확도가 기준 이하면 자동 알림
    ''')
    if st.button('🔍 성능 점검 시작', type='primary'):
        all_results = []
        codes_to_check = list(featured_data.keys())
        prog = st.progress(0)
        for i, code in enumerate(codes_to_check):
            name = get_kospi_all_stocks().get(code, code)
            prog.progress((i+1)/len(codes_to_check), text=f'{name} 점검 중... {i+1}/{len(codes_to_check)}')
            try:
                df = featured_data[code]
                dates_arr, probs, signals = get_predictions(df, code, model, scalers)
                df_raw = df.copy(); df_raw.index = pd.to_datetime(df_raw.index)
                for dt, prob, sig in zip(dates_arr, probs, signals):
                    idx = df_raw.index.get_loc(dt) if dt in df_raw.index else None
                    if idx is None or idx+FUTURE_DAYS >= len(df_raw): continue
                    future_price = df_raw['close'].iloc[idx+FUTURE_DAYS]
                    curr_price   = df_raw['close'].iloc[idx]
                    actual_ret   = (future_price - curr_price) / curr_price * 100
                    all_results.append({
                        'code': code, 'name': name, 'date': dt,
                        'prob': prob, 'signal': sig,
                        'actual_ret': actual_ret,
                        'actual_up': actual_ret >= RISE_THRESHOLD*100,
                    })
            except: pass
        prog.empty()

        if not all_results:
            st.error('점검 데이터 없음')
        else:
            df_res  = pd.DataFrame(all_results)
            sig_df  = df_res[df_res['signal']]
            overall_acc = sig_df['actual_up'].mean() if len(sig_df) > 0 else 0
            st.divider()
            m1,m2,m3,m4 = st.columns(4)
            m1.metric('매수신호 정확도', f'{overall_acc:.1%}', '✅ 양호' if overall_acc>=0.55 else '⚠️ 개선 필요')
            m2.metric('총 신호 횟수', f'{len(sig_df)}회')
            m3.metric('신호 시 평균 수익률', f'{sig_df["actual_ret"].mean():+.1f}%' if len(sig_df)>0 else '-')
            m4.metric('전체 평균 수익률', f'{df_res["actual_ret"].mean():+.1f}%')

            if overall_acc < 0.55:
                st.error('⚠️ 모델 정확도가 55% 미만입니다. 재학습을 권장합니다.')
                if st.button('🔄 즉시 재학습', type='secondary'):
                    for f in [MODEL_PATH, SCALER_PATH]:
                        if os.path.exists(f): os.remove(f)
                    st.cache_resource.clear(); st.rerun()
            else:
                st.success('✅ 모델 성능이 양호합니다.')

            st.subheader('📊 확률 구간별 정확도')
            bin_rows = []
            for lo,hi in [(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
                sub = df_res[(df_res['prob']>=lo)&(df_res['prob']<hi)]
                if len(sub)==0: continue
                bin_rows.append({'확률 구간':f'{lo*100:.0f}~{hi*100:.0f}%','신호 횟수':len(sub),
                                 '정확도':f"{sub['actual_up'].mean():.1%}",'평균 수익률':f"{sub['actual_ret'].mean():+.1f}%",
                                 '최대 수익':f"{sub['actual_ret'].max():+.1f}%",'최대 손실':f"{sub['actual_ret'].min():+.1f}%"})
            if bin_rows: st.dataframe(pd.DataFrame(bin_rows), use_container_width=True)

            st.subheader('⚙️ 최적 임계값 탐색')
            best_thresh, best_acc = THRESHOLD_PRED, 0
            thresh_rows = []
            for thresh in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
                sub = df_res[df_res['prob']>=thresh]
                if len(sub) < 10: continue
                acc = sub['actual_up'].mean()
                thresh_rows.append({'임계값':f'{thresh*100:.0f}%','신호 횟수':len(sub),
                                    '정확도':f'{acc:.1%}','평균 수익률':f"{sub['actual_ret'].mean():+.1f}%",
                                    '현재 선택':'✅' if abs(thresh-THRESHOLD_PRED)<0.01 else ''})
                if acc > best_acc: best_acc, best_thresh = acc, thresh
            if thresh_rows:
                st.dataframe(pd.DataFrame(thresh_rows), use_container_width=True)
                st.success(f'🎯 최적 임계값: **{best_thresh*100:.0f}%** (정확도 {best_acc:.1%})')
                st.info(f'💡 사이드바 슬라이더에서 임계값을 **{best_thresh*100:.0f}%** 로 조정해보세요.')

    st.divider()
    st.subheader('📡 신호 추적 현황')

    # 5일 후 결과 업데이트
    if st.button('🔄 결과 업데이트', key='update_tracking'):
        with st.spinner('5일 후 수익률 업데이트 중...'):
            updated = update_signal_results()
            st.success(f'✅ {updated}개 신호 업데이트 완료')

    try:
        sb  = get_supabase()
        res = sb.table('signal_tracking').select('*')\
            .order('signal_date', desc=True).limit(50).execute()
        if res.data:
            df_track = pd.DataFrame(res.data)
            # 완료된 신호만 정확도 계산
            done = df_track[df_track['checked'] == True]
            if len(done) > 0:
                acc = done['is_correct'].mean()
                avg_ret = done['actual_ret'].mean()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric('실시간 정확도', f'{acc:.1%}')
                m2.metric('평균 수익률', f'{avg_ret:+.1f}%')
                m3.metric('추적 완료', f'{len(done)}회')
                m4.metric('추적 중', f'{len(df_track)-len(done)}회')

            # 테이블 표시
            display_rows = []
            for _, row in df_track.iterrows():
                display_rows.append({
                    '종목': row['name'],
                    '신호일': row['signal_date'],
                    '신호가': f"{int(row['signal_price']):,}원",
                    '확률': f"{float(row['prob']):.1%}",
                    '결과가': f"{int(row['result_price']):,}원" if row['result_price'] else '추적 중',
                    '수익률': f"{float(row['actual_ret']):+.1f}%" if row['actual_ret'] else '-',
                    '결과': '✅' if row['is_correct'] else ('❌' if row['checked'] else '⏳'),
                })
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True)
        else:
            st.info('아직 추적된 신호가 없습니다. 종목 분석 시 매수 신호가 발생하면 자동으로 기록됩니다.')
    except Exception as e:
        st.error(f'추적 데이터 로드 실패: {e}')
                        
# ── 탭6: 매수 신호 TOP 20 ──
with tab6:
    st.subheader('🏆 매수 신호 TOP 20')
    st.info('학습된 전체 종목 중 현재 매수 확률이 높은 종목 TOP 20을 보여줍니다.')


        
# 저장된 결과 불러오기
df_top20 = load_top20_from_supabase()

if not df_top20.empty:
    updated = pd.to_datetime(df_top20['updated_at'].iloc[0])
    st.caption(f'마지막 스캔: {updated.strftime("%Y.%m.%d %H:%M")}')

if st.button('🔄 전체 종목 스캔 (약 10~20분)', type='primary'):
    all_stocks = get_kospi_all_stocks()
    results = []
    pb  = st.progress(0, text='전체 종목 스캔 중...')
    total = len(featured_data)

    for i, (code, df) in enumerate(featured_data.items()):
        name = all_stocks.get(code, code)
        pb.progress((i+1)/total, text=f'스캔 중... {i+1}/{total} — {name}')
        try:
            _, probs, _ = get_predictions(df, code, model, scalers)
            latest_prob = probs[-1]
            if latest_prob < 0.5:
                continue
            atr_mult = st.session_state.get('atr_mult', 2.0)
            close_now, atr14, stop_atr, stop_10low, stop_20low, _ = calc_turtle(df, atr_mult)
            results.append({
                'code': code, 'name': name,
                'prob': round(float(latest_prob), 4),
                'price': int(close_now),
                'stop_atr': int(stop_atr),
                'stop_10low': int(stop_10low),
                'stop_20low': int(stop_20low),
                'signal': '🟢 매수' if latest_prob >= THRESHOLD_PRED else '🟡 관심',
            })
        except:
            pass
    pb.empty()

    if results:
        df_result = pd.DataFrame(results).sort_values('prob', ascending=False).head(20)
        save_top20_to_supabase(df_result)
        df_top20 = df_result
        st.success(f'✅ 스캔 완료 — {len(results)}개 종목 중 TOP 20 선정')
        if st.button('📧 이메일로 받기', key='send_email'):
            send_signal_email(df_result)
    else:
        st.error('스캔 결과 없음')

# 결과 표시
if not df_top20.empty:
    st.divider()
    display_rows = []
    for _, row in df_top20.iterrows():
        display_rows.append({
            '순위': f"{list(df_top20.index).index(_)+1}위" if hasattr(_, '__index__') else '',
            '종목': row['name'],
            '코드': row['code'],
            '매수 확률': f"{float(row['prob']):.1%}",
            '현재가': f"{int(row['price']):,}원",
            'ATR 손절': f"{int(row['stop_atr']):,}원",
            '10일 손절': f"{int(row['stop_10low']):,}원",
            '20일 손절': f"{int(row['stop_20low']):,}원",
            '신호': row['signal'],
        })
    df_display = pd.DataFrame(display_rows)
    df_display['순위'] = [f'{i+1}위' for i in range(len(df_display))]
    cols = ['순위'] + [c for c in df_display.columns if c != '순위']
    df_display = df_display[cols]
    st.dataframe(df_display, use_container_width=True)
else:
    st.info('스캔 버튼을 눌러 전체 종목을 스캔해주세요.')                