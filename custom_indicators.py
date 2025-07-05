"""Technical indicator helper functions."""
import pandas as pd
import numpy as np
import logging
from scipy.fft import fft
from ta.trend import MACD, ADXIndicator, SMAIndicator, PSARIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator

EPSILON = 1e-9
DEFAULT_RSI_WINDOW = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGN = 9
DEFAULT_ADX_WINDOW = 14
DEFAULT_BB_WINDOW = 20
DEFAULT_BB_DEV = 2
DEFAULT_SMA_SHORT_PERIOD = 10
DEFAULT_SMA_MEDIUM_PERIOD = 20
DEFAULT_SMA_LONG_PERIOD = 50
DEFAULT_EMA_WINDOW = 10
DEFAULT_ATR_WINDOW = 14
DEFAULT_DONCHIAN_WINDOW = 20
DEFAULT_VWMA_WINDOW = 20
DEFAULT_FFT_WINDOW = 10
DEFAULT_SLOPE_WINDOW = 10
DEFAULT_RETURN_LOOKBACKS = {"Return_3p": 3, "Return_6p": 6, "Return_12p": 12, "Return_24p": 24, "Return_78p": 78}
DEFAULT_PSAR_STEP = 0.02
DEFAULT_PSAR_MAX_STEP = 0.2
DEFAULT_ST_ATR_PERIOD = 10
DEFAULT_ST_MULTIPLIER = 3.0
DEFAULT_CHANDELIER_LOOKBACK = 22
DEFAULT_CHANDELIER_MULTIPLIER = 3.0
DEFAULT_KC_WINDOW = 20
DEFAULT_KC_ATR_WINDOW = 10
DEFAULT_KC_MULTIPLIER = 2.0
DEFAULT_VOL_PROFILE_SLOPE_WINDOW = 5


def calc_slope(series):
    """Calculate the slope of a series using a simple linear regression."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series))
    try:
        slope, _ = np.polyfit(x, series.values, 1)
        return slope
    except Exception:
        return np.nan


def add_macd_features(df, price_col="price", window_slow=DEFAULT_MACD_SLOW,
                      window_fast=DEFAULT_MACD_FAST, window_sign=DEFAULT_MACD_SIGN):
    out = pd.DataFrame(index=df.index)
    if price_col in df:
        macd = MACD(df[price_col], window_slow=window_slow,
                    window_fast=window_fast, window_sign=window_sign, fillna=True)
        out["MACD"] = macd.macd()
        out["MACD_Signal"] = macd.macd_signal()
        out["MACD_Diff"] = macd.macd_diff()
    return out


def add_rsi_features(df, price_col="price", window=DEFAULT_RSI_WINDOW):
    out = pd.DataFrame(index=df.index)
    if price_col in df:
        out[f"RSI_{window}"] = RSIIndicator(df[price_col], window=window, fillna=True).rsi()
    return out


def add_bollinger_bands_features(df, price_col="price", window=DEFAULT_BB_WINDOW,
                                 window_dev=DEFAULT_BB_DEV):
    out = pd.DataFrame(index=df.index)
    if price_col in df:
        bb = BollingerBands(df[price_col], window=window, window_dev=window_dev, fillna=True)
        out["BB_High"] = bb.bollinger_hband()
        out["BB_Low"] = bb.bollinger_lband()
        out["BB_Mid"] = bb.bollinger_mavg()
        out["BB_Width"] = bb.bollinger_wband()
        out["BB_Pband"] = bb.bollinger_pband()
        out["BB_Pband"].replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def add_sma_feature(df, price_col="price", window=DEFAULT_SMA_SHORT_PERIOD):
    out = pd.DataFrame(index=df.index)
    if price_col in df:
        out[f"SMA_{window}"] = SMAIndicator(df[price_col], window=window, fillna=True).sma_indicator()
    return out


def add_ema_feature(df, price_col="price", window=DEFAULT_EMA_WINDOW):
    out = pd.DataFrame(index=df.index)
    if price_col in df:
        out[f"EMA_{window}"] = df[price_col].ewm(span=window, adjust=False, min_periods=max(1, window-1)).mean()
    return out


def add_price_derivative_features(df, price_col="price", volatility_window=10):
    out = pd.DataFrame(index=df.index)
    if price_col in df:
        out["Return"] = df[price_col].pct_change()
        out["Price_Change"] = df[price_col].diff()
        out["Magnitude"] = out["Price_Change"].abs()
        out["Direction"] = np.sign(out["Price_Change"]).fillna(method="ffill").fillna(0).astype(int)
        out["Volatility"] = out["Return"].rolling(window=volatility_window, min_periods=1).std()
    return out


def add_lookback_return_features(df, price_col="price", periods_dict=None):
    if periods_dict is None:
        periods_dict = DEFAULT_RETURN_LOOKBACKS
    out = pd.DataFrame(index=df.index)
    if price_col in df:
        for name, per in periods_dict.items():
            out[name] = df[price_col].pct_change(periods=per)
    return out


def add_adx_features(df, high_col="high", low_col="low", price_col="price", window=DEFAULT_ADX_WINDOW):
    out = pd.DataFrame(index=df.index)
    if all(c in df for c in [high_col, low_col, price_col]):
        adx = ADXIndicator(df[high_col], df[low_col], df[price_col], window=window, fillna=True)
        out["ADX"] = adx.adx()
        out["ADX_PosDI"] = adx.adx_pos()
        out["ADX_NegDI"] = adx.adx_neg()
    return out


def add_atr_and_natr_features(df, high_col="high", low_col="low", price_col="price", window=DEFAULT_ATR_WINDOW):
    out = pd.DataFrame(index=df.index)
    if all(c in df for c in [high_col, low_col, price_col]):
        atr = AverageTrueRange(df[high_col], df[low_col], df[price_col], window=window, fillna=True)
        out[f"ATR_{window}_raw"] = atr.average_true_range()
        safe_close = df[price_col].replace(0, EPSILON)
        out[f"NATR_{window}"] = (out[f"ATR_{window}_raw"] / safe_close) * 100
        out[f"NATR_{window}"] = out[f"NATR_{window}"].replace([np.inf, -np.inf], np.nan)
    return out


def add_donchian_pos_feature(df, high_col="high", low_col="low", price_col="price", window=DEFAULT_DONCHIAN_WINDOW):
    out = pd.DataFrame(index=df.index)
    if all(c in df for c in [high_col, low_col, price_col]):
        dc = DonchianChannel(df[high_col], df[low_col], df[price_col], window=window, fillna=True)
        dc_high = dc.donchian_channel_hband()
        dc_low = dc.donchian_channel_lband()
        dc_mid = (dc_high + dc_low) / 2
        dc_width = dc_high - dc_low
        pos = (df[price_col] - dc_mid) / (dc_width.replace(0, EPSILON))
        out[f"Donchian_Pos_{window}"] = np.clip(pos, -1.5, 1.5)
    return out


def add_vwma_sma_diff_features(df, high_col="high", low_col="low", price_col="price", volume_col="volume",
                              vwma_window=DEFAULT_VWMA_WINDOW, sma_ref_col=f"SMA_{DEFAULT_VWMA_WINDOW}"):
    out = pd.DataFrame(index=df.index)
    if all(c in df for c in [high_col, low_col, price_col, volume_col, sma_ref_col]):
        vwma = VolumeWeightedAveragePrice(df[high_col], df[low_col], df[price_col], df[volume_col],
                                          window=vwma_window, fillna=True).volume_weighted_average_price()
        out[f"VWMA_{vwma_window}"] = vwma
        sma = df[sma_ref_col].replace(0, EPSILON)
        diff = (vwma - df[sma_ref_col]) / (sma + EPSILON)
        out[f"VWMA_SMA_Diff_{vwma_window}"] = diff.replace([np.inf, -np.inf], np.nan)
    return out


def calculate_fft_features_df(df, price_col="price", window=DEFAULT_FFT_WINDOW):
    out = pd.DataFrame(index=df.index)
    if price_col not in df:
        out["fft_real"] = np.nan
        out["fft_imag"] = np.nan
        return out
    real_list, imag_list = [], []
    prices = df[price_col].values
    for i in range(len(prices)):
        start = i - window + 1
        if start >= 0:
            window_data = prices[start:i+1]
            if np.isnan(window_data).any():
                real_list.append(np.nan)
                imag_list.append(np.nan)
                continue
            try:
                vals = fft(window_data)
                real_list.append(np.real(vals[1]) if len(vals) > 1 else 0.0)
                imag_list.append(np.imag(vals[1]) if len(vals) > 1 else 0.0)
            except Exception:
                real_list.append(np.nan)
                imag_list.append(np.nan)
        else:
            real_list.append(np.nan)
            imag_list.append(np.nan)
    out["fft_real"] = real_list
    out["fft_imag"] = imag_list
    return out


def calculate_phase_features_df(df, real_col="fft_real", imag_col="fft_imag"):
    out = pd.DataFrame(index=df.index)
    if real_col in df and imag_col in df:
        re = df[real_col].fillna(0)
        im = df[imag_col].fillna(0)
        phase = np.arctan2(im, re)
        out["phase_angle"] = np.degrees(phase)
        start = phase.iloc[0] if len(phase) > 0 and pd.notna(phase.iloc[0]) else 0
        diff = np.diff(phase, prepend=start)
        out["phase_change"] = np.degrees((diff + np.pi) % (2*np.pi) - np.pi)
    else:
        out["phase_angle"] = np.nan
        out["phase_change"] = np.nan
    return out


def add_rolling_slopes(df, columns_to_slope=None, slope_window=DEFAULT_SLOPE_WINDOW):
    out = pd.DataFrame(index=df.index)
    if columns_to_slope is None:
        columns_to_slope = ["price"]
    for col in columns_to_slope:
        if col in df:
            slope = df[col].rolling(window=slope_window, min_periods=max(2, slope_window//2)).apply(calc_slope, raw=True)
            out[f"Slope_{col}_{slope_window}p"] = slope.replace([np.inf, -np.inf], np.nan)
    return out


def add_adx_derived_features(df, adx_col="ADX", pos_di_col="ADX_PosDI", neg_di_col="ADX_NegDI"):
    out = pd.DataFrame(index=df.index)
    if all(c in df for c in [adx_col, pos_di_col, neg_di_col]):
        out["DI_Diff"] = df[pos_di_col] - df[neg_di_col]
        out["ADX_Trend_Confirm"] = df[adx_col] * out["DI_Diff"]
    return out


def add_macd_volatility_interaction(df, macd_diff_col="MACD_Diff", volatility_col="Volatility"):
    out = pd.DataFrame(index=df.index)
    if all(c in df for c in [macd_diff_col, volatility_col]):
        out["MACD_Diff_x_Vol"] = df[macd_diff_col] * df[volatility_col]
    return out


def add_sma_comparison_features(df, price_col="price", sma_configs=None):
    out = pd.DataFrame(index=df.index)
    if price_col not in df:
        return out
    if sma_configs is None:
        sma_configs = {
            "Price_vs_SMA50": ("vs_base", f"SMA_{DEFAULT_SMA_LONG_PERIOD}"),
            "SMA10_vs_SMA50": ("cross", (f"SMA_{DEFAULT_SMA_SHORT_PERIOD}", f"SMA_{DEFAULT_SMA_LONG_PERIOD}")),
        }
    for new_col, cfg in sma_configs.items():
        op, cols = cfg
        if op == "vs_base" and cols in df:
            base = df[cols].replace(0, EPSILON)
            out[new_col] = (df[price_col]/base) - 1
        elif op == "cross" and all(c in df for c in cols):
            short = df[cols[0]].replace(0, EPSILON)
            long = df[cols[1]].replace(0, EPSILON)
            out[new_col] = (short/long) - 1
    return out


def add_rsi_divergence_flags(df, rsi_col=f"RSI_{DEFAULT_RSI_WINDOW}",
                             rsi_slope_col=f"Slope_RSI_{DEFAULT_RSI_WINDOW}_{DEFAULT_SLOPE_WINDOW}p",
                             ob_level=70, os_level=30):
    out = pd.DataFrame(index=df.index)
    if rsi_col in df and rsi_slope_col in df:
        rsi = df[rsi_col]
        slope = df[rsi_slope_col]
        out["RSI_OB_Maybe_Div"] = ((rsi > ob_level) & (slope < 0)).astype(int)
        out["RSI_OS_Maybe_Div"] = ((rsi < os_level) & (slope > 0)).astype(int)
    return out


def add_pqs_indicators(df, price_col="price", volume_col="volume", phase_angle_col="phase_angle"):
    out = pd.DataFrame(index=df.index)
    if all(c in df for c in [price_col, volume_col, phase_angle_col]):
        price = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
        volume = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
        phase_rad = np.radians(pd.to_numeric(df[phase_angle_col], errors="coerce").fillna(0))
        out["P_indicator"] = volume * price * np.cos(phase_rad)
        out["Q_indicator"] = volume * price * np.sin(phase_rad)
        out["S_indicator"] = np.sqrt(out["P_indicator"]**2 + out["Q_indicator"]**2)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def add_custom_interaction_polynomial_features(df, **kwargs):
    out = pd.DataFrame(index=df.index)
    macd_col = kwargs.get('macd_col', 'MACD')
    atr_col = kwargs.get('atr_col', f'ATR_{DEFAULT_ATR_WINDOW}_raw')
    rsi_slope_col = kwargs.get('rsi_slope_col', f'Slope_RSI_{DEFAULT_RSI_WINDOW}_{DEFAULT_SLOPE_WINDOW}p')
    volume_col = kwargs.get('volume_col', 'volume')
    adx_col = kwargs.get('adx_col', 'ADX')
    macd_diff_col = kwargs.get('macd_diff_col', 'MACD_Diff')
    bb_high_col = kwargs.get('bb_high_col', 'BB_High')
    bb_low_col = kwargs.get('bb_low_col', 'BB_Low')
    price_col = kwargs.get('price_col', 'price')
    rsi_col = kwargs.get('rsi_col', f'RSI_{DEFAULT_RSI_WINDOW}')
    macd_diff_poly = kwargs.get('macd_diff_col_poly', 'MACD_Diff')

    if macd_col in df and atr_col in df:
        safe_atr = df[atr_col].replace(0, EPSILON)
        out['MACD_div_ATR'] = (df[macd_col] / safe_atr).replace([np.inf, -np.inf], np.nan)

    if adx_col in df and macd_diff_col in df:
        out['ADX_x_MACD_Sign'] = df[adx_col] * np.sign(df[macd_diff_col])

    if rsi_slope_col in df and volume_col in df:
        vol_med = df[volume_col].rolling(window=20, min_periods=5).median()
        vol_std = df[volume_col].rolling(window=20, min_periods=5).std().replace(0, EPSILON)
        vol_spike = (df[volume_col] > vol_med + 2 * vol_std).astype(int)
        out['RSI_Slope_x_VolSpike'] = df[rsi_slope_col] * vol_spike
        out['Volume_Spike_Binary'] = vol_spike

    if all(c in df for c in [price_col, bb_high_col, bb_low_col]):
        width = (df[bb_high_col] - df[bb_low_col]).replace(0, EPSILON)
        out['User_BB_Pband'] = ((df[price_col] - df[bb_low_col]) / width).clip(0, 1)

    if rsi_col in df:
        out[f'{rsi_col}_sq'] = df[rsi_col] ** 2
    if macd_diff_poly in df:
        out[f'{macd_diff_poly}_sq'] = df[macd_diff_poly] ** 2

    return out


def add_trend_holding_features(df, **kwargs):
    out = pd.DataFrame(index=df.index)
    price_col = kwargs.get('price_col', 'price')
    high_col = kwargs.get('high_col', 'high')
    low_col = kwargs.get('low_col', 'low')
    volume_col = kwargs.get('volume_col', 'volume')
    atr_col = kwargs.get('atr_col', f'ATR_{DEFAULT_ATR_WINDOW}_raw')
    sma_long_col = kwargs.get('sma_long_col', f'SMA_{DEFAULT_SMA_LONG_PERIOD}')
    adx_col = kwargs.get('adx_col', 'ADX')
    pos_di_col = kwargs.get('pos_di_col', 'ADX_PosDI')
    neg_di_col = kwargs.get('neg_di_col', 'ADX_NegDI')
    macd_col = kwargs.get('macd_col', 'MACD')
    macd_sig_col = kwargs.get('macd_signal_col', 'MACD_Signal')
    rsi_col = kwargs.get('rsi_col', f'RSI_{DEFAULT_RSI_WINDOW}')

    if price_col in df and sma_long_col in df and atr_col in df:
        safe_atr = df[atr_col].replace(0, EPSILON)
        out['Price_Dist_SMA_Long_ATRNorm'] = ((df[price_col] - df[sma_long_col]) / safe_atr).replace([np.inf, -np.inf], np.nan)
        out['Price_Above_SMA_Long'] = (df[price_col] > df[sma_long_col]).astype(int)

    if all(c in df for c in [adx_col, pos_di_col, neg_di_col]):
        out['Is_Strong_Uptrend_ADX'] = ((df[adx_col] > 25) & (df[pos_di_col] > df[neg_di_col])).astype(int)

    if rsi_col in df:
        out['RSI_In_StrongZone'] = ((df[rsi_col] > 60) | (df[rsi_col] < 40)).astype(int)

    if all(c in df for c in [macd_col, macd_sig_col]):
        macd_above = df[macd_col] > df[macd_sig_col]
        out['MACD_Sustained_Bullish'] = (macd_above & macd_above.shift(1) & macd_above.shift(2) & (df[macd_col] > 0)).astype(int).fillna(0)

    if all(c in df for c in [high_col, low_col, price_col, atr_col]):
        try:
            psar = PSARIndicator(df[high_col], df[low_col], df[price_col],
                                 step=kwargs.get('psar_step', DEFAULT_PSAR_STEP),
                                 max_step=kwargs.get('psar_max_step', DEFAULT_PSAR_MAX_STEP), fillna=True)
            psar_val = psar.psar()
            out['PSAR_Value'] = psar_val
            out['Is_PSAR_Bullish'] = (df[price_col] > psar_val).astype(int)
            safe_atr = df[atr_col].replace(0, EPSILON)
            out['Price_Dist_PSAR_ATRNorm'] = ((df[price_col] - psar_val) / safe_atr).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            logging.error(f'PSAR error: {e}')

    if all(c in df for c in [high_col, low_col, price_col, atr_col]) and len(df) > kwargs.get('st_atr_period', DEFAULT_ST_ATR_PERIOD):
        st_df = get_supertrend_df(df, high_col, low_col, price_col,
                                  atr_col_for_norm=atr_col,
                                  st_atr_period=kwargs.get('st_atr_period', DEFAULT_ST_ATR_PERIOD),
                                  st_multiplier=kwargs.get('st_multiplier', DEFAULT_ST_MULTIPLIER))
        out = pd.concat([out, st_df], axis=1)

    if all(c in df for c in [high_col, low_col, price_col, atr_col]) and len(df) >= kwargs.get('ch_lookback_period', DEFAULT_CHANDELIER_LOOKBACK):
        ch_df = add_chandelier_exit_trend_features(df, high_col, low_col, price_col, atr_col,
                                                   ch_lookback_period=kwargs.get('ch_lookback_period', DEFAULT_CHANDELIER_LOOKBACK),
                                                   ch_multiplier=kwargs.get('ch_multiplier', DEFAULT_CHANDELIER_MULTIPLIER))
        out = pd.concat([out, ch_df], axis=1)

    if all(c in df for c in [high_col, low_col, price_col]):
        kc_df = add_keltner_channel_trend_features(df, high_col, low_col, price_col,
                                                   atr_col_for_norm=atr_col,
                                                   kc_window=kwargs.get('kc_window', DEFAULT_KC_WINDOW),
                                                   kc_atr_window=kwargs.get('kc_atr_window', DEFAULT_KC_ATR_WINDOW),
                                                   kc_multiplier=kwargs.get('kc_multiplier', DEFAULT_KC_MULTIPLIER))
        out = pd.concat([out, kc_df], axis=1)

    if all(c in df for c in [price_col, volume_col, high_col, low_col]):
        vol_df = add_volume_profile_features(df, price_col, volume_col, high_col, low_col,
                                             slope_window=kwargs.get('vol_profile_slope_window', DEFAULT_VOL_PROFILE_SLOPE_WINDOW))
        out = pd.concat([out, vol_df], axis=1)

    return out


def get_supertrend_df(df, high_col='high', low_col='low', price_col='price', atr_col_for_norm='ATR_14_raw',
                       st_atr_period=DEFAULT_ST_ATR_PERIOD, st_multiplier=DEFAULT_ST_MULTIPLIER):
    out = pd.DataFrame(index=df.index)
    high = df[high_col]
    low = df[low_col]
    close = df[price_col]
    if len(high) <= st_atr_period:
        return out
    atr = AverageTrueRange(high=high, low=low, close=close, window=st_atr_period, fillna=False).average_true_range().bfill()
    st = pd.DataFrame(index=high.index)
    st['basic_ub'] = (high + low)/2 + st_multiplier * atr
    st['basic_lb'] = (high + low)/2 - st_multiplier * atr
    st['final_ub'] = 0.0
    st['final_lb'] = 0.0
    st['value'] = 0.0
    st['direction'] = 1
    st.iloc[0, st.columns.get_loc('final_ub')] = st['basic_ub'].iloc[0]
    st.iloc[0, st.columns.get_loc('final_lb')] = st['basic_lb'].iloc[0]
    st.iloc[0, st.columns.get_loc('value')] = st['final_lb'].iloc[0]
    for i in range(1, len(st)):
        prev_close = close.iloc[i-1]
        fub_prev = st['final_ub'].iloc[i-1]
        flb_prev = st['final_lb'].iloc[i-1]
        bub = st['basic_ub'].iloc[i]
        blb = st['basic_lb'].iloc[i]
        st.iloc[i, st.columns.get_loc('final_ub')] = bub if (bub < fub_prev) or (prev_close > fub_prev) else fub_prev
        st.iloc[i, st.columns.get_loc('final_lb')] = blb if (blb > flb_prev) or (prev_close < flb_prev) else flb_prev
        prev_dir = st['direction'].iloc[i-1]
        close_i = close.iloc[i]
        fub = st['final_ub'].iloc[i]
        flb = st['final_lb'].iloc[i]
        if prev_dir == 1 and close_i <= flb:
            st.iloc[i, st.columns.get_loc('direction')] = -1
            st.iloc[i, st.columns.get_loc('value')] = fub
        elif prev_dir == -1 and close_i >= fub:
            st.iloc[i, st.columns.get_loc('direction')] = 1
            st.iloc[i, st.columns.get_loc('value')] = flb
        else:
            st.iloc[i, st.columns.get_loc('direction')] = prev_dir
            st.iloc[i, st.columns.get_loc('value')] = flb if prev_dir == 1 else fub
    out['SuperTrend_Value'] = st['value']
    out['SuperTrend_Direction'] = st['direction']
    out['Is_SuperTrend_Up'] = (close > out['SuperTrend_Value']).astype(int)
    if atr_col_for_norm in df:
        safe_atr = df[atr_col_for_norm].replace(0, EPSILON)
        out['Price_Dist_SuperTrend_ATRNorm'] = ((close - out['SuperTrend_Value']) / safe_atr).replace([np.inf, -np.inf], np.nan)
    return out


def add_chandelier_exit_trend_features(df, high_col='high', low_col='low', price_col='price',
                                       atr_col='ATR_14_raw', ch_lookback_period=DEFAULT_CHANDELIER_LOOKBACK,
                                       ch_multiplier=DEFAULT_CHANDELIER_MULTIPLIER):
    out = pd.DataFrame(index=df.index)
    if atr_col not in df or len(df) < ch_lookback_period:
        return out
    atr_series = df[atr_col]
    highest_high = df[high_col].rolling(window=ch_lookback_period, min_periods=1).max()
    out['Chandelier_Exit_Long'] = highest_high - atr_series * ch_multiplier
    out['Price_Above_ChandelierLong'] = (df[price_col] > out['Chandelier_Exit_Long']).astype(int)
    safe_atr = atr_series.replace(0, EPSILON)
    out['Dist_to_ChandelierLong_ATR_Units'] = ((df[price_col] - out['Chandelier_Exit_Long']) / safe_atr).replace([np.inf, -np.inf], np.nan)
    return out


def add_keltner_channel_trend_features(df, high_col='high', low_col='low', price_col='price',
                                       atr_col_for_norm='ATR_14_raw', kc_window=DEFAULT_KC_WINDOW,
                                       kc_atr_window=DEFAULT_KC_ATR_WINDOW, kc_multiplier=DEFAULT_KC_MULTIPLIER):
    out = pd.DataFrame(index=df.index)
    try:
        kc = KeltnerChannel(df[high_col], df[low_col], df[price_col], window=kc_window,
                            window_atr=kc_atr_window, multiplier=kc_multiplier, fillna=True)
        out['KC_Upper'] = kc.keltner_channel_hband()
        if atr_col_for_norm in df:
            safe_atr = df[atr_col_for_norm].replace(0, EPSILON)
            out['Price_vs_UpperKC_ATRNorm'] = ((df[price_col] - out['KC_Upper']) / safe_atr).replace([np.inf, -np.inf], np.nan)
    except Exception:
        out['KC_Upper'] = np.nan
        out['Price_vs_UpperKC_ATRNorm'] = np.nan
    return out


def add_volume_profile_features(df, price_col='price', volume_col='volume', high_col='high', low_col='low',
                                slope_window=DEFAULT_VOL_PROFILE_SLOPE_WINDOW):
    out = pd.DataFrame(index=df.index)
    if volume_col in df and price_col in df:
        obv = OnBalanceVolumeIndicator(df[price_col], df[volume_col], fillna=True).on_balance_volume()
        out['OBV'] = obv
        if pd.to_numeric(obv, errors='coerce').notna().sum() >= slope_window:
            out['OBV_Slope'] = pd.to_numeric(obv, errors='coerce').rolling(
                slope_window, min_periods=max(1, slope_window//2)
            ).apply(calc_slope, raw=True).replace([np.inf, -np.inf], np.nan)
    if all(c in df for c in [high_col, low_col, price_col, volume_col]):
        adi = AccDistIndexIndicator(df[high_col], df[low_col], df[price_col], df[volume_col], fillna=True).acc_dist_index()
        out['AccDist_Line'] = adi
        if pd.to_numeric(adi, errors='coerce').notna().sum() >= slope_window:
            out['AccDist_Slope'] = pd.to_numeric(adi, errors='coerce').rolling(
                slope_window, min_periods=max(1, slope_window//2)
            ).apply(calc_slope, raw=True).replace([np.inf, -np.inf], np.nan)
    return out


def add_relative_volume_feature(df, volume_col='volume', window=20):
    out = pd.DataFrame(index=df.index)
    if volume_col in df:
        avg_vol = df[volume_col].rolling(window=window, min_periods=1).mean()
        out['Relative_Volume'] = df[volume_col] / (avg_vol + EPSILON)
    return out


def add_sandbox_pqs_features(df):
    logging.debug('Sandbox PQS features placeholder - no features added')
    return pd.DataFrame(index=df.index)


def calculate_all_features(df):
    feature_funcs = [
        add_price_derivative_features,
        add_lookback_return_features,
        add_macd_features,
        add_rsi_features,
        add_bollinger_bands_features,
        lambda d: add_sma_feature(d, window=DEFAULT_SMA_SHORT_PERIOD),
        lambda d: add_sma_feature(d, window=DEFAULT_SMA_MEDIUM_PERIOD),
        lambda d: add_sma_feature(d, window=DEFAULT_SMA_LONG_PERIOD),
        add_ema_feature,
        add_adx_features,
        add_atr_and_natr_features,
        add_donchian_pos_feature,
        add_volume_profile_features,
        add_vwma_sma_diff_features,
        add_sma_comparison_features,
        calculate_fft_features_df,
        calculate_phase_features_df,
        lambda d: add_rolling_slopes(d, ['price', 'MACD', f'RSI_{DEFAULT_RSI_WINDOW}']),
        add_pqs_indicators,
        add_custom_interaction_polynomial_features,
        add_trend_holding_features,
        add_adx_derived_features,
        add_macd_volatility_interaction,
        add_rsi_divergence_flags,
        add_relative_volume_feature,
        add_sandbox_pqs_features,
    ]

    out = df.copy()
    for func in feature_funcs:
        try:
            new_cols = func(out)
            out = out.join(new_cols, how='left') if isinstance(new_cols, pd.DataFrame) else out
        except Exception as e:
            logging.error(f'Error applying {func.__name__}: {e}')
    return out
