import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import numpy as np
import os
from pathlib import Path
import requests

warnings.filterwarnings('ignore')

class MomentumStrategy:
    def __init__(self, tickers_dict, initial_capital=100000, momentum_threshold_min=1.2, momentum_threshold_max=3.0,
                 max_positions=4, risk_on_leverage=2.0, sma_filter_months=6,
                 macro_filter_ticker='^GSPC', macro_filter_sma_months=10, bok_api_key=None,
                 telegram_token=None, chat_id=None):

        self.tickers_dict = tickers_dict
        self.initial_capital = initial_capital
        self.momentum_threshold_min = momentum_threshold_min
        self.momentum_threshold_max = momentum_threshold_max
        self.max_positions = max_positions
        
        self.risk_on_leverage = risk_on_leverage
        self.current_leverage = self.risk_on_leverage

        self.sma_filter_months = sma_filter_months
        self.macro_filter_ticker = macro_filter_ticker
        self.macro_filter_sma_months = macro_filter_sma_months
        
        self.bok_api_key = bok_api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.price_data = None
        self.monthly_returns = []
        self.portfolio_history = []
        self.trade_history = []
        self.selected_assets_history = []
        self.bok_rates_cache = self._get_default_bok_rates()
        self.default_bok_rates = self._get_default_bok_rates()
        self.is_risk_on = True

        filter_info = f"{self.sma_filter_months}개월 SMA 필터" if self.sma_filter_months > 0 else "개별 필터 없음"
        macro_filter_info = f"{self.macro_filter_sma_months}개월 SMA*1.01 매크로 필터 ({self.macro_filter_ticker})" if self.macro_filter_ticker else "매크로 필터 없음"
        print(f"--- 전략 초기화: Max Positions={self.max_positions}, Max Leverage={self.risk_on_leverage}x, {filter_info}, {macro_filter_info} ---")

    def send_telegram_message(self, message):
        """텔레그램으로 메시지 전송"""
        if not self.telegram_token or not self.chat_id:
            print("텔레그램 설정이 없어 콘솔에만 출력합니다.")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                print("✓ 텔레그램 메시지 전송 완료")
            else:
                print(f"❌ 텔레그램 전송 실패: {response.status_code}")
        except Exception as e:
            print(f"❌ 텔레그램 전송 오류: {e}")

    def _prepare_data(self, start_date_str, end_date_str):
        print("\nFetching all historical data in one batch... this may take a moment.")
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        required_offset = max(14, self.sma_filter_months + 2, self.macro_filter_sma_months + 2)
        data_start_date = start_date - pd.DateOffset(months=required_offset)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        all_tickers = list(self.tickers_dict.values())
        if self.macro_filter_ticker and self.macro_filter_ticker not in all_tickers:
            all_tickers.append(self.macro_filter_ticker)

        try:
            full_data = yf.download(all_tickers, start=data_start_date, end=end_date + timedelta(days=1), progress=True)
            if full_data.empty or not isinstance(full_data.columns, pd.MultiIndex):
                print("🔥 Critical error: Failed to download valid data or data has an unexpected format.")
                self.price_data = None
                return

            if 'Adj Close' in full_data.columns.get_level_values(0):
                self.price_data = full_data['Adj Close']
                print("--> Using 'Adj Close' for price data.")
            else:
                self.price_data = full_data['Close']
                print("--> Warning: 'Adj Close' not found. Falling back to 'Close' prices.")

            self.price_data.ffill(inplace=True)
            print("✓ Historical data has been successfully downloaded and prepared.")
        except Exception as e:
            print(f"🔥 Critical error during data download or processing: {e}")
            self.price_data = None

    def _get_default_bok_rates(self):
        return {
            "2018-07": 1.50, "2018-08": 1.50, "2018-09": 1.50, "2018-10": 1.50, "2018-11": 1.75, "2018-12": 1.75,
            "2019-01": 1.75, "2019-02": 1.75, "2019-03": 1.75, "2019-04": 1.75, "2019-05": 1.75, "2019-06": 1.75,
            "2019-07": 1.50, "2019-08": 1.50, "2019-09": 1.50, "2019-10": 1.25, "2019-11": 1.25, "2019-12": 1.25,
            "2020-01": 1.25, "2020-02": 1.25, "2020-03": 0.75, "2020-04": 0.75, "2020-05": 0.50, "2020-06": 0.50,
            "2020-07": 0.50, "2020-08": 0.50, "2020-09": 0.50, "2020-10": 0.50, "2020-11": 0.50, "2020-12": 0.50,
            "2021-01": 0.50, "2021-02": 0.50, "2021-03": 0.50, "2021-04": 0.50, "2021-05": 0.50, "2021-06": 0.50,
            "2021-07": 0.50, "2021-08": 0.75, "2021-09": 0.75, "2021-10": 0.75, "2021-11": 1.00, "2021-12": 1.00,
            "2022-01": 1.25, "2022-02": 1.25, "2022-03": 1.25, "2022-04": 1.50, "2022-05": 1.75, "2022-06": 1.75,
            "2022-07": 2.25, "2022-08": 2.50, "2022-09": 2.50, "2022-10": 3.00, "2022-11": 3.25, "2022-12": 3.25,
            "2023-01": 3.50, "2023-02": 3.50, "2023-03": 3.50, "2023-04": 3.50, "2023-05": 3.50, "2023-06": 3.50,
            "2023-07": 3.50, "2023-08": 3.50, "2023-09": 3.50, "2023-10": 3.50, "2023-11": 3.50, "2023-12": 3.50,
            "2024-01": 3.50, "2024-02": 3.50, "2024-03": 3.50, "2024-04": 3.50, "2024-05": 3.50, "2024-06": 3.50,
            "2024-07": 3.50, "2024-08": 3.50, "2024-09": 3.50, "2024-10": 3.25, "2024-11": 3.00, "2024-12": 3.00,
            "2025-01": 3.00, "2025-02": 2.75, "2025-03": 2.75, "2025-04": 2.75, "2025-05": 2.50, "2025-06": 2.50,
            "2025-07": 2.50
        }

    def get_bok_rate(self, date_str):
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        date_key = date_obj.strftime('%Y-%m')
        available_dates = sorted([k for k in self.bok_rates_cache.keys() if k <= date_key])
        return self.bok_rates_cache[available_dates[-1]] if available_dates else 1.50

    def initialize_bok_data(self, start_date, end_date):
        print(f"기준금리 데이터 초기화 중...")
        self.bok_rates_cache = self.default_bok_rates.copy()

    def get_month_end_dates(self, start_date_str, end_date_str):
        all_months = self.price_data.loc[start_date_str:end_date_str].index.to_period('M').unique()
        month_end_dates = [self.price_data.loc[self.price_data.index.to_period('M') == m].index.max() for m in all_months]
        return [d.strftime('%Y-%m-%d') for d in sorted(month_end_dates)]

    def get_trading_day_price(self, ticker, target_date_str):
        try:
            target_date = pd.to_datetime(target_date_str)
            price = self.price_data[ticker].asof(target_date)
            return float(price) if pd.notna(price) else None
        except (KeyError, IndexError):
            return None

    def calculate_momentum_score(self, ticker, end_date_str):
        try:
            end_date = pd.to_datetime(end_date_str)
            ticker_prices = self.price_data[ticker].dropna()

            if ticker_prices.empty: return None

            current_price = ticker_prices.asof(end_date)
            if pd.isna(current_price): return None

            price_ratios = []
            for months_back in range(6, 12):
                target_date = end_date - pd.DateOffset(months=months_back)
                past_prices_in_month = ticker_prices.loc[target_date.to_period('M').start_time : target_date.to_period('M').end_time]
                if past_prices_in_month.empty: continue

                past_price = past_prices_in_month.iloc[-1]
                if pd.notna(past_price) and past_price > 0:
                    price_ratios.append(current_price / past_price)
            
            if len(price_ratios) < 4: return None
            
            return { 'score': sum(price_ratios) / len(price_ratios), 'current_price': current_price }
        except Exception:
            return None

    def analyze_monthly_momentum(self, date_str):
        momentum_results = []
        print(f"\n=== {date_str} 모멘텀 분석 ===")
        target_date = pd.to_datetime(date_str)

        if self.macro_filter_ticker:
            macro_window = self.macro_filter_sma_months * 21
            macro_sma_series = self.price_data[self.macro_filter_ticker].rolling(window=macro_window).mean()
            
            current_macro_price = self.get_trading_day_price(self.macro_filter_ticker, date_str)
            macro_sma_value = macro_sma_series.asof(target_date)

            if current_macro_price is not None and pd.notna(macro_sma_value):
                macro_threshold = macro_sma_value * 1.01
                self.is_risk_on = current_macro_price > macro_threshold
                status = "ON" if self.is_risk_on else "OFF"
                print(f"  [매크로 필터] 시장 '위험 {status}' 상태 ({self.macro_filter_ticker} 현재가 {current_macro_price:.2f} vs {self.macro_filter_sma_months}개월 SMA*1.01 {macro_threshold:.2f})")
            else:
                self.is_risk_on = False
                print("  [매크로 필터] 데이터 부족으로 '위험 OFF' 처리")
        
        if self.is_risk_on:
            self.current_leverage = self.risk_on_leverage
        else:
            self.current_leverage = 1.0
            print("  - 매크로 필터 '위험 OFF' 상태. 전량 현금 보유로 전환합니다.")
            print(f"\n선택된 자산 (0개):")
            print(f"\n선택된 자산이 없음 → 현금성 자산 투자 (기준금리: {self.get_bok_rate(date_str)}%)")
            return []

        sma_df = None
        if self.sma_filter_months > 0:
            window_size = self.sma_filter_months * 21
            sma_df = self.price_data.rolling(window=window_size).mean()

        for name, ticker in self.tickers_dict.items():
            if self.sma_filter_months > 0 and sma_df is not None:
                current_price = self.get_trading_day_price(ticker, date_str)
                sma_value = sma_df[ticker].asof(target_date)

                if current_price is None or pd.isna(sma_value): continue
                if current_price <= sma_value:
                    print(f"  - {name}: {self.sma_filter_months}개월 SMA 아래. [필터링됨] (현재가 {current_price:.2f} <= SMA {sma_value:.2f})")
                    continue
            
            result = self.calculate_momentum_score(ticker, date_str)
            if result is not None:
                filter_status = f"({self.sma_filter_months}개월 SMA 통과)" if self.sma_filter_months > 0 else ""
                print(f"  - {name}: 모멘텀 점수: {result['score']:.3f} {filter_status}")
                momentum_results.append({
                    'name': name, 'ticker': ticker,
                    'momentum_score': result['score'], 'price': result['current_price']
                })

        momentum_results.sort(key=lambda x: x['momentum_score'], reverse=True)

        qualified_assets = []
        for asset in momentum_results:
            if asset['name'] in ["BTC/KRW", "ETH/KRW"]:
                if self.momentum_threshold_min <= asset['momentum_score'] < 6.0:
                    qualified_assets.append(asset)
            elif asset['name'] == "미국 20년 국채 ETF":
                if 1.1 <= asset['momentum_score'] < self.momentum_threshold_max:
                    qualified_assets.append(asset)
            else:
                if self.momentum_threshold_min <= asset['momentum_score'] < self.momentum_threshold_max:
                    qualified_assets.append(asset)
            if len(qualified_assets) >= self.max_positions:
                break
        
        final_assets = self._apply_crypto_weight_limit(qualified_assets)
        print(f"\n선택된 자산 ({len(final_assets)}개):")
        for i, asset in enumerate(final_assets, 1):
            weight_info = f" (Weight: {asset.get('target_weight', 1/len(final_assets))*100:.1f}%)"
            print(f"{i}. {asset['name']}: {asset['momentum_score']:.3f}{weight_info}")
        if not final_assets:
            print(f"\n선택된 자산이 없음 → 현금성 자산 투자 (기준금리: {self.get_bok_rate(date_str)}%)")
        return final_assets

    def _apply_crypto_weight_limit(self, qualified_assets):
        crypto_assets = [a for a in qualified_assets if a['name'] in ["BTC/KRW", "ETH/KRW"]]
        non_crypto_assets = [a for a in qualified_assets if a['name'] not in ["BTC/KRW", "ETH/KRW"]]

        if not crypto_assets: return qualified_assets
        if not non_crypto_assets:
            selected_cryptos = crypto_assets[:2]
            weight = 0.5 / len(selected_cryptos) if selected_cryptos else 0
            for c in selected_cryptos: c['target_weight'] = weight
            return selected_cryptos
        else:
            max_crypto_count = min(2, len(crypto_assets))
            selected_cryptos = crypto_assets[:max_crypto_count]
            remaining_slots = self.max_positions - len(selected_cryptos)
            selected_non_cryptos = non_crypto_assets[:remaining_slots]
            total_assets = selected_cryptos + selected_non_cryptos
            crypto_ratio = len(selected_cryptos) / len(total_assets)

            if crypto_ratio > 0.5:
                crypto_weight = 0.5 / len(selected_cryptos)
                non_crypto_weight = 0.5 / len(selected_non_cryptos) if selected_non_cryptos else 0
            else:
                equal_weight = 1.0 / len(total_assets)
                crypto_weight, non_crypto_weight = equal_weight, equal_weight

            for asset in total_assets:
                asset['target_weight'] = crypto_weight if asset in selected_cryptos else non_crypto_weight
            return total_assets

    def calculate_monthly_return(self, selected_assets, buy_date, sell_date):
        if not selected_assets:
            bok_rate = self.get_bok_rate(sell_date)
            cash_return = (bok_rate / 100) / 12
            print(f"현금 투자 - 월 수익률: {cash_return * 100:.4f}%")
            return cash_return

        total_weighted_return, total_weight = 0.0, 0.0
        for asset in selected_assets:
            ticker, weight = asset['ticker'], asset.get('target_weight', 1.0 / len(selected_assets))
            buy_price, sell_price = self.get_trading_day_price(ticker, buy_date), self.get_trading_day_price(ticker, sell_date)

            if buy_price and sell_price and buy_price > 0:
                asset_return = (sell_price - buy_price) / buy_price
                leveraged_return = asset_return * self.current_leverage
                total_weighted_return += leveraged_return * weight
                total_weight += weight
                print(f"  {asset['name']}: {buy_price:.2f} → {sell_price:.2f} ({asset_return * 100:+.2f}%) | {self.current_leverage}x 레버리지 적용: {leveraged_return * 100:+.2f}% (비중: {weight*100:.1f}%)")
            else:
                print(f"  {asset['name']}: 가격 데이터를 가져올 수 없어 계산에서 제외됩니다.")

        cash_weight = 1.0 - total_weight
        if cash_weight > 1e-6:
            bok_rate = self.get_bok_rate(sell_date)
            cash_return = (bok_rate / 100) / 12
            total_weighted_return += cash_return * cash_weight
            print(f"  현금: (월 수익률 {cash_return*100:.4f}%) x {cash_weight*100:.1f}%")

        return total_weighted_return

    def analyze_current_portfolio(self):
        print(f"\n{'=' * 80}\n현재 포트폴리오 분석 (Today's Portfolio)\n{'=' * 80}")
        today, analysis_date = datetime.now(), datetime.now().strftime('%Y-%m-%d')
        self._prepare_data("2015-01-01", analysis_date)
        if self.price_data is None:
            error_msg = "데이터를 가져오지 못해 현재 포트폴리오 분석을 종료합니다."
            print(error_msg)
            self.send_telegram_message(f"❌ <b>포트폴리오 분석 실패</b>\n\n{error_msg}")
            return

        last_trading_day = self.price_data.index.max().strftime('%Y-%m-%d')
        print(f"분석 기준일: {last_trading_day}")
        selected_assets = self.analyze_monthly_momentum(last_trading_day)

        leverage_info = f"{self.current_leverage}x 레버리지" if self.is_risk_on else "현금 보유 (위험 OFF)"
        
        telegram_msg = f"📊 <b>모멘텀 전략 포트폴리오</b>\n"
        telegram_msg += f"📅 <b>분석일:</b> {last_trading_day}\n"
        telegram_msg += f"⚡ <b>상태:</b> {leverage_info}\n\n"
        
        if selected_assets:
            total_risk_weight = sum(a.get('target_weight', 1/len(selected_assets)) for a in selected_assets)
            telegram_msg += f"✅ <b>투자 대상 자산 ({len(selected_assets)}개):</b>\n\n"
            for i, asset in enumerate(selected_assets, 1):
                weight = asset.get('target_weight', 1/len(selected_assets))
                telegram_msg += f"{i}. <b>{asset['name']}</b>\n"
                telegram_msg += f"   • 비중: {weight * 100:.1f}%\n"
                telegram_msg += f"   • 모멘텀 점수: {asset['momentum_score']:.3f}\n\n"
            
            cash_weight = 1.0 - total_risk_weight
            if cash_weight > 1e-6:
                telegram_msg += f"💵 <b>현금성 자산:</b> {cash_weight * 100:.1f}%\n\n"
        else:
            bok_rate = self.get_bok_rate(last_trading_day)
            telegram_msg += f"❌ <b>투자 조건 만족 자산 없음</b>\n"
            telegram_msg += f"💵 전액 현금성 자산 투자\n"
            telegram_msg += f"📈 기준금리: {bok_rate}%\n\n"
        
        if self.macro_filter_ticker:
            macro_status = "ON ✅" if self.is_risk_on else "OFF ❌"
            telegram_msg += f"🌍 <b>매크로 필터:</b> 위험 {macro_status}\n"
            telegram_msg += f"📊 기준: {self.macro_filter_ticker} > 10개월 SMA*1.01"
        
        print(f"\n{'=' * 60}\n📊 현재 투자 포트폴리오 요약 ({leverage_info})\n{'=' * 60}")
        if selected_assets:
            total_risk_weight = sum(a.get('target_weight', 1/len(selected_assets)) for a in selected_assets)
            print(f"\n✅ 투자 대상 자산 ({len(selected_assets)}개):\n")
            for i, asset in enumerate(selected_assets, 1):
                weight = asset.get('target_weight', 1/len(selected_assets))
                print(f"  {i}. {asset['name']} (비중: {weight * 100:.1f}%, 점수: {asset['momentum_score']:.3f})")
            cash_weight = 1.0 - total_risk_weight
            if cash_weight > 1e-6:
                print(f"  💵 현금성 자산 (비중: {cash_weight * 100:.1f}%)")
        else:
            print("\n❌ 현재 투자 조건을 만족하는 자산이 없습니다. 전액 현금성 자산 투자.")
        print(f"\n{'=' * 60}")
        
        self.send_telegram_message(telegram_msg)
        return selected_assets


if __name__ == "__main__":
    tickers = {
        "S&P 500": "^GSPC", "나스닥 종합": "^IXIC", "니케이 225": "^N225",
        "인도 Sensex": "^BSESN", "브라질 Bovespa": "^BVSP", "FTSE 100": "^FTSE",
        "인도네시아 JSX": "^JKSE", "독일 DAX": "^GDAXI", "상해 종합": "000001.SS",
        "KOSPI 200": "^KS200", "홍콩 H지수": "^HSCE", "BTC/KRW": "BTC-KRW",
        "ETH/KRW": "ETH-KRW", "금 선물": "GC=F", "미국 20년 국채 ETF": "TLT"
    }
    
    strategy = MomentumStrategy(
        tickers_dict=tickers,
        initial_capital=330000000,
        momentum_threshold_min=1.2,
        momentum_threshold_max=3.0,
        max_positions=4,
        risk_on_leverage=2.0,
        sma_filter_months=6,
        macro_filter_ticker='^GSPC',
        macro_filter_sma_months=10,
        bok_api_key="YOUR_API_KEY",
        telegram_token="7200427583:AAE6ZBTRvhfSnrYWstUsOGdgnN4YUxy7OcQ",  # 텔레그램 봇 토큰
        chat_id="6932457088"         # 텔레그램 채팅 ID
    )

    # 현재 포트폴리오 분석만 실행
    strategy.analyze_current_portfolio()
