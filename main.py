import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import numpy as np
import requests
import json

warnings.filterwarnings('ignore')


class PortfolioSelector:
    def __init__(self, tickers_dict, momentum_threshold_min=1.2, momentum_threshold_max=3.0,
                 max_positions=4, leverage_factor=1.0, sma_filter_months=4, 
                 telegram_bot_token=None, telegram_chat_id=None):
        self.tickers_dict = tickers_dict
        self.momentum_threshold_min = momentum_threshold_min
        self.momentum_threshold_max = momentum_threshold_max
        self.max_positions = max_positions
        self.leverage_factor = leverage_factor
        self.sma_filter_months = sma_filter_months
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        self.price_data = None
        self.bok_rates_cache = self._get_default_bok_rates()
        
        filter_info = f"{self.sma_filter_months}개월 이평선 필터 적용" if self.sma_filter_months > 0 else "이평선 필터 미적용"
        print(f"--- 포트폴리오 선택기 초기화: Max Positions={self.max_positions}, Leverage Factor={self.leverage_factor}x, {filter_info} ---")

    def _prepare_data(self, start_date_str, end_date_str):
        print("데이터 다운로드 중...")
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        required_offset = max(14, self.sma_filter_months + 2)
        data_start_date = start_date - pd.DateOffset(months=required_offset)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        all_tickers = list(self.tickers_dict.values())

        try:
            full_data = yf.download(
                all_tickers,
                start=data_start_date,
                end=end_date + timedelta(days=1),
                progress=False
            )

            if full_data.empty or not isinstance(full_data.columns, pd.MultiIndex):
                print("데이터 다운로드 실패")
                self.price_data = None
                return

            if 'Adj Close' in full_data.columns.get_level_values(0):
                self.price_data = full_data['Adj Close']
            else:
                self.price_data = full_data['Close']

            self.price_data.ffill(inplace=True)
            print("데이터 준비 완료")

        except Exception as e:
            print(f"데이터 다운로드 오류: {e}")
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

            if ticker_prices.empty: 
                return None

            current_price = ticker_prices.asof(end_date)
            if pd.isna(current_price): 
                return None

            price_ratios = []
            for months_back in range(6, 12):
                target_date = end_date - pd.DateOffset(months=months_back)
                past_prices_in_month = ticker_prices.loc[target_date.to_period('M').start_time : target_date.to_period('M').end_time]
                if past_prices_in_month.empty: 
                    continue

                past_price = past_prices_in_month.iloc[-1]
                if pd.notna(past_price) and past_price > 0:
                    price_ratios.append(current_price / past_price)

            if len(price_ratios) < 4: 
                return None

            return {
                'score': sum(price_ratios) / len(price_ratios),
                'current_price': current_price,
            }
        except Exception:
            return None
    
    def analyze_current_portfolio(self):
        """현재 포트폴리오 분석 및 선택"""
        today = datetime.now()
        analysis_date = today.strftime('%Y-%m-%d')
        
        # 데이터 준비
        self._prepare_data("2015-01-01", analysis_date)
        if self.price_data is None:
            return None

        last_trading_day = self.price_data.index.max().strftime('%Y-%m-%d')
        
        # 모멘텀 분석
        momentum_results = []
        
        # 이동평균선 계산 (필터가 활성화된 경우)
        sma_df = None
        if self.sma_filter_months > 0:
            window_size = self.sma_filter_months * 21
            sma_df = self.price_data.rolling(window=window_size).mean()

        for name, ticker in self.tickers_dict.items():
            # 이동평균선 필터 검사
            if self.sma_filter_months > 0 and sma_df is not None:
                current_price = self.get_trading_day_price(ticker, last_trading_day)
                sma_value = sma_df[ticker].asof(pd.to_datetime(last_trading_day))

                if current_price is None or pd.isna(sma_value):
                    continue

                if current_price <= sma_value:
                    continue
            
            # 모멘텀 점수 계산
            result = self.calculate_momentum_score(ticker, last_trading_day)
            if result is not None:
                momentum_results.append({
                    'name': name, 'ticker': ticker,
                    'momentum_score': result['score'], 'price': result['current_price']
                })

        momentum_results.sort(key=lambda x: x['momentum_score'], reverse=True)

        # 자산 선정
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
        
        return {
            'date': last_trading_day,
            'assets': final_assets,
            'bok_rate': self.get_bok_rate(last_trading_day)
        }

    def _apply_crypto_weight_limit(self, qualified_assets):
        crypto_assets = [a for a in qualified_assets if a['name'] in ["BTC/KRW", "ETH/KRW"]]
        non_crypto_assets = [a for a in qualified_assets if a['name'] not in ["BTC/KRW", "ETH/KRW"]]

        if not crypto_assets: 
            return qualified_assets
        if not non_crypto_assets:
            selected_cryptos = crypto_assets[:2]
            weight = 0.5 / len(selected_cryptos) if selected_cryptos else 0
            for c in selected_cryptos: 
                c['target_weight'] = weight
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

    def format_portfolio_message(self, portfolio_data):
        """포트폴리오 정보를 텔레그램 메시지 형식으로 포맷팅"""
        if not portfolio_data:
            return "❌ 포트폴리오 분석 실패"
        
        date = portfolio_data['date']
        assets = portfolio_data['assets']
        bok_rate = portfolio_data['bok_rate']
        
        filter_info = f"{self.sma_filter_months}개월 이평선 필터" if self.sma_filter_months > 0 else "필터 없음"
        
        message = f"📊 **모멘텀 전략 포트폴리오** ({self.leverage_factor}x 레버리지)\n"
        message += f"📅 분석일: {date}\n"
        message += f"🔧 설정: {filter_info}, 최대 {self.max_positions}개 자산\n\n"
        
        if assets:
            message += f"✅ **선택된 자산 ({len(assets)}개):**\n"
            total_risk_weight = 0
            for i, asset in enumerate(assets, 1):
                weight = asset.get('target_weight', 1/len(assets))
                total_risk_weight += weight
                message += f"{i}. {asset['name']}\n"
                message += f"   📈 모멘텀 점수: {asset['momentum_score']:.3f}\n"
                message += f"   💰 비중: {weight * 100:.1f}%\n\n"
            
            cash_weight = 1.0 - total_risk_weight
            if cash_weight > 0.01:  # 1% 이상인 경우만 표시
                message += f"💵 **현금성 자산:** {cash_weight * 100:.1f}%\n"
                message += f"   🏦 기준금리: {bok_rate}%\n"
        else:
            message += "❌ **투자 조건을 만족하는 자산 없음**\n"
            message += f"💵 전액 현금성 자산 투자\n"
            message += f"🏦 기준금리: {bok_rate}%\n"
        
        return message

    def send_telegram_message(self, message):
        """텔레그램으로 메시지 전송"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            print("텔레그램 설정이 없습니다. 콘솔에만 출력됩니다.")
            return False
        
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        data = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("✅ 텔레그램 메시지 전송 성공")
                return True
            else:
                print(f"❌ 텔레그램 메시지 전송 실패: {response.status_code}")
                print(response.text)
                return False
        except Exception as e:
            print(f"❌ 텔레그램 전송 오류: {e}")
            return False

    def run_portfolio_analysis(self):
        """포트폴리오 분석 실행 및 텔레그램 전송"""
        print("📊 현재 포트폴리오 분석 시작...")
        
        # 포트폴리오 분석
        portfolio_data = self.analyze_current_portfolio()
        
        # 메시지 포맷팅
        message = self.format_portfolio_message(portfolio_data)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("현재 포트폴리오 선택 결과")
        print("="*60)
        print(message.replace('**', '').replace('*', ''))
        print("="*60)
        
        # 텔레그램 전송
        self.send_telegram_message(message)
        
        return portfolio_data


if __name__ == "__main__":
    # 자산 티커 설정
    tickers = {
        "S&P 500": "^GSPC", 
        "나스닥 종합": "^IXIC", 
        "니케이 225": "^N225",
        "인도 Sensex": "^BSESN", 
        "브라질 Bovespa": "^BVSP", 
        "FTSE 100": "^FTSE",
        "인도네시아 JSX": "^JKSE", 
        "독일 DAX": "^GDAXI", 
        "상해 종합": "000001.SS",
        "KOSPI 200": "^KS200", 
        "홍콩 H지수": "^HSCE", 
        "BTC/KRW": "BTC-KRW",
        "ETH/KRW": "ETH-KRW", 
        "금 선물": "GC=F", 
        "미국 20년 국채 ETF": "TLT"
    }
    
    # 텔레그램 설정 (봇 토큰과 채팅 ID를 입력하세요)
    TELEGRAM_BOT_TOKEN = "7200427583:AAE6ZBTRvhfSnrYWstUsOGdgnN4YUxy7OcQ"  # 봇파더에서 받은 토큰
    TELEGRAM_CHAT_ID = "6932457088"     # 본인의 채팅 ID 또는 그룹 ID
    
    # 포트폴리오 선택기 초기화
    selector = PortfolioSelector(
        tickers_dict=tickers,
        momentum_threshold_min=1.2,
        momentum_threshold_max=3.0,
        max_positions=4,
        leverage_factor=2.0,
        sma_filter_months=6,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )
    
    # 포트폴리오 분석 및 텔레그램 전송
    selector.run_portfolio_analysis()
