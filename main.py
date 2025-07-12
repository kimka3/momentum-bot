import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import numpy as np
import requests
import os
import logging
from typing import Dict, List, Optional, Tuple

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ======================== [로깅 설정] ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('momentum_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================== [설정] ========================
# 환경변수에서 토큰 정보 가져오기 (보안 강화)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '7200427583:AAE6ZBTRvhfSnrYWstUsOGdgnN4YUxy7OcQ')
CHAT_ID = os.getenv('CHAT_ID', '6932457088')

# 분석 대상 자산
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
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "금 선물": "GC=F",
    "미국 20년 국채 ETF": "TLT"
}

# 모멘텀 분석 설정
MOMENTUM_MIN_SCORE = 1.2  # 최소 모멘텀 점수
MOMENTUM_MAX_SCORE = 3.0  # 최대 모멘텀 점수
MAX_ASSETS = 4  # 최대 선택 자산 수
ANALYSIS_PERIOD_YEARS = 2  # 분석 기간 (년)


# ======================== [함수] ========================
def send_telegram(message: str) -> bool:
    """텔레그램으로 메시지 전송"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"  # HTML 포맷 지원
    }

    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            logger.info("✓ 텔레그램 메시지 전송 성공")
            return True
        else:
            logger.error(f"⚠️ 텔레그램 전송 실패: {response.text}")
            return False
    except Exception as e:
        logger.error(f"⚠️ 텔레그램 전송 오류: {e}")
        return False


def calculate_momentum(prices: pd.Series, date: datetime) -> Tuple[Optional[float], Optional[float]]:
    """모멘텀 점수 계산"""
    try:
        # 현재 가격
        current_price = prices.loc[:date].iloc[-1]

        # 6개월, 12개월 전 날짜
        six_months_ago = date - pd.DateOffset(months=6)
        one_year_ago = date - pd.DateOffset(months=12)

        # 과거 가격 찾기
        price_6m = prices.loc[:six_months_ago].iloc[-1]
        price_12m = prices.loc[:one_year_ago].iloc[-1]

        # 모멘텀 점수 계산 (6개월 수익률 + 12개월 수익률의 평균)
        momentum_6m = current_price / price_6m
        momentum_12m = current_price / price_12m
        score = (momentum_6m + momentum_12m) / 2

        return round(score, 3), current_price

    except Exception as e:
        logger.warning(f"모멘텀 계산 실패: {e}")
        return None, None


def download_data_alternative(tickers_dict: Dict[str, str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """개별 종목별 데이터 다운로드 (대안 방법)"""
    logger.info("대안 방법으로 개별 종목 데이터 다운로드")

    all_data = {}

    for name, ticker in tickers_dict.items():
        try:
            logger.info(f"다운로드 중: {name} ({ticker})")
            ticker_data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)

            if ticker_data.empty:
                logger.warning(f"⚠️ {name} ({ticker}) 데이터 없음")
                continue

            # Adj Close 또는 Close 컬럼 선택
            if 'Adj Close' in ticker_data.columns:
                price_data = ticker_data['Adj Close']
            elif 'Close' in ticker_data.columns:
                price_data = ticker_data['Close']
            else:
                logger.warning(f"⚠️ {name} ({ticker}) 가격 데이터 없음")
                continue

            all_data[ticker] = price_data

        except Exception as e:
            logger.warning(f"⚠️ {name} ({ticker}) 다운로드 실패: {e}")
            continue

    if not all_data:
        raise ValueError("다운로드된 데이터가 없습니다.")

    # DataFrame으로 결합
    data = pd.DataFrame(all_data)
    data = data.ffill()

    logger.info(f"✓ 대안 방법 다운로드 완료: {len(data)} 일, {len(data.columns)} 종목")
    return data


def download_data(tickers_dict: Dict[str, str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """가격 데이터 다운로드"""
    logger.info(f"데이터 다운로드 시작: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

    try:
        # 첫 번째 시도: 일괄 다운로드
        raw_data = yf.download(
            list(tickers_dict.values()),
            start=start_date,
            end=end_date + timedelta(days=1),
            progress=False
        )

        logger.info(f"다운로드된 데이터 구조: {raw_data.columns}")
        logger.info(f"데이터 타입: {type(raw_data.columns)}")

        # 데이터 구조 확인 및 처리
        if isinstance(raw_data.columns, pd.MultiIndex):
            # 다중 종목인 경우
            logger.info("다중 종목 데이터 구조 감지")
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns.get_level_values(0):
                data = raw_data['Close']
            else:
                # 가능한 모든 컬럼 출력
                logger.error(f"사용 가능한 컬럼: {raw_data.columns.get_level_values(0).unique()}")
                raise ValueError("가격 데이터 컬럼을 찾을 수 없습니다.")
        else:
            # 단일 종목인 경우
            logger.info("단일 종목 데이터 구조 감지")
            logger.info(f"사용 가능한 컬럼: {list(raw_data.columns)}")

            if 'Adj Close' in raw_data.columns:
                data = raw_data[['Adj Close']].copy()
                data.columns = [list(tickers_dict.values())[0]]
            elif 'Close' in raw_data.columns:
                data = raw_data[['Close']].copy()
                data.columns = [list(tickers_dict.values())[0]]
            else:
                # 모든 컬럼 출력하여 디버깅
                logger.error(f"사용 가능한 컬럼: {list(raw_data.columns)}")
                raise ValueError("가격 데이터 컬럼을 찾을 수 없습니다.")

        # 결측값 전진 채우기
        data = data.ffill()

        # 데이터 검증
        if data.empty:
            raise ValueError("다운로드된 데이터가 비어있습니다.")

        logger.info(f"✓ 데이터 다운로드 완료: {len(data)} 일, {len(data.columns)} 종목")
        logger.info(f"최종 데이터 컬럼: {list(data.columns)}")
        return data

    except Exception as e:
        logger.error(f"❌ 일괄 다운로드 실패: {e}")
        logger.info("대안 방법으로 개별 다운로드 시도...")

        try:
            return download_data_alternative(tickers_dict, start_date, end_date)
        except Exception as e2:
            logger.error(f"❌ 대안 방법도 실패: {e2}")
            raise e


def analyze_individual_asset(name: str, ticker: str, data: pd.DataFrame, date: datetime) -> Optional[Dict]:
    """개별 자산 분석"""
    if ticker not in data.columns:
        logger.warning(f"⚠️ {name} ({ticker}) 데이터 없음")
        return None

    try:
        score, price = calculate_momentum(data[ticker], date)
        if score is None:
            logger.warning(f"⚠️ {name} 모멘텀 계산 실패")
            return None

        # 변동성 계산 (추가 정보)
        returns = data[ticker].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 연환산 변동성

        return {
            'name': name,
            'ticker': ticker,
            'score': score,
            'price': price,
            'volatility': round(volatility, 3)
        }

    except Exception as e:
        logger.error(f"❌ {name} 분석 실패: {e}")
        return None


def select_assets(results: List[Dict]) -> List[Dict]:
    """모멘텀 기준으로 자산 선택"""
    # 점수 조건 필터링
    filtered = [r for r in results if MOMENTUM_MIN_SCORE <= r['score'] < MOMENTUM_MAX_SCORE]

    # 점수 기준 정렬 (내림차순)
    sorted_assets = sorted(filtered, key=lambda x: x['score'], reverse=True)

    # 상위 N개 선택
    selected = sorted_assets[:MAX_ASSETS]

    logger.info(f"선택된 자산: {len(selected)}개 (전체 분석: {len(results)}개)")
    return selected


def create_portfolio_message(selected_assets: List[Dict], date: datetime) -> str:
    """포트폴리오 메시지 생성"""
    date_str = date.strftime('%Y-%m-%d')

    message = f"📊 <b>{date_str} 모멘텀 포트폴리오</b>\n\n"

    if not selected_assets:
        message += "❌ 조건에 맞는 자산 없음 → 현금 보유 권장\n"
        message += f"(점수 범위: {MOMENTUM_MIN_SCORE}~{MOMENTUM_MAX_SCORE})"
        return message

    # 동일 비중 계산
    weight = round(100 / len(selected_assets), 1)

    for i, asset in enumerate(selected_assets, 1):
        message += f"<b>{i}. {asset['name']}</b>\n"
        message += f"   • 모멘텀 점수: {asset['score']}\n"
        message += f"   • 포트폴리오 비중: {weight}%\n"
        message += f"   • 현재가: ${asset['price']:,.2f}\n"
        message += f"   • 연환산 변동성: {asset['volatility']:.1%}\n\n"

    message += f"📈 총 {len(selected_assets)}개 자산 선택\n"
    message += f"⚖️ 각 자산 동일 비중: {weight}%"

    return message


def analyze_momentum():
    """메인 모멘텀 분석 함수"""
    logger.info("🚀 모멘텀 분석 시작")

    try:
        # 날짜 설정
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * ANALYSIS_PERIOD_YEARS)

        # 데이터 다운로드
        data = download_data(tickers, start_date, end_date)

        # 각 자산 분석
        results = []
        for name, ticker in tickers.items():
            result = analyze_individual_asset(name, ticker, data, end_date)
            if result:
                results.append(result)

        logger.info(f"분석 완료: {len(results)}개 자산")

        # 자산 선택
        selected_assets = select_assets(results)

        # 메시지 생성 및 전송
        message = create_portfolio_message(selected_assets, end_date)

        # 콘솔 출력
        print("\n" + "=" * 50)
        print(message.replace('<b>', '').replace('</b>', ''))
        print("=" * 50)

        # 텔레그램 전송 (실패해도 계속 진행)
        try:
            success = send_telegram(message)
            if success:
                logger.info("✓ 텔레그램 전송 성공")
            else:
                logger.warning("⚠️ 텔레그램 전송 실패 (분석 결과는 콘솔에서 확인 가능)")
        except Exception as e:
            logger.warning(f"⚠️ 텔레그램 전송 중 오류: {e}")

        logger.info("✓ 모멘텀 분석 완료")

    except Exception as e:
        error_msg = f"❌ 모멘텀 분석 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"오류 상세: {type(e).__name__}: {str(e)}")

        # 텔레그램 전송 시도 (실패해도 괜찮음)
        try:
            send_telegram(error_msg)
        except:
            pass

        raise


def validate_environment():
    """환경 설정 검증"""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == 'YOUR_TELEGRAM_TOKEN':
        logger.error("❌ 텔레그램 토큰이 설정되지 않았습니다.")
        return False

    if not CHAT_ID or CHAT_ID == 'YOUR_CHAT_ID':
        logger.error("❌ 텔레그램 채팅 ID가 설정되지 않았습니다.")
        return False

    # 텔레그램 연결 테스트
    try:
        test_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
        response = requests.get(test_url, timeout=5)
        if response.status_code != 200:
            logger.error(f"❌ 텔레그램 봇 토큰 검증 실패: {response.text}")
            return False

        # 채팅 ID 검증을 위한 간단한 테스트 (선택사항)
        logger.info("✓ 텔레그램 봇 토큰 검증 완료")

    except Exception as e:
        logger.warning(f"⚠️ 텔레그램 연결 테스트 실패: {e}")
        logger.info("분석은 계속 진행하지만 텔레그램 전송은 실패할 수 있습니다.")

    logger.info("✓ 환경 설정 검증 완료")
    return True


# ======================== [실행] ========================
if __name__ == "__main__":
    try:
        # 환경 설정 검증
        if not validate_environment():
            exit(1)

        # 모멘텀 분석 실행
        analyze_momentum()
        print("\n✅ 모든 작업이 성공적으로 완료되었습니다!")

    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"💥 예상치 못한 오류: {e}")
        exit(1)