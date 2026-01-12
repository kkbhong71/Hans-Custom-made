import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter

# 1. 페이지 설정
st.set_page_config(page_title="Han's Lotto AI", page_icon="🎱", layout="wide")

st.title("🎱 Han's Custom Made: 로또 당첨 예측 AI")
st.markdown("데이터 기반 통계 분석과 AI 알고리즘을 융합한 번호 생성기입니다.")

# -----------------------------------------------------------
# 2. 데이터 로드 함수
# -----------------------------------------------------------
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        if 'round' in df.columns:
            df = df.sort_values(by='round', ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame()

# -----------------------------------------------------------
# 3. 분석 유틸리티 함수 (기존 로직 유지)
# -----------------------------------------------------------
def calculate_ac_value(numbers):
    diffs = set()
    for a, b in combinations(numbers, 2):
        diffs.add(abs(a - b))
    return len(diffs) - 5

def get_high_low_ratio(numbers):
    low = sum(1 for n in numbers if n <= 22)
    high = 6 - low
    return low, high

def analyze_last_digit(numbers):
    last_digits = [n % 10 for n in numbers]
    s_last = sum(last_digits)
    if not (15 <= s_last <= 35): return False, s_last
    counts = Counter(last_digits)
    if max(counts.values()) >= 3: return False, s_last
    return True, s_last

def analyze_section_pattern(numbers):
    sections = [0] * 5
    for n in numbers:
        if 1 <= n <= 10: sections[0] += 1
        elif 11 <= n <= 20: sections[1] += 1
        elif 21 <= n <= 30: sections[2] += 1
        elif 31 <= n <= 40: sections[3] += 1
        else: sections[4] += 1
    return sections

def get_ball_color(number):
    if 1 <= number <= 10: return '#FBC400'
    elif 11 <= number <= 20: return '#69C8F2'
    elif 21 <= number <= 30: return '#FF7272'
    elif 31 <= number <= 40: return '#AAAAAA'
    else: return '#B0D840'

# -----------------------------------------------------------
# 4. 핵심 알고리즘 (기존 로직 A~G)
# -----------------------------------------------------------
def get_lotto_numbers(algo_type, hot_pool, cold_pool, weights):
    pool = hot_pool
    if len(pool) < 6: return []

    def pick_random(p, k=6):
        return sorted(random.sample(p, k))

    # [A] 랜덤
    if algo_type == 'A': return pick_random(pool)
    
    # [B] 가중치
    elif algo_type == 'B':
        try:
            probs = np.array(weights) / sum(weights)
            sel = np.random.choice(pool, 6, replace=False, p=probs)
            return sorted([int(n) for n in sel])
        except: return pick_random(pool)

    # [C] 홀짝 밸런스
    elif algo_type == 'C':
        for _ in range(500):
            cand = pick_random(pool)
            odd = sum(1 for n in cand if n % 2 != 0)
            if 2 <= odd <= 4: return cand
        return pick_random(pool)

    # [D] 합계 구간
    elif algo_type == 'D':
        for _ in range(500):
            cand = pick_random(pool)
            s = sum(cand)
            if 100 <= s <= 170: return cand
        return pick_random(pool)

    # [E] 패턴 분산
    elif algo_type == 'E':
        for _ in range(500):
            cand = pick_random(pool)
            sec = analyze_section_pattern(cand)
            if max(sec) >= 5: continue
            is_cons = False
            for i in range(len(cand)-2):
                if cand[i+1] == cand[i]+1 and cand[i+2] == cand[i]+2:
                    is_cons = True; break
            if not is_cons: return cand
        return pick_random(pool)

    # [F] AI 초정밀
    elif algo_type == 'F':
        for _ in range(10000):
            cand = pick_random(pool)
            if not (100 <= sum(cand) <= 170): continue
            odd = sum(1 for n in cand if n % 2 != 0)
            if not (2 <= odd <= 4): continue
            low, high = get_high_low_ratio(cand)
            if not (2 <= low <= 4): continue
            if calculate_ac_value(cand) < 7: continue
            valid_last, _ = analyze_last_digit(cand)
            if not valid_last: continue
            is_cons = False
            for i in range(len(cand)-2):
                if cand[i+1] == cand[i]+1 and cand[i+2] == cand[i]+2:
                    is_cons = True; break
            if is_cons: continue
            return cand
        return pick_random(pool)

    # [G] 과적합 방지
    elif algo_type == 'G':
        if len(cold_pool) < 2: return sorted(random.sample(hot_pool + cold_pool, 6))
        for _ in range(2000):
            mix_ratio = random.choice([(4, 2), (5, 1)])
            n_hot, n_cold = mix_ratio
            try:
                part1 = random.sample(hot_pool, n_hot)
                part2 = random.sample(cold_pool, n_cold)
            except: continue
            cand = sorted(part1 + part2)
            if not (80 <= sum(cand) <= 200): continue
            is_cons = False
            for i in range(len(cand)-2):
                if cand[i+1] == cand[i]+1 and cand[i+2] == cand[i]+2:
                    is_cons = True; break
            if not is_cons: return cand
        return sorted(random.sample(hot_pool + cold_pool, 6))
    
    return pick_random(pool)

# -----------------------------------------------------------
# 5. 메인 실행 UI
# -----------------------------------------------------------
# 파일 로드 (같은 폴더에 있는 파일)
file_path = 'new_1206.csv'
df = load_data(file_path)

if df.empty:
    st.error(f"'{file_path}' 파일을 찾을 수 없습니다. 저장소에 파일을 업로드했는지 확인해주세요.")
else:
    last_round = df['round'].iloc[0]
    st.info(f"📅 최신 데이터: {last_round}회차까지 업데이트됨")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 분석 옵션 설정")
    window = st.sidebar.selectbox("분석 구간 선택 (최근 N회)", [30, 50, 100], index=0)
    
    if st.button("🚀 번호 생성 시작"):
        st.divider()
        st.subheader(f"📊 최근 {window}회 분석 결과")
        
        # 데이터 전처리
        number_cols = [f'num{i}' for i in range(1, 7)]
        subset = df[number_cols].head(window)
        counts = pd.Series(subset.values.flatten()).value_counts().sort_index()

        # Hot/Cold 분류
        hot_mask = counts >= 2
        hot_target = counts[hot_mask]
        hot_pool = hot_target.index.tolist()
        weights = hot_target.values.tolist()
        cold_pool = [n for n in range(1, 46) if n not in hot_pool]

        col1, col2 = st.columns(2)
        col1.metric("🔥 Hot Pool (2회 이상)", f"{len(hot_pool)}개")
        col2.metric("❄️ Cold Pool (1회 이하)", f"{len(cold_pool)}개")

        # 그래프 시각화
        if len(hot_pool) >= 6:
            fig, ax = plt.subplots(figsize=(10, 3))
            ball_colors = [get_ball_color(n) for n in hot_pool]
            sns.barplot(x=hot_pool, y=weights, palette=ball_colors, hue=hot_pool, legend=False, ax=ax)
            ax.set_title(f"Hot Number Frequency (Last {window} rounds)")
            ax.set_ylabel("Count")
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig)

        st.divider()
        st.subheader("🎲 알고리즘별 추천 번호")

        algo_names = [
            ('A','랜덤'), ('B','가중치'), ('C','밸런스'),
            ('D','합계구간'), ('E','패턴분산'),
            ('F','AI초정밀'), ('G','과적합방지')
        ]
        
        results = []
        
        for code, name in algo_names:
            nums = get_lotto_numbers(code, hot_pool, cold_pool, weights)
            if not nums: continue
            
            sec = analyze_section_pattern(nums)
            
            # 추가 정보 텍스트
            extra_info = ""
            if code == 'F':
                ac = calculate_ac_value(nums)
                _, s_last = analyze_last_digit(nums)
                extra_info = f"(AC:{ac}, 끝수합:{s_last})"
            elif code == 'G':
                cold_cnt = sum(1 for n in nums if n in cold_pool)
                extra_info = f"(❄️Cold: {cold_cnt})"

            # 결과 저장
            results.append({
                "타입": f"{code} ({name})",
                "추천 번호": str(nums),
                "구간 분포": str(sec),
                "특이사항": extra_info
            })
            
            # 카드 형태로 출력
            with st.container():
                nums_str = "  ".join([f"{n}" for n in nums])
                icon = "🛡️" if code == 'G' else ("🌟" if code == 'F' else "🔹")
                st.write(f"### {icon} [{code}] {name}")
                st.code(nums_str, language="text")

        # 요약표
        st.divider()
        st.write("📋 **한눈에 보기 (복사용)**")
        st.dataframe(pd.DataFrame(results))