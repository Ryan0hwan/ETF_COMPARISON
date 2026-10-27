"""
비율 차트(Ratio chart) 감시 → 디스코드
- ratios.txt의 비율(A/B)마다 추세(20·50일선 기준), 1주·1개월·3개월 변화, 52주 범위 내 위치를 계산
- 위험선호 종합, 교차확인(XLY/XLP·HYG/IEF·RSP/SPY), 다이버전스 경고, 주제별 해석 문장을 만들어
- 매일 장 마감 후 디스코드로 보냅니다.
"""
import datetime as dt
import os
import sys
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 ─────────────
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"                       # 휴장일·주말에도 가장 최근 거래일 기준으로 실행
DIVERGENCE_NEAR_HIGH = float(os.environ.get("DIVERGENCE_NEAR_HIGH", "3"))  # SPY가 52주 고점 3% 이내일 때 다이버전스 점검
# ───────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
ALIAS = {"SPX": "^GSPC", "NDX": "^NDX", "VIX": "^VIX", "VVIX": "^VVIX", "VIX3M": "^VIX3M", "DXY": "DX-Y.NYB"}
CROSS_CHECK = ["XLY/XLP", "HYG/IEF", "RSP/SPY"]
TREND_ICON = {1: "📈", 0: "➖", -1: "📉"}
CATEGORIES = ["위험선호", "성장가치", "기술내부", "시장폭", "경기인플레", "금리금융", "금리민감", "원자재통화"]


def load_ratios() -> list[dict]:
    out = []
    for ln in (ROOT / "ratios.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(",", 3)]
        if len(parts) < 4 or "/" not in parts[1]:
            continue
        a, b = parts[1].upper().split("/", 1)
        out.append({"cat": parts[0], "name": f"{a}/{b}", "a": ALIAS.get(a, a), "b": ALIAS.get(b, b),
                    "pol": int(parts[2]), "desc": parts[3]})
    return out


def tickers() -> list[str]:
    return list(dict.fromkeys(t for r in load_ratios() for t in (r["a"], r["b"])))


def _trend(r: pd.Series, at: int = -1) -> int:
    """비율이 50일선 위 + 20일선 > 50일선 → +1 / 반대 → -1 / 그 외 0"""
    s20, s50 = r.rolling(20).mean(), r.rolling(50).mean()
    v, a, b = r.iloc[at], s20.iloc[at], s50.iloc[at]
    if np.isnan(b):
        return 0
    return 1 if v > b and a > b else (-1 if v < b and a < b else 0)


def _pct(r: pd.Series, n: int) -> float:
    return float((r.iloc[-1] / r.iloc[-n - 1] - 1) * 100) if len(r) > n else np.nan


def analyze(data: dict[str, pd.DataFrame]) -> dict:
    res, missing = {}, []
    for spec in load_ratios():
        da, db = data.get(spec["a"]), data.get(spec["b"])
        if da is None or db is None:
            missing.append(spec["name"])
            continue
        r = (da["Close"] / db["Close"]).dropna()
        if len(r) < 60:
            missing.append(spec["name"])
            continue
        tail = r.tail(252)
        lo, hi = tail.min(), tail.max()
        t_now, t_prev = _trend(r), _trend(r, -6)
        res[spec["name"]] = {**spec, "trend": t_now, "flip": t_now != t_prev and t_now != 0,
                             "w1": _pct(r, 5), "m1": _pct(r, 21), "m3": _pct(r, 63),
                             "pos": float((r.iloc[-1] - lo) / (hi - lo) * 100) if hi > lo else 50.0}
    return {"ratios": res, "missing": missing, "summary": interpret(res, data)}


def _t(res, name) -> int | None:
    return res[name]["trend"] if name in res else None


def _avg(res, names) -> float | None:
    v = [res[n]["trend"] for n in names if n in res]
    return sum(v) / len(v) if v else None


def interpret(res: dict, data: dict) -> list[str]:
    out = []
    # ① 위험선호 종합
    counted = [r for r in res.values() if r["pol"] != 0]
    if counted:
        on = sum(r["pol"] * r["trend"] == 1 for r in counted)
        off = sum(r["pol"] * r["trend"] == -1 for r in counted)
        score = (on - off) / len(counted)
        verdict = "🟢 위험선호" if score >= 0.33 else ("🔴 위험회피" if score <= -0.33 else "🟡 혼조")
        out.append(f"위험선호 종합: {verdict}  ({len(counted)}개 중 선호 {on} · 회피 {off} · 중립 {len(counted) - on - off})")

    # ② 교차확인
    cc = [(n, _t(res, n)) for n in CROSS_CHECK if n in res]
    if len(cc) == 3:
        ts = {t for _, t in cc}
        if ts == {1}:
            out.append("교차확인 (소비재·신용·시장폭): ✅ 모두 상승 → 위험선호가 넓고 건강함")
        elif ts == {-1}:
            out.append("교차확인 (소비재·신용·시장폭): 🔻 모두 하락 → 위험회피가 확인됨")
        else:
            desc = " · ".join(f"{n} {TREND_ICON[t]}" for n, t in cc)
            out.append(f"교차확인: ⚠️ 불일치 ({desc})")

    # ③ 다이버전스: 지수는 고점 근처인데 시장폭·신용이 꺾임
    spy = data.get("SPY")
    if spy is not None and len(spy) > 252:
        c = spy["Close"]
        gap = (c.iloc[-1] / c.tail(252).max() - 1) * 100
        weak = [n for n in ("RSP/SPY", "HYG/IEF", "XLY/XLP") if _t(res, n) == -1]
        if gap >= -DIVERGENCE_NEAR_HIGH and weak:
            out.append(f"⚠️ 다이버전스: SPY는 고점 {gap:+.1f}%인데 {', '.join(weak)} 하락 추세 → 상승이 좁아지는 경고")

    # ④ 주제별
    lines = []
    g = _avg(res, ["QQQ/SPY", "IWF/IWD"])
    if g is not None:
        lines.append("스타일: 성장·대형기술 주도" if g > 0.5 else ("스타일: 가치·비기술 로테이션" if g < -0.5 else "스타일: 성장/가치 혼조"))
    # 기술주 내부 이동: 반도체 vs 소프트웨어·클라우드, 메모리
    smh_igv, cloud, mem = _t(res, "SMH/IGV"), _t(res, "WCLD/SMH"), _t(res, "DRAM/SMH")
    if smh_igv == -1 and cloud == 1:
        lines.append("기술 내부: 하드웨어(반도체) → 소프트웨어·클라우드로 자금 이동")
    elif smh_igv == 1 and cloud == -1:
        lines.append("기술 내부: 소프트웨어 → 반도체(AI 인프라)로 자금 이동")
    elif smh_igv:
        lines.append("기술 내부: 반도체(AI 인프라) > 소프트웨어" if smh_igv > 0 else "기술 내부: 소프트웨어 > 반도체")
    if mem:
        lines.append("메모리: 반도체 전체보다 강함 → 메모리 사이클 상승" if mem > 0
                     else "메모리: 반도체 전체보다 약함 → 메모리 사이클 둔화")
    # 금리 하락 기대 로테이션: 장기채·지역은행·주택건설·리츠가 함께 움직이는지
    rate = [n for n in ("TLT/SHY", "KRE/SPY", "ITB/SPY", "SCHH/SPY") if n in res]
    if len(rate) >= 3:
        ups = sum(res[n]["trend"] == 1 for n in rate)
        downs = sum(res[n]["trend"] == -1 for n in rate)
        early = sum((res[n]["m1"] or 0) > 0 and (res[n]["w1"] or 0) > 0 for n in rate)
        if ups >= 3:
            lines.append(f"금리 민감 업종: {ups}/{len(rate)}개 상승 추세 → 금리 고점 반영 로테이션 확인")
        elif early >= 3:
            lines.append(f"금리 민감 업종: {early}/{len(rate)}개가 1주·1개월 모두 반등 → 금리 고점 기대 초기 신호 (추세 전환 전)")
        elif downs >= 3:
            lines.append(f"금리 민감 업종: {downs}/{len(rate)}개 하락 추세 → 고금리 부담 지속")
    b = _avg(res, ["RSP/SPY", "IWM/QQQ"])
    if b is not None:
        lines.append("시장폭: 상승이 넓게 퍼짐" if b > 0.5 else ("시장폭: 소수 대형주 쏠림" if b < -0.5 else "시장폭: 혼조"))
    e = _avg(res, ["XLI/XLU", "XLB/XLU", "COPX/GLD"])
    if e is not None:
        lines.append("경기: 확장 기대" if e > 0.33 else ("경기: 둔화 우려 (방어주·금 강세)" if e < -0.33 else "경기: 중립"))
    i = _avg(res, ["TIP/IEF", "XLE/SPY"])
    if i is not None and abs(i) > 0.5:
        lines.append("인플레: 기대인플레 상승 (에너지·물가연동채 강세)" if i > 0 else "인플레: 기대인플레 하락")
    t = _t(res, "KRE/XLF")
    if t:
        lines.append("은행: 지방은행 강세 → 은행 스트레스 완화" if t > 0 else "은행: 지방은행 약세 → 은행 스트레스 주의")
    t = _t(res, "TLT/SHY")
    if t:
        lines.append("채권: 장기채 강세 → 경기 둔화 우려 / 금리 하락 기대" if t > 0 else "채권: 장기채 약세 → 장기금리 상승 압력")
    t = _t(res, "VVIX/VIX")
    if t:
        lines.append("변동성: VVIX가 VIX보다 빠르게 오름 → VIX 급등 전조 주의" if t > 0 else "변동성: VVIX/VIX 하락 → 변동성 안정")
    t = _t(res, "SPX/VIX")
    if t:
        lines.append("SPX/VIX: 주가 대비 공포 감소 → 위험선호" if t > 0 else "SPX/VIX: 주가 대비 공포 증가 → 헤지 수요 확대")
    for name, up, dn in (("GDX/GLD", "금광주 > 금 → 금 랠리 지속성 높음", "금광주 < 금 → 금 랠리 힘 약함"),
                         ("SLV/GLD", "은 > 금 → 위험선호·산업 수요", "은 < 금 → 방어적 귀금속 수요"),
                         ("EEM/SPY", "신흥국 > 미국 → 달러 약세 장세", "미국 > 신흥국 → 달러 강세·미국 쏠림"),
                         ("IBIT/QQQ", "비트코인 > 나스닥 → 투기 수요 강함", "비트코인 < 나스닥 → 투기 수요 약화"),
                         ("EWY/SPY", "한국 > 미국 → 한국 상대 강세", "한국 < 미국 → 한국 상대 약세" +
                          (" (메모리 약세와 동행)" if mem == -1 else ""))):
        t = _t(res, name)
        if t:
            lines.append(up if t > 0 else dn)
    out += [f"· {ln}" for ln in lines]

    # ⑤ 최근 1주 안에 추세가 바뀐 비율
    flips = [f"{n} {TREND_ICON[r['trend']]}" for n, r in res.items() if r["flip"]]
    if flips:
        out.append("최근 추세 전환: " + ", ".join(flips))
    return out


def table_lines(res: dict, cat: str) -> list[str]:
    rows = [r for r in res.values() if r["cat"] == cat]
    out = []
    for r in rows:
        f = lambda v: "    -" if np.isnan(v) else f"{v:+5.1f}"
        flip = "*" if r["flip"] else " "
        out.append(f"{TREND_ICON[r['trend']]} {r['name'] + flip:<11}{f(r['w1']):>6}{f(r['m1']):>7}{f(r['m3']):>7}"
                   f"{r['pos']:>5.0f}  {r['desc']}")
    return out


# ───────────── 시세 · 디스코드 ─────────────
def download(tks: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    data = yf.download(tks, period="2y", interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    if not isinstance(data.columns, pd.MultiIndex):
        data = pd.concat({tks[0]: data}, axis=1)
    have = set(data.columns.get_level_values(0))
    out = {}
    for t in tks:
        if t in have:
            df = data[t].dropna(subset=["Close"])
            if len(df):
                out[t] = df
    return out


def post(content: str):
    if not WEBHOOK:
        print(content)
        print("-" * 40)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def fit(s: str, w: int, right: bool = False) -> str:
    out = ""
    for ch in s:
        if width(out + ch) > w:
            break
        out += ch
    pad = " " * (w - width(out))
    return pad + out if right else out + pad


def send_block(title: str, head: str | None, lines: list[str]):
    first = True
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first and title else ""
        h = f"{head}\n" if head else ""
        post(f"{header}```\n{h}" + "\n".join(block) + "\n```")
        first = False


def main():
    need = list(dict.fromkeys(["SPY"] + tickers()))
    data = download(need)
    if "SPY" not in data:
        post("⚠️ 비율 차트: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return 1
    session = data["SPY"].index[-1].date()
    if session != dt.datetime.now(NY).date() and not FORCE:
        print(f"휴장일로 보임 (최근 거래일 {session}). 종료.")
        return
    rt = analyze(data)
    if not rt["ratios"]:
        post("⚠️ 비율 차트: 계산된 비율이 없습니다.")
        return 1
    send_block(f"📊 비율 차트 해석 | {session} 마감", None, rt["summary"] or ["(해석할 비율 없음)"])
    head = ("   " + fit("비율", 11) + fit("1W%", 6, True) + fit("1M%", 7, True) + fit("3M%", 7, True)
            + fit("위치", 5, True) + "  설명")
    for cat in CATEGORIES:
        lines = table_lines(rt["ratios"], cat)
        if lines:
            send_block(f"· {cat}", head, lines)
    post("📈 상승추세 ➖ 횡보 📉 하락추세 (비율이 50일선 위·아래 + 20일선 방향) · \\* 최근 1주 안에 추세 전환\n"
         "위치 = 52주 범위에서 현재 비율 위치 (0 = 52주 최저, 100 = 52주 최고)")
    if rt["missing"]:
        post("⚠️ 계산 못 한 비율: " + ", ".join(rt["missing"]))
    print(f"완료: 비율 {len(rt['ratios'])}개")


if __name__ == "__main__":
    sys.exit(main())
