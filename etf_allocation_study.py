"""月配息 ETF 配置研究:掃描所有 10% 刻度的權重組合,並用前半/後半期間檢驗「挑出來的」有沒有延續。"""
import numpy as np
import pandas as pd

from twdt import etf_alloc as ea

UNIVERSE = [("00900", "TW"), ("00929", "TW"), ("00934", "TW"), ("00939", "TW"),
            ("00940", "TW"), ("00937B", "TWO"), ("00945B", "TW")]
MINE = {"00939": 0.5, "00929": 0.3, "00937B": 0.2}
METRICS = ["ann", "max_dd", "yoc_annual", "yoc_cv"]            # 前三個越大越好,最後一個越小越好
BETTER_HIGH = {"ann": True, "max_dd": True, "yoc_annual": True, "yoc_cv": False}
LABEL = {"ann": "年化含息報酬", "max_dd": "最大回檔", "yoc_annual": "年化配息率(對成本)", "yoc_cv": "配息變異係數"}


def evaluate(close, adj, div, W, last_n):
    t = ea.tr_metrics(adj, W)
    inc = ea.income_metrics(ea.dca_simulate(close, div, W), last_n=last_n)
    return pd.DataFrame({"ann": t["ann"], "vol": t["vol"], "max_dd": t["max_dd"],
                         "yoc_annual": inc["yoc_annual"], "yoc_cv": inc["yoc_cv"],
                         "yoc_min": inc["yoc_min_month"], "dca_ret": inc["total_return"]})


def balanced_rank(df):
    ranks = pd.DataFrame({m: df[m].rank(ascending=not BETTER_HIGH[m], na_option="bottom") for m in METRICS})
    return ranks.mean(axis=1)


def name(w, codes):
    return " ".join(f"{c}:{int(round(x * 100))}%" for c, x in zip(codes, w) if x > 0)


def main():
    codes = [c for c, _ in UNIVERSE]
    frames = {c: ea.fetch_etf(c, s) for c, s in UNIVERSE}
    close, adj, div = ea.align(frames)
    print(f"共同期間 {close.index[0].date()} ~ {close.index[-1].date()},{len(close)} 個交易日,"
          f"{(close.index[-1] - close.index[0]).days / 365.25:.2f} 年,{len(codes)} 檔")
    months = close.index.to_period("M")
    print("各檔期間內配息次數 / 有配息的月份數:",
          {c: (int((div[c] > 0).sum()), int(pd.Series(div.index[div[c] > 0].to_period('M')).nunique())) for c in codes},
          f"(期間共 {months.nunique()} 個月)")

    W = ea.weights_grid(len(codes), 10)
    res = evaluate(close, adj, div, W, 12)
    res["rank"] = balanced_rank(res)
    print(f"\n掃描 {len(W)} 種權重組合(10% 刻度)。以下指標全部是「此期間」的歷史結果。")

    ref = {"你的 5:3:2 (00939/00929/00937B)": np.array([MINE.get(c, 0) for c in codes]),
           "7 檔等權重": np.full(len(codes), 1 / len(codes))}
    for i, c in enumerate(codes):
        e = np.zeros(len(codes)); e[i] = 1; ref[f"只買 {c}"] = e
    rr = evaluate(close, adj, div, np.array(list(ref.values())), 12)
    rr.index = list(ref)
    pd.set_option("display.width", 220)
    fmt = lambda d: d.assign(**{k: (d[k] * 100).round(1) for k in ("ann", "vol", "max_dd", "yoc_annual", "yoc_min", "dca_ret")}).assign(yoc_cv=d["yoc_cv"].round(2))
    cols = ["ann", "vol", "max_dd", "yoc_annual", "yoc_cv", "yoc_min", "dca_ret"]
    ren = {"ann": "年化含息報酬%", "vol": "年化波動%", "max_dd": "最大回檔%", "yoc_annual": "年化配息率%", "yoc_cv": "配息變異係數", "yoc_min": "單月最低配息率%", "dca_ret": "定期定額累計報酬%"}
    print("\n【參考組合】\n" + fmt(rr)[cols].rename(columns=ren).to_string())

    print("\n【各目標下歷史表現最好的 3 組】(合併 8008 組挑最好,結果已被『挑選』高估)")
    for m in METRICS:
        top = res.sort_values(m, ascending=not BETTER_HIGH[m]).head(3)
        print(f" ◆ {LABEL[m]}:")
        for i, r in top.iterrows():
            print(f"    {name(W[i], codes):<62} 報酬 {r.ann*100:5.1f}%  回檔 {r.max_dd*100:6.1f}%  配息率 {r.yoc_annual*100:4.1f}%  CV {r.yoc_cv:.2f}")
    print(" ◆ 綜合(四項排名平均,等權重,這個權重是我任訂的):")
    for i, r in res.sort_values("rank").head(5).iterrows():
        print(f"    {name(W[i], codes):<62} 報酬 {r.ann*100:5.1f}%  回檔 {r.max_dd*100:6.1f}%  配息率 {r.yoc_annual*100:4.1f}%  CV {r.yoc_cv:.2f}")
    mine_i = int(np.argmin(np.abs(W - np.array([MINE.get(c, 0) for c in codes])).sum(axis=1)))
    print(f"   (你的 5:3:2 在 {len(W)} 組中的綜合排名:第 {int(res['rank'].rank().iloc[mine_i])} 名)")

    # ---- 前半/後半:挑出來的有沒有延續?
    mid = close.index[len(close) // 2]
    h1 = (close[close.index < mid], adj[adj.index < mid], div[div.index < mid])
    h2 = (close[close.index >= mid], adj[adj.index >= mid], div[div.index >= mid])
    r1, r2 = evaluate(*h1, W, 6), evaluate(*h2, W, 6)
    print(f"\n【前後半驗證】前半 {h1[0].index[0].date()}~{h1[0].index[-1].date()}、後半 {h2[0].index[0].date()}~{h2[0].index[-1].date()}")
    print(" 各指標在前半與後半的排名相關係數(+1=完全延續,0=毫無關聯,負=反轉):")
    for m in METRICS:
        print(f"   {LABEL[m]:<12} {r1[m].rank().corr(r2[m].rank()):+.2f}")
    r1["rank"], r2["rank"] = balanced_rank(r1), balanced_rank(r2)
    best1 = int(r1["rank"].idxmin())
    pct = lambda col: float((r2[col].rank(ascending=not BETTER_HIGH[col]) <= r2[col].rank(ascending=not BETTER_HIGH[col]).iloc[best1]).mean())
    print(f" 前半「綜合最佳」:{name(W[best1], codes)}")
    print("   它在後半各指標的位置(前百分之幾,越小越好;50%=和隨機差不多):", {LABEL[m]: f"{pct(m)*100:.0f}%" for m in METRICS})
    print(f"   它在後半的綜合排名:第 {int(r2['rank'].rank().iloc[best1])} / {len(W)} 名")
    topret1 = int(r1["ann"].idxmax())
    print(f" 前半「報酬最高」:{name(W[topret1], codes)} → 後半年化報酬 {r2.ann.iloc[topret1]*100:.1f}%(後半全部組合中位數 {r2.ann.median()*100:.1f}%,排名前 {pct('ann') if False else (r2['ann'] >= r2['ann'].iloc[topret1]).mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
