# 2026-10-19 m5：回路縮減を残す代数的修正とH6準備

利用者の「続きを行って」を受け、1qubit再合成を無効にしたH6 PASSを保全し、回路縮減を残す方式を実装した。隣接RZの正確な有理数和とSX/X恒等式だけを使い、非ZERO小角度の閾値削除、2π wrap、Euler生成、scalar加算を避ける。

prototype SOURCE29f5d5f0、H6専用SOURCE0d47f59c。人工19件＋adapter11件PASS。7回路×2仮設定の14条件を2processで実行し、native exit0・約1.38秒。丸め予算0でも人工RZ24→1、entangled例RZ32→24を確認したが、実H6での縮減率・raw精度・時間は未検証である。正予算1e-11は人工の台帳比較だけで、H6には採用していない。

H6 one-shot案は丸め0、同じcosine一軸・新compile1/作用1、保存lowering/PASS再利用、raw3検査各1e-9未満を維持。SOURCE/input/runtime2016seal、metadata20項目と改変拒否11種PASS、無効grant/実CLI拒否を保存。有効grant・新run/marker・実H6 QPY/NPZ復元・科学計算・会計・retry・push0。2720file、H6対照PASS・旧STOP/58軸・AD03保留を保全した。

[実装・人工結果・固定scope・次の担当](../track_a_m5_algebraic_oneq_readiness_20261019.md)。次はユーザーが新SOURCEで固定一軸を一回実行するか判断し、承認後にCodexが新receipt/grant/fresh gateを結合する。local commit後mandatory STOP。
