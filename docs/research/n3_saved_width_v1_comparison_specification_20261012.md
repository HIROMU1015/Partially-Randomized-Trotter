# N3 保存証拠の事後保証幅比較：結果取得前の実行仕様

2026-10-12 JST。今回ここまでの認可は準備のみ。以下の実8入力への数値適用は別の明示認可後、一回だけ。
[数学契約](n3_saved_width_v1_mathematical_contract_20261012.md)、[準備報告](n3_saved_width_v1_preparation_20261012.md)。

## 1. Immutable inputsと母集団

元payload `e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c`、公開HEAD `86f57f05a36e7e6639e827486d300bd049bf2438`。
primary seal `8b1e75e89da78c13b82f24896db387adec6e76597ec12669a7549279592d0d69`。
input SHA `c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4`。
同じ8入力をposthoc development資料として扱い、独立移送・新しいblind検証と呼ばない。

実対称one-body＋非負unordered density、一外部mode。
C1/R1/L1/S1はactive m=6/7/8/9、ν=2/3/3/4、full modes7/8/9/10、外部mode1/4/0/8、scale1/1.5/.75/1.25。
各baseの同norm2方向、元δgoal=.004/.006/.003/.005。
geometry・化学basis・DF rank/方針・L_D・PF/time windowは適用なし。
元δgoal、採否、NO_ADDITIONAL_ACCEPTANCE_OBSERVED、2-base未達、Mandatory STOPを変更しない。

## 2. 比較先と保存候補の固定

主比較は各caseの保存済みU,c_starを共通にする。c_starを新計算・新下界へ置き換えない。
母集団は元ledgerの全三行c0/c_occ/c_starに保存されたraw候補。
PSDはcやδに依存しないので、元行のbox/affineで不採択でもPSD-validなら含める。
保存されたκそのものを使い、xからκを作り直さない。

| pool | 含める証拠 | 含めないもの |
|---|---|---|
| NORM | 保存入力のvからの同じnorm-only identity、原raw証拠へ参照 | 新しい下界、真値 |
| FIVE | rawに保存された全PSD-valid κ,μ | 旧λだけから新しく作るκ、追加五点 |
| OLD-DIRECT | rawに保存された全PSD-valid κ,μ | optimizer、未保存κ/μ |
| BASE saved | 上の三方法の安全なOR、有限保存集合の最小幅 | NORMを落とした対照 |
| OCC raw all saved spectra | raw κに対する保存selected spectralと全spectral_attemptsのPSD-valid τ/Z（または保存full trace） | fallback、継承コピー、新κ、新τ/Z |
| OCC selected spectra | 各raw attemptのgenerator-selected spectralだけ、補助集計 | 新たな発見集合という解釈 |

OCCは元のselected証拠だけでなく、途中に保存されたPSD-valid spectral pairも主保存集合へ含める。
BASEも採用証拠だけへ絞らず全PSD-valid候補を含める。この扱いを新しい幅結果を見る前に固定する。
selected-onlyの補助集計を併記して、保存途中証拠を加えた効果と元generatorの選択を分離する。
同一κ・μまたはτ/Zの重複は内容hashでまとめ、元row/method/attempt/pointerは全て残す。
継承コピーは同じ固定証拠なので新候補に数えない。fallbackもOCC固有の集合へ混ぜない。

## 3. 評価・厳密監査

新source/input/planに結び付く別authorizationを必要とする。旧科学batch認可はschema段階で拒否。
原pack/ledger/secondary/source/planのbyteを確認し、旧checkerで保存U/cと全raw証拠を確認する。
全保存spectral pairのZとresidualも、保存proof poolから独立LDLで再確認する。新しい証拠は生成しない。

各固定証拠のqと幅区間を128-bit外向き計算で得る。安全側δ_proof、元goal、元証拠ID・matrix hash、原pointerを別fieldへ保存する。
`verify_n3_saved_width_v1.py`は保存候補集合の完全性、固定κ/μ/τ/Z、q、PSD、poly endpoints、各poolのminimum envelope、比較labelを独立確認する。
scientific outputは別namespace `artifacts/n3_saved_width_v1/2026-10-12/evaluation1/`。元run1には書かない。

caseの分類はstrictに小さいOCC、strictに小さいBASE、exactな一点幅一致、区間重なり未判定、保存coverage不足。
8入力を分母から除かない。技術失敗なら部分resultとfailure/STOPを保存して停止する。
解析成功後も、元gateを成功へ書き換えたり条件族のglobal optimum・不可行性を主張したりしない。

## 4. 計算費用と資源

支配費用は保存された有理LDL/行列再構成。既存checkerのcase内cacheを再利用し、caseごとのimmutable入力を2 spawn workersへ渡す。
secondary監査は別proof poolへ切り替わるため、事後評価側でprimary proof pool/cacheを復元する。凍結済み旧checkerは変更しない。
Python3.12.3・stdlib。BLAS/OMP/MKL/Numba/Rayon各1、NumPy・GPU・eigensolver・compilerなし。
CPU32/affinity32の環境でsyntheticの2-worker経路を確認する。actual8の速度は測定していない。

width候補cap512/case、精度128bits固定、有理4096bits、sqrt scaling integer8192bits。
worker AS4GiB、runner＋全descendantsのsampled RSS8GiB、開始時available10GiB以上。
各output32MiB、全output64MiB（通常62MiB＋停止記録予約2MiB）、failure record64KiB、stdout/stderrは計画上8MiB以内に保存する。
wall/CPU time capなし。RSSは50ms目標samplingで瞬間peakを測定済みとしない。
checkerの費用、width算術、wall/CPU、RSS、監視bit長を分けて保存する。
旧producer費用は保存eventの参照として報告できるが、cold timingや新しい量子資源costを生成しない。

## 5. STOPとGPTへ戻す条件

数学入力・source/input hash・proof・最低幅検査不整合は停止。bit/memory/output capは技術STOP。精度増・再試行で埋めない。
新Hamiltonian、δgoal変更、κ最適化、τ/Z探索、eig、E_forcedの数値評価、元batch retry、N1/N2再開は0。
一回評価後は全statusでMandatory STOP。中心テーマnull、次段認可false。

OCC固有のstrict幅差が確認されたら、有限保存集合の事後知見として、差の大きさ・旧法coverage・費用・研究taskをGPTへ渡す。
差なしなら中心候補から外すかの研究判断へ、未判定なら追加計算の情報価値の判断へ戻す。
同内容のGPT科学レビューを再要求するための準備ではない。次に必要なのは、この固定scopeに対する実行認可である。
