# Track A PM-0：post-M2証拠帰属・資源機構の事後解析

2026-10-04。**POSTHOC、保存値のみ、追加科学計算なし。** PM-1以降の実行認可ではない。
M1/M2のraw result、正式status、source、contract、authorization、manifestは変更しない。

## 採用するRQと結論

固定DF Hamiltonianの残差をdiscard、deterministic保持、canonical finite-RTE補完のどれで処理すると、
同じcomplex-signal精度で資源競争力が残るか。その境界をbias、normalization、shot、実装費用から説明する。
新しいalgorithm、selector、sampling分布、一般的resource最適化法の研究へ戻さない。

登録した二次DF-PFの有限候補集合内では、intermediate partialは**固定q=8でも**primary RZ最小であり、
可変qでは絶対資源がさらに下がる。旧S2からのrank6→rank3変化を「q最適化が初めてpartialを有利にした」
証拠とはしない。M2はdevelopmentで固定した5構成のtransfer支持を保持するだけで、held-out method最適性ではない。

利用者提供の方針review `pr2_post_m2_research_redesign_20261004.md`
（project root、SHA-256 `0819a0f358cdbb2029baa9e83c004b870c819100053ae209c9399c6e9707eb78`）を受けた
PM-0である。review原文は移動・編集しない。旧暫定PMラベルを本reviewのPM-0〜PM-3に整合させる。

## 範囲・provenance

入力をcommit `b6e65c6123475add5e620ec1064f361378bead95`のblobとbyte照合してから再集計した。
allowlistはM1-A/M1-B1/M1-B1 validation/M2/旧S2のJSON 5件とsource text 4件だけ。
入力path、bytes、SHA-256は[summary](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/summary.json)に保存する。
artifact内のNPZ/runtime/cache pathを辿らず、分子データのresolve/stat/hash/load、signal、sampling、
circuit build/compile、GPU、旧validatorの再実行は0。現在のworktreeで生成した未commitのlocal posthoc evidenceで、
immutable CI、外部再現、新しい検証、formal confidence intervalではない。

H4 linear、STO-3G、DF rank12、8 system qubits、T=0.8、ε_complex=0.05、各軸ε=0.05/√2、α=0.025。
M1 geometryは1.00 Å、L_D=0/3/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1、random K=2/4。
rのbase gridは1/2/4/8/16/32、r64は下記の2件だけ。
M2 geometryは1.30 Å、以下の共通5構成のみ。B2/B0/B1はdelta=0.8、B3はdelta=0.1。
compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/couplingなし。
primaryは状態準備を含まないfull measured Hadamard wrapperの
`G_RZ=N_real E[C_cosine,RZ]+N_imag E[C_sine,RZ]`。
secondaryはrz_depth、cx_count、cx_depth、total_depth、circuit_size。P感度はRZ-equivalent per-shot costであり実際の状態準備compileではない。

## A. 同じ比較集合で見直す

| 比較集合 | 登録/適格 | primary RZ point最小構成 | G_RZ |
|---|---:|---|---:|
| M1 q=8部分集合 | 53 / 52 | B2 rank3 q8 r2 K2 | 684,061,479.375 |
| M1全登録集合 | 210 / 206 | B2 rank3 q1 r4 K2 | 130,774,896.656 |
| 旧selectorのrandom16件 | 16 / 16 | 同じB2 rank3 q1 r4 K2 | 130,774,896.656 |
| M1のM2共通5構成 | 5 / 5 | 同じB2 rank3 q1 r4 K2 | 130,774,896.656 |
| M2固定5構成 | 5 / 5 | 同じ構成（pointのみ） | 111,753,794.438 |

M1の同じmethod/split候補domainでは固定qでもB2 rank3がprimary最小。
可変qでのB2最小Gはq8最小Gより約80.88%低いが、**最小methodを逆転させた証拠ではない**。
B0/B1もqを選ぶと資源が下がる。例えば最小B2/最小B0比はq8で0.396147、可変qで0.516123であり、
絶対資源の減少をpartialの相対利益の増加と混同できない。B0最小splitはq8では9、可変qでは6である。

旧S2のprimary B2はrank6、rank3/9はcontrolsに限られた。候補集合、trajectory標本もM1と違う。
旧S2の正式decisionを変更せず、S2→M1差をqだけへ帰属しない。
同じmethod/q/r/Kでrankを比較する全登録組と欠測は[rank presence CSV](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/fixed_q_r_K_rank_presence.csv)に保存した。
NOT_REGISTEREDとaccuracy INELIGIBLEとを区別する。異なるq/r/Kのrank別最小値はrankだけの効果ではない。

## B. selectorのprimary regretとfrontier損失は別

proxy非支配64件はactual six-metric Paretoの2件を両方保持する。
旧random16件selectorはr4/K2を保持し、r8/K2を落とした。baseline16件は別枠であり、この16の件数に混ぜない。

| 指標 | 旧16件の最小値regret |
|---|---:|
| primary rz_count | 0% |
| circuit_size | 0% |
| cx_count | 0.6813% |
| cx_depth | 2.2554% |
| rz_depth | 1.2176% |
| total_depth | 1.1995% |

`SELECTION_LIMITED`を撤回しない。結果前には未compile候補を安全に除外できず、frontier全保持にも失敗した。
しかし「primary RZ最適点を落とした」「proxyは研究上大きな主要意思決定誤差を起こした」とは主張しない。
ここはpoint regretの再集計であり、MC uncertaintyを考慮した厳密winner判定ではない。

## C. N×1-shot費用・誤差の分解

`N=N_real+N_imag`、`C_eff=(N_real C_cosine+N_imag C_sine)/N`とすると`G=N C_eff`。
全M1 210件とM2 5件について、各軸のbias/allowance/shots、normalizationとその二乗、
one-shot RZ、n_det/expected random actions/n_fixed、6指標workを[candidate CSV](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/candidate_decomposition.csv)に残した。
不適格4件のN/GはMISSINGとし、compileした1-shot費用をaccuracy適格な比較へ入れない。

| B2 r4/K2 対 endpoint | N比 | C_eff比 | G_RZ比 |
|---|---:|---:|---:|
| M1 common5：B0 rank6 q1 | 0.856185 | 0.602817 | 0.516123 |
| M1 common5：B1 rank12 q1 | 1.113259 | 0.315337 | 0.351052 |
| M1 common5：B3 rank0 q8 r32 K4 | 0.570687 | 0.198781 | 0.113442 |
| M2 common5：B0 | 0.983086 | 0.596173 | 0.586090 |
| M2 common5：B1 | 1.127543 | 0.311504 | 0.351234 |
| M2 common5：B3 | 0.588862 | 0.177455 | 0.104496 |

B1よりshotが少ないからB2が安い、とは説明できない。これらのB2はB1よりNが多く、one-shot費用の低さが勝る。
またnormalizationの二乗だけでshot比を説明せず、各軸の残りallowanceも見る。
各比は登録集合内での点推定比較であり、一般因果分解や強いbaselineへの優位性証明ではない。

保存済み複素値からpartialの外側PF誤差とfinite truncation誤差を**符号付きで**分けた。
絶対値の和をtotal biasと同一視しない。B0の`outer_pf_bias_abs`はfull-targetに対するdiscard＋PF総biasであり、
pure PFではない。exact truncated-H signal `z_D`がこれらの入力にないため、discard/PFの純分解はMISSING。
compiled費用のdet/random/basis/primitive別内訳も保存集約値からは得られず、action proxyへRZを割り当てて捏造しない。

## D. 同じR=q rでqを変える

同じrank/K/T/Rの56比較groupでtau、normalization、expected random applicationsが一致することを照合した。
これは浮動小数のbookkeeping checkで、科学gateの変更ではない。
例：M1 B2 rank3 K2、T=0.8、R=8。全4件でtau=0.05832704、B=1.02753344、expected random actions=8.02712923。

| q / r | n_det proxy | total bias abs | N | C_eff,RZ | G_RZ |
|---|---:|---:|---:|---:|---:|
| 1 / 8 | 6 | 0.00725187 | 19,489 | 6,809.1875 | 132,704,255.188 |
| 2 / 4 | 12 | 0.00173267 | 15,693 | 12,446.9063 | 195,329,299.781 |
| 4 / 2 | 24 | 0.000428445 | 15,015 | 23,791.3125 | 357,226,557.188 |
| 8 / 1 | 48 | 0.000106825 | 14,858 | 46,332.6563 | 688,410,606.563 |

qを増やすとこの組ではbiasとNが減るが、deterministic/fixed workと実際の1-shot費用が増える。
同じRは同じsignalや同じ回路costを意味しない。列の順序、PF分割頻度、boundary fusionを含む。
n_detは旧定義のaction proxyで、fusion後のcompiled block数ではない。
全比較は[same-R CSV](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/same_R_candidate_comparison.csv)から追跡できる。

## E. P感度を共通5構成で比較する

| 候補集合 | B2 r4 → B2 r8 のP | B2 r8 → B1 のP |
|---|---:|---:|
| M1共通5構成 | 1,796.423 | 235,578.461 |
| M2固定5構成 | 385.078 | 208,735.132 |

どちらも大きなPでB1が入る。M1全206適格候補のenvelopeはB2だけだが、これをM2 5候補と直接比べて
geometry-induced method changeとは言わない。閾値の数値はgeometry間で違うが、候補集合の効果を先に分ける。
affine不等式から全P>=0を計算し、P gridを追加しない。交点のtieを保持する点推定感度で、formal intervalではない。
ε sweepはPM-2に属するため今回は行わない。

## F. candidate/baseline監査と次に反証する一点

r64は`B2-rank3-q1-r64-K2`と`B3-rank0-q8-r64-K2`だけ。全q/Kを網羅したboundary試験ではない。
B3 q1/r256など未登録endpointの有利・不利を補間しない。
B0 rank4/5は今回のM1/M2比較domainに存在しない。rank3不適格／rank6適格だけから、その間の性能は決まらない。

source textを読むと、basis changeは既に無制御、diagonal primitivesだけがcontrolledであり、
repeated builderは同一blockのstep境界half-sweepを既に融合している。
全basisを無駄にcontrolしている、fusionが全くない、という反証前提は棄却する。
一般の相対Gaussian変換融合やcontrol-aware合成で改善できるかは**未検証**。
文献再監査、operator/ancilla相対位相test、新policy実装、費用計測は今回行っておらず、opt1だけで強いbaselineと呼ばない。
静的function excerpt・行番号・source hashはsummaryに収録した。

**次に一つだけ数値反証するなら、development B0 rank4/5の近接discard穴を第一候補とする。**
具体的な欠測が確認でき、未証明のoptimizer利得を仮定せず、random samplingなしでB0比較の結論を反証できるため。
提案上限は同じT/task/state/DF表現でq=1/2/4/8、最大8 signal records＋16 deterministic wrappers。
これは**PM-1案、未認可、未実行**。旧M1/M2 authorizationは再利用しない。
登録rank・二次PFの固定実装classに限定したnoteとして閉じるなら、追加計算自体を必須にしない。
強いdeterministic実装一般への優位性をclaimするなら、別途一つの適用可能な合成policyを監査・認可する必要が残る。
高次PF、独立instance、energy/RPE接続は自動追加せず、Track Bのmethod co-designも本作業に混ぜない。

## 成果物・再集計・STOP

新しい[Track A artifact directory](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/)だけへ保存した。
summary fingerprint、入力identity、生成source hash、test記録、new-output manifestは同directoryで確認できる。
専用moduleは[src/trottertracks](../../src/trottertracks/resource_applicability/pm0_evidence_attribution.py)に分離し、
M1/M2のsource collector対象である既存`trotterlib`科学コードを追加・変更しない。
[runner](../../scripts/resource_applicability/run_pr2_post_m2_evidence_attribution.py)はbundleをstdoutへ出すだけで、ファイルを上書きしない。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" \
scripts/resource_applicability/run_pr2_post_m2_evidence_attribution.py --project-root "$PWD"
```

専用test 18 passed、fail/skip0。入力9件を前後byte照合し、candidate/保存signal fingerprint、N×cost、
旧validation regret、same-R identity、P envelope、missing扱い、access allowlist、改変拒否を検査した。
全repository suiteは実行していない。
元M2の`TRANSFER_SUPPORTED`・mandatory STOP・next_stage=falseは維持する。
PM-0終了後STOP。PM-1、PM-2、PM-3、追加96、held-out再開封、別分子、compile、commit/pushは今回行わない。
