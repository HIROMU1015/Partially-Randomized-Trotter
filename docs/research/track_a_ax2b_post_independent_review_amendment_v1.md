# Track A AX-2B：独立レビュー後の準備追補 v1

2026-10-10 JST。利用者共有の[GPT独立科学レビュー](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)と「作業を進めて」を受けた**準備のみ**の追補。
Codexによる独立科学レビューの代行・H6 launchの承認ではない。
`H6_NOT_AUTHORIZED`、`DRAFT_NOT_AUTHORIZATION`、mandatory STOPを維持する。
旧[H4後レビュー案](track_a_ax2b_h4_post_scientific_review_v1.md)、[不足11項目](track_a_ax2b_h4_validation_gaps_v1.md)、
[H6案 v1](track_a_ax2b_h6_pilot_contract_draft_v1.md)は当時の草案として保全する。

## 方針と契約の変更点

RQ-Rを主、RQ-P1を補助とする。H4 v5を登録範囲の技術証拠として扱う。
総u未認定・accuracy UNDETERMINED・N/G未評価は旧結果のまま。
全演算の厳密certificateをあらゆる技術pilotの一律前提にしない。
一方、technical agreement・empirical numerical validation・certified boundを別fieldにし、経験的差をu_boundと呼ばない。
厳密な総資源認定とは別に、明示した数値仮定への感度を持つ条件付きresource studyを設計する。
この追補で新しい精度・総資源結論を得たわけではない。

targetは保存binary64 DF係数が表す数学的H_DFと、保存vectorを数学的に正規化した指定state。
元積分Hamiltonian、理想DF分解、真の基底stateへ置換しない。
normalization/Hermitization/cutoff/one-body correction/表現誤差の層を追跡し、方式biasと演算誤差を分ける。
状態準備を除くlogical RZ-workを主指標として維持し、物理T costやenergy estimationへ拡張しない。

## 今回の実装と検証範囲

共有`trotterlib`、H4 v5 controller/native、旧freeze/results/失敗記録を変更せず、追加モジュールへ分離した。

| 追加source | 実装した準備機能 | 今回確認した範囲 |
|---|---|---|
| [numerical accounting](../../src/trottertracks/resource_applicability/ax2b_numerical_accounting.py) | 三状態、u-aware/log-domain shot診断、非unitary誤差伝播の一般会計 | 合成値、u=0旧整数再現、境界・overflow。uの妥当性を認定しない |
| [independent reference](../../src/trottertracks/resource_applicability/ax2b_independent_reference.py) | 独立occupation ladder構成、G²前のfull occupation作用、binary64整数比→mpmath参照 | 小型合成行列、scalar/符号/sector、80/120桁のfixture。H4実入力未使用 |
| [H6 input adapter](../../src/trottertracks/resource_applicability/ax2b_h6_input.py) | tol-only直接integral-to-DF、actual kwargs/rank、one-body補正・Hermitization/cutoff記録 | mock decompositionのみ。None→config rankを通さない |
| [H6 contract](../../src/trottertracks/resource_applicability/ax2b_h6_contract.py) | 7 cell/36 wrapper、seed、actual-rankによる時間集合、未認可proposal | rankは合成整数のみ。分子rankやinputは取得していない |
| [H6 controller](../../src/trottertracks/resource_applicability/ax2b_h6_controller.py) | 別orchestration、before-call matvec/solver cap、bounded sector matrix、writer、scale guard、partial/terminal | fake port/solverと合成行列。molecular backendは未接続 |
| [H6 watchdog](../../src/trottertracks/resource_applicability/ax2b_h6_watchdog.py) | 外部一worker、phase別wall/total/log/output、process group停止 | dummy process。H6分子workerは起動していない |

[専用tests](../../tests/tracks/resource_applicability/test_ax2b_post_review_preparation.py)は
real artifacts/NPZ読み込み、分子builder、random sampling、QuantumCircuit構築/transpileを拒否する。
finite eventは小型fixtureを完全列挙し、確率重み・phase・独立2 occurrence・Hadamard X/Y期待値を照合する。
これはmolecular native wrapperとの接続完了やfresh-shot samplerの独立性を認定するものではない。
local synthetic検証であり、H4/H6科学結果、CI再現、総uの認証とは扱わない。

[metadata専用CLI](../../scripts/resource_applicability/prepare_track_a_ax2b_post_review.py)は
実rank=nullのproposalだけを保存する。`--execute`を持たず、input生成・scientific launcherを供給しない。
H6分子backendの統合は後続準備。旧H4 launcherをH6へ転用しない。

## H4-N/A/E/M：結果前に固定する限定検証

入力はlinear H4 1.00 Å、STO-3G、legacy DF rank12/generation-prefix、保存指定state/36-dimensional sector、T=0.8。
旧8登録cellのみを対象にし、新Hamiltonian/stateを生成しない。
新科学実行は未認可。以下の上限は**提案**で、割当や実行認可ではない。

| 作業 | 手順・backend | 保存項目と判定 | 提案上限 |
|---|---|---|---|
| H4-N | 同一保存係数を独立occupation構成し既存sector matvec全列と比較。mpmath1.3.0の80/120 decimal digitsで係数を整数比からliftし、direct expmで同じ指定stateのsignalを照合 | input/basis/source hash、構成差、normalization差、精度間/経路差。expm/eigh差×任意係数を厳密uにしない | reference dimension36、2精度、phase900秒、AS8GiB |
| H4-A | 既存8 cellの実scheduleを展開。Horner経路と独立構成に基づく別精度の多項式作用、negative S4/merged/half/undo時間を記録 | local/stage norm、raw/corrected/logB、inner product、数値差とsigned bias分解。中間stateを正規化しない | 8 cell×2精度、phase1800秒、total2700秒 |
| H4-E | 小型finite完全列挙→平均operator→各unitary eventのcontrolled/Hadamard期待値。代表prepared molecular eventのphase/basis/registerとfresh IID draw範囲の接続を準備 | 小型の完全確率質量、phase、独立occurrence、ordinary/directionalと両axis、molecular未検証範囲 | toy event200/fixture、科学sampling/compile0。molecular確認は別scope |
| H4-M | u=0旧式、境界、片軸不明、largeB/logN、allowance伝播を合成値で検査 | `evidence_kind`、両軸の三状態、headroom、u/h、logN/整数。実N/Gは別解析として保存 | 今回syntheticのみ。実会計未実施 |

H4-N/Aの具体的scientific runner・全stage高精度port・上限付き実行監査はまだ未接続。
future実行前に、solver不要/再生成なし、実matvec/primitive counts、CPU/output/log/diag上限、source-bound planを固定する。
80/120桁の一致だけからroundoff保証を導かない。構成の独立性と演算精度の独立性を別々に確認する。
未来のprobe coverageが上限を超える場合は削らずSTOPして計画を見直す。

## H6：v1への追補条件

[H6準備契約 v2](track_a_ax2b_h6_pilot_preparation_contract_v2.md)が新しい準備入口。
7 cell/36 wrapper、n=2工程sample、symmetric_directional primary/ordinary paired sensitivityを維持する。
tol-only案はadapterでconfig fallbackを無効化する。旧H4 explicit rank12の結果は変更しない。
DF政策が異なるH4/H6を共通rank-policyのサイズ系列と呼ばない。main比較のlegacy bridgeは未完了。

## 11項目の判断点別優先順位

| ID | 今回の進展 | 次の必須確認 |
|---|---|---|
| U-N1 target/state/u | targetと3証拠区分を明文化 | H4の同一保存target確認、H6の入力固定 |
| U-N2 reference roundoff | 独立構成・mpmath fixtureとH4-N計画 | 分子H4での構成/精度比較、u_empiricalの説明 |
| U-N3 stage/非unitary | schedule列挙、一般伝播、scale guard | actual primitive/全stageのnorm記録と累積会計 |
| U-N4 u-aware shots | synthetic会計実装 | justified u/B上側値による条件付き解析。technical N/G=nullの間は未使用 |
| U-N5 estimator | toy完全列挙と理想測定axisを検査 | prepared molecular phase/register、native control、fresh IIDの実接続 |
| U-C1 exploration | 7/36技術scope維持 | 本比較の共通prefix/q/R/K、強いbaseline、boundary/quota |
| U-C2 cost mean | n=2工程用途を維持 | 母平均/rare event/独立confirmationのmain標本設計 |
| U-H1 input/DF | tol-only adapterのmock検証 | 別認可での入力生成、actual kwargs/L/ordering/cutoff/sector/hash |
| U-H2 orchestration | 別controller、bounded calls/writer、watchdogのfixture検証 | molecular port、CPU/RAM、source/input/env、probe/gate上界、別launch grant |
| U-P1 predictor | 旧FEW凍結、domain外N/A | mainでのsize対応特徴・予測時点・取得費用 |
| U-I1 independence | H6はdevelopment、H8接触0 | H8前のモデル/仮説/候補/標本freeze |

B0のexact truncated-H/signed分解は**既存H4 p6/q4**で保存済み。旧全candidateの分解が揃ったとはしない。
旧v3 MemoryError/v4 wall STOPは失敗として保持する。
H4全28 wrapper再compile、標本増量、FEW refit、QPE/物理資源化は今回実施しない。

## 成果物・停止位置

準備の[別inventory](../../artifacts/resource_applicability/track_a_ax2b_post_review_preparation/2026-10-10/preparation_inventory_v1.json)、
[H4限定検証案](../../artifacts/resource_applicability/track_a_ax2b_post_review_preparation/2026-10-10/h4_limited_validation_plan_v1.json)、
[H6 metadata案](../../artifacts/resource_applicability/track_a_ax2b_post_review_preparation/2026-10-10/h6_preparation_proposal_v1.json)へ辿る。
旧科学manifest・結果statusは不変。新しいsource固定は旧v5実行時freezeとは別であり、旧実行を新commit実行へ書き換えない。
prepared componentsはH6 launch-readyではない。入力作成と科学実行の別scope/予算/認可が残る。
target/metric/PF-control意味論/独立性/主要比較範囲の変更、正しさの未解決矛盾が出たら早期にGPTへ戻る。
通常の保全的な実装修正ごとに本格科学レビューをやり直す必要はない。
H4限定検証・H6 technical結果後にGPTがH6 mainを判断し、H8へはその後のfreezeと別判断が必要。
