# SP-0.5 synthesis-economics gate：結果前契約・source review

2026-10-06 JST。**準備実装・限定technical tests完了、登録target計測は未実行・未認可。**
利用者のGPT review `PROCEED_TO_SYNTHESIS_ECONOMICS_GATE_BEFORE_PRIMITIVE_PILOT`を受領した。
全面再設計は行わず、16-cell wrapper pilotの前に、合成器一つ・cheap catalogue一つの存在可能性gateを置く。
本書とJSONを結果前sourceとしてreviewし、通過後に別authorization-only childと明示的実行指示を固定する。
全outcomeでmandatory STOPし、次段の必要性・scopeはGPT側へ戻す。

## 1. 出典・作業境界

| 資料 | 固定identity／役割 |
|---|---|
| [返却reviewの原文bytes](inputs/sp05_synthesis_economics_review_20261006.txt) | 添付一件だけを選択して保存。元path、bytes、SHA-256は[preparation manifest](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/preparation_manifest_v1.json) |
| [直前のplacement仕様](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3861e6b941745e43863f6a62cd25fe36f3b3e108/docs/tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md) | 設計・実行未認可の履歴。今回はSP-0.5を先行させ、catalogueは一つへ縮小 |
| [SP05 contract](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/contract_v1.json) | target／norm／options／caps／判定／launch方式の正本 |
| [実装](../../../src/trottertracks/algorithm_codesign/synthesis_placement/economics.py)、[launch guard](../../../src/trottertracks/algorithm_codesign/synthesis_placement/launch.py)、[runner](../../../scripts/tracks/algorithm_codesign/run_sp05_synthesis_economics.py) | B専用namespace。共有library実装をimport・変更しない |
| [focused tests](../../../tests/tracks/algorithm_codesign/test_sp05_synthesis_economics.py)、[local report](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/focused_test_report_v1.json) | 34件pass、失敗・skip 0。synthetic gate／launch／budget fixtures。immutable CIや科学結果ではない |

branch `track-b-sp05-economics-preparation-20261006`、worktree
`/home/abe/Project/prt-worktrees/track-b-sp05-economics-preparation-20261006`。
baseは上記`3861e6b...`。root、A、以前のB worktreeは編集しない。
B-F限定negative closure、原BF1 INCONCLUSIVE／R0 BF-A、BM-0.5三次同値性と現adapter new-method closure、
旧BM-1未実行、過去STOPは保持する。Aのsource/result/authorization/artifact/fingerprint/status/pathは変更しない。
M1/M2、H4 1.00／1.30 Åを入力・独立validationに使わない。

## 2. 一つの実際に呼べる合成器

[pygridsynth公式実装](https://github.com/quantum-programming/pygridsynth)の公開release **2.0.0**を使用する。
mutable mainではなく[公開wheel](https://files.pythonhosted.org/packages/ac/3c/f5e71d2ad2bc756fa0bbbb5d10b1adef69fabb722eabb56735381e8341e5/pygridsynth-2.0.0-py3-none-any.whl)の
SHA-256 `30b5b15e9383a8ea8510d54f28e2de0385d102cb54fd4440071abf4525027568`と
installed Python source-tree SHA-256を固定した。これはrelease bytesのsource identityであり、未確認のGit commit SHAは記入しない。
wheel内27 Python filesとinstalled bytesが一致することも確認した。

[tool identity](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/tool_identity_v1.json)と
[runtime lock](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/requirements_runtime.lock)に全28 package版、
Python 3.10.12、pygridsynth／mpmath source hashを保存し、実行時に照合する。
隔離`.venv`だけを準備した。shared projectの`pyproject.toml`はPython>=3.11であり、
今回のstandalone namespace使用を共有packageのインストール・互換性検証とは呼ばない。
distributionのcvxpy等はimport依存として導入したが、mixed-synthesis／solver／plot／numba compilationを実行しない。

固定options：`dps=80, seed=0, dloop=floop=10, dtimeout=ftimeout=500 ms`、
`up_to_phase=true, verbose=0, measure_time=false, show_graph=false`。
内部のfactorization試行数10は既知合成器の固定設定であり、外部gate runのretryは0。
実際のAPI呼出しはidentity角0だけで確認した。登録targetの一般角合成・T-count・Jは未測定。

TとT†は各1、CliffordはT metricで0、full-joint global scalarは0。
返却されたgate stringを保存して実数える。global scalar `W`は別countに保存する。
gate最適化・RUS・catalyst・state reuseを追加しない。T-optimal formulaの証明を主張しない。

## 3. catalogueとtargetを結果前固定

catalogueは **一つ**、`Θ_k=kπ/4, k=0,...,7`、spacing `Δ=π/4`。
exact **channel** Clifford+Tで、even kはClifford、odd kは1 T/T†の明示gate stringを持つ。
たとえば`Rz(π/4)=exp(-iπ/8) T`、`Rz(-π/4)=exp(+iπ/8) T†`。
このscalarはjoint-space native rotation全体のglobalとしてのみ扱う。
π/8等の一般dyadic angleを無料・exactとはしない。
ordinary baselineにも同じexact fast pathを与え、PAIだけに低cost例外を与えない。

登録targetは以下の8件のみ。正負、π-rational、radian-rational、exact Clifford controlを含め、
合成結果を見てtarget・catalogueを追加しない。

| ID | target θ |
|---|---|
| pi16_pos／pi16_neg | ±π/16 |
| pi8_pos／pi8_neg | ±π/8 |
| 3pi16 | 3π/16 |
| rad_1_5_pos／rad_1_5_neg | ±1/5 rad |
| clifford_control | π/2 |

各targetでordinary Rzとcontrolled Pauli loweringのpairを評価する。
target／native signed half-angle／catalogueの重複をexact rational keyで共有し、**23 unique keys**。
これはplan列挙であり、合成実行回数ではない。key capは32。
一度のrun内で同一keyを再利用する以外、以前のA/B cacheを使用しない。

## 4. phaseとfinite-error意味論

`R_P(θ)=exp(-iθP/2)`として、controlled gateは先に

$$C(R_P(θ))=\exp[-iθ(I\otimes P)/4]\exp[+iθ(Z\otimes P)/4]$$

へloweringする。native angleは **+θ/2と-θ/2**、basis/parity loweringはCliffordだけなので
SP05の論理T metricには加算0。実際のwrapper回路構築・compile・workspace評価は行わない。
native pairを独立PAI化し、weight積と`γ_1²γ_2²`を戻す。
system Rzをglobal-phase同値にしてからcontrolする操作は禁止する。relative ancilla phaseを失う。
native joint rotation全体のscalarだけがchannelで消える。

native primitiveごとに共通`ε_op=10^-6`。
一般合成APIへのrequestは`ε_op/4`として余裕を残すが、その返却保証を鵜呑みにしない。
返却gate stringを独立interval arithmetic（mpmath iv 90 dps）でdecodeし、exact target angleと比較する。
16個のscalar witness `exp(ikπ/8), k=0,...,15`のそれぞれについて
`||phase*U_sequence-Rz(θ)||_F`を外向きに評価し、その上限の最小を保存する。
Frobenius上限はprojective operator distanceの上限、unitary channel diamond distanceはその2倍以下。
smallest witness phase／sequence SHA／operator upper／diamond upperをrowに保存する。
これは数学的phase witnessを伴う数値guardであり、合成器の一般optimality証明ではない。

ordinaryと各notchに同じnative error ruleを適用する。
catalogueのchannel exactnessは上の有限gate代数による。数値decode上限も併記する。
approximate branchを使う場合のchannel biasは`Σ_j |g_j| δ_j`、pairでは逐次合成して集約する必要がある。
今回はexact catalogueなのでその理論的biasは0、native deterministic pairのdiamond上限は`4ε_op`。
ordinary対controlled間を同一gateとして比較せず、それぞれ同じnative precisionでdet／PAIを比べる。
finite synthesis bias・wrapper Bernstein range・init／state cost・finite-RTEを戻した最終GOは行わない。

## 5. Jと判定

[PAI v2 §II.2–II.4／App.A](https://arxiv.org/html/2305.19881v2)の既知3-notch channel decompositionを用いる。
`k=floor(θ/Δ)`, `t=θ-kΔ`としてnotchは`k,k+1,k+4`（mod8）、

$$g_2=\sin t/\sin Δ,\quad
g_1=[1+\cos t-g_2(1+\cos Δ)]/2,\quad
g_3=[1-\cos t-g_2(1-\cos Δ)]/2.$$

`γ=Σ|g|`, `p_j=|g_j|/γ`、canonical Wは符号×γ。
g/p/γとJを外向きintervalで保存する。cell locationが不確定なら停止する。

$$J_{ordinary}=\frac{γ²\sum_jp_jC_j}{C_{det}},\qquad
J_{controlled}=\frac{γ_1²γ_2²(\sum_jp_{1j}C_{1j}+\sum_jp_{2j}C_{2j})}{C_{det,1}+C_{det,2}}.$$

`C_det=0`はdivisionせず`ZERO_COST_BASELINE_NO_STRICT_GAIN`。PAI側がcheapでもpositiveにはしない。
各rowはupper J<1なら`STRICT_TRADEOFF`、lower J>=1なら`NO_STRICT_TRADEOFF`、
1を跨ぐintervalなら`NUMERIC_INCONCLUSIVE`。
全23 keysの合成・誤差guardが成功した後だけ、16 primitive rowsを採点する。

- strict witnessが一つ以上：`PRIMITIVE_TRADEOFF_EXISTS`。存在可能性のみ。5% materiality・新規性・最終GOなし。
- witnessなしで不確定rowあり：`INCONCLUSIVE`。
- 全eligible rowでlower J>=1：`NO_PRIMITIVE_TRADEOFF_IN_REGISTERED_SET`。登録domainに限るSTOP情報。
- semantic／error／runtime identity不整合、resource cap、exception：`INCONCLUSIVE`またはlaunch拒否でSTOP、retryなし。

この粗いexact catalogueで成立し得るcheap-gate/sampling trade-offそのものは既知PAIの機構であり、
研究Bの独立method delta・DF-native改善の証拠とはしない。
不通過後の第二catalogue追加・角度追加は禁止。別resource mechanismは新しい結果前review事項。

## 6. 計算上限・実行境界

| 上限 | 固定値 |
|---|---:|
| synthesis keys | 32（plan 23） |
| per-key wall／CPU | 30／20 seconds |
| total wall／CPU | 1200／900 seconds |
| 同時parent＋child RSS | 1024 MiB |
| child virtual address | 2048 MiB |
| 保存result | 2 MiB |
| sequence string | 8192 characters/key |
| 外部retry | 0 |

同時workerは一つ。Linux `fork` workerにCPU／address limit、parentでwall・combined RSSを監視する。
strict存在可能性を測るだけなので20分はhard ceilingであり、所要時間の測定値ではない。
run中cap hitはmarkerを残してINCONCLUSIVE。部分rowを保存し、再開しない。
runtime lock照合、focused-test passとsource SHA照合、clean source、別authorizationを起動前に検証する。

現時点の[authorization JSON](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/authorization.json)は
`PENDING_FINAL_SOURCE_REVIEW`、source commit=null、science execution=false。
source commit **S**をこの準備公開commitに固定し、review通過後に、その**直下child A**で
authorization JSONと任意receipt文書だけを変更する。Sをauthorizationから参照するため自己参照SHAは不要。
HEAD=A、parents=[S]、source/contract/tool/test identity不変、明示実行指示receiptをrunnerが要求する。
fresh SP05 markerをexclusive createしてから最初の登録測定へ入る。旧BF/BM markerは流用しない。

入口は以下。`plan`はtarget/keyを列挙するだけ、合成0。現在の`run`は未承認として拒否する。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/tracks/algorithm_codesign/run_sp05_synthesis_economics.py plan
```

actual runはsource-bound review・別authorization・明示指示後に同じ入口の`run`のみ。
成功・失敗にかかわらず**mandatory STOP**。16-cell wrapper pilotへ自動進行しない。
後続pilotの案は4 templates×4 masks×1 catalogue、keys<=384／branch-axis<=10,368だが、
これはSP05の実行予算でも認可でもない。wrapper側materiality／precision／capsは別contractへ固定する。

## 7. GPTへ返す最終review事項

1. 一つのexact π/4 catalogueとcommon fast pathが、このprimitive gateの目的に適合するか。
2. native error rule、joint-space phase witness、controlled pairのmoment/cost会計に欠落がないか。
3. 8 target／23 key、strict existence判定、資源上限、source→authorization-only child／one-shotが適切か。

必要資料をこの独立B branchへcommit/pushし、完全なsource SHAと固定URLを渡してSTOPする。
このreview packetとlocal testsは科学実行承認ではない。SP05を一回実行する場合も、終了後はGPT側の研究判断へ戻す。
