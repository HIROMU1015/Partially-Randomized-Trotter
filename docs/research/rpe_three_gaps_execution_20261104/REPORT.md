# RPE A/C限定実行：95%条件と保存DF referenceを確認、Float64 decoderは未保証

2026-11-04 JST / LOCAL。ユーザーの別途明示実行許可後、公開SOURCE
[`d29a86963f91044a95217d7d2e2ee9703d2f1dc1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/d29a86963f91044a95217d7d2e2ee9703d2f1dc1)
を無変更でA+Cの**初回1 run・各1 pass**実行した。exit0、`COMPLETED_MANDATORY_STOP`。

**保存αを変更せず、Aの直接Hoeffding失敗上界の和≤.05が成立した。
Cも70個のLDL* pivotすべてのstrict正値から保存DF sectorの最小固有値を包含し、reference誤差込みenergy目標を満たした。**
Bは未実行・UNKNOWNであり、実装Float64 decoder、実測定全体の成功保証には昇格しない。

## 固定条件・入力・SOURCE

Linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D=3、単一δ=.02、
保存physical stateのnormalized ray、8qubit・4電子sector70、m0..17の保存q/r/K/N/α/β/ε/E_ref。
保存E_refは−2.1663874486347616 Ha。energy目標は元binary64の0.00015936001019904 Haを維持した。

[固定計画](../rpe_three_gaps_preparation_20261104/EXECUTION_PLAN.md)、
[証明・対象の区別](../rpe_three_gaps_preparation_20261104/PROOF_AND_SCOPE.md)、
[固定17入力/SOURCEのcommit/path/blob/SHA256](../rpe_three_gaps_preparation_20261104/fixed_inputs.json)。
SOURCE8ファイルと17入力が公開commitのbyteと一致することを実行前後に照合した。
旧D結果89cb94fe、旧D SOURCE6942e517、旧36軸包含d2e7a7db/f4905fff/m1734840ce9、V1/V2は再計算・変更しない。
旧Dの幾何・理想branch・alias/literal energy結果は固定rawから再利用した。

## 個別結果

| 課題 | 実行状況 | 保証可能範囲 | 未保証事項 | 次の科学実行 |
|---|---|---|---|---|
| αの95%保証 | A160-bit・36軸各1 pass、成立 | 同じideal IID有界shotの全軸G。成功下界≥0.950307030063252182 | fixed PRNGのIID・実測定bridgeは仮定のまま | 今回の固定A範囲では追加不要 |
| Float64 decoder | B未実行、UNKNOWN | 前回の人工有限prefixのみ | 全域の一様angle/root/distance誤差・coverage | 本SOURCEにH4実行入口なし。別指示で静的証明の準備が先 |
| E₀ reference | C256-bit・1 pass、成立 | 保存DF係数のexact polynomialの4電子sector最小固有値。η_ref上端≤1.00000000000000001e−8 Ha | upstream構築誤差、元分子/他sector/実state preparation、実装decoder | 今回の固定C範囲では追加不要 |

### A：保存αを変更しない確率上界

全36軸の保証付きU_cat_upper、保存N/εを使い、t=ε−U_cat_upper>0を確認した。
各軸の2 exp(−N t²/2)を160-bit Arbで外向き包含し、そのdyadic上端をexact rationalで加算した。

    failure_upper_exact =
      1218468077447179233166734734097595332511768353948680311
      / 24519928653854221733733552434404946937899825954937634816

    success_lower_exact =
      23301460576407042500566817700307351605388057600988954505
      / 24519928653854221733733552434404946937899825954937634816

失敗上界の表示は**≤0.0496929699367478180**、成功下界の表示は**≥0.950307030063252182**。
exact1/20との比較が成立し、`DIRECT_HOEFFDING_95_SUFFICIENT`となった。
全36軸の保存入力・t下端・tail上端表示は[AXIS_TAIL_BOUNDS.tsv](AXIS_TAIL_BOUNDS.tsv)、
exact端点は[原A.json](../../../artifacts/rpe_three_gaps_validation/2026-11-04/run01/A.json)。

旧保存Σα=0.050000000000000004626684は変更していない。
旧α和によるUNKNOWN/STOPを上書きせず、同じN/ε/Uから得た直接失敗上界で95%条件を確認した別結果である。
新しいα配分、N/ε/U_catの調整、二項確率の中心点移送はない。
union boundにround間・軸間の独立性は不要だが、各軸Hoeffdingのshot内IID・有界性は条件として残る。
U_catをphase予算へ再加算しない。

### C：保存DF sector最小固有値とenergy接続

保存NPZの係数とstateを読み、完全なsorted4電子sector、H/state identity、shape/dtypeを確認した。
保存one_bodyと12個のgの**exact binary64 Hermitian性**が成立した。
係数をsymmetrizeせず、H=cI+T(h)+Σλ_l T(g_l)²を256-bit Acbで包含した。
既存の整数excitation符号・basis順序を変更せず再利用し、SCF/DF再生成やfloat materializer/eighは呼んでいない。

固定trial L=lower(E_ref−10^−8 Ha)に対するH−LIのLDL*は**70 pivotすべてstrict正**。
従ってL<λ_min。保存raw state vのnormalized-ray Rayleigh商からλ_min≤Rを得た。
この下界と変分上界の組合せが最小固有値を識別し、stateの小residualだけでgroundと判断していない。

| 保存結果 | 外向き表示 |
|---|---:|
| λ_minの包含下端 | −2.16638745863476163 Ha |
| λ_minの包含上端 | −2.16638744863476234 Ha |
| η_ref上端 | 1.00000000000000001e−8 Ha |
| 旧literal energy上端＋η_refの上端 | **0.000138689200360119854 Ha** |
| 保存energy目標とのstrict比較 | SUFFICIENT |
| 同じsector固有値のalias window内判定 | SUFFICIENT |

η_refの幅は事前固定したtrial shiftに対応する保証付き上界であり、実際のreference誤差が10^−8であるという推定ではない。
追加shift・precision・別solverで幅を狭める計算は行っていない。
formal端点・全pivot・Rayleigh/norm²区間は[原C.json](../../../artifacts/rpe_three_gaps_validation/2026-11-04/run01/C.json)。
表示は既存端点の上向き／下向きdecimal整形のみで、新しい科学式の再評価ではない。

移送先は**保存DFのbinary64係数をexactとするHamiltonianの4電子sector最小固有値**。
丸め済みfloat dense行列や元の厳密な分子Hamiltonian、他sectorのglobal groundとは区別する。
upstream積分/DF構築・basis/model誤差の上界を新たに取得したわけではない。
η_refは旧literal energy上端へ一度だけ加え、qδ η_refをphase予算へ重複加算していない。

## 条件付き数学的成功と、残る保証の区別

旧Dで確認したG→共通位相→理想branch→literal energyの含意へ、AのPr(G)下界とCのreference包含を接続できた。
従って指定ideal categorical/IID測定・**数学的decoder**では、保存DF sector ground energyに対して
上記energy上界を満たす条件付き成功確率が少なくとも上記0.950307030063252182である。
これは実decoderのjoint失敗確率そのものを求めた結果ではない。

実装Float64 decoderの一様a/c/dは未取得で、Bは`UNKNOWN/STOP`、scientific_calls=0。
`implementation_energy_upper=null`、`real_molecular_energy_decision=UNKNOWN/STOP`を保持した。
その誤差を0として実装全体へ移送していない。physical stateとPF有効固有stateを同一視せず、
経験的PF係数をrigorous上界へ昇格しない。
fixed PRNG/IID、全18round実回路、compiled/noise、実state preparationのbridgeも未保証のまま。

## 実行・資源・原本保全

Aは36 tailセルとexact上端和1回。Cはsnapshot検査・sector構成1セル、13 one-body lifts、
12 square、LDL*1回、norm²/Rayleigh1回、η/energy移送1回。
source中のshape/dtype照合等に伴うNPZ member再読込みを含むが、科学的なH構成・reference検証は各1 passである。
旧mean/phase/branch包含の再計算、sampling/RNG replay、回路build/compile、費用評価、eigsh/eighは0。
retry/resume、精度追加、shift/予算/decoder/対象変更も0。A/Cとも例外・途中UNKNOWNはなかった。

Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、python-flint0.9.0／FLINT3.6.0、IEEE binary64。
CPU affinity32、事前MemAvailable54,287,812 kB。GPU driverは利用不可、固定CPU Arb/Acb経路を使用した。
整数mapとT(g)を再利用し、native FLINTの行列積と一度の下三角LDL*で評価した。
process poolや全スペクトル計算を追加せず、人工CPU/RAM/worker/wall-time上限・共有設定変更なし。
task評価は約0.339秒、process wall0.52秒、maximum RSS103,648 kB。比較実測によるspeedupは主張しない。

原RESULTのSHA256は`707c22df15d690348bc081d9249a5598deeec1353447a4e5c85a12a9be5434f3`。
A/C/RESULTの原byteを保存し、生成後の編集・再生成はしていない。stdout/stderrも原byte、stderrは空。
resource公開記録は私有interpreter pathを省いた[resource_summary.json](../../../artifacts/rpe_three_gaps_validation/2026-11-04/run01/resource_summary.json)。
原time記録は私有領域に保全した。
[VALIDATION.json](VALIDATION.json)にSOURCE/入力/原出力hash・判定/inventory照合を記録する。
準備時の28人工PASSを再利用し、無変更のテストは再実行していない。
実行後の確認は保存結果・identity・転記の技術照合であり、科学式の再実行ではない。
LOCAL固定SOURCEの主agent結果であり、immutable CI・外部再現・独立科学レビューではない。

## 次の担当・mandatory STOP

原checkout、Codex C、他branch/worktree・SERVER1/2、旧SOURCE/raw/grant/ledger/marker/STOPは無変更。
新結果に対応する概要・validation-specific報告・当日研究ノート・machine manifestを追記し、過去の判断を保存する。
今回のA/C範囲に追加科学実行は不要。
次は**ユーザー**がBの一様primitive/全距離証明の技術準備を進める範囲を指定し、
**Codex（別指示後）**がその静的準備を行う。BのH4科学実行や重要レビューへ自動移行しない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持。
**結果公開・remote SHA/公開成果物照合後mandatory STOP。**
