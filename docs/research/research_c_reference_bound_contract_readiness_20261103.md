# 研究C：B2の状態非依存reference上界・固定契約の完成

2026-11-03。前回B2のNOT_READYを、保存stateの精度を推定せずに進められる**状態非依存のoperator上界**へ具体化した。
既存CAR/ArbとB2 adapterを再利用し、限定B2科学評価を未承認27-call契約として準備した。
**技術的に実行準備完了、実H4科学callは0。Stage C・Whole-P0・actual CAは未資格のまま。**

## 出典・対象と今回の変更

Branch：`codex/research-c-reference-bound-contract-readiness-20261103`。
起点：`2ac4a38dced97406d15fcd157e9d7a0ec7aba982`（[前回B2報告](research_c_reference_numerical_uncertainty_readiness_20261103.md)）。
独立worktree：`/tmp/prt-c-reference-bound-contract-readiness-20261103`。
適用AGENTS/PROJECT_MAP、正式共通指示 `research_project_instructions_unified_2026-10-25.md` §8/8.4/10に従った。
旧B2・B1・統合branch/worktreeのSOURCE・契約・STOPは変更しない。前回NOT_READY文書は当時の記録として保持する。

Stage A科学SOURCE `ddd8333488e07a3b3af9095b5cb193ef9f90b3c3`、結果 `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53`、
元PR科学SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe` を保持する。Stage Aを再実行しない。
Endpoint SHA256：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。

固定条件はLinear H4・隣接1.00 Å/STO-3G、Nα=Nβ=2、sector36、Number CDF rank8、元frame順LD2、
partial ordered symmetric S2、s1/r32/K0、h=δ=.1、m0..12。CA=.00159 Ha、DF reserve=CA/8、
own-DF ground oracle・状態準備費用除外。独立53-bit IID-gridを条件とし、PCG64 joint streamへ拡張しない。
保存P0/Qiskit1.3/opt1/seed17/rz,sx,x,cx/MPFR160 phase/Gaussian・relative fusionは変更せず、native build/compileは行わない。

前回は保存stateで計算した複素信号の精度を基準に、oracleとの距離上界が未取得であることを阻害条件とした。
今回は任意の正規化sector状態ψについて `|ψ†Dψ| <= ||D||2 <= ||D||F` を使う。
したがって、保存stateの数値精度・groundとの距離・ground solveをこのreference誤差の入力にする必要がない。
**保存stateが正確なgroundであると認定したわけではない。** 旧state-specific観測の誤差を取得済みとも説明しない。
これは同じ固定条件に対する保守的な評価経路の選択であり、新geometry・新oracle・新たな保証方式の採択ではない。

## 三つのreference対象を固定する

PとCARの意味を曖昧にしないよう、次の三点を明示する。保存binary64値は正確なdyadic入力として扱い、元積分や物理モデルの誤差とは区別する。

| 対象 | 正確な定義と元資料 |
|---|---|
| M_atom | Stage A atoms.jsonの216 signed atom係数・support・frame、categorical_table.jsonの正確なp、保存Uのliteral CAR Z/ZZ、ideal_event_mean.jsonの保存angle bitを使う数学的平均。S_rev×mean^32×S_fwdとtail identity phaseを含む |
| C_math | 保存generators G0..8を正確な行列入力とする数学的K0 corrected kernel。tailはsum(G3..8)−identity I、one-stepはI−i(τ/λ)tail、32乗。headはh/2=1/20、phaseはexp(−i identity/10)。λ/τ/identityは保存値の正確なbit値 |
| R_num,m | 保存sector_reference.npzのcorrectedと保存normalization.jsonのA_mを使い、元のbinary64 power/scaling経路を新しい限定評価内で再生した行列。浮動小数出力bitを正確な点として比較する |

正確なCAR raw referenceは `R_math,m = (1+τ²)^(-16*2^m) C_math^(2^m)`。
このAは保存normalization.jsonの明示定義。`math.log/exp`やBの浮動小数近似を正確と仮定しない。
CAR側のτ/λは保存configの数学的比であり、黙ってh/rへ簡約しない。
Atom側のangleは保存されたatan出力bitであり、正確なatan(τ)へ置換しない。
原h/gからGを新たに再構築したり、κをHermitian化したり、Uをpolar/unitaryへ置換したりしない。
これらの表現差やscalar/time/normalization経路の差を、下記二項で実際に囲う。

保存atom係数を正規化したpが入力である。係数生成誤差を新たに0と認定したのではなく、Stage Aと同じ固定入力の定義。
G自体の生成誤差、元積分/DF/モデル誤差、既存PF/RTE numerical pilotの厳密化はこの限定評価の対象外。
科学SOURCEと入力identityは保持し、これらの対象外成分が解消したと主張しない。

## 二重計上しない誤差配分

各roundで、Arb384により次の別々のtriangle legを評価する。

```
B_m >= ||M_atom^(2^m) - R_math,m||2
U_m >= ||R_math,m - R_num,m||2
|psi†(M_atom^(2^m)-R_num,m)psi| <= B_m+U_m   (||psi||=1)
```

- `reference_bridge=B_m`：atomと数学的CARの表現差。係数/p/angle/τ/λ・literal U・head/scalarの固定定義に起因する差。
- `reference_uncertainty=U_m`：数学的CARから保存corrected＋binary64 power/scalingへの評価誤差。
- Arbの指数・積・冪・差・Frobenius計算の半径はそれぞれの上界に保持する。半径だけを第三の誤差として再加算しない。
- `reference_uncertainty_included=False` は、B_mにU_mという別legを加えていないことを表す。同じ丸め誤差を両fieldへ重ねない。
- Categoricalのp→qは別枠のまま。qの新評価・TV再計算・categoricalの再加算はしない。
- PF/corrected-RTEは既存engine側、DFはenergy側で一度。今回のbridgeに混ぜない。

この分割では中心残差だけを上界と命名しない。Arbの包囲計算で得る二項なので、値が有限で全coverageが完了したときだけ
固定入力モデルに対する`rigorous_bound`を発行できる。現在は未実行のためH4のB/U値は未取得。
384-bitで非有限・構造不一致・失敗なら後続をSTOPし、precision escalationやfallbackは行わない。
巨大だが有限の上界が得られた場合も、RPE/CAに有用とは認定しない。

Literal saved Uによるatom mapは厳密なunitaryと未確認なので、旧categorical上界のunitary-trajectory前提は未解消。
本評価はnorm growthを保持し、勝手にunitary化したり2でclampしたりしない。
B1・categorical前提・元PF/RTE等の資格を待たずにB2限定評価は準備できるが、Stage C全体へREADYを出さない。

## 実装・再利用・資源

新module：[c_reference_bound_contract.py](../../src/trottertracks/c_reference_bound_contract.py)。
新entry：[run_research_c_reference_bound.py](../../scripts/run_research_c_reference_bound.py)。
既定はpassive preflight。`--execute`は完成した固定契約への別途実行許可後だけ使用する。
契約：[research_c_reference_bound_contract_unapproved_20261103.json](research_c_reference_bound_contract_unapproved_20261103.json)。

再利用した主なものは、`SectorCAR`の整数励起table、`stage_b.atom_operators`、旧B2の`weighted_events`、
`fixed_car.Arithmetic/Matrix/pack/unpack`、`zsxx_enclosure._real/_matrix/_norm`、`_ordered_exponentials`、既存`Calls/resource_plan`。
共有moduleは無変更。新しいCAR engine・native compiler・ground solver・RPE/shot/energy engineは作っていない。
旧state-specific converterのoracle距離条件を真と偽装したり、unknown記録をboundへ書き換えたりせず、
任意正規化状態という別の換算モードを明示した最小の記録assemblerを追加した。
K0の既存Taylor定義をArb行列に接続し、p/角度の既存定義と保存correctedへの二つの比較を追加した。

Arb384で三つのhead指数を独立process並列化し、元と同じcache・順序で左右headを組む。
Atom/CARの二本のArb chainと、保存correctedの一本のbinary64 chainを共有し、各chain内の12回の反復二乗は順に行う。
13roundのnorm評価も独立processへ分配する。Coherent occurrence数2^mと12回の計算上のsquaringは区別する。
FLINTの行列積を使い、不要な全word列挙・ground期待値・full512行列を作らない。

CPU/RAM/worker/wallの人工capは設けず、既存resource_planがaffinity/quota/空きRAM/独立task数から並列数を選ぶ。
小行列に対し、各独立worker内のBLAS1でoversubscriptionを避ける。Arb contextはprocess内に分離する。
shared venv/cache/OS設定/jobを変更せず、将来のrunはfresh output内の専用cacheを使用し、process環境を復元する。
今回の人工試験も既存python-flint0.8.0/FLINT3.3.1をread-onlyで使用した。実H4性能やspeedupは未計測。

## 27 logical callsと停止条件

旧38-call・Stage C16-callの流用ではなく、実装dispatcherと人工E2Eで次を照合した。

| 処理 | calls |
|---|---:|
| 保存metadata / identity / arrays / sector / atom context / atom平均 | 6 |
| 3つのhead exponential | 3 |
| head products / CAR tail polynomial / kernel pair / shared power chains | 4 |
| m0..12の二項上界 | 13 |
| B2専用summary | 1 |
| **計** | **27** |

216atomや各chainの依存積は該当logical unit内の固定した計算。新候補・q/他geometry・追加精度は含まない。
Fresh outputは `artifacts/research_c_reference_bound_contract/2026-11-03/run01`。
retry/resume0、使用済み出力は拒否。first failureは主journalへ記録し、依存後続を停止してpartial出力とSTOPを保持する。
同時に開始済みの自分の独立taskは既存executorが回収する。他jobを止めない。完了後もmandatory STOP。
科学実行は会話上の別途許可で行い、証明書・承認ID・grant新設・承認原文保存は要求しない。

## B1/統合へのinterfaceと検証

将来の完了出力は `reference_bridge` と `reference_uncertainty` の各13row。
既存共通identity（candidate、Stage A SOURCE、IID-grid model、compiler、conditions）を維持し、
sector36・raw complex signal・model_reference・phase sensitive・independent IID kernelsを記録する。
`raw_conversion`は任意正規化状態へのoperator上界。saved_state_used=false、ground_oracle_precision_claimed=false、
physical_input_error_included=falseを添え、scopeを誤解させない。
両legは異なるerror tagを使い、SOURCE/固定契約/resultから出典を辿れるようにする。
実評価のB2記録を出しても、B1 synthesis不在なら既存統合gateがStage Cを拒否することを人工検証した。

新しい人工試験は27件。sector4の発明入力で実際の27-call dispatcher、独立mpmath128桁による既知の対角mapとの照合、
反復二乗、norm growth、二項配分、保存state未使用、candidate/array/schema不一致、SOURCE/model/precision変更、
並列head、fresh/partial FAIL/STOP/retry拒否、Stage C拒否を確認した。
実H4保存行列の数値decode・演算・再解析は0。既存B2/統合/accuracy等との回帰結果はtechnical成果物へ記録する。

## 技術判定と次の段階

**TECHNICAL_GO_UNAPPROVED_B2_REFERENCE_ONLY**。固定された384-bit/27-callの限定評価を準備した。
H4 bound取得済み・Stage A再実行済み・all-law/actual CA達成とは説明しない。

次はユーザー：この新しい固定B2契約に対する27-call実行を別途許可する。
その後Codex：fresh run01を一度評価し、B/U各13rowと未解決の全体前提を報告・公開し、mandatory STOPする。
Stage C、sampling、native、別geometry、重要科学レビューはその許可に含めない。
今回の作業は契約準備と人工検証・公開までで停止する。

## 固定・検証記録

人工・回帰 **355 PASS、新規27件・既存328件、既知の旧2件を理由付きdeselect**。
旧Stage A消費後のfresh PASS期待と並行失敗の表面例外順序という既存2除外で、
消費済みSTOP拒否と主journalの両失敗順序は代替テストで通過した。旧テストは変更していない。
[実行条件・結果](../../artifacts/research_c_reference_bound_contract/2026-11-03/technical/technical_checks.json)、
[passive preflight](../../artifacts/research_c_reference_bound_contract/2026-11-03/technical/passive_preflight.json)。

実装SOURCE commit：`918c63cc2a959f59f2817388bf6be549b905e3e6`。
[SOURCE manifest](../../artifacts/research_c_reference_bound_contract/2026-11-03/technical/SOURCE.json) SHA256：`07eef25fe16ee42beaad79f02caacb406a7ca08b71ce72e973c7f6b1521ac443`。
固定未承認契約SHA256：`a346b066f65e071f4e8c5d6912dfb8e62a45d9eea260161d8c4e04617b29c44b`。
新規3 SOURCEと継承171 SOURCEを固定し、新契約の8保存入力もbyte照合した。
Passive preflightは数値libraryのimportを遮断してPASSし、Git内の実装snapshotとも照合した。

元checkoutのHEAD `e098c54c78f589055082f9cfc2b13de50c90ca94`、index SHA256 `68a93531c7bc1993d49f6750ac8d98a54fafb8e153ddd0b887dd17b28ddd9c4c`、
開始時のstatus178行が一致した。旧B2/Stage A/B1/統合の固定SOURCE・raw・契約・STOPは無変更。
報告内のローカルリンク7件を確認した。今回のGit書込みは専用branchとworktree登録に限る。
