# C価格モデルに必要な実compiled証拠：取得準備と有限案

2026-11-13 JST。**初回はPR-LD2/m0のnormal 1 path・X/Y・最大2 compileを提案する。科学実行は未承認・未実行。** 旧LD2/m12 T2を次の仕事とはしない。既存D selector/runtime/fast pipelineとC forwardingで技術接続を満たすため、production code・request builder・compiler・科学runnerを新設していない。

[必要セル・再利用表](NEEDED_CELLS.md)／[有限科学案](FINITE_SCIENCE_PLAN.md)／[C HANDOFF](HANDOFF.md)／[選択metadata・全seed](FINITE_CELLS.json)／[技術確認](TECHNICAL_CHECKS.json)／[回帰ログ](TESTS.txt)。Dの既存branch `codex/d-compiled-residual-c-minus-s-20261113`を継続した。

## 固定SOURCEとCとの整合範囲

| 用途 | 固定SOURCE・対象 |
|---|---|
| D基点・scalar修正済みselector | `e827367629ca6c2920d9d908fc1631314399db86`、`src/trottertracks/ld12_price_requests.py`、`docs/research/d_t2_scalar_binding_fix_20261113/` |
| Cの最新公開価格モデル | `1914d0764115d82ec860e08810dafeb50f8d9a42`、`docs/research/rpe_mean_price_model_preparation_20261113/{REPORT,HANDOFF,NEXT_SCIENCE_PLAN}.md`、`src/trottertracks/rpe_mean_price_model.py` |
| CのD転送 | `3802bcc0212325091515325a889fcab20eabd909`から上記1914まで、`rpe_optimization_pipeline.py`と`rpe_d_price_connection.py`はbyte不変 |
| A固定候補 | `a9abf8de75bf3432fcd714e868e5c14c8884b8f8`、MENUとCONFIGURATION_LD1/LD2 |
| D adapter / fast | `0c6e16342b3c804fa0bd89a4028d6b5bea7b0322` / `5a67cea7705e80ff295306606611d774724ae2e2`からの現行実装 |
| P1 / P2 direct証拠 | `54b64ef72bf507be25970f2d3362d3be091321c3` / `39ddd59edf54a3a901e5302207b2765ddbd03fbb` |
| Full PF receipt正本 | `e7dca5f7301c8633b17a59b6a7161331303aef2d`、`docs/research/d_full_pf_full_h_v2_final_receipts_20261113/{HANDOFF_C.md,COVERAGE_V2.csv,RECEIPTS.json}` |

C remote branch `codex/r-variable-compiled-price-design-20261111`のHEADが1914であることを`git ls-remote`で確認した。Cの最新計画は**検証点・候補・cohort・compile数を選定していない**。Cの加法モデルとS＋単一full-wrapper残差モデルは全て未資格であり、値の供給だけで正式Gへ入らない。本成果はCが記した不足方向とDの候補/取得APIを照合したD側の提案で、C/Bによる科学的な採択・価格精度の指定を代行しない。C branch/worktreeは変更していない。

## 回路・quantity・SOURCE

対象はLinear H4/1.00 Å/STO-3G、保存tol-only DF10・元H/state/snapshot、元順prefix、one-body deterministic、S2、ΔPF=h=.1、K2、M12、元exact有理数κ、P0。E0/E1はいずれもA固定PR-LD2のscheduleから選び、q/Rやlawを新たに作り直さない。

- **S**：元Accountingによる同一trajectory/axisのfull-wrapper構造RZ。basis folding等を含む構造quantity。
- **C**：controlled evolutionから測定までのfull Hadamard wrapperを一体transpileしたRZ数。X/Yは個別取得。
- **D=C−S**：同一trajectory/axisのpaired residual。値の計算・保存は将来の承認済み取得範囲で行う。

`NativeRuntime.context`→元prepare/head構築、元carrier compose→native emitter、`compiled_price_fast`→Accounting/測定付きwrapper/fingerprint/cache/compilerを再利用する。PF head、RTE tail、PF/RTE・piece・outer境界とgate順、controlled evolution、X/Y suffix、controlled scalar phase、最終測定を含む。P0は状態準備**費用**を除外する指定で、ancilla準備・basis change・scalar・測定の削除ではない。H/stateの保存bindingも維持する。

新tail loweringは `CONTROLLED_ZZ_CX_CRZ_CX_V1`。compilerはQiskit1.3.0/NumPy1.26.4、basis rz/sx/x/cx、opt1、seed17、symmetric_directional、backend/target/couplingなし。compiler hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。同じhashだけではlowering/physical/wrapper一致にならない。

scalarはD e827のsource-bound selectorを使い、保存snapshotのc_HとA tail-only s_Rを元順のbinary64 `c_H+s_R`でexpected totalへ接続する。runtimeは復元Hと実prepareから独立に同じ量を求め、従来の`float.hex()`完全一致guardを通す。metadataのexpected値はphaseを追加しない。composeのscalar一回適用を変えず、Cで再加算・丸め直し・tolerance追加をしない。LD2以外のFull/Truncated/Full-Hはそれぞれの既存scalar契約に従う。

## 証拠の再利用と初回案

既存P1/P2のLD1 m0/m6は新ZZ/full-wrapperのdirect path証拠であり、今回再compileしない。A固定LD1とのphysical metadataは一致したが、canonical configuration/実行SOURCE/request形式は異なる。新requestへの完全な科学的identity bridgeを今回成立させたとはしない。元development/replay/fresh roleを保った参照であり、平均資格や独立holdoutへ付け替えない。LD2へ流用しない。

Full PFは既存Dが73候補の両軸receiptをEXACTと照合済み。Cで同じ監査・compileを繰り返さない。例としてH4/S2のT=.8 q8、T=1.6 q16の既存EXACT metadataだけを確認した。元profileで同じ物理候補に使える資産であり、S2/Δ=.1 scheduleのm3/m4に対応し得る。今回このFull PF構成の正式採択や新loweringへの付替えはしない。新ZZ overrideがRTE eventだけに作用するというSOURCE上の事実と、全native/wrapperの価格同一性を承認するbridgeとは区別する。

初回E0はLD2/m0、T=.1、q1/R1、normal index0、X/Y最大2 compile。scalar guard・prefix2 context・paired取得・C mapping用の最小direct receiptを得る案。複数outerを持たないためouter依存を検証できない。別の保留案E1はLD2/m4、T=1.6、q16/R10、normal index0、X/Y最大2 compile。160論理eventのouter/PF-RTE接続を確認するための案で、E0から自動進行しない。**1 pathのmean予測差はmodel誤差と標本変動を分離できない。いずれも適格E[C]、CI、model holdout、長時間外挿の証拠にはならない。**

Cが必要とするR-only、q/R同時増大、Full/Truncated/Full-H別domain、law/rare、長時間のmodel検証は残る。初回1 pathを口実にその全実験を開始しない。実mean modelの検証点/n/誤差基準は、Cの固定仮説とBの比較family・N/必要精度から別途設計する。

## 技術確認・限界・STOP

既存scalar修正17項目＋LD1/LD2 metadata12項目＝**29 PASS（39.70秒）**。人工constant/preparation/word recorderを使い、量子回路を生成・compileしない既存回帰を再利用した。C380転送の境界確認は既存testで実施し、C1914の該当2 moduleが不変であることを追加確認した。作者自身の技術確認である。

E0/E1のrequestを現selectorでmetadata生成し、既存`check_request`を通した。保存snapshotのconstant memberだけを読み、H/stateは復元しない。seedは元carrierのSHA256関数で算出し、RNG/eventを生成しない。metadata生成中はRuntime.context/sampleとAdapter.acquireを禁止した。新たな価格値・保存科学数値のfit/再解析・S/C平均/CI/N/RMSE/Gは0。production/runtime/compilerは基点e827と不変。

**修正SOURCEでの実preparation/scalar guard通過は未取得。** wall/RSS/compile可否も未取得で、event数から実行時間や価格を認定しない。巨大m12実行可能性・価格、window tail bound、ideal/実sampler差、rare寄与はUNKNOWN。旧T2のrun01・STOP、診断、source/input/raw/cache/grant/ledger/marker/KEEPを保全した。新実行には新SOURCE/出力先と明示された科学scopeが必要で、旧one-shot未使用枠を転用しない。

技術成果をcommit/push・remote SHA/tree/主要成果物照合後 **mandatory STOP**。
