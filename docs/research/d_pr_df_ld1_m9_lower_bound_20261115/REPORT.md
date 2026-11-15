# PR-DF LD1 m9-X：期待compiled RZ価格の片側下界のSOURCE調査

2026-11-15 JST。対象はphysical `06a78c7f9856f373336f`、m9-X、q=512/R=1020、X軸25 shotの1軸のみ。

- **結果**：固定compilerの最適化後Cについて、順位判定に使える非自明な全support下界h、又は一様な残差下限bを今回のSOURCE論証では確立できなかった。状態は **UNKNOWN_NONTRIVIAL_LOWER_BOUND_NOT_PROVED**。下界が存在しないと証明したものではない。
- **数値評価**：非自明な論証が未成立のため、ℓと25ℓの新規評価は行わずnull。既存のC≥0という自明な下界は有効だが、両PF基準への順位情報を与えない。モデル参照額約28.29億RZを下界に使用しない。
- **PFとの比較**：B0I p7 S2の593,555,352 RZ、Full PF S2の838,693,580 RZに対して、固定LD1が高い／安いのいずれも今回からは認定できない。

[機械可読正本](ASSESSMENT.json)／[統合担当・C向けHANDOFF](HANDOFF.md)／[静的検査](CHECKS.json)。既存LD1 branch `codex/d2-pr-df-ld1-m3-acquisition-20261115`、起点85dab58dを再利用した。LD2専用branchは編集していない。新path・build・compile・再fit・Q3・F0再実行・E[S]再計算は0。

## 1. 対象と保存入力

保存linear H4/1.00 Å/STO-3G/tol-only DF10、PR-DF LD1、S2、Δ=.1、K2、元exact κ/H/state/snapshot/fragment・basis順。価格CはP=0、scalar c+s_Rを総Tに一回、symmetric-directional control、測定付きfull controlled Hadamard X wrapperの最適化後RZ数。new ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend/coupling/layout追加指定なしを固定する。

| 固定資料 | 今回使う入口・役割 |
|---|---|
| a82b0f0e `…/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json` | candidates[0].rows[18]：physical、X、N=25。allocation `PR_DF_LD1_TOTAL_Q3_SAVED_CONTROL_01` |
| 950577f4 `…/d_pr_df_ld1_ec_plan_20261115/AXIS_EVIDENCE.json` | rows[18]：m9、T51.2/q512/R1020、configuration/canonical IDと元Cの対応 |
| 9d95d3ea `…/h4_structural_schedule_results_20261113/run01/COVERAGE.json` | values[18]：同physical/cosineの保存E_I[S]と数値enclosure、law_binding |
| dcf905fc `…/d_pr_df_ld1_f0_rz_range_20261115/RANGE_EVIDENCE.json` | rows[18]とsources：既存のC/S上下側envelope、compiler SOURCE調査 |
| 85dab58d `…/d_pr_df_ld1_reference_budget_20261115/run01/RESULT.json` | rows[18]の加重参照額2828507555.268906326 RZを**対象選定の動機だけ**に引用。新しい数値入力・下界にはしない |
| 8dc41b35 `…/d_pr_df_ld2_saved_price_assessment_20261115/NEXT_ONE_TASK.json` | 前回の1軸下界案。今回ユーザーが指定した範囲を実施 |
| 1002f1df `…/pf6_allround_cost_20261115/RESULT.json` | PFの採択済み閉包条件付き2基準。再集計しない |

全commit/path/selectorとgit blobはASSESSMENT.inputs/sourcesに固定した。Q3・既存軸表・構造表のm/physical/axis/q/RとNをmetadataで照合した。構造表binding内のT=.8は参照profileの値で、m9の回路Tへ流用せず、candidate側T51.2を用いる。

対象Iは保存SOURCEの理想finite-RTE law。同じlabel ωは、outer順、各eventのorder、rotation選択、順序付きproduct選択を持ち、保存component/basis/sign/angle/phaseから固定wordへ写る。order0の観測だけへsupportを限定せず、idealで正のlabelをCDF格子上の欠落や未観測を理由に除かない。SOURCE宣言71ba49bbとprovider c31ef424の `src/trotterlib/rte.py` は同じblob `53601b7c…` だった。IをPCG64 seed ensembleへ変える作業ではない。

## 2. §10.5：前回以降の差分

前回8dc41b35の照合を再利用し、originをfetchして、D2 LD1/LD2、C、Dの残差/P2、B2、統合ロードマップ、Bの最新計画、AのPauli計画の固定refと関連履歴を限定照合した。m9-Xの同等な全support下界証明・評価は確認範囲で未同定。既知の同名local jobも見つからず、新出力先は開始時に存在しなかった。取得不能な他serverや未列挙作業の不存在は主張しない。

前回以降のLD2原receipt公開追補は条件不一致の旧候補、CのF1はFull-H、B2はPauliの参照総額、Bの新計画は決定論PFの幅/shot比較、Aの更新もPauliであり、今回のLD1下界を供給するものではない。LD2 receiptの公開を理由にLD1の数値へ移送していない。比較に必要な既存の上界調査・P0の相殺証拠を再利用し、新規の回路検証はしなかった。checked_refsはASSESSMENT.duplication_deltaに保存した。

## 3. 必要な論証と、今ある量の区別

正常に終了する固定生成・compilerが定める価格関数Cについて、Iの全supportで次のいずれかが必要である。取得失敗・OOM・未定義の回路を0価格へ補完してはいけない。今回の否定的結果は、正常終了を仮定した標準SOURCE上でも非自明な下側不変量が未成立、という意味である。

```text
h(ω) ≤ C(ω)  ⇒  E_I[h] ≤ E_I[C]
C(ω) − S(ω) ≥ b  ⇒  E_I[S] + b ≤ E_I[C].
```

保存E_I[S]のcentralは **108641024.20905557 RZ**、保存数値enclosureは **[108641024.0948432,108641024.3358428] RZ**、statusはK0_PARITY_PASS_K1_NUMERIC_ENCLOSURE。すべて読取・引用で、今回再計算していない。将来bが得られても、下側enclosureにはcentralをそのまま使わず、受入済み数値契約の下端を用いる必要がある。この区間はCIでもcompiler補正でもない。

既存dcf905fcのm9-Xは **C_lower=0、S_upper=1088759301、direct_residual_lower=null**。そこからの既存の粗い下限 `C−S ≥ −U_S` は有効だが、保存構造enclosureの上端もU_S未満なので、`E_I[S]−U_S` は負となりC≥0を改善しない。この不要な差額の数値計算は行っていない。C上界、総basis gate数上界、残差の変動幅やF0のlaw間期待残差差上界を、残差期待値又はcompiled価格の下界へ反転できない。

## 4. 実際の入口と下界を妨げた処理

SOURCEの存在だけで適用済みとせず、同path S/C入口→compose→wrapper→compile→active Rust最適化まで追った。Accountingの旧 `predict` 入口にはq8/q16のguardがあるが、保存m9のSは既存構造kernel、同path一般qの経路は `structural_pair` であり、旧predictをm9へ呼ぶ前提ではない。共通のAccountingクラスとFastPath/NewZZのworking fileは各固定blobと一致した。

| 経路・固定SOURCE | 確認できたこと | 下側の論証に不足する点 |
|---|---|---|
| Accounting 8457c443 `h4_df10_price_v1.py` :26–49,114–145,167–229 | SはliteralなGaussian/CRZ等の展開、全op列一致の隣接逆basis削除、wire別の隣接RZ加算。非RZが当該wireを使うとpending Zをflush。compiler再合成の結果ではない | Sがcompiled後に残るRZ数以下だという不変量はない |
| EventLabels c31ef424 `rte.py` :1594–1659、LegacyBuilder `df_rte_qiskit.py` :244–351 | K2 product→rotation、符号/相対phase、同basis共有を保持してtailを構成 | ordered productの削除、basis遷移、触れないwireでの部分相殺を、独立eventの価格へ足し上げられない |
| NewZZ 39ddd59e `efficient_controlled_zz.py` :21–43 | 二wire rotationのみCX–CRZ–CXへ変更、他の役割・位相は元経路 | 正しいloweringと「そのCRZの2 RZが最終的に必ず残る」は別の主張 |
| Compose 155529e2 `h46_long_time_native.py` :38–67、NativeBuilder c31ef424 :169–205,259–283 | 各outerのhead→tail→reverseを1回路へ連結し、scalar総T一回、H/evolution/H/measureのX wrapperを作る | event・request・outer・head/tailの境界に保護barrierはない。headだけを別compileした価格も全wrapper価格の下界とは言えない |
| FastPath 39ddd59e `compiled_price_fast.py` :43–71,111–159 | Sはwrapperのpending basis/Zをfinish前に維持、Cは同じevent requestから元compose・X wrapperを通り、固定compilerの最終RZを数える | X/Yの同値、wrapper費の独立な加算を仮定できない。今回Yは調べていない |
| CompilerCall c31ef424 `rte_compiled_cost.py` :843–893 | rz/sx/x/cx、opt1、seed17等をtranspileへ渡し、post_counts['rz']がC | untranspiled命令数又は局所のRZ数はCへの下界にならない |

Qiskit1.3.0のrelease commitは `896d1c8b72888a52a46081a6d1f4de8c56db8aea`。次のactive処理を読んだ。

1. `builtin_plugins.py` :565–586,633–674のopt1は **Optimize1qGatesDecomposition→InverseCancellation** をdepth/sizeのfixed pointまで繰り返す。layout/routing追加はないが、1回の局所処理だけで終了するわけではない。
2. Python `optimize_1q_decomposition.py` :204–219はRust `optimize_1q_gates_decomposition` を呼ぶ。Rust :1083でDAGの1q runを集め、:1179–1241で行列積から再合成・置換・削除する。SOURCE上のeventやAccountingのroleを保存する処理ではない。[active Rust SOURCE](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs#L1076)
3. 同Rust :543–574のZSXXはmod 2π後の角が閾値以下ならRZを発行しない。ANGLE_ZERO_EPSILON=1e−12、`circuit_psx_gen` :277–346には0/π/π/2付近とXへの短縮分岐がある。Accountingの「加算結果が浮動小数点でちょうど0か」とは異なる。全supportの合成角が消去境界から離れること、又は境界に入っても必要なRZが残ることを保証する証明はない。
4. [InverseCancellation Rust](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/inverse_cancellation.rs#L42) は同wireのCX等のrunを削除する。**このpassがRZを直接削除するという主張ではない。** CX等が消えると、次のiterationで前は別runだった1q列が連結し、さらにRZが減り得る。1q runを一回数えた数や、局所の孤立RZをそのまま下側に固定できない。
5. [BasisTranslatorの探索](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/basis_search.rs#L96) は総basis gate数のcostを使う。既存の上側envelopeには有効だが、RZという一種類のgateの最低残存数を与えない。CX/SX/Xだけの存在からRZの正の下限は出ない。

例えば記号的に `RZ(t) X RZ(t)=X` なので、wire-local Z-foldではXで区切られた2個のRZを数えても、1q再合成はRZを残す義務を持たない。これは一般の推論「SのRZは消えない」を否定する代数的説明で、**m9のlegal pathを構成・観測した反例ではない**。

より直接な既存証拠としてP0 `442cafa1:docs/research/d_controlled_zz_p0_20261113/REPORT.md` §3は、同じ保存basis対・new-ZZ形の孤立小回路で **S=334、C=194、C−S=−140** を記録する。触れないwireの逆basis削除と1q再合成が理由として報告されている。これは単純なC≥Sや「new-ZZへ揃えれば残差は常に正」を採用できない根拠であり、**−140を全event/全pathの下限bとして使える根拠ではない**。今回P0を再実行せず、この値をqR倍していない。m0/m3/m6の正の観測残差、m4の記述的差、凍結モデル係数も同じ理由で一様bにしない。

## 5. 試した下界経路の到達点

| 経路 | 到達点 |
|---|---|
| C≥S、又は保存/モデル残差の最小をbにする | 不採用。Sと最終Cの関係に相殺が入り、有限観測は全supportを覆わない |
| C−S≥−U_Sを再利用 | 有効だが非負性を改善しない。既存下限の読取のみ |
| event/head別の最低RZを加算してhにする | 未証明。最適化の境界越え・wire別部分相殺・繰返しによる削除を差し引く不変量が必要 |
| 最終的に保護されるRZ又は1q runを数えるh | 未証明。全supportのDAG残存構造・合成角の閾値・fixed pointまで保存される対応が必要 |

固定m9のwordは有限長なので、有用な下界の存在や有限証明の可能性を否定したものではない。今回確認できないのは、その下側を全supportで保証するlemma/証明書付きobservableである。単に「Qiskitは複雑だから」とはせず、上記の削除・再合成・boundary・SOURCEのどこを閉じる必要があるかを残した。最適化を無効化したりbarrierを追加すれば別のCとなり、今回の固定価格関数を保たない。

## 6. PF比較とロードマップ

他軸が非負なので、正当なℓが得られた場合のみ `T_LD1 ≥ 25ℓ` を使う。現在の非自明ℓ・25ℓはnullである。

| PF基準 | 固定額 (RZ) | 厳密に超える下界が得られたとき | 等号の扱い | 今回 |
|---|---:|---|---|---|
| B0I p7 S2 | 593,555,352 | 25ℓ>基準ならB0Iが固定LD1より低費用 | LD1はB0Iを下回らないまで | 未判定 |
| Full PF S2 | 838,693,580 | 25ℓ>基準なら両PFが固定LD1より低費用 | LD1はFull PFを下回らないまで。実総額が等しいとは言わない | 未判定 |

下界が基準未満でもLD1が安いとは結論しない。PF値は採択済みcompiler閉包・固定shot配分の条件付き値を引用している。全law価格・全G・winner・PR-DF方式全体を新たに認定していない。今回の対象はIなのでPCG64 joint-law接続を今回の論証の障害として挙げない。F0がその接続や有限標本/モデル誤差を解決したともしない。

ロードマップ差分：LD1 m9-Xだけの安価なSOURCE下界確認を実施したが、PF比較を決着する非自明下界は未成立だった。次に必要な証拠は **この固定wordとopt1のfixed point全体に対する、相殺を織り込んだ下側observable又はpotential不変量の証明1件**。Sの閉状態kernelがあることだけではcompiler後の下側証明にならない。この調査を理由に新規pathやY/m12/LD2へ進まず、別判断を待つ。LD2価格整備の要否やPR-DF代表選択の問題も未解決のまま。ロードマップ本体は統合担当が更新する。

## 7. 検査・保全・STOP

固定locator/selector、対象ID/axis/q/R/N、law SOURCE blob、Accounting/FastPath/NewZZ blob、既存compiler Python hash、Rust release SOURCEの取得とhash、文書/JSONリンク、旧成果保全を作者自身で確認した。独立review又は新しい量子作用gateのPASSではない。Qiskit/科学moduleのimport、pass manager構築、path/回路生成、数値suite・保存H4演算子回帰は行っていない。静的な読取を並行実施しただけで、科学workerは0。

m4独立holdout、m3/P2較正、保存E_I[S]、凍結モデル、旧SOURCE/raw/SCOPE_DEVIATION/STOPは改変しない。論証調査1件、非自明証明0、下界・部分額の新規数値評価0。REPORT/HANDOFFと機械可読状態をcommit/pushし、remote SHA・公開file一致を確認した時点で **mandatory STOP**。
