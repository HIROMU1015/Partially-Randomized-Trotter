# PR-DF LD1：固定Q3候補と保存価格証拠の接続

2026-11-15 JST。**保存E[S]を26/26軸に接続。m0・m6の各X/YについてP1各2path、P2 fresh各4pathを有限path価格として受入、保留0。P2によるP1 replay各1pathは参照として分離した。全lawの適格E[C]は0/26軸のまま。** 新しい科学計算・実compileは0件。

[ADMISSION.json](ADMISSION.json)は既存ALIGNMENTへの差分索引、[COVERAGE.csv](COVERAGE.csv)は26行の一覧。[SOURCE_FACTS.json](SOURCE_FACTS.json)に実行済みSOURCEとscalar/preparation/profileを結び、[HANDOFF.md](HANDOFF.md)からB原Q3結果と元receiptをたどれる。[NEXT_VALIDATION_PLAN.md](NEXT_VALIDATION_PLAN.md)は未採択・未承認の追加検証1案。

## 重複確認と再利用

既存C branch `codex/m5-m4-receipt-admission-20261114`を継続。開始HEAD/公開refは`04a68f5932f53377595b2d69bbf8f46e02e82a01`、worktree clean。関連branch/refとtracked結果path名に同等のLD1-Q3価格受入を認めず、同Linuxユーザーの関連Python script名にも同じ受入jobを観測しなかった。確認範囲外の別名archive/別serverを網羅する主張ではない。他担当のworktree/process/cacheは変更していない。

既存`855573064…/PR_DF_VALIDATION_PLAN.md`、52軸coverage、`0b4b2d941…/ALIGNMENT.json`、law調査`d72dc4e6…`を再利用した。新規部分はBの完了Q3とのjoin、E[S]の旧canonicalから元/total candidateへのmetadata対応、P1/P2の元path・軸別receiptのconsumer受入。既存の構造値・cohort統計・残差・Q3を再計算していない。

## SOURCEと評価対象

- B結果carrier：`a82b0f0ec7f0f11d9981ddb2343fcd5378a87503`、`docs/research/pr_df_fullh_q3_fixed_results_20261115/{C_RESULT_JOIN,INPUTS}.json`、`run01/PR_DF_LD1/RESULT.json`。科学実行SOURCEは`27e70efe533862f38f4080aabb9cdd3ad1749fe9`。
- 元configuration：`a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`。元configのSOURCEはB INPUTSが示す`a9abf8de75bf3432fcd714e868e5c14c8884b8f8:docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json`。
- total decoder configuration：`9c3d392ec5c5ad373aaa1d96a8ba0828ddd268e6fef35da229029beec0229e8b`、binding `3c7a55ba69fcdbd7530bfe75167e85804e8e672fc4add768aee7422176f72c43`。価格の元IDは書換えない。
- E[S]：`9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f:docs/research/h4_structural_schedule_results_20261113/run01/COVERAGE.json#values`。旧configuration `587d20f7…`、各kernel SOURCEと数値区間を保持。
- P1/P2結果carrier：`39ddd59edf54a3a901e5302207b2765ddbd03fbb`。P1科学SOURCE `20fe11942369253361068dd2b00f98b26350ba8d`、P2科学SOURCE `ea0706ab05313423f336614912758a93e9e8285c`。

物理条件はLinear H4/1.00 Å/STO-3G/tol-only DF10、保存H/state/snapshot、S2、LD1（one-body＋保持DF head、残りtail）、fixed ΔPF=.1、finite K2、h=.1/M12、S0 exactκ。元fragment/basis順、signed piece [1]、各roundの固定q/R（m7以降255/510/1020/2040/4080/8160）を保持する。binding内T=.8は元入力の参照metadataで、実時間は各physical candidate/roundのTを照合した。

## ① E[S]：26/26軸を接続

各行の`structural.source/mean_selector/enclosure_selector/law_source/kernel_source`を参照する。元値のSOURCE・selectorを保存し、値そのものの新計算・再集計はない。

旧構造configurationと新signal/total configurationは文字列一致ではない。physical ID、round/axis、q/R/K/prefix/formula、T/Δのbinary64表現、signed allocation、H/state/snapshot、compiler/control/full-wrapper/P0、law definition・exactκ・fixed-grid mappingを展開して照合した。lawのconfiguration/canonical carrierだけを区別し、内容を維持した。total decoderは価格回路変更ではないというBの保存bindingも確認した。

quantityは`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`、期待値は`IDEAL_FINITE_K_SOURCE_LAW`。保存K0/K1のstatus・数値区間を保持し、E[C]へ改名しない。compiled価格がない22軸は欠測のまま。

## ② 有限path C/S：4 round×axisで受入

| round | physical ID | axis | P1元path | P2 fresh path | P1 replay参照 | 保留 |
|---|---|---|---:|---:|---:|---:|
| m0 q1/R2 | 53e00d8b07d1547d19a7 | X | 2 | 4 | 1 | 0 |
| m0 q1/R2 | 53e00d8b07d1547d19a7 | Y | 2 | 4 | 1 | 0 |
| m6 q64/R128 | 2a8c6718220eec59d419 | X | 2 | 4 | 1 | 0 |
| m6 q64/R128 | 2a8c6718220eec59d419 | Y | 2 | 4 | 1 | 0 |

pathはX/Yで共有する。P1は計4元trajectory・8軸receipt、P2 freshは計8元trajectory・16軸receipt。replay計2trajectory・4軸receiptは元P1 index0を再現した参照で、新しい独立標本へ数えない。P1/P2 fresh/replayをpool・再平均していない。cohortはそれぞれ`D_P1_EFFICIENT_CZZ_20261113`／`D_P2_DIRECT_PILOT_V1`、roleは元のDEVELOPMENT。

元private resultとcosine/sineの個別receiptを、固定公開結果が指す正確なpathから読み、C/S、candidate、cohort/index、event digest、seed selector、law/outer配分、axis fingerprint、compiler、compiled metricsを照合した。全receiptを元bytesのまま`receipts/<SHA-256>.json`へ保存（38個の異なる原JSON、replayとP1の同一軸receiptは同bytes）。locatorsに元path・hash・公開copy pathを併記する。同じRZ数だけを根拠にX/Yを共有していない。

## scalar・lowering・wrapperの照合範囲

元C1 Backend.context（`1a87e533…`）は保存input bytesとpreparation hashを検査し、**`ham.constant + prep.extracted_identity_coefficient`**をscalarにする。H4 completionの元PR preparation hashは物理closureの保存hash `b99720d6…`と一致。保存scalarは`2.7108187593826942`、binary64 `0x1.5afc1bee774aap+1`。この値はclosureからのmetadata転記で、今回cやs_Rの再計算・加算はない。

P1はそのctx.scalarをC1 structuralと元`compose`へ渡す。P2は同contextを継承した`compiled_price_fast`から同じctx.scalarを両方へ渡す。元`compose`はpiece内scalar=0で構成し、全outer後に`p(-scalar*T)`を一回適用する。Bの保存signalの定義もc＋prefix tail scalarを総Tで一回適用する。P1/P2の元resultにはresolved scalar数値を直記していないため、ここは**保存input/preparationと実行SOURCEの式・演算位置によるbinding**であり、原receiptに数値欄があったようには表現しない。

新ZZは`CONTROLLED_ZZ_CX_CRZ_CX_V1`。P1の保存tail parity、P2の新lowering有効化経路とreplay checksを再利用。既存C guardの`evidence` readerと`_rte_source`を使い、既存guardの変更・緩和はない。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional control、full measured Hadamard wrapper、P0を照合。元wrapperはcosine→X、sine→YでSdgの有無と測定を個別に定義する。

今回の受入は**SOURCE付き有限path証拠のconsumer接続**。平均価格recordを発行していないため、平均価格を前提とする`G.check`全体やBのcostは呼んでいない。P1/P2の`full_wrapper_statevector_validation=false`も保持し、新たな物理作用/全wrapper検証PASSを付与しない。旧C1のlowering/domainは異なり、新ZZの代替にしていない。

## ③ E[C]の証拠とQ3の扱い

全26軸で`E_C/law_distance/rare_contribution/total_CI/shared_covariance=null`。同pathのC/Sがあることと、全law期待価格の資格は別である。現時点で正式Gに使えるE[C]はない。受入済み有限pathは明示したfinite-cohort scenarioの原証拠になり得るが、今回その平均・G_paperは計算していない。

E[S]はideal law I。C−S標本はPCG64/CDF sampler Q。既存law調査d72dc4e6は同SOURCEの生成構造を確認したが、実H4 categorical/CDF距離・長trajectory joint-law・固定seed IIDは未評価。Iをtargetとして両者を足す場合のlaw systematicは`E_Q[C−S]−E_I[C−S]`。Qをtargetにすれば`E_I[S]−E_Q[S]`。距離を0とせず、rare未観測の寄与・長q/R移送・共有誤差も未知のまま。

BのLD1配分は`[64,60,56,52,47,43,38,34,30,25,21,17,12]`をX/Yに使った固定USER_SUPPLIED vector。**この固定配分で、保存raw・ideal IID・solver proxy・total decoder等に条件付くQ3閾値へ適合済み**と参照する。十分N/global RMSEを認定せず、価格との積算も0。LD2は別候補で、LD1の価格・E[S]・残差係数を移送しない。

## 実装・検証・次の一案

追加は`scripts/admit_pr_df_ld1_q3_prices.py`のmetadata reader/joinと人工testのみ。SOURCE/identity・scalar処理・law projection・axis receipt・replayの分離を確認した。人工新8＋既存guard回帰14＝**22 test method PASS**、作者自身の技術確認であり独立科学レビューではない。[TESTS.txt](TESTS.txt)。sampling/build/compile/fit/CI/Q3再評価/N/G=0。

最初の追加検証は**保存P2 fresh m0/m6を較正に再利用し、m3 q8/R16を追加較正、係数凍結後にm4 q16/R32で独立予測検証**する案を推奨する。最初の新規取得はm3 normal4path×X/Y＝8wrapper、後続holdoutも仮に4path×X/Y＝8wrapper。4pathは既存pilotと同規模の開発仮定で精度保証ではない。詳細・事前判断・STOPはNEXT_VALIDATION_PLANに記す。実行承認もPR用許容誤差も未採択。

ロードマップ差分：LD1の「E[S]とP1/P2のconsumer binding待ち」を本scopeで解消し、Bの完了Q3候補から原価格へたどれる。残課題は平均価格資格、law/rare/有限cohort誤差、較正・独立holdoutと長q/R移送。D/D1/D2/C2の進行は待機条件ではなく、ロードマップ本体は統合担当が更新する。

**mandatory STOP：保存価格の接続と計画公開で終了。新sampling・S/C計数・compile・平均/残差再解析・fit・CI・Q3再評価・N/G/winnerへ進まない。**
