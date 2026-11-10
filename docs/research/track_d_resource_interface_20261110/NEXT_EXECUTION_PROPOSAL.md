# 次の単一PR設定：同じCDF8のr=16（未実行）

[REPORT](REPORT.md)／[interface](INTERFACE.md)。**推奨はr32基準に対しrだけを16へ変更する限定検証。今回の指示では実行しない。** α最適化、新A Q、Closed P3、別のReference算法を混ぜない。

## 1. 選定理由と固定条件

PR本体のevent負担とattenuation/十分Nのtradeoffを調べる、最小の一変数変更である。Hamiltonian、DF、LD、PF、K、時間、state、測定を維持でき、Reference/controlの効果を同じPR law内で分離しやすい。rを小さくすると単価が軽くなる可能性がある一方、RTE誤差や減衰が悪化して総費用は増え得る。改善は予測・認定していない。

固定はLinear H4／1.00Å／STO-3G／Nα=Nβ2、同じ元H packetとnumber-CDF8 endpoint、元frame順LD2、ordered partial S2/s1、K0、h=δ=.1、元H厳密ground仮定・準備費除外、X/Y、C1 OFF、同P0・位相規約。13round、ε=.00159 Ha／failure=.01／βtotalπ/6／βstatπ/12／一様α=.01/26／同prior・decoderを維持する。

元H packetは `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF endpointは `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。**r32のlaw hash、N、A/F/Sはr16のものではない。** K0のlabel確率に共通部分があっても、角度・step数・whole-path lawを新rにbindする。

## 2. 再利用可能なSOURCEと範囲

| 資産 | 再利用する部分／制約 |
|---|---|
| B1 d3913866・SOURCE8c6eeb02 | 元H operator誤差の会計、保存DF norm＋guard、同s/hのPF入力。r非依存性をSOURCEで確認した項だけ再利用し、旧RTE/attenuationを流用しない |
| B4 07d18bfe・track_d_original_h_accuracy | 元H信号のradius/phase/coordinate/十分Nとdecoder。新r16 evidence・SOURCEで新Nを算定する。旧結果を編集しない |
| rte.make_rte_config／finite_rte_distribution／attenuation、df_partial_s2 | rte_stepsを明示する汎用構成。新r16 lawと角度・normalizationを扱う。今回呼出し0 |
| CDF science SOURCE4df8eca29c59ffd55e2dd898f05da16bb246ddbe | 保存number-CDF因子・signed係数、c_pr_aware_df_stage_b_runner等の既存configure/native処理。旧runnerの固定s4/r16やr32 contractをそのまま新s1/r16結果に読み替えない |
| Stage A/C1/C2/C5 adapter・StrongBuilder・P0/postcheck | whole-path AB=V、actual adjoint、χ、vacuum、X/Y observable、強対照とReferenceのfull-wrapper。新rに必要な最小metadata拡張を新SOURCEで固定 |
| B6b 7ffe8b28 | ideal測定同値性の一般条件。whole Vのvacuum eigenray、actual A†、scalar/χ、同X/Y・同lawを新設定で検査 |
| 今回のresource interface | 科学inputsとcontrol価格の分離、未資格返却、上流evidenceから旧総集計への接続。SavedCDF8Studyはr16を拒否したまま残す |

既存研究にr16を扱うコード・別条件の保存結果はあるが、同じCDF8/s1/r16・元H ground・Strong/Referenceの全13round資格と価格が既にあるとは確認していない。own-DFや別endpointのr16 N/価格を移さない。

## 3. 最初に許可を求める最小科学実行案

### S1：新lawの元H誤差・十分N

新SOURCEで固定CDF8のr16法則・normalization/attenuationを構成する。保存DF/PF入力の再利用条件を確認し、既存operator誤差pipelineで有限RTEと元H信号の十分条件を全13roundへ接続する。ground solve／DF fit／新state準備は不要。B4方式でradius、phase余裕、alias/decoder、十分Nを検査する。

既存r32の元H ground仮定から半径1を仮定しない。DFは元H信号誤差へ一回だけ入れる。r16で正半径又はphase/energy条件が成立しなければ、結果と理由を保全してSTOPし、r/K/hを無断で変えない。この段階で否定的結果なら新回路価格取得は不要である。

### S2：PASS時のみ短い全wrapperのpaired gate

S1のPASS後、**q4、complete step2後cut、一つの新r16 ordered path**を結果非依存のseed/sample IDで実行前固定する。必要な新samplingはこの一pathだけとし、原本を新namespaceへ保存する。r32のsampleを途中で切ってr16保存原本と呼ばない。

同じr16 pathを両方式に使い、既存のraw harnessでAB=V、actual adjoint、vacuum ray、χ、scalar/identity、signed係数、X/Y測定演算子を比較する。raw PASSかつ強symmetricのbasis共有・relative/tail共有・step融合が公平に維持される場合だけ、Strong/Reference×X/Yの**最大4 P0 wrapper compile/postcheck**を行う。Referenceの全path一gadgetとcut固定、追加register/CSWAP/χ/readoutを含む。既存precisionと閾値は緩和しない。

r32の保存q4/C1 OFF回路価格は背景参照として再利用できるが、別path・別rとの価格差をpaired cohort推定にしない。pairedは新r16の同じordered path内のStrong対Referenceを意味する。

S2の指標はRZ/CX count、各depth、total depth、size、peak qubitsを別々に記録する。主資源や重み付き目的関数は今回採択しない。一pathの価格は全sampling lawの期待価格でも、全RPE費用保証でもない。

## 4. 全総資源評価に残る追加証拠

S1＋S2だけでは、r16の13round総費用を評価できない。r16用の期待full-wrapper価格が各roundに必要である。旧r32 A/F/Sは使えない。直接価格・新設定の校正／独立検証・適用域を明示した条件付きモデル、又は正当な構造的価格境界を別途取得する必要がある。

そのため、最小の**次回実行契約はS1＋条件付きS2まで**とし、総費用取得の標本数・モデルfamily・価格幅・long-q範囲はS2結果と研究上必要な精度に基づいてユーザー／GPTへ戻す。今回はそれらの新fit/校正や13round集計を許可した計画にはしない。大量長q compileを自動必須にしない。

価格coverageが揃った別の許可範囲では、新r16の十分Nと価格を同じScientificInputsへbindし、両方式の総6資源と幅を既存集計へ接続する。共通Nは新law内のideal測定同値性と同じ誤差条件に基づき確認し、actualは別資格。実行前に価格の母平均/CI/外挿のどの主張を採用するかを固定する。

## 5. 承認・失敗時・担当

必要な次回の許可は、新r16 SOURCE/lawの構成・元H operator誤差/N評価、PASS時一pathのsampling/raw correctnessと最大4 compile/postcheck、保全・公開まで。SOURCE/input/seed/cut/compiler/精度/上限を実行前固定する。既存r32のgrant・ledger・marker・STOPを再利用・解除せず、新namespaceで扱う。

SOURCE/入力不一致、correctness不一致、全round十分条件不成立、公平な強対照を構成できない場合は依存実行STOP。失敗後に別r・別path・cut・閾値へ変更しない。技術修正を許可する場合のretry条件も次の契約で明示する。S2後は価格校正や総費用へ自動進行せずmandatory STOPする。

**ユーザー／GPT**：このr16限定案の科学実行範囲と主張水準を判断する。**Codex**：現在は実行せず、明示許可後のみ固定範囲を行う。新A／Closed P3／他Track統合・新Reference算法・詳細科学レビューは含めない。
