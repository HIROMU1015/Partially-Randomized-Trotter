# 研究C：basis-aware saved64・修正後148-callの結果

2026-10-29 JST。保存直鎖H4・1.00 Å・STO-3G、通常number CDF rank8、元順L_D=2、partial ordered S2、δ=h=0.1／s=1／r=128／K=0。M4 Proposedとtail-P4の保存各32 trajectoryを独立cohortとして扱った。Own-DF ground-state oracle、状態準備費用除外。今回の対象は保存native価格の記述的予測で、PF/RTE・shots・総資源の新計算ではない。

**Fresh run02は148/148 logical units完了、correctness FAIL 0。主RZ判定は両cohortとも`SURROGATE_NOT_QUALIFIED`となった。** 複合モデルCはM4ではA、tail-P4ではBよりheld-out MAEが大きく、両cohortでmean/A/Bのすべてを改善する登録条件を満たさない。新DF fit GO=false、optimizer目的関数・η採択=false。これは今回固定した3特徴量モデルの不適格であり、basis-aware DF全体の否定ではない。

## 修正と固定SOURCE

[旧run01](research_c_df_basis_surrogate_execution_20261029.md)は3 attempts＝2 COMPLETE＋1 key FAILで停止した。元の`np.kron`経路と保存key生産側のspin liftでは、数値が同じでもsigned-zeroのreprが異なった。[修正準備報告](research_c_df_basis_surrogate_repair_preparation_20261029.md)に従い、新v2 geometryで既存general-Kの`_matrix`／`_spin_lift`をそのまま使う。保存key・U・frame順は変更せず、近似key照合やfallbackは導入していない。旧99 SOURCE・旧契約・run01の原本/STOPを保全した。

- 科学SOURCE commit：`0301286493ebf2b95a1786c2bd78756015c1015d`（実行前にpush・remote照合）。
- Branch：`codex/research-c-basis-native-saved64-repair-20261029`。
- [修正契約](research_c_df_basis_surrogate_contract_repair_v2_20261029.json) SHA256：`365d52a7df356a4162dbfd65e9810b7820b5418ac3dcb2b83688c789ba3a81d3`。
- [SOURCE103 manifest](../../artifacts/research_c_df_basis_surrogate/2026-10-29/technical_repair_v2/SOURCE.json) SHA256：`dc3890d10fdda3c378fd8d35c386a5a709b81634d0988e3c5c8c1d056204bce1`。
- 実行入口：[v2 runner](../../scripts/run_research_c_df_basis_surrogate_v2.py)。既存engine/trajectory/statisticsを継承し、task process内のgeometry/scope/fresh pathだけをv2へ接続する。旧SOURCEは上書きしていない。
- 人工新7＋回帰43＝**50 PASS（2.09秒）**。[ログ・verification](../../artifacts/research_c_df_basis_surrogate/2026-10-29/technical_repair_v2/)。Reflection、det±1、π、複数frame、signed zero、真のkey不一致、binding復元、旧統計関数の同一性を確認。
- [Preflight](../../artifacts/research_c_df_basis_surrogate/2026-10-29/execution_preflight_v2.json)／[postflight](../../artifacts/research_c_df_basis_surrogate/2026-10-29/execution_postflight_v2.json)：SOURCE103・保存入力275・旧保護1570件のhash一致。旧SOURCE99、旧run01、元科学入力267件は不変。

ユーザーのチャット許可に基づく技術修正後の一度のfresh実行である。旧契約の歴史的metadataは保全し、承認証明・新grantは作成していない。

## 事前登録したモデルと評価

各cohort内だけで、切片を持つmean、A（basis_switches）、B（overlap_path）、C（basis_switches／ZZ_count／same_basis_support_pairs）を比較した。`basis_key`は保存生産側の厳密keyを使い、frame IDだけの共有を仮定しない。固定4-foldは`sample_index % 4`、各24 train／8 test、NumPy OLS・lstsq rcond=1e-12。係数は各train内で推定し、最適化の重みには転用しない。候補追加、特徴選択、ハイパーパラメータ探索、pooling、paired比較は行っていない。

主指標はheld-out **RZ MAE**。Cがmean/A/Bすべてより厳密に小さいことを両cohortで要求する。加えて全rankがfull、Cの負予測0を要求する。

| Cohort | Model | RZ MAE | RZ RMSE |
|---|---|---:|---:|
| M4 Proposed | mean | 1248.5547 | 1539.0269 |
| M4 Proposed | A switch | **221.4073** | 274.7401 |
| M4 Proposed | B overlap | 305.6085 | 377.7858 |
| M4 Proposed | C composite | 227.7980 | 273.1524 |
| tail-P4 | mean | 1082.2266 | 1433.1541 |
| tail-P4 | A switch | 287.4242 | 348.6428 |
| tail-P4 | B overlap | **261.5380** | 345.6792 |
| tail-P4 | C composite | 273.5792 | 339.1421 |

CのRZ RMSEは両cohortで最小だが、副指標を事後的に主判定へ置き換えない。全48 cohort/metric/model、192 foldでrankはmean=1／A=B=2／C=4のfull rank、負予測は0件。Rank不足、constant correlationによるUNRESOLVEDはなかった。

全6指標のheld-out MAE（精密値・RMSE・fold係数は保存CSV/JSON参照）：

| Cohort | Metric | mean | A | B | C |
|---|---|---:|---:|---:|---:|
| M4 Proposed | RZ count | 1248.55 | 221.41 | 305.61 | 227.80 |
| M4 Proposed | RZ depth | 336.78 | 23.76 | 61.07 | 20.93 |
| M4 Proposed | CX count | 382.99 | 65.96 | 94.76 | 63.71 |
| M4 Proposed | CX depth | 175.75 | 15.02 | 30.48 | 11.35 |
| M4 Proposed | total depth | 730.61 | 51.85 | 131.49 | 43.29 |
| M4 Proposed | circuit size | 2515.82 | 428.17 | 607.75 | 435.11 |
| tail-P4 | RZ count | 1082.23 | 287.42 | 261.54 | 273.58 |
| tail-P4 | RZ depth | 327.57 | 32.29 | 64.43 | 32.93 |
| tail-P4 | CX count | 337.23 | 85.90 | 79.56 | 80.80 |
| tail-P4 | CX depth | 168.88 | 21.30 | 30.85 | 18.29 |
| tail-P4 | total depth | 714.80 | 67.25 | 134.38 | 68.38 |
| tail-P4 | circuit size | 2197.13 | 558.33 | 512.26 | 534.13 |

登録predictorとRZ countのcohort内記述相関：

| Cohort | Predictor | Pearson | Spearman |
|---|---|---:|---:|
| M4 Proposed | basis switches | 0.9844 | 0.9851 |
| M4 Proposed | overlap path | 0.9682 | 0.9666 |
| M4 Proposed | ZZ count | 0.4512 | 0.4978 |
| M4 Proposed | same-basis support pairs | -0.2511 | -0.2436 |
| tail-P4 | basis switches | 0.9733 | 0.9696 |
| tail-P4 | overlap path | 0.9718 | 0.9501 |
| tail-P4 | ZZ count | 0.3524 | 0.3206 |
| tail-P4 | same-basis support pairs | -0.0364 | 0.0901 |

相関はp値・有意差・因果効果を認定しない。有限保存cohort内の価格変動には切替/overlapと強い記述的関係が観測されたが、新DFへの予測性能や母平均の費用改善は未検証である。

## 静的DF差と保存raw census

静的DFの標本は**2個**。各32 event列から64個の独立DF観測が得られたとは扱わない。

| 保存law descriptor | M4 Proposed | tail-P4 |
|---|---:|---:|
| λR / Ha | 1.442430 | 1.516166 |
| P_same | 0.540242 | 0.486313 |
| expected switches | 58.389326 | 65.238272 |
| expected runs | 59.389326 | 66.238272 |
| expected Z events | 48.245626 | 82.100260 |
| expected ZZ events | 79.754374 | 45.899740 |
| raw central RZ | 256 | 256 |
| expected raw central CX | 415.508747 | 347.799479 |
| expected overlap path | 121.856666 | 135.090445 |
| head F0/F1 overlap | 2.043459 | 2.039616 |

q_frame/q_basis・run分布は[static2 CSV](../../artifacts/research_c_df_basis_surrogate/2026-10-29/presentation_v2/static2.csv)と各cohortのstatic.jsonに保存した。Tail-P4はこのlaw descriptor上でZZが少なく切替が多いが、この2点から平均価格差を因果分解できない。Head・静的basis・係数分布の効果はcohort別interceptにも吸収される。Head価格／isolated basis価格／basis-pair価格はnullのままである。

保存raw QPY64件はdeserializeしてopcode数だけ集計した。新しいdecomposition/build/compileはない。[64行CSV](../../artifacts/research_c_df_basis_surrogate/2026-10-29/presentation_v2/features64.csv)とsample00～31.jsonに、元seed・順序保持特徴・保存P0全6metrics・raw opcode countsを保存した。Raw censusをhead/basis単独の合成価格と同一視せず、学習predictorにも追加していない。

## 科学call・資源・証拠

| Kind | COMPLETE |
|---|---:|
| saved bundle | 2 |
| identity gate | 2 |
| static features | 2 |
| trajectory features | 64 |
| saved raw census | 64 |
| descriptive models（2 cohort×6 metrics） | 12 |
| surrogate qualification | 1 |
| result summary | 1 |
| **Total** | **148** |

Run02 retry/resume=0、技術relaunch=1。旧run01の3 attemptsを含む履歴累計は151 attemptsで、旧残枠の再利用やresumeはない。新DF fit／sampling／build／compile／PF-RTE／shots／総資源再計算／詳細科学レビューは0。

P0は保存済みQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／qubits_initially_zero=false、MPFR160位相条件を維持した。入力と結果はread-onlyで再利用し、task-local venv/cache以外の共有環境を変更していない。

実行前32 affinity CPU、空きRAM約52 GiB、load 1.09/1.10/1.09を確認。今回の支配処理は64×128 event走査、QPY I/O、小規模4-fold OLSであり、schema順序を維持する既存逐次engineを使用した。CPU/RAM/worker/wallの人工上限は追加していない。GNU time実測wall **5.04秒**、user10.25秒、system2.97秒、max RSS393424 KiB、swap0、exit0。Runner内wall4.696741秒／CPU13.004448秒。速度向上倍率は主張しない。

- [raw76件・result・SOURCE・contract・call journal・STOP](../../artifacts/research_c_df_basis_surrogate/2026-10-29/run02/)。
- [モデル48行](../../artifacts/research_c_df_basis_surrogate/2026-10-29/presentation_v2/models48.csv)、[fold192行・係数](../../artifacts/research_c_df_basis_surrogate/2026-10-29/presentation_v2/folds192.csv)、[相関48行](../../artifacts/research_c_df_basis_surrogate/2026-10-29/presentation_v2/correlations48.csv)。CSVは登録結果の表示用転記だけで、再fit/追加相関計算ではない。
- [Preflight](../../artifacts/research_c_df_basis_surrogate/2026-10-29/execution_preflight_v2.json)、[postflight・全raw hash](../../artifacts/research_c_df_basis_surrogate/2026-10-29/execution_postflight_v2.json)、[process time](../../artifacts/research_c_df_basis_surrogate/2026-10-29/process_time_v2.txt)。

## 未資格事項・次の担当・STOP

今回の登録Cモデルを新目的やηへ採択しない。異なる静的DFへの汎化、head/basisの因果分解、全law費用support、actual CA、保証shots、総資源優位は未資格で、未知量を0で補完していない。回帰係数をDF最適化の重みに置き換えない。

次の判断は**ユーザー／研究C**が、切替/overlapの単純モデルと今回の負結果を踏まえて独立DF検証やbasis-run機構検証を行うか決める。**Codex**は具体的依頼を受けた範囲で最小の技術設計を準備し、新数値再解析・新fit・新compileは別の明示許可後だけ行う。今回の負結果を避ける事後的な特徴追加や条件変更はしない。

結果・SOURCE・旧新原本をcommit/pushしremote SHAと成果物を照合後、**mandatory STOP**。追加科学計算へ自動進行しない。
