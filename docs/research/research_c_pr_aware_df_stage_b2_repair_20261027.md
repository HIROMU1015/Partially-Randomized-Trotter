# PR-aware DF B2：Qiskit共有equivalence libraryの技術修正と再実行固定

2026-10-27 JST。ユーザーがB2実行と、意図しない技術中断の修正・再実行を明示許可した範囲で対応した。科学的FAILの再試行を許可と解釈していない。

## run01を保全

元科学SOURCE e74226a817e65918c0e00a735498e5bada77146d／実装4d7d9dbd3…・652-call契約は変更していない。run01は429 calls開始、428 COMPLETE、kernel_build `proposed_17`でRuntimeError `Already borrowed`が1件。数値correctness FAILは0であり、部分18 kernel metricsからarm比較・母平均推定を作らなかった。全workerを既存STOPで回収し、run01 retry/resume0を維持した。

[run01 failure/STOP/raw](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run01/)と[postflight](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run01_postflight.json)はcommit abc06e759で保存した。439 raw files、QPY165、reference49、wall385.928s、peak RSS2575364KiB。SOURCE47/input22は不変。run01のSTOP.failed=trueは技術例外を含む実行状態で、科学的不適格や数値同値性FAILを意味しない。

## 原因と最小修正

固定Qiskit1.3.0の `circuit/add_control.py` はSessionEquivalenceLibraryを`sel`として使い、RZZGate.controlからUnrollCustomDefinitions／BasisTranslatorを実行する。`ControlledGate.inverse`も`base_gate.inverse().control(...)`を呼ぶ。従来のP0 compile専用mutexとarm別build lockでは、build・compile・actual-adjoint constructionが同じRust-backed equivalence libraryへ同時アクセスし得る。Tracebackはこの経路のBasisTranslator.base_runで停止している。

新しいtask-local `c_pr_aware_df_stage_b2_compat.py`／専用entryを追加した。旧B2 executorには手を入れず、kernel buildとP0 compileのcallbackを同一RLockで実行する。Actual-adjointの**回路構築だけ**も同じlockで保護し、処理終了後にprocess-local wrapperをrestoreする。CAR reference、Operator decode、行列照合・unitarity・X/Yは並列のまま。Package・共有venv・共有cache・OS設定は変更しない。CPU／RAM／worker／wallへの人工上限ではなく、共有内部libraryの排他条件である。

共通lock取得後に既存CallsがSTOPを確認してからbuild/compileを計上する。科学call上限、実際のgate・目的operator・P0/MPFR160・allowanceは変更していない。

## 人工検証と新SOURCE

共有libraryを使う3種類の操作の同時実行を禁止する人工stress、実人工RZZ.control／P0／inverseの行列一致、STOP後lock取得、人工complete pipelineと元seed SOURCEの保持が4 PASS。既存B2のfull512等16 testsと合わせ**20 PASS**（34.08s）。旧strong-baseline/feasibility等56 PASSも再利用する。

- 新数値実装：`3e1e8d1446de04b9ebec5c0bcf455e7e7a47d17a`
- [新SOURCE50件](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/compat_technical/SOURCE.json) SHA256：`e6facbf59838f93dcdd61258c0589da7b3dadb2116a041209d7a08fb8f471d28`
- [技術再実行契約](research_c_pr_aware_df_stage_b2_compat_contract_20261027.json) SHA256：`23763d603c242f59001239e132c604f3f4f3b3cb7c71efc520bd30a65e09f7ce`

[test log](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/compat_technical/tests.log)／[passive preflight](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/compat_technical/passive_preflight.json)。元SOURCE47を含む新SOURCE50、保存input22、run01のSTOP／failure／journal等をbyte照合した。

## 同じ科学条件・64標本をfresh再実行

保存H4・1Å/STO-3G、sector36、通常CDF8両arm、元順LD2、partial symmetric S2 δ.1/s4/h1/40/r16/K0、own-DF oracle／状態準備除外、共通precision/P0・全512検証を維持する。

修正SOURCE identityとsampling用SOURCEを分離した。新executorは元の固定科学契約を既存base.executeへ渡すので、元sampling digest `cdf79ed569127f5cb6c7ee017b76bce6422166a656bb3ae866de6e86917c45bf`、master/cohort/arm/final-JU-hash/indexを保持し、最初から同じ64標本を再実行する。run01の成功prefixの結果・cacheを再利用せず、標本探索・選択・拡張もしない。

Fresh outputは`artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run02`。今回明示許可の技術的再起動1回、run02内retry0／resume0、最大652 units。旧429と合わせ累積最大1081 unitsを明記する。旧retry/resume0契約を無断で書き換えたのではなく、今回のユーザーが許可した予期しない技術中断の修正・再実行だけを新SOURCE／fresh outputへ固定した。追加の自動retry、科学的FAILの回避、別cohort・条件・rankへの変更は許可していない。

全正しさgate後の有限bit費用校正のみで、shots／CA／総資源・support／全law合成上界は未資格、nullを保持する。run02終了後は両runの証拠・科学call・新SOURCEを公開照合しmandatory STOPする。
