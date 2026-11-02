# Track A 科学レビュー前の技術整理（2026-10-15）

**独立native B0+I prototypeとH8 D0 adapterを用意し、23人工検査に合格した。H8 D0用の保存JSON4件を受領し、既報SHAと一致した。実値の再会計・科学計算・compileは0。** 本資料は実装準備と証拠整理であり、baseline採用、実Hamiltonianのcorrectness、母平均優位を認定しない。

次の指定は[科学判断handoff](track_a_pre_review_scientific_decision_handoff_20261015.md)から行える。比較条件は[比較matrix](track_a_pre_review_comparison_matrix_20261015.md)、希少費用の未証明条件は[技術調査](track_a_pre_review_rare_cost_guarantee_20261015.md)に分けた。

## 1. 固定点と今回の境界

基礎D0 result `8dbdb68293c988a999645e21a82b53966f9ef6cd`。H8 result `681ec23c93fdd1090d325908f4b155a79115afdf`は別identityとして参照し、mergeしていない。追加設計 `2ccc79502f043b4885e2337ffa98ff45f1875ae4`、共通判断 `4675730e7e40d25ba3618334a4a0e28b4bd2f145`、H6 rare report `eee1343f6e99aa7661892ddfbe1067e974e2b74a`を読了した。指定5資料はGitHub固定rawとGit blobのbytes/SHA一致、originの4対象branch HEAD一致を確認した。

今回の独立prototype SOURCEは **`3c513d1ad52ce99f0d2df7d8a300e4d58d15a553`**。[source closure](../../artifacts/resource_applicability/track_a_pre_review_technical_closure_v1/2026-10-15/source_closure_v1.json)は新code/tests4件＋既存D0依存2件の6file。D0解析SOURCE `fa75c7997e135da7b7a0c4e23949e47252bd7a2e`の2fileはbyte不変。原科学registry/runner/source、旧結果/grant/ledger、private venvを変更していない。新しいgrantを発行していない。

全固定参照のcommit/blob/bytes/SHAと入力identityは[証拠索引](../../artifacts/resource_applicability/track_a_pre_review_technical_closure_v1/2026-10-15/fixed_reference_index_v1.json)。新規prototype SOURCEは将来の実行SOURCEとは区別する。production bridgeやbounded runnerを追加した際には新しいactual SOURCEで再固定する。

## 2. Native B0+I：確認済みの構造

H4規約は、fragment `lambda * (dGamma(G))²`のfull-Fock Pauli identityを

`C_fragment = lambda/4 * [(Tr G)² + Tr(G²)]`

としてdiscarded tailから加えるもの。固定粒子数sectorのtraceや、絶対値lambdaを使う規約とは異なる。signed lambda、generation order、Hermitian Gを保持する。H4実装のlower triangle/real diagonal（UPLO=L）算術と、新prototypeの算術はHermitian G上で対応する。新prototypeは全entryのfiniteと**exact Hermiticity**を確認し、非Hermitian/非finite/次元不一致を拒否する。実入力の許容差やscalar roundoff評価を今回新たに採択していない。

`_prepare_discard`はHamiltonianから先頭prefixのみを選び、tail空の`prepare_df_partial_s2`へ渡す。核・元constant・one-bodyは保持される。tail空なので`extracted_identity_coefficient=0`、deterministic buildの`p.constant_coefficient`とsignalのconstant+extractedの違いはこのB0経路で矛盾しない。retained DF block内部のscalar/global phaseはそのblockに既に含まれる。**retained identityをもう一度取り出して加えない**。

native deterministic builderはordinary/symmetric_directional両方で、最後に`P(-T*scalar)`をancillaへ一度加える。提案はこのscalar引数だけを`original constant + discarded C`へ置換する。既存Pに同じconstantを追加する方式、PF各stageへの重複追加、whole-circuitの無制御global phaseへ置く方式は採らない。Hadamard wrapperではcosineがRe z、Sdg→HのsineがIm z。block内部phase、outer q、負Yoshida時間、軸・control・orderingの既存意味は保持する。

[独立prototype](../../scripts/pre_review_native_b0i_prototype_v1.py)はB0のdeep copyにscalar overlayの提案を返す。元B0のblocks/global phases/constantは不変で、二重overlayを拒否する。SOURCE、H/DF/state/input/compiler、元numeric fingerprint、fragment orderをbindingする。native builder・Qiskit・科学input loaderをimportせず、registry/runnerへ接続していない。

## 3. 保存signalから導出できる条件と残る実装

同一H/DF/state/T/prefix/q/PFの保存B0 signal `z_B0`があれば、提案baselineのsignalは`z_I = exp(-i*T*C_tail) z_B0`。exact truncated referenceも同じphaseで回転できるが、**full-H referenceは回転しない**。biasはそのfull-H referenceとの差へ更新する。入力を同じ名前だけで同一視しない。採用するidentity規約とCの符号・算術・receiptが一致することが条件になる。

旧axis allowanceをそのままコピーすることは避ける。回転角thetaについて、旧Re/Im errorを`|cos theta| u_Re + |sin theta| u_Im`等へ伝播し、scalar phase誤差とrotation roundoffを別に加える純粋関数を用意した。旧uがEMPIRICALなら新uもEMPIRICAL。Cの誤差推定や実入力の数値guard合格は**未実施**である。

H6最小候補は設計通りp5/q1、p10/q2、p15/q8、T=.8/S2。3組の保存B0 corrected/exact-truncated/referenceを固定し、実入力からCの受領receiptを別実行で作る必要がある。analytic signal導出3件、両軸I新compile6件。matched B0の旧numeric/profile/source/phaseが不足する場合だけ**事前登録した別scopeで**B0最大6件、合計最大12件。旧B0や未取得Iの費用を0、又は同じ費用として補完しない。

新SCF/DF/state、trajectoryは不要。scalar extraction、signal/bias/u再会計、必要なcontrol probesとnative wrapper build/compileは将来の実行に含む。retained block・入力・scalar・numeric fingerprintを結合するproduction bridgeとbounded runnerはREQUIRES_IMPLEMENTATION。科学採択後に別SOURCEとして作る。H8は同構造を再利用できるが、rank28/p7,14,21/modes16、別H/state、CSR reference/uを再bindingする。H6 correctnessだけでH8の実測費用やΔmetric=0を認定しない。

H4 ordinary六距離の240軸で六metric差0だった結果は既報のまま参照する。native H6/H8の保証として使わない。

## 4. H8 D0：入力不足の解消とadapter

完了runの`signal_summary.json`、`exploration_resource_map.json`、`confirmation_resource_map.json`、`selection_before_confirmation.json`をbyte/SHA受領した。期待SHAはH8 completion auditの`result_file_SHA`。公開`wrapper_metrics.csv`と`completion_audit.json`を合わせたfuture必須6入力は**6,518,976 bytes**。原JSON、NPZ、runtimeを新commitへコピーせず、[入力索引](../../artifacts/resource_applicability/track_a_pre_review_technical_closure_v1/2026-10-15/h8_input_inventory_v1.json)だけを収録した。`candidate_status_by_precision.signal_precision`は説明stringであり、signal recordの代用品ではない。

H8 recordはH6同様`error_decomposition.total_signed`、EMPIRICAL `allowance`、`log_B`、旧4ε/factor1/10の完全`precision_rows`を保存している。H6 spectral参照とH8 independent occupation CSR/expm参照は数値経路が異なる。policyの名前が同じことからuの完全同一性を推定しない。

科学SOURCE c31ef424…、seed cb8a56c6…、input byte SHA d4c43461…、compiler1239304c…を区別する。CSVのsource列は3 execution SOURCE（cb8a56c6…/c6692964…/c31ef424…）を持ち、同一datasetでの合法resume lineageを表す。単一SOURCEへ書き換えない。confirmation registrationsのseed、expected_signal、union6、exploration8/confirmation32を照合した。[schema receipt](../../artifacts/resource_applicability/track_a_pre_review_technical_closure_v1/2026-10-15/h8_adapter_schema_receipt_v1.json)は**metadataのみPASS**。accounting/weighted_metric/analyze_scopeを禁止mockとし、実旧precision equalityを実行していない。

[独立adapter](../../scripts/pre_review_h8_d0_adapter_v1.py)はfrozen D02fileのSHAをimport前に固定し、92候補とunion6＋det20（26構成）を扱う。両scopeの全旧precisionと六metric mapの完全一致を先に要求し、合格後に.0005/metric/Pへ進む順序を人工検証した。確認detは探索n1をreuseし、B2の探索n8は確認n32へpoolしない。ordinary anchor4軸はprimary statistical costから分離する。

paired SE、RZ選抜の他metric値、metric別再選択、固定RZ pairと全候補P lower envelope、旧union外候補の表示は旧D0純粋APIで処理する。MISSING/INELIGIBLE/NUMERICALLY_UNRESOLVED/UNOBSERVEDを別に保持し、未観測layer平均をnullにする。実解析入口は閉じており、将来のbounded execution driver/contract bindingはREQUIRES_IMPLEMENTATION。H8 real old equalityも別承認後のPhase Aである。

## 5. 追加実行の静的予算案

[future contracts](../../artifacts/resource_applicability/track_a_pre_review_technical_closure_v1/2026-10-15/future_execution_drafts_v1.json)は**DRAFT_NOT_AUTHORIZATION**。approved=false、runtime_authorization=false、allowed_cpus=[]、absolute_launch_command=null。既存grant・carryを消費/改訂していない。

|案|最小・最大work|未実測量と提案STOP予算|
|---|---|---|
|H6 native I pilot|3 analytic signal、3 tail scalar receipt、I6 axes。matched B0は別登録最大6。新trajectory/SCF/DF/state0|実wall/RSS/費用delta UNKNOWN。worker1/thread1、AS32GiB/RSS9GiB等は既設計の候補上限。stage4h/output512MiB/log32MiB/journal64MiB/inode10,000は停止予算で必要量の予測ではない|
|H8 D0|input6/6,518,976B、Phase A old precision944 scope行/旧費用3,576行、Phase B precision1,180 slots/六metric最大7,080行、P120曲線/固定pair最大240|新compile/trajectory/signal0。actual wall/RAM/output UNKNOWN。single process/thread1、AS512MiB/RSS256MiB/CPU240s/wall300s/output32MiBは未承認STOP候補。FS free256MiB/20,000inode・quotaはfuture fresh gate|
|新rare layer pilot|最小correctness1 trajectory/2axesではSEなし。記述的pilot候補1層8 trajectories/16axes、精度保証の標本数ではない|H6既済q8/r2/K2 s2の8/16を再提案しない。新K4 s4又はH8 conditional adapterは別source/承認/資源見積りが必要|

H8 .0005の実適格数、point/SE、P最良、union外の新候補は**UNKNOWN/NOT_RUN**。入力metadataから候補slot上限は数えたが、実値の数値予測はしていない。H4/H6 D0の既報wall2.223s/peak54.14MiBは参考実測で、H8予測へ外挿しない。storageは入力copy、output temporary/final二重、log/journal/temp、terminal reserve、filesystem block丸め/inode/quotaを含む。carryは旧成功/失敗/reserveを保持し、未知historical actualを0で埋めない。

## 6. 検証と公開境界

native11＋H8 adapter12、計**23 distinct人工検査PASS**。初回22件中1件はfixtureがunion外のmetric最小を作っておらずFAIL、その証跡を保持した。CXだけ異なる人工候補を加えたattempt02で22件PASS。最終の次元guard追加はnative11だけをattempt03でPASSし、H8の合格campaignは反復しなかった。

主な独立oracleはtiny3-mode full-Fock `Tr(dGamma(G)²)/2^n`、直接ancilla amplitude/Sdg/H、手計算paired totalのsample SE。残りは二重scalar、signed/ordering/Hermitian/nonfinite、旧scope両方のexact gate先行、map欠落/paired欠落、input/source/selection/seed、union外表示、real解析禁止を検査した。Qiskit/NumPy/CuPy/native科学module importは0。単一process/thread1、AS512MiB/CPU60s/wall120s/output1MiB以内。attempt02 wall3.296s/peak68,423,680B。これはlocal人工検査でありCI・外部再現ではない。

今回の作業は全てhome配下。新科学計算・実データ再会計・compile0、既存科学SOURCE/worktree/grant変更0。新worktreeと今回の新branchに限って軽量code/tests/docs/receiptをcommitする。publish状態はbundleのpublication receiptで記録し、認証障害なら設定を変更せず手動pushを案内する。科学判断・新計算へ自動進行せずmandatory STOP。
