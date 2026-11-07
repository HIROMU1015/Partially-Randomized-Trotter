# 新A N4-C：全support CT誤差・保存199-shot条件

2026-11-08 JST、SERVER 2。固定基準 **112d2f7eeaa6a0adc8bea2c14347900266d3c7ea**、branch `codex/pr-reference-n4c-accounting-20261108`。

**固定m7/q128/r1・B′／Reference×X/Yの4条件すべてで、shell位相を含めても保存199 shots/axisの十分条件が成立した。** 判定は **SAVED_199_SHOTS_SUFFICIENT**。必要連続shot数の最大保証上界は198.683571850558719、整数上界は全件199。指定再会計・検算を完了し、**mandatory STOP**。

これは固定SOURCE、compiler条件C、role閉包、共通word policy、C_phi/runtime、保存raw信号モデルを前提とする**m7全supportの条件付き十分条件**である。有限2 trajectoryの観測を全supportへ外挿した結果ではない。条件Cを独立に証明したものでも、全18round CT成功・総T／T-depth／FT優越の資格でもない。

## 固定条件・実行範囲

Linear H4、1.00 Å、STO-3G、DF12、元H厳密ground供給・状態準備費除外。Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1・保存73 bits。Qiskit1.3.0/compiler A、native rz/sx/x/cx、全結合を維持した。

B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar/event・X/Yは不変更。最終native RZ strict Frobenius10^-11、b36／j=-36…6／43基本word／内部10^-13、C0全299 word、既存emitter／inplace consumerをそのまま用いる。保存199/axis、alpha1/720、phase cap.4、failure.05、energy1.5936e-4 Haも維持した。

実行は保存入力と既存SOURCEによる4条件の誤差・shot再会計だけ。新規合成、word取得、strict guard、provider実行、native build/compile、Hamiltonian生成、ground solve、simulation、sampling、trajectory列挙、価格計算、shot／precision／alpha最適化は行っていない。199-shot積のT会計、他round再解析、全18round総Tも0。

## 1. 全supportのshell上界

[N4-B PHASE_ACCOUNTING](../pr_reference_n4b_consumer_20261108/PHASE_ACCOUNTING.md)を変更せず、追加shellを

`epsilon_shell < (16+64S)2^-53`

として独立に扱った。全supportで **S<=1** を使える根拠は保存8 wholeの観測だけではない。

- [wrapper](../../../src/trottertracks/pr_reference_interference_pilot.py)はH、core、任意のY用Sdg、H、最後のmeasureからなり、body内にreset／barrierを入れない。
- 同SOURCEのreference_core／symmetric_fused、[B′ build_saved／rewrite_rte](../../../src/trottertracks/pr_reference_bprime.py)、[transfer.assemble](../../../src/trottertracks/pr_reference_transfer_qualification.py)はunitary bodyを組む。固定DF RTE／partial-S2 builderにもreset／measure／barrier呼出しがなく、SOURCEの有限gate族は条件C下のunitary templateへ閉じる。
- [CT-4 prepare_one](../../../src/trottertracks/pr_reference_ct4.py)はReferenceだけに先頭8 resetを付け、その後にcompile済みwrapperをcomposeする。B′にはこのprefix resetがない。固定compiler Aのpassと条件C下の同値則は新しいbody fenceを生成しない。
- 従って非empty unitary segmentは高々1。仮に空でもこの上界は安全であり、有限trajectoryで非emptyだったことを一般証明の代用にしない。

初期outer phase設定1回とsegment→outer join高々1回から、両arm・両軸で同じ保守的spectral上界を用いた。

`epsilon_shell <= 80/2^53 = 5/562949953421312`

これは約8.8817841970012523e-15。Raw diskへの寄与の外向きbinary64上界は **2.5121479338940402e-14**。局所1-qubit scalarのFrobenius上界をspectralへ移したもので、巨大joint Frobeniusへdimension係数なしに適用していない。

SOURCE関数hashと上界の帰属は[SHELL_ERROR_ACCOUNTING](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/SHELL_ERROR_ACCOUNTING.json)に保存した。新しいQPY／copy／to_instruction経路は追加していない。

## 2. Joint synthesis誤差とdisk

CT-0の `count_assets / rzcount / worst_counts` をそのまま再利用し、固定保存loweringからm7のrole上界を対応させた。

| Arm | D | Frame | Rotation | Product | Event phase | Scalar/wrapper X/Y | CSWAP | RZ合計 X/Y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B′ | 791704 | 393216 | 256 | 2048 | 128 | 5 / 6 | 0 | 1187357 / 1187358 |
| Reference | 790848 | 393216 | 128 | 512 | 0 | 5 / 6 | 176 | 1184885 / 1184886 |

Actual SOURCE inverseの保存loweringも旧CT-0と同じに扱う。V3期待多重度、有限trajectoryのgate数、signed D融合価格補正は誤差数へ使っていない。両armのframe393216個をそれぞれ数え、価格取消しを誤差取消しへ流用しない。

全位置の既存certificateがstrict Frobenius10^-11未満なので、spectral embedding／unitary telescopingで

`epsilon_RZ <= N_RZ_max * 10^-11`

`epsilon_syn <= epsilon_RZ + epsilon_shell`

を用いた。各RZの量子化・基本guard・局所emitterは10^-11へ既に含まれ、再加算していない。Shellはそれらと別の実操作として1回だけ加えた。[JOINT_SYNTHESIS_ERROR](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/JOINT_SYNTHESIS_ERROR.json)に正確なFractionを保存した。

複素信号はX/Y両座標からなるので、軸別joint上界をそのまま片方だけで両座標へ適用しない。既存CT-1 error_and_shotsの**両軸worst-case数のmaxを取る代入文**をそのまま再利用し、各armで `epsilon_pair=max(epsilon_syn_X,epsilon_syn_Y)` を選ぶ。これは新しい推定器や誤差規約ではなく、同じ4条件を既存の一様複素diskで安全に閉じる保守的接続である。

既存のraw信号規約のまま、`d_syn <= 2sqrt(2)*epsilon_pair`、`d_total = d_old + d_syn`を192-bit外向き区間で評価した。各armではY側上界が大きく、X/Yに同じRPE disk・shot十分条件を使う。以下のjoint列は各軸固有の上界、raw／RPE列はその両軸maxを用いた値。上限小数は区間上端から上方向へ丸めた表示で、判定は元の有理端点による。

| 条件 | Joint epsilon_syn上界 | Raw d_syn上界 | 合算d_total上界 | 必要連続N上界 | 保存N |
|---|---:|---:|---:|---:|---|
| B′ X | 1.18735700088817842e-5 | 3.35835557649653472e-5 | 3.38670714792282589e-5 | 198.683571850558719 | 199 / PASS |
| B′ Y | 1.18735800088817842e-5 | 3.35835557649653472e-5 | 3.38670714792282589e-5 | 198.683571850558719 | 199 / PASS |
| Reference X | 1.18488500088817842e-5 | 3.35136370464416214e-5 | 3.37974110992686846e-5 | 198.683477052163953 | 199 / PASS |
| Reference Y | 1.18488600088817842e-5 | 3.35136370464416214e-5 | 3.37974110992686846e-5 | 198.683477052163953 | 199 / PASS |

保存d_oldはB′ **2.8351571426291167e-7**、Reference **2.837740528270632e-7**。各値はCT-0の採用値と旧B′ ERROR_RPE／Reference STAGE3_ALL_SUPPORTのtotal_raw_disk_upperが同じであることを確認した。B′では旧SOURCE／finite-RTE／phase／finite-bit proposalとnative compiler diskを含み、Referenceではvacuum生成のIEEE誤差も既に含む。元の古いcompiler diskを復活させたり、これらの成分やπ差を別に加えたりしていない。

Nominal raw radiusは保存 **0.9827337331952227**。Sampling law、normalization、attenuation、reset/readoutとX/Y raw signalの定義を変えていないため、追加のnormalization倍率を入れない。点ごとの一様誤差から同じcanonical lawのcoherent meanを囲む議論であり、平均channel一致を複素信号の保証へ代用しない。

[RAW_DISK_ACCOUNTING](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/RAW_DISK_ACCOUNTING.json)に旧disk、合算、半径、位相の区間端点を保存した。これらの区間は**保守的上界式・下界式の数値包絡**であり、未知の実誤差自体に正の下界を与える区間ではない。

両軸maxを入れる前のper-axis診断値とそのSOURCEは[INITIAL_PER_AXIS_UNADOPTED.tar.xz](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/INITIAL_PER_AXIS_UNADOPTED.tar.xz)へ保全した。これは採用結果ではなく、公開採用は本表とRPE_199_SHOT_CHECKSの両軸max込みの値である。保存CT-4Bのper-axis条件行も式の再現照合に使い、両座標を覆う新しい保証の代用にはしない。

## 3. 半径・位相・保存199の検算

CT-0／CT-4Bと同じ式を使った。

`rho_lower = rho0 - d_total`

`phi_disk_upper = asin(d_total/rho0)`

`beta_stat_lower = B_lower - phi_disk_upper`

`N_required = 4 log(1440) / (rho_lower^2 sin(beta_stat_lower)^2)`

Asinは既存の `atan2(ratio, sqrt(1-ratio^2))` で評価し、PF予算控除済みの保存B_lower=0.3999863216400871を維持する。位相の変換順を変えず、PF誤差をraw diskへ重複加算しない。

| 条件 | Raw半径の下界 | Disk位相の上界 rad | 統計位相の下界 rad |
|---|---:|---:|---:|
| B′ X | 0.982699866123743496 | 0.0000344621033571502546 | 0.399951859536729922 |
| B′ Y | 0.982699866123743496 | 0.0000344621033571502546 | 0.399951859536729922 |
| Reference X | 0.982699935784123456 | 0.0000343912190701375723 | 0.399951930421016934 |
| Reference Y | 0.982699935784123456 | 0.0000343912190701375723 | 0.399951930421016934 |

半径・統計位相の正値、disk/rho0<1、PF＋disk位相<.4、必要連続N上端<=199、同上端の整数ceil<=199をすべて有理端点で検算した。Integer ceilは4条件とも199。新しいshot scheduleを選んだわけではない。

Alpha1/720の既存Hoeffding十分条件からlog(2/alpha)=log(1440)をそのまま使った。36/720=.05は各armの元の18round×2軸protocolにおける保存配分で、両armを一つのprotocolへ合算していない。今回はm7以外のCT成功条件を再会計していないので、18round CT全体のfailure.05を新たに認定した結果ではない。

[RPE_199_SHOT_CHECKS](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/RPE_199_SHOT_CHECKS.json)が判定の正本。ShellなしではCT-4Bのconditional uniform行8件を同じ式・192-bit精度で再現し、CT-0純粋geometry関数とも区間端点が一致した。新しいshell込みの判定を、旧保存PASSの引用だけで済ませていない。

## 4. SOURCE再利用と資格・保全

新規[限定runner](../../../scripts/account_pr_reference_n4c.py)は、既存I／upper／lower、CT-0のrzcount／count_assets／worst_counts／geometryを変更していない関数ASTとして再利用した。CT-4.validateは価格集計や旧出力への副作用も持つため起動せず、**元の6つの数値代入文だけをそのまま抽出した純粋wrapper**で再利用した。CT-1からは両軸maxの一様上界代入文だけを再利用し、q1の科学計算や結果を移送していない。区間演算、disk伝播、shot式を再実装していない。新規差分はFractionのRZ＋shell項、入力・SOURCE対応と結果保存だけである。

Mpmath1.3.0の192-bit区間端点を既存mpf→rational変換でFraction化し、表示用binary64へ変換する前にPASSを決めた。新規provider／word emitter／quantum moduleは実行していない。N4-Bの85 certificate・C0299保持・共通policy・局所34検査・保存whole8検査・SOURCE hash、N3の全vacuum域／allocated upper<10^-11の証拠を読取照合した。旧検証suiteや科学runnerの再実行ではない。

資格は **SOURCE_AND_COMPILER_CONDITIONS_PRESERVED／ROLE_WORD_POLICY_QUALIFIED／CONSUMER_SHELL_BOUND_INCLUDED／JOINT_SYNTHESIS_DISK_EVALUATED／SAVED_199_SHOTS_SUFFICIENT**、および **CONDITIONAL_ON_COMPILER_CONTEXT**。固定m7に対する条件付き十分条件が今回の到達点である。無条件の元H表現保証、他roundのCT成功、全support実T価格、18round総T／T-depth／FT優越は今回認定していない。これらへ結果を拡張した意味での **FULL_SUPPORT_CT_RPE_UNQUALIFIED** は維持する。

[SOURCE_INPUTS](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/SOURCE_INPUTS.json)、[ENVIRONMENT](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/ENVIRONMENT.json)、[CHECKS](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/CHECKS.json)に実入力hash・再利用箇所・条件・検算を保存。既存固定venvで `PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python scripts/account_pr_reference_n4c.py` が再現入口。FreshなN4-C出力領域を使う条件を持つ。実venv pathはENVIRONMENTに記録した。

採用再会計の実処理は約1.88秒、CPU affinity64・空きRAMを確認。4つの小さい区間式なので1 processで完了し、資源capや共有設定変更はない。著者Codexによる技術再会計であり、独立詳細科学レビューではない。正式共通運用指示2026-10-25、親／worktree AGENTSとPROJECT_MAPを適用し、旧SOURCE／raw／結果／grant／ledger／marker／STOP、他Track・他ユーザーは不変更。

**[MANDATORY STOP](../../../artifacts/pr_reference_n4c_accounting/2026-11-08/STOP.json)。次担当はGPT／ユーザー**：本4条件の条件付き資格を研究上どう採用するか判断する。追加資格・他round・総T等が必要なら、対象と実行承認を別途定める。CodexはN5／N2／CT-3B／SK／全18round等へ自動進行しない。
