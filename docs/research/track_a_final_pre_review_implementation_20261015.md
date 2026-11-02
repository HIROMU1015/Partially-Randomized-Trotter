# Track A：科学レビュー前の最終技術実装（2026-10-15）

**H8 D0 driverとH6 native B0+I bridgeを独立実装し、最終41件の人工検査に合格した。両方とも実装範囲のTECHNICAL_READY。実値解析・実signal・trajectory・native build/compileは0、実行認可は発行していない。**

[科学判断handoff](track_a_final_pre_review_handoff_20261015.md)、[監査bundle](../../artifacts/resource_applicability/track_a_final_pre_review_implementation_v1/2026-10-15/README.md)から次の指定ができる。未採択の科学条件と、将来の実行直前bindingを分けて記録した。今回、希少次数の上界調査、H5/H7/T実験、原稿改訂、詳細GPT科学レビューは行っていない。

## 1. 固定参照と新SOURCE

今回SOURCE **`60eeb51c148007359cf6e8bcf6288e6837a39701`**。基礎は指定prototype SOURCE `3c513d1ad52ce99f0d2df7d8a300e4d58d15a553`。初期実装SOURCE df743438…にH6の明示roundoff政策を加えた最終SHAである。branch `track-a-final-pre-review-implementation-20261015`、独立worktreeはhome配下。旧資料branchをmergeせず固定参照として利用した。

前回技術HEAD `560ba600dce8737f588e030fbd3f1f6c5559cbbc`、D0 result `8dbdb68293c988a999645e21a82b53966f9ef6cd`、H8 result `681ec23c93fdd1090d325908f4b155a79115afdf`はorigin一致。共通科学判断commit `4675730e7e40d25ba3618334a4a0e28b4bd2f145`と合わせ、4指定主要資料のGitHub raw/Git bytes/SHAを確認した。[固定参照receipt](../../artifacts/resource_applicability/track_a_final_pre_review_implementation_v1/2026-10-15/fixed_github_reference_receipt.json)。変更のない共通指示のSHAは709524f8…、全文読了済みの規則を引き継いだ。

[closure](../../artifacts/resource_applicability/track_a_final_pre_review_implementation_v1/2026-10-15/source_closure_v1.json)は新code/tests8件と凍結依存4件、計12fileのactual Git blob/bytes/SHA。既存D0 SOURCE `fa75c7997e135da7b7a0c4e23949e47252bd7a2e`、prototype2件とその旧23検査は不変。既存科学source/registry/runner、結果、grant、ledger、private venvは変更していない。

## 2. H8 D0 driver：完成した実行経路

[driver](../../scripts/h8_d0_saved_execution_v1.py)、[共通実行guard](../../scripts/final_pre_review_execution_v1.py)、[sequential launcher](../../scripts/launch_h8_d0_saved_once_v1.sh)を追加した。shot、weighted cost、paired SE、selection、fixed crossing、lower envelopeは**旧D0純粋関数を変更せず再利用**する。前回adapterの実解析入口を開放したものではなく、新driverが明示認可を要求する。

未認可contract/receiptは入力mapping loaderの呼出前に拒否する。contract exact byte SHA、user承認とreview証拠、SOURCE、input/binding/profile/resource/CPU/carry digest、exclusive output、retry=falseを検査する。CLIは人工grantを拒否する。現在の[実行契約草案](../../artifacts/resource_applicability/track_a_final_pre_review_implementation_v1/2026-10-15/h8_d0_execution_contract_draft_v1.json)はapproved=false、runtime_authorization=false、allowed_cpus=[]、未seal、command=nullであり、実H8を起動できない。

認可後の順序は以下。

1. 新SOURCEのdisk/Git bytesとruntime profile、process thread1、許可CPU、memory/pressure/OOM、output filesystem block/inode/quotaを確認。durable exclusive one-shot markerとown output directoryを作る。
2. 受領済み6入力、**6,518,976 bytes**の正本receiptに完全一致したJSON/CSVだけをロード。科学arrayは扱わない。H8 SOURCE/seed/compiler/input/selection/paired phase/replicaを確認。
3. Phase A：探索92と確認union6＋det20を分離し、旧4ε `.05/.01/.005/.001`・factor1/10の**944 precision scope行と3,576費用map行**を完全一致で照合する。bias=corrected−full reference、`log_B=R log(b)`（bはevent normalization）、det b=1を確認。全accounting dictのheadroom/eligibility/整数shot/log-only/statusに加え、六metric point/paired SE、axis mean/min/max/n/SD/SE/shot、map phase/methodを検査。旧RZ上位2×4εのunionが事前selectionと一致することも要求する。
4. **両scopeのPhase A PASS後だけ**、Phase Bの5ε `.0005`を含む1,180precision slots、六metric、paired SE、RZ選抜の他metric/metric再選択、固定RZ pairのP感度と全適格候補lower envelopeへ進む。最大metric slots7,080、P曲線120。新metric点最小が旧union6外ならOUTSIDE_ORIGINAL_UNION6と表示する。
5. input/source/profileを再確認し、出力bytes/SHA、provenance、STOP/FD/timer終了を保存。native exitはdriver自身から推定せず、**同じshellがforeground processをwaitした後**、別の逐次metadata専用呼出でkernel identity不在とshell wait statusを記録する。解析再実行ではない。

探索B2 n8を確認B2 n32へpoolしない。確認のdetは旧探索n1だけreuseする。ordinary anchor4軸はprimary statistical costから除外する。MISSING eligible costはSTOP、INELIGIBLE/NUMERICALLY_UNRESOLVEDとは別。保存histogramの未観測layerはUNOBSERVED/平均nullで、費用0としない。no new estimator、no fallback、no tolerance relaxation、no retry。

### 監視と資源・容量

候補STOP上限は**解析process1/thread1、AS512MiB、RSS256MiB、CPU240s、wall300s、output32MiB**。1秒未満の明示checkpointと0.5秒のown-process signal timerを使い、AS/RSS/CPU/wall、affinity/thread変化、host PSI<1%・visible own cgroup PSI0・OOM増分0、memory余裕2GiB、filesystem/inodeを読む。5秒以上の観測interval又は観測I/O所要ならfail-closed。最後の64観測、最大interval/所要、phase、最初のSTOPを保存する。他PIDへのsignal/共有設定変更はない。

signal timerも長いC処理中にPythonへ戻らなければ配送が遅れる。戻った時点の5秒stalenessでSTOPする。独立observerを追加していないため、**SIGKILL/長いC処理中の即時RSS停止とterminal作成は保証しない**。kernel AS/CPU制限、shellのnative exit receipt、fresh headroomを併用する。これは軽いsaved-scalar one-shotの運用設計で、全環境での成功保証ではない。

契約はinput copy6.22MiB、output final/pending各32MiB、log2MiB、monitor journal reserve8MiB、temp32MiB、terminal2MiBとblock丸めを積算する。fresh free256MiB/20,000inode、own output100file＋terminal4inodeを候補にした。actual outputはstreaming JSON/FD O_EXCL/fsyncで計数。実wall/RSS/outputは**UNKNOWN**。H4/H6のD0実測や今回人工fixtureの時間をH8予測へ外挿しない。

quota-enabled mountはall-user/group/projectのfresh effective quota receiptがなければSTOP。単なるfilesystem空きからquota余裕を推定しない。現在CLIのdefault monitorはquota receiptを供給しないため、そのfilesystemでは起動が拒否される。必要なら科学条件に依存しないfresh receipt接続を実行前に追加する。quota/config自体は変更しない。native exit記録は逐次process、source verificationのGit呼出も短い読取childで、科学解析process/observerを増やすものではない。

## 3. H6 B0+I bridge：完成した接続

[独立bridge](../../scripts/h6_native_b0i_bridge_v1.py)は既存B0を置換せず、H4 full-Fock Pauli identity規約を**未採択の対照候補**として実装した。`C_tail=Σ_tail lambda/4[(TrG)²+Tr(G²)]`、signed lambda、exact Hermitian finite G、generation order/prefixを維持する。sector平均/energy fitへ変更していない。

`prepare`は認可/SOURCE/profile/inputをloader callback前に確認する。tail extractionはcaseごとに一度だけ予約し、失敗後に同じGrantで再試行しない。元prepared B0はtail空/extracted identity0/先頭prefixと一致するものだけ受理。旧metadataのpreparation hash、constant、block order/mode/prefix/q/inputから[静的blueprint3件](../../artifacts/resource_applicability/track_a_final_pre_review_implementation_v1/2026-10-15/h6_prepared_metadata_receipt_v1.json)を固定した。これはcompiled numeric fingerprintではない。

`build`は核・元constant・one-body・retained block phaseをそのまま保ち、元`build_deterministic_native`へ**scalar=original constant+Cを一度だけ**渡す。元blockはdeep copyし、元preparedのbefore/after byte digestを確認する。ordinary/symmetric_directionalのancilla relative P(-T scalar)を維持し、cosine/sineをRe/Imへ対応させる。request/scalar/input/blueprintの改変・二重overlay/buildを拒否する。matched B0がない場合の費用deltaはnullで、自動buildしない。

`prepare_from_snapshot`は固定科学checkout、snapshot/DF receipt/保存signal byte SHAを確認してから、旧`load_h6_snapshot`と`_prepare_discard`へ遅延接続するready helperである。今回このreal経路は未実行。元H6 provider SOURCE `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`のPython230fileを固定参照closureにし、profileは既存Python3.11.0rc1/private venvを参照する。H8 SOURCE/D0解析Python3.12.3とは別identity。library installやcompiler options変更を要求しない。

### 保存signalとallowance

`z_I=exp(-i*.8*C) z_B0`、exact truncatedも同じ位相で回転し、**full-H referenceは回転しない**。更新biasはcorrected_I−full reference。元recordはdeep copyで保持する。

旧uはfrozen H6 `safety*max(floor,rounding,reference_error+max(path)+raw_closure)`によるEMPIRICAL診断。独立寄与receiptがある場合はsignalとreferenceを別々に使う。ない場合の明示候補政策は、旧combined uを**各寄与の保守的な経験的診断上限**として二つに分け、signal側を`|cosθ|u_Re+|sinθ|u_Im`で回転し、reference側uは回転せず加算する。旧uをそのまま新axisへコピーする処理ではない。厳密なtrue-error上界にも昇格させない。

stored binary64 coefficientsに対するidentityをFractionで独立に評価し、float Cとの差を外向き丸めしたarithmetic receiptへ保存する。constant+Cと-T scalarのrounding、declared input coefficient errorを別項目にする。complex/libm rotation errorは**明示EMPIRICALレシピ** `max(1e-12,64*epsilon*(1+abs(.8*C))*max(1,abs(z)))` 又は別receipt値。algebraは外向き丸めするが、libm errorの厳密certificateではない。原HamiltonianのDF近似誤差を解消したと扱わない。

将来pilotはp5/q1、p10/q2、p15/q8、S2/T=.8、I6軸のみ。旧snapshot/DF receiptとscalar signal4入力は**1,577,349 bytes**、NPZは今回未読。H6 candidateのinput/constant/preparation metadataと新bridgeを結合する。matched B0最大6軸は別scope・別承認、H8への拡張/full mapは別科学判断。

bridgeはcompileを実装/自動呼出しない。pilot runnerが新grant、monitor、bounded artifact、既存固定compiler cost APIに結合して6軸を実行する。今回のbridge APIはその接続を用意したもの。real native correctness/fingerprint/metric deltaは実測後に確認する。

## 4. 人工検査と残る条件

最終**41 distinct tests PASS**：H8 driver14、H6 bridge17、実行guard10。attempt04で40件PASS後、最後のH6 roundoffレシピ差分をattempt05で17件PASS。H8/guardの合格campaignはその後反復していない。旧prototype23件は不変で再実行0。

人工H8は92候補/union6/JSON/CSV/n8/n32、旧値exact gate、axis statistic、seed/source/axis、欠測、status、Pのrational交点、union外表示、未認可loader拒否、source/input/contract改変、one-shot、same-parent native wait metadataを検査。H6はtiny2-mode/rank19 mock、signed/full-Fock/prefix/phase、reference不回転、allowance、元blockの不変、fingerprint/改変/重複を検査。guardはwall/欠測/遅延/PSI/OOM、block/inode/quota、output/terminal reserve、FD二重close等をfixture検証した。

人工testsはforeground1process/thread1、AS512MiB/CPU120s/wall180s、fixturefile64MiB/log1MiB、scratch home専用。attempt04 wall6.223s/peak75,440,128B、attempt05 wall.126s/peak28,835,840B。attempt01のfixture file1MiB不足、attempt02の人工RZ selection不一致を修正し、失敗logを保持した。NumPy/Qiskit/CuPy/科学module import0、real build/compile0。local人工検証でありCI・外部再現・科学reviewではない。

残件は実装上の技術不合格とは分ける：科学採択、new user receipt、CPU/output/carryの実行前binding、fixed provider/profileをcontractへ展開・fresh byte検査、resource/quota、real old equality/native build/control/fingerprintの実証。長いC/SIGKILLへの制限、quota-enabled時のfresh receipt接続は上記の運用限界として明示する。実needed wall/RAM/費用優位はUNKNOWN。新科学計算・実数値再会計・grant発行0。資料公開後mandatory STOP。
