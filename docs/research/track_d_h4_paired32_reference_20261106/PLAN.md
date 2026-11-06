# Track D Stage A：保存32 trajectoryのpaired Reference検証

2026-11-06 JST。ユーザーの今回の明示許可に従い、必要最小限のadapter拡張、correctness、全precompile PASS時の最大128 wrapper compile、有限cohortのpaired資源比較だけを実行する。旧STOP・grant・SOURCEは保持し、承認証明や新grantを作らない。

## 固定対象・SOURCE

- 基点は静的調査 `523efe89fde52a14a8e0a4eba4ea6b70e3f512e1`、既済H4結果 `7d0dc55199c49a3f49a9ba93323843d8a336cce1`。
- linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、δ=h=.1の**1 h-step**。元H厳密groundを仮定し、状態準備費除外。保存own-DF groundを測定入力に使わない。
- 元C科学SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` の143 source・16保存入力を、元manifest／契約のhashに照合する。旧runner・4-word実行は起動しない。
- 保存carrier `3649b75af191916daf44abb91e14d34c43b83e8f` の `artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/C/` から、sample00..31のevents、system_reference、ordinary rawを同一bytesで回収する。96 input bindingをSOURCE.jsonに保存する。
- Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、元h/g/scalar `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。
- Cohort `near_practical_PR_r32_fixed32_v1`、master2026110304、SOURCE manifest `103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739`。元policy paired=false／adaptive=falseを保持し、今回同pathの2方式をpaired比較する。新sampling・endpoint・cut・K変更なし。

## 最小拡張と公平性

旧 `track_d_h4_reference.py`、runner、check harnessは変更しない。新 `track_d_h4_paired_reference.py` は既存のprepare_arm／configure、StrongBuilder、full-frame provider、signed-event emitter、control_one、directional_head、reference_gadget、wrapperを再利用する。既済word02固定の入口を呼んで余分な分子wordを生成せず、共通headをworker内で再利用し、保存32 eventsを渡す部分だけを拡張する。

S2の前半headはuncontrolled、後半逆順headはdirectional、中央tailはordinary controlled。headのrun grouping／fsum／解析位相付きrelative Gaussian、tailのexact basis共有を維持し、cross-head/tail relative fusion=falseを両方式で保持する。Gaussian全制御や融合を外した弱い対照へ戻さない。

各pathはcomplete event16直後で一度だけ分割する。B=前半head＋先頭16 events、A=後半16 events＋逆順head＋scalar/identity一回、A†はactual inverse。χはscalar/identity・event.phase・signed rotationからpath別に求め、ancilla P(χ)へ入れる。B_Kやattenuationをunitary／observableへ乗せない。cutで共有を失う場合も分割回路へ実際に反映し、費用を差し引かない。切断端のbasis一致を記録するが、別cutのcompileによる因果ablationは行わない。

## correctness → compile → postcheck

1. 元CAR reference・保存raw、A B、actual A†、unitarity、両halfの真空ray、χを既存check harnessで照合する。direct armは強いsymmetricだけ。Referenceはreference/ancilla=0で全256 system入力列・全131072出力振幅を既存CPU Aer double state-actionで評価する。raw complex coherenceとX/Y observable、位相欠落負対照を確認する。
2. 全32 pathのprecompile検査がPASSするまでcompileを開始しない。Referenceとsymmetricのraw X/Y observableを直接paired照合する。
3. 2方式×X/Y×32のfull wrapperをP0で各一回だけcompileする。最大128件。H→core→YのみSdg→H→ancilla測定、状態準備なし。既存 `compile_checked`→MPFR160 declared-phase→位相保持unrollを再利用する。
4. compiled directは全512 operator、Referenceは全256 system入力列のisometry／observableをraw・CARへ照合し、compiledの2方式もX/Yでpaired照合する。17-qubit全Operatorは作らない。

P0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initial-zero false、topologyなし。Gaussian `nature_jw_exact_zero_lapack_atan2_v1`。binary64/complex128、component・unitarity abs1e−10、observable Frobenius／signal2e−10、rtol0、global phase fitなし。旧package10 pinsを専用venvに導入し、Python版など実際の実行contextを記録する。MPFR160を全数値評価の精度とは呼ばない。失敗後のallowance緩和・再実行は禁止する。

## 資源・出力・判定

初期観測は64 CPU・空きRAM約465 GiB。旧Reference worker実測peak約6 GiBを参考にした9 GiB/workerの計画見積もりと実空き資源から並列path数を選ぶ。CPUを独立pathと各pathのAer／大きいGram contractionへ配分し、小行列は既済のBLAS1方針を使う。OSのCPU/RAM/worker/wall制限や共有設定変更を新設しない。開始済み処理を通常の競合・監視遅延だけでkillしない。

Fresh outputは `artifacts/track_d_h4_paired32_reference/2026-11-06/run01` 一回。retry/resume0。SOURCE、96入力identity、条件、fusion、raw/compiled wrapper、A/B/actual A†、checks／observable、paired CSV、有限cohort集計、inventory、STOPを保存する。rawは必要なら元pathとhashを保つ通常archiveへまとめ、非公開環境path・cacheを公開しない。

各軸32本のRZ/CX各count/depth、total depth、size、qubit数について2方式の価格と差、平均・最小・最大・cohort SD、改善／同値／悪化数を記述する。X/Yを64独立標本として数えず、母平均CI、全law保証、量子shots/N、全round RPE総費用は算定しない。2 CSWAP cascade・各8対・P(χ)・追加8 qubitsとcut損失を含む実際のwrapper価格を使う。

SOURCE/入力不整合、correctness不一致、公平なbaseline構築不能では新call・新task投入を停止し、開始済み処理のpartialを回収して保全する。failed pathの置換、対象変更、別cut、sampling、無断retryはしない。科学SOURCEを実行前にcommit固定し、SOURCE/入力を実行後も照合する。

成功・失敗どちらも専用branch `codex/track-d-h4-paired32-reference-20261106` へ公開し、remote SHA・artifact bytesを照合してmandatory STOPする。次担当はユーザー／GPT。B（元HのRPE統計・N）、C（全round総費用）、重要科学レビューは開始しない。
