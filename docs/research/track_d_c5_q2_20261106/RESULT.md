# Track D C5：共通whole-path入口と固定q2公平比較

**固定q2のraw correctness、全4 P0 wrapperのpostcheck、最終paired X/Y・vacuum χ照合はPASS。SOURCE/inputを保全し、TECHNICAL_PASS_MANDATORY_STOPで科学実行を終了した。共通入口と資格付き有限fixture価格recordを取得した。**

実行SOURCE `976c898fbdb79477c1db3178b5d32df1fe5aeb5c`。[固定PLAN](PLAN.md)／[人工・回帰16件](SYNTHETIC.md)。C4 `a5784c4bd3e3fb9779849a1e1fc1759559e2caea`を基点とするC5専用branchで実施した。C3/C4の旧worktree・SOURCE・結果・STOP、旧C2の価格は変更していない。

## 固定入力・SOURCE・費用境界

linear H4、隣接1.00Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=.1。元H厳密groundを仮定し、状態準備費は除外する。保存own-DF ground vectorは使わず、全256 system入力列の状態非依存observable/coherenceを検査する。

cohort `near_practical_PR_r32_fixed32_v1` のsample00→01だけ、q=2、64 events、V=U01 U00=A B、B=U00/A=U01、complete step1後cut。fixture/cutを価格結果で変更していない。endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、元H packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`を保持した。

既存preflightで元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`のSOURCE/input/lawと、carrier `3649b75af191916daf44abb91e14d34c43b83e8f`の96保存binding、Stage A/C1/C2の固定証拠、C3完了cc554のSOURCE/STOP/資源を照合した。[今回SOURCE](../../../artifacts/track_d_c5_q2/2026-11-06/run01/SOURCE.json)に全bindingとpackageを保存する。normalization/attenuationをunitaryへ重ねて乗算せず、scalar/identityは各step一回だけ会計した。

## 共通入口・再利用・最適化

新`track_d_whole_path_price.OrderedRequest`はq、時間順保存sample、cut、元SOURCE/carrier/endpoint/cohort/law/inputを持つ。`prepare(context,saved,request)`で同一requestのbundleを一度構成し、`wrapper_for(bundle,request,method=...,axis=...)`で両方式・X/Yを選ぶ。context identity、request identity、method/axis不一致を拒否する。

C3の演算bodyを`prepare_ordered`へ、raw保存/checkerを`save_and_check`へ最小抽出した。旧C3 `prepare(context,saved,q)`は同じ固定q4/q8 map・cut・metadataで委譲する。C5以外の既存科学SOURCE commitを変更した意味ではない。旧sourceと新固定入口は2-mode人工q4/q8でmetadata/operator回帰を確認し、実H4 q4/q8の新生成・compile・postcheckはしていない。

Stage A `prepare_common`、StrongBuilder、C3 `mixed_head/physical_segment`、Reference gadget、actual inverse、独立保存CAR、correctness harness、C2のP0 compile/postcheckを再利用した。新しいGaussian合成器・sampler・compiler・state simulatorはない。共通入口は現時点で固定cohortの明示保存列用で、C6のfresh law生成器まで実装済みとはしない。

強いsymmetricはuncontrolled Gaussian、directional center、head relative fusion、tail同basis共有、1 step境界のdirectional half＋uncontrolled half→ordinary controlled full融合、全path scalar/identity集約を保持する。Referenceは同じVに一つのgadget、Aのactual inverse、path全体χ、追加8 reference qubits、16 CSWAPを含む。global cutでその1境界融合を失い、内部step融合は0。両方式とも新しいhead-tail横断融合は追加していない。弱いbaselineへの置換・別方式の節約値の加減算はない。

価格recordは方式/q/axis/request/cut/SOURCE/最適化政策/compiler/precision/full-wrapper scope、RZ/CX数・各depth・total depth・size・logical peak幅、raw/post/paired資格を持つ。`finite_fixture_compiled`と明記し、law期待値/CI/model価格/RPE総費用はnull。今回のq2/00→01/cut1以外や未PASS、kernel/shell scope、異なるP0を誤受理しない。将来のlaw mean providerの完成・長q資格を意味しない。

## 人工・回帰検証

C5新規8、既存C3 4、C2 2、Stage A 2の計16件PASS。要求14項目の対応は[SYNTHETIC.md](SYNTHETIC.md)に記録した。順序/q/cut、SOURCE/context/law、AB/actual adjoint/vacuum/χ/scalar一回、Gaussian/signed tail、両方式X/Y、合法な融合とcut損失、旧C3入口、費用境界/未検証q/未資格価格、保存compile byte再利用、公開projectionを含む。人工fixture API/真空位相とStage Aの起動方法を科学実行前に修正し、閾値緩和・phase fitはしていない。人工PASSをH4の性能証拠としない。

## 実H4 raw・compile後correctness

raw既存22項目＋paired X/YはPASS。最大成分差は2.22044604925e-14、raw reference leakage成分/L2は0/0。χは−0.4574879096539928で、AB=V、actual A†、vacuum/解析位相、scalar一回、strong両branch、complex coherence、X/Y、χ省略negative controlを含む。

|wrapper|判定|最大成分差|CAR observable Frobenius|reference leakage L2|
|---|---|---:|---:|---:|
|symmetric_directional_X|PASS|1.97555101088e-12|2.78446339463e-13|非該当|
|symmetric_directional_Y|PASS|1.28016654201e-12|3.43597315274e-13|非該当|
|reference_X|PASS|1.4701431642e-12|1.06240198211e-11|1.39560553359e-13|
|reference_Y|PASS|8.43392904813e-13|2.59726653634e-12|1.39560553822e-13|

最終paired X/Yと両方式のvacuum χ照合もPASS：

|検査|最大成分差|数値Frobenius差|
|---|---:|---:|
|X|7.13762382535e-13|1.0834559642e-11|
|Y|3.17579297102e-13|2.63338059508e-12|
|symmetric_directional_vacuum_chi|7.35182276441e-15|7.35182276441e-15|
|reference_vacuum_chi|6.88521377278e-13|6.88521377278e-13|


component abs1e−10、observable数値Frobenius/signal2e−10、rtol0、global phase fitなし。17-qubit全Operatorは作らず、既存state-actionでreference/ancilla=0の全256 system入力列・各131072出力振幅を扱う。directは全512 operatorを照合する。local binary64照合であり、丸め誤差の厳密包絡、全law実装bound、immutable CI、独立した詳細科学レビューではない。

P0はQiskit1.3.0、optimization_level1、seed17、rz/sx/x/cx、approximation1、initial-zero false、topology-free、位相保持。既存declared-phase compile規約を維持し、許容誤差やcompilerを変更していない。

4 wrapperともC2保存rawとの完全byte一致は得られず、compiled再利用は0、各1回の新P0 compileを計4回実施した。[再利用判定](../../../artifacts/track_d_c5_q2/2026-11-06/run01/reuse.json)を保存する。raw bytesの不一致は演算子不一致を意味せず、その原因を名称差等と断定していない。今回の同一性規則に従って新SOURCEの別回路として扱い、4 postcheck・paired/χとSOURCE/input再照合を完了した。

## Full-wrapper資源と旧C2・保存C3との関係

各方式でX/Yの全7資源が一致した。各軸1 full-wrapperの値を示す。

|方式|RZ count|CX count|RZ depth|CX depth|Total depth|qubits|size|
|---|---:|---:|---:|---:|---:|---:|---:|
|strong symmetric-directional|10773|4100|2624|1996|6096|9|21970|
|Reference|11114|3860|1357|860|3014|17|22263|
|Reference−symmetric|+341|-240|-1267|-1136|-3082|+8|+293|


**C5と旧C2は、両方式・両axisの全7資源値が一致した。** C3型の手動境界融合・位相集約をq2へ明示適用した今回、P0後の追加価格節約は観測されていない。raw bytesの不一致と資源指標の一致は別の事実であり、旧compilerがどの段階で同じ節約を得たかの機構・単独因果効果は未分離である。

保存C3の同政策q4/q8を再実行せず転記する：

|保存q・方式|RZ count|CX count|RZ depth|CX depth|Total depth|qubits|size|
|---|---:|---:|---:|---:|---:|---:|---:|
|4 symmetric|19867|7686|4746|3744|11116|9|40572|
|4 reference|20092|6998|2516|1570|5583|17|40301|
|8 symmetric|39610|15360|9492|7515|22259|9|81015|
|8 reference|39602|13776|4531|2870|10093|17|79615|


RZ/CX/depth/sizeを一つの指標へ合成せず、qubitsは別の幅制約とする。追加register・CSWAP・actual inverse・χ・global cutの融合損失は上表のfull-wrapperに含む。旧kernel＋shell会計へ渡していない。routing・実機接続制約、RZのfault-tolerant合成/T費用、reference維持/reset等は未評価であり0と認定していない。topology-free depthを実機時間と同一視しない。

旧C2の同q2は追加手動cross-step融合・全path位相集約を含まない。C5とのSOURCE上の差は実装政策全体の変更であり、今回P0後の7指標は一致した。融合とphase集約の単独の因果的節約内訳を測ったものではない。C3 q4/q8とは同じ最適化政策を使うが、path長・実際のevent列・cut位置は異なる。保存q1/q2/q4/q8から母平均・q依存model・長q価格をfitしていない。

## 保全・公開・未解決事項

[result](../../../artifacts/track_d_c5_q2/2026-11-06/run01/result.json)、[STOP](../../../artifacts/track_d_c5_q2/2026-11-06/run01/STOP.json)、[価格record](../../../artifacts/track_d_c5_q2/2026-11-06/run01/prices.json)、[旧結果との比較](../../../artifacts/track_d_c5_q2/2026-11-06/run01/resource_comparison.json)を保存した。raw/compiled QPYと検証NPZは[lossless archive索引](../../../artifacts/track_d_c5_q2/2026-11-06/run01/payload_archive.json)で保全し、archive/member bytesを元出力と照合した。

元checker出力・内部CPU/RAM観測・cacheは非公開localにそのまま保持する。公開JSONは科学check/phase/metricsの値を保持したviewであり、実行時間・peak RSS・thread観測欄を除いた。元出力は編集していない。公開archiveはowner/path/mtimeを中立化し、内部環境path・非公開raw入力・資格情報・運用logを含めない。共有OS/venv/scheduler/他jobを変更せず、実空き資源に基づく配分で実行した。

SOURCE/inputの実行前後一致を確認した。新科学実行は固定q2の1回、compile4、保存compile再利用0、retry/resume0。新sampling、ground solve、DF refit、N評価、量子Born shots、q>2の新科学実行、paired母平均校正、価格model fit、holdout、全RPE総費用評価は0／未実施。Aer snapshot内部のshots=1は量子shot実験ではない。

残るのはC6以降の対象law・標本単位・必要数の採択、fresh pathの同request paired価格校正、独立holdout、共通校正の共分散、長q適用domain、B側の元H N・同一精度/成功条件との接続である。価格recordや同一fixtureの数値PASSだけでこれらを資格付けしない。

**mandatory STOP。次担当はユーザー／GPT**：今回の公平q2証拠と共通入口を評価し、C6の実装/新sampling/校正/holdout等の対象・SOURCE・入力・科学call範囲を判断する。Codexは新しい明示許可の範囲だけを担当し、C6、q>2の新科学検証、長q評価、N/成功率/全13-round費用、B2/B4-2、ground solve/DF refit、詳細科学レビューへ自動進行しない。
