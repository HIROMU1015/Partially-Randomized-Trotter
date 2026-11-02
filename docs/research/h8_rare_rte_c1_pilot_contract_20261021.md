# H8希少RTE C-1：S2_SINGLE pilot固定契約

2026-10-21 JST。**契約・標本設計・read-only照合・人工検証まで完了。C-1実科学は未承認、mandatory STOP。** 基準commit `cb9c994be608233fccc49ff2e44007bbb32e2fb4`。契約用SOURCE `bc1a77a89584427605b1ff287aff6e809877dd09`。このSOURCEはC-0.5 adapterを変更せず継承し、契約算術・検査だけを追加した。

## 固定した対象と境界

線形H8/1 Å/STO-3G/tol-only 1e-8 DF28/T=0.8/P=0、通常signed bias・allowance1。H8独立Pを変更・再選択せず、p8/q2/r4/K4（ε=.05）とp7/q16/r4/K4（ε=.0005）を扱う。**新cohortは各条件S2_SINGLE 43本、計86本**。S2_MULTI/S4_PRESENTは未取得・平均/分散null。S0は既存P n48を別anchorのまま保持し、今回の会計に混合・再解析しない。

目的は条件付きpaired費用の経験平均・不偏標本分散・SEと、実行負荷の取得。production本数設計、全層期待費用の新評価、順位・感度の再解析、数学的上界採択、追加sampling、詳細科学review/final freezeは対象外。pilotを完了・中断したらSTOPする。

[条件別契約・承認書案・予算・監査packet](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/README.md)。旧[C-0.5実装](h8_rare_rte_stage_c05_implementation_20261021.md)と[数値規約](h8_rare_rte_stage_c05_numeric_pilot_20261021.md)は保全した。

## 標本数43本の決定根拠

各条件の条件付きpaired費用を、計画上、独立同分布のNormal(μ,σ²)、0<σ²<∞と仮定する。実費用は離散・非負であり、正規モデルは未検証の理想化である。PRNG namespaceの分離も数学的独立性の証明にはならない。このモデルの採用は標本設計に限定し、実H8の母分散・母平均の保証としては採用しない。

ν=n−1、α=.10。正規モデルの分散のequal-tail区間は

`[ν s² / χ²(.95,ν), ν s² / χ²(.05,ν)]`。

式の参照：[NIST χ² confidence interval](https://www.itl.nist.gov/div898/handbook/prc/section2/prc231.htm)（2026-10-21取得。標準偏差の式を二乗して分散へ対応）。η=1/2の基準を、**区間の両端がs²の±50%以内**、すなわち`[ν/χ²(.95,ν),ν/χ²(.05,ν)] ⊆ [.5,1.5]`と明示した。これは全区間幅だけを50%にする基準でも、`P(|s²/σ²−1|≤.5)≥.9`という別基準でもない。

|n|ν|χ²(.05,ν)|χ²(.95,ν)|区間下端/s²|区間上端/s²|採否|
|---|---|---|---|---|---|---|
|42|41|27.325551469999|56.942387146808|0.720026013210|1.500427175093|上端超過|
|43|42|28.144049496685|58.124037680865|0.722592608425|1.492322560225|基準内|

n=2..42をすべて拒否し、最小n=43を得た。分散の値に依存しない設計なので両条件とも43本。[計算結果](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/sample_size_design.json)、[算術SOURCE](../../scripts/h8_rare_c1_contract.py)。正のgamma級数と、exp/erfからの半整数漸化式という別経路で全判定を照合し、逆関数も照合した。binary64 CDF差の受入は5e-13、分位点二分法は相対1e-12。これは数値照合でありinterval certificateではない。

**90%は各条件の計画水準**。2条件同時90%を設計したとは言わない。実pilotでは正規性の適合検定や追加fitを自動実行せず、取得したmean/variance/SEは経験的推定として表示する。重い尾・非正規性・分散0ではη精度を主張しない。本数増量、外れ値除去、結果依存の停止・やり直しも行わない。

## 凍結した科学call予算

各条件独立one-shot。cacheは今回は無効とし、成功した全43本は測定付きfull Hadamard wrapperのcosine/sineを各1 compileする。evolutionは同じevent列から各trajectory1回。両軸は同event・同evolutionを使う。

|call|p8/q2|p7/q16|合計上限|
|---|---:|---:|---:|
|保存input/preparation decode|1|1|2|
|純粋prefix復元|1|1|2|
|元law形成|1|1|2|
|canonical対応の数値law照合|1|1|2|
|S2_SINGLE count表準備|1|1|2|
|component表準備|1|1|2|
|trajectory|43|43|86|
|occurrence|86|688|774|
|RTE event|344|2752|3096|
|native evolution|43|43|86|
|測定full wrapper build|86|86|172|
|実compile（cache無効）|86|86|172|
|control probe（4 statevector simulationを含む）|4|4|8|
|algebraic event oracle|1|1|2|
|非測定validation wrapper|2|2|4|
|paired pilot会計|1|1|2|
|別経路の数値moment検査|1|1|2|
|新signal/SCF/DF/state solver/Gaussian生成|0|0|0|

予算は階層的なcallである。controlの4 simulationはcontrol_probeの内数で、さらに8回を加算しない。event_oracleは固定関数全体1回で、保存stateのcopy、native block-action準備、各stepのdeterministic作用、明示event作用（q×r=8/64）、full-Fock scalarと内積を含む。これは新signal推定ではないが、科学的なvector計算を含むため**将来の実行承認書に明記する**。固定元関数以外のvalidation、追加oracle/compileによる再確認は予算外。

validation代表は両条件とも、結果を見る前に登録した`S2_SINGLE replica000`の1本。native control toleranceは元の1e-8。代表を先に完了・回収・control確認し、その後残42本を投入する手順を契約化する。科学callは呼出し前に非返還予約し、失敗・中断・部分結果でも消費を戻さない。予算未満でSTOPしたものを成功86本と数えない。

## SOURCE・入力・compiler/runtimeと数値規約

新SOURCEの[ファイルSHA](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/SOURCE_manifest.json)を保存。元科学SOURCE745a6745、元helper固定Git・provider c31ef424、原profile補足5f267e96、段階B3d5ab5bbを参照継承する。Hamiltonian/state/snapshot/preparation/compiler/profileのSHA、analytic shotsとPFは条件別契約へ転記した。

|条件|preparation SHA先頭|cosine/sine shots|固定PF|
|---|---|---|---|
|p8/q2|52bdd60b64c3823f|7659 / 8832|B0I_p11_q1_2nd|
|p7/q16|f794c14366f60812|79863816 / 99933856|B0I_p15_q4_4th|

元Qiskit1.3.0、basis rz/sx/x/cx、opt1、transpiler seed17、symmetric_directional control、identity/full-Fock scalar・ancilla相対位相・state preparation費用外を維持する。原exportは完成prefix完全一致ではないため、元の純粋復元でp8/p7 preparation hashを必ず検査する。SCF/DF/state solverやGaussian再生成へfallbackしない。max_instructions=1,000,000は元実装guardでありcompiled費用上界ではない。

検査初回に日本語ファイル名のGit引用表記を誤比較したため、NUL区切りへ修正した新SOURCE bc1a77a8で再結合した。[修正来歴](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/technical_correction.json)。旧SOURCE d1da0ea9・未使用registry/draftは保全、科学call/有効grantは前後とも0。

今回read-onlyでruntime34273、native provider265、helper21、原P931閉包を両scope、旧運用/input87、元nativeconfig等の**重複を除く35,292 physical file**を照合した。[照合記録](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/readonly_provenance.json)。現Python executableは新たにobserved hashを固定したが、過去にexecutable hashがなかったため歴史的完全同一性とは言わない。実package import、loader、実library観測は0。private locator・seed・rawは公開packetに含めない。

数値候補IDはC-0.5と同じ`EXACT_INTEGER_CONDITIONING_OF_STORED_BINARY64_V1_PROPOSED`。保存binary64の厳密整数比、整数CDF、元componentのbinary64除算後の整数比、PCG64 raw-word規則、全支持・元event係数を維持する。候補選択済みだが、実行承認も実データ整合も未取得なので`numeric_model_approved=false`のままにした。

[保存law転記](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/saved_law_bindings.json)には元law dict、τ、canonical p/B、保存log B、回収λ_Rとその制約をそのまま保存した。hash一致を構造確認しただけで、確率差・希少質量・期待費用を再計算していない。回収λ_Rは原order0係数由来であり、独立full-tail norm auditではない。

別途科学承認後、**最初のtrajectory抽選より前**に、元law hashとq/r/K/T/τ/normalization、正の次数・component支持を確認し、保存binary64の有理数正規化pとB canonical保存文字列の最大絶対差≤1e-12を確認する。C-0.5の1回law_compatibility callを使う。これはengineering整合gateで、canonical期待費用のbias上界・旧NumPy choiceのbit列再現・母平均保証ではない。不整合ならgrant/予約を保全してSTOPし、許容値やlawをその場で変更しない。

## seed・grant・one-shot

SOURCE/input/preparation/law binding・固定epoch・candidate/stratum/phase/replica/streamからC-0.5の規約でprivate registryを作成した。43本×(order+q component stream)で129/731、計860 stream seedと、2個のlaw root seed metadataを封印。旧除外5960値との衝突0、新旧/新新衝突0。これはSHAによるregistry metadata作成でありRNG drawではない。実seed値は保存先privateのみ。[receipt](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/seed_receipt.json)。旧registryからのreuse、worker順/PID/時計によるseed変更、retry用seed更新を禁止する。

[実行制御schema](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/execution_control_schema.json)と条件別の**無効承認書案**を保存した。有効grant・ledger・one-shot markerは今回作成していない。将来、ユーザーの明示的な実科学承認receiptと完成contract SHA/SOURCE/budget/private閉包を結び、新規condition-specific grantを1つずつ発行する。markerは科学load/law以前にO_EXCLで消費し、存在時は再起動拒否。旧消費済みgrantは使わない。load/law/control/compile失敗後も非返還、retry/resumeなし。

条件はcoarse→tightの順。coarseが失敗したら全pilot STOPし、tightへ自動進行しない。coarse成功後だけ、別の未消費tight grantのgateを行う。条件間同時実行を今回の契約に加えない。条件内はtrajectory/両軸並列・COW共有を使う。これは科学的な開始順・失敗境界の固定で、CPU/worker数の低い上限ではない。

## 実行前・実行中gateと残る技術事項

|gate|今回の状態|将来の確認と失敗時|
|---|---|---|
|n/scope/budget/保存binding|固定・構造PASS|SOURCE/入力SHA変更でSTOP|
|ユーザーの実科学承認|未取得|現在の指示は契約専用。draft falseでは起動拒否|
|canonical数値整合|NOT_RUN|承認したload/law後、抽選前に1回照合。不整合STOP|
|実import・SessionEquivalenceLibrary|未確認|元のisolated import recipeと実module file/extensionを照合。libraryのgate定義、parameter/global phase、挿入順をcanonical化しSHA封印。出所不明の現状態を黙って採用しない|
|fork安全性|mockのみPASS|import前の技術probeに加え、loader後・各fork直前に自分の全native thread/task、BLAS等のfork条件、親子library identityを確認。unsafeならSTOP、spawnやthread policyへ自動変更しない|
|private memory/thread/IO|実H8 footprint未確認|両軸・control/oracleを含む根拠あるpeakと実thread/IOをfreshな空き資源へ照合。未確認を0や小値で補わずSTOP|
|実行entry point/manifest|C-0.5 APIは固定、将来の実結合は未完|代表先行barrier、2 batchのreceipt統合・一度だけの最終会計、pre/post-loader gateを結合し封印。C-0.5 run_registered単独を2batch driverと扱わない。新entry point追加時は新SOURCE/closureとして記録・検査し、未検査のcodeを今回のSOURCEと称さない|
|cache|今回は無効|172 compile予算を固定。後から結果依存で有効化しない|
|native終了・保全|mock回収とread-only SHA PASS|own PID/start/UID/PGID、wait4 exit、子孫回収・原bytes一致がなければ科学結果をPASSにしない|

C-0.5 begin_approvedのverified flagだけではpost-loader安全性を証明できない。将来driverはpre-importの許可gateとpost-loaderの実安全gateを区別する。marker後にpost-loader gateが失敗した場合も消費を維持する。現在の契約は科学範囲とgateを確定したもので、**実分子launcherの完成・検証済み、即起動可能とは主張しない**。

[今回の資源観測](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/resource_observation.json)も保存したが、H8 footprintや将来の空き容量保証ではない。資源はfresh affinity/physical core idle、有効quota/cgroup memory、IO/PSI、親入力RSS/PSSとworker private peakをread-only観測し動的admissionする。代表1本で負荷を測る場合も、その1本は43本と予算内であり、別benchmarkや捨て標本を追加しない。代表を安全に開始できる根拠がなければSTOP。残42本は実測を使い並列化する。固定の低いCPU/RAM/worker/thread/wall capは設けない。一時PSI上昇・監視遅延だけで正常workerをkill/restartしない。OOM/入力破損/数値不一致/継続不能では部分結果を保全してSTOPする。

自分のowned processだけを回収し、他worktreeの通常HEAD/status更新を拒否条件にしない。共有環境・他ユーザー・別4次PF研究の設定やprocessを変更しない。session library/bytesの不一致を共有packageの上書きや環境の無断変更で直さない。

## 出力・検証・終了

pilotの`G_i=Ncos*RZcos+Nsin*RZsin`は同一event両軸の値を結合する。整数/Fractionでmean・不偏variance、Decimal100でSE。別実装raw整数momentでmean/varianceを照合し、両軸・trajectory/occurrence/event件数、count層、全43 IDを検査する。loadは実時間、CPU、RSS/PSS/private、並列度・cache無効、native exitと回収receiptとして記録。mean/variance/SEの値も、未知層の費用上界にはならない。[出力schema](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/result_schema.md)。

[今回のpreflight](../../artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/preflight.json)は`PASS_CONTRACT_TECHNICAL_PREFLIGHT_SCIENCE_BLOCKED`。新11テスト＋既存55 toy/mock、9 mock native receiptの所有子/孫回収、元35,292 byte保全と構造/SHAを検査した。著者自身の別経路検査で、別AIによる独立科学reviewではない。新H8 sampling/law/simulation/circuit/compile/保存科学再解析、有効grant発行/消費はすべて0。標本設計χ²算術とregistry metadata以外の実科学数値解析は0。

専用branch `codex/h8-rare-rte-c1-pilot-contract-20261021`にSOURCE・契約・監査をローカルcommitする。pushは未実行・別公開承認が必要。**ユーザー**の次作業は、この43本×2と全call予算・数値gate・残る技術gateを含むC-1 pilotの実行を別途明示承認するか判断すること。**Codex**はmandatory STOP。承認前にgrant発行・実結合を用いたH8科学起動・productionへ進まない。公開範囲は実行承認と別に判断する。
