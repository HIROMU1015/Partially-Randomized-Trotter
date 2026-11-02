# GPUサーバー側への依頼 H4資源mapの正式契約準備

## 今回の依頼と停止位置

公開済みserver preparationを基に、Track A H4 geometryと要求精度のresource mapの
正式契約案・schema・zero-compute plan・契約検査を準備してください。
利用者は追加6距離と最大12 CPU workersを契約条件として確認しました。
この確認は本計算、science source port、execution authorizationの作成や実行の認可ではありません。

今回は、数値回路identityを欠くcheckpoint草案を修正し、未固定の生成・資源・判定規則を明文化する段階です。
**分子snapshotを開く／生成すること、science runnerの実装／run／resume、signal・trajectory・build・compileは行わないでください。**
新しいbenchmarkやsynthetic transpileも不要です。契約検査は合成JSONと静的sourceだけで行い、報告後STOPしてください。

元の原稿・科学結果・STOPと公開準備草案は変更せず、今回の資料を別pathへ追加します。
原稿作成は引き続き保留。Track Bには触れません。

## 固定identityと資料

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 公開準備commit：`c2ab34fed49bb1fb104d39fe83b36858a2c92c2a`。
- 準備branch：`track-a-h4-geometry-resource-server-prep-20261005`。
- 元handoff commit：`48f9ac3b756bcf572aeb7c594a0c88791a458725`。
- 旧証拠commit：`4c23453c541700c6a41ba71fc5ec9323b53858d6`。
- 今回の利用者確認を記録した[scope JSON](track_a_h4_geometry_contract_preparation_scope_v1.json)。
  未commitの文書を渡された場合は全文をhandoffとして保存し、公開済みcommitだと仮定しません。

公開準備資料のprefixは
`artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05/`です。
同commitのREADME、report、environment inventory、dependency closure、static audit、planと二つのschema、
checks/final auditを読み、以下を照合してください。

| 公開準備artifact | SHA-256またはfingerprint |
|---|---|
| preparation artifact manifest SHA-256 | `33d8a7a5986bd4a29cbe7d678874d54d859614bfab8643b88de2e9f613651741` |
| zero-compute draft plan SHA-256 | `6b535440ccca73ed826f809382e629fb54cd043a7bd4332db01fdad133b177d3` |
| 同plan fingerprint | `fd0dda4e5f33a75f6b66d13f8d04a148da2e0d446a52d570d718106b4e5e4f53` |
| template集合fingerprint | `a939f5602a7840d92d93dfa9be40db40af52dede4483091c649413f480ef03c4` |
| future checkpoint draft schema SHA-256 | `1567533226a11ee0c1b2611709570cc4fdda8c99d8b57740406380281fa3db80` |
| environment fingerprint | `7d397d409aa05efac027463b671cfe5608f554e09e464fa1e87eed1965b41cd1` |
| compiler fingerprint | `46ce3bcc45cbb53dbe021ca67e4253156597fa5ae9c034cfeb4cddfe6828f30d` |

origin ownerとcommit系譜を確認し、上記準備commitを起点にします。追従branch tipを使いません。
別branch/worktreeの推奨名は `track-a-h4-geometry-contract-20261006`。同名を上書きしません。
新worktreeを作る場合は前回と同じworktree限定のsparse/no-checkout手順で分子snapshot・runtimeを除外し、
通常の全面checkoutによる科学入力のmaterializeも避けます。共有Git設定・既存worktreeは変更しません。
この指示ではcommit/pushを依頼していません。後続の明示指示までlocal資料として報告してください。

最初にAGENTS、PROJECT_MAP、研究概要を読みます。旧normative記述は当時の状態として保持し、
このhandoffで確認された条件と科学未認可を新資料・研究概要・dated noteへ記録してください。
旧validation manifest/result/statusや247 sourceを編集しません。

## 利用者が確認した距離と資源上限

追加geometryはH4 linearの隣接H–H距離
**0.70、0.80、0.90、1.10、1.40、1.60 Åの6点だけ**です。
canonical distanceは上記二桁小数のstringとし、単位・座標生成規則を固定してください。
これらには過去の別研究で用いたgeometryを含むため、fresh blind/未使用held-outとは呼びません。

H4、STO-3G、requested/actual DF rank12、8 system qubits、T=0.8、
二次DF-prefix PF、canonical finite-RTE、delta=T/qを契約のモデルscopeとします。
参照は各geometryのDF Hamiltonianの固定物理sector内の最低固有状態であり、
未切断化学Hamiltonianのexact energyやchemical-accuracy QPEではありません。

各geometryは保存inventoryと同じ218 templateです。

| method | template | count |
|---|---|---:|
| B0 discard | L_D=3/4/5/6/9、q=1/2/4/8、r=K=0 | 20 |
| B1 deterministic | L_D=12、q=1/2/4/8、r=K=0 | 4 |
| B2 partial | L_D=3/6/9、q=1/2/4/8、r=1/2/4/8/16/32、K=2/4 | 144 |
| B3 random-dominant | L_D=0、同じq/r/K | 48 |
| 既登録r64 | B2-rank3-q1-r64-K2、B3-rank0-q8-r64-K2 | 2 |
| 合計 | B2/B3 random194、B0/B1 baseline24 | 218 |

旧selector/r64選抜を再実行せず、候補追加・除外・置換をしません。
保存inventoryのtemplate IDとparameter集合を照合し、旧geometry-bound candidate fingerprintはコピーしません。
accuracy-ineligibleも登録・compile監査に残し、shots/matched-workをnullとします。欠測を0で埋めません。

将来の本計算の契約上限は以下です。今回これを消費しません。

- 各random cellは32 trajectories。Re/Imは同trajectory、wrapper keyはaxis別。
- 1点のrandom wrappersは194×32×2=12,416、baselineは24×2=48、計12,464。
- 6点のsignal recordsは1,308、logical wrapper records上限は**74,784**。
- actual science transpile invocationsも74,784以下。cache reuse・reserved/ambiguous・完了を別countにする。
- 外側spawned CPU workerは**最大12**。BLAS/OMP/MKL/NumExpr/Rayon各1、Qiskit内部process 1。
- 追加96、adaptive extension、geometry/seed救済、GPU、実quantum shotsは0。

12は上限であって共有CPUの予約ではありません。科学launch前に共有資源を再確認する運用を契約へ入れます。
16 workersへ自動拡大しません。snapshot/solverの段階を含め、内外二重並列を禁止します。
science memory/wall/outputの正式上限は未固定です。syntheticのAS4 GiB/CPU600秒をそのまま採用せず、
静的な配列・回路・worker見積りから数値案と根拠を示し、未解決なら未固定としてSTOPします。

## サーバー既存環境のbinding

採用予定は既存 `/home/AbeHiromu/venvs/trotter-common/bin/python` です。
公開準備時はPython3.12.3、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、rustworkx0.17.1、
OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0でした。
metadataで現在の差を読み取り専用確認し、変化があれば公開準備との不一致を報告します。
install/upgrade/downgrade、共有shell/Qiskit/OS/CUDA/driverの変更はしません。

compilerは公開準備の全option/default/plugin identityを引き継ぐ案です。
basis rz/sx/x/cx、opt1、seed17だけの一致を「完全identity一致」と呼びません。
依存45件のmetadata closureは記録済みですが、全wheel/binaryの独立同値検査ではありません。
旧Python guardは保持し、新sourceでserver-native bindingが必要であることを契約に書きます。
**今回はそのscience sourceを実装せず、guard除去した旧runnerも起動しません。**

benchmarkは完了済み128 synthetic callsの記録だけを使います。追加transpile 0。
サーバー内scalingからローカル速度倍率や科学ETAを断定しません。

## checkpoint schemaの必要な修正

公開draftは数値full-wrapper完全一致をcache条件にしていますが、
`additionalProperties=false`のrecord schemaには数値回路fingerprintを保存するfieldがありません。
公開draftはそのまま保存し、別v1 schemaと契約に次を明示してください。

1. 各recordの `wrapper_key` と `numerical_circuit_fingerprint` を明示fieldとして持たせる。
   REGISTERED等で未buildの後者はnullを許し、COMPLETEは64桁SHA-256を必須にする。
2. wrapper keyはgeometry/H/DF/state/candidate/axis/trajectory seed/index/compiler/environment/source/
   wrapper semanticsへ結合する。deterministic recordのnull seed/index規約も統一する。
3. 数値回路fingerprintはcompiler投入前のordered full circuitを結び、全角度・global phase・
   qubit/classical-bit順序・測定axis・instruction/measurementを含める。symbolic skeletonや角度を丸めたkeyではない。
   非有限値を拒否し、floatやcomplexのcanonical encodingを結果前に明示する。
4. 完了record全体のdigestを**外部completion ledger**に保存・照合する。
   record自身のhashを同recordへ埋める自己参照形式は使わない。
5. 再利用は同geometry/candidate/axis、同source/compiler/environmentで数値回路まで完全一致した
   COMPLETE ownerだけ。trajectoryが異なって同一回路になる場合はowner linkを残し、各sampleのweightを保持する。
   cross-geometry/cell/axis、位相・角度を無視した再利用は禁止。
6. atomic record/ledger保存、reservationをcompile前に消費、未解決予約はAMBIGUOUSでSTOP。
   欠測を補完せず、automatic retry/resumeを認可しない。

契約validatorは架空recordを使い、正当なcomplete/reuseを受け入れ、
geometry/candidate/axis/seed/index/source/compiler/environment/角度/global phase/key/record digestの
改変・欠落、owner欠測、RESERVEDの再利用を拒否するtestsを作ります。
JSON schemaだけでfield間一致やowner/ledger照合を保証したとは呼ばず、意味論validatorと分けてください。
これらは契約検査用コードだけであり、science builder/compilerを呼びません。

## science sourceより前に明文化する規則

公開planの未固定項目をsource textと既存normative文書から精査し、正式契約案へ記録してください。
新規分子計算や古いNPZ/runtimeのloadによって選びません。

- snapshot生成：座標・単位、charge/spin、SCF/積分/DF algorithm・収束、requested/actual rank、
  fragment orderとtie-break、物理sector、solver、縮退時の扱い、状態phase、normalization/residual gate。
  rank不足/不収束を別rank・geometry・状態で救済しない。
- seed：campaign master seed、domain separation、canonical encoding、step/occurrence独立samplingとaxis共有。
  source/H/DF/stateへ後から一意に結合する生成規則を固定し、duplicate seed時はSTOP、別seed救済なし。
- 新snapshot・actual sourceのhashは、まだ存在しない段階ではnull。
  生成後のH/DF/state freeze barrierとsignal/cost一致gateを将来runnerの設計へ明記する。
  「契約条件固定」と「source/input-bound execution plan sealed」を区別する。
- precision：PM-2の0.005〜0.1の301幾何gridにexact 0.05を追加した302表示点、alpha_axis=0.025。
  headroom=epsilon/sqrt(2)-bias_axis、strict headroom>0、
  N_axis=ceil(2 B²/headroom² log(2/alpha_axis))を維持する。
- primaryはN_real E[C_cosine,RZ]+N_imag E[C_sine,RZ]。
  secondaryはrz_count/rz_depth/cx_count/cx_depth/total_depth/circuit_sizeと共通P>=0感度。
  paired-axis covariance、ties、null/missing、precision/geometry境界とuncertainty表示を固定する。
  点±2SEはengineering intervalでありformal CIや厳密winnerではない。
  科学GO/winner閾値を結果後に作らず、今回runnerの研究判断はnull。
- 長時間実行：fixed root/output案、one-run registry、memory/wall停止・partial ledger・中断時STOPの規則。
  shared serverの観測空きmemoryを利用予約として扱わず、runtime移送やautomatic resumeを認可しない。

未固定の科学的選択が一意に決まらない場合は、推奨案・source根拠・必要な判断を分け、
「契約完成」と偽って後続source実装へ進まないでください。

## 旧証拠layerと次段階

旧1.00 Å218候補と1.30 Å固定5構成は旧source/environment/compiler layerとして保持します。
新6点同士は同一の新identityで比較する契約です。旧costを新mapの同条件点へ直接混ぜません。
新環境1.00 Å anchor、1.30 Å全候補化、8点化は今回に含めません。
anchorの12,464 wrapperや6点+anchorの87,248を自動追加しないでください。

次の順序を保ちます。

1. 今回の契約・schema・zero-compute planをreviewし、未固定事項を閉じる。
2. 別指示で新science module/runner/testsを実装し、actual source commitを固定する。
3. source-bound planと別result-prior authorizationを作る。
4. 最終pre-execution reviewと利用者の明示launch後だけ一回実行する。
5. GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEWまたはIMPLEMENTATION_GATE_FAILEDでmandatory STOP。
   next_stage_authorized=false、automatic_research_decision_authorized=false、research_decision=null。

旧science source/authorizationは流用・編集しません。H6/H8/H12、新geometry、高次PF、strong synthesis、
energy/RPE/QPE、Track B統合、投稿判断を自動的に追加しません。

## 成果物と最終報告

別の契約準備pathへ、正式契約案、scope/plan/result/checkpoint schema、zero-compute plan、
synthetic record tests・結果、identity/access audit、file manifestとreview依頼を保存してください。
準備planではdistance scopeとworker capを固定し、science_execution_authorized=false、
execution_plan_sealed=false、actual source/snapshot identity=null、mandatory_stop=trueを保ちます。
実runtime/cache/registryを作成せず、zero-compute planのsymbolic slotを完成科学recordに見せません。

最終報告の先頭は、readyなら
`H4_GEOMETRY_CONTRACT_READY_FOR_SOURCE_REVIEW_SCIENCE_NOT_AUTHORIZED`、
未解決条件があるなら
`H4_GEOMETRY_CONTRACT_REQUIRES_DECISION_SCIENCE_NOT_AUTHORIZED`、
identity不一致なら `HANDOFF_IDENTITY_MISMATCH` とします。

commit/worktree、公開25 filesと旧source/保存証拠の不変性、6距離・218 template・74,784上限・12 workers、
schema修正とmutation tests、未固定条件、生成path/hashを示します。
科学入力アクセス、分子計算、signal/sampling/build/compile、新synthetic transpile、
旧runtime操作、GPU、共有環境変更、他job変更、commit/pushが全て0であることを報告します。

**報告後STOP。今回はscience source port、authorization作成、本計算へ進みません。**
