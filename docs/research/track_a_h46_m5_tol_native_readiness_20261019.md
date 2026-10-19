# H4/H6 m5_best：tol-only保存入力adapterと4lane native実行準備

2026-10-19。**技術準備完了・実行未承認・mandatory STOP。** [計画差分](track_a_h46_m5_protocol_alignment_20261019.md)に沿い、H4の保存tol-only入力接続と新4契約を封印した。実H4/H6の配列復元、sampling、signal、compile、SCF/DF/state生成、保存値の数値再解析は0。有効grantは0。m5_bestの主比較採用（AD03）と後続Coreの実行は保留のまま。

科学実行用SOURCEは `c7d53f04c3be131078ee91e8bef3c78ca434ed3e`。作業branchは `track-a-h46-m5-tol-native-readiness-20261019`。資料入口：[summary](../../artifacts/resource_applicability/track_a_h46_m5_tol_native_readiness/2026-10-19/summary.json)、[入力](../../artifacts/resource_applicability/track_a_h46_m5_tol_native_readiness/2026-10-19/input_manifest.json)、[closure](../../artifacts/resource_applicability/track_a_h46_m5_tol_native_readiness/2026-10-19/source_runtime_closure.json)、[検査](../../artifacts/resource_applicability/track_a_h46_m5_tol_native_readiness/2026-10-19/validation_summary.json)。

## 1. 再利用と新adapter

旧SOURCE `7a4d126ad2df622873aebdb372487c8fd956de4c` のm5係数、11段signed S2合成、Gaussian、scalar一回、occupation oracle、full-wrapper・paired compile、fingerprint処理は保全した。新しい `h46_m5_tol_native.py` は旧関数のcode objectを独立したglobalsへ束縛して再利用し、旧moduleの定数・設定を書き換えない。新policyは `M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1`、候補IDとscopeに入力系列を含める。

H4の保存復元はF/FP `97d25cc431516d720ce91c7cb4f148e061d8ea34` の `h4_tol_only_input.py` から、`array_record` と `load` の関数本体だけをそのまま固定した。AST一致を人工検査し、SCF/DF/state生成の関数は新moduleへ持ち込まない。H6は既存保存loaderを維持する。実preparation生成・reference作用は実行承認後のscience preflightでのみ行う。

新contractは入力role・series・rank policy・Hamiltonian/state/snapshot hashを固定する。H4 rank10は保存分解の観測値であり、新生成のfinal_rank指定ではない。H4 legacy12の旧binding・snapshot・D/P・grantは混ぜない。全SCF/MO/積分/DF/state規則がH4/H6で一致するという認定もしていない。

## 2. 保存入力・runtime・封印

H4はFP configのcanonical digestと固定公開FP_bindingを照合し、Fのexit0/final gate PASS、保存tol-only receiptのSHA、NPZの全bytes/SHAを確認した。**FP_bindingの新入力**を使い、同bundleのlegacy input_bindingを使っていない。H6も旧m5保存入力の全bytes/SHAを確認した。配列としては開かず、新signal・statevector・bias計算をしていない。Fの元providerと新m5 providerのDF class・Hamiltonian hash・state hashを含む3つのSOURCE fileも全bytes一致を確認した（saved_loader_provider_compatibility.json）。この静的互換性確認だけで全入力生成規則の同一性を認定しない。

SOURCE16 file、provider278 file、package/Python runtime11 fileとseed metadataを結合した。各契約の封印はH4が309 file、H6が310 file。Python/runtimeは既存m5専用環境をそのまま使用し、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・symmetric-directionalを維持する。OS、共有package、他ユーザーの環境は変更していない。

旧main profileは参照情報で、新policyとの費用同一性は未認定。旧ordinary高次PRや旧main model/compiled費用を同じ単価へ無条件に接続しない。旧profile bytesの全同一性や全入力生成規則の未確認も、この封印で解消したと表示しない。

初回SOURCE `a6c19ba98bb1931031f0b9c81138c02d550867b1` の未承認proposal01は保全した。その後、configが実際に使う入力・seed・SOURCE pathの全てを封印一覧へ結合する検査を補強し、別SOURCE/別proposal02で再sealした。初回に科学計算・有効grant・markerはない。旧proposalを上書き・再利用・遡及PASSしていない。

## 3. 固定technical pilotの範囲

対象は線形H4/H6、隣接1 Å、STO-3G、tol-only1e-8、T=.8、保存Hamiltonian/state、P=0。公式はS2/Yoshida4/`4th(m5_best)`、m5係数版は `PF_M5_BEST_20D64C2_BINARY64_V1`。新係数最適化はしない。

|新契約|内容|最大軸|技術PR本数|reference枠|
|---|---|---:|---:|---:|
|H4_TOL_PILOT_DET|rank10、p3/full10、q2、3公式×full/B0/B0I|18|0|1|
|H4_TOL_PILOT_PR|rank10、p3、q2、3公式、K4、signed piece当たりr2|12|6|0（共有）|
|H6_TOL_PILOT_DET|rank19、p5/full19、q2、3公式×full/B0/B0I|18|0|1|
|H6_TOL_PILOT_PR|rank19、p5、q2、3公式、K4、signed piece当たりr2|12|6|0（共有）|

合計30 evolution task、60 measured wrapper軸、PR12 trajectory、full-H reference2 job。referenceは系ごと一回だけ、DET契約の予算に入り、両laneが同じhash付き作用を使う。各代表sample0でsignal/controlの確認を行い、PR sample1は同じ群の代表PASS後に投入する。正負時間と元finite-RTEの無条件lawを保ち、scalarは全piece後に一回。B0は元constant、B0I/PRは元constant＋tail identity。referenceの位相を合わせ直さない。

native correctnessは既存と同じfiniteかつ1e-9未満、両軸のHadamardとcompile前後・occupation oracleを照合する。この値をshot用allowanceへ流用しない。shot allowance/headroom適格性はnull、signal診断は `EMPIRICAL_DIAGNOSTICS_ONLY_NOT_ADOPTED_FOR_SHOTS`。費用winner選択・5ε会計・独立確認Pではない。model fit/2%検証も今回のpilotには含めない。

## 4. 新plan SHAと承認境界

|契約|封印したplan SHA256|
|---|---|
|H4_TOL_PILOT_DET|`d9908a4a3238834a10fbed9090b96ebf59dcb8fb2f00a8af4c52866870f77af0`|
|H4_TOL_PILOT_PR|`2b0bb77dea0e616e8aba59d0a4971be58ae4427ec5fa34e3177b9e1f53c76072`|
|H6_TOL_PILOT_DET|`1ffd4f0d072f94edb6b5e901e58584577d464f6b8596e5f0172b3b838a746470`|
|H6_TOL_PILOT_PR|`d9b6cfdd384d649eb631cde70fa5eb02a9c1e15a7fc562e02473536df9c76ba7`|

公開bundleは4契約のscope/digestと無効grantの投影だけ。private config/plan/seed/入力mapping原本は `/home/AbeHiromu/projects/h46-m5-tol-readiness/20261019/private/proposal02/` に保存した。`future_run01` は存在せず、実行output/ledger/markerも0。grantは全てvalid=false、grant_id/approval_reference=nullである。

暫定seedは120、既知除外は4,483、衝突0。旧高次PF・旧m5・未使用tol proposal01のseedも除外した。未知の外部jobや別チャットの全seedを網羅したとは主張しない。後続の独立確認では別namespace・新cohortとし、技術pilotの標本を混ぜない。

## 5. 並列化・fresh gate・STOP

新launcherは既存のround-robin/spawn pool、代表先行、lane別STOP、wait/reapを再利用する。変更はnew contract/nativeへの接続と、DET reference laneを新scopeから取得する箇所だけ。H4/H6・DET/PRの4laneを並行し、各lane内も候補/trajectoryで並列化する。evolutionをcos/sin間で共有し、reference/preparation/spectraとcanonical fingerprintをprocess内でcacheする。cosine確認後にsineへ進む。

workerは実行直前の有効affinity・idle CPU・空きRAMと各taskのRSS計画proxyから決め、現在は未割当。旧4/32/60 workerをcapにしない。RSS proxyは旧Yoshida由来の計画推定で、m5実測上界ではない。内側BLAS/transpilerはown process内1threadとして過剰なnested並列を避ける。fresh資源gateは承認後に実行し、今回の観測を将来の起動PASSへ読み替えない。既存OS/cgroup制御は変更・回避しない。

SOURCE/input/compiler/runtime/seal・grant・新output/marker・seedの一致を起動前後に照合する。実際に使うfileだけを拒否gateへ結合し、無関係worktreeのHEAD/statusを凍結しない。PSI・CPU競合だけでkillせず、実OOM/継続不能な不足、own process異常、native不一致、nonfinite、identity破損は対象scopeをSTOPする。共通reference失敗は依存する同一系の両laneを止めるが、無関係系の正常laneへ伝播させない。

旧grant/ledger/marker/STOPは保持し、予算融通・自動retry・再fit・追加compileなし。各lane終了後mandatory STOP。後続124 signal候補案、allowance採択、単価/model、5ε再会計、独立確認、機構分解、主比較採用へ自動移行しない。

## 6. 人工検証・限界

新しい16項目を人工入力で検査した。初回14件（人工native2件を含む）PASS、契約path補強後は関連するmetadata/fork14件（新2件を含む）を再検査してPASS。数値native/保存loader/launcher bytesは不変だったため、人工native2件は既検査を再利用し再実行していない。計28回を独立28項目として水増ししない。

人工8 qubit・rank10のDFと人工NPZ/receiptだけで、保存loader復元、hash改竄拒否、full/cut preparation、共有reference、m5 PRのnegative piece/scalar・control・occupation、paired2軸compile・integer ledger・結果保存までPASS。これは実分子のF入力や実H4/H6経路の成功ではない。作者自身の技術検査で、別reviewerの独立レビューでもない。

実際に封印したconfigのmetadataを複製し、SOURCE・rank・series・snapshot path/hash・role・runtime/provider・registry/exclusion・重複seal・欠落SOURCE・native policy・旧lane・control・plan SHAの16種を改変し、全て拒否した。元config/plan/receiptへは書かない。実際の無効bundleも固定PythonのCLIへ渡し、`EXPLICIT_PILOT_SAMPLING_COMPILE_APPROVAL_REQUIRED`、exit2、future root/ledger/marker0で拒否された。

## 7. 成果物・次の担当

新codeは `h46_m5_tol_contract.py`、`h46_m5_tol_native.py`、保存loader、`launch_h46_m5_tol_pilot.py`、`prepare_h46_m5_tol_pilot.py`、固定入力参照、人工test。公開可能な資料は専用bundleへ保存し、原NPZ・seed値・private grant/path一覧は公開資料に含めない。既存H4 legacy、旧m5 SOURCE/4契約、V1/V2、H5/H7、別担当のH6新A確認とA82は変更していない。

**次のユーザー判断**：SOURCEと新4契約に対する「H4/H6 tol-only technical pilot、最大60軸・PR12本・reference2 jobの一回実行」を承認するか判断する。m5の主比較採用や後続Coreの承認とは分離する。

**Codex（承認後の次作業）**：最新の既知seed receipt、SOURCE/input/runtime・資源・保全・未使用rootのfresh gateを通過した場合だけ有効grantを結合して一回起動し、結果・native exit・全worker回収を保存してSTOPする。現時点では未実施。

今回の作業はlocal commit後mandatory STOP。公開pushと新科学計算は行わない。
