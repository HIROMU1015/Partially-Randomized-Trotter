# Track A H4 geometryと要求精度の資源map拡張設計案

原稿作成・投稿先検討はいったん保留にし、H4のgeometryを増やした全候補資源mapを次の計算候補とする。
これは利用者の新しい拡張意向を記録した**設計案**であり、固定済み実行契約・science authorizationではない。
状態は `DESIGN_PENDING_NO_SCIENCE_AUTHORIZATION`。原稿v0.1・v0.2、図、旧結果とSTOP判断は保存する。
[別JSON](track_a_geometry_precision_extension_proposal_v0.json)に未確定条件と計算量を分離した。

その後の利用者指示により、実行環境はサーバーの既存Python/venv・依存を優先する。
[GPUサーバー側への準備指示](gpu_server_track_a_h4_geometry_resource_preparation_prompt.md)に環境inventory・CPU benchmark・契約草案までを依頼した。
旧local versionとの完全一致は要件にせず、新campaignの環境を固定し、compiler差のある旧証拠は別layerとして扱う。
本計算は引き続き未認可で、距離の組も未確定。以下の初期環境案はこの方針で具体化する。

## 今回先に調べる問い

固定した二次DF-prefix PFとcanonical finite-RTEの実装で、discard・deterministic・partial・random-dominantを
同じ有限coherent-signal精度で比較したとき、geometryと要求精度によって資源競争力がどう変わるかを調べる。
partialが勝つことは成功条件にせず、消える領域・accuracy不適格な領域・不確実な境界も保存する。

H4の複数geometryは適用条件の証拠を増やすが、新手法の新規性や投稿可能性を自動的に保証しない。
先の `CLOSE_AS_TECHNICAL_RESOURCE_NOTE` reviewは当時の証拠に対する判断として残す。
その後の原稿v0.2と編集判断の別bundleはローカルに保持し、今回の公開対象には含めない。
H6/H8による系サイズ拡張、energy/RPE/QPE総cost、strong synthesis、高次PF、Track B統合は今回に混ぜない。

## geometryとモデルの未確定条件

追加6点の候補は隣接H–H距離 **0.70、0.80、0.90、1.10、1.40、1.60 Å**。
圧縮側から伸長側まで含める設計提案であり、利用者の確認をまだ受けていない。8点案なら距離を別途固定する。
これらには過去のP-C等で使用したgeometryが含まれるため、fresh blind検証とは呼ばない。
新しいfull-grid resource studyとして、結果を見る前に距離と候補集合を固定する。

H4 linear、STO-3G、8 system qubits、explicit DF rank12、T=0.8を第一案とする。
「H4だから実rankが常に12」は仮定せず、各snapshotのrequested/actual rank・fragment列・物理sectorを検査する。
rank不足やSCF/参照状態の不収束を、padding・別rank・別geometryへの差し替えで救済しない。
DFの並びと位相規約もsnapshotへ保存し、結果に応じてprefixを並べ替えない。

新geometryではSCF・積分・DF構築と、rank12 DF Hamiltonianの同じ物理sector内の最低固有状態の生成が必要になる。
既存S0のreference state・global phase・残差検査を監査し、新しい生成契約に明示する。
比較対象signalはそのDF参照状態の `exp(-i E_DF T)` であり、未切断分子Hamiltonianの厳密chemistry energyとは区別する。
今回はその構築・solve・snapshotアクセスを一切実行していない。

既存1.00 Åは218候補の保存証拠を持つが、1.30 ÅのM2は固定5構成だけである。
1.30 Åを218候補の完全mapとして数えたり、未測定213候補を補間したりしない。
既存二点を参考表示する場合もdomainを区別する。1.30 Åの全候補化は別途明示予算に含める必要がある。

## 全218候補を同じtemplate集合にする

| method | 固定template | 候補数 |
|---|---|---:|
| B0 discard | L_D=3,4,5,6,9、q=1,2,4,8、r=K=0 | 20 |
| B1 deterministic | L_D=12、q=1,2,4,8、r=K=0 | 4 |
| B2 partial | L_D=3,6,9、q=1,2,4,8、r=1,2,4,8,16,32、K=2,4 | 144 |
| B3 random-dominant | L_D=0、同じq/r/K grid | 48 |
| 既登録r64境界 | B2-rank3-q1-r64-K2 と B3-rank0-q8-r64-K2 | 2 |
| 合計 | random194＋deterministic/discard24 | 218 |

ここで引き継ぐのは `method,L_D,q,r,K` の集合であり、development snapshotに結合された旧fingerprintではない。
各geometryのHamiltonian・DF・状態identityを含む新しいcandidate fingerprintを生成する。
旧r64境界の選抜規則を各geometryで再発火させず、上記二templateを初めから含める。
旧16-cell selectorを使わず、追加・除外・有利な候補への置換を行わない。

新geometryでも194 random全件がaccuracy適格とは仮定しない。
完全map案ではineligible候補も登録とcompile監査に残すが、matched-accuracy resourceはnullとし、0やwinnerにしない。
実装gate failureと科学的なaccuracy不適格を別statusで記録する。

## 計算量

各random候補は32 classical trajectoryを使い、cosine/sineで同じevolutionを共有する。
二軸は別wrapper identityとし、量子shotは実行せず解析式から必要数を算出する。

| 追加geometry数 | random wrapper | deterministic/discard wrapper | 合計wrapper record |
|---|---:|---:|---:|
| 1 | 12,416 | 48 | 12,464 |
| 6 | 74,496 | 288 | 74,784 |
| 8 | 99,328 | 384 | 99,712 |

これはfull-wrapper recordの上限案であり、同一cell内の厳密identity一致cacheで減るactual transpile数とは区別する。
異なるgeometry/cell間の角度差無視・partial cache流用は行わない。
追加96、adaptive sampling、別seed救済、新しいr/K拡張は含めない。

[保存済みM1-B1結果](../../artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json)では、
6 worker・12,448 recordのcompile段階が21,021.267秒、約5.84時間だった。
同程度の1geometry処理時間という**計画上の仮定**なら6点約35時間、8点約47時間となる。
新geometryの回路複雑さ・signal・snapshot生成・共有負荷は未測定であり、これは完了時刻の保証ではない。

## ローカルとサーバーの役割

第一候補はサーバーの**CPU**で長いcompileを行い、ローカルで契約・tests・保存値解析・図を準備する構成。
GPUを使う前提ではない。現在のローカルはCore i9-13900、32 logical CPU、RAM約62 GiB。
P/E coreを同等の32 physical coreと数えない。

以前のサーバー報告はEPYC7742・128 physical/online core・約1 TiB RAMだったが、現在の空き資源は未確認。
その `NO_MATERIAL_SERVER_ADVANTAGE` は進行中runを捨ててfresh再実行する移行判断であり、
今回の新規6〜8点キャンペーンの否定材料にはならない。
一方、以前のnative-basis synthetic benchmarkはローカルとの同一fixture比較がなく、
高level controlled gateの分解・builder負荷を再現していないため、実wrapperの短縮率は確定できない。

サーバー側で科学データを使わないfixtureのbenchmarkを先行し、同じfixtureをローカルでも再利用可能にする。
ローカルとの同一fixture実測がない間はserver内worker scalingだけを報告し、host間の速度倍率は断定しない。
9 qubit・1 classical bitの高level controlled Gaussian/Givens類の完全synthetic列を設計し、
build/transpile時間、1/6/12/16 workerのthroughput、RSS、共有負荷、failureを記録する。
研究Hamiltonian、candidate record、trajectory seed、snapshotは読まない。
benchmark上限案は各host128 transpile以下・総wall30分以下。危険なworker条件は理由を残して未実行にする。

[Qiskit1.3 compiler文書](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/compiler)には
複数回路のmultiprocessingが記載されている。外側worker poolと二重並列にせず、
各workerのBLASと内部process/thread設定を固定・監査する。
最大16 workerは比較候補であって認可済み上限ではない。benchmark後にsafeなcapを決める。

Qiskit versionだけでなくPython・NumPy・SciPy・OpenFermion・PySCF・rustworkx・生成sourceを固定する。
既報のサーバーPython3.12.3は旧local3.11.0rc1と異なる。
新しい環境はサーバーの既存versionを優先し、旧実行との完全identity一致を要求しない。
数学的意味論・測定wrapperは維持し、純synthetic照合と新sourceへの互換化で適合させる。
compiler version/policyが異なる旧costは同条件のmapへ直接混ぜず、別layerで表示する。
直接比較用の新環境1.00 Å anchorが必要なら12,464 wrapper追加を提案し、別認可なしに再評価しない。
本設計でpackageのinstall/upgrade、旧runtimeのcopy、サーバーlaunchを行わない。

## 実行の順序とcheckpoint

1. 距離・モデル・reference state・worker cap・評価規則を契約として固定する。
2. 新しいscience module、runner、schema、synthetic testsを作り、source commitを先に固定する。
3. sourceに結合したseed/key/予算/固定root/outputのplanと、別result-prior authorizationを作る。
4. 独立実行前reviewと利用者の明示launch後だけ、新geometry snapshotを生成・hash固定する。
5. 各geometryの218 signalを一巡し、32 trajectoryの全候補compileを行う。
6. 保存bias/normalization/軸別costだけから要求精度依存のresource mapを作り、必ずSTOPして研究判断に戻る。

既存M1/M2/PM-1 runnerはsnapshot・source・workerに結合されているため、そのauthorizationを書き換えて使い回さない。
追加sourceはTrack Aの新しいpathへ置き、旧sourceとevidenceはそのまま残す。

checkpoint再利用はgeometry、H/DF/state hash、candidate、axis、trajectory index/seed、
compiler/dependency identity、source commit、wrapper semanticsが全て一致する完成recordだけに限る。
atomic保存、予約中と完了の区別、実compile数のledgerを新しい契約へ入れる。
未解決予約の自動retry・欠測値の推定はしない。中断時の継続可否も実行前に定め、旧no-resume規則を黙って解除しない。

## 保存値からのprecision解析と終了

第一案はPM-2と同じε=0.005〜0.1の表示grid、軸別α=0.025、bias・normalization補正付きshot式。
primaryは `N_real E[C_cosine,RZ] + N_imag E[C_sine,RZ]`、
secondaryはRZ/CX count・depth、total depth、sizeの6指標Paretoと共通P>=0感度。
精度ごとの新signal、sampling、compileは不要であり、表示点を独立実験と数えない。

候補domain内のpoint最小、method間差、bias適格境界、shotと1-shot費用の寄与、
MC uncertaintyと未評価領域を分けて示す。±2SEはengineering intervalで、formal CIではない。
geometry離散点の間を連続した厳密境界と呼ばず、ε受理境界を原理的精度限界と呼ばない。
境界表示・materiality・uncertaintyの機械規則はsource実装前に固定し、結果後にGO閾値を調整しない。

成功terminal案は `GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEW`、実装失敗は `IMPLEMENTATION_GATE_FAILED`。
どちらもmandatory STOP、next-stage=false、研究判断の自動分類なし。
H6/H8、追加96、別basis、別PF、高次PF、強いsynthesis、energy/RPE接続へ自動で進まない。

## この準備で行ったこと

利用者の添付案と保存JSON・source・現行文書を確認し、候補とwrapper数を照合した。
科学入力のresolve/stat/hash/load、分子計算・reference solve・新signal・sampling・circuit build・transpile・GPUは0。
旧runtime/checkpoint操作、科学tests、環境変更、commit/pushも0。
現段階で距離・実行host・worker・新source/authorizationは未固定。これは成果物ではなく実行準備の案である。

準備照合はPASS。保存済み[候補台帳](../../artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/candidate_inventory_v1.json)の
218 templateとの完全一致、wrapper上限、保存入力4件のhash/commit blob一致、46 frozen evidence filesの不変性、
v0.2 manifestの6件、文書のlocal links、git diff whitespaceを確認した。科学testsの再実行結果ではない。
