# Track B D1：finite-RTE development結果・GPT引き継ぎ

2026-10-13 JST。**D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW**。
固定sourceの一回実行は12 rowsすべて正常完了し、外側processもexit 0。
保存値・来歴監査PASS。**mandatory STOP、retry=0、次の科学実行は未認可。**

## Source、authorization、入力

- source S：`874ca12d347f8f35e5d8d64f65f24abc4a2df523`。
- authorization-only直接子A／実行HEAD：`f803291638d453873e4ab6f1ecf7bcc607f48e4b`。
- execution branch：`track-b-rte-development-d1-one-shot-execution-20261013`。
- 明示指示原文：「続きの作業を行って大丈夫です」。直前に提示した固定D1一回実行の承認として受領。
- [authorization receipt](rte_development_d1_execution_authorization_receipt.md)／[authorization JSON](../../../artifacts/track_b_rte_development_source/2026-10-13/v1/authorization.json)。
- [固定contract](../../../artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json) SHA256：`afa174cc7aba9ff8a14ae1a77886e50792daaf8be342ea6c1f2d3024a4e543b8`。
- authorization SHA256：`7536635537bba0b62758f016f6f47d3755943ca6a3a88e2eaad908f5502a015c`。
- [one-shot marker](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/one_shot_consumed.json) SHA256：`9c0459452b3ab3c3bb4d4f328ec042edd046a44c5b6fd0da064df9a539714e2b`。

契約・runtime・source・比較条件・capsは変更していない。
実行前にSの直接子A、authorization-only diff、remote=A、clean worktree、critical/protected hashes、runtime lock、namespace未使用を確認。
Sの`source_preparation_only`等の準備時metadataは原文のまま保持し、今回の別authorizationが登録一回実行を束縛した。
結果公開後のHEADはAの後続結果commitになる。将来のsource/authorizationと混同しない。

対象は分子/DFではない明示3-system-qubit synthetic involution provider。
`Q0=Z0, V1=R_XX01(pi/4), V2=R_XX12(pi/4)R_ZZ01(pi/4), Qi=Vi†ZiVi`、
`R_P(theta)=exp(-i theta P/2)`、`p=(1/5,3/10,1/2)`、`tau=5/7`、sigma=+、state=`000`。
DF rank、geometry、orbital basis、split L_Dは適用しない。既知providerのdevelopmentでありheld-out/独立再現ではない。
P5、r=1,2、x=tau/r、ordinary／partial-return-tail／closed P3-tail／local full／closed P5／matched CTSの六方式。
同じr内は同finite mean、r間は異なるfinite approximationで同じ物理target `exp(-i tau R)`を比較する。
r間の差をrepresentationだけに帰属しない。

## 正本と小さなレビュー入口

| 用途 | 保存資料 |
|---|---|
| 完了・外側process・STOP | [terminal](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/terminal.json)、[outer receipt](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/outer_process_receipt.json)、[STOP](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/STOP.json) |
| 全12行の原資源表 | [resource map](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/resource_map.json) |
| 正確な有理数のコンパクト表 | [JSON](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/saved_output_audit/native_resource_rows_exact_v1.json)、[CSV](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/saved_output_audit/native_resource_rows_exact_v1.csv) |
| 主比較の差と交点 | [affine relations](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/saved_output_audit/primary_affine_relations_v1.json) |
| inventoryと全sequence | [inventory](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/inventory.json)、[cache](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/synthesis_cache.json)、[sequence index](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/saved_output_audit/synthesis_sequence_index_v1.json) |
| raw row/eventと分割索引 | [原evidence manifest](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/evidence_manifest.json)、[最終inventory](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/saved_output_audit/final_evidence_inventory_v1.json) |
| 保存値・保護監査 | [saved output audit](../../../artifacts/track_b_rte_development_d1_result/2026-10-13/v1/saved_output_audit/saved_output_audit_v1.json) |

resource mapは約167 KiB、compact tableとkey/関係表は512 KiB未満。
六つのevent raw JSONは512 KiBを少し超えるため、それぞれ二つのUTF-8 `.partNNNN.txt`へlosslessに分割。
12 chunksすべて512 KiB以下で、順序結合のbytes/SHA256が原event JSONと一致した。
残る六event raw、12 row JSON、job/started/result/logは直接取得可能な小さなファイル。
原JSON・marker・terminal・STOPは変更せず、監査表を別directoryへ追加した。

## 完了coverageと共通誤差会計

12 rows、24 axes、1890 positive event bindings、全supportのnative/error coverageを保存した。
固定inventoryは36 keys：G9 v2の固定primitive19 keysをbyte一致でreuse、新規17 keysを各一回合成。
新angle inventoryは費用取得前にfreezeし、SHAをjob、started receipt、worker result、集計へ束縛した。
全新workerはKEY_COMPLETE、sequence hash・T/T†・1Q/W count・strict error recordの照合PASS。
新規上限53 keys、全cache上限72、event上限2000の範囲内。r=1への新合成・第二seed・precision調整はない。

| r | ordinary | partial | closed P3 | local full | closed P5 | matched CTS |
|---|---:|---:|---:|---:|---:|---:|
| 1：N/axis | 2088098 | 1590438 | 1564916 | 1803142 | 1536780 | 2675065 |
| 2：N/axis | 926574 | 778141 | 776048 | 847756 | 773966 | 1185751 |

Nは実測shot数ではなく共通Bernstein十分予算。各axis誤差1/200、alpha=49/24000、row resource failure=1/12000、familywise総計1/20。
Taylor bias、係数丸め、native synthesisを共通に戻し、全行の残余幅は正。
有限meanの非unitarityを積誤差へ反映し、exact signalや浮動小行列残差でNを減らしていない。
probabilityは実dyadic law、weightはcoefficient/proposal。
local fullはzero-fillを含め、全step nonzeroのときだけ量子初期化するpolicy。zeroを除いた平均ではない。

T/CX/1Q cost、moment、acceptance、range、期待native費用、prep/readout slope、accepted-tail/hard upperは原資源表とコンパクト表にexact rationalで保存。
監査は保存されたIR/sequenceを数え直し、weight/momentと積費用の式を照合した。
監査で新しい回路生成、matrix evaluation、合成、sampling、最適化を行っていない。

## 保存された主比較の関係

主比較はclosed P5（left）とpartial（right）。各resource kについて
`G_k(h_k)=native_intercept_k+prep_readout_slope*h_k`、`h_k>=0`。
以下は固定policyの保存済み代数分類であり、研究GO・新規性判断ではない。

| r | T | CX | native1Q |
|---|---|---|---|
| 1 | P5 nonworse、全h_T>=0 | P5 nonworse、全h_CX>=0 | P5 nonworse、全h_1Q>=0 |
| 2 | P5 nonworse、全h_T>=0 | trade-off。h_CX交点≈3.51494987806 | P5 nonworse、全h_1Q>=0 |

r=2 CXではP5のinterceptが大きく、slopeが小さい。h_CXが交点未満ならpartial側、交点超ならP5側のこのCXモデル値が小さい。
交点のdecimalは表示用。正確な有理数と差の符号はaffine relations JSONを正本とする。
T/CX/1Qを単一scalarへ加算しない。Tは1Qに含まれる。
全baselineとlocal fullの値も保存してあり、主二方式だけの関係から最良algorithmを採択しない。

## 実行資源・技術監査

外側wall 631.280043 s、CPU（children込み）631.759466 s、parent peak RSS165108 KiB（約161.24 MiB）。
内側guardのpeak observed aggregate RSS366428160 bytes（約349.45 MiB）。
二worker・各numeric threads=1。aggregate CPU1800 s、parent/worker RSS512 MiB、aggregate RSS1536 MiB等の固定上限内。
wall capなしという新契約の仕様を維持。log cap超過・partial files・technical failureはない。
stdoutは完了status一行、stderrは0 bytes。外側logsの原bytes/hashを監査directoryへ保存した。

completed worker CPU7.855002 s、constructor/reference setupの記録CPU0.176721 s。
parent側の残りにはoperator/cost validation、serialization、頻繁なfilesystem-based guard検査が含まれるが、phase別profileは未取得。
保存sourceからserialization/guard overheadの可能性はある。秒数の原因を一つに断定せず、speedup倍率を主張しない。
完了済みsourceへ性能修正を入れたり再計算したりしていない。

監査で確認した内容：原manifest全fileのbytes/SHA、分割復元、全36 sequences、全1890 eventのnative count/strict-phase記録、dyadic law、moment、bias・積費用、外側exit/terminal/STOP/source/authorization/contract/marker/inventoryの整合。
source manifest133 critical files、protected ledger181 filesは不変。
親worktreeのHEAD、dirty diff、raw indexは実行前snapshotと一致。旧G10等の結果・markers・closures、Track A/N3、共有APIは保持。

## GPTへ渡す科学判断と停止条件

今回取得したのは、既知3-qubit development providerの同physical taskに対する、有限confidenceのadditive native resource map。
whole-circuit最適compiler、分子/DF、独立validation、actual quantum shots、online classical samplingは未実施。
classical per-trial costは未知で、0としない。final total-cost improvementやuniversal dominanceを主張しない。

次のGPT科学判断の材料は、主に次の点。

1. partial/P5の同confidence関係とr=2 CX交点は、既存成果を発展させる説明として何を支持するか。
2. r間で異なるTaylor biasとapproximationを、representation/normalizer/native cost効果とどう区別するか。
3. local fullとmatched CTSを含む全表、I1 access、加法的費用・未知classical費用の制約のもとで、限定したclaimと次の検証に価値があるか。

GPTによる重要科学レビューは、利用者の開始承認後に行う。
この報告自体で新規性、algorithm採択、独立論文成立、次stageを判断しない。
**science/synthesis runは一回のみ、retry=0、mandatory STOP。追加grid、SP、DF、GPU、G11、RA-D0、他Trackへ進まない。**
