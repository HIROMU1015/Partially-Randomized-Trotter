# Parent allocation S4：訂正KKT補完実行・保存監査・GPT引き継ぎ

2026-10-15 JST。**PARENT_ALLOCATION_KKT_SUPPLEMENT_COMPLETE_AWAITING_GPT_REVIEW**。
採択sqrt(C)目的へ訂正した固定sourceを一回実行し、4構成・20新lawを完遂した。
旧50 control lawは保存値を再利用し、旧20 proxy lawは元namespaceで保持した。
retry=0、exclusive marker消費済み、mandatory STOP。科学的採否・新規性・研究方針をCodexで判断していない。

## 固定identityと実行承認

| 項目 | 値 |
|---|---|
| Science source S4 | `4ccc1832f63cd323b23fc05623f7e0b58cf34f37` |
| Authorization-only直接子A4／実行HEAD | `71cf1b76492ed01fc5bb8363246b5f13f7e166d7` |
| 実行branch | `track-b-parent-allocation-s4-one-shot-execution-20261015` |
| contract SHA256 | `f76a10c6f04972494d5b7607a1a32f425f47c52b5bdb6083f8a983a827056e7d` |
| authorization SHA256 | `f0a296c4a9028185cc50df051dbc2b2a32bc6c48a5f6b62b292ecdbf0211ea1f` |
| source manifest SHA256 | `7a0612cae3bf4a18e434a4ab2d759ac001c4d928ed4c02504453b4d750a2f4a9` |
| v4 marker SHA256 | `f04d5b249f549c1a7300e0dca6bd365155b464da9c30ccb6e0e1f8aaa73caea0` |
| resource map SHA256 | `8281cf66c1ce7b5abf1a46ba6d4c4725c994eabf3dadcc143a9877387cc117c5` |
| candidate freeze SHA256 | `c9624113afaec7af550842ef1fc25eef3dd347c8c1dbfc07ce909984360d7e0c` |
| inventory SHA256 | `b1d0ce222e52afacd5bbd770938522824809a930bf6ab5b3f26f4111820831bb` |
| 保存監査 SHA256 | `d832c80fd33807f4de0f56f8f83c8755396829c7fc35a05e9f6367613de56103` |

利用者の直近指示原文「次の作業に進んで大丈夫」を、直前に実行承認待ちとした固定S4補完one-shotの承認として
[A4 receipt](parent_allocation_s4_execution_authorization_receipt_20261015.md)とauthorizationへ記録した。
A4の唯一のparentはS4で、変更はv4 authorizationとreceiptだけ。
公開確認HEAD `b2b80bf82abfe2f3ac3aedaf189979cdd9cdb3bc` はscience source／実行HEADに使用していない。

[準備記録](parent_allocation_s4_kkt_repair_and_supplement_preparation_20261015.md)の114登録外testsは参照し、同一sourceで再実行していない。
実行前にpending preflight、承認後launch gate、remote A4と主要4ファイルのbytes、187 critical・844旧保護hash、固定runtime、cleanと新namespace未存在を確認した。
Python3.10.12、pygridsynth2.0.0、mpmath1.3.0とsource tree・tool optionsは固定どおり。
32 CPU／約51 GiB availableのhostで2 synthesis workers・numeric threads1を使用した。GPUは使用しない。

## 訂正の範囲と比較条件

[GPT科学レビュー原本](../../research/track_b_parent_allocation_v3_scientific_review_20261015.md)の採択目的

\[
S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2},\qquad z_i=b_i\lambda/\sqrt{C_i-\lambda^2}
\]

へ戻す訂正。S3はraw Cを受けながら分母にC²を用いていた。
旧config_04／05／10／11・計20 lawは、実際に評価された別のraw-C proxy配分の原記録として保持する。
旧20の費用・classificationを訂正後KKTの性能結果へ読み替えない。
新sourceはt=lambda²の固定128 bisection、outward root256bits、share64bitsの有理化を保存した。
**保存された配分は認証区間と丸め距離付きのsurrogate候補で、exact KKT最適点やnative費用の最適解ではない。**

既知development用3-qubit distinct-basis synthetic：p=(1/5,3/10,1/2)、x=tau=5/7、状態000、sigma+1、r1、finite P5。
Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4)R_ZZ01(pi/4)、Qi=Vi†ZiVi。
分子geometry／basis／DF rank／split L_Dは対象外で、独立held-out検証でもない。
root256bits／probability160bits、rho=eta=1e-12、primitive1e-6、strict phase100dps。
axis epsilon=1/200、alpha=1/40、complex失敗率1/20、Taylor上界|x|⁶/720を維持した。
同じ一complex taskの十分N規則を用い、law数でalphaを割り直していない。
実dyadic q、w=c/q、moment=sum(qw²)、range=max|w|と係数・strict native誤差を会計する。

対象はordinary／closed P5の訂正KKT・halfwayのみ。
各構成にcanonical、analytic IS、regularized ISの3条件（C*/16、C*、16C*）、計5 proposalを付けた。
今回は4構成すべて異なる候補として存在し、全20 lawが保存された。省略・不適用・失敗は0。
旧50 controlに含まれる5件のanalytic IS正常不適用はv3契約どおりで、性能negativeへ変更しない。
共通・頂点・参照構成の候補／law／N／native費用を再取得せず、50保存rowをそのbytesと値で取り込んだ。
C*とbaseline_Tもv3 freezeと完全同一。共通rootの保存full-event T価格[136,138,140]だけをproxyに使用し、新角度価格を見る前に候補・inventoryをfreezeした。

## 完遂した取得範囲

| 処理 | 完成件数 |
|---|---:|
| 新構成 | 4（config_04／05／10／11） |
| 新law／資源row | 20／20 |
| 再利用control law | 50 |
| 新旧を結ぶ訂正済み比較 | 70 law（旧proxy20は新比較から除外） |
| 新event binding | 672（上限1,092） |
| 新positive root合成 | 12 unique keys、各一回、全KEY_COMPLETE |
| 再利用sequence metadata／strict certificate | 11 keys |
| inventory全keys | 23＝旧11＋新12 |
| 新root event／ideal+realized controlled診断 | 12／24 |
| 旧operator guard再実行／非root native再計算／control N再計算 | 0／0／0 |
| quantum shots／retries | 0／0 |

signed inverseはactual sequence adjoint、W位相を保持し、up_to_phase=false。
保存root診断はindependent targetに対するideal／realizedの点検で、bias certificateは固定strict boundから取得する。
新sequence、T／T†／1Q／W count、strict error、key spec、worker開始・一回予約・結果hashを保存監査で照合した。
非root bindingは元common eventとevent／IR／native certificate／診断metadataを含めて同一。

## 新20 rowの表示用抜粋

以下は新resource mapの保存順で、追加のランキング・N再計算はしていない。
小数は保存exact rationalの表示丸め。全exact費用と元row/hashは
[new_law_index.csv](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/new_law_index.csv)へ対応付けた。
primaryはh=0の2N E[T]。Nは共通信頼保証による十分試行数で、必要最小や実測shot数ではない。
T、CX、native1Qは別指標で、Tをnative1Qに重複加算しない。

| ID | N/axis | E[T]/trial | 2N E[T] | 2N E[CX] | 2N E[native1Q] |
|---|---:|---:|---:|---:|---:|
| config_04:canonical | 924,298 | 139.9320 | 258,677,702.49 | 15,217,913.06 | 659,484,863.20 |
| config_04:analytic_IS | 924,761 | 139.8745 | 258,700,946.38 | 15,135,736.87 | 659,426,971.00 |
| config_04:regularized_1/16 | 924,724 | 139.8777 | 258,696,515.10 | 15,140,351.54 | 659,422,382.95 |
| config_04:regularized_1 | 924,488 | 139.9024 | 258,676,088.35 | 15,175,831.30 | 659,421,311.56 |
| config_04:regularized_16 | 924,316 | 139.9284 | 258,676,093.66 | 15,212,846.70 | 659,473,669.61 |
| config_05:canonical | 924,271 | 139.9555 | 258,713,583.23 | 15,244,050.31 | 667,654,462.51 |
| config_05:analytic_IS | 924,780 | 139.8840 | 258,723,906.84 | 15,224,602.39 | 667,664,974.56 |
| config_05:regularized_1/16 | 924,738 | 139.8881 | 258,719,651.02 | 15,225,350.81 | 667,654,731.53 |
| config_05:regularized_1 | 924,472 | 139.9190 | 258,702,405.93 | 15,232,520.99 | 667,616,594.76 |
| config_05:regularized_16 | 924,289 | 139.9511 | 258,710,512.71 | 15,242,495.72 | 667,645,373.56 |
| config_10:canonical | 680,288 | 138.4331 | 188,348,722.59 | 11,568,727.69 | 480,615,387.35 |
| config_10:analytic_IS | 680,727 | 138.3042 | 188,294,772.71 | 11,594,116.26 | 480,569,275.44 |
| config_10:regularized_1/16 | 680,682 | 138.3120 | 188,292,983.58 | 11,592,147.73 | 480,558,942.03 |
| config_10:regularized_1 | 680,425 | 138.3697 | 188,300,418.48 | 11,579,354.79 | 480,536,216.04 |
| config_10:regularized_16 | 680,296 | 138.4257 | 188,340,938.17 | 11,569,772.63 | 480,600,542.95 |
| config_11:canonical | 680,248 | 142.9955 | 194,544,810.46 | 11,587,064.48 | 510,452,538.10 |
| config_11:analytic_IS | 680,386 | 142.9769 | 194,558,960.82 | 11,546,066.94 | 510,482,186.44 |
| config_11:regularized_1/16 | 680,375 | 142.9780 | 194,557,295.77 | 11,548,405.45 | 510,478,254.52 |
| config_11:regularized_1 | 680,306 | 142.9862 | 194,548,699.78 | 11,566,266.54 | 510,458,988.99 |
| config_11:regularized_16 | 680,253 | 142.9944 | 194,544,741.88 | 11,584,574.26 | 510,451,914.99 |

config_04／05はordinary KKT／halfway、config_10／11はclosed P5 KKT／halfway。
新resource mapの`combined_comparisons`は50 inherited＋20 correctedからsourceが生成した比較であり、旧20 proxyは含まない。
保存された`finite_registered_best`は**config_07:analytic_IS**（closed P5 vertex_0）。
これは有限候補・h=0・固定十分N・加法的native費用モデルの出力であり、連続配分最適性・独立新規性・研究継続の認定ではない。
common/noncommonの保存mechanismはordinary `config_00:regularized_16` 対 `config_03:regularized_1/16`、
closed P5 `config_06:regularized_1` 対 `config_07:analytic_IS`。
保存`scientific_GO=false`、`scientific_adoption=false`、`priority_established=false`を維持する。

準備・読出しh≥0の関係は保存affine comparisonへ戻る。新たな交点・最適包絡線は本監査で計算していない。
NやCXの増加を伴うT低下を全資源dominanceと扱わない。
classical setupの候補生成／ledger CPUは保存されているが、online sampling費用は`NOT_MEASURED`。
共有acquisitionの実行時間を選択された方式一つのsetup費用と同一視しない。

## process・資源・停止記録

| 指標 | 保存値 |
|---|---:|
| public runner／ledger exit | 0／0 |
| outer wall秒 | 9.447381626348943 |
| completed ledger tree CPU秒 | 12.457264 |
| outer CPU秒 | 1.020047735 |
| observed aggregate peak RSS bytes | 396926976 |
| completed child peak RSS KiB | 165436 |
| runner原出力最終bytes（監査追加前） | 12537928 |
| outer final_output_bytes | 12535325 |
| workers／numeric threads | 2／1 |
| resource failure／technical reason | null／null |

observed peakは約378.54 MiB、aggregate上限3,072 MiB。worker512／ledger2,048 MiB、address space4,096 MiBを維持。
CPU上限はper-key120秒／total3,600秒、wall上限なし。出力64 MiB／log64 KiB／keys12／root診断24等の固定capを超過していない。
原出力総bytesとouter記録bytesは計測時点が異なる。outer記録は自身・terminal・STOPの保存前で、原receiptを合わせて変更しない。
旧v3との実行時間比較は異なるworkloadのため速度改善率と呼ばない。

[public runner観測](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/public_runner_completion_observation.json)は
public tool exitとledger subprocess exitを分離して記録した。
terminal／inner／outerのclassificationが一致し、marker→job／terminal→outer→innerとSTOP→terminalのhash chainを照合済み。
資源・numeric failureとpartial fileは0。失敗prefixを使った科学的な分類はない。

## 正本・監査・取得経路

[結果directory](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/)、
[evidence_manifest](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/evidence_manifest.json)、
[resource_map](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/resource_map.json)、
[inherited_rows](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/inherited_rows.json)を正本とする。
inherited row内の相対pathは`source_result_directory`に明記したv3 namespaceを基準とする。v4のrow番号と混同しない。
旧正本はcommit `cd055232f9612982a60953341b9cbf6168dd0bd8`／v3結果。

[保存監査](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/saved_output_audit_s4.json)は**371 checks PASS**。
[実行したread-only script](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/audit_saved_output_as_executed.py)を保存した。
S4/A4/contract/tool/source/marker・worker PID/一回実行・sequence hash/count/error、KKT保存区間／simplex／halfway、
新20 lawのdyadic q／weight／moment／range／native費用、50 controlの元row bytes/value、旧非rootデータ、C*を照合した。
旧918監査は繰り返さず、監査の新synthesis／operator guard／matrix／N forecast／science runner呼出しは0。
監査scriptの初回には旧bindingから既存`new_root=false` fieldを誤って除去して比較するASSERTIONがあったが、
完全な辞書一致へ監査側だけ訂正してPASSした。science source・raw結果・停止classificationを変更していない。

187 critical files／844旧保護files／10歴史的Git source blob、旧v1/v2/v3 markerが不変。
親worktreeはHEAD／index／71 status entries／132既存filesのsnapshotと一致し、dirty編集をstageしていない。
原出力169ファイルと11 chunkのpath／bytes／SHAを監査JSONに記録した。
大きなresource_mapとevent JSONは既存`.partNNNN.txt`を順に結合してbytes／SHAを完全復元できる。
全review chunkはUTF-8かつ512 KiB以下。raw JSONの値を丸めて保存し直していない。

公開後のremote SHA・主要資料と全新資料のbytes/SHA再取得は
[publication_receipt](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/publication_receipt.json)へ記録する。
result/publication commitをS4/A4と混同しない。

## 次にGPTへ戻す判断

訂正された内部候補と頂点／commonの関係、proxyの目的とnative十分T費用の違い、ISとの独立性、
TとN／CX／準備費用の交換、有限候補・合成器依存・未知online費用をどう解釈するかが科学判断の対象。
本報告は技術的な完遂と固定費用表の確認であり、新規性・一般化・研究継続を採択しない。
旧20 proxyと訂正20を同じKKT最適化の反復試行と扱わない。次の科学レビューは利用者の明示的な開始承認後。

**mandatory STOP。** v4 runner再起動、追加candidate／precision／seed／provider／r、新compile／合成、
D1/D2/D3・219,303 pair、DF／分子、GPU、LP、trajectory、SP接続、次stageを開始しない。
