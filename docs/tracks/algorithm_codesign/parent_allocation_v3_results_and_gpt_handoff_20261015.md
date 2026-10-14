# Parent allocation v3：一回実行完了・保存監査・GPT引き継ぎ

2026-10-15 JST。**PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW**。
保存処理の限定修正後、固定S3のauthorization-only直接子A3から一回実行し、15構成・70 lawの資源表を完成した。
retry=0、marker消費済み、mandatory STOP。科学的な採否・新規性・一般化は判断していない。

## 固定identityと承認

| 項目 | 値 |
|---|---|
| Science source S3 | `53a36adf9bdd84690e9cdf3e1f97f38a178441dc` |
| Authorization-only直接子A3／実行HEAD | `b1dfe53ca98a80d60831dbf9d28c5c9fc6a08f31` |
| 実行branch | `track-b-parent-allocation-v3-one-shot-execution-20261015` |
| Contract SHA256 | `adf2a75ecaadbd012f1ec20bb6b741979e5542614f151da6988d26938dfd82ad` |
| Authorization SHA256 | `2297724b815e3538599cd62d82de08b8d977757b2ffd4977c20bf6af6bc9c2b3` |
| Source manifest SHA256 | `24f465cb705baccbdea2ea5d509f41f15ab9aa9f3c381f42048ff46702f417c3` |
| v3 marker SHA256 | `3d30e8710abccf81c71b5ae2dba9270b0f8a866780b8e07e0e845b06a3f15b28` |
| Resource map SHA256 | `a781242e3136c51253862d9569388c0ac7fc9093d4c1f28c2fbe378523ba94fe` |
| 保存監査JSON SHA256 | `2f34a8a4c2f517d1b976c1aa7d378da0494f102b5d16a7b0d2545fd2719cf008` |

利用者の原文「次の作業に進んで大丈夫　修正が終わったらその先もできるなら行って」を、
科学条件を維持した限定修正と技術検証が通った後の一回実行への承認として、
[A3 receipt](parent_allocation_v3_execution_authorization_receipt_20261015.md)と承認JSONへ記録した。
A3の唯一のparentはS3。差分はv3承認JSONとreceiptの2 pathのみ。
source公開確認HEAD `7c0fcf368e76a1dacb1ed160be72ff0d6f72be27`を実行source／HEADとして使っていない。
launch前にclean、remote A3一致、169 critical hashes、478旧保護ファイルと5歴史的source Git blob、固定runtime、新namespace未存在を確認した。

入力・数学・費用会計の正本は[採択済みGPT科学設計](../../research/track_b_parent_allocation_scientific_design_review_20261014.md)、
[固定contract v3](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/execution_contract_v3.json)。
v3では[Parent専用16 KiB writer](parent_allocation_v3_buffered_source_preparation_20261015.md)とnamespace／歴史的source保護を更新した。
全科学条件・capsはv2と同じ。共有EvidenceWriter、pipeline、candidate、law、価格・N・誤差会計は不変。
新15＋既存78＝93登録外tests、人工出力bytes比較PASSを事前証拠として参照し、実行後に繰り返していない。

## 完遂した範囲と費用モデル

既知development用3-qubit distinct-basis synthetic：p=(1/5,3/10,1/2)、x=tau=5/7、状態000、sigma=+1、r=1、同じfinite P5。
Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4)R_ZZ01(pi/4)、Qi=Vi†ZiVi。
geometry／basis／DF rank／split L_Dは対象外で、分子DF・独立held-out検証ではない。

ordinary／closed P5のcommon、全3頂点、旧common価格の固定surrogateによるKKT、halfwayと、
partial-return／P3+tail／matched CTSを含む15構成。
全構成にcanonical、適用可能なglobal analytic IS、regularized ISの3条件を公平に与える。
候補・C*・inventoryは新価格取得前に凍結した。

| 処理 | 完成保存された件数 |
|---|---:|
| 使用旧unique sequenceのstrict phase/error guard | 20 |
| inventory unique keys | 36＝旧20＋新16 |
| 新合成 | 16、各一回・全KEY_COMPLETE |
| root ideal／realized controlled診断 | 72 |
| 構成／event binding | 15／2,562 |
| law／資源row | 70／70 |
| inapplicable analytic IS | 5（positive supportにzero Tがあるため）。固定契約の正常除外 |
| 既存sourceが保存した全law間affine費用関係 | 7,245（T／CX／native1Q別） |

law上限75に対し70完遂。inapplicableを失敗やnegative evidenceとしない。
新16価格を見て候補・proposal機会・precisionを選び直していない。
primitive 1e-6、root bits256、probability bits160、rho=eta=1e-12、strict phase 100dps。
axis精度1/200、alpha_axis=1/40、complex失敗率1/20。70 lawへalphaを割り直していない。
Taylor上界は|x|^6/720、実dyadic qとw=c/qを使い、係数・strict native誤差を別途戻す。

主費用は同じ十分試行数規則の`2 N E[T]`、h=0。
CX／native1Qを別指標とし、Tをnative1Qへ重複加算しない。
準備・読出し費用は`2N h`として保存affine関係を参照できる。
Nは共通信頼保証からの十分試行数で、必要最小shotでも実測quantum shotsでもない。
共有acquisition・ledger setup時間は記録したが、選択されたアルゴリズムのonline費用は`NOT_MEASURED`のまま。

## 保存比較の技術的な読取り

以下はresource mapが既に選んだ各armの最良common／noncommonの保存行を抜粋したもので、新しい最適化ではない。
小数はexact rational正本の表示丸め。固定h=0・十分試行数・additive nativeモデルに限る。

| arm／配分／proposal | N/axis | E[T]/trial | 2N E[T] | 2N E[CX] | 2N E[native1Q] |
|---|---:|---:|---:|---:|---:|
| ordinary common／regularized_16 | 924,186 | 139.9428 | 258,666,316.46 | 15,263,828.18 | 694,789,433.91 |
| ordinary vertex_2／regularized_1/16 | 1,587,522 | 70.0559 | 222,430,656.80 | 21,883,969.29 | 590,555,983.32 |
| closed P5 common／regularized_1 | 680,318 | 139.4729 | 189,771,851.79 | 11,574,441.76 | 479,835,618.53 |
| closed P5 vertex_0／analytic_IS | 2,270,017 | 33.8711 | 153,775,923.47 | 29,657,567.70 | 414,245,982.26 |

sourceが保存したcommonとの差はordinaryで約−36,235,659.66 T（−14.0087%）、closed P5で約−35,995,928.32 T（−18.9680%）。
両armの`accounting_relation`は`NONCOMMON_LOWER_UNDER_FIXED_SUFFICIENT_POLICY`。
登録70 lawのh=0最小は`config_07:analytic_IS`（closed P5 vertex_0）。
同時にN・CX費用が増えており、全resource座標のdominanceやh≥0一様優位を意味しない。
second momentはP5 common約1.6601に対しvertex_0約5.5381、range約1.3344に対し約5.1148。
回路単価低下とmoment／range／N増加の交換が記録されている。

`scientific_GO=false`、`scientific_adoption=false`、`priority_established=false`は保持する。
連続配分familyの最適性／no-go、ISを超える独立新規性、他provider・分子PR・QPE全体の改善は、この有限比較から認定しない。
全70行・係数・proposal・guard・正確な費用差は原資源表へ戻って確認する。

## 実行資源と保存処理修正の確認

| 指標 | 実行記録 |
|---|---:|
| public runner／ledger subprocess exit | 0／0 |
| outer wall | 26.670214937秒 |
| completed ledger tree CPU | 30.236055秒 |
| outer CPU | 2.709452322秒 |
| completed CPU合計 | 32.945507322秒（上限3,600秒） |
| sampled aggregate peak RSS | 391,434,240 bytes（373.30 MiB、上限3,072 MiB） |
| completed child peak RSS | 165,188 KiB |
| raw出力最終総bytes | 27,946,909 bytes（64 MiB上限内、監査追加前） |
| outer receipt内final_output_bytes | 27,944,322 bytes（outer／terminal／STOP保存前の計測） |
| workers／numeric threads | 2／1 |

CPU／RSS／output／log／call capの超過なし、technical reason=null。wall capは固定契約どおりなし。
二つのoutput bytesは記録時点が違う。原bytes・receiptを修正して一致させない。
v2はCPU上限で未完遂だったため、v2/v3を同じ完成workloadの速度比較とは扱わない。
ただし、v2の完成保存prefix（15 eventファイル＋61 law＋61 row＝137ファイル）とv3は**bytes／SHA256完全一致**。
旧v2 prefixをv3の入力として流用せず、この一致は修正の保存意味論を確認する事後のread-only照合である。

## 正本の取得経路・監査

原出力は[結果ディレクトリ](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/)。
3,647,912 bytesの[resource_map.json](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json)は
7個の`.partNNNN.txt`から損失なく取得できる。
[evidence_manifest](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/evidence_manifest.json)が
イベント／law／row／chunkのpath・bytes・SHA256を結ぶ。
9大きなJSON群・23 UTF-8 chunkを確認済みで、各chunkは512 KiB以下。順番に結合して原bytes／SHAに一致する。

[read-only監査](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/saved_output_audit/saved_output_audit_v3.json)は**918チェックPASS**。
[実行script](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/saved_output_audit/audit_saved_output_as_executed.py)を保存した。
source／contract／A3／manifest／tool／candidate・inventory freeze、worker owner・一回予約・sequence・保存error、
root保存診断、全lawの実dyadic確率・weight・moment・range・費用、marker→terminal→outer→inner／STOPのhash chainを照合した。
監査では新N予測、合成、operator guard、行列、runner、shotsを実行していない。
原出力342ファイルのpath／bytes／SHAを記録し、complete coverage・partial0・旧v1/v2 marker不変を確認した。
169 critical hashes、478旧保護ファイル、親HEAD／index／70 status entries／131既存ファイルも不変。

[public runner観測](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/saved_output_audit/public_runner_completion_observation.json)はtool exitとledger exitを区別する。
[公開確認receipt](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/saved_output_audit/publication_receipt.json)に結果commit、remote SHA、再取得したpath／bytes／SHAを記録する。
結果commitと公開確認commitはS3／A3と混同しない。

## 次の判断と停止

次に必要なのは、登録配分の利益をどこまで独立した資源価値・一般化・研究成果と解釈できるかという**GPT科学判断**。
T低下とshots／CX増加、既知ISとの関係、合成価格・providerへの依存、未知online費用、有限候補と連続familyの境界を検討対象として渡す。
新規性・科学的GO/STOP・追加実験をCodexが採択しない。
資料公開はGPTレビュー開始承認でも次の科学実行承認でもない。レビューは利用者の明示的な開始承認後。

**mandatory STOPを維持する。** v1/v2を再試行せず、v3を再起動せず、D1/D2/D3／219,303 pair再評価、
追加precision／seed／provider／r、native compile、LP、DF／分子、GPU、trajectory、SP接続へ進まない。
