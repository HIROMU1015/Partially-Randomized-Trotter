# PRアルゴリズム改善：3環境統合Gap

2026-11-05 JST。固定入力はLOCAL `ba14f71128362d197bfa47c9d0effacd0d7174d9`、SERVER 1 `b6a6fa81b98307beae1d89eef34eeab0c8805b6e`、SERVER 2 `0df7646826b3f863b19e69b204140059f9d09112`。
新しい科学評価・新規性審査・方向採択を行わず、保存状態の更新関係を統合した。
[台帳](IMPROVEMENT_VERIFICATION_LEDGER.csv)、[重複・再利用](DUPLICATION_AND_REUSE_MAP.md)、[残検証](REMAINING_VALIDATION_MAP.md)、[報告](REPORT.md)と併読する。

## 元暫定版から解消・更新したもの

| 元のGap・読み方 | 固定証拠による更新 | 残る境界 |
|---|---|---|
| 改善専用SERVER 1/2報告未取得 | 指定2commitの全14ファイルを取得し、LOCALの5ファイルと合わせ19ファイルの本文・blob一致を確認 | 各hostのprivate原本をLOCALが現物確認した意味ではない |
| B/F1/F2/PS48のrun/input対応が台帳で薄い | SERVER 1の6 bindingとSERVER 2の公開input識別子を段階行へ関連付け。F1旧/新、F2旧/新のSOURCEと原結果を分離 | 同じphysical/arm入力でもlaw・時間・K・compile経路・attemptは別。全run SOURCEの一括同定ではない |
| G6のlive接続を一つの限定pilot行で参照 | G6-ADAPTER-INDEPENDENT、G6-LIVE-512、SUP-G6-LIVEを区別 | 独立人工受入は別session/worktreeの技術確認。512は固定実API correctness。16,896は別workloadで、物理host独立や資源優位ではない |
| UWCを単純に「raw未発見」と読む | UWC-DISCOVERYに指定JSON等のFOUND_UNVERIFIEDを反映 | 元run SOURCE/command/当時環境/H-state-groupingへの結合は未完了。旧科学結果を復活させない |
| BF1旧中断と復旧をfamily要約だけで読む | BF1-ORIGINALとBF1-RECOVERYを分離し、二層SOURCE・保存cell再利用・bridge欠落を明示 | 元runはINCONCLUSIVEのまま。復元BF-Aは事後解析で独立確認でない |
| コードの存在・過去版差が概略だけ | SERVER 1の93資産、SERVER 2の固定SOURCE表、LOCAL参照を統合。全179 snapshot/path/blobと143同path同blob群を再利用表に保持 | 静的存在・AST確認であり、新条件correctness、実行互換性、test再実行PASSではない |
| SERVER 1の原稿・案内が未公開とみなされ得る | S1-DOC-01〜03に、現存原稿と公開archive同blobというSERVER 1の限定custody観測を保持 | 新しい科学run・独立レビュー・現在の再採択ではない |

## RQ-01〜07

番号はPhase 2Aから継承した分類で、科学的な優先順位でない。検証済みは元条件の範囲を指す。

| 課題 | 検証・実装済みの範囲 | 部分検証・負結果・既知同値 | 未検証・資料不足・保留 | 統合台帳の入口 |
|---|---|---|---|---|
| RQ-01 表現・辞書・normから総資源へ | Pauli/DF基盤、N1〜N3、core/pair、混在S/P、関連研究C。F1修正版の登録correctness、F2修正版の全wrapper数値検査と1-shot期待資源 | 固定F2でPP端点が低単価。混在必勝ではない。core/pairの強対照負結果、N1 exact fallbackを保持。旧UWC資料発見は科学的再検証でない | gamma/認証N/N加重利益、未使用入力selector、分子移送・全law精度・取得費込み。旧DF screeningは失効維持 | IDX-REP-01〜09、MIX-*、UWC-DISCOVERY、SUP-C-*、SUP-PAULI |
| RQ-02 非prefix・grouping・split | PR-2/M1/M2/PMのprefix、P-C/P-D、BF/BM/R3、一体再配分の既存研究 | BMはcompact BCH対応。P-D強baselineは同候補。BF保存復旧は固定familyでF固有decision gainを支持しない。P-C局所支持と広域失敗を分離 | 非prefix固有の実装可能な方法差・同情報既知対照後の増分。prefix勝者を非prefix成功へ移さない | GAP-NONPREFIX、IDX-PR-01、IDX-OPT-03〜05、IDX-RB-01/02、BF1-* |
| RQ-03 basis共有・control・Givens・生成 | WP06 full/support-run、P-A、D1〜D3、既存diagonal-only control/境界融合、混在辞書、PS48、関連固定word | P-A非退化対照は一区間baselineと差なし。Gaussian積は固有追加手順不足。PS48 correctness成功と一律RZ/CX削減不支持を両方保持 | 既存共有・合成後の独立差、未使用contextの選択性能、因果分離。singleton独立方向は正式一時終了 | IDX-CIR-01/02、IDX-RB-12/13、MIX-CAUSE/PS48/SELECT/CLOSE、SUP-C-WORD |
| RQ-04 law・RTE生成・return・有限bit | R0/R1、RA-D0/G1〜G6構成、zero-fill、G7/G8、G9/G10、B3受入/512/pilot、C1、M1-S。sampling freedomの静的調査と人工testも既存 | G9/G10は旧技術停止後に別run完了。Closed P5は内部特殊化。相関marginalだけではcoherent mean保存でない。SAMPLING-FREEDOMは限定NO_DISTINCT_FREEDOM | 全law精度・全N古典費・物理provider・同等cache/取得費込み全task比較。512 report/trialのartificial_only矛盾は保持。C1凍結 | IDX-RB-04〜10/15〜18、TB-*、G6-*、C1-*、SAMPLING-FREEDOM |
| RQ-05 native辞書・allocation・合成配置 | BS/SP、R1、D1〜D3、Parent/S4、G9/G10のnative/十分N会計、F2実費 | T利点とCTSのCX利点。SP長modelのloss/cap。訂正KKTと旧proxy目的、best-to-bestのproposal差を分離。F2は単回費用まで | 分子strict native・全wrapper・共通認証N、古典取得費と準備/readoutを含む同条件利益。T/RZ/CX/秒を混合しない | IDX-RB-03/04/09/10/13/14/17、TB-G9/G10、MIX-F2、資産profile ALLOCATION |
| RQ-06 round別表現/splitとenergy | fixed-splitのr/K・予算配分、短4round、全18roundモデル、V1/V2/V3/R1、条件付きenergy接続、M1-S局所信号 | 条件付き成功・energy接続は実施済み。費用モデルの点差と総合未決定を両立。研究C旧roundwise全D選択と後続candidate/r変種を区別 | roundごとに表現/rank/splitを変える新方式の同target・state・branch整合と資源差。全実装/実機成功や最終費用優位は別 | IDX-OPT-01、SUP-RPE-*、IDX-RB-15、SUP-C-*、SUP-RESEARCH-B |
| RQ-07 connectivityが選択を変える条件 | topology-freeでのDF/Pauli/混在共有・control、分子logical接続がbaseline | logical接続・topology-free correctnessはgraph/routing検証ではない | 同graph/layout/routing budget/precisionでの改善比較は、選定3調査の範囲で証拠未確認。全repository不存在とは断定しない | 既存回路baseline、GAP/remaining map、DF-CONTROL/MD-CONTROL資産 |

## RQ外の関連系列

| 分類 | 既存到達点 | 限界と状態 |
|---|---|---|
| ADD-PF：高次PF・BCH・processed | Yoshida/Morales/新4次、YP8m8、D6/支配位相、有限時間校正・response、BF/BM、高次PR、研究B | 既知法の適用・局所機構・正負比較を保持。次数一様優越、新selector、H12係数は未確立。別PF repositoryは旧索引の継承参照 |
| ADD-FR：phase/radius分離 | scalar補正、toy/非一様検証、technical note | strict witnessと固定予算での実用認証差なしを分離。未解決T3等で旧STOPを自動解除しない |
| ADD-C：PR-aware DF | 10/31旧結果→11/2小step→11/3候補005/064・独立価格→11/4固定word | 小step自己目的、nominal資源、有限operator correctness、厳密CA/全lawは別。SERVER 2旧summaryを最新全系列の否定へ広げない |
| ADD-RPE：統計・normalization・energy | 補正平均/実平均、B/attenuation、十分shot、branch/alias、条件付きenergy | 固定TのG、全roundモデル、理想IID＋数学的decoderの成功、実backendを別層で保持 |
| ADD-ACCESS：生成/access周辺 | bus/CAR、joint-loop、integrable insertion、集団遷移、adaptive、条件付き測定、unknown-gap等 | 既知還元・具体則不足・固定ansatz不成立・保留を継承。今回の科学的再審査ではない |

## 主要な後続更新の固定鎖

| 元状態 | 後続・現時点の読み方 | 追跡 |
|---|---|---|
| F1 metadata setter停止 | SOURCE `8192530b17d6bc4af195c734ff43a6e78c506942`、結果 `b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381` で登録PASS。未資格fallbackは最適選択でない | MIX-F1-OLD→MIX-F1、6公開binding |
| F2資格STOP/旧compiled FAIL | 資格STOPは残し、別資源契約のSOURCE `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`、結果 `dc7378ab34015287215bcf4ce52147e46d94ecd9` は全件数値・単回費用PASS | MIX-F2-QUAL、MIX-F2-OLD→MIX-F2 |
| singleton設計・局所改善 | PS48の固定subset結果→選択規則未確立→結果 `69f8e1cac23a4924ab81aebe399657a705024f6a` の正式一時終了 | MIX-SINGLETON-DESIGN→MIX-PS48/SELECT/CLOSE |
| G6 native未実装という初期記述 | G7/G8取得、G9 v2/G10 v3のnative会計、人工受入/512、H4 C1が後続。全scope一般保証ではない | TB-G6〜G10、G6-*、C1-* |
| C1旧P5 worker停止 | 別SOURCE P5R完走。旧Ordinary/Fullを再利用して3方式各1024 logical path。追加研究凍結 | C1-5A→C1-P5R→C1-PAUSE |
| BF1 serialization中断 | SOURCE `f226f8c81b5ba07b6d0c4b248c0eb15bf080a622` の保存解析で限定BF-A復元。原中断runは未完了のまま | BF1-ORIGINAL→BF1-RECOVERY→IDX-RB-01 |

F1/F2/PS48は4mode・2粒子・occupied(0,1)・LD1・残差2fragmentの合成fixture。分子geometry/AO basis/分子DF rank非該当。
F1はT1/5,q2,delta1/10,r2,h1/20,K2、F2/PS48はT1/5,q1,delta1/5,r2,h1/10,K0。
Qiskit1.3.0/opt1/seed17/rz-sx-x-cx/topology-free、phase fit禁止等の元精度を維持する。詳細条件・90/128桁参照・許容差は台帳と固定原報告へ戻る。

未取得raw、未結合SOURCE、未実施、技術失敗、限定負結果、既知同値、厳密保証未成立、正式凍結は異なる。
このGapは新しいPASS/FAIL/GO/STOPを発行せず、既存の採択・停止・科学条件を変更しない。
