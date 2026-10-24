# Track A：PRリソース評価データの体系化・論文利用準備

2026-10-24 JST。**データ整理を完了。原結果を維持する参照ビューを作成した。** 既存3環境の49 catalog行と最新結果への9 access viewを接続し、科学条件・証拠の強さ・比較可能性・採用状態を保持した。これは58独立実験の集計ではない。CEM v3全23主張、主表6条件とFigure 2、最新の標準Yoshida4独立確認を固定原典へ接続した。

新規科学計算、保存値からの新しい費用・比・順位・統計の解析、sampling、回路生成、compile、simulation、図表生成、原稿執筆、主張の正式変更は0。第2研究C0の進行を待たず、そのworktree・SOURCE・仮説・モデル・結果へ介入していない。**旧strict COMPAREはBLOCKEDのまま、mandatory STOP。**

## 正本と成果物

現在の正式採用証拠は [09e71910のpaper manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/09e71910ee49e0d9d3a2f2f410249f6623022c0d/artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/paper_manifest.json)。同packetのCEM v3・ledger v5・paper data manifest・evidence roles・source catalog・adoption recordを再利用し、変更していない。旧f75採用版と6dafの3環境索引は来歴を保持する。旧版と新版を連結して証拠数を増やしていない。

|成果物|内容|
|---|---|
|[研究索引](study_index.csv)|49元索引行＋9最新参照ビュー。物理条件・DF・候補/構成・cohort・n・費用単位・SOURCE/result・制限|
|[比較可能性](comparability.csv)|16組の比較条件。完全に比較可能／限定条件下／参照のみ／完了確認として比較不可|
|[主張の利用ビュー](claim_usage.csv)|既存CEM23行の主張・科学条件・証拠強度・未解決をliteral保持。原SOURCE、結果path、反対例・利用強度・配置候補を接続|
|[図表用データ索引](paper_data_index.csv)|既存主表6行、Figure 2、geometry、baseline、588候補、精度別方向、同一q、独立確認、C/N/G、希少event・否定的履歴|
|[未解決事項](unresolved_issues.csv)|既存不足項目と最新scopeを5つの論文化影響へ分類。必要な場合に限る最小対応を記載|
|[SOURCE catalog](source_catalog.json)|固定commit/path/SHA/URL。旧catalog全3,538件は複製せず、既存ref_idへ戻る|
|[schema](schema.json)・[方針/版差](policy_and_scope.json)|三層、unknown/non-applicable/not-verified、n/N/C/G/SE、採用と配置の意味|

## 整理した研究系列

|系列|含めた主な資料|科学条件・利用上の区別|
|---|---|---|
|Geometry|H4距離依存map・native B0+I・保存主張/台帳|legacy DF12。tol-only DF10と同一条件にしない。各距離・測定subset・元baselineを保持|
|Baseline|B0/B0+I/B1/B2/B3、強化PF域、H6 native B0+I、H8保存map|実測とmodel、元比較集合と強化集合を分ける。B2/B3の存在を全系列で評価済みと推定しない|
|PF構成|共通PF588候補、S2・標準Yoshida4・m5_best、q/prefix/partition|H2/H4/H6、1Å、STO-3G、tol-only1e-8 rank3/10/19、T=.8/P=0。全公式15比較と同一公式45比較は別の問い|
|PR探索|元A n8、旧union n32、新A拡張、outer32、高次PR保存会計|原候補・pilot・探索・選択後固定を区別。有限標本の点比較。全PR/PF同域最適化ではない|
|独立確認|元H2 P、H4/H6/H8主表、元/再構成PR対PR、最新標準Y4|別cohortをpoolしない。H4粗精度非再現、H6粗精度PF有利、H8同一観測aliasと公開監査留保を保持|
|Costモデル|理論/compiled S2、固定N G00/G01、最新C/N/G分解|C・N・Gとnを分ける。N_circuit/G10/G11 nullを0補完しない。新S2 H6/H8補助FAILは条件付き参照|
|診断|normalization/headroom/bias、rare event、compiler/native・runtime、旧中断/FAIL|技術PASS・実行完了は科学的優位の保証ではない。未観測費用・母平均・不明な親終了はunknown|
|論文用固定|現行採用体系、主表6条件、Figure 2、M1〜M3対応|既存図表・表示規則・原数値を維持。掲載候補分類は正式採択でも原稿でもない|

Hamiltonian・state・reference・r allocation・scalar/phase・native lowering等は、系列名や同じrankから推測しない。索引のcondition_referenceと個別候補/input manifestへ戻る。全fieldが存在するわけではなく、不明値は明示した。

## 既存結果として把握できること

以下は既存CEM・採用記録・保存REPORTの記述を関連付けたもの。新しい科学判断・数値解析ではない。

- PRが低費用となる登録条件と、PFが低費用となる条件の双方がある。既存主表のH4 ε=.05はNOT_REPRODUCED/PF低費用、H6粗精度はPF有利方向をREPRODUCEDとした結果であり、REPRODUCEDを一律PR勝利と読まない。[主表原典](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1923fce8f7461513d1bc47b5061e87cd70917e87/artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/main_table.csv)
- 決定論baselineの強化やPR候補domainの拡張で、保存点の費用方向・分母が変わる場合がある。CE1〜CE3・COMMON588の固定集合内の主張として整理した。全候補最良・一般的優位へは拡張しない。[CEM原典](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/09e71910ee49e0d9d3a2f2f410249f6623022c0d/artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/claim_evidence_matrix_v3.csv)
- 共通PF588候補の採用範囲ではH2全5精度はPF低費用、H4/H6はε=.001/.0005でPR低費用、他3精度でPF低費用。保存探索・固定域点比較であり、独立確認を588件実施した意味ではない。[共通域採用](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f75a7c25d584b69d90de454731a7f343a421e059/artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/paper_manifest.json)
- 固定S2/Yoshida4構成の精度別比較では、H2全5精度、H4 ε=.005/.001、H6 ε=.001に異なるqのFull/PR方向逆転が保存されている。H4 ε=.005の弱い差も有限点として扱う。同一qのH4/H6 ε=.001ではq4は双方Y4、q8は双方S2が低費用で逆転なし。q・signed piece・RTE配分・partitionも変わるため、次数単独の因果効果ではない。[保存精度/対照結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23/artifacts/resource_applicability/track_a_h246_reversal_batch_run01/2026-10-24/REPORT.md)
- H4/H6 ε=.001の標準Y4/q4対S2/q8は、独立PR費用方向の条件付き近似再現として正式採用済み。再利用S2 n25/23、新Y4 n45/32を別群のまま保持する。新Y4計77は取得総数で、2比較familyの数や合算推定量ではない。m5_bestの確認を流用したものでも外部施設での再現でもない。[正式採用REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/09e71910ee49e0d9d3a2f2f410249f6623022c0d/artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/REPORT.md)
- 最新C/N/G保存分解では、H6 PRは高次化でCが上がってもN削減が上回る。単価が必ず小さくなるという説明は成立しない。これは元の軸別G会計の説明で、operation別compiled寄与・独立N推定・純粋なcompiler因果効果の認定ではない。[保存C/N/G解析](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2536c4f959bfb4edd7d2deeceb155585fb53eb0/artifacts/resource_applicability/track_a_h246_order_cost_analysis_run01/2026-10-24/REPORT.md)

主表と最新高次PR確認は異なるcandidate/prefix/K/cohortの比較であり、主表を差し替えない。H2の探索会計完了はstrict native identityの解消を意味しない。H6四方式の保存値回収は、新しいH6取得・新順位の実行ではない。

## 比較可能性と論文利用

完全に比較可能と記載したものも、各固定resultが定義する一組のcontrastに限定する。分子・geometry・DF政策/rank・state/reference・T/ε/P・公式・q/prefix/r/K/partition・scalar/control/wrapper・compiler/native・会計政策・cohort・費用単位が必要な範囲で対応することを前提とする。family名や設定名の一致を完全identity証明にしない。

[比較可能性一覧](comparability.csv)では、同一q内部の固定比較と採用済み独立PR比較を「固定contrast内」、同一入力の固定N会計・共通PF対探索PRを「限定」、DF12対DF10・異なる分子・元と再構成cohortを「参照のみ」、終了監査のないH6run02を「完了確認として比較不可」にした。異なるqの主比較と同一q対照は別contrastであり、どちらかを他方で置換しない。

Main候補は現行主表、採用共通域の限定結果、条件付き独立Y4方向等。Supplementary候補はgeometry/baseline詳細、精度別・同一q対照・C/N/G・構成診断。監査留保・補助FAIL・unit-cost witness・未評価/失敗は条件付き参考または限界表示とし、未評価を有利な候補へ格上げしない。配置候補を整理しただけで、論文A/Bの選択や中心主張は採択していない。

## 未解決事項と論文化への影響

[未解決事項一覧](unresolved_issues.csv)は次の5分類を使う。同じ不足でも、選ぶ主張の強さで必要性が変わる。

|分類|例|今回の扱い|
|---|---|---|
|1：既存限定主張には影響しない|旧原resultの追加来歴、資料版差、未取得Level1原文|未取得を明記。独立して可能な整理を停止しない|
|2：限界表示が必要|経験的allowance/reference、H6旧parent、H8公開監査、候補域非対称、q交絡|原audit/status・有限点scopeと共に引用。無条件の新計算を要求しない|
|3：主張を強める場合のみ必要|母平均/rare-tail保証、旧原27候補identity、独立N、全履歴seed除外、旧strict比較|現在の限定結果と分ける。必要性が採択された時だけ別scope/承認を準備|
|4：結論/採用に影響しうる不整合|旧H6全域非相殺の欠測、GAP-001/008失効・原本不足|対象値を復活させず、全域主張・正式証拠への格上げを拒否。別の限定採用結果へFAILを転移しない|
|5：別研究|C0予測/選択、非DF別契約、energy/QPE/RPE全体・FT資源への拡張|今回へ混入せず、追加実行も開始しない|

O00002の原本発見・operation stream回収は後続資料で更新されているが、独立diagonal角度案やB2の過去load/完全stream範囲等の制限は残る。旧索引の「未発見」は過去時点として保持する。strict COMPARE未解除を研究全体の保存探索引用の一括禁止とはしない。逆に探索会計の完了をstrict契約PASSともしない。

元run01のH6 SIGBUS/NOT_EVALUATED、別run02の単価64軸とend audit未取得、後続再構成確認、標準Y4旧技術STOPと成功recoveryは別履歴である。後続成功は旧失敗の上書きでも同じgrantの再利用でもない。

H2他4精度の未評価という旧C/N/G REPORTの記述は、後続eebの固定会計で補完済み。H4/H6標準Y4未確認という旧計画も後続採用09e71910で更新済み。原文・STOPは変更せず、新旧を参照で接続した。

## 共通指示・取得範囲・実装

取得可能な共通運用指示のSHAは `050e5dc7055a1c114c35eb17913f3ee050092b57444dcffb70949e896548ac28`。指定正式原本 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` は未取得で、取得可能版には§10.3がない。§8.1/8.4/10.1/12と、今回ユーザーが明示した科学的理解・再利用優先方針を適用した。元指示を変更せず、正式原本を読了済みとはしない。

指定 `PR_resource_evaluation_Level1_review_2026-10-24.md` は既知の共有入口・今回worktreeで未取得。新レビューを実施・再採択せず、既存M1〜M3対応とreview source receiptを参照する。索引と既知の固定資料を読み、3,538原参照やprivate rawを再取得・再監査していない。

既存 `integrate_h468_paper_evidence.py` のCSV/JSON/pointerと、`adopt_h46_standard_y4_confirmation.py` の固定Git読取・token保持を再利用した。`adopt_h246_pf_common_results.py` は固定588採用専用なので調査のみで再実行しない。追加adapterはmetadata join/serializer/schema・参照検査だけを実装し、科学runnerをimportしない。軽量なcached Git metadata処理で、恣意的なCPU/RAM/worker上限や共有環境変更は導入していない。

## 技術検査・保全・終了

[検査記録](validation_summary.json)：25項目PASS。固定commit/path/SHA、49行の元key、CEM23行のliteral、主表6行、n/N、DF12/DF10、標準Y4/m5、失敗/未取得のstatus、pooling禁止、C0独立性、公開差分のprivate locator/認証情報除外を確認した。作者自身の技術検査であり、別担当の独立科学レビューではない。

[保全監査](preservation_audit.json)：変更を新しい参照bundle・metadata adapter・小さなデータ入口へ限定。原SOURCE・入力・raw/QPY・結果・grant・ledger・marker・STOP・既存CEM/manifest・主表・Figure 2・原稿を変更しない。他worktreeのHEAD/statusを凍結条件にせず、他チャットのprocessへ操作を行っていない。

今回の公開範囲はmetadata・固定公開参照・技術記録のみ。private raw/seed/grant原本・内部絶対path・認証情報は含めない。共通AGENTSのコミットに含まれるpush許可と今回の明示指示で専用branchを公開し、remote HEAD照合を行う。commit/remoteの最終値は完了チャットで報告し、自己参照のために旧成果物を再commitしない。

次担当：**ユーザー**が必要な段階で論文に使う主張・図表候補・配置を選ぶ。現行の限定主張を使うだけなら、未解決事項一覧を理由に新sampling/compileを一律要求しない。**Codex**は本データ整理を終了し、原稿・新図表・科学計算・詳細レビュー・最終freeze・strict COMPARE解除を開始せずmandatory STOPを維持する。
