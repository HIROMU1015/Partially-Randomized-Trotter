# Track A V2：保存来歴の回収・照合と固定fit条件の件数確認

**PR158件のdataset/seed来歴を回収し、262 scope entriesのうちH6の25件を条件付きmetadata結合済みへ更新した。残る237件は未解決である。fit可能なstratumは依然0件で、実データfit・残差計算は行っていない。**

起点は`83824ada333c0d00c07adc9c7de8225c4715811a`、branchは`track-a-v2-residual-analysis-readiness-20261016`。残差runnerの固定SOURCE`5703dbda58ebb8b7a9d18dce18ba33ed45f3c11c`とその9実行ファイルは変更していない。既存832理論行・保存compiled接続の正本は`7e41670565d9284da9aded94063be591fdcaa3dd`である。[前回報告](track_a_v2_residual_implementation_20261016.md)、[今回の軽量証拠・監査手順](../../artifacts/resource_applicability/track_a_v2_provenance_reconciliation/2026-10-16/README.md)を参照する。

## 回収・照合した証拠

対象は既存linear H6/H8、1.00 Å、STO-3G、保存DF rank19/28、T=.8、旧prefix/q/S2/S4/PR登録集合である。V1追加格子や新候補を加えず、保存CSV、task、worker claim、manifest、両軸wrapper、preparation/scalar receipt、固定SOURCEのcodeを読取専用で照合した。理論費用JSON・compiled接続gzip・NPZはparseせず、保存lawは記述・binary64 bitsの一致確認だけに使用した。確率正規化・期待値・費用の減算や比は計算していない。

|確認項目|結果|
|---|---|
|旧wrapperの独立照合|H6 848・H8 784のsymmetric-directional paired groupsを元task/worker/manifestと両軸receiptへ結合。scope間のPF再利用は独立標本として加算しない|
|PR dataset/seed来歴158 entries|全158件を回収。H6は`H6_MATCHED_V1_FRESH_COST`固定namespaceでSOURCEをseedの引数にしない。H8は元seed SOURCE `cb8a56c6b2a772827275f9200b02638030f33f72`をmanifestで保持し、後続実行SOURCEと区別。保存seed bitsをSHA256規則に照合した。新samplingは0|
|origin SOURCE欠測16 entries|すべて未測定の最短prefix（H6 p5/H8 p7）のB0各4候補×2 scope。実行SOURCEで穴埋めせず、`PRESERVED_UNMEASURED_NO_ORIGIN_TO_RECOVER`を保持|
|B0I wrapper identity欠測7 entries|H6 3・H8 4の不適格・未測定候補。`NOT_ACQUIRED_INELIGIBLE`を保持。新fingerprint・compileは生成しない|
|測定済みB0I 17候補|元両軸fingerprintと保存scalar/preparation receiptを照合。理論評価時に保存された24 receiptの参照と対応するpreparation hashへ一致。新取得14候補のdeclared profileを結合し、旧pilot再利用3候補のprofile同値は昇格させない|
|共通SOURCE意味論|旧nativeの9 origin SOURCEについて、共通library6ファイルとnative v5の7ファイルが固定providerとbyte一致。B0I bridgeのscalar applicationも原SOURCEで確認。これはprovider全closure・全binary・全basis同値の証明ではない|

旧索引には262 entriesのうち23未測定entriesがある。上の16と7はその内訳であり、「原receiptが見つかれば実測値が存在するはず」とは扱わない。H6/H8それぞれ探索/確認scopeの重複を含むため、262は固有candidate数ではない。

## 条件付き結合できた25件と残る不足

H6の探索17 PR候補と確認8 PR候補では、各候補の全paired groupsについて、manifest→task→worker→両軸のidentity、固定archive SHA `ad52f9f6…`、旧profile receipt SHA `4e32bb9b…`、H/state/DF/input、保存native lawのbitsが結合できた。固定archiveと理論preparation identityへの参照、共通nativeのSOURCEを保持し、`METADATA_RESOLVED_CONDITIONAL_ONLY`とした。完全数値wrapper envelopeや局所行列・compiler outputの再認証ではない。

|項目|保存証拠を結合できたentries|未解決entries|
|---|---:|---:|
|compiler profile|39|223|
|basis variant|25|237|
|control、scalar policy、native policy|各239|各23（未測定）|
|preparation binding|42|220|
|law binding|129（PR25＋決定論104の非該当）|133 PR|
|lineage compatibility|25|237|
|dataset/seed|PR158を回収、決定論104は非該当|0|

H6の旧探索では、初期SOURCEでその都度準備したbasisと、後半で固定archiveを読む実行が混在している。共通codeの一致だけで旧basis variantを一括置換しない。同一候補の8標本の一部にしかarchive証拠がない場合も、候補全体を結合済みへ昇格させない。

H8の旧PR manifestにはpackage/versionと実行pathの宣言があるが、候補群のcompiler binary/profileと全prepared basisを独立に固定する証拠は今回確認できなかった。後続B0Iの`622ce4a8…` profileを旧PRへ遡及適用しない。H8の補完preparation hashやB0Iのscalar/preparation hashの一致も、全旧compiled Gaussian lineageを認定する代わりにはしない。これらを未解決のまま残した。

## 固定train/holdout条件の確認

前登録の`(system,prefix,q,formula)`→canonical ASCII JSON→SHA256 big-endian `mod 5 == 0`を変更せず、candidate metadataだけで件数を数えた。費用値、fit係数、残差、rank、condition numberは評価していない。

|条件付きmetadata結合済みの層|train PR/PF|holdout PR/PF|判定|
|---|---:|---:|---|
|H6探索、旧profile・固定archive|11 / 0|6 / 0|PF不足。fit不可|
|H6確認、旧profile・固定archive|8 / 0|0 / 0|PFとholdout PR不足。fit不可|

各層はA/B・cosine/sine・lower/upperに共通するmetadata上の判定であり、次fitを予約した表ではない。train/holdout双方のPR/PF各2以上という条件を満たすstratumは0。未知のcritical fieldを持つ237 entriesを足して件数を作らず、profileやbasisをまたいで統合しない。

さらに、**来歴未解決を無視して全測定候補を数える上限診断でも、H8探索PR72候補・H8確認PR6候補・H6確認PR8候補のholdout PRはすべて0件**だった。この3集合は、来歴が将来すべて揃っても現契約の共通fitには進めない。H6探索の全測定候補はtrain PR66/PF13、holdout PR6/PF3だが、これは来歴を無視した件数上限にすぎず、fit適格性を認定しない。[全件数表](../../artifacts/resource_applicability/track_a_v2_provenance_reconciliation/2026-10-16/fit_count_precheck.json)にscopeの重複を明記した。

分割探索、別hash、method別fit、最小件数緩和、確認/探索の統合は行っていない。distinct theory x/rankの条件は件数不合格の後段にあるため未評価である。件数の構造的不足は、既存証拠が実際に失われたという意味とは区別する。

## 台帳・検証・保全

更新台帳は新しいprivate metadata出力へ保存し、SHA-256は`3dbb141ed4e11f6d09038cedd9067a20cfd575610e1b779462cc0ff2531e5be0`。旧registry、SOURCE、未承認config/grant、fresh-gate拒否fixture、grant/ledgerは上書きしていない。更新台帳は旧実行configへ設置しておらず、新config・新grantの発行も0である。

parameterized rootを使う[metadata収集手順](../../artifacts/resource_applicability/track_a_v2_provenance_reconciliation/2026-10-16/collect_saved_task_metadata.py)と[照合手順](../../artifacts/resource_applicability/track_a_v2_provenance_reconciliation/2026-10-16/reconcile_saved_metadata.py)を公開した。既存worker/producerを実行せず、stdlibで新しいmetadata出力だけを作る。各手順を実行し、source/seed/task/manifest digest・両軸・既存receiptのassertionと、固定libraryのmetadata-only分類を確認した。今回の検査はprimary agent自身のlocal checksであり、独立レビューではない。既存43人工testsや科学campaignは反復していない。

7,043入力receiptのhash台帳をprivateで保持し、公開はそのdigest・固定SOURCE/pathのreceipt・entryごとの判定・件数表に限定した。private manifest/config/registry本体、分子配列、保存費用値、他job一覧は含めない。旧667保護ファイル、V1・旧V2・理論worktreeのHEAD/status、旧V2未追跡script、残差SOURCE9ファイルを前後照合して保全した。旧runtimeをloadせず、環境・affinity・他jobは変更していない。

## 次の判断とSTOP

現契約のまま残差解析を承認する段階にはない。H6探索については、比較対象PFの旧basis/profileと、mixed lineageのPR候補を独立receiptへ結合できるかが残る。H8全体と両確認集合については、来歴回収に加えて固定holdoutでPRが0という設計上の問題があり、利用者による解析設計の再判断が必要である。変更案を採る場合も、結果を見る前に別契約で固定する。今回その変更・新計算・追加レビューは行わない。

**実データfit・残差・費用再計算・compile・signal・sampling・primitive/law/DPは0。成果をcommit・pushしてremote SHA/treeを確認した後、mandatory STOPする。**
