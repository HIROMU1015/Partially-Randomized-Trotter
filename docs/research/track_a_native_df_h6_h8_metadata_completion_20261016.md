# H6/H8 Native DF：保存metadataの追加受領と後続実行案

**H6の完全なpreparation archiveを既存保存資料から取得した。H8の完全archiveは確認した範囲で未取得。**
前回のH6の「全block/basis/component law未取得」は、今回の発見で更新する。
H6 archiveのscalar/Gaussian記述をstdlibだけで照合し、実Gate・配列・回路を復元していない。理論ゲート数・期待費用・比・候補順位・shot数の新計算は0。

[証跡入口](../../artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/README.md)／[前回監査](track_a_native_df_h6_h8_metadata_binding_20261016.md)。
本書は技術的な追加受領・準備記録であり、科学的採択、Claude Level 1レビューの代行、本計算の成功認定ではない。サブエージェント0。

## 起点と認可範囲

- 起点branch `track-a-native-df-h6-h8-metadata-binding-20261016`、固定commit `46c06e5653152eb4ce786d69d927ecbf12f9a2de`。originの同branchがこのSHAと一致することを確認した。
- 新branch `track-a-native-df-h6-h8-metadata-completion-20261016`、独立worktree。既存root・実行worktreeを変更しない。
- 利用者の「次の段階に進んで大丈夫」「続きを行って」を受け、既存資料の取得・照合・後続準備を実施した。実入力の補完生成と理論費用評価について出している範囲確認には、具体的な選択回答は未受領。今回その数値処理は実施していない。
- 前の読取は一度、自動承認レビューの利用上限で未実行になった。今回同じ認可範囲の読取を再開できた。設定変更や承認経路の迂回は行っていない。

## 1. H6の新たな受領資料

旧希少次数解析の[固定script](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/820287551b3b6049b30bdb1ef46766c396e14d9f/artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/analyze_saved_cost.py)から、private `h6-roundoff-preparations-v1.json` の所在を追跡した。
原本5,670,237 bytes、SHA-256 `ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010`。
同じ固定commitには[全component catalogの公開抽出](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/820287551b3b6049b30bdb1ef46766c396e14d9f/artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/component_and_structure_v1.json)も存在した。既存抽出の費用・確率解析scriptを再実行せず、必要な記述の所在だけを参照した。

|archive root|保存された用途|block記述|全component数（保存宣言と件数を照合）|
|---|---|---|---:|
|discard:p5|B0 prefix5・B0Iの元prefix構造|one-body＋fragment0〜4|0|
|discard:p10|B0 prefix10・B0Iの元prefix構造|one-body＋fragment0〜9|0|
|discard:p15|B0 prefix15・B0Iの元prefix構造|one-body＋fragment0〜14|0|
|faithful:p5|PR prefix5|prefix block・完全tail registry|1,092|
|faithful:p10|PR prefix10|prefix block・完全tail registry|702|
|faithful:p15|PR prefix15|prefix block・完全tail registry|312|
|faithful:p19|full PF|one-body＋全19 fragments|0|

条件は保存linear H6/1Å/STO-3G/12 modes/sector400/DF19、元signed generation order、T=.8の既存研究系列。新しい分子や分割を作成していない。
全10,687 content-addressed nodesのhash・codec digest、root参照、finite float.hex、tuple順を確認した。
blockではbasis記述hash、runtime gate名/ordered qubits/parameter bits/局所行列hashの記録値が一致することを確認した。tailでも全componentのbasis記述とregistry runtime記述を照合した。
**局所行列を数値生成してhashを再計算した検査ではない。** 原本の`decode_preparations`は実Gate/Operatorを作るため呼ばず、description専用readerを追加した。

保存値の所在・node hash・preparation/partition/tail hash・block/registry完全性は[受領検査](../../artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/h6_existing_archive_inspection_v1.json)に保存した。原本codecの全payloadやprivate絶対pathは公開しない。

## 2. 実行来歴への結合と限界

H6確認の実行SOURCE `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` のmanifestは上記archive SHAを固定していた。
同SOURCEの[確認worker](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trottertracks/resource_applicability/h6_confirmation_fast_execution_v1.py)は、archiveを読み `use_preparations` でfrozen preparationを接続する。今回このSOURCEとmanifestの参照を照合した。
原本archiveには旧profile SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75` が記録されている。後続B0Iのprofile digest `622ce4a8…` と同一profileだと扱わない。

前回確認したH6探索K2/K4各1件・確認K2 1件については、選択されたcomponent/basis ID/hashがarchiveと一致した。これは代表3件の一致であり、全探索のbasis variantを一括で置換する認可ではない。旧希少次数資料も旧/新basis variantを区別している。
H6 B0Iの新p10/q1に保存されたcurrent-profile preparation hashはarchiveのdiscard:p10と一致する。ただし全B0I軸の旧/新SOURCE・compiler bit同一性を今回新たに認定しない。

H6で今後必要な処理は、既存lambda/eta/eigenvaluesから元producerのRZ/RZZ/global phase列を指定時間ごとに得る処理である。これらの**係数とGaussian操作列は既存archiveにある**ため、H6分子入力の再diagonalizationやGaussian再生成を先行させる必要はない。primitiveの数値評価・Fraction lawの正規化・期待値DPは今回未実行。

## 3. H8の取得状況

private evidence領域とH6/H8 provider checkoutの関連artifactで、保存preparation・component・archive名を限定探索した。最初の探索は818 directoryを対象とし、test/cache/worker group等2,445 directoryを除外した。希少次数資料を経由してH6 archiveへ辿り着いた。これは全storageの不存在証明ではない。

H8の固定SOURCEでは `_prepare`/`_prepare_discard` の結果はprocess内cacheに保持され、`prepared_representation.json` には主にpreparation hash・lambda・constant等が保存される。
[H8 signal port](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trottertracks/resource_applicability/h8_resource_map_port_v1.py)、[cost port](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trottertracks/resource_applicability/h8_fast_cost_port_v1.py)の保存範囲も確認した。
H8の完全registry・全component law・全block係数を含むarchiveは未取得である。

既存の完全exportが別途あればbytes/SHAで受領する。それがなければ、保存H8 snapshotから固定providerのpreparationを作る処理が必要となる。これは既存DF係数の読み出し・小行列diagonalization・Gaussian Gate記述生成を伴うため、前依頼で禁止した「未取得metadataの再生成」と同じ範囲に含まれる。今回はNPZを開いていない。
これは新SCF/DF分解/state solver/trajectory生成が必要だという意味ではない。

## 4. 全登録候補と既存wrapperの索引

既存D0 JSON・B0I resultから候補identityと10条件の会計receipt digestを転記し、公開済みwrapper CSVからSOURCE/seed/replica/axis/event/compiler/fingerprintだけを索引化した。
gate metrics・総費用を加算/平均したり、shot数を導出したり、候補を再選択したりしていない。

|索引|H6|H8|
|---|---:|---:|
|固有candidate ID|104（旧92＋B0I12）|104（旧92＋B0I12）|
|scope別candidate entries|132（探索92＋確認28＋B0I12）|130（探索92＋確認26＋B0I12）|
|既存会計条件への参照|1,320|1,300|
|旧wrapper paired group索引|850|786|
|そのうちordinaryの参考pair|2|2|
|B0I未測定を保持したcandidate|3|4|

850/786は回路費用ではなくmetadata pairの件数で、独立trajectory標本数でもない。旧deterministic・ordinary検証pairを含む。探索8/確認32とdeterministic1を分離した。
両軸が同じSOURCE・seed・replica・event digest・compilerへ結合することを照合し、missing/duplicate axisや異SOURCEの結合は受理しない。
H6の確認CSVにはSOURCE列がないため、固定integrated inventoryと256個の既存task_bindingの保存SHAから対応付けた。全256 task hashの一致を確認し、推測でSOURCEを補わなかった。
どの精度でも不適格だった未測定B0やB0Iはmissingのままとし、0 gateや成功実測へ置換しなかった。

[候補索引](../../artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/saved_candidate_connection_index_v1.json)と[被覆・元file receipt](../../artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/saved_identity_coverage_v1.json)を保存した。
`dataset_seed_source`はこの索引だけで確定しない行にnullを保持した。実行SOURCEで代用せず、将来の評価で元seed manifest・law variantへ結合する。
scope索引が揃ったことと、前回adapterの完全connection envelopeが全候補で認証済みであることは別で、全`full_connection_envelope_validated=false`を保持した。

## 5. 実装と限定検査

- [reader](../../src/trotterlib/native_df_archive_metadata.py)：typed JSON DAGのhash・参照・float.hex・Gaussian記述の比較専用。数値library/元codec decoderをimportしない。
- [archive inspection CLI](../../scripts/inspect_native_df_preparation_archive.py)：実archiveを読むが実Gate/Operatorを作らず、node digest・取得範囲を報告する。AS512MiB/CPU120秒、科学import拒否。
- [12人工tests](../../tests/test_native_df_archive_metadata.py)：改竄hash、missing ref、重複JSON key、unsupported gate/array node、非canonical/nonfinite float、signed zero、qubit順序、runtime parameter/hash記述の不一致を確認。12 PASS。

既存system Python3.12.3、process1/thread1、Python -P -B、新作業・tmp/logはhome内。H6実archive記述検査は約1.77秒、科学import・実Gate/Operator・compile・sampling・理論費用呼出0。旧20/42人工campaignは反復していない。共有環境・科学venv・他job・GPUは変更しない。

## 6. 次の実行案（未起動）

[未認可実行案](../../artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/next_execution_draft_v1.json)に対象と段階を固定した。`approved=false`, `runtime_authorization=false`, `allowed_cpus=[]`, `absolute_launch_command=null`。

1. H6は受領archiveのscalar/操作記述を使い、元producerから指定時間のdiagonal primitivesを取得する。H6の再SCF/DF/state/preparation生成は不要。旧/新variantとSOURCEを保持する。
2. H8は既存完全exportを優先し、未取得なら固定保存snapshot/providerでprefix7/14/21のfaithful/discardとfull28の**7 preparation**だけを補完する。新sampling・native full wrapper・transpile/compileは0。別のprepared hashが出たら過去値を同一扱いせずSTOPする。
3. law/source/profile/input/primitive/phase/axis/shot receiptのbindingを完成する。旧profile/current-profileと全結果の出自を混ぜない。
4. 別途認可された場合だけ、固定208 candidate IDに対してA/Bの費用・有限law期待値を評価し、既存compiled記録へ対応付ける。実nativeはA-prefix/B-tailの混合政策であり、Bの仮想PF相殺を実施済みcompilerの効果とはしない。scope重複の262 entriesを262固有candidateと数えない。

一般identity phaseは既存 **0〜1 P/occurrence**幅を維持し、未観測次数を0費用にしない。期待費用DPの既存transition capで停止した場合、無断でcapを増やしたり近似fallbackへ切り替えたりしない。source/modelの科学的意味の変更はしない。

数値補完・理論費用評価まで進む場合は、その実行範囲の明示承認が必要である。前依頼の第12節で「実H6/H8の理論費用の新規計算」「未取得metadataの科学的な再生成」が対象外とされていたためであり、新たな科学レビューを一律に要求する趣旨ではない。
実行前のSOURCE/profile/input bytes・必要資源・出力exclusive/予算確認は、認可範囲に合わせて具体化する。今回のread-only取得を、本計算のgrantや科学的採択へ拡張しない。

## 7. 保全と公開

旧科学SOURCE・counter・native builder・実行worktree・grant/ledger/NPZ/raw/cacheを保持。private archiveを新しい科学SOURCEの実行結果として再ラベルしない。原稿改訂・新科学計算・新grant・subagents0。
軽量reader/tests・報告・索引・receiptだけを明示stage/commitし、non-force pushを試す。認証設定は変更しない。完了報告後mandatory STOPし、未認可のmetadata補完・費用評価・詳細科学レビューを自動開始しない。
