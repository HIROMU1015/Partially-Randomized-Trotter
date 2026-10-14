# Parent allocation：採択済み科学設計の実行source準備

2026-10-15 JST。判定 **READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION**。
登録比較の実行承認ではない。科学的採否、新規性、継続・終了は判断していない。

## 正本と停止位置

[GPT科学レビュー原本](../../research/track_b_parent_allocation_scientific_design_review_20261014.md)を変更せず保存した。
41,911 bytes、SHA256 `11d1e389190091221fd2a282e931c229a52bda1dcc8c8f30ec5d3bcf7aa853bf`。
入力保存commit `117000dfbe7e0a8f8166303c8c0a24049dbd19ee`からGitHub再取得してbyte一致を確認した。
技術基点は`bf1366ac5832136ab8038d160e9a0285da69e79e`、従来adapter/native bridgeは変更していない。

[固定契約](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v1/execution_contract_v1.json)に科学レビュー§6–14を反映した。
[manifest](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v1/source_manifest.json)がsource・依存library・契約・原本・入力のbytes identityを保護する。
今回の固定source SHAは、公開確認receiptの`science_source_commit`を正本とする。
公開確認commitと将来のauthorization-only直接子を混同しない。

## 固定した科学条件

| 項目 | 固定条件 |
|---|---|
| 入力 | 既知developmentの3-qubit distinct-basis provider、p=(1/5,3/10,1/2)、x=τ=5/7、σ=+1、状態000、r=1 |
| target | 共通finite P5(-ixR)、physical exp(-ixR)。基底状態・exact signalは選択やshot削減に使わない |
| 中心／参照 | ordinary／closed P5、partial-return-tail／closed P3+tail／matched CTS |
| 配分 | 各中心common、全3頂点、KKT surrogateの有理点、半移動点。重複は記録して削除し補充しない |
| proposal | canonical、全positive supportのTが正ならglobal解析IS、共有κ=C*/16,C*,16C*のglobal正則化IS |
| C* | 五つの旧common baselineを新しい共通canonical規則で会計した期待Tの最大。新価格前にfreeze |
| 精度 | primitive 1e-6、root256/probability160 bits、rho=eta=1e-12、guard/atan100dps |
| accuracy/confidence | axis誤差1/200、選択した一complex taskのfailure≤1/20、axis各1/40。law数で割らない |
| primary | 十分Nによる期待logical Tのh=0切片。CX/native1Q、N、range、moment、h感度を別々に保存 |

再利用cacheはD1の1e-6のみ。D3の1e-7との安い方取りはしない。
原D1 resultは`fc96571881f301435a655f057709df4d6c1678e0`、sourceは`874ca12d347f8f35e5d8d64f65f24abc4a2df523`。
五つのr1保存event/IR、cache、inventory、原contract、保存監査をbytes/hashで結び付ける。
旧N・旧ランキングは再利用せず、旧結果を書き換えない。

## 実装と数値規約

新namespaceは[専用package](../../../src/trottertracks/algorithm_codesign/parent_experiment/)。
[one-shot入口](../../../scripts/tracks/algorithm_codesign/run_parent_allocation_one_shot.py)、
[個別合成worker](../../../scripts/tracks/algorithm_codesign/run_parent_allocation_synthesis_key.py)、
[zero-compute preflight](../../../scripts/tracks/algorithm_codesign/preflight_parent_allocation_experiment.py)を分離した。

1. 承認・直接子・clean・remote SHA・契約・manifest・旧保護対象を確認する。
2. 新namespaceへexclusive markerを作り、isolated ledger ownerを一回起動する。
3. 必要な旧unique sequenceを固定strict guardで一度確認する。metadata変換だけでは使用可能にしない。
4. 旧commonの完全root event T価格をKKT proxyへ渡す。provider/control/phase/literalを含む。新角度価格は渡さない。
5. 候補・C*・key inventory・登録root診断件数を保存しSHA256を固定する。
6. inventoryが許す新keyを各一回取得し、全native価格／strict guard coverageを揃える。
7. 新rootのcontrolled作用を独立ideal定義と照合する。非rootは旧固定証拠を再利用する。
8. 全law ledger、対比較・hのaffine関係、取得費用／classical setup、内外status、STOPを保存する。

KKTは固定cost問題のλ bracketを有理intervalで最大128回二分する。sqrt256 bits。
解をshare denominator 2^64のlargest-remainderで有理simplexへ写す。和は厳密にa。
元KKT解へのL∞距離上界、λ/z区間、実施回数、残差を保存し、native最適点やexact KKT最適解とは呼ばない。
数値intervalが方向判定できなければ既存bracketを保持する。精度不足はtechnical failure、正性仮定不成立は数学的適用外。

canonicalもISも有限event表全体を対象とする。canonicalはdigital係数に比例する全表lawをH bitsへ固定丸めする。
旧階層samplerのproposalを新Nへ黙って流用しない。基礎common係数・event順序の旧adapter互換性は別に照合する。
確率丸めはweight= digital c/qで補償しbiasへ二重課金しない。κはactual TにもKKT乗数にも足さない。
解析ISはzero Tをsupportから削除して救済せず適用外。C*=0では正則化floorを作らない。

誤差は `e_rep=coefficient_error_upper+Σ digital_c*strict_native_error`。
中央方式は変更root／不変nonrootのinterval幅の和、保存参照方式は既存の共通8rho mean誤差envelopeを再利用する。
独立に正当化済みのHermitian contraction Taylor上界 `|x|^6/6!`を共通使用し、axis bias=2*(Taylor+e_rep)。
exact atan targetまで含むRZ guardにangle誤差を重ねて加算しない。
全表ISのmomentは巨大なexact fractionになり得るため、decimal I/Oを65,536桁に制限したcontextで保存し、終了時に元の設定を戻す。これはsolver精度・係数・確率の変更ではない。人工273 atomで既定4,300桁超と復元一致を検証した。
logは既存96項の有理outward interval上端を用い、ceilで十分Nを安全側へ確定する。
Δ≤0とshot cap超過はlawの不適格として記録し、missing price/guard/数値failureとは区別する。

費用は共通literal簡約＋保存sequenceの加法native計数。Wとsigned adjoint、endpointのancilla位相を保持する。
Tはnative1Qに含まれるので合算しない。未知online費用はNOT_MEASURED。
新lawのsampling/lookupをベンチマークする追加samplingは含めない。

## 実行上限と性能方針

| 項目 | 上限 |
|---|---:|
| 構成／law | 15／75 |
| 新positive root synthesis key | 18、各1回、retry0 |
| 使用旧unique sequenceの再guard | 最大36、各1回 |
| strict primitive guard総数 | 最大54（旧36＋新18のbackend guard） |
| 新root event／controlled matrix診断 | 最大36／72（ideal・realized各一回） |
| 保存event bindings | 全構成4096、各構成273 |
| exact integer decimal I/O | 65,536桁 |
| worker／numeric threads | 2／各1 |
| CPU | key120秒、compute tree＋observer合計3600秒 |
| RSS | key512 MiB、ledger/observer各2048 MiB、aggregate3072 MiB |
| virtual address | ledger/key各4096 MiB |
| wall | phase／totalとも上限なし。backend固定options・tool polling timeoutとは別 |
| output／log | aggregate64 MiB、各log64 KiB |
| cache／sequence | 4 MiB／32,768文字 |
| 十分N | 1e9/axis。clipやprecision変更による救済なし |
| review chunk | 512 KiB以下、UTF-8 lossless再結合／bytes・SHA照合 |

dominant workloadは新合成とinterval guard。固定runtimeを再利用し、独立keyを2 worker、BLAS/OMP/MKL/NumExprを1 threadに制限する。
最大18×120=2160 CPU秒のkey上限に対し、全体3600秒は起動・旧guard・ledger・監査の余地を含む技術予算で、完走時間の予測ではない。
新入力生成によるtiming probeは行っていない。host資源のmetadataと実テスト資源を[準備記録](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v1/)へ保存した。
D2の219,303 pair、全nonroot matrix、旧science runner、whole-circuit compilerは再実行しない。speedup倍率は未測定。

外側observerと周期guardがCPU/RSS/output/logを監視する。key CPUにはPOSIX hard limitも設定する。
Linux RSSは周期観測なので、瞬間peakを完全に事前抑止するとは主張しない。workerのwait4 peakとledger/outerのRUSAGEを終了時に照合し、超過結果を成功としない。
raw key、予約token、start receipt、partialファイル、内外terminalを保存する。全取得完了前のwinnerや失敗prefixの科学利用は禁止する。

## focused validationとzero-compute確認

[試験](../../../tests/tracks/algorithm_codesign/test_parent_experiment.py)の新45件＋既存native bridge回帰26件、**71 PASS、FAIL/ERROR/SKIP0**。
[実行記録](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v1/focused_tests_execution.json)は固定Python runtime、約1.32秒wall、peak69,388 KiB。
local testsでありimmutable CI／外部再現ではない。人工p=(1/7,2/7,4/7),x=1/3、人工KKT、1-qubit controlled位相、mock取得・jobを用いた。

確認：simplex/KKT距離、重複・zero-cost、global dyadic補償、有限confidence会計、欠測、bias/shot不適格、旧metadata/new key、freeze前後の分離、endpoint、raw_word root識別、source/authorization/marker不一致、worker restart、chunk改ざん、保存失敗、alarm復元、資源超過、外側failure優先。
接続テストで発見した「縮約後empty wordの非rootをrootに含める」問題はraw_wordで区別して修正済み。
registered Q/T/U・旧登録sequenceのoperator再評価・新角度・N・ランキング・合成はすべて0。
mock backend callbackの予約試験は実合成ではない。

[zero-compute記録](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v1/zero_compute_preflight.json)でcontract/protected hash/runtime/namespaceを照合する。
既存authorizationはpending、承認済みauthorization-only child／marker／result namespaceは作成しない。新contractのauthorization templateはpendingで保存する。

## 未検証事項と次の境界

登録candidate・角度・inventory件数の実値、旧operator再guard、新sequence価格、新root provider診断、N・比較結果は未取得。
これは準備中に避けるべき結果先取りであり、manifestと固定guardによる実行時の拒否条件を用意した。
人工end-to-end試験は本番provider／synthesizerの成功や資源内完遂を保証しない。
科学条件の変更を要するblocking issueは現在発見していない。実取得でcache不足・key上限超過・科学意味論の問題が出ればtechnical STOPまたはGPT判断へ戻す。

次は利用者の**固定sourceに対する別途one-shot実行承認**。
承認後だけ、固定source Sのauthorization-only直接子Aを作成・pushし、A/remote一致・cleanを確認して一回実行できる。
Aで変更できるのはcontract指定authorization JSONと任意receiptのみ。
runnerは `python scripts/tracks/algorithm_codesign/run_parent_allocation_one_shot.py --source <full S>`。
今回はそのrunnerや科学workerを起動していない。

成功は`PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW`、失敗は`PARENT_ALLOCATION_TECHNICAL_INCONCLUSIVE`。
いずれもretry0・mandatory STOP。配分利益、ordinary/P5差、IS差、参照集合更新を科学GOへまとめない。
結果の重要な科学的判断は新たなレビュー開始承認後のGPTへ戻す。
