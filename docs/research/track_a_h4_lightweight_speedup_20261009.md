# H4の軽量高速化・限定同等性確認（2026-10-09）

距離ごとの共通準備の再利用と、ledgerの変更キーだけの差分保存を実装した。
限定人工48テストに合格した。**本計算は起動していない。実測の高速化率は未確認。**
累積chargeの再実行見積り12.261694GiBは現10GiBを超えるため、plan未seal、approved/runtime_authorization=falseを保持する。

資料入口は[固定bundle](../../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/README.md)。
起点REVIEW `f25284607fd5d52da3b7fbddebee3005a22f57de` と旧SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、
旧worktree・停止証拠・one-shot・課金を保持した独立worktreeで作業した。

## 固定sourceと実装

branchは `track-a-h4-lightweight-speedup-20261009`、SOURCEは `4d2d1492fc23d0736c305533d78967cc1db8a7c8`。
[source closure38・旧/new blobとSHA](../../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/source_freeze_v1.json)を固定した。
SOURCE commit対象はsignal.py、execution.py、ledger.py、新人工test、新限定runnerの5file。
schema、科学式、登録templates、compiler options、serialization規約、monitor間隔・5秒制限・全資源上限は変更しない。

`signal.GeometryPreparation` をdriver generator内で距離ごとに一度作る。
one/DF12の13ブロックの対角化・basis、Operator、dense actionとmethod/L_Dに依存するprepare結果を同じ距離内だけ再利用する。
原入力はread-onlyとして、別geometry・入力object置換・writeableへの変更を検出してSTOPする。
coefficientsは変更禁止、戻り値の小さなlist/dictはコピーし、回路builderが共有basisを変更しないことを確認した。
tailの係数順・加算順・符号・constant・dtype/shapeを維持し、浮動小数点加算の組み替えはしない。
同じrunのdriverメモリ内だけの再利用であり、永続科学cacheや旧partial結果を使わない。

| 距離ごとの処理 | 旧 | 新 |
|---|---:|---:|
| one/DF block準備呼出 | 13×218 = 2,834 | 13 |
| method/L_D別の全prepare | 218 | 10 |
| ledger deltaのhistory走査 | 全entry/reservation | 変更entry/reservation各最大1 |

218全件のoldブロック数は静的な算出値で、旧218回の重いprepare campaignを再実行していない。
10種類の旧prepareと全218新テンプレート結果を比較した。新13ブロック呼出は人工fixtureで観測した。
登録templatesに対するcacheのdense/Operator/tail ndarrayは30MiB。basis回路・Python metadata・scratch・元入力はこの値に含めない。
driverのAS/RSS8GiBと既存監視は維持し、実入力での全driver RSS・速度は未測定。

ledgerはsingle driver writerのlock内でreserve/completeが変えたキーを直接deltaへ渡す。
初期空delta、reserve前のdurable予約、record→completion delta→headの順とcanonical bytes/digestを維持する。
saved-history全コピーを除き、campaign全体で差分生成にかかる反復走査をなくした。
最終auditの全chain・record・owner・orphan検査は保持する。失敗時のRESERVED課金・STOP・返却なしも維持する。

既存の12常駐worker、candidateをまたぐ上限12のqueue、完了workerへの補充、結果の順序・内部thread1は維持する。
この変更はdriver側の重複処理を減らすもので、transpile自体の高速化率や12倍加速を保証しない。

## 限定テストと限界

[実行前計画・結果](../../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/limited_tests_v1.json)は48件PASS、16.052秒、当該process peak RSS250,023,936 bytes。
単一test process・内部thread1、AS2GiB/RSS512MiB、wall240秒、output16MiB上限。
所有範囲は当該test driverとhomeのexclusive evidenceだけ。12workersはmock、observer/実worker/affinityなし。
旧SOURCEのsignal/ledger Git blobをreferenceとして直接ロードした。

- 全218templatesのprepare: 256×256 complex dense/tailのdtype/shape/bytes、constant、support・coeff・sign順を比較。
- 4method代表×cosine/sineの8組: 8-system+ancillaの9qubit wrapper canonical bytes/digestを比較。4method代表のfinite corrected signalも一致。
- 別の256×256 complex UnitaryGate parameter case: 9qubit人工wrapperをbounded chunksとdigestで比較し、共有basis parameterのbytes/writeability不変を確認。
- 同一距離cacheのreadonly・入力置換・別geometry拒否、失敗prepare未cache、nonfinite STOPとsigned zero保持を確認。
- 旧/new ledger13fileのbytesとchain、予約carry・順不同完了・合法ownerリンク・二重完了・crash windowを比較。
- 10,000件のmock historyで全件iteration/itemsを禁止してもdelta保存が成功することを確認。
- 既存signal/ledger/12worker mock queue回帰を限定実行し、candidate順序、metrics、pending owner、worker死亡、monitor failureと予約capを確認。

最初の人工試行はdriver wiring mockにconsumed_secondsが欠け1errorとなった。fixture修正後47件PASS、
dense UnitaryGate caseを追加して最終48件PASS。全試行証拠はhomeに保持する。
標準gateを使った人工basisであり、Gaussian/OpenFermionを実行していない。
全218実wrapper、実Gaussianの性能、旧compiler output完全同一性、実H4 signal/compile成功やworker占有率は未検証。
追加transpile・実NPZ科学array読込・分子/SCF/DF/input生成・GPU・production起動は0。
既存28/64 synthetic・128件benchmark・受領native proof campaignは再実行しない。

## binding・承認・予算

[plan/auth/review binding](../../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/binding_audit_v10.json)をactual SOURCEへ更新した。
旧環境との差18version/45raw RECORD（normalized22差）の記録とcandidate environment/compiler profileを保持し、
既存venvを変更せずlive metadata一致を確認した。旧compiler output同一性は未確認のまま。
受領済み6NPZ・freeze・旧native control83件と新host停止14identityのreceiptをbyte-onlyで結合した。
旧hostの過去exit code/正確な終了時刻の未記録nullは今回から補完しない。

carryは20 actual /4,428,938,712 bytes /5,472.345380863175秒、
承認済みactual cap74,804・残74,784。全74,784 logicalは最悪74,784 actualを要し、節約は保証しない。
既承認worker12 CPUs [2,4,5,6,8,9,10,11,12,13,14,15]、driver16、observer18、内部thread1、
observer AS256MiB/RSS64MiB/admission120.25GiBと候補environment採用は保持する。
今回はそのCPUを使用せず、本番roleを起動していない。

output/controlはhomeの新evidence内の未使用pathに結合した。
SOURCE変更後のseed identityは将来再結合が必要で、旧random/partialと混合しない。
新cacheはdiskへ保存しないため保存形式/最大observer trace/charge/inode見積りは不変。
累積worst charge13,165,893,832 bytes（12.261694GiB）>現cap10GiB、physical新output案5GiB/301,000 inodes。
再実行planは未seal、auth/review approved=false/runtime_authorization=false、absolute_launch_command=null。
このSOURCEには10GiBを維持するgateがある。13GiBは以前の未承認提案のまま、この依頼では採用しない。

残る条件は[従来の再実行案](../../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/relaunch_proposal_v1.json)の累積charge問題を解決し、
変更後のSOURCE/binding/独立最終review・fresh CPU/memory/pressure/OOM/fs/block/inode/quota・未使用root/one-shotを確認して、
利用者の一度のmap再実行指示を受けること。今回の速度改善を容量問題の解決とは扱わない。

## 独立reviewと公開

修正authorとは別の担当による[独立review](../../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/independent_speedup_review_v1.json)はIMPLEMENTATION_SCOPE_PASS_LAUNCH_BUDGET_FAIL。
source closure38、旧/new演算・ledger順序、限定結果のscope、profile/input/stop/carry/CPU/caps/digest整合を独立確認した。
blocking実装findingsなし。担当は既存48人工結果を原本と照合し、test・本番起動を再実行していない。
実装合格をlaunch予算合格やproduction成功とは扱わない。

SOURCEと軽量REVIEW_BUNDLE commitを分離し、[commit対象一覧](../../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/commit_inventory_v1.json)を固定する。
NPZ、実runtime/checkpoint/cache、raw test log、credential・内部SSH情報・home準備utilityをcommitしない。
originはnon-force pushを試み、認証失敗時は設定を変更せず手動commandを最終報告する。
actual REVIEW_BUNDLE/remote SHAは自己参照を避けhome publication receiptと最終報告で記録する。

## 2026-10-09 H4利用者が13GiB累積charge・一度の再実行を明示認可

[13GiB改定と一度の再実行認可](track_a_h4_approved_relaunch_20261009.md)へ継続。利用者が再実行budget問題を承認し、新SOURCE/output改定schemaを固定した。上記の10GiB不合格はその時点の履歴として保持する。
