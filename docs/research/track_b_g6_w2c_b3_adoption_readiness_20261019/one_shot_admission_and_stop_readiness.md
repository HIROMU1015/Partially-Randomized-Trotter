# External admission・one-shot・failure/STOPの技術準備

**本番portの実装・有効化・実authorization/grant/marker/claimの発行／消費は未承認・未実施。**
既存driverと新しいread-only準備guardのscience entryは、今回の未採択候補を拒否する。
新mock portは`admit_mock`だけを持ち、driverが要求する`admit_once`を持たない。返すものも本番receiptではない。
`ARTIFICIAL_*`識別子、private `b3-artificial-*` /tmpでのみ動く。通常の科学namespaceを受け付けない。

## 1. 本番に必要な結合（設計、未実装）

| 結合対象 | 固定する値／検査 |
|---|---|
| 正式契約 | 採択されたR1または別profile、exact canonical contract digestと公開commit |
| 科学input/policy | input digestとH/K_root/η/ρ/error/confidence/root/ceil/phaseの不変 |
| 科学SOURCE | 既存10 pathの完全SHA256、import実path、読み込み前後の照合 |
| driver/guard/port | 既存driver、新準備guard、将来の本番portの完全SOURCE SHA／release commit |
| schedule | 16 cells、保存N、correctness16、二pass、24 constructor、保存上限を含むhash |
| call budget | 256＋15,855,296＋15,855,296＝31,710,848。256を二重加算しない |
| authorization | ユーザーの実在する明示承認への参照と内容hash。boolや候補の自己申告を根拠にしない |
| one-shot | scenario scope定数と別の一意なlogical scope ID、fresh claim ID、未消費の検査 |
| output/marker | 指定専用namespace、symlink/path逸脱なし、新規output、旧marker不操作、二重起動拒否 |
| STOP | 初回科学構築より前にscopeを予約。成功／失敗／中断とも同scopeのretry/resume拒否 |

本番adapterはconstructorで正式契約・SOURCE root・output・authorization locatorを保持し、
`admit_once(contract_digest)`で全てを実在照合する。既存Boundaryはportを信頼するinterfaceなので、
そのreceiptのboolだけを本番authorization確認だと扱わない。
source、input、selected angle、seed、全cell/call予算、output/markerをclaim前にpreflightし、
atomic reserve直後にも結合を確認する。driverの再source照合・exclusive output・各guardは残す。
TOCTOUで衝突／変化した場合も上書き／rollback／再起動をせず、その束をSTOPする。
今回これらの本番binding値を発行していない。production commit／formal digest／authorization／fresh実IDはnull。

## 2. Atomicモデルの人工検証

mockはscope keyをcontract digestから独立させ、scopeとclaim IDの双方をO_EXCLで一意に予約する。
同一scopeの32 thread要求と、別の32 owned-process要求の各人工raceで、それぞれ1件だけがsessionを得た。別adapter、契約digest変更、別scopeでの旧claim ID再利用も拒否。
二ファイル予約はtransactionと同一とは主張しない。途中で片方だけ書けた場合はその予約を消さず、
sessionを返さない。空／partialファイルもspentとして扱い、fail closedで同scopeを焼く。
正常なsessionの返却は両bindingの永続化後のみ。mockのfsyncはprivate local tempだけで行う。
本番のfilesystem／再起動／クラッシュ耐久性について、このprivate tempの人工thread/process raceを証明として流用しない。
本番portを実装する際は同じat-most-onceとbinding条件を、採用storageで再度人工検証してからfreezeする。

## 3. Failure prefix・cost・retry0

入力／SOURCE/input SHA、contract digest、angle採択、seed、cell/N/call予算、承認、旧claim、二重起動、
partial reserve、partial trial、保存失敗、output衝突、STOP後retry/resumeを人工検査した。
mockは供給された人工eventを記録するだけで、generatorやbitstreamを呼ばない。
失敗はZERO_R/ZERO_Cと区別し、完了prefix、row/hash、fake clockの消費費用、保存失敗情報を保持する。
失敗trialのfake費用も合計へ残し、0とは置かない。自己control書込費はnullと明示する。
保存失敗時に既存prefixを上書きせず、同じ保存操作やsamplingを自動retryしない。
scopeの予約自体が再開を拒否するので、STOPログが書けない技術障害でも勝手に再開しない設計である。

## 4. 初回correctnessのgate

本番は16 cells×16 trials＝256。event、phase、元rawと逆時間word、πW、A/B trace、
F3/canonical audit、要求angle、source/state/output identityを検査する。
各cellにacceptedが1件もなければSTOPし、coverage目的のtrial追加は行わない。
既存engineを人工all-zero endpointへ接続した検査では、最初のaxisの16 trialsでSTOP、17回目を呼ばなかった。
raw zero/coin rejectの両方が実prefixで観測される保証を要求せず、人工path coverageと実観測coverageを分ける。
今回の実G7 correctnessは0。実G7接続・全N・科学的accuracy・CPU費用の検証ではない。

## 5. 残る本番作業の境界

allowed scopeの設計・mock検証は完了。本番portの実装・authorization実在確認・live scope store接続と
本番クラッシュ／process間検査は未実施で、今回の「未実装・未承認」という境界を維持している。
これらは新科学計算なしに実施できる後続技術作業だが、明示的な準備承認後に行う。
実装後はSOURCE・正式契約を完全SHA/digestで固定し、それを対象に最後の科学実行承認を受ける。
今のmockを本番portとして渡したり、未固定の将来SOURCEで科学実行する承認文にはしない。
mandatory STOP。
