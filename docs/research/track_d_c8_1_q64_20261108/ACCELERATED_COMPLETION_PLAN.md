# C8-1：利用者承認のCPU予約修正・残task引継ぎ

2026-11-08 JST。ユーザーの「改善して 引継ぎまで行ってよい」に基づき、旧一campaignの未着手taskだけを新SOURCEへ引き継ぐ。元PLANのSOURCE切替・未着手処理継続に関する制約を、この特定の引継ぎについて変更する。科学条件・精度・cohort・seed・events・cut・P0・32 compile総上限は変更しない。

## 問題と変更

旧SOURCE6b1d613dはcompileをkind=compiledとしながらmethod=referenceのまま既存allocatorへ渡したため、referenceを重いnative taskとして31 CPU枠ずつ予約していた。P0 compileは実際には1 threadで、4 compileが64枠を予約する状態だった。

新SOURCEはC7-3高速化で既に使う方式を再利用する。スケジューリングproxyだけmethod=symmetric_directionalへ正規化し、raw/postはkind=raw、compileはkind=compiledにする。元task objectとmethodを保持するので回路やbaselineは変わらない。**compile予約は各1、raw/postへ残りCPUを配る。** compiler/数値backend/cache/全256列/位相/精度は変更しない。

## 旧runの調整停止と保全

旧controllerだけへSIGINTを一回送り、新規投入を止める。workerへsignal/killを送らず、ProcessPoolExecutorのactive task完了を待つ。旧SOURCE/HEAD/入力を変更しない。旧STOP_FAILURE/KeyboardInterrupt記録を消去・PASS化せず、承認された調整停止として新来歴から参照する。

保存8 path/512 occurrence/16384 events、sample00 raw PASSの完了済み入力を、compileのdrain中にtechnical/fixed_prior_evidence.jsonへhash固定する。未完了compiled出力の未知hashを作らない。新SOURCEを先に固定できるが、新controllerは旧STOPを待つだけで、新しい科学taskを投入しない。

旧STOP成立後、sample00 raw PASS、実行中だった4 compileの正常完了、旧SOURCE/input前後一致を確認する。STARTED未完了compile、raw/post科学失敗、不整合があれば引継がない。固定済みpriorを再照合し、drain完了時の全継承hashを新出力のinherited_evidence.jsonへ保存してからraw/event/QPY/NPZをlocalへbytes一致でコピーする。コピー後・終了時も旧原本/STOPを照合する。待機中はSTOPの有無だけを10秒間隔で確認し、原本の繰返し大量hashを行わない。

## 許可された残task

samplingは追加0。sample00 raw再実行0、旧4 compile再実行0。完了済みpostがあれば再実行0。sample00の未実施postとpairedχを完了し、全数値gate PASS後だけ残7 raw・最大28新compile・未実施postへ進む。旧新合算raw8、compile最大32、post最大32。correctness failure後の再実行ではなく、ユーザーが許可した未着手task継続である。

q64/cut32/同H4/CDF8/LD2/S2/s1/r32/K0/h=.1、元H厳密ground仮定・準備費除外、component1e−10/observable2e−10/rtol0/phase fitなし、同P0、強い対照の63境界融合、Reference一gadget/62内部融合/χ/actual inverseを維持。A/F/S予測・係数・共分散・CX候補−14128と既存評価規則は不変。新fit/N/全RPE費用更新/q≥128/actual保証/詳細レビューは0。

## 新namespaceと来歴

新branch/worktreeを使い、出力は`artifacts/track_d_c8_1_q64/2026-11-08/run01_accelerated_completion`。実行前に本PLAN、新コード/人工テスト、旧runの完了済みprior hash索引をGit固定する。残るcompiled入力は旧SOURCEで正常完了したbytesを、新しい科学task投入前に固定する。旧sampler SOURCEは6b1d613d、rawはsample00が旧SOURCE・残7が新SOURCE、compile4が旧SOURCE・残28が新SOURCE、postは実際のSOURCEを別々に記録する。旧eventのrequest identityは保持し、価格recordにはraw requestとcompile/post SOURCEの対応を明示する。

CPU/メモリは実affinity/quota/MemAvailableとtaskメモリadmissionを再利用。compile/postの初期メモリestimateは旧初期値を維持し、rawはsample00実測peakの2倍を反映する。これらは人工的な使用上限ではなくadmission estimate。共有環境・他jobへ変更しない。

## 検証と終了

人工テストで4 compileが4 CPU枠、compile1＋post63等の予約、raw/postへのremainder、旧raw1/compile4再利用、7 raw/28 compile/32 postの一回性、sample00 gate、未完了/科学失敗拒否、旧STOP保全を検証する。実H4の性能を人工PASSから主張しない。

不一致・精度超過・実際の資源継続不能・保全不成立ならpartialを保持しSTOP。追加SOURCE切替や科学retryを自動実行しない。最終C8-1結果と長q戦略を公開可能scalar/hash/文書にまとめ、remote SHA/tree/public bytes照合後mandatory STOP。次の研究判断はユーザー/GPT。
