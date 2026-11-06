# A1 N3案B：新しい独立campaignの科学実行契約草案

2026-11-06 JST。**提案のみ／未採択／科学実行は未許可。** 本書の公開・人工PASSは新seedや実科学callの承認ではない。

## 1. 固定対象・継承事項

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits、全4電子number sector、L_D3、m0/q1/δ=.02/r1/K2、coefficient_atol0、X/Y。
N1=`744e1a09661ebba6841fe5faf456f88c41c47780`、N2=`5e9bd08068738cdf1070bdc43bee8194b113ae33`、engine=`fdf00cee3c87d804845497afc89ac6a71f3b3f59`。
一次targetは同じactual finite-grid Q・selected full interrogation・保存DF groundによる複素mean。理想gate/独立uniform53-bitモデル限定。
Hamiltonian/partition/preparation/tail/law/policy/state/order/phaseとengine証明を変更せず、再生成・再認証しない。

旧397件・旧係数を新train/cal/prodへ混入しない。旧中断結果d97eと費用・UNKNOWN・STOP/markerをそのまま保全する。
旧を消して新しい完走実験だけを選ぶことはしない。新campaignと研究全体の費用を分け、全体を述べる場合は旧中断費用も含める。
事前判定a53cの全方式完了gateとA1対強いB1-OL主比較を継承する。実現差から期待費用改善を保証したと主張しない。

## 2. 新identity・streamの提案

候補campaign ID：`A1-N3-B-H4-m0-q1-20261106-run01`。
候補fresh output：`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_run01`。
候補master seed：**20261107**。旧20261106のstream/word/stateは使わない。これは数値の提案で、今回生成・採択していない。

PCG64/SeedSequenceと既存sorted domain/spawn規則を再利用し、method×axis×stage×stratum×path/bitを分離して一回固定する。
既存Domain tokenは統計moduleのため維持し、外側ledgerのkeyを(campaign ID, domain token, trial ID, attempt ID)とする。
新seed選別、成功seedだけの採用、旧397件を見たcontrol/予算調整を禁止する。異なるseedのPRNGを数学的に独立と証明した意味にはしない。
乱数の保証は元と同じ独立uniform53-bitモデルに条件づける。path64word/trial、別measurement word1、旧と異なる消費履歴を開始前保存する。

## 3. 継承する採択済み科学条件

| 項目 | 新Bへの継承案 |
|---|---|
| 方式/軸 | B0、B1-O、B1-OL、A1 × X/Y |
| 座標精度/同時失敗確率 | ε_X=ε_Y=1/8、4方式×2軸のα=1/20 |
| Control | 既存g1/g2/g3。希少性を保持し、自動変更しない |
| A1 train | 五層×64 fresh conditional shots/軸、OLS一回、rank4、凍結後D_CV=2+Σabs(λ)≤4 |
| B1 pilot/B0 | B1の各正層64/軸、B0 pilot0 |
| Calibration | 各評価単位512 fresh shots。A1はunconditional Q |
| Production | 独立cal/pilotから十分Nを一回凍結、≤8192/method/axis。A1はunconditional Q |
| 条件付き取得 | actual Qの整数count直接条件付き取得、全正層維持 |
| Confidence/N | 固定03928のcal/prod配分、Bernstein/Hoeffding、外向き丸め、B1のJ8有限配分をそのまま使用 |
| Engine誤差 | 保存された厳密有理数b_engine（表示≤7.8079008e-7）。1/32は採択しない |
| Full価格 | 同selected full no-prep compiled RZ、Qiskit1.3.0/(rz,sx,x,cx)/opt2/seed17。状態準備だけ除外 |

上表の旧採択条件を変える提案ではないが、**新しい独立campaignへの適用と実行自体には明示採択・許可が必要**。
独立train/cal/prod、bit0→+1/bit1→−1、X/Y/phase/order規約を維持する。Born/state/真信号/E0/gap/真varianceをCVへ渡さない。

## 4. 一campaignの有限科学scope

| 処理 | 最大範囲・条件 |
|---|---|
| 保存bundle binding | 同N1/N2/proofのdecode一回、identity/環境照合。preparation/solve/qualificationへfallbackしない |
| Train/pilot | A1計640、B1-O計256、B1-OL計640、B0なし：計1536 shots以下 |
| Calibration | B0/A1各1024、B1-O2048、B1-OL5120：計9216 shots以下 |
| N/配分凍結 | 4方式×2軸で各一回、fresh cal/pilotのみ。cap違反でSTOP |
| Production | 8方式軸×8192以下：計65536 shots以下 |
| 測定・path word | 合計76288 trials以下、trialごとに同exact engine一回、measurement word一回、path word≤64 |
| build/compile/価格 | 実際に取得したtrialのselected fullだけ。cache missのみcompile、hitもshot RZを課金 |
| 結果保全 | 全開始・call・bit・価格・stage・凍結係数/N・費用・失敗を保存。全supportの実path価格列挙はしない |

これらは既存quota/capの有限会計で、実際のN・variance・総費用・所要時間の新規科学評価ではない。
最初のrank/range/identity/probability/word/stream/price/resource不整合で全campaignをSTOPし、baseline救済実行や追加shotsをしない。
旧opt2のhash/旧bias/旧価格/旧stateを無条件に流用しない。

## 5. Manager・observer・SOURCEの固定

管理adapterはqualified `5cb542ece4d208878a1dfc7011007b99297a5a55`を再利用候補とする。
既存cronのown-user one-shot投入、消費済receipt、開始前trigger撤去、application restart=no、外側recorderと親/worker観測を使う。
共有daemonのRestart設定、linger/systemd/cgroupは変更しない。自分の既存crontab/新規jobと競合すれば勝手に上書きしない。

本番Bの薄いentryは、旧Campaign/serializer/測定・統計関数を再利用し、新output/campaign/seedだけを採択内容でbindする。
旧`run()`のhard-coded output/marker/profileを使って起動しない。旧marker削除やnamespace変更によるretryではなく、新しい実験を明示的に一回開始する。
CALL_START/RETURN/EXCEPTION、identity、実child/ancestor cgroup・memory/pressure、worker exit status、STOP_REQUESTED先行保存を接続する。
親/managerのprocess exit0だけを科学成功とせず、semanticなCOMPLETED/STOPPED、STOP/CAMPAIGN_RESULT、全保証/価格の完備性を照合する。

このB entry・SOURCE/依存環境・固定input・compiler profile・stream一覧を人工検証後にcommit・公開照合し、実科学call前に固定する。
今回の人工専用dispatchはscience payloadを拒否するため、本番への薄い入口接続と環境lockは別の技術差分として必要。
技術変更は旧exact kernel/乱数/測定/統計の引数・結果を変えず、float kernel/真信号cache/別control等は含めない。

## 6. 費用・STOP・承認

全段階のfull RZ、cache hit/miss、generation/fit/cal/補正/compile/cache/setup、engine simulation、wall/peak・欠測を分離記録する。
共有cron cgroup値をjob単独peakと呼ばない。未完了attempt/親wait owner消失のexitcodeや古典費はUNKNOWNで、0にしない。
実現campaign差と期待価格/期待総費用保証は別。後者は今回scopeに含めない。

manager/worker異常、source/input/engine不一致、rank不足、D_CV>4、N>8192、word/probability/stream/価格不整合、共有環境への干渉のおそれでSTOP。
receipt/取得bitを保持してから回収し、同じjobの自動再起動・再学習・seed変更・精度緩和・不足shot補充をしない。
終了後commit/push・remote SHA/本文/blob/tree照合を行い、成功でも中断でもmandatory STOPする。

次に必要なのは、(1)新Bのidentity・候補seed/stream方針と上記科学契約の採択、(2)本番entry/SOURCE/環境固定、(3)一条件4方式×X/Y・一campaignの有限科学実行への別途明示許可。
今回、どの採択・科学callも行っていない。N1/N2/engine再計算、追加campaign、18round展開、C1解凍、重要科学レビューは含めない。
