# B：RPE全体の精度・shot数評価の再利用設計

2026-11-13 JST。**供給signalに対する既存統計経路は再利用できる。新しい全RPE N solverを既に完成したとは扱わず、whole-shot-vector family・claim判定・価格付き内側選択を分けて設計した。** 新H4 MC、保存結果再解析、新N最適化、RMSE計算、実Gは0。L1経験的評価やnominalの十分性は採択していない。

[再利用設計](DESIGN.md)／[signal・N・G HANDOFF](HANDOFF.md)／[最小科学実行案](SCIENTIFIC_PLAN.md)／[新7＋decoder回帰](TEST_DECODER_SHOTS.txt)／[controller回帰](TEST_CONTROLLER.txt)／[技術確認](TECHNICAL_CHECKS.txt)。production codeは無変更、新しいframeworkは追加せず、必要な接続を既存APIで検証する人工テスト7件だけを追加した。

## 確認したSOURCEと再利用

| SOURCE/実装 | 確認・結論 |
|---|---|
| S3 `2084791b1b09cff65e07fa8566bb734e1e6bff8e` | 固定1配分の経験的条件付き結果・未資格/null。runのHANDOFF/REPORTを参照し、trial原本/Nを再集計していない |
| R3 `7f1f9053dfbbb81646ac9e483a0354d01cbec535`、B統合`ab12b69120323379b0aae572e7f62692a5819125` | math.atan2/元scalar branch・zero/tie・exact count変換を保つbatch hook。現Bに存在し無変更で再利用 |
| A調査 `97146481f7b05e9418d8fc016f133c7b78d93d08` | 全事象loss/population risk・target/raw/law移送、直接全riskと既定Minkowski個別budgetの違い。重複N理論/物理評価器を作らない |
| B比較 `51500b724a75f9535a2c831d50bcfe09eabb9326` | 4方式11案、N/C資格、共通target/価格/P0、保存signal/receiptの再利用。menu未採択を保持 |
| 直前B `cdf6a80671b2d0bd58983fbbbaf7334f58dc8422` | 価格精度・同時区間/ΣNb・N未資格の扱いを今回のG接続へ再利用。既存branchを継続 |
| `rpe_decoder_statistics.prepare/simulate_prepared/summarize/trial_budget/exact_counts` | 任意の供給mean＋whole-shot vectorを扱える。MC/解析/十分Nの区別を保持。13round count直積の完成risk solverはない |
| `pr_rpe_rmse`、`n_from_a`、V2 `assess/cost/select` | nominal配分は十分Nでない。診断packetのN_X/Yはnullに保つ。適格packetだけexisting exact費用へ渡す |

## 主要な設計判断

1. **候補生成とsignal計算を分離**。Bは全round raw/attenuation/誤差/SOURCEを受け、物理actionを再呼出ししない。meanのV2 identity/law/target照合は既存prepareを使用する。
2. **Nは全scheduleベクトル**。元decoderは前roundの推定を使い、早いbranch誤りが後段へ伝播する。round別の独立最適化は仮定しない。初期案は供給base×共通scale等の有限低次元familyで、scale値・X/Y制約は未採択。
3. **評価と資格付けを分離**。conditional RMSE/未解決率を得ても十分Nは出さない。必要Nの認定には採択loss/claimに対する有効population上側riskと入力適用誤差が必要。point RMSE、L1ラベル、局所phase/Tでglobal PASSを作らない。
4. **N選択はGを目的にする**。同じ物理configurationで複数allocationを評価し、適格Nと同じcandidateのcompiled価格を使う。価格未取得なら資源winnerを出さない。normalizationや価格標本数を二重乗算しない。
5. **既存selectの入力制約を明示**。現selectはunique physical configuration IDに対しevaluation1個。allocation違いを物理IDで偽装せず、将来は内側の有限allocation表を通常metadataで管理し、1代表＋未探索域を外側へ渡す最小接続が必要。今回は科学policyが未確定なのでoptimizerを先行実装しない。

## 検証結果

**計69件PASS**：新7、既存decoder統計16、A高速decoder12、B batch結合5、controller17、evidence planner12。8次を検査する旧1methodは除外し、新8次の接続・検証は行っていない。作者自身の技術確認で、独立科学レビューではない。

新7件は、同physical config/異whole allocation、SOURCE/identity/law/target不一致拒否、欠測・未資格gate、人工MC診断のN昇格拒否、Gとshot総数の異なる順序、巨大整数N保全/backend上限、N偶奇によるzero massの例を確認する。

人工例では全2round・両軸同Nの(1,5)は12shot、(8,1)は18shotだが、価格を1/100と置いたmock費用は1002/216。これはshot総数最小とG最小が異なる例であり、実H4候補や最適化の結果ではない。また人工M0/μX=μY=0でN1からN2へ変えるとzero事象のmassが現れるため、decoder規則だけでresolutionの単調性は仮定できない。

実H4の保存mean/価格/N/S3 trialを新しい数値評価へ渡していない。raw/model/error/sourceの受入仕様は既存形式、未解決loss/claim/statistical precisionはユーザー判断に残す。現在の十分N/global bound/G/winnerはnull。

## 次の担当・STOP

**signal担当**が候補正本＋全round mean/normalization/誤差を供給。**ユーザー＋B/A**がclaim・loss・有限shot family・精度/MC反復scopeを決める。**Codex B**はその明示許可後に、供給signalからのみ全RPE統計/N資格を評価する。**C/D**は同candidate/law/loweringの平均compiled価格を供給する。候補物理生成/新signal/CモデルをBで重複実装しない。

今回はcommit/push・remote SHA/tree/主要成果物照合後mandatory STOP。新科学実行、保存再解析、shot最適化、実G比較へ自動進行しない。
