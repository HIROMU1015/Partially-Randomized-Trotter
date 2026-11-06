# A1 N3範囲見直し：752-shot予備検証案・設計STOP

**設計・人工検査のみ完了。新しい科学計算/測定/compile/再実行、現行jobの停止/変更は0件。752-shot案は未採択・未実行。**

## 推奨案

[範囲見直し](N3_SCOPE_REDESIGN.md)と[契約草案](N3_PILOT_CONTRACT_DRAFT.md)に、Step 1の学習成立・Step 2のA1対強いB1-OL予備比較・Step 3の独立保証付き本検証を分けた。
対象は保存linear H4/1.00 Å/STO-3G/legacy DF12、8system qubits/full4-electron sector、L_D3/m0/q1/δ=.02/r1/K2/atol0、X/Y、N1 actual Q/selected full・N2保証ground・certified exact Fraction engine。
既存3-controlと全5正層を維持し、B0/B1-Oを初回pilotから外す。

第一候補は新train5×16×2=160、B1-OL配分probe5×8×2=80、fresh評価128×2方式×2軸=512、合計**752 shots以下**。
既存の新B学習結果は参考資料として使い、旧397件・現行campaignのraw data/係数を新pilotへ暗黙に混入させない。primary案はfresh train込みである。
B1-OLは全5層にfloor4、残り108/軸をexact質量・probeのvariance proxy・平均RZで一回配分する。実入力へallocatorを適用していない。
各単位512 cal、十分N、prod8192 cap、4方式8座標保証を持ち込まない。旧固定train/pilot/cal10752に対して約93%、旧最大76288に対して約99%小さい新予算案で、旧支出や実行時間/検出力が同率で減るという主張ではない。

## 技術成果と人工test

既存のConditionalSampler/bit-only facade/freeze/range/disjoint/価格/provider/worker/managerを将来再利用する。
追加helperは[a1_n3_pilot_design.py](../../../src/trottertracks/algorithm_codesign/a1_n3_pilot_design.py)、人工testは[test_a1_n3_pilot_design.py](../../../tests/tracks/algorithm_codesign/test_a1_n3_pilot_design.py)。科学実行runner/CLIは追加していない。
既存rational/integer/root_intervalとn3.production_meanを再利用し、5層で同じcontrol patternを持つ二つの層を誤って合併しない。
[人工11件PASS](ARTIFICIAL_TESTS.txt)：tiny positive層の保持、variance/priceに応じた配分、zero variance/price guard、quota/target/stream不一致、A1凍結補正と独立性、5-cell合成、real binding/RNGを起動しないこと。
人工PASSは科学的なCV改善・厳密精度・期待費用改善の証拠ではない。future driverには外側campaign provenanceの検査、stream/SOURCE固定と保存/会計接続が必要である。

## 現行jobと承認

[CURRENT_N3_STATUS.md](CURRENT_N3_STATUS.md)に実SOURCE ee49b53f…・実worktree・契約・親/manager/worker観測と、承認後のみ使う停止手順を記した。現行jobは観測時RUNNING、A1 X/Y train各320・rank/range検査が完了していた。
最新の取得時刻・stage・bit/価格数はartifactのCURRENT_OBSERVATION_FINALを参照する。設計中も旧許可されたjobは独立に進行しており、その増分を本taskが新規起動したものとしない。
次に必要なのは、(1)現行新Bの正規停止・回収・保全への明示許可、(2)752-shot条件/seed/配分/診断方針の採択、(3)新SOURCE/環境/薄いrunnerを固定した後の有限実科学実行許可。
今回のユーザー指示は「今回は現行ジョブの停止…を許可しません」としているため、停止操作を行っていない。一般的な技術修復許可を今回の停止/再開へ流用しない。

## 限界・保全・STOP

小pilotは分散/価格の不確かさ、zero/極小層、整数floor、startup費の影響によりINCONCLUSIVEを返し得る。少数測定でCVが有望と必ず判断できる設計ではない。
既存a53cの構造上の留意点（3-controlが各5層内で定数）を継承し、異なる精度の費用差や小さいsample varianceを正式優位にしない。
科学条件・SOURCE・旧結果/STOP/grant/ledger/marker、元main/dirty/untracked、C1凍結と現行jobの実装/入力を保全した。既存の原本は上書きしない。
専用branchへ公開し、remote SHA・本文・blob/treeを照合後、本設計taskはmandatory STOPする。**現行科学processは停止させない。**
