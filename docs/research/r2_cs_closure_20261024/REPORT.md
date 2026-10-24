# R2-CS：正式研究終了・成果保全記録

2026-10-24。研究ID **R2-CS-20261024**。ユーザーは固定C0.6の提案Cを正式採択した。本記録はその決定の反映であり、科学結論の再解析・再採点・詳細レビューではない。

**正式終了status：`R2_CS_INDEPENDENT_SELECTION_RESEARCH_CLOSED`**

終了するのは、PF次数・q・deterministic/random partition等から低費用構成を事前予測・選択する**独立アルゴリズム研究R2-CS**である。終了状態を他研究へ伝播させない。正式採択、scope、固定根拠、P1の最終状態は [decision_record.json](decision_record.json) に記録した。

## 経緯と直接根拠

|段階・固定commit|その時点の位置付け|今回の取扱い|
|---|---|---|
|[C0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8cb41c40aec9c31c69a5a50e8e5fe93aa6d2eb88/docs/research/r2_configuration_selection_c0_20261024/REPORT.md) `8cb41c40…`|初期設計。構成長と有料適格性照会、既存R3との差分不足、P1非実行案|初期仮説・先行研究・旧契約を歴史資料として保全|
|[C0.5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25cbd26d27057bd3d5af9c18d1c4b7be13d9cedf/docs/research/r2_pr_structure_c05_20261024/REPORT.md) `25cbd26d…`|提案B。有用な既存構造はあるが、独立選択法の差分は未確立。P1 HOLD|横断索引・反例・再利用資産を保全|
|[C0.6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dee25ba13ee8ebcb23c7d294c659ea678d818277/docs/research/r2_one_sided_c06_20261024/REPORT.md) `dee25ba1…`|最終限定調査。3候補の成立部分は既存理論・実装の範囲。提案C、当時は正式採択前|**正式終了の直接的な科学的根拠**。今回ユーザーが提案Cを採択|

旧REPORT・旧STOPの「未採択」「HOLD」は各段階の履歴であり、誤記として書き換えない。本記録がR2-CSおよびP1の最終的な方針を追加する。固定commit/path/SHAと必須資料はdecision recordから取得できる。

## 正式に採択された終了理由

1. 既存R3、SPRINT、cost-aware sampling、一般multi-fidelity最適化との重複が大きい。
2. G6・finite-RTEの有用な構造は既存成果であり、第2研究の新しい選択機構として再計上できない。
3. 安価な情報から分子PRの適格性・総費用上の劣位を判定する、新しい独立機構を特定できなかった。
4. 一般の上下界比較、既知誤差保証、law情報共有だけでは方法差が不十分。
5. C0の16 endpoint P1を実行しても、この方法的新規性の不足を解決する見通しがない。
6. 同種の調査・pilot・候補拡張の反復を現時点では推奨しない。

これは**一般的な構成選択法の不可能性を証明したという意味ではない**。現在の資産と確認済み先行研究の範囲で、独立研究として継続する十分な根拠が得られなかった、というユーザーの終了判断である。

## 保全する知見と未確立事項

同じR=qrの条件付きlaw共有、finite-RTEの係数・normalization・signed pieceと整数allocation、G6/P5の局所query・有限構成、既存の上下界・誤差保証、情報取得費用とOracle漏洩の制約、既存コードの再利用先を資産として保持する。数式・SOURCE・raw結果の大量複製はしない。[保全概要・資産参照](preservation_summary.md)

これらから新しい安全除外・適格性・総費用選択の独立機構へ接続する点は未確立だった。取得可能性と安価さ、十分予算と最低shot数、scalar lawと非可換path、点推定と認証上界の区別を保つ。新たな普遍則・新規性・性能値は採択していない。

## P1の最終取扱い

`P1_FULL_SAVED_INFORMATION_VALUE` は **`REMOVED_FROM_R2_CS_EXECUTION_PLAN_UNEXECUTED`** とする。

- 正式な研究継続候補から除外し、本研究の実行予定に含めない。
- 未実行、科学SOURCE未固定、grant未発行、新数値再解析なし、自動実行なし。
- 旧C0のpilot契約DRAFTとC0.5/C0.6のHOLD・STOPは変更しない。
- 将来の類似検証にも、独立した研究契約と明示ユーザー承認が必要。旧P1の承認を復活させたとは扱わない。

## 他研究・将来の扱い

Track Aの正式採用結果・データ整理・論文化保留、PR-2のmatched accuracy/資源比較/transfer、G6・P5の数理と生成法、他Track Bの限定成果・否定・休止・STOP、R3の`STOP_R3_NO_METHOD_DELTA`は維持する。G6やPR-2の結果を否定的に再分類せず、他の研究テーマも開始・採択しない。

将来の再検討には、既存法との差を持つ具体的命題、安価に取得できる非自明な誤差/費用certificate、または新構造と構成選択上の利益を結ぶ明確な研究課題が必要となる。これは将来の再評価条件であり、新研究計画・計算承認ではない。具体的新規性がないままpilot・proxy探索を再開しない。

## 作業実績と終了

新科学計算、保存値数値再解析、費用比/regret/順位/統計値算出、simulation/sampling/回路生成/compile、学習/最適化、詳細GPT/Claudeレビュー、旧STOP解除、新研究採択、論文・新図表はすべて0。実施したのは終了記録、固定参照整理、metadata/保全検査だけである。

既存C0/C0.5/C0.6の検査を再実行せず、既存管理helperを使った [終了記録の技術検査](verification.json) 16件はPASS。固定参照33件と保全資産8群を接続した。科学的再検証や独立レビューを意味しない。第1研究の資料・既存索引を含む既存ファイルは改訂せず、本終了ディレクトリだけを追加した。

共通運用規約の指定SHA原本は未取得のままで、取得可能版のSHA・§10.3欠落をdecision recordへ記録した。現在の明示ユーザー指示を適用し、未取得原本を読了済みとは扱わない。

**[mandatory STOP](STOP.json)。次の作業なし。** 公開・remote HEAD一致・worktree cleanを確認して終了する。研究判断、科学計算、別研究、詳細レビューを自動開始しない。
