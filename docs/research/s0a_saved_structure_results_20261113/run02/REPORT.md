# S0a run02：保存4path・8軸の構造RZ照合（2026-11-13）

**指定4保存pathのX/Y計8件すべてで、新Streaming Sと元C1の保存Sが整数で完全一致した。** 実入力のSOURCE/identity・preparation・law/basis/phase・signed allocation・event digestを固定runnerで確認し、旧Sを再計数せず対照に用いた。これは一pathの構造価格実装parityのdevelopment診断であり、平均S・平均compiled Cや小さい物理閉状態の検証ではない。

[RESULTS](RESULTS.json)／[PLAN](PLAN.json)／[STOP](STOP.json)／[CHECKS](CHECKS.json)／[実行log](../run02_EXECUTION.txt)／[HANDOFF](HANDOFF.md)。旧[run01の失敗記録](../REPORT.md)と[run01 STOP](../run01/STOP.json)は変更せず保存した。

## 数値結果

比較単位は、**controlled PR + full Hadamard wrapper（measure付き、P=0）の測定1回分に対応するliteral structural RZ S**。X/Yは元C1のcosine/sine順であり、実compiled価格Cとは区別する。

| 系 | T | qPF | R_outer | 元role | S_stream,X = S_saved,X | S_stream,Y = S_saved,Y | 判定 |
|---|---:|---:|---:|---|---:|---:|---|
| H4 DF10 | 0.8 | 8 | 8 | CALIBRATION | 21,664 | 21,664 | 完全一致 |
| H4 DF10 | 1.6 | 16 | 8 | VALIDATION | 46,559 | 46,559 | 完全一致 |
| H6 DF19 | 1.6 | 1 | 32 | CALIBRATION | 158,669 | 158,669 | 完全一致 |
| H6 DF19 | 3.2 | 2 | 32 | VALIDATION | 321,151 | 321,151 | 完全一致 |

candidate IDは順に`6e1efff1ba1f7ac95a3b`、`201d498a3fa5eaa3b5c5`、`4269eafc8c0ef8922c23`、`24fec6bdd44d6a728f1e`。すべて元normal cohortのindex0だけを使用した。元n8のcohortとCALIBRATION/VALIDATIONの役割を変更せず、新平均法のfresh holdoutとは扱わない。新S計数8件・保存S比較8件・旧Sの新再計数0。平均・比率・新しい価格残差統計は算出していない。

## 固定条件・SOURCE

- 実行SOURCE：`9516351e639278349acf55334e838520a7b3833e`。run01のast scope不備を修正した公開版をそのまま用い、今回追加のコード修正・科学retryは不要だった。着手時公開commit：`3f2a696ec98f7ba69529d06094531b80dc837643`、既存C branchを継続。
- C1対照：`2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`の`run01/CALIBRATION.json`と`VALIDATION.json`。元科学SOURCEは`1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`。実行後の成果物照合でも、保存S/identity/role/axis/digestが同じ公開recordに対応することを確認した。
- Linear H4/H6、1Å、STO-3G、保存tol-only DF10/DF19・weighted completion・元fragment順/H/state/snapshotを維持。H4はS2/δ.1/prefix1/K2、H6はm5_best/δ1.6/prefix7/K2・元11 signed piece/負時間/整数allocation。
- symmetric-directional controlled PR、scalar identityは全時間で一回、元full Hadamard wrapper/measure/P0。Qiskit1.3.0、NumPy1.26.4、元compiler profile（rz/sx/x/cx、opt1、seed17とhash）をbindingとして保持した。compilerは呼んでいない。
- 元lawの宣言`ideal_finite_RTE_law`と実sampler`PCG64_numpy_float64_CDF_saved_cohort`、provider SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`を保持。RNGを再生せず、元の保存event列を読んだ。

## 確認できたこと

保存inputのbyte/hash、H/state/snapshot/preparation binding、元counter/law SOURCEのチェックを通過した。eventのcomponent・basis操作/hash・正負時間のangle・product順/符号・relative phase・probability/normalizationは元SOURCEから合法metadataを再構成して照合した。全tail数/event数・元seed/law/conditional・canonical event digestはS開始前と各軸の読取で確認した。これらは完了した固定runnerの検査結果であり、別の独立科学レビューではない。

前段のSavedEventsから、同じ合法tailを一つのiteratorとして旧stream_structureへ渡した。Accounting/structuralの状態はpiece/outer間で継続し、scalar/wrapper/terminalは元loopのまま一回処理した。H6の11pieceを含む実保存入力で、8出力が元Sと一致した。

readerはevent列全体を一括保持しないが、全eventを読んで処理する。事前検査とX/Yのために同じrawを複数回読むため、O(qR)時間は残る。preparation/headおよびseeds/laws metadataも保持する。実行全体のRSS・速度向上率は測定しておらず、巨大q/Rでの性能を実証したものではない。

旧C1のAccounting、stream_structure、MeanKernel、connected-cluster、C1モデル係数/資格を変更していない。修正版SOURCEの人工14件PASS記録は[前回成果](../TECHNICAL_TESTS.txt)を再利用し、科学run02を確認するためだけに同じS計数や人工suiteを再実行していない。実行後のCHECKSは保存された出力と固定対照を読む整合性確認で、新Sの再計数ではない。

## 範囲と限界

保存inputから元構造metadataを再構成し、law/digestを検査したうえで、承認された4path×XY8件だけを計数・照合した。新sampling、SCF/DF/H/state再生成、実回路build/compile/transpile、価格fit/残差再校正、実平均価格予測、signal/N/G、他path/round、C2の40/16wrapperは0。

8件の一致は同じ保存event列での実装parityを支持する。全supportに対する閉包や縮約の正しさ、全event生成なしのE[S]、全lawのE[C_compiled]、compiler残差の長q/R移送、モデル採用・価格精度・全RPE成功/資源優位を保証しない。保存数値stateのground-state資格やsamplerとideal lawの距離も新たに保証しない。physical_state_closure/mean_compiled_price/full_rpe_resourcesはnull、qualified_for_candidate=falseを維持する。

## 引渡しとSTOP

**Codex B**は今回の8件を構造計数のparity証拠として参照できる。実compiled DIRECT価格や平均MODEL価格のrecordとして取り込まない。共通controller/provider/API・A/Bの作業領域は変更していない。

次の価格科学検証へ進む場合は、**ユーザー**が目的と有限scopeを判断する。**Codex C**はBの実RMSE/M/κ候補domainに合わせ、物理閉包の証拠または局所期待モデルとC−S境界残差の検証を選ぶ。保存同条件の結果を優先再利用し、新compiled校正や独立cohortが必要な場合だけ範囲を明示する。今回その計算は開始しない。

run02は`S0A_PARITY_COMPLETE_MANDATORY_STOP`。旧SOURCE/input/raw/結果/cohort/grant/ledger/marker/KEEP/STOPとrun01の失敗を保全し、結果をcommit/push・remote SHA/主要成果物照合後mandatory STOPする。
