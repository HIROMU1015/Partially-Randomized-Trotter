# m17局所周辺平均の固定実行計画（準備のみ・科学実行待ち）

2026-11-03 JST。本計画・runner・人工testを含む公開commitを固定SOURCEとする。
full SHAと固定URLは完了チャットで示す。
現在の方針への同意は技術的準備の許可として扱い、実H4評価を開始していない。

## 問いと固定条件

samplerの宣言されたcategorical lawを全supportにわたって平均する経路、
保持係数/paired-Taylorの経路、保存physical attenuated event-meanが、選択した一つのroundで整合するか。
さらに固定N/α/epsilonの残余と、観測差に対して残る数値不確かさの余地を記録する。

linear H4/隣接1.00 Å/STO-3G/保存legacy DF12、8 qubit、4電子sector70、Bの保存physical ground state。
Partial L_D3、単一δ=.02、m17/q131072、r32/K2、extract_identity_phase、coefficient_atol=1e-12。
F/Rは同じδのdeterministic half、eventのtauはlambda_R*δ/32。

| 入力 | 固定所在・identity |
|---|---|
| H/state | [前段の固定入力表](../rpe_mean_tolerance_validation_selection_20261103/NEXT_STAGE_DECISION.md)を継承。carrier4591b8d75a2a6a55c74a51225f0c61648023c8eeとB commit e22d7718c47358ff650dbc04e8e72f6ab1c8bf41 |
| H file hash | 13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b |
| H hash | 56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5 |
| State file hash / fingerprint | 793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4 / a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90 |
| Preparation | hash89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919、218 retained component |
| 保存μ | V2 commit78c49d0b5f1d71e63b5959e78813237923bbc648のV2_ROUND_RESULTS.csv、ld3/round_index17。Re=.32902182375994465、Im=−.46875529425442597（保存値の転記） |
| N/α/epsilon | 各軸N411、α=.016957089147912167、epsilon=.15244325851620388 |
| β | PF=.014050012886294481、RTE=9.899584668737594e-8、stat=.38594988811785885 |

N/α/β/epsilonはRPEモデルの入力であり、本検証で生成するshot数ではない。
Full・m8・残り17round・新schedule・最適化・価格取得を含めない。

## 科学callと出力

1. 固定1行・2軸のscalar残余をDecimal80桁で一度評価する。これは新しい保存値の科学的再解析である。
2. 保存H/stateをload・照合し、DF preparationは1回。basis cacheからA_prob/A_coeffとF/Rを構成する。
   DF preparation内部のGaussian gate定義、sectorの数値operator/state-actionを含む。
3. literal-lawのone-event平均を構成してr32へ合成する。参照は既存finite_rte_operator_momentsを1回使用する。
   外側q131072のmatrix powerにより、最終complex平均を各経路1個、計2個求める。
4. law対reference、law対保存μ、reference対保存μのcomplex/座標残差を保存する。
   AとtailのFrobenius差も非gating診断として保存する。

新H/state生成・対角化・ground-state solve、sampling/RNG draw・仮想shot、
controlled/Hadamard build、compile、phase成功率や価格の再計算は各0。
出力はfresh `artifacts/rpe_m17_mean_validation/2026-11-03/run01/`を候補とする（準備段階では未作成）。
RESULT.jsonにSOURCE/hash・保存条件・段階・versions/config・資源・結果を保存し、
operators.npzにA_prob/A_coeff/F/R/reference tail/law one-event/law tailの7行列を保存する。
中断時も得られた段階のRESULTを残し、既存directoryを上書きしない。

## 数値判定とその限界

行列はcomplex128/float64、basis定義はQiskit1.3.0を維持する。
scalarの80桁計算は小さな差の評価に使うが、外向き丸めの区間証明ではない。

- **数値再現の診断**：3つのcomplex絶対残差がすべて1e-10以下ならNUMERICALLY_CONSISTENT、
  超過ならINCONSISTENT。座標残差も併記する。thresholdは取得結果に合わせて変えない。
- **RPEに対する解釈**：軸ごとに`epsilon−s_alpha`と、さらに観測law対保存μの座標差を引いた値を出す。
  u_numの上界は未確立なのでnull、RPE_tolerance_certified=false、科学status=UNDETERMINEDを保持する。
  正の余地・二経路一致・1e-10の診断だけで、真のmean差やenergy成功を認定しない。
- **異常時**：入力・preparation不一致、非有限値、不正な最終mean、中断をエラーとして記録する。
  Rが非正でも条件を緩めず、数値平均自体の局所接続は別に記録する。

この検証は、RPEの平均差許容条件へ接続するための局所数値証拠を得るもの。
数値誤差上界そのものや、全18round・厳密半径・PRNGの厳密law・全回路・実backend/noiseを証明するものではない。
二経路に共通するDF/basis/state/halfの誤りも残る。既存V2/R1/run02のstatusを上書きしない。

## 資源と実行コマンド

許可後にCPU/GPU/RAM/共有負荷を確認する。basis集約とmatrix powerで主要負荷を処理し、
人工70x70試験で適したBLAS4 threadを初期構成とする。実環境の必要に応じて自job内の構成を選び、記録する。
人工CPU/RAM/wall-time上限を置かず、共有環境や他jobを変更しない。実H4での速度倍率は未測定。

metadataだけの確認：

```bash
python docs/research/rpe_m17_mean_validation_preparation_20261103/run_m17_mean_validation.py --describe
```

**公開SOURCEに対するユーザーの明示的な科学実行許可後だけ**、実行専用worktreeで次を使う。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python docs/research/rpe_m17_mean_validation_preparation_20261103/run_m17_mean_validation.py --run --output-dir artifacts/rpe_m17_mean_validation/2026-11-03/run01
```

--runは処理選択であり許可を証明しない。SOURCE一致検査は科学的な再現性のためだけに行い、
承認証明書・承認原文保存・grant・台帳・署名・管理者権限を要求しない。

## 許可・再試行・STOP・次担当

次はユーザーが、この固定SOURCEでの**1セルの保存scalar再解析とDF/sector平均評価**を明示許可する。
Codexは許可後にその範囲を実行・記録・公開照合して停止する。重要科学レビューの許可は別である。

意図しない技術的中断には既存ユーザー指示の修正・再実行許可を適用し、
失敗記録を残し、修正SOURCEを固定し、次のfresh結果directoryで同条件を再実行できる。
正常完了のINCONSISTENT/UNDETERMINEDを技術的中断へ読み替えて試行を繰り返さない。
正常完了後は診断結果にかかわらずmandatory STOP。別round、追加sampling、robust二項領域評価、
精度目標の変更、R2/R3・重要レビューへ自動移行しない。
今回の準備も公開照合後mandatory STOPで、科学実行を自動開始しない。
