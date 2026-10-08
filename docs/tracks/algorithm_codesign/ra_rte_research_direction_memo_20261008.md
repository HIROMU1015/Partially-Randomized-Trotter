# Track B / RA-RTE 研究方針共有メモ

2026-10-08 JST。利用者が共有したGPT側の研究方針を記録する。
**本書は方針メモであり、実装・数値計算・source変更・execution authorizationではない。**
T0.1については別途具体的な指示を待つ。既存STOPを維持する。

## 基本方針と中心RQ

当面はRA-RTEを継続候補とする。目的はRA-RTEを完成させること自体ではなく、
既存アルゴリズムでは得られない独立した資源削減の価値を、できるだけ早く判別することである。
情報価値の低い技術修正を無制限に繰り返さない。

中心RQ：

> 同じ有限Taylor平均を実現するランダムunitary ensembleについて、degree-localな係数再配分を許すことで、既存representationの混合や合成精度の最適化よりも少ない総量子資源を実現できるか。

- B1：representation固定、合成精度の配分を最適化。
- B2：既存representation全体の混合を許す。
- B3：degree-localな係数再配分まで自由化。

中心比較は固定resource constraintsのもとで、certifiedに
`U_Q(B3) < L_Q(B2_outer)`が成立するか。
B3がB2に勝つことだけでは、独立した新アルゴリズムの新規性が成立したと判断しない。
importance sampling、CTS等の強い既存法との比較は、その後に必要性とscopeをGPT側で判断する。

## 現状と証拠境界

- v3 source S：`45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`。
- authorization A：`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`。
- one-shot result R：`35f8b949079f15d0348bc082b916324870da7246`。
- T0 audit：`72192b3475d59f5c56370cb4068d5659789f0ef4`。

v3の最終分類は`D0_TECHNICAL_INCONCLUSIVE`。
最初のB2 T minimumの数値certificate failureで停止し、B2/B3 comparisonは0件。
このrunからB3の優位性・非優位性・RA-RTEの研究価値を結論しない。

T0ではraw double-derived nominal点が量子化前からmembership条件を満たさず、
保存N3点のconfidence marginも不足することを確認した。
zero-mass groupの扱いによりprojectionは未実施、T0分類は`T0_TECHNICAL_INCONCLUSIVE`。
詳細は[監査](ra_d0_t0_read_only_failure_audit_20261007.md)と
[GPT handoff](ra_d0_t0_gpt_handoff_20261007.md)を参照する。
元v3 result・消費済みmarker・authorizationを保持し、既存one-shotを再実行しない。

## 条件付きの優先順位とGO/STOP判断

1. **T0.1 read-only診断**：inactive representationのzero-mass groupを扱う数学的規約を結果前に固定し、
   同じ保存nominal点への一回のprojectionを評価する。membership、mean、confidence、workspaceを確認する。
   本メモでは規約を採択せず、診断を実行しない。別の具体的指示が必要。
2. **限定的な数値実装再設計の検討**：T0.1で修正可能性が認められた場合だけ検討する。
   対象候補はstructure-preserving B2 parameterization、数値的mean保存、confidence認証余裕、
   dyadic sampler construction。科学仮説・候補集合・比較baselineを結果後に変更しない。
3. **B2/B3中心比較**：数値実装が成立し、固定条件と別の実行認可が整った場合に実施を検討する。
   追加自由度の資源的価値をcertified comparisonで判断する。
4. **研究価値の再判断**：B3の有利性を確認できれば強いbaseline比較の必要性・範囲をGPT側で検討する。
   有利性を確認できなければ縮小・停止も候補とする。
   限定修正でも中心比較へ到達できなければ、LP/certificate設計を継続する情報価値を再評価する。

修正を繰り返すこと自体を研究成果とは扱わない。
上記の条件を満たすことは次stageの自動認可ではない。科学的解釈とGO/STOP判断はGPT側に戻す。

## Track B全体との関係

Track Bの大きな目的は、PRのランダム化部分に適用できる新しいアルゴリズムや、
より資源効率のよい時間発展方式を提案すること。
Taylor法を最良と仮定せず、qDRIFT、高次乱択法、importance sampling、CTSとの比較可能性を意識する。
これはbaseline追加の認可ではない。
PR専用Hamiltonian変換・軌道最適化等は別研究経路とし、現在のRA-RTE作業へ混ぜない。

## 担当と作業制限

GPT側：研究仮説の選択、研究方針・RQ・新規性・論文着地点、新アルゴリズムの採択、
科学的解釈、追加検証の必要性・範囲。

Codex側：承認済み固定契約内の実装、数学的・数値的検証、証拠保存、source review、
provenance audit、GPTへの正確な報告。GPTへ渡す必要資料はcommit・pushして固定identityを提示する。

明示指示なしに研究方針変更、baseline追加、tolerance/solver設定調整、STOP後の再実行、
次stageへの自動移行をしない。本書の追加を根拠にsource・contract・authorization・result・markerを変更しない。
**現在の実装・計算authorizationは追加されていない。T0.1の具体的指示までSTOPを維持する。**
