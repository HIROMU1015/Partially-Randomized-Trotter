# SERVER_A：DF H4一件の限定数値整合性検証

2026-10-28 JST。**独立数値一致4項目、metadata一致2項目、配線一致1項目、INCONCLUSIVE3項目。不一致0、禁止API実行0。** 新しい専用harnessから実際に決定論的評価を行った。全B2/S2/finite-RTE経路の科学的同等性を認定する結果ではない。

固定ケース：Linear H4、0.90 Å、STO-3G、保存DF rank12/generated order、B2、S2、L_D=3、q=1、r=4、K=4、T=delta=0.8、`B2-rank3-q1-r4-K4`。固定条件の採択変更・候補探索は0。

| 項目 | 判定 | 実行した比較範囲 |
| --- | --- | --- |
| DF_HAMILTONIAN | MATCH_INDEPENDENT | 元保存full256×256 HとP1既存coreの再構成全要素 |
| SCALAR_ONE_BODY | MATCH_INDEPENDENT | 元hspin＋correctionとP1 one、nuclear一回適用 |
| FRAGMENT_ORDER | MATCH_METADATA_ONLY | 保存lambdas/G順序・原版identity |
| FIXED_PARTITION | MATCH_METADATA_ONLY | 保存列0,1,2と3..11の明示partition |
| S2_PREPARATION | INCONCLUSIVE | 完全prepareは今回禁止の回路生成を伴う |
| FINITE_RTE_LAW | INCONCLUSIVE | 候補固有のexact tail/config未適格。静的意味だけ確認 |
| EXPLICIT_EVENT_REQUEST | INCONCLUSIVE | A確認範囲で適格な保存明示event列なし |
| PHASE_SIGN_CONTROL | MATCH_INDEPENDENT | nuclear identity scalarの負符号位相・diag(I,U)だけ |
| REAL_INPUT_CONTEXT_PROPAGATION | MATCH_WIRING_ONLY | 実adapter→同一coreへの入力/文脈配線だけ |
| SAVED_NUMERICAL_REFERENCE | MATCH_INDEPENDENT | 保存H/state/energy/sector/bit/phase。保存確率signalは対象外 |

## 実測誤差と実行

- full H最大絶対要素差：`7.1054273576010019e-15`（固定atol=rtol=1e-12）。one-body補正差0。
- state norm差：`6.6613381477509392e-16`（atol1e-12）。保存energy差：`4.4408920985006262e-16`。
- H-state残差L2：`2.1300460134223203e-15`（上限1e-9）。sector H最大差：`2.6159630017730251e-15`。
- nuclear scalar位相/control行列差0（atol=rtol=1e-14）。bit順・sector埋込・phase条件も固定条件内。

[harness](../../../scripts/integration/validate_h4_one_case_equivalence.py)はH4GeometryAdapter入力1回、既存Python df_linear_operator1回、identity256のmatmat1回（内部matvec256回）、identity位相2 API各1回を実行した。科学実入力の再試行0。数値段階wall 14.415秒、peak RSS 591528 KiB。別の[人工回帰テスト](../../../tests/test_h4_one_case_equivalence_harness.py)は11 PASS / 0 FAIL / 0 SKIP。

P1固定commitは `2c865c66171fc9d1972b67c1257c57e5684fe26c`。独立参照は真正性確認済みの旧保存行列/状態で、同一coreを二回呼んだ値をoracleにしていない。核scalar位相は別のcos/sinの数学的基準と比較した。配線・metadata一致は独立数値一致へ昇格させていない。4項目は比較範囲を分けた分類であり、4つの独立campaignや全pipeline保証を意味しない。

## 未確定項目の具体的理由

P1 prepare_df_partial_s2はGaussian block構築からBogoliubovTransformを作り、実Qiskit circuit生成へ進む。今回それを呼ばず、禁止modeを有効化せず、science kernel/initializerを修正していない。そのためS2 half/reverse全演算子は未検証。

旧r=4はq=1の中の4 short steps、K=4はeven paired Taylor order。P1の意味は静的確認したが、同一候補のexact RTE tail/normalizationを回路なしで適格に渡す経路または保存configが不足した。partition proxyをlambda_Rに置き換えたり、保存normalizationから逆算して比較を成立させたりしていない。

Aの既知旧5runと後続固定locatorでは、この候補のraw signal/event列は取得できなかった。公開posthoc保存投影・candidate fingerprint・MAP_COMPLETE_STOPは照合できるが、有限trajectoryのsignal/cost/SEを決定論的oracleとはしない。SERVER_B報告だけでA上の実体やeventを適格と認定していない。

## 資料と結論の範囲

[入力/SOURCE適格性](INPUT_AND_SOURCE_QUALIFICATION.md)、[実行前固定contract](FIXED_COMPARISON_CONTRACT.md)、[10項目CSV](NUMERICAL_EQUIVALENCE_RESULTS.csv)、[実測JSON](VERIFIED_RESULTS.json)、[テスト・再現](TECHNICAL_AND_SCIENTIFIC_TESTS.md)、[保全/次段階](PRESERVATION_AND_NEXT_STEPS.md)を参照する。

入力NPZ/freezeと32配列の完全manifestは適格になった。今回の一致はHamiltonian表現・補正・保存stateとnuclear identity phase/controlという限定範囲。完全なS2/finite-RTE event経路、Qiskit price、N/G/SE/RPE、別距離/分子/rankの正しさは未認定。P1のpublic scientific/real/resumeとStageResultのUNQUALIFIEDは保持する。

旧SOURCE/input/raw/result/grant/ledger/marker/STOPを変更せず、B3・SERVER_B・Research C・他研究job・共有環境へ操作しない。結果の通常fast-forward公開・local/remote一致・公開bytes読戻しまで完了したらmandatory STOP。P3/P4、H6/H8、新候補、旧再開には進まない。
