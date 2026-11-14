# 保存4件の正式binding完了・共通運用指示の確認

2026-11-14、Codex D。ユーザーの共通指示追加連絡と継続指示を受け、Stage 0の保留部分を接続した。
**Full PF 2件＋Truncated p7 2件のC受入成果を固定blobで結合し、保存4・新規20＋12・全36価格の技術準備が完了した。**
比較policy・holdout許容差/分母は未採択であり、今回の指示も技術作業までなのでStage 1は開始しない。
新実H4 preparation/sampling/build/compile、保存値の再集計、実fit、holdout判定、N/G、Pauli B0再実行は0。
公開照合後 **mandatory STOP**。

## 固定SOURCE

| 用途 | SOURCE |
|---|---|
| D接続SOURCE | `22ccf3972a38b8bf0496a1472360e6eec85f1fae` |
| D初版（保留状態、保全） | `0501096bd9385157dbbe660bc59a5d00f2232e37`、実装`2b9983590` |
| Cの公開HANDOFF・4件参照正本 | `e50524f4b59f268735a682bd4b2eaa8278949c89`、`docs/research/truncated_p7_admission_pr_df_evidence_20261114/SAVED_RECEIPT_SOURCES.json` |
| C新版CELL_PLAN・p7入力/価格/proof | `43c15bfa34a22f313e55b824223b200332b7f23f`、同dirの`CELL_PLAN_V2.json`等 |
| C p7 bound sidecar/guard | `2b94539784995790c8ff1ba7dee578137bfc6034`、同dirの`SIDECARS.json`・`GUARD_CHECK.json` |
| C Full PF旧受入（変更なし） | bundle `3368805575`、bound sidecar/guard `2fa88065318f0f2ab13937ef610828200b35874c` |

Cの新版36行は[D初版提案](CELL_PLAN_PROPOSED.json)と全fieldで一致した。
旧C計画との差は、p7 M5/m4のX/Yのactionだけである。configuration・physical ID・PF公式/signed schedule・
実幅・q/round・保持block/fragment順・scalar quantity・compiler/control/wrapper・model group/roleは不変。
`P7_CELL_PLAN/P7_ADMISSION`を上記固定SOURCEへ設定した。旧planの`CELL_PLAN`参照は
差分検査用に保持し、実際の選択・preflight・実行provenanceは新版を用いる。

## 保存bindingとcoverage

| route | 保存m4/X/Y | 新規較正 | 新規holdout |
|---|---|---:|---:|
| Full PF p10 M5 | physical `0cd1527c80f7b9841a83`、各85960 RZ | 10 | 6 |
| Truncated p7 B0I M5 | physical `13d1cd4869951cdf1e2c`、各60950 RZ | 10 | 6 |

価格はCが受け入れた保存整数の参照であり、Dによる新取得・平均化ではない。
元price SOURCEは155529e20、cohortは両routeともEXPLORATIONのまま。
Cの元入力recordとDの返却recordの完全一致を、各route・各axisで確認した。
Full PFの既存bundle参照は変更せず、Cの新しい4件参照表が指定する公開版ともrecord内容が一致する。
保存p7のscalarはCの元契約で`c+s_tail=2.293146549384713`、Fullはcのみ。
どちらも元scalar quantity・一回適用・測定付きfull wrapperを保持する。
今回新しくruntime preparationのscalar gateを実行したわけではない。

- [保存binding](admission_bound/SAVED_BINDINGS.json)：4 model groupの元recordと固定参照。
- [新版metadata preflight](admission_bound/PREFLIGHT_METADATA.json)：C plan束縛済み、両routeそれぞれ10/6。
- [SOURCE状態](admission_bound/SOURCE_STATUS.json)：解消した不足と残るSTOP条件。
- [tests](admission_bound/TESTS.txt)：Stage parent **18件PASS**（16.509秒）。

既存stage選択・保存binding・C_INPUTS生成・凍結model/holdout gateを再利用した。
方式2×公式3×軸2の12 group、m4/m5較正・m6holdoutを維持する。
両routeのM5/X/Yは保存m4＋新規m5となり、元SOURCEの異なる2点を平均化・同cohort化しない。
実m5は未取得なので実C_INPUTS・実モデルは未作成。人工fixtureだけでstage実行と凍結gateを確認した。

## 共通指示§10.5の照合

追加された正本
`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-11-14.md`
の全文を読んだ。初版時点の「正本未取得」は解消した。§8の許可/STOP、§10の再利用・適正管理、
§10.4のbranch再利用、§10.5の既存成果の照合を適用する。承認証明・grant発行等は新設していない。

§10.5の別名/旧IDの扱いに合わせ、既存`duplicate_preflight`の照合を最小拡張した。
ID一致だけでなくsystem/method/prefix/formula/T/q/実幅/R/Kが一致する保存表・catalog行も
再利用候補として検出する。物理scheduleの一致だけで採用せず、元SOURCE/H/state/scalar/compiler/axisの
証拠確認を要するSTOPとし、独立検証や同等性を偽装しない。
人工試験で旧IDへの変更後も検出することを確認した。

[更新した重複snapshot](admission_bound/DUPLICATE_PREFLIGHT.json)では、既知保存表・catalog・元trial path・
当route既存stage結果・関連running jobの範囲で新規32件の追加重複はなかった。
未知のprivate archive・別サーバーまで網羅した主張ではない。
実行までにSOURCE/取得状況が変われば影響するcellを再照合し、証拠不足なら該当分をSTOPする。
他担当worktree・process・fileや共有設定を変更していない。

## 現在の開始判定と次の担当

**技術bindingはPASS、Stage 1開始は不可。** 保存4件の受入とC新版表・共通指示取得の不足は解消した。
残るのはユーザーによる比較policyとholdout許容差/分母の正式採択、および次段の実行指示である。
2%案を自動採択しない。旧最大34件の承認範囲と固定STOPを保持し、32件へ減ったことを新しい科学実行許可とは扱わない。

- **ユーザー**：比較policy・許容差/分母を判断する。
- **Codex D**：条件成立後、固定SOURCE・対象・既存承認範囲・実行直前重複/入力/import/資源条件を確認して開始可能状態を報告する。今回は実行しない。
- **Codex C**：較正価格を受け取った後の実fit・係数凍結・holdout判定には、引き続き別途科学実行承認が必要。Dは代行しない。

初版[REPORT](REPORT.md)・[HANDOFF](HANDOFF.md)・[NEXT_EXECUTION](NEXT_EXECUTION.md)は
保留時点の記録として残し、本追補を現在の状態とする。旧SOURCE/raw/入力/価格/STOPを保全した。
