# O2-SCV：結果統合・要因整理・次研究方向の提案

2026-11-07 JST／SERVER1（GPUA100-gleap-01）。**静的調査・既報値の引用・文書公開。新科学計算・保存結果の数値再解析・新control実装・人工実験・重要科学レビューは0。旧mandatory STOPを維持する。**

## 1. 統合した証拠と範囲

保存linear H4／1.00 Å／STO-3G／legacy DF12／8 system qubits／全4電子sector／L_D=3／m0,q1,δ=.02,r1,K2／coefficient_atol=0.0、X/Y、理想gate。N1 actual finite-grid Q・全5正層・selected full、N2保証ground、認証済みexact Fraction engineに限定する。保存DFモデル外、実機noise、RPE energy精度、全round最終総費用には拡張しない。

| 固定資料 | 確認した役割 |
|---|---|
| [224結果 ba483ab3][r224]／科学SOURCE996e434e | REPORT・RESULT_SUMMARY・EXECUTION・FINAL_METADATA_AUDIT、SOURCE・manifest・STOP |
| [448結果 a7b70ff][r448]／科学SOURCEf70e555 | REPORT・RESULT_SUMMARY・SAVED_RECORD_INTEGRITY・CLASSICAL_RESOURCE_SUMMARY・EXECUTION、manifest・STOP |
| [Signed metadata b4512675][meta]／科学SOURCE71778dae | exact mean/m₂/Σ・構造的ELIGIBLE。再実行なし |
| [旧方向整理4741cae][old] | 3比較文書。Signedが未実施だった時点の記録として保持 |
| [候補設計84e021b6][design]、[有限confidence設計af47cf9][confidence] | 既存式・SOURCE/API・費用・未保証項目 |
| SOURCE560c6092・f70e555 | sampler、CV、h1-only model、RTE/phase/basis、価格/保存・既存人工テストを静的に確認 |
| N1 744e1a0／N2 5e9bd08／engine fdf00ce | target/state/engineの固定参照。preparation/solve/認証は再計算しない |

正式共通指示2026-10-25（§8.1/8.2/8.4/10/10.3）、適用AGENTS、PROJECT_MAP、研究概要、研究目的、VALIDATION_STATUS/validation manifestも確認した。固定資料45ファイルのcommit/path/blob/byte SHA256と公開APIのblob一致は[EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json)。SOURCEは読解のみで、科学moduleをimport・実行していない。

## 2. 2件から共通して言えること

**CONFIRMED**：両件ともexact中心化、fresh独立train、係数凍結、全5層paired評価、rank2/range guard、全bit・価格・会計を実H4で接続し完了している。

**SUPPORTED_BUT_LIMITED**：具体的な2control・凍結係数・固定quotaにおいて、学習費込みRZ改善を支持する明確な材料は得られなかった。

| pilot | 取得 | 分散点指標の既報傾向 | 実現physical RZ |
|---|---|---|---:|
| Fragment群indicator | train64＋共通eval160＝224 | Xは実質改善なし、Yは約0.024711%の微小点低下。両軸の学習込みscoreは不利 | 1,079,241 |
| Signed Support | train128＋共通eval320＝448 | X/Yとも追加点改善なし、Yは点増加。両軸の学習込みscoreは不利 | 2,156,805 |

出典は固定REPORT/RESULT_SUMMARY。全比較項目・係数・quota・paired限界・train/eval費・scoreは[統合証拠表](SCV_TWO_PILOT_EVIDENCE.md)に転記した。共通評価を実費で二重課金しない。2pilotは同じfinite targetだがcontrol・学習量・評価量・quota・recordが異なり、pilot間の絶対V̂からcontrolを順位付けしない。

精度保証・真variance差・期待費用改善・最適B1-OLへの優越・SCV全体の無効性は**UNKNOWN**。探索的point scoreを厳密な結論に格上げしない。

## 3. 改善しなかった理由はどこまで分かるか

| 区分 | 整理結果 |
|---|---|
| CONFIRMED | feature非定数性とrank2は確認済みだが、相関や係数精度を保証しない。SCVには有料の独立trainが追加される |
| SUPPORTED_BUT_LIMITED | 今回の凍結係数・評価ではその追加費用を回収する点材料が不足 |
| HYPOTHESIS | metadataと信号の弱い関係、Bornノイズ残存、有限学習損失、少数評価の変動が要因の候補 |
| UNKNOWN | 各要因の寄与、真の相関/oracle係数。Born支配や「真の分散悪化」は確定できない |
| 将来への注意 | rangeは有限confidence/shot式へ影響するが、今回は保証付きcal/Nを実施しておらず、range penaltyをpoint score悪化の直接原因とはしない |

[要因整理](SCV_NONIMPROVEMENT_HYPOTHESES.md)に、既存のvariance展開、Born分解、有限学習損失、固定費回収式と、判断を変える証拠を整理した。新しい共分散・相関・break-even値を計算していない。

## 4. 未検証controlの価値

Product完全label一致は、rotation一位置情報を使った現2候補と異なり、同一signed involutionのP_l²=Iというproductsの関係を問える。exact conditional meanを平方和/比で取得でき、feature取得の追加量子測定は不要。**構造的に検討する根拠はあるが、bit相関・全RZ改善の証拠はまだない。**

主な弱点は活動h2/h3が希少で、層内削減が全targetで小さく重み付けされる一方、各層の独立trainにはfull価格がかかること。productsが恒等化してもrotation/phase/D/wrapperは残る。現h1専用modelもそのまま使えない。[候補比較](SCV_NEXT_CONTROL_COMPARISON.md)でProduct一致・products sign積・basis群・操作数proxy・特定labelの構造、mean/range、UNKNOWNと最小差分を比較した。

## 5. 技術的な第一提案と正式採択の境界

**NOT_ADOPTED：現2種類の具体的controlは一区切り（A）、SCVの追加科学計算は当面保留（D）する案を第一提案とする。** 同controlの追加取得に、費用に見合う新しい問いがまだ定まっていないため。2pilotからSCV全体の終了を確定する提案ではない。

将来別構造を問うならProduct一致（C）が候補。ただし「未検証だから次も測る」とせず、[方向A/B/C/D比較](SCV_RESEARCH_DIRECTION_OPTIONS.md)の判断条件を先に確認する。研究方針の採択はユーザーが行い、今回の資料追加では科学的status/GO/NO-GO/旧契約を変更しない。

## 6. 次の担当・最小作業・承認

今すぐ追加計算する必要はない。ユーザーがCを検討すると決めた場合、SERVER1 Codexの次の最小作業は**限定adapter実装・人工correctness**。その指示後に新SOURCEを固定し、さらに別の明示実行許可で**N1 LAW/catalogからProductの全5層exact mean/momentを一回取得するmetadata-only**へ進む案を用意した。[最小検証草案](NEXT_MINIMAL_VALIDATION_DRAFT.md)

metadata-onlyにstate/Born/RNG/shot/compile/fitを含めない。fresh train/pairedは別段階とし、活動層・独立stream・train/eval量・guard・診断・全費用・STOPをその後の契約で固定する。metadata適格だけで自動測定しない。今回は新しいshot数/seed/guardを正式採択せず、新control実装・人工試験も開始していない。

同controlの継続を選ぶ場合も、現λの独立評価、再学習の品質検証、保証付きcal/productionを分けて目的と有限scopeを許可する必要がある。承認証明書・grant発行等の手続きは追加しない。重要科学レビューは別の明示開始指示なしに開催しない。

## 7. 保全・公開・技術検証

- base=`a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e`、専用branch=`codex/o2-scv-two-pilot-synthesis-20261107`、worktree=`/tmp/o2-scv-two-pilot-synthesis-20261107`。
- 追加は本directoryの資料・技術照合JSONと必要な索引/資料追加履歴だけ。src/scripts/tests/artifacts、VALIDATION_STATUS/validation manifestの科学的statusは変更しない。新しい検証経路や科学的証拠を登録したものではない。
- 元mainのHEAD・dirty/untrackedと3,087既存ファイル、旧224/448/metadata原本を開始前後のhashで照合する。既存SOURCE/結果/STOP/grant/ledger/marker、C1凍結、752以前の結果と並行O1/O2を保全する。保全の実測範囲は[TECHNICAL_CHECKS.json](TECHNICAL_CHECKS.json)に明記する。
- 技術確認は引用/リンク/固定blob・追加範囲・文書整合・byte保全のみ。科学的統計再計算、新control/人工実験、science worker/manager/GPU起動0。自分以外のprocessや共有設定/venvへ操作0。
- 公開はユーザー指示に従い既存GitHub APIを優先し、共有HTTPS認証は変更しない。remote commit/parent/tree、公開全変更ファイルの本文とblob、固定資料リンクを照合する。最終固定SHAは公開commitと最終チャット報告を正本とする。公開後の照合receiptは作業用`/tmp/o2-scv-synthesis-publication-verification.json`へ保存する。

## 8. Mandatory STOP

資料収集・統合・静的調査・次段階設計・公開照合を完了した時点で**mandatory STOP**。新科学計算、旧rawの数値再解析、Product/他control実装・人工試験、旧224/448再開、cal/production、重要レビュー、別のengine高速化作業は自動開始しない。

本書はユーザー依頼による結果整理資料であり、実際に開催していない重要GPT/Claude科学レビューの記録ではない。提案の正式採択はNOT_ADOPTEDのまま維持する。

[r224]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/REPORT.md
[r448]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/scv_signed_support_448_science_20261107/REPORT.md
[meta]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4512675ae1d650858256547d359210361d77f68/docs/research/scv_signed_support_h4_metadata_science_20261107/REPORT.md
[old]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/4741cae05af662dfc42d0519ff977756a94cb7b7/docs/research/scv224_next_directions_20261107
[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107
[confidence]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107
