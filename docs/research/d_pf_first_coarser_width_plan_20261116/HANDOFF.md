# B・次の信号取得担当へ：p7/Yoshida4・32h/5を先行

**推奨はB0I p7／標準Yoshida4／PF要求幅32h/5。保存7roundを再利用し、m6〜m11の6 complex meanを補完して、現幅Yoshida4の保存B_PRICE配分1本でQ3を比較する案。初回compile上限0。** 今回は計画のみ。科学実行の許可・配分採択・価格規則採択は新しく発生していない。

## ID・schedule・入力

元configuration：**`dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`**。元scheduleの表示・physical候補は次を併用する。

- `5fb2c32e39e63267ea1748861269c362a93a1b86:docs/research/truncated_pf_optimization_plan_20261114/PLAN_METADATA.json#configurations_table[69]`。
- `5fb2c32e39e63267ea1748861269c362a93a1b86:docs/research/truncated_pf_t2_results_20261114/run01/cell_p7_4th_32o5/CELL_SCOPE.json#uses`（m5〜m12のT・physical candidate）。
- B索引：`7f27e18f06dde12127c85be898777b685d29bacc:docs/research/pf_joint_width_shot_plan_20261115/WIDTH_SIGNAL_COVERAGE.json` の同configuration。D索引：`486fa84ccf2b63c73f71d3adcedada72dc6ebe99:docs/research/d_pf_width_price_coverage_20261115/PRICE_COVERAGE.json` の同configuration。

完全な固定selectorは [CANDIDATES.json](CANDIDATES.json) と [FIRST_FINITE_PLAN.json](FIRST_FINITE_PLAN.json)。B索引ではこの粗幅の `configuration_locator` / 完成packetはnullなので、旧canonicalのIDだけを書き換えたpacketを渡さない。既存spec constructor・original schedule・bindingからcanonical metadataを組み、元IDとの一致を人工検査し、正しい新packetとして公開する必要がある。元scalarやwidthの意味が解決しなければSTOP。

保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、元H/state/snapshot、一体項＋fragment0..6保存順、P0除外。h_RPE=.1は固定。PF幅は保存hのexact有理数×32/5で、表示0.64とexact rational descriptorを混同しない。標準Y4 signed weightsは元scopeの `[1.3512071919596578,-1.7024143839193153,1.3512071919596578]`、元演算順序とblock順を保持する。p7のscalarはc_H＋tail identity、total Tへ一回。保存raw meanへ再度位相を掛けない。Fullのc_Hだけの経路をp7へ転用しない。

## 保存7roundと不足6round

| 区分 | m | q_PF | T | physical ID |
|---|---:|---:|---:|---|
| 新規候補 | 6 | 10 | 6.4 | `77d24715893869133820` |
| 新規候補 | 7 | 20 | 12.8 | `7e23a1be7c7ffb2e536d` |
| 新規候補 | 8 | 40 | 25.6 | `443fdeda245dc18375fc` |
| 新規候補 | 9 | 80 | 51.2 | `f0a99e91997c44c59ad4` |
| 新規候補 | 10 | 160 | 102.4 | `61d8331798b0671efa95` |
| 新規候補 | 11 | 320 | 204.8 | `b1bd9b1c025f7348bf92` |

全新規点のactual PF幅は32h/5。同じcomplex meanのRe/ImをX/Yとして扱う。normal samplingや実回路compileの計画ではない。

再利用するm0〜m4：
`06a15ee45beb177c018f6260ea4a5ac8cf66cfe6:docs/research/pf_shortlist_mean_completion_20261115/run01/B0I_P7_Y4_16h_over_3/B_SIGNAL_INPUT.json#means[0..4]`。physical IDは順に `6c100b1fcb67060431a8`、`c841e095f0f12a2fcdbe`、`9531e5eb957c0b00e08a`、`f144b792e8889063951f`、`91100691c8f55c5a3ab0`。early roundは現幅と同じT/q/actual幅なので、元記録を保持した候補bindingで再利用する。

粗幅固有のm5(q5/T3.2/ID `4eccd52bab8d4d790ab9`) とm12(q640/T409.6/ID `e9a98f2bfa3f17a497c2`)：
`5fb2c32e39e63267ea1748861269c362a93a1b86:docs/research/truncated_pf_t2_results_20261114/run01/cell_p7_4th_32o5/round_first_check.json` と `round_m12_check.json`。元 `gate_pass` とscalar一回のmetadataがある。元SOURCE/roleを保持し、数値の再解析・再評価で置き換えない。

保存step locator：同commit・同cellの `OPERATORS.npz#PF_step_without_global_scalars`。今はNPZを開いていない。後続では元 `SavedStepContext`・保存step loader・`baseline_signal_numerics.evaluate`・reference reader・packet producerを再利用する。再利用実装SOURCEは `06a15ee45beb177c018f6260ea4a5ac8cf66cfe6:src/trottertracks/pf_shortlist_mean_completion.py` と `scripts/run_pf_shortlist_mean_completion.py`。旧48mean guardとnamespaceは変更せず、新6点の薄いselectorを別SOURCE/出力で固定する。

## 比較する保存配分と上限

元比較configuration：`defc5e88ff4da2e43c99ecdf7157aa41cc47cd32f905f5aa3a1266aa3ecf5416`、p7/Y4・現幅16h/3。

元配分locator：
`b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/SUMMARY.json#rows[15].allocation`。

**X/Yとも `[59,55,51,47,43,39,35,31,27,23,19,7,2]`** を固定する。元allocation ID `B0I_P7_Y4_16h_over_3_B_PRICE_COST_BUDGET_3O4_01` はsourceとして保存し、新幅用IDは実行SOURCE固定時に別に付ける。p7/S2の末尾13/4の配分へ変更しない。

提案する一つの有限パッケージは次の上限。取得できない点を別round/seedへ置換せず、再利用分が増えれば新規枠を減らす。

| 作業 | 提案上限 |
|---|---:|
| 元7roundの再利用・source binding | 7 complex mean |
| 不足m6〜m11の新mean | 6 complex mean |
| 新source-bound完成packet | 1 configuration |
| 保存配分の新Q3 | 1 whole-allocation call |
| 新reference計算・step/回路build・compile・価格・fit・新配分・retry | **各0** |

新Q3は全13roundのintake/numerical gateを通過したときだけ。元decoder/target/loss/law/exact τを維持し、新signalに元の閉包PASSを流用しない。現幅の同配分はMSE約2.418371982e−6 Ha²で適合。上界≤exact τなら同配分の粗幅適合、下界>τならこの配分の不適合、跨ぐ場合はINCONCLUSIVE、証拠/gate不成立はUNKNOWN。

全価格がないままでもこの精度比較はできる。Q3後は結果を保全してSTOPし、候補を残す場合だけ粗幅価格の別有限案へ進む判断を返す。不適合でも粗幅全体を棄却せず、shot増・他の保存列を試す許可を自動的に推定しない。

## 価格側と二つの費用基準

同物理点の価格証拠はm0〜m4・X/Yの10軸。元trial/receipt・C受入根拠へのlocatorは `FIRST_FINITE_PLAN.json#saved_direct_price_evidence`。元値を再計数せず、候補binding/C guardが必要なら取得前に確認する。特にm4の元X receiptは `a04d40fe4d8f4bcf8083e4feebdf1535b5e7f5cd:docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/PRICES.json#[4]`、Yは別recordでJSONに保持している。

不足価格はm5〜m12・X/Yの16軸。**保存10軸はactual幅32h/5の実測ではない**。新幅のruleは未採択で、hの5513q+307や現幅の閉包係数を流用しない。

候補を残す場合の後段案は、m5(q5)・m6(q10)の開発2点＋m9(q80)の未使用確認点、各X/Y・最大6 wrapper。これは今回の第一有限枠の外で、SOURCE・出力・採択条件の別承認が必要。q640までの保証や全軸直接compileを前提にしない。

費用評価が可能になった後は、(1) **Y4現幅の435,641,568 RZを下回るか**、(2) **p7の保存最安S2 435,481,090 RZを更新するか**を別々に判定する。現時点のcoarse価格・総費用・shot増でのQ3・費用優位はUNKNOWN。Fullの基準615,304,502 / 619,717,524 RZと混ぜない。

## §10.5の開始前差分照合とSTOP

今回引き継いだ資料はB7f27、D486、Q3 b429、細幅hのB3b5cd、D ecf455、Bの後続局所計画9992c594。確認時D remoteは `ecf45508a63ddf75a4806b47179f5f5102fa90e0`、B remoteは `9992c594d4c72d1e8b65b42029d2b231641c0d17`。4粗幅の追加packet/receiptを示す後続成果はこの関連範囲で未確認。関連Python caller名・候補IDの同ユーザーrunning jobは確認0。網羅的312軸監査やprivate配列/QPY探索はしていない。

科学開始直前に、今回の6physical roundと保存7点のSOURCE・H/state/snapshot・scalar・law・signed順・q/T/幅、元packet/別名ID、関連branchの後続run・trial・receipt・進行jobを差分確認する。受入可能な新保存meanは再利用、同等性不明は該当点を保留。Bの現幅最安点が更新されても全worktreeを凍結せず、費用challengeの最新版を出所付きで別記する。今回実際に使うSOURCEとinputだけを固定する。

新SOURCEと新namespaceをscience前にcommitし、人工fixtureで旧ID/Fullとの混同、6点scope、phase一回、packet保存・Q3保存経路を確認する。必要資源に恒常的なworker/CPU/RAM上限を追加せず、既存saved-step共有・独立処理の並列化・協調logを使う。旧contract/one-shot/STOPを開き直さない。

SOURCE・scalar・元数値gate不整合、未解決重複、保存不能等では部分結果と理由を保存してSTOP。無断retry・別点代替・閾値緩和・追加配分・追加compileはしない。成功時も最大1 Q3の公開・照合でSTOP。

統合担当へのロードマップ差分：**粗幅の最初の1候補と未実行の6mean＋1 Q3案が確定した。新しいsignal/直接価格/Q3適合点/総費用/winnerは0。** 今回の資料をcommit/push・remote照合して **mandatory STOP**。実行には別の明示承認が必要。
