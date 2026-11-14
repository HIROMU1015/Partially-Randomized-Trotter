# 必要Nの有限計画と共通total decoderの他方式受入準備

2026-11-14。H4/1.00 Å/STO-3G/保存tol-only DF10、sector(2,2)、元H/state/snapshot、h=.1/M12/P0を対象とする技術成果。実科学call、保存数値の新算術・ranking、MSE/N/G評価は0。主agentによる確認であり、独立レビューではない。

## 完了結果の位置付け

正本は固定[`caf1417ef`のREPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caf1417ef20d0a5ab2030a2d957ee4f09443be43/docs/research/pauli_total_q3_accelerated_results_20261114/REPORT.md)、同directoryの`SUMMARY.json`、`INPUTS.json`、`run01/RESULT_LD1.json`と`RESULT_LD2.json`。以下は保存報告からの転記で、新計算ではない。

| Pauli構成 | 保存直接モデルMSE上界 Ha² | 状態 |
|---|---:|---|
| LD1 | 4.7246214689726850477135E-7 | 条件付きopt-inモデル診断完了 |
| LD2 | 4.7241534608509211675289E-7 | 条件付きopt-inモデル診断完了 |

両軸のcontrol vectorは`[64,60,56,52,47,43,38,34,30,25,21,17,12]`。USER_SUPPLIEDであり、原論文nominal・十分N・最適Nではない。今回、目標値との新しい比較filter・比率・平方根・LD順位を計算していない。

SUMMARYのMSE上下界とDP投影fieldをrawと同一性照合した。各DPはm0〜12の残余・zero保持・tie argmin fieldを持ち、元の保存検算はPASS。INPUTSのallocation ID/policy/shotsはrawと一致し、rawの追加allocation SOURCEは実行SOURCEを指す。新しいmass総和やmoment分解の再算術は行わず、[既存VERIFY_RESULTS](../pauli_total_q3_accelerated_results_20261114/VERIFY_RESULTS.txt)とexact原本を再利用した。[READINESS](READINESS.json)に今回の照合を記録した。

旧partial DP（`e7ce2380`）はPR-DF raw＋zero/tie未解決loss契約、新Q3はPauli raw＋全関数推定量。物理signal・辞書・normalization・decoder・lossが異なるため、上界の大小をdecoder改善率やLD優劣と解釈しない。同じPauli run内の`partial_decoder_reference`も旧推定量用の別診断であり、total riskへ加算しない。

total decoderはzeroを定義済み更新規則で保持し、tieは決定論argminでenergyを返す。未解決確率0は全関数の定義による。unsafeは十分条件から外れた事象で、誤branchや未解決そのものではない。tracked zero/tie確率は残余内事象を含む全law値とは区別し、残余lossを捨てない。固定kernelにはzero/tie別の独立loss加算分解はない。

## 今回の実装と確認

- `rpe_total_binding.bind_signal(..., original_packet_source=...)`をbaseline専用に限定拡張した。元packetを固定Git JSONと一致確認し、元configurationからのdecoder-only bridgeとCの物理context検査を使う。
- baseline packetの単一numerical SOURCEを、既存callerが要求する`implementation_source`へ明示したviewを追加。元packet、raw、target、normalization、qualificationは変更せず、kernel viewと元carrierを両方保全する。
- `rpe_c_interfaces`に、C `c628e9e84`の既存`baseline_signal_pipeline.prepare`本体と必要定数だけを読み込んだ。C planner・decoder・risk・DP・controllerの再実装はない。数値providerは読み込まない。
- Full-H κ2とTruncated p7/B0Iの保存13roundずつがmetadata接続PASS。Full PFは人工13roundで同じ経路を確認した。PR-DFの固定grid拒否とPauliの専用intakeは維持した。
- 新7＋既存8＝15技術test PASS。[TESTS](TESTS.txt)。実H4旧DP・Pauli Q3の再実行はしていない。

具体的なSOURCE・ID・不足は[SIGNAL_APPLICABILITY](SIGNAL_APPLICABILITY.md)、APIは[HANDOFF](HANDOFF.md)、契約と有限実行案は[FINITE_N_PLAN](FINITE_N_PLAN.md)。

## 残る判断と終了

Z1＋T1の主推定量採択、model/physical riskの主張、target ground包含、state・sampler law移送、数値certificate、元Minkowskiへの接続は未確定。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`を維持する。raw直接riskは近似を含み得るのでAのδEを加算しない。

指定の`research_project_instructions_unified_2026-11-14.md`は既知の共通配置先と関連Git資料で取得できず、絶対pathをユーザーへ照会した。§10.5を読了したとは扱わない。現行`/home/AbeHiromu/projects/AGENTS.md`と取得済み11/09版・今回の明示制約に従い、Pauli完了2件・旧DF DP・coarse/lemma・保存26round・Full-H/PF既存成果の重複を確認し、科学計算を起動していない。

**ユーザー**：[有限案](FINITE_N_PLAN.md)の契約・範囲を判断し、必要なら別途科学実行を明示許可する。**Codex B**：許可後だけ固定範囲を実行する。**A/C/D**：不足入力・価格を取得する場合も各担当の別scopeとする。今回、他担当への実行依頼や起動はない。

成果の公開・remote照合後は**mandatory STOP**。新しいDP、N探索、価格・Gへ進まない。
