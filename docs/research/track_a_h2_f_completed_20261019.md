# H2 F：tol-only入力生成・状態検証完了

2026-10-19（Asia/Tokyo）。**COMPLETE_MANDATORY_STOP。承認されたFをone-shotで1回だけ実行し、native exit0・同一親wait4回収・終了監査PASS。** 後段FP/D/A/S_STATS/Pは未実施・未承認。H2 P最終sealの完了ではない。

## 承認と固定SOURCE

ユーザーの承認原文は「固定SOURCEと予算によるFのone-shot実行を承認します」。[承認receipt](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/F_completed_run01/approval_receipt.json)をFだけへ結合し、元の承認待ちspecとINVALID grantを保持したまま、新しい私有execution spec/grantをactivation01へ作成した。

- 科学SOURCE：`37b1cf72990885962d8449d8bd5ec7e303329350`
- 準備commit：`7b8570dc526aa664d918e9346fa964d81e6f5b53`
- 実行spec digest：`218c8cf9629569604466a2ef8489d8923418e0027e87d2f84b7ae2cfcd40b5a7`
- [固定契約と準備時の検証](track_a_h2_phase0_f_preparation_20261019.md)
- 原本：`.h2_pr_resource_private/2026-10-19/attempts/F_run01/`
- [保存結果](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/F_completed_run01/result.json)、[終了監査](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/F_completed_run01/completion_audit.json)、[実行binding](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/F_completed_run01/execution_binding.json)

今回の実行承認は公開pushの承認を含めない。完了証拠はH2専用branchへlocal commitで保存し、今回のpushは行わない。

## 新規H2入力と検証結果

linear H2、結合距離1.0 Å、STO-3G、charge0・multiplicity1、RHF、SCF conv_tol=1e-9・max_cycle50、symmetryなし。SCF・integral・DF・状態を新規に生成した。DFはtol-only threshold1e-8、final_rank未指定、実返却rankは**3**。生成順・signed係数・全返却fragment・後続cutoff0を維持した。

system 4 qubits、Nα=Nβ=1、spin sector次元4の最低 `scipy.linalg.eigh` vectorを保存した。最大振幅を実正にする位相規約を使用し、保存後再正規化は行っていない。正規化の1e-12 gateは固定SOURCEの生成時に通過し、normの値そのものは別fieldへ保存していない。以下は保存値の転記であり、終了監査で再計算していない。

| 検証 | 保存値 | 固定受入条件 |
|---|---:|---|
| SCF | converged=true | 収束・finite |
| DF truncation返却値 | 0.0 | 0以上1e-8以下 |
| 独立occupation oracleとのsector行列差 | 6.972611193684198e-32 | <1e-9 |
| sector Hermiticity誤差 | 9.8607613152626476e-32 | <1e-10 |
| 状態残差 | 2.3625993563384045e-16 | <1e-10 |
| 最低固有値gap | 0.35527853719705116 | >1e-12 |
| weighted projection η（Hartree） | 5.85869134813549551920102418333447909024244891760283816731653E-15 | ≤9.9e-11 |

保存state energyは`-1.1011503302326187` Hartree。これらは固定binary64入力・工学的gateでの局所検証であり、ground-state certification、DFと元電子Hamiltonianの厳密誤差保証、最終資源優位の主張ではない。

旧H2の設定選択rank3・sector6は使用していない。新しいtol-only入力もrank3になったが、rankの一致は入力同一性を意味しない。今回のspin sector4・snapshot/state hashを固定し、旧H2やH4/H6/H8の状態・結果・cohortへ混ぜない。

## 科学callの実績と終了

| カテゴリー | 固定予算 | 実績 |
|---|---:|---:|
| SCF kernel | 1 | 1 |
| integral build | 1 | 1 |
| tol-only DF decomposition | 1 | 1 |
| sector DF作用column | 4 | 4 |
| 独立occupation行列 | 1 | 1 |
| dense eigensolver | 1 | 1 |
| signal / sampling / compile / shots | 各0 | 各0 |

各callを非返却の予約記録へ結合し、保存ledger・result・native completionとの完全一致を確認した。予約したSCF kernelの内部反復はmax_cycle50の範囲であり、新しいkernelの自動retryを意味しない。

native wallは`14.233091045171022`秒、worker wallは`12.416704092174768`秒、peak RSSは`392396800` bytes。native exit0・同一親wait4完了、中断0・retry0・新seed0。CPU/RAM/worker/wallの恣意的上限・affinity変更は導入せず、起動直前の許可CPUと空きRAMを観測した。Python backendの小さい4×4検証を一度だけ実施した。

## byte同一性と保全

- Hamiltonian hash：`d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80`
- state hash：`080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf`
- snapshot SHA256：`62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa`
- result SHA256：`63acdc965e57670531fa689824c8efddd489374e95a03c05ca723a038d6ed2c5`

integrals・raw DF・最終snapshotのSHA/bytesを保存receiptと照合した。[生成入力manifest](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/F_completed_run01/generated_input_manifest.json)。NPZ原本とSCF checkpointはH2私有attemptへ保全し、元の準備seal266レコードを照合、新しい実行記録29レコードを私有manifestへ固定した。

固定SOURCE5 file・provider262 file・runtime12,202 fileの実行時照合を維持した。worker終了時と親の終了時に封印済みrunnerがruntime bytesを検証した。今回のmetadata auditorはSOURCE/providerとruntime manifestのbytes、承認・marker・ledger・native exit・保存結果を照合し、NPZ decodeやSCF/DF/solver/残差の数値計算を再実行していない。作者自身の確認であり、独立科学レビューではない。

H4/H6/H8のSOURCE・契約・入力・結果・grant・ledger・marker・STOP・実行プロセスに書き込む操作は行っていない。無関係なworktreeのHEAD/statusを凍結対象にせず、他チャットの通常作業を妨げる操作も行っていない。

## 到達範囲と次の担当

Fの入力生成・状態検証だけを完了した。full Hadamard/native compilerのH2実検証、D/AのH2候補集合・対照・seed・科学call予算、S_STATSのn設計、P最終sealは未完了。Fの成功でこれらの実行を許可したとは扱わない。

次は**ユーザー**がFの結果を受け入れ、次の技術準備を指示する。**Codex**はその指示までmandatory STOPし、保存済みF入力を後段で再利用する。後段の科学実行はSOURCE・入力・予算を具体化したうえでstageごとの別承認とし、D/Aも別々に承認を得る。P sampling/compile・詳細科学レビュー・論文最終evidence freezeには進まない。
