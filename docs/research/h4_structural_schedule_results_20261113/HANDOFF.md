# B/C/D HANDOFF：H4 PR全13roundのE[S]

[REPORT](REPORT.md)／[全26record](run01/COVERAGE.json)／[CSV](STRUCTURAL_MEANS.csv)／[CHECKS](CHECKS.json)／[STOP](run01/STOP.json)。実行SOURCEは`de2427d4a31d6b4b39d2c8eb8b63e3c972ffb89e`、計画公開は`3a12c962f75d530a30884a47280c53fb72a02318`。既存C branchを継続し、B/Dの作業領域や共通APIを変更していない。

## 結果と意味

- 残11roundすべてK0 PASS、新22軸のK1を完了。m0/m6の四値は`83d219cc68c25540507a5539ebd2da94c29d06b3`から完全copyし、科学再実行していない。
- 全13round・26recordは同canonical configuration `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`へ結び付く。
- quantity=`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`、expectation=`IDEAL_FINITE_K_SOURCE_LAW`。
- E[C]/N/RMSE/Gは未取得、`mean_compiled_C=null`、`qualified_for_compiled_price=false`、`full_rpe_resources=null`。**FULL_WRAPPER_MEAN_COMPILED_PRICEとして供給しない。**

## Bの受入単位

`run01/COVERAGE.json:values`をそのまま読む。順序は旧copyが先のため、配列位置でround/axisを推測せず、`m`と`axis`で照合する。元B identity・law_binding・physical ID・kernel_source・数値区間を保持し、IDを再ラベルしない。m0/m6のkernel_sourceは旧19b95d8b3、今回22値はde2427d4aという出典の違いも維持する。

B公開source chainは3216d27aのintake→030f3f54の`docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json`、SHA256 `22a1bd1af8704006568c545eb8bcbce5bd476202ad59aaeaf79f50850815f943`。preflightで一致を確認した。R=255/510/1020/2040/4080/8160はそのままで、exactκの近似や2冪への丸めはない。

対象は同H4 DF10/S2/L_D1/δ.1/K2/h.1/M12/P0、保存H/state/snapshot、元fragment/basis/qubit順・scalar一回・symmetric-directional/full Hadamard X/Y。preparation hash `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、324非identity/9 basis。全条件は`run01/INPUT_METADATA.json`と前段数値契約へ戻る。

## kernelとparity証拠

各新roundの`run01/m{m}/`にK0.json・KERNELS.npz・KERNEL_STATES.json・K1.json。K0のkernel_sha256とCHECKSのarchive hashで対応を確認できる。11archive/693配列/99状態辞書を保存し、m0/m6の旧archiveは旧public sourceへの参照とする。

共通source分岐21,474件、event metadata17,820件、phase/control73,443件一致、毎R/9wireの閉包・local/terminal integer rewardとpending-Z hex完全parity、unit mass/interval/longdouble検査を通過。最大mass defectは1.7141843500212417e−12（tolerance1e−10）。全99 graphでstate数はwire0〜6各56・wire7 48・control22だが、これは今回検査の結果で、別lawの前提へ移さない。

K0 PASSとkernel/state保存後だけK1を実行。K0のK1_started=false、SHARED_METADATAのphysical_K0_for_each_R_still_required=trueは各検査前snapshotで、最終coverage/STOPと区別する。prefix/steady entry、close/reverseのterminal配列、outer固定出口、scalar/XY最後のsuffixを保存し、counterを途中でfinish/resetしていない。

数値区間は元γ/IEEE/BLAS丸め契約での構造・確率伝播enclosure。confidence interval、ideal対実sampler距離、compiler差・rare寄与・共有covariance/total CI、全RPE保証ではない。値の再集計やK0/K1再実行を受入処理に含めない。

## Dとの接続条件

Dの`af6f27d7a6467bb355686421acd067544edcbbd2:docs/research/d_compiled_residual_c_minus_s_20261113/HANDOFF_C.md`を確認した。同じcanonical/physical candidate/axis/law/full wrapperと元AccountingのSという基準を維持する。

今回のE[S]はideal finite law。Dの残差標本はPCG64/CDF samplerの有限cohortであるため、law一致又は距離の証拠なしにE[C]へ資格しない。C/Sのpath digest・seed/cohort/index一致、forced rareの別stratum、残差modelを一つだけ使うこと、未知の系統差/共分散をnullにすることを保持する。

controlled ZZの元loweringとAccountingの差はDの検証課題で、今回Cは元Sを変更しない。S′を採用する場合はユーザーの科学判断と別SOURCEが必要で、旧Sや本結果を上書きしない。CはC−S fit・compiled取得を重複実装していない。

## 保全・実行範囲・次の担当

一括実行1回、science restart/retry0。監視のread-only commandが利用上限に伴う自動承認レビュー失敗で止まったが、workerは停止・再起動していない。既存session exit0と完了STOPを後から確認した。新sampling/full-path S/build/compile/transpile/compiled C/fit/N/RMSE/G、H6/他preparation/8次は0。

元C1/C2/connected-cluster/S0a・旧run01の失敗・m0/m6の成功・SOURCE/input/raw/cohort/grant/ledger/marker/KEEP/STOPを保全した。今回の26技術test記録を再利用し、新code/testsや科学再実行は追加していない。独立科学レビューを実施したとは記載しない。

次：**Codex B**が同canonicalの構造packetを受入、**Codex D**が許可された範囲で残差・compiled側を検証、**ユーザー**が必要な有限scope/精度/S′方針を判断する。**Codex C**は今回完了。commit/push・remote照合後mandatory STOPし、追加科学計算へ進まない。
