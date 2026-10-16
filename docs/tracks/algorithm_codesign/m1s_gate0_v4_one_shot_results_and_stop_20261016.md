# M1-S Gate 0 v4 初回one-shot結果・技術停止（2026-10-16）

採択済みv4を別の実施承認に基づいて一回実行し、TECHNICAL_STOP / PRECISION_UNRESOLVEDで終了した。
全G3/G6/G9でW・vの事前誤差予算が採択精度を超え、W/v/rhoは計算せずnullで保存した。
これは当該精度モデル下での技術的未解決であり、M1-Sの科学的no-go、rho=1、資源優劣を意味しない。
再実行・閾値緩和・SOURCE改変は行わずmandatory STOP。今回のone-shot実施承認は消費済み。

## 実施承認と固定対象

ユーザーは直前の「初回数値correctnessを含むGate 0 one-shot」の提案に対し、別turnで「次の作業に進んで大丈夫」と承認した。
原文と文脈は[approval basis](../../../artifacts/track_b_m1s_gate0_execution_preparation/2026-10-16/v4/user_execution_approval_basis.json)に記録。
正式採択時の科学未承認記録は履歴として変更していない。

|役割|固定commit／path|
|---|---|
|科学SOURCE|606036db52da2f080cf846e48824e5e418430762|
|正式契約・採択|5e57955baa79e42f37aa233ac011c60539e431fd|
|非数値実施準備・外側launcher|de379c2771b6431d53a05cc4b57c4c9593f7a312|
|別実行authorization A|b70abf111e6459c45c24331a2c7d99c05e7dc228|
|[正式machine契約](../../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/execution_contract_v4_adopted.json) SHA256|bcabddea9c87a6da37ef9499fe0ec5311a530415445dadf0a73d908d4646ef2f|
|[別実行authorization](../../../artifacts/track_b_m1s_gate0_execution_authorization/2026-10-16/v4/execution_authorization.json) SHA256|9ede1985d87eb6770d47111b290b2875fb33e034b29ff093f148bf941a64e24e|
|[raw result](../../../artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/result.json) SHA256|83bf30919202b045101fba6aed9e7aec9e149581c4696c8c9e7212c70ae593c0|

物理系はlinear H4、1.00 Å、STO-3G、8 spin orbitals、DF rank12。
保存DF順のprefix G3/G6/G9をHD、残りをHRとし、非冗長signed Pauli集約後に絶対値を取る。
HR identityをHDへ移動、coefficient cutoffは0、exact floating zeroだけを除外。
delta window／finite-step評価なし、保存数値sector oracleのみ。exact固有状態・full-Fock ground・非縮退・pair-sum条件・gapは認証していない。
入力NPZは8652 bytes、SHA256 3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a。
17項目、残差品質式、条件付きcoverage、runtime/capsを一切変更していない。

## 保存結果と実際に到達した検証

初回runの入力・保存sector埋込み・exact Hermiticity・16 ladderのJW対応・reference Hermiticity・再構成・identity phase ledger・残差等、到達したgateは132件すべてPASS。
一方、精度事前判定は全3行で未解決。132 PASSは17項目全体やW/v/rhoのnumerical correctness完了ではない。
確率正規化、Pauli期待値の虚部、Pauli mean consistency、W/v range、rho interval等は未到達。

以下はraw resultの保存値の引用であり、監査で算定式を再評価していない。
W_error_budgetとv_error_budgetの採択許容値はともに2^-24 = 5.960464477539063e-8。

|split（HD prefix）|lambda_R [Ha]|W_error_budget|v_error_budget|metric status|W / v / rho|
|---|---:|---:|---:|---|---|
|G3（0–2）|0.8137790393725155|1.5123734243732403e-6|6.042595235294155e-6|PRECISION_UNRESOLVED_BEFORE_METRICS|null / null / null|
|G6（0–5）|0.026294457645549787|4.675057562570242e-5|1.8699376654180338e-4|PRECISION_UNRESOLVED_BEFORE_METRICS|null / null / null|
|G9（0–8）|5.47382434715216e-7|11.67336502712622|27.033167003942463|PRECISION_UNRESOLVED_BEFORE_METRICS|null / null / null|

各行のcanonical Pauli表、table hash、再構成値、phase ledger、事前formula ledgerはraw resultに保存済み。
law誤差予算超過は実際のW/v誤差がその大きさだと観測したことを意味しない。ρ順位・基底エネルギー推定資源の結論は出さない。

17項目の適用状態は、basis・factor Hermiticity・state norm・H再構成・H Hermiticity・phase・coefficient虚部・energy虚部・residual品質のgate通過、
W/v error budgetの事前適用で精度未解決、W_floorは算定値保存のみでmetricへの適用未到達、
probability／Pauli expectation／W／v／rhoの実metric correctnessは未検証、である。

## 残差品質とcoverage

保存full-H residualは1.022703352687838e-15 Ha。
採択式のquality成分1e-8 Ha、B_res=5.333090193902594e-9 Ha、合計gate限界1.5333090193902595e-8 HaでPASS。
quality成分は旧solverの数値受理基準。旧matrix-free sector/eigsh経路と新full reference/Rayleigh経路の差は測定・認証していない。
stateの厳密固有状態からの距離保証ではない。

全結果の区分はCONDITIONAL_ENGINEERING_NOT_CERTIFIED。
採択済み条件付きIEEE binary64・BLAS・math.fsum・underflow/FTZ等の仮定は今回もプラットフォーム全体として証明していない。
132件の有限入力検査で厳密数値certificateへ格上げしない。

## Runtimeとone-shot保全

実際のrunはPython3.11.0rc1、NumPy1.26.4、CPU1 worker、thread環境5項目を各1、GPUなし。
SOURCEの共有DF block/reference、cached JW bilinear/maskをそのまま再利用した。別benchmarkやspeedup比較は行っていない。
runner記録のelapsedは0.44177887216210365 s、maxrss_linux_KiB=62088、product contributions=27781。
SOURCEが設定したRLIMIT_ASは4 GiB、per-file（result/log）は256 MiB、contextは4 MiB。scientific wall-time capはnull。
この一回での記録であり、一般のruntime/精度保証や未到達pathの検証ではない。

外側launcherはA固有の/tmp exclusive markerを作ってからfrozen runnerへexecした。
その[markerのbyte copy](../../../artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot/exclusive_launch_marker.json)、[launch完了記録](../../../artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot/launch_completion.json)、
[execution context](../../../artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/execution_context.json)、[raw runner.log（0 bytes）](../../../artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/runner.log)を保全した。
outer exit0は結果保存処理の終了であり、科学的PASSを意味しない。判定元はresult.status=TECHNICAL_STOP。
結果directoryのexclusive guardも保持し、marker消去・retry・resumeは0。

## 保存監査・公開・停止

[saved-only auditor](../../../scripts/tracks/algorithm_codesign/audit_m1s_gate0_v4_saved_result.py)はstdlibのbytes/JSON/hash/status確認だけを実施。
保存bytes・JSON・固定hash・statusについて65項目PASS、FAIL0。
NPZ配列decode、Pauli再生成、誤差式再評価、W/v/rho再計算は0。
[監査記録](../../../artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot/saved_identity_status_audit.json)、
[evidence index](../../../artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot/evidence_index.json)、
[publication receipt](../../../artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot/publication_receipt.json)を参照。
SOURCE・入力・正式契約・旧採択記録・旧成果・親worktreeを保持する。

現状は技術的precision unresolved。新契約案の作成、閾値やcoverage変更、新SOURCE／新authorization、再計算、研究方向の採否は今回行っていない。
次の科学判断・再実施は別のユーザー指示を要する。T1c/Q4/Q5、simulation、compile、総資源評価へ進まずmandatory STOP。
