# M1-S Gate 0 v5：正式実験契約・採択記録（2026-10-16）

**FINAL_EXPERIMENT_CONTRACT_ADOPTED_BY_USER / CONDITIONAL_ENGINEERING_NOT_CERTIFIED。**
ユーザーの明示指示を受け、独立数学レビューの条件付き受理に基づくv5実験契約を固定した。
採択・公開までが今回の承認範囲であり、**科学計算は明示的に未承認**。mandatory STOP。

## 採択の文脈と固定資料

- [独立数学レビュー原本](../../research/track_b_m1s_gate0_v5_gpt_independent_mathematical_review_20261016.md)：ユーザー指定ファイルからbytes不変で保存。条件付き数学的受理であり、platform coverage認証ではない。
- [採択文脈](../../../artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/adoption_context.json)：今回のユーザー指示全文と、提示されたGPTレビュー全文・§8推奨を結び付けた。別の直前チャット全文は提供されていないため再構成していない。「その方針で進めて」は今回の指示内でユーザーが引用した発言として区別する。現在の直接指示が採択固定の権限根拠である。
- [ユーザー採択記録](../../../artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json)／[正式machine契約](../../../artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/execution_contract_v5_adopted.json)。
- 候補の基点：47da54279838a6b3e1497f36204ba7e53c4857f9。
- 固定SOURCE S：69de022dc8142f703423b1ce0bd42f05e123954e。
- 固定proof：b5a185cc0c3aae0e477bfc52b93061e86feb7dec。
- 正式machine契約SHA256：361e7034850512fcfde76a68ebd5dc3b58de945d1075a696196d75af8ae7547b.

新しい科学レビューをCodexが実施したものではない。受領済みレビューの条件付き判断を採択記録に反映した。
旧[候補](m1s_gate0_headroom_execution_contract_v5_candidate_20261016.md)、証明・SOURCE・既存記録はそのまま保持する。

## 採択した科学scope・精度条件

対象はlinear H4 1.00 Å、STO-3G、8 spin orbitals、保存DF rank12。
HDは保存DF順prefix G3/G6/G9、HRは残りの非冗長signed Pauli-tail。
signed集約後にabs、lexicographic IXYZ、cutoff=0、exact floating zeroのみ削除、HR identityをHDへ移動する。
保存数値sector-ground oracleを再正規化・上書きせず、期待値は保存norm²で割る。delta window／finite-step評価なし。
保存binary64入力が算術誤差targetであり、積分・DF近似・厳密固有状態への距離を保証するものではない。

局所係数上界 alpha_R=gamma_(713+r)*S_R、S_R=sum_tail |lambda_l|G_l²、G_l=sum_pq(|Re g|+|Im g|)を採択。
G3/G6/G9のr=9/6/3、gamma index=722/719/716。観測counter・結果を上界選択に使わない。
既存global再構成・phase・Hermiticity・残差予算と、HR局所係数/law予算の分離を維持する。
解析的十分領域S_R/lambda_hat<=8等は実行gateに追加しない。

[精度17項目](../../../artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/precision_specification_17_adopted.json)は候補と同一。
8 scalarと9 formula selectorを固定し、W/vは2^-24、rhoは2^-12の要求を保持。
oracle絶対残差品質は1e-8 Ha+B_res。1e-8 Haは旧solverの数値受理基準、B_resは別の条件付き演算誤差予算。
IEEE binary64 round-to-nearest、scalar複素積、BLAS、math.fsum、未会計overflow/underflow/FTZなし、exact Hermitian入力等の条件を採用する。
不成立や精度不足なら既存technical STOP／PRECISION_UNRESOLVEDとし、結果を見た閾値変更・implicit retryを認めない。

adopted=trueとcoverage_model_adopted=trueは**ユーザーによる条件付きモデルの採用**を表す。
条件が実プラットフォームで成立すると認定した意味ではない。CONDITIONAL_ENGINEERING_NOT_CERTIFIEDを維持する。
固定SOURCE/candidateのAUTHOR_CONDITIONAL_DERIVATION_NOT_INDEPENDENTLY_REVIEWEDは導出時の起源ラベルとして保持し、
その後のGPT独立レビューと条件付き受理は別の採択文脈へ記録する。SOURCEのラベルを改変しない。

## Runtime・保存・実行境界

[Runtime/resource採択値](../../../artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/runtime_resource_adopted.json)：Python3.11.0rc1、NumPy1.26.4と6 build-file hash、
CPU1 worker、GPUなし、5 thread環境各1、4 GiB memory、result/log各256 MiB、context4 MiB。
Pauli65536、global contribution1000000、scientific wall-time capなし。候補から変更していない。
runtime・memory成功・numerical correctness・実入力の局所予算・W/v/rhoの成功は未検証。

全3分割のstatusを報告し、one-shot、排他出力、未到達理由とfailure分類、mandatory STOPを維持する。
実行にはproof < SOURCE S < 採択契約C < 別実施authorization Aのcommit順とhash整合を要求する。
今回の正式契約固定だけで科学実行authorization Aは発行しない。
契約shape検査のexecution_ready=trueは**契約情報の完備性**であり、actual_execution_permitted=falseである。
科学実施にはユーザーの別途明示承認と新しいv5専用authorizationが必要。

## 非数値検査と保全

[採択static checker](../../../scripts/tracks/algorithm_codesign/check_m1s_gate0_v5_adoption_static.py)で、
候補との4フィールド差分、17条件・runtime不変、原本／SOURCE／proof／入力のhash、commit順、旧authority拒否を確認する。
原本引用・metadata／JSON・AST・raw byte hashingのみであり、NPZ配列decode、予算算定、数値correctness、科学処理は行わない。
前段の122 SOURCE静的PASSは既存証拠として保持し、その全suiteを今回再実施したとは扱わない。
採択検査結果と公開照合は[採択監査artifact](../../../artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1)を参照。

v4のTECHNICAL_STOP、W/v/rho=null、消費済みone-shot、既存SOURCE・結果・契約・authorization・markerを保全。
親worktreeのdirty/untrackedを保持し、他Trackの成果を取り込まない。
新科学計算・実入力/模擬入力の予算評価・Pauli生成・数値correctness・simulation/compile・W/v/rho評価0。
新科学実施authorization/grant/marker0、v5への自動実行なし。**mandatory STOP**。
