# 2026-10-16 M1-S Gate 0 v4 初回one-shotの実行・停止記録

正式契約採択後、初回数値correctnessを含むGate 0 one-shotの具体的提案に対して、ユーザーが別turnで「次の作業に進んで大丈夫」と実施承認。
正式採択時の科学未承認記録は変更せず、別authorization A=b70abf111e6459c45c24331a2c7d99c05e7dc228を公開してclean HEADから一回実行した。
SOURCE606036db52da2f080cf846e48824e5e418430762、正式契約5e57955baa79e42f37aa233ac011c60539e431fdを保持。

H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・保存順HD G3/G6/G9と非冗長Pauli-tail、delta評価なし、保存数値sector oracle。
到達132gate PASS、残差1.022703352687838e-15 Haは固定1e-8 Ha+B_resを通過。
全3splitで事前W/v誤差予算が2^-24を超え、PRECISION_UNRESOLVED_BEFORE_METRICS、W/v/rhoはnull。
probability・Pauli expectation・W/v/rho correctnessは未到達。最終TECHNICAL_STOP、科学的no-goは判断しない。
CONDITIONAL_ENGINEERING_NOT_CERTIFIEDを維持。新科学one-shot1、retry0。閾値緩和・SOURCE変更・研究拡張なし。

raw result/context/log/checksum、消費済みexclusive markerを保存し、saved-only bytes/status監査と専用branch公開を行う。
事前のSOURCE準備・正式採択段階のscience0記録はその時点の履歴として保全する。
詳細：[実行結果と停止](../../tracks/algorithm_codesign/m1s_gate0_v4_one_shot_results_and_stop_20261016.md)。
新契約や再実施は開始せずmandatory STOP。
