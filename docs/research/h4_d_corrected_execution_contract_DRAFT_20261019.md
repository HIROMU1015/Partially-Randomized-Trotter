# H4 D 修正後の実行契約 DRAFT

状態：`DRAFT_UNAUTHORIZED`。元runの停止を保全し、新しい実行承認を待つ。
修正code SOURCE：`8a82b9272ee24896e7d7cbb7400d9b8c12d8f804`。

元[固定D契約](track_a_h4_dp_phase0_20261019/h4_deterministic_execution_contract_DRAFT.md)の科学条件、
104登録、55 compile候補、保存適格性/shot、110 compile、最大220 control、決定論会計1 passを継承する。
H4/1.00 Å/STO-3G/legacy固定DF12/8 system qubits/T=.8/P=0、
native symmetric-directional、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/no backend/couplingを変更しない。
新SCF/DF/state/reference/signal・PR sampling/compile・shot再算出・fit・n計算は0。

変更する技術事項：

1. full inputのhash照合は維持し、discard preparationのhashは元builderどおりのtruncated inputへ照合する。
   prefix順序、system数、scalar、空のrandom tailを明示検査する。元builderと実験回路を変更しない。
2. 親で検証するmanifestをspawn workerへ複製しない。各workerへ必要なloader設定だけを渡す。
   SOURCE/input/compiler/runtimeの前後照合、各候補control→cosine→sine、call ledger、STOPは維持する。
3. SOURCE・spec digest・承認receipt・grant・output/lockを新しくする。
   停止runのSOURCE95fdfc8c…・grant・STOP・消費済みone-shot・preparation試行は不変。

新code SOURCEと各file SHAはtechnical_source_binding.json、候補・input SHAはD_corrected_contract_DRAFT.json。
runtimeの科学起動前後のfresh byte照合は承認後に行う。過去のworker55を恒常上限として採用しない。
実効CPU/RAMと既存制御を読み取り、有限55 taskに適した並列度を選ぶ。affinity等は変更しない。
一時的PSI/CPU競合/無関係worktree更新で科学workerを強制終了しない。
own異常・実OOM・入力破損・科学不一致はSTOPし、未確認をPASSにしない。

再実行でもその新scopeはone-shot、自動retryなし。全55候補/110軸・全回収・native driver exit0・
final closure・1040 endpoint会計の受入後に保存・監査しmandatory STOPする。
P/S_STATSはこの承認に含めない。公開範囲を超えるpushは行わずlocal成果物を保全する。

次の担当：ユーザーが、この修正SOURCEと同じ科学範囲の新規D実行を別途明示承認する。
