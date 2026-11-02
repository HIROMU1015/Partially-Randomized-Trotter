# H6原metadata 63候補：追加取得と固定共通域の実行契約

2026-10-22 JST。利用者は旧STOPの63 B0I候補の不足観測取得を別科学予算で明示承認し、その成功後の固定H6 276候補signal/shot/compile/会計まで包括承認した。追加確認待ちは挟まない。基点0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c、専用branch `pf-common-h6-metadata-execution-20261022`。旧H2/H4結果・raw・grant・STOPは読み取り参照のみ。

対象はlinear H6、1 Å、STO-3G、tol-only1e-8 rank19、T=.8/P=0、L_D=5..15/Full19、q1/2/4/8、δ=.8/.4/.2/.1、元S2/Yoshida4/m5_best signed係数。Qiskit1.3.0・rz/sx/x/cx・opt1・seed17、full Hadamard、symmetric-directional、元scalar一回。原snapshot SHA99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005とHamiltonian/stateを照合し、入力生成・新reference・PR samplingは0。

## 観測方法

旧B0I serializerはB0のcorrected/exact-truncated scalar signalをexp(-i*.8*C)で回転し、full-H referenceは回転しない。厳密算術ではscalarが全blockと可換でnorm不変。しかし旧観測は各merged primitive・各outer scalar後を含み、保存B0 scalar signalだけからbinary64中間norm/closureを復元することはできない。保存B0 metadataの無条件転用は行わず63件を新取得する。

固定providerの `h6_matched_port_v1.compact_signal`、`ax2b_h6_pilot_port_v1.recurrence_paths`、`ax2b_h6_contract.primitive_time_schedule` を使用する。既存common nativeと同じ明示Gaussian準備のblockに元native primitive作用を適用し、400次元spin sectorへのprimitiveごとのleakageを1e-12未満で検査する。独立occupation行列eigh forwardと比較し、元のmerged S2/Yoshida4作用順序・outer scalar順序を維持する。最終B0I scalarは元overlayと同じ位置で作用させる。signed PF数式・元Hamiltonian/state・referenceを変更しない。

maximum_intermediate_normは初期1・各primitive/outer scalarの原観測点・最終B0I vector回転後でのnorm最大。raw/correctedを別実行し、最終vector間のnorm(raw−exp(−log_B)corrected)を観測する。決定論log_B=0でも0を代入せず、実vector差を保存する。新scalar後normを観測することはB0Iの最後の作用を含めるためで、元の途中観測点を省かない。scalar信号closureとvector closureを混同しない。

52旧signalはcorrected/reference/signed bias/旧Cを各1e-9未満で照合し、旧signalとu/marginは保持して新観測だけを結合する。原raw未回収11件は新signalとして明記。元B0Iの保存endpoint allowanceを読み、保存u未満にしない。原log marginは旧overlayがB0からそのまま継承する定義である。新データを原本復元と呼ばない。

## 予算・grant・STOP

実行前のstdlib準備が対象63IDと元merged scheduleから内部callを正確に算出し、`plan_APPROVED.json`・承認receipt・SOURCE/input/runtime closureへ封印する。別METADATA grantはnative corrected63/raw63、独立forward63、vector closure63、旧signal比較52、最終vector回転189。native primitive build/作用は元schedule作用数の各2倍、forward作用は元schedule作用数。outer scalarはnative2×Σq・forwardΣq。context snapshot/reference/oracle/full preparation各1、11prefix tail scalar、独立sector matrix/eigh各16。追加SCF/DF/state/reference/compile/samplingは0。具体的整数はgrant/ledgerと結果receiptを正とする。

METADATA成功後、新しいCOMMON packを作成し、旧原観測111と新63を参照結合した174候補を保存値として受け入れる。固定不足102=基礎51+overlay51、forward51、signed S2 piece1419・block作用各30588、shot1380 endpoint、compile pool224候補448軸・native piece3625以下を維持する。同一expm行列の再利用だけを追加し、ユニークexpm構築数も別counterに封印する。科学条件・作用順序・算術は不変。

SIGNAL/QUALIFY/COST/ACCOUNTは各別grant・ledger・exclusive marker・終了receipt。入力seal相違、旧signal/forward/native数値不整合、予算超過、科学call消費後の例外は当該stage STOP。同grant retry/resume禁止。正常終了stageから次の承認済みstageへ進む。CPU/RAM/ディスクの実空き観測に従い外側process数を動的選択、内側BLAS/compiler1。共有設定・他ユーザー操作は0。

共通政策はH2/H4で固定した通常signed、5精度、α_axis=.025、経験的u=max(floor1e-9,10closure+64eps(1+norm)(1+stages),saved u)、log_B margin>=1e-12/saved margin、u/headroom≤.01、凍結binary64 shotと別整数照合、compiled RZ×analytic shot。元PRのcohort/mean/variance/SE/sourceは変更しない。外部レビュー・新sampling・pushはしない。H6完了後588候補の参照統合、局所監査・local commit後mandatory STOP。

実装 `scripts/h6_pf_metadata_completion.py`、H6 loader/cache接続 `scripts/pf_common_run_native.py`、one-shot `scripts/pf_common_run_contract.py`、人工 `tests/test_h6_pf_metadata_completion.py`。人工4件PASS、既存10件の結果を確認後にSOURCEを固定する。

## 取得後記録

科学SOURCE3cff9a4baのMETADATA stageはsnapshot_restore1後にloader shape検査でSTOP。旧入力SHAは一致。signal作用・再観測0。修正SOURCE458ea00ddを別に保存し人工5PASSだが、失敗時STOP指示に従い実H6は再実行しない。[終了監査・不足状況](track_a_h6_pf_metadata_STOP_20261022.md)。
