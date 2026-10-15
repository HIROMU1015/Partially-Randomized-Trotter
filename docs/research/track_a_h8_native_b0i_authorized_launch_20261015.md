# H8 native B0+I 全登録候補：4 worker one-shot実行認可

利用者の「計算を開始して」を、直前の4 worker SOURCEと固定した全12構成・最大24軸の実行承認として受領した。SOURCE `54ed98c5fa5b7190f77255271f4e3bec8db15630`、準備REVIEW `b654e90e25a59011b1763dbf8ab29f7ae93f6e56`。このcommit時点は起動前であり、最終auth/sealとfresh gate合格後に一度だけ起動する。

[承認plan・byte/profile/input/carry照合・主担当確認](../../artifacts/resource_applicability/track_a_h8_native_b0i_authorized_launch_v1/2026-10-15/README.md)。[実装と旧未承認準備の履歴](track_a_h8_native_b0i_readiness_20261015.md)。

対象はlinear H8/1Å/STO-3G、保存DF28、16 modes/sector4900、T=.8、S2、prefix7/14/21×q1/2/4/8。全12構成を保存signalから会計し、適格な構成を最大2件ずつnative correctness確認した後、cosine/sine最大4軸をcompileする。PR比較は保存確認union6/独立32 trajectoryに限定する。新SCF/DF/state/reference exponential action/PR trajectory生成、matched B0再測定、範囲拡張を行わない。

driver CPU63、worker CPUs[62, 61, 59, 57]、内部thread1。初期gateで別々のidle physical coreを選んだ。affinityは新しいown-runと子孫だけに適用し、共有設定・他jobを変更しない。既存metadata launcherがdriver CPUを共有して1秒観測、最大5秒欠測STOPを維持する。

数値role各AS64GiB/RSS32GiB、metadata AS512MiB/RSS256MiB、headroom16GiB。336.5GiBはAS上限の保守的な起動判定値であり、実際の確保・消費量ではない。stage累積CPU86,400秒、metadata CPU1,800秒、phase/total wallはnull。host full PSI>=1%、cgroup full PSI非zero、OOM増加、資源・容量・inode不足時はfail-closedとする。FS8GiB/40,000inodesと有効quotaをfresh gateで再確認する。現在の/home mountはquota無効のread-only記録を保存した。

固定source13/provider265/profile34,273 file（約1.50GB）と入力7件のstreaming bytes/SHAを照合した。旧H8 native exit0/wait-reap、元driver/launcher identityの不在、既知消費・保留予約・不明消費を保持した。古いexit時刻や未知actualを補完しない。準備中の科学import・NPZ array読込・新科学呼出は0。既存venv変更・install/upgrade/GPU利用・新科学review・subagent利用も0。

人工45件の既存合格を再利用し、新しい人工campaignや詳細科学reviewを繰り返さない。確認は主担当による限定binding照合であり、別担当の独立reviewではない。旧host compiler出力のbit同一性、経験的uの限界、実H8最大RSS・速度の未測定は保持する。

最終contract/auth/review/seal・launch wrapper hashを結合し、直前fresh CPU・memory/pressure/OOM・filesystem/inode/quota・SOURCE/profile/input/carry・未使用output/control・durable one-shot lockが合格した場合だけ一度起動する。完了またはfail-closed STOP後は証拠を保存してmandatory STOP、自動retry・次stage移行は行わない。

privateな実行入口：`/home/AbeHiromu/projects/h6-handoff-evidence/20261015/h8-native-b0i-full-family-run01/`。`launcher_identity_v1.json`、`control01/external_resource_observation_v1.json`、`control01/launch_stdout.log`、`run01/terminal_v1.json`で実際の起動・進捗・終了を区別する。runtime/checkpoint/科学cache/venv/NPZはcommitしない。GitHub pushは今回未実施。
