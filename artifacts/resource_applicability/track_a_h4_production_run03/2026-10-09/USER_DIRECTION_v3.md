# H4 run03の利用者指示と未承認の累積上限

2026-10-09 JSTの最新利用者指示（原文）:
「これはしなくてよい。実際に本計算をしていき問題が出たら解決していく感じにする」

対象は「人工回路の限定compile検査」。この追加diagnosticを省き、実H4 signal/compile mapで問題が出た場合は
first STOP原因を保存してown-runだけ停止し、その証拠から修正する。元run02科学compiler原因の未特定は記録するが、
追加人工compileを起動の条件にはしない。自動retry・次stage・入力再生成・旧partial/cache混合は認可しない。
一度の本計算への利用者意図は受領済み。過去に採用されたenvironment/compiler・CPU・observer条件を再質問しない。

今回の指示に新しい数値上限の承認は含まれない。現累積cap13GiB/74804 actualを保持する。
run02の消費/予約を返却せずcarry21 /8692723164B /5766.582514658794秒へ固定する。
次の全map worstは17429694796B=16.232668232172728GiB、actual21+74784=74805。
必要な追加承認案は累積output charge13→17GiB、累積actual74804→74805だけ。
17GiBは整数GiBで必要量を満たす最小値、余裕823916212B。数値以外の72h/監視/各role資源/科学/compilerは維持。
予算改定は未承認であり、draft approved=false/runtime_authorization=false/allowed_cpus=[]/sealed=false。
承認後にartifact/binding/seal・独立最終review・fresh resource/未使用one-shot検査が合格した場合だけ一度起動する。
この準備作業では科学array読込/回路build/compile/transpile/affinity/worker/GPU/本体起動を行わない。
