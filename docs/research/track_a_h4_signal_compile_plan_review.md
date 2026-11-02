# H4 signal／compile次段の準備と停止条件

2026-10-06 JST。利用者の次段準備指示を受け、
[資料入口・承認対象・未実行command](../../artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/README.md)へ
INPUT_BOUND plan、別認可/review草案、source/input binding、stage容量とread-only CPU/memory/quota観測をまとめた。

**H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP**。
H4 linear/neutral singlet/STO-3G/DF12、追加6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、
8 system＋ancilla1、T=0.8、二次DF-prefix PF/canonical finite-RTE、
L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1、
登録218 template/距離・random32 paired trajectoriesの科学条件は不変。
将来signal1,308、logical wrappers/actual invocation cap74,784、6 CPU workersの案である。
SOURCE `049e69919af16ad29a67a217dc7a407d6b1754a6`の19 paths、run02出力先、既存plan/auth/review/bundleは不変。

入力生成は[run02完了監査](track_a_h4_worker_bootstrap_run02.md)でINPUTS_FROZEN_STOPを確認済み。
今回6 input recordとgeneration-freeze fingerprintを新planへ機械転記した。
plan SHA-256は`0d3d7fbda75bec6b4629b99fb783ae216006565c0ea3def3506e673ead54d641`。
この準備ではNPZ/NPYを開かず、generation-freeze JSONとbyte-budget.journalだけread-onlyで確認した。
production schema/固定source・contract・dependency・compiler metadataのbindingを検査し、false review、
旧generation認可、明示launchなしを拒否した。signal/seed/sampling/build/compile/transpile/worker/taskset/GPUは0。

容量planning bound3.328216552734375 GiBを0.25 GiB単位で上方丸め、
**必要3.5 GiB・560,000 inodes**とした。
2026-10-06 18:44:16 JSTにext4 /homeの利用者向けavailable **3.4193763732910156 GiB**、
available inodes225,814,677、user/group/project quota非有効をread-only確認。
余裕込み必要量から**約82.6 MiB不足**のため起動不可。
current freeは小計を上回るが、余裕込み判定を引き下げてlaunch成立とはしない。

最大72時間の監視259,200 files、74,784 checkpoint、149,569 ledger deltas、全workerログ8 KiB、
1,308 signal JSON（302表示点/paired metrics、512 KiB余裕）、128-byte journal row、
temp/final同時出版、4 KiB block丸め、directory/extent/ACL/journal余裕を含めた。
JSON幅/directory余裕は仮定、ログ/出版上限等はsource値と区別する。
ext4の既存固定inode tableはfree-blockから重ねて確保せず、必要inodeとdirectory等を別途計数した。
根拠は[Linux kernel documentation](https://www.kernel.org/doc/html/latest/filesystems/ext4/inodes.html)。
以前の10 GiB全量空き確保案、input stage3 GiBと4 KiB/inode余裕は旧監査履歴として残す。
campaign charge cap10 GiBは維持。input既消費164,698,732 bytes＋次段見積りの累積charge合計は4,737,395,180 bytes。
wall2.679710050113499秒も累積72 hへ引継ぐ。この準備はキャンペーン完了や性能を保証しない。

CPU候補[3,5,6,7,8,9]・6 workers・own-run mask0x3e8、effective memory約983.119 GiB、
fresh準備観測のCPU/memory/PSI/OOMはPASS。別stageのCPU利用許可や資源予約ではない。
**stage reviewはapproved=false、別のCPU利用許可・最終review・明示launch未取得**。
最小解決はfixed filesystemの利用者向けavailableを3.5 GiB以上にする状態の確認であり、
削除・移設・予約・output root/共有設定変更は実行せず利用者への提案に留める。

承認後もfresh resource/capacity検査が不合格なら起動しない。
この段階の将来実行は6 frozen inputsのsignal/compileだけ、一度、失敗retry/resumeなし、
map完成後MAP_COMPLETE_STOP/研究判断null/next-stage=false/mandatory-stop=true。
H6/Track B/追加trajectory/長RPE/最終total-costへは進まない。
今回旧科学結果/status/validation manifest・source・保存証拠・共有環境・他jobは不変。
軽量の準備bundleと索引だけを固定し、科学NPZ/runtime/checkpoint/cacheはGitへ収録しない。
