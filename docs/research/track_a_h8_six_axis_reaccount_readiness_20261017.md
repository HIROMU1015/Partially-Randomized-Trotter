# H8主要6軸：通常・非相殺の保存値限定再会計準備（公開認証待ち）

利用者の一回実行承認を受領済み。保存結果の公開・元receipt照合、専用runner、人工30検査、SOURCE固定を完了した。
新SOURCEのnon-force pushはHTTPS認証エラーで失敗した。公開確認が実行条件のため、有効grant発行0・科学dispatch0、実H8の費用更新・ランキング0。
追加承認は要求しない。利用者の手動push後に公開SOURCEとfresh条件を照合し、既存承認で一回実行する。

## 公開確認と受領

主要6軸結果branch `track-a-h8-key-compile-parallel-handover-20261017` のorigin HEADは
`ce6cd8ead5c0311ec72165c04869e7df4050b75b`。fetchした固定blobと入力copyを照合した。
最終compile SOURCEは`41e466e13f32e9224bb37790cc16f0ff6621c399`、先頭cosine軸は元SOURCE
`352581103b7dbd4fa58e28af7a6ca65570d36e8e`を保持する。

- [6軸結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce6cd8ead5c0311ec72165c04869e7df4050b75b/docs/research/track_a_h8_key_compile_completed_results_20261017.md)：元scalar結果6件、各native receiptのbytes/SHA・identity・exit0/reap、public projection、最終gate、外側native exit0を照合。
- [通常Phase C](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/docs/research/track_a_h8_phase_c_run01_results_20261017.md)：SOURCE931755c4…、保存1900行/10条件/旧32・184。
- [非相殺COST_COMPARISON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23d46e489af054735d9d328aad0342ecdbddf305/docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md)：SOURCEac932974…、別namespaceの保存1900行。
- B1正本`83a3c4a6688e66576f85471bf41a45e29cbfb705`、固定model `8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`を維持。
- B2正本`fef2db7a79886ab682b652da7266b463c98c7d3a`、登録S4全4軸の既存受入・軸別SOURCE・旧STOP保全を元gateで照合。

固定入力45件/4,162,148 bytes。public commit/path・bytes/SHAと、private scalar/native証拠のpathを含まないreceiptを
[入力manifest](../../references/h8_six_axis_reaccount_inputs_v1.json)へ固定した。NPZ/QPY/trajectory配列/元grantを取り込んでいない。
通常の公開会計行には軸別shotが省略されているため、その元入力と同一SHAのD0・B0I・Phase A保存accountingを結合する。
3,800行のID/精度/倍率/適格性と保存shotの構造検査PASS。shot算術・新費用・ランキングは実行していない。
旧B0I保存倍率の文字列`"1.0"/"10.0"`はjoin keyだけ正規化し、元値を変更しない。
3構成が旧32に存在しないこともregistryで確認した。

## 固定SOURCEと実装範囲

- branch：`track-a-h8-six-axis-reaccount-20261017`。SOURCE：`aa984d98a7b438dc953a829a24fe3d7edc9d0a6b`。
- plan canonical SHA256：`bee16900432f5d69369449f134d1ea5b6e02b84f6e66dcb81cdcaea8c392e800`。
- SOURCE closure7 files：新module、contract、runner、native witness、tests、入力manifest、既存B2検証module（変更0）。
- [準備bundle](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_readiness/2026-10-17/README.md)。closure・stdlib profile68 files・入力/終了監査・無効grant・test記録を含む。
- 科学条件はlinear H8/1Å/STO-3G/保存DF rank28/16 modes/sector4900/T=.8、q1/2/4/8、ε=.05/.01/.005/.001/.0005、倍率1/10、α_axis=.025、P0。state preparation除外のfull-wrapper RZ。

新6軸実測→B2登録S4→B1 anchor→既存full PF→固定モデルの優先順を維持する。
変更する単価はB0I p15/q4 S4各668636、B0I p11/q1 S2各42268、B0 p15/q4 S2各223940のみ。
未測定単価は元結果のまま使い、予測呼出し・refitを行わない。費用は保存N_real/N_imagと単価のFraction算術。
両bias modeを分離し、全184/旧32をexact cost→lexical IDで再選択する。
PRの保存費用・paired SE・保存選択・n32は再計算せず、不変性を検査する。
旧32の行・winner・exact削減率も完全不変gateとする。

各mode10条件×旧32/184=計40比較、更新前後の候補・費用・exact削減率・差pp、通常/非相殺差、適格/不適格/未解決/欠測数、
6軸の旧model単価と新単価、全3800行、各新候補の費用変更とwinnerへの寄与を別結果へ保存する。
未解決domainの最良を確定せず、適格費用欠測はSTOP。不適格費用を0にしない。負の削減率も保持する。

## 人工検証と資源・停止

人工30 tests PASS。2mode×1900行の人工fixture、軸別非対称単価/shot、全184選択、exact tie、
旧32/PR/SE/shot/未変更行、型違い倍率join、欠測・未解決・不適格、SOURCE/input/receipt/profile/compiler不一致、
無効grant、再使用拒否、native exit7、所有不一致、ESRCH、非ESRCH、FD/pipe回収、最初の停止理由、外側協調停止を検査した。
新規science・人工compile・既存campaign再実行0。主担当自身の限定差分確認であり、独立別担当reviewではない。subagent0。

新しいCPU/RAM/worker/wall capを設けない。stdlibのみ、native wait witness＋launcher＋単一軽量算術child。
推定peak RSS256MiB、出力64MiB/inode32、数秒程度はadmission用の容量見積りであり実行上限ではない。
実測6軸のcompiler/profileは入力の費用来歴で、今回compiler環境をimport/変更しない。
解析runtimeは`/usr/bin/python3.12 -I -S -P -B`と読み込むstdlib/extensionのbytes/SHAを別profileに固定する。
OS全体の完全なimage固定・旧compilerとの全格子同一性を主張しない。

実行直前にSOURCE/input/plan/grant/公開確認、未使用output/control/lock、CPU許可範囲/負荷、
実効memory/cgroup限界・OOM、FS/block/inode、quotaを確認する。quota utilityが無ければUNKNOWNを明記し、
同filesystemの64MiB割当probeで必要容量を確認して削除する。利用可能な容量をquota無制限と読み替えない。
PSI単独でSTOPせず、own-runの不整合・資源不足・native異常を保護する。
launcherは自分の未回収direct childだけをpidfdで停止・wait4し、ESRCHだけ退出競合として扱う。
外側witnessは停止要求をlauncherへ渡して内側childの回収を待ち、その後launcherのnative終了を保存する。
自動retry/resume0。旧STOP・既存grant/ledger/marker・他worktree・他job・共有venv/設定の変更0。

## 公開待ちと次の作業

実際のnon-force pushは`could not read Username for 'https://github.com': terminal prompts disabled`でexit128。
認証設定は変更していない。新SOURCEの公開未確認を、実行承認未取得とは扱わない。
有効grant0、未seal draft/無効grant保持、科学attempt0。未使用の専用output/control/one-shotを保持する。

ユーザー：以下の資料branchを手動pushする。

```bash
git -C "/home/AbeHiromu/projects/h8-six-axis-reaccount/20261017-01/repo" \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h8-six-axis-reaccount-20261017
```

Codex：remote HEADとSOURCEの祖先関係・公開blobを確認し、元draft/無効grantを変更せず、既存ユーザー承認に結合した新sealed configと一回grantを発行する。
fresh gate後に通常/非相殺を一つのstageで一回実行し、結果・native終了・保全を保存公開してmandatory STOP。
異常STOP時も科学retry/resumeしない。現時点では結果更新・削減率更新は未取得。

184候補全体の直接実測最良、未測定格子の誤差保証、PR母平均winner、厳密bias certificateは保証しない。
対象は保存DFモデルの有限時間complex signalであり、実機のend-to-end energy/QPE費用ではない。
