# H8 非相殺COST_COMPARISON：実装・人工検証・未承認契約 v3

準備は完了。実H8の新しい費用・順位・削減率・SE再解析は0件である。
`approved=false`、`runtime_authorization=false`、`execution_sealed=false`、CPU `[]`、absolute command `null`、有効grant0件を維持する。
今回の依頼を費用再会計の実行承認として扱わず、資料固定後にmandatory STOPする。

## 固定SOURCEと資料入口

- SOURCE：`ac932974a0f250be0df5720e1b6dad06f9ff9535`。資料commitはこのSOURCEを親とする別commitに固定する。
- 専用branch：`track-a-h8-noncancellation-cost-readiness-20261017`。
- plan canonical SHA256：`f171099baeb4fdd56fb40748292db1c70074479b711702f360c9d88921fc96ad`。
- input contract byte SHA256：`8960fdc1df16b71fd1afe73a409daa7ba05e26b36a7a5b8bee788b13fc213ce0`。
- private config canonical SHA256：`0050ca6b6cb3669e19f1ea71954429a3d5d90c7081265161a04dd992643581fd`。
- 無効grant canonical SHA256：`8083f477874c4dc8b5a988bf33f2f6e3a8dbfb0507d43371c6065ae558cd80f8`。
- [準備bundle](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_readiness/2026-10-17/README.md)、[SOURCE closure 23件](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_readiness/2026-10-17/source_closure_v3.json)、[入力46件のbytes/SHA・公開commit/path](../../references/h8_noncancellation_cost_inputs_v3.json)。
- [runner](../../scripts/run_h8_noncancellation_cost.py)、[新契約](../../scripts/h8_noncancellation_cost_contract.py)、[cost adapter](../../src/trotterlib/h8_noncancellation_cost.py)、[人工tests](../../tests/test_h8_noncancellation_cost.py)。

SOURCE/test/reference 13ファイルを明示commitした。うち7件は旧非相殺SOURCEからのbyte同一copyで、旧runner・旧planを変更していない。
通常Phase Cの固定kernel・資源監視・wait4 child cleanupを再利用し、新cost-only adapterと契約を6ファイルで追加した。
SOURCE専用cloneと資料cloneを分離し、既存repositoryのworktree登録を行っていない。
public config bindingは監査用であり、実行可能configではない。private path mapping、raw、消費済みgrantは公開しない。

## 完了済み証拠の接続

|対象|固定結果と確認範囲|
|---|---|
|非相殺shot|[完了結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f902229cef9da26cb6feeff09eb35860b6115a6/docs/research/track_a_h8_noncancellation_shots_result_20261017.md)。SOURCE `cbfd7cacad8a5828c0dbfd284a99ef0871860e66`、plan/grant/input/raw/terminal SHA、native exit0/reap、1,900行・380軸・PR84 checksの保存projectionを照合。shot算術は再実行しない。|
|通常Phase C|[通常bias完了結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/docs/research/track_a_h8_phase_c_run01_results_20261017.md)。SOURCE `931755c4c84aa0751dbde7ddf84cb5187b5172ae`、native完了・最終gate・同一入力joinを照合。旧CSV/JSONは比較基準として読取のみ。|
|B2最終|[登録4軸結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fef2db7a79886ab682b652da7266b463c98c7d3a/docs/research/track_a_h8_b2_completed_results_20261017.md)。`PASS_REGISTERED_FOUR_AXES`、軸別SOURCE・compiler/profile/fingerprint・実測・2%未満・native/reap/final gateを受理。単一SOURCEの旧未完了契約を使用しない。|
|B1・D0|B1 `83a3c4a6688e66576f85471bf41a45e29cbfb705`、model `8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`。既存D0 weighted_metricとexact rational、確認paired n32の来歴検証を変更しない。|

B2はB0 p15/q1 S4 cosine/sine各168353 RZ、SOURCE `244c45e752fa703e286d40ece9ad641430ea9c6e`。
B0+I p21/q8 S4 cosine/sine各1852640 RZで、SOURCEはそれぞれ
`2e604e4a3a04eee5090e03d9bb83818d7afd3c87` / `f58da9590597f42ff7260c22d771734096a706bb`。
保存3軸＋最終新規1軸の独立受入を維持し、過去STOP・消費済みgrant・ledger・未settle予約を変更しない。
4軸PASSを全prefix/qのモデル精度保証へ拡張しない。

入力46ファイルは計11,281,117 bytes。旧22入力のSHA、raw結果・元terminal・消費済みgrantのbytes/SHAを限定的に再照合した。
科学array読込・入力再生成・新しいH8数値比較は実施していない。

## 将来の費用比較と解釈

linear H8/隣接1.00 Å/STO-3G/保存tol-only DF rank28/16 modes/sector4900/T=.8。
旧32＋追加152の決定論184候補と、PR確認union6・各n32、q=1/2/4/8、ε=.05/.01/.005/.001/.0005、allowance倍率1/10、α_axis=.025、P=0、state preparation除外のfull-wrapper RZを維持する。

新namespace `EMPIRICAL_NONCANCELLATION`、通常固定結果 `USUAL_SIGNED_SAVED`、厳密bias保証 `null`を分離する。
B0/B0+Iはabs(discard)+abs(PF)、full PFはabs(PF)、PRはabs(outer_PF)+abs(finite_RTE)という完了済みshot結果をそのまま使用する。
保存combined uは一回、headroom・normalization・shot算術は凍結D0どおりで、再計算しない。

単価優先順位はB2登録S4実測→B1 anchor実測→来歴確認済み旧full PF実測→固定B1モデル。
S2のprefix14〜21外は外挿、未実測S4は登録4軸だけで検証したモデル推定と表示する。
旧B0未確立来歴の単価を復活させない。各軸の単価と実測区分は固定通常Phase Cの保存行とも照合する。

決定論費用はN_real×RZ_cosine＋N_imag×RZ_sine。PRは凍結D0 weighted_metricでpaired n32を新shot重みへ再重み付けし、標本SEを計算する。
通常bias SEはコピーしない。保存PR paired metric再現gateはcost stage内の必須条件として人工検証済みで、実値でのgateは将来承認後に実行する。
探索n8混合・新sampling・希少Taylor次数の未観測分を0扱い・母平均優位や選択後winner保証への昇格は禁止する。

exact rationalで全対象から最小を選択し、同費用はlexical ID。旧32と184、PR union6を分け、通常Phase Cとの差はpercentage pointsで報告する。
必要単価の欠測や未解決を除外して全候補最良を主張しない。該当集合の全候補が不適格なら明示nullとし、費用0やwinnerを作らない。
最良候補がモデル推定を含む場合は明記する。全候補実測最良・厳密bias保証・PR母平均優位は主張しない。

## 人工検証と資源・終了保護

新26＋旧非相殺56＋通常Phase C49＝131件PASS。[結果・ログ](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_readiness/2026-10-17/artificial_validation_v3.json)。
B2部分/過去STOP/誤SOURCE/native/profile拒否、shot領域/重複/非finite/receipt拒否、PR seed/SOURCE/compiler/paired再現gate、
非相殺SE再重み付け、exact tie/欠測/null、無効grant、SOURCE/input変更、one-shot/output再利用、first STOP保持と直接child終了・wait4回収を確認した。
実H8データでの費用runner end-to-endは未実施であり、将来の実行成功は保証しない。検証者は主担当自身で、独立reviewとは呼ばない。

固定解析Pythonは`/usr/bin/python3.12 -P -B`。元の非相殺解析と同じstdlib算術を使用し、既存venv・package・共有設定を変更しない。
compiler/profileは保存単価の来歴として固定し、新たなcompileは0。
11MB程度の入力・1,900行をindex化し、軸単価をcacheして一回走査する。重い科学workerや並列stressは不要。
資源上限は設けず、解析1 process＋直接childをwait4するsupervisor、内部thread1を選ぶ。
RSS256MiB・output64MiB/32inode・wall秒単位はadmission見積りであり、上限や実H8測定値ではない。

起動前にSOURCE/input/environment、選択CPU、実効host/cgroup空きメモリ、filesystem/block/inode/quota、未使用output/control/lockを検査する。
PSIと他worktree更新は観測のみでSTOP根拠にしない。自run異常・OOM終了・整合性不一致はSTOPし、所有する未回収の直接childだけをpidfdで停止・回収する。
未承認grantは入力読込・書込・子process・affinity変更前に拒否する。現時点でoutput/control/lock/fresh gateは未作成。

## 残る事項と次の担当

未解決の準備上の技術不合格は0。実H8費用・ranking・削減率・paired SEは未知である。
実行には新しい明示承認・有効grant・fresh資源検査・sealが必要で、現在の準備指示や過去shot/通常Phase C承認は代用できない。

- ユーザー：上記SOURCE/planで、完成済みshotを再利用するCOST_COMPARISON一回（1900行、旧32/184×5精度×2倍率、PR n32 paired費用/SE、通常保存結果との差）だけを実行承認するか判断する。
- Codex：承認後に新grantと未使用output/control/one-shotを結合し、fresh gateを通過した場合だけ一回実行する。完了/STOP後は自動retry・次stage進行なし。

追加signal/compile/sampling/fit・shot stage再実行・通常Phase C再実行は承認案にも含めない。
今回の準備完了でmandatory STOP。
