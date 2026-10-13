# mixed-access source v1 公開確認・次の一回実行scope

2026-10-13 JST。技術準備完了、**科学実行未認可・Mandatory STOP**。

固定payloadは`102e6b31401356d6eea775169041ebdf2cfa6ba0`、branchは`qc-mixed-access-feasibility-source-v1-20261013`。
[GitHub固定準備資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/102e6b31401356d6eea775169041ebdf2cfa6ba0/docs/research/qc_mixed_access_source_preparation_v1_20261013.md)、[runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/102e6b31401356d6eea775169041ebdf2cfa6ba0/scripts/run_qc_mixed_access_feasibility_v1.py)、[source manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/102e6b31401356d6eea775169041ebdf2cfa6ba0/artifacts/qc_mixed_access_source_v1/2026-10-13/source_manifest.json)。

[独立remote取得receipt](../../artifacts/qc_mixed_access_source_v1/2026-10-13/publication/receipt.json)：GitHubから35変更blobs/1,261,361 bytes、16 locked files、12旧優先blobs、27 relative links、6 synthetic source bindingsを照合。local object alternatesなし。科学sourceやtestsをremote確認のために再実行していない。

新runner/module、独立saved checker、正負時間/phase/sector/DF/2workerの22登録外tests、hash付きdependency lock、source freeze、read-only preflightを完了した。sourceと元科学契約は固定。constructorの独立新規算法非採択、N1STOP/N2HOLD/N3CLOSEDは不変。
ここまでの準備は実分子feasibilityの結果でもCI証拠でもない。分子生成・SCF/AO2MO・登録PF queries・quantum compileは0。

次に承認する対象は[既存の固定H4 scope](qc_mixed_access_feasibility_approval_scope_v1_20261013.md)一回のみ：

- H4直線鎖、隣接1Å、z=(-1.5,-.5,.5,1.5)Å、STO-3G/実RHF、charge0/spin0、8 modes/ν4、frozen coreなし。fresh pack一つを生成してSHA封印。SCF max_cycle50、conv_tol1e−10、minao、標準DIIS、再試行なし。
- 全DF factorsと補正を保存し、P2/E2/E4/D2/D4/R2/R4を同じ元Hで比較。L_Dなし、PRなし。
- T1/4、r=1,2,…,1024、PF error154 queries以下。256 output rows×70 input columns、漏れとcontrolled phaseを保持。εU=.001はoperator errorで、基底エネルギー1mHa達成を意味しない。
- coefficient/PF/function/synthesis予算を固定し、partial primitive ledgerとremaining functionのsymbolic frontierを保存。κ3/6/12は価格hypothesis、function/depth/workspace/ground-energy総費用はUNKNOWN。
- CPU2workers/thread1、aggregate RSS4GiB、output64MiB/log4MiB、one attempt/retry0。wall/CPU capなし。FCI/ground/gap eig、量子compile/T synthesis、shots/PR、H2/別geometry/追加gridは0。
- 結果または失敗、独立監査、GitHub公開・再取得を保存しMandatory STOP。重要な結果から継続/backend投資を判断する際はGPTレビューの開始承認を別途得る。

science承認書は、ユーザーの明示承認参照と下記固定bindingを必要とする。

| binding | 値 |
|---|---|
| payload commit | `102e6b31401356d6eea775169041ebdf2cfa6ba0` |
| source manifest SHA256 | `42348ed9329b42de7bbb7495ee260a502e6e4975077b54a40fe1de8af49cd99f` |
| science contract SHA256 | `fbdb937fd8ec60a836260b36fd26d43e007b62dbfb75c52b4ebb35475cc2bf2f` |
| dependency lock | [requirements.lock](../../artifacts/qc_mixed_access_source_v1/2026-10-13/requirements.lock)（SHAはDRAFT参照） |
| sole output | `artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1` |

[DRAFT template](../../artifacts/qc_mixed_access_source_v1/2026-10-13/authorization_DRAFT_NOT_AUTHORIZED.json)はtypeがDRAFTのままで、現在のrunnerは拒否する。今回の技術準備承認や前の別実験の承認を転用しない。
明示承認後にだけ承認記録を作り、runtime host/source/envチェックを通した後、attemptを消費して入力生成へ進む。/tmpの準備venvが失われた場合はhash lockで再構築する。

global manifestの既存description欠落FAILは保持し、個別source保全のPASSと区別する。rootのdirty125 filesは変更せず、独立worktreeで公開した。新しいGPTレビューはこの通常技術準備だけを理由に必要としない。
