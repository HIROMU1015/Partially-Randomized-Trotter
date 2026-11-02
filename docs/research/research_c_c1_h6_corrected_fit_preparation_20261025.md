# C1：H6修正codecのffsim接続・rank14最小1 fit準備

2026-10-25。起点 `00dedc8d4730c3db29794d9f64ec6015308736d1`。
branch `codex/research-c-c1-h6-corrected-fit-preparation-20261025`。

**技術準備完了。新規27＋既存134 tests、29 subtests PASS。実H6のdecode・新fit・数値再解析は0。科学実行は未承認、公開照合後mandatory STOP。**

## 接続したものと維持したもの

[前段の修正round-trip結果](research_c_c1_h6_codec_roundtrip_execution_20261025.md)を来歴として保全する。今回、保存値から科学結果を再算定してはいない。

新 [c1_ffsim_corrected_codec.py](../../src/trottertracks/c1_ffsim_corrected_codec.py) は、専用fit process内だけで次の参照を一時的に差し替え、例外時も元へ戻す。

1. ffsimの `_double_factorized_explicit_cholesky` の呼出しには、同条件で保存済みのXDF rank14をcopyして返す。過去の診断で実initializerとのcomponent同一性を確認した保存入力を再利用する。元g、tol=1e-8、rank14を照合し、新たな分解を重複計算しない。
2. fitterのmoduleへ直接importされた `df_tensors_to_params` を、既存 `real_schur_column_gauge_v1` の `roundtrip` へ接続する。単にutil側を差し替えるだけでは実fitterへ届かないため、この参照を使う。既存の列符号／real Schur／π平面・各projector・J・tensorのgateは無変更。
3. `scipy.optimize.minimize` のprocess内呼出しを観測する薄いwrapperで、渡されるx0が事前sector検証済みsnapshotと完全一致することを確認する。実際のffsim `scipy_func` で開始時value/gradientを記録後、元のminimizeを同じ引数で呼ぶ。結果の生x/jac/status等と終了時診断を保存する。

インストール済みffsim/SciPy、既存venv、旧bridge、旧H2/H4/H6 callerは変更しない。スレッド間で共有するpatchではなく、専用の単一fit processでのみ使う接続である。旧codecへのfallbackはない。

目的関数 `.5*||g-g_DF||F^2`、既存JAX x64解析gradient、ffsimのparameter形式／unpackとL-BFGS-B本体はそのまま。旧H6 callerはoptionsを省略していたので **options=None、callback=None、jac=True** を維持する。ffsimのtolは初期分解打切り用であり、SciPy minimizeへ新たなtolやoptionsを渡さない。

この保存initializer再利用は、修正前の初期XDFを変更する操作ではない。将来の実行では再度保存XDF14とfull-XDF先頭14 frameのrank・J/U・列符号を照合する。保存旧CDF終了結果は初期値に使わない。

## 実行経路

[c1_h6_corrected_fit.py](../../src/trottertracks/c1_h6_corrected_fit.py) と [CLI](../../scripts/run_research_c_c1_h6_corrected_fit.py) を追加した。既存 `SectorCAR`、`rank_assessment`、`tensor_metrics`、`component_check`、`Calls`、`resource_plan` を再利用する。旧one-fit／rank-ladder runnerは再起動しない。

1. default CLIはbyte・JSON・version preflightのみ。科学実行は別途のチャット許可後に限る。
2. fit環境のprepare workerで、保存J/Uを修正codecへ通し、before／restored initial／x0とtensor診断をfresh outputへ保存する。この段階はminimizeを明示遮断する。
3. 評価環境の親processで、元のh/g/scalarからphysical sector400を一度だけ構築。before／initialのsector誤差・guard、codec前後のsector operator同値性を確認する。両状態がDF予算内でない場合、新fitへ進まない。
4. 専用fit workerが実ffsim fitterを最大1回呼ぶ。保存初期J/Uから再packしたx0と資格済みx0の完全一致をoptimizer開始前に確認する。H6では504 parameters（14×6²）。
5. actual optimizer x/jac、success/status/message/fun/nit/nfev/njev、開始・終了gradient、返却J/Uのtensor誤差を保存。親は同じphysical/contextを再利用して終了後sector誤差・guardを評価する。
6. optimizer successとDF予算PASSを別に報告。初期からのtensor objective／sector errorの差と増減も保存する。増減は数値点比較であり厳密な大小証明ではない。結果に応じた初期値への自動置換はしない。

元gから補正したκ、scalar、frame順、sector Nα=Nβ=3、既存engineering guardを維持。4096／8192の密行列や回路は使わない。

## 人工・回帰検証

[接続test](../../tests/management/test_research_c_c1_ffsim_corrected_codec.py) は実ffsimのfit経路を通し、minimizeをmock化した。**実optimizer iterationは0**。2／6次元、det±、π／π近傍、複数frame・一般対称Jを使用した。

| 試験 | 件数・結果 |
|---|---:|
| 新接続：x0一致、pack/unpack、tensor、実loss/gradient、独立有限差分、binding復元、不正入力STOP、人工worker証拠 | 15 PASS |
| 新runner：人工sector end-to-end、予算未達、optimizer不成功、初期gate FAIL、worker例外、fresh output、passive preflight | 12 PASS |
| 既存codec50＋ffsim/SciPy bridge14＋初期化診断fit側11 | 75 PASS |
| 旧round-trip親4＋H2/H6経路14 | 18 PASS |
| 旧初期化診断親 | 1 PASS |
| Stage2 native回帰（独立process） | 27 PASS |
| 一般K回帰（独立process） | 13 tests＋29 subtests PASS |

合計161 tests＋29 subtests PASS。[原本・JUnit・集計](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/technical/)。初回runner試験ではsubprocess mockがlazy import中の`uname`にも適用される試験側の不具合があった。対象worker以外を実呼出しへ返すよう修正し、最終12件PASS。初回失敗ログも保全した。これは人工試験の修正で、科学的FAILや科学retryではない。

旧venv起動回帰を含むbridge14件がPASS。新runnerもPython pathに`.absolute()`を用い、symlinkを`.resolve()`してvenvを失う問題を避ける。保存H6をdecodeしない両環境preflight、固定ffsim util／fitter source照合がPASS。

人工回帰には人工Hamiltonianでの回路検証を含むが、実分子の回路・fit・PF・科学再解析は実施していない。技術PASSをH6 fit成功と扱わない。

## 固定した未承認科学契約

[契約](research_c_c1_h6_corrected_fit_contract_unapproved_20261025.json)／[SOURCE](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/technical/SOURCE.json)／[依存lock identity](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/technical/dependency_identity.json)。SOURCEは既存42 filesを保全し、新規5 filesを追加した47 files。元入力・保存XDF・旧identity／STOP等18 filesはhash確認のみ。

- scope：`C1_H6_RANK14_CORRECTED_CODEC_ONE_FIT_V1`
- SOURCE manifest SHA256：`bf1b942618301ef5a9c8fc466b746bef9ef115e0a631ed57e870e1671c848b14`
- 契約SHA256：`996bb93c44d42a10108dd49ba3ea70dbd5d4dd32e315748d6146dc22b552e55d`
- 保存入力：直鎖H6・1.00 Å・STO-3G、元input SHA256 `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447`
- 初期XDF14 SHA256：`559dc43bf923a421ccc62d05c4cefa0bdf39a48ba68824abf3c8f770c04642b1`
- CDF：通常number、rank14、ffsim0.0.83、L-BFGS-B、tol1e-8、options=None、最大1 fit。新初期分解0、rank探索0、retry/resume0。
- DF予算：CA/8=`0.00019875 Ha`。同じsector norm＋engineering guardで判定。guardは厳密enclosureではない。
- component allowance1e-10、codec tensor allowance `1e-10*max(1,||g_before||F)`、objective照合 `1e-12+1e-8*abs(expected)` は既存のまま。qualified x0は完全一致。
- fresh output：`artifacts/research_c_c1_h6_corrected_fit/2026-10-25/run01`。旧run／STOPは変更しない。

| 科学call群 | 上限 |
|---|---:|
| prepare：保存load1、初期identity1、codec1、tensor2 | 5 |
| fit：input load1、実fitter1、開始value/gradient1、終了value/gradient1、終了tensor1、objective照合1 | 6 |
| 親：snapshots load2、sector context1、physical1、sector assessment3、開始gate1、終了判定1 | 9 |
| **合計logical units** | **20** |

fit内部の反復／value-gradient呼出しは既存方式と同様に1 fit単位へ含め、実績nfev/njev/nitを別保存する。明示的な前後diagnostic2回は別計上。codecはprepare1＋実fit pack1の最大2回、optimizer呼出しは最大1回。callの定義・順序は契約に列挙した。

数値・identity・codec不一致、非有限値、optimizer例外ではpartial evidenceを残してSTOP。通常のoptimizer不成功returnでは登録済みの終了状態tensor／sector診断まで保存し、optimizer不成功と予算を区別してSTOPする。optimizer成功でも予算未達なら`DF_BUDGET_NOT_MET`。初期値採用・追加fit・rank変更へは進まない。正常完了もmandatory STOP。

## 資源・保全・残る制約

技術試験開始時は32 CPU、利用可能RAM約23 GiBでCPUは混雑していた。小さい人工行列の試験をnice10で実行して他jobへCPUを譲り、複数環境と小行列での過剰BLAS並列を避けるため当該test processのみBLAS/OMP1とした。共有設定や他jobに変更はない。恣意的なCPU/RAM/wall上限は設けていない。

将来は実際の空き資源を確認し、独立な初期sector2状態を既存resource_planで並列評価する。physical/contextは終了後も再利用し、主負荷のfitは既存JAX loss/gradientをそのまま使う。新たな高速化を固定SOURCEへ加える場合は実行前に準備し直し、実行中にはSOURCEを変更しない。

[保全確認](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/technical/preservation.json)の通り、旧SOURCE42 filesと保存入力18 hashを照合した。旧結果・grant・ledger・marker・STOPは改変していない。

残る技術的blockerはない。実H6でoptimizerが成功するか、最終誤差が改善するか、DF予算を保持するかは未検証。tensor目的関数の低下だけではsector予算を保証しない。π branchのchartは一意ではなく、optimizer収束／条件数の改善を証明してはいない。厳密CA、PF・native・shots・資源優位は今回の範囲外。

**次はユーザーが固定SOURCE・契約・入力を確認し、このrank14最大1 fitを別途チャットで許可する。許可後にCodexがこのfresh runnerで登録範囲だけ実行する。今回の準備から自動実行しない。**
