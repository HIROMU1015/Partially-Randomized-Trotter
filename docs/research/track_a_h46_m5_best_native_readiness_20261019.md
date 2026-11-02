# H4/H6：m5_best native pilotの実行準備

2026-10-19。**adapter・並列launcher・4実行契約の封印を完了。実分子のsampling、signal、compileは0、有効grantは0、mandatory STOP。** [初期計画](track_a_h46_m5_best_parallel_plan_20261019.md)から、実行用の接続を追加した。係数最適化や旧検証の再実行から始めていない。

実行SOURCE：`7a4d126ad2df622873aebdb372487c8fd956de4c`。branchは `track-a-h46-m5-best-parallel-readiness-20261019`。人工検査は作者自身によるもので、独立reviewとは呼ばない。

## 固定した取得範囲

既存の数値4次公式 `4th(m5_best)` をPF側固定commit `20d64c2f…` のbinary64係数・元の演算順で使用する。11段のsigned S2合成。係数残差が非ZEROであることだけを棄却理由にしない。比較公式はS2、Yoshida4、m5_bestの3つ。旧new_2をm5へ改称しない。

対象はH4/H6直鎖1 Å・STO-3G・T=.8、保存Hamiltonian/state。H4 legacy rank12・打切りp3、H6 tol-only rank19・打切りp5。full PFは全rank。全群q2。PRはK4・signed piece当たりr2、各群技術確認2 trajectory。これは独立確認標本でも費用winner選択でもない。

| 契約 | 取得内容 | evolution | compile軸 | full-H reference job |
|---|---|---:|---:|---:|
| H4_P2_DET | 3公式 × full/B0/B0+I | 9 | 18 | 1 |
| H4_P2_PR | 3公式 × 各2 trajectory | 6 | 12 | 0（共有） |
| H6_P2_DET | 3公式 × full/B0/B0+I | 9 | 18 | 1 |
| H6_P2_PR | 3公式 × 各2 trajectory | 6 | 12 | 0（共有） |
| 合計 | 24 signal条件、PR12 trajectory | 30 | 60 | 2 |

referenceは系ごとに1回だけ、DET契約の予算として独立予約し、hash付き保存値を両laneで共有する。以降のworkerがfull-H reference solverを重複起動しない。既存NPZの復元とnative preparationの構築は登録するが、SCF・DF分解・新状態生成はしない。preparationはfull/打切りの2種類をprocess内にcacheする。

## 実装した接続

`scripts/h46_m5_native.py` は既存のexplicit Gaussian、DF native S2 lowering、RTE sampler、occupation oracle、full-wrapper compilerを再利用する。m5を既存nativeの未対応labelへ直接渡さず、順序を保ったsigned S2 pieceとして下ろす。scalarは全q・全pieceの最後に一度だけ適用する。B0は元constant、B0+I/PRは元constant＋抽出tail identity。full-H referenceを位相回転して合わせない。

controlは `symmetric_directional`。compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/couplingなし。compiler hashは `1239304c…`。native policyは新しい `M5_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1` とする。

**旧main profileは入力参照情報であり、新native系列との費用同一性を認定していない。** 出力に `old_main_profile_is_reference_only=true`、`old_main_costs_reused=false` を保存する。旧ordinary高次PR、旧main model/compiled費用を無条件に同じ費用表へ混合しない。主表への接続は後続で完全なnative/preparation/profile joinが必要。

各代表24本で低branch、漏洩、独立occupation vector、Hadamard両軸、compile前後両軸の計168主要比較を行う。log_Bの平均lawとの照合24件は別。各誤差はfiniteかつ1e-9未満を要求する。cosine PASS後にsineへ進み、同じtrajectory/eventsをpairとして保持する。60軸以外の隠れたcompileは追加しない。

signal出力はfull-H reference、exact-tail、finite-RTE corrected/raw、bias成分、log_B、正規化closure、最大中間normを対応させる。allowanceの区分は `EMPIRICAL_DIAGNOSTICS_ONLY_NOT_ADOPTED_FOR_SHOTS`。shot用u・headroom適格性・厳密bias保証はnull。今回のtechnical pilotで新allowanceを採択したり、旧1e-9をshot保証へ流用したりしない。shot再会計・ranking・最良q・fitは対象外。

## 並列化とSTOP

`launch_h46_m5_pilot.py` は4つの個別grantを要求する。grant・plan digest・marker・ledger・output・STOP eventはlaneごとに独立する。1つのspawn process poolを共有し、round-robinで4laneへ仕事を配る。PRの同一群は代表sample0のPASS後にsample1を投入する。logical failureでは該当laneの新規呼出しを止め、無関係な正常laneは継続する。STOP前に入った処理・費用は保存し、返却・自動retryしない。pool自体の崩壊・own signalなど実際のプロセス異常では全体STOPを記録する。

worker数はfresh idle CPU・RAM・job数で決め、過去4/32/60 workerを上限として引き継がない。RSS計画値は旧Y4実測×11/3＋oracle workspaceの推定であり、実測上界・固定メモリcapとは呼ばない。実測RSSと空きRAMを記録し、次のtaskを収容できなければ新規投入を止める。既存の正確なcanonical fingerprintキャッシュもprocess内で使う。BLAS/transpilerは内側1、外側候補/trajectoryで並列化する。launcherの環境変数は終了時に復元する。

host/上位cgroupの一時的なPSI、CPU競合、無関係なworktreeのHEAD/statusは停止条件にしない。SOURCE/入力/runtime破損、数値不一致、own process異常、実際の継続不能な資源不足はSTOP。変更拒否の範囲は実際の入力・SOURCE・compiler/runtime・grant・ledger・marker。科学SOURCEや旧grantを実行中に書き換える設計ではない。

各laneは成功・失敗にかかわらずmandatory STOP。Phase3・モデルfit・追加compile・費用再会計・sampling拡張へ自動移行しない。pilotに費用モデルの2%判定はなく、その基準は別契約のモデル検証段階で固定する。

## 封印・人工検証

4契約のprivate plan SHAは以下。公開bundleにはscope・digest・無効grantを保存し、内部pathと実データ・raw logを公開しない。

| 契約 | plan SHA256 |
|---|---|
| H4_P2_DET | `e5102574c9d7fc518cea965874ce1995edf0f72144365d0e507e574aed341400` |
| H4_P2_PR | `7b96232a6b1c7faea178afb1ce3a160208f291676062c833dd1efdec89bce88c` |
| H6_P2_DET | `8bcabef8420e8776607b5aba6d1df3d47703c845c70f2a8a39a289b2150983ea` |
| H6_P2_PR | `2103dbe0db08d286eef3c41f4af4ad52df3e348db8e9f4a8bbd817d127d34ff5` |

H4各303、H6各305 fileのSOURCE/input/provider/runtime closureを照合した。暫定seed120個を、既知高次PF探索・確認等4,243個と照合し衝突0。未知の外部jobを網羅したとは主張しない。暫定seedはevent抽選をしておらず、目的はtechnical pilot、後の独立確認は別namespace・新標本にする。

人工・既存回帰120件PASS。うちm5の計画/新native系28件（初期12＋今回16）。人工1/4 system qubitの負時間、scalar、directional branch、独立oracle、両軸compile、paired台帳から結果保存までを確認した。4 processの人工dispatchでも全lane完走と、1lane故障後の他3lane完走・全worker回収を確認した。実分子用経路の成功はまだ主張しない。

実際の4無効契約をCLIへ渡し、`EXPLICIT_PILOT_SAMPLING_COMPILE_APPROVAL_REQUIRED`、exit2、output/marker生成0で拒否されることを確認した。有効grantは発行していない。旧SOURCE・結果・grant・ledger・他worktreeを変更していない。

## 次の担当

**ユーザー**：上記SOURCE・4契約・technical pilot限定の取得範囲を確認し、4lane計60軸/PR12 trajectory/full-H reference2件の一回実行を承認するか判断する。今回の準備指示を実行承認へ読み替えない。

**Codex**：明示承認後にだけ新grantを結合し、SOURCE・入力・runtime・fresh資源・未使用root/markerを再照合して一回起動する。取得結果を保存し、全worker回収・native exitを確認してSTOPする。後続Phaseの実行は別承認。
