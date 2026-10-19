# H6 full m5：CT01による保存cosine受入とsine一軸の続行準備

2026-10-19。**元compilerの保存cosineを新基準で受け入れる記録を作成し、未実施sine一軸の実行準備を完了した。実分子compile・回路作用は今回0件。local commit後mandatory STOP。**

利用者の「その方針で進んて」を、直前に示した保存cosineの受入記録と未実施sineの続行準備への承認として扱った。新規compileの実行承認は含めず、専用proposalのgrantはINVALID、実行承認receiptはfalseのままである。公開pushなし。

- branch：`track-a-h6-m5-ct01-sine-readiness-20261019`
- 実装SOURCE：`885bbd7199c9ada26704b7db75b99767734e92e7`
- 採択済みCT01文書：`94d18a979cdc1a390fa7e2eb3ae2f18f700b47d5`
- DRAFT plan SHA256：`2a91769d1c38ffa6b8c9c098f79783b5303a39880f423f617b1ced7b49a0d0b4`
- 証拠入口：[summary](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_readiness/2026-10-19/summary.json)、[plan公開projection](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_readiness/2026-10-19/plan_DRAFT_projection.json)。projectionは実行configではない。

## 対象と元compilerの固定

H6直鎖・隣接1 Å・STO-3G・保存tol-only1e-8/observed DF19、full PF p19/q2、T=.8、P=0、`4th(m5_best)` / `PF_M5_BEST_20D64C2_BINARY64_V1`。11段のsigned S2合成、symmetric-directional control、scalar一回、full Hadamard wrapper、13qubit/ancilla12、保存low状態を維持する。

Qiskit1.3.0、rz/sx/x/cx、元のpreset opt1、seed17、NumPy1.26.4/Python3.12.3。compiler設定hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、native policyは`M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1`。provider SOURCEは`c31ef424175c02379eafebe8fc6c7548c1d9aabe`。

直前の代数的1qubit縮減や再合成無効のcompilerは、今回のsine経路に使用しない。旧58軸と同じ元compilerを使い、CT01で変更したのはcompile前後のraw状態ベクトル2-norm差の閾値だけである。finiteかつ**厳密に1e-7未満**を要求する。occupation/native/Hadamard等の既存検査は1e-9未満を維持し、phase alignment・事後位相補正・再normalizationは行わない。

## 保存cosineの受入（metadataのみ）

元recovery SOURCE `f743a88225f7cb2c8d2c91d7e19813a5780f803b`、plan SHA256 `ffe5d880ecdebdc6616a0e4a41bd3432115503e2feba16f2414a081a2ac098da`の保存receipt、STOP、診断JSON、native exit/reapとQPY/NPZのbyte SHAを結合した。配列・回路は復元していない。

| 項目 | 保存値・判断 |
| --- | --- |
| cosine raw差 | 1.504043707445665e-9 |
| 元1e-9未満基準 | FAIL、旧STOPを保持 |
| CT01 1e-7未満基準 | 新しい受入記録でPASS |
| 保存compiled RZ | 831,662／cosine軸 |
| native/occupation/Hadamard等 | 元の5チェックPASSを再利用 |
| cosineの新compile・再作用 | 各0件 |

[保存cosine受入記録](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_readiness/2026-10-19/saved_cosine_acceptance.json)はpilotの一軸についての新しい記録である。元のquarantine、compiled receiptの`UNVALIDATED_NOT_FOR_ACCOUNTING`、診断のold FAIL、STOP、grant、ledger、markerは書き換えていない。主表への採用・shot込み費用再会計は未実施で、m5のAD03は保留する。元58軸にこの別受入1軸が加わった状態を、旧runの全60軸正常完了とは記載しない。

## 未実施sine一軸の処理

[adapter](../../scripts/h6_m5_ct01_sine_native.py)、[契約](../../scripts/h6_m5_ct01_sine_contract.py)、[launcher](../../scripts/run_h6_m5_ct01_sine.py)、[metadata準備器](../../scripts/prepare_h6_m5_ct01_sine.py)を新規追加した。旧科学SOURCEは変更していない。

1. 保存cosine wrapperの末尾H直前にSdgを一つ挿入してsine wrapperを作る。native evolution・scalar・preparationを再構築しない。
2. 保存cosineの未compile最終ベクトルに、ancillaの`H Sdg H`だけを作用させ、sineの期待ベクトルを得る。ベクトルの二つの半分をa,bとすると、x=(a+b)/√2、y=(a−b)/√2として、結果は((x−iy)/√2,(x+iy)/√2)。保存状態の単一ancilla変換であり、新しい物理的入力状態・signal/oracle/referenceの生成はしない。
3. その期待ベクトルのHadamard値と保存signalのimagを1e-9未満で照合する。
4. 元compilerでsineを一度compileし、そのcompiled回路を保存初期状態に一度作用させる。期待ベクトルとのraw差が1e-7以上・非finiteならSTOPする。
5. 軸別単価receipt、全結果SHA、native exit・worker回収を保存してmandatory STOPする。

末尾変換は人工回路について元providerのsine builderとfingerprint・raw演算で一致した。実H6での変換・sine精度・単価は未検証で、実行時に検査する。sine期待ベクトルの全未compile回路による再生は省き、compiled一作用との照合を残す。

新規compile上限1、compiled状態作用1、sampling0。順序固定の3段階台帳は各stageへ入る前に一回予約し、再入場を拒否する。失敗時の自動retry・追加軸・再fit・費用再会計・次Phaseは認めない。

## 並列化・資源・停止

今回は一つのsine回路にcompile→作用の依存があるため、productive worker一つと独立監視を用いる。元Qiskitのnum_processes=1を維持し、BLAS等の過剰な内側並列を抑える。末尾変換はNumPyでベクトル化し、保存native/occupation/cosineを再計算しない。独立した追加候補を作って並列数を増やさない。

起動直前には、SOURCE・入力13 role・runtime全2025file・protocol Git object・保存cosine受入・新承認・新grant・未使用output/markerを再照合する。今回の封印は実際に使うSOURCE/runtime/入力と旧結果の保護に限定する。他worktreeのHEAD/statusを固定しない。

同一点の旧実測由来RSS proxyは9,066,639,360 bytes、artifactサイズ由来disk proxyは259,342,552 bytes。いずれも起動可否の見積りであり、恒久的なメモリ上限・runtime capではない。fresh gateで利用可能CPU・実効空きRAM・diskを確認し、一jobを収容できなければ起動しない。host/上位cgroupのPSI上昇や他worktree資料更新だけでworkerをkillしない。自身の異常、OOM、継続不能な資源不足、入力破損、科学的不一致は未検証PASSにしない。

## 検証・保全

- 新規人工13 test method＋関連する元recovery回帰15件、全28件PASS・skip0。新adapterの人工end-to-endは、同じ13qubit形の小回路でPASSとraw不一致STOPの両方を確認した。実分子の回路・配列は使用していない。
- 元sine builderとの一致、非zero global phase、paired入力の独立性、terminal変換、厳密な閾値境界、非finite、先行PASS・来歴改変、stage再入場、spawn/回収、draft拒否を検査した。[人工log](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_readiness/2026-10-19/test_h6_m5_ct01_sine.py.txt)、[回帰log](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_readiness/2026-10-19/test_h6_m5_recovery.py.txt)。
- 人工試験の初回不一致は、Qiskit Statevectorへ同じndarrayを渡し、global phaseの入力変更が次の比較へ伝わったことによる。各比較の入力をcopyして解消した。末尾変換式・元科学SOURCE・保存値は変更していない。
- metadata検査16項目、封印等の改変10種拒否、DRAFTを渡した実CLIのexecute/superviseがともにexit2で拒否されることを確認した。prepare-only receiptはflagだけ変えても実行承認として拒否される。[監査](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_readiness/2026-10-19/metadata_audit.json)。
- 旧SOURCE/結果/grant/ledger/marker/STOP等2791fileは作業前後のSHA不変。既存58受理軸と旧pilot/recoveryのSTOPを保持した。[保全](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_readiness/2026-10-19/preservation.json)。

報告・リンク・manifest・旧履歴・無効grant・未起動等の[構造検査40項目](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_readiness/2026-10-19/structure_validation.json)もPASSした。

以上は作者Codex自身の検査で、独立レビューではない。新sampling・実分子compile/作用・signal/oracle・SCF/DF/state/reference生成・費用再会計は全て0。privateのQPY/NPZ・実行経路・有効grantは公開していない。

## 次の担当

**ユーザー**：固定SOURCEに対する、未実施sine一軸のみの一回実行を判断する。保存cosineと事前検査を再実行する必要はない。

**Codex（別途実行承認後）**：新しい承認receiptを結合した別authorization proposal・有効grantを作り、同じscope/SOURCE/入力/runtimeをfresh gateで照合して一回実行する。今回のprepare-only receiptとINVALID proposalを上書きしない。結果保存・終了監査後mandatory STOPし、主比較採用や費用再会計へ自動進行しない。
