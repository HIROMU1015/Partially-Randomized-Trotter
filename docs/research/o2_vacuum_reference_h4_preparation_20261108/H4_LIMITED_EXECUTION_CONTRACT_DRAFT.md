# 実H4限定資源比較：実行契約草案

状態：SOURCE_PREPARED / SCIENCE_NOT_AUTHORIZED / DRAFT_NOT_ADOPTED。今回の技術作業は実H4の実行許可ではない。この草案の採択と、一回限定の科学実行への明示許可を受けるまでCLIを起動しない。承認証明書・grantを新設する必要はない。

## 固定候補・入力

保存Linear H4、1.00 Å、STO-3G、legacy DF12、8 system qubits、L_D3、m0/q1/delta=.02/r1/K2、coefficient_atol=0、N1 actual finite-grid Q、selected full、理論stateは元Hの厳密ground。同じ有限複素targetを維持する。state準備は共通の研究scopeとして除外する。

N1結果744e1a09661ebba6841fe5faf456f88c41c47780、科学SOURCE b6dbec9f80da35a88fba28795ad02ca74c9adfa0。旧engine認証fdf00cee3c87d804845497afc89ac6a71f3b3f59は参考契約として保持し、新gadgetの認証に流用しない。旧N2 state・spectral witnessは今回の次段階にもロードしない。

既存m_n0_metadata_executionの五入力bindingを再利用する。directoryは artifacts/o2_a1_new_target/2026-11-06/n1/。今回blobの科学内容を抽出・分類していない。

| 入力 | 固定Git blob |
|---|---|
| SOURCE.json | 9cf56d6570133bde49a692ca1cb7cb064382a0a1 |
| CONFIG.json | ad272e31d0dfc2b5087721a2ed7ab488d663724e |
| LAW.json | 15b40241ba648c0f8750e087425d3baeb053730f |
| POLICY.json | c089717e43b3ab5b1a637e802471a2a6bb40ba9c |
| PREPARATION.json | d7cf105c31b30100ee16dc4fefe41999cd5c009f |

実行前に、今回公開する専用SOURCE commitをfull SHAで明示する。preflightはHEAD／使用SOURCEファイルのbytes／依存環境を検査し、五blobと固定モデルscopeを照合する。元mainや旧科学worktreeを実行領域にしない。

## 結果を見る前に固定するpath・split

LAWのmass exact>0のlabelをUTF-8 bytesの昇順へ並べる。n=0は先頭labelをrotationに使う1 path。正次数は固定K2のorder2とし、先頭3 labelを順にproduct0、product1、rotationへ割り当てる1 path。必要な正label3件がなければSTOP、代替選択なし。order0/order2がともに正、dyadic53-bitと全mass1を確認する。

これはsamplingでも高費用path探索でもない。path取得・正support確認・exact path質量計算・identity確定は次回科学実行のmetadata段階として一回だけ行う。今回、具体的label名やH4 path identityは未取得である。path質量は記録するが、2 pathを代表標本と見なしてQ期待価格を推定しない。

splitはdeterministic forwardの各block、RTEの各application、deterministic reverseの各blockをsemantic unitとし、先頭floor(B/2)をR、残りをLにする。奇数時は早い境界。gate数・angle・価格によるsplit再選択は禁止。既存builderのbasis fusion、開いているframe、scalar/event位相を含むgate列を保持する。最後のRTE applicationには閉じるbasisとevent phaseを含める。

## 方式・compile

1. Controlled PR：既存selected full、diagonal_only control、boundary_optimizedをそのまま取得する。
2. 既知SE：同じL/R、真空phase、CSWAP/P/reset/readoutで構成する。
3. Candidate：同じIR。比較表で別行とするが、IR hash一致時はcompile・数値資格・価格を共有する。

Controlledは9 qubits、参照は17 qubits、fanoutなし。参照8 reset、16 CSWAP、補正P、X/Yとsignal1bit読出しを含める。target state準備だけを除外する。後からcat版やpostselectionを加えない。

候補profileは全方式でQiskit1.3.0、basis=(rz,sx,x,cx)、optimization level2、seed17、approximation_degree1、qubits_initially_zero=False。既存phase-preserving definition展開を全方式に適用する。level1の旧人工価格と数値を混ぜない。現在のA1価格providerと基本profileが一致しても、新しい前処理のため旧価格を流用せず全wrapperを新規compileする。

topologyは全結合と固定line。全結合でnative compileを一度行い、lineは同じnative IRからlevel2、trivial layout、identity initial layout、sabre routing、同seedで追加routing compileする。最終permutationだけでなくinitial layoutも追跡する。これは固定の比較profileであり、hardware routing最適性の保証ではない。

## 科学call上限と保存

一条件・2 path・X/Y・3方式・2 topology =24結果行。重複compileなし。cold/warm・回路取得・phase・compile・operator資格の費用を分けて記録する。

| 科学入口 | 上限 |
|---|---:|
| 五入力load／identity照合 | 1 |
| cost-blind path選定 | 1 |
| state-free保存metadata復元 | 1 |
| boundary traceを伴うbody build | 2 |
| 既存selected X/Y wrapper build | 4 |
| 真空参照X/Y wrapper build | 4 |
| native compile | 8 |
| line routing compile | 8 |
| raw/compiled全入力POVM資格 | 16 |

復元中の小gate matrix検証と数値資格の局所gate certificateを、そのunique keyとhashで記録する。gate variant数は実入力前にはUNKNOWN。得られた有限gate列の各variantを一回証明し、同じSOURCE内のcacheを使う。旧cache・bit・乱数・係数・学習結果は使わない。callごとのSTART／RETURN／EXCEPTIONを耐久journalへ記録し、開始回数も数える。

raw/compiledの資格は全入力operatorの線形代数である。8-qubitの全256入力columnに対するPOVMを評価し、保存groundやpath Born値は取得しない。17-qubit isometry一個は約512 MiB、複数copy/作業配列が必要となる。実行時の空きRAMと他jobとの非干渉を確認し、不成立ならSTOP。CPU/RAM/workerに恣意的な低い固定上限を置かず、依存関係を守りcacheを再利用する。今回の人工時間を実H4所要時間へ外挿しない。

保存はSOURCE、INPUTS、PATH_DESCRIPTORS、phase ledger、局所gate証明、compiled QPY、initial/final layout、RESULT_ROWS、CALLS、CALL_COUNTS、RESULT、STOP。RZ/CX count、各depth、total depth、qubits、reset/measure、classical elapsed/peak RSSを記録。QPYは再compileを要しない成果物として保存する。

T count／physical reset-readout latency／hardware校正誤差はUNKNOWN。variance、shot数、Q期待費用、全RPE総費用を算出しない。

## 数値保証と受理

H4_NUMERICAL_CORRECTNESS_AND_ERRORの方式で、各取得pathの任意正規化入力stateに対するcompiled座標差を上から評価する。scalar-free理想gadgetと元finite gate列の差は補正角丸め、compile差は全入力POVMの差。参考state数件のallcloseを一様保証へ格上げしない。

実H4のepsilon_phi・gate誤差・effect差・reference leakageは、この限定科学callで初めて取得する。未知値を0にしない。新しい科学的総error allowanceや1/32は採択しない。結果は上界と資源を記録する技術的比較で、正式な測定精度達成や科学的GOではない。精度を満たす実測へ進む場合は上界の採択と他誤差項の契約が別途必要。

## STOP・retry・承認境界

SOURCE／input／policy／gate順序／matrix／path support不一致、未対応gate、非有限値、phaseやinitial/final layout未追跡、reset/readout不一致、informativeな丸め/effect上界が得られない、出力衝突、共有環境非干渉不成立ではSTOP。STOP済みoutputへのresume・seed/path/profile変更・科学retryは禁止。retry0・一campaign。技術的不備が実入力で判明した場合も、その科学結果を保全し、新SOURCEや再実行を自動採択しない。

次に必要な明示許可は、公開固定SOURCEと本契約候補による上記一回のmetadata選定・state-free復元・2 path資源compile・全入力POVM数値資格である。現時点では未許可。state solve、preparation、CDF再計算、保存state作用、Born、sampling/shot、calibration/production、18round、他方式、旧研究再開、重要科学レビューは含まない。

完了・中断のどちらでも結果を専用branchへ公開照合してmandatory STOP。元main、旧raw/SOURCE/結果/STOP/grant/ledger/marker、C1凍結を保全する。
