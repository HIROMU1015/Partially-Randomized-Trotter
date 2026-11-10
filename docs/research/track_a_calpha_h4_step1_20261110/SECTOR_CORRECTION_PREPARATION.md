# C-α／α0-P：sector条件訂正の反映とStep 1残作業の実行準備

2026-11-10 JST。担当：Codex。

**状態：訂正条件の反映・mean-only runnerの準備・非科学的mock確認を完了。訂正後scopeの科学実行許可待ち。**

今回のcomment.md §2に従い、前回の「実行してよい」をSTOP後の科学実行へ引き継いでいない。今回の「comment.mdを確認して」は、同文書が許す静的照合・実装準備・非科学的確認の開始として扱った。新しいoperator、配列再構成、moment、m0/m17信号・上界はまだ評価していない。

## 1. 原資料と訂正内容

`track_a_calpha_sector_condition_review_20261110.md` の追加された原本を確認した。§3–6、§8、§11–12を含め全文を読解。原本のSHA256は **d681afee3c3ed579506f691ab0538d76341b6746509bcc4b52526988264c3108**。

元の `track_a_calpha_theoretical_review_20261110.md` は履歴として不変更。位相・半径定理、arctan補正、Γを含むsplit誤差、head/model/実装誤差の分離を引き続き用いる。変更するのは§1.3から旧runへ持ち込まれた一律の個別primitive保存条件であり、新しい研究方向や新規性のレビューは行っていない。

| 項目 | 旧run01での扱い | 訂正後の準備 |
|---|---|---|
| 個別atomのNα/Nβ保存 | 必須条件として検査しSTOP | 平均信号の必要条件にしない。46件は既知の事実として保全 |
| tail | 元fragmentの保存と個別atom検査 | 固定signed atom和・full辞書mass・identityをmeanへ結合し、解析用Rとの差も評価 |
| head | 旧のprimitive保存条件が問題となった | generic basis変換・対角指数・逆変換からなる完成blockを扱う |
| sector制限 | 条件不成立で未実施 | 保存する完成平均／完成blockの境界で制限。個別trajectoryを射影しない |
| 近似的な非保存 | 未評価 | sector外成分を含むcolumn normの接続誤差へ入れる |
| 実装未資格 | 従来の留保 | UNKNOWNを維持しつつ、条件付き理想lawの数値診断を取得できる構造 |

正式共通運用指示 `research_project_instructions_unified_2026-11-09.md` は、今回確認した環境では未取得のまま。GPTレビュー側の読了記載をCodex自身の読了としていない。提示されたAGENTS.md・今回のcomment.md・追補原本に従う。

## 2. 作業場所と旧成果の保全

同じ専用branch `codex/track-a-calpha-h4-step1-20261110`、worktree `.worktrees/track-a-calpha-h4-step1-20261110` を継続使用。基点は `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`。root checkoutや仮説Iのworktreeへ実装を移していない。branchの新設・統合・削除・履歴書換えはない。

旧runner、旧run01、旧log、旧REPORTはすべて原内容のまま保持した。

- `scripts/run_calpha_h4_step1_applicability.py`
- `artifacts/calpha_h4_step1/2026-11-10/run01/applicability.json`
- `artifacts/calpha_h4_step1/2026-11-10/execution.log`
- `docs/research/track_a_calpha_h4_step1_20261110/REPORT.md`

これら4ファイルを [preserved_run01.tar](../../../artifacts/calpha_h4_step1/2026-11-10/preserved_run01.tar) にも保存し、格納内容が元ファイルと一致することを確認した。archive SHA256は `8cf61d048e3cdd6d2ff94a5c6f75d65a95598787913ca92053f4fb4cc09ce62b`。private stateや積分は含まない。

旧statusは `TECHNICAL_STOP_PRIMITIVE_SPIN_SECTOR_NOT_PRESERVED` のままであり、PASSへ変更していない。46件を誤検出扱いにせず、元の科学SOURCE確認・sector検査も再実行していない。旧実行SOURCEの記録hashとも一致した。

保全archiveは研究SOURCEと結果の保持のためのものであり、承認証明書・grant・approval receiptではない。

## 3. 準備した実装と再利用

新規の実装は [run_calpha_h4_step1_mean.py](../../../scripts/run_calpha_h4_step1_mean.py) 一つ。library側の固定SOURCE・旧runnerは変更しない。新runner SHA256は **05974d2ac2372e360aab0933ffbf4a8b3b562608d31f4f6c52bd726046610436**。

| 再利用元 | 使用する機能・限定 |
|---|---|
| `b2_p1_signal_validation.py` | `load_private_inputs`、`compose_step`、信号の記録・差。旧`run()`やnative correctness buildは呼ばない |
| `b2_p1_remaining_rounds.py` | `checked_prior`、`mean_preparation`：固定K2保存資料・係数projection・λ/identityの結合。旧の16round実行入口は呼ばない |
| `scripts/run_o1_m1_h4_operator_correctness.py` | P1 loaderから取得するCAR・vectorized exterior-minor `fock_lift`。Gaussian回路を生成しない |
| `df_trotter.decompose.diag_hermitian` | 同じgeneric固有基底・sort。spin-awareな別基底へ置換しない |
| `rte.py` | `make_rte_config` と `finite_rte_operator_moments` をK0/r4で使用。RTE polynomialやnormalization式を新実装しない |
| 既存の高精度診断 | `e217d145b…/scripts/diagnose_b2_p1_strong_pf_numerics.py` のexact binary64→mp変換と80/120桁累乗の方法を小さく移植。旧script全体は本基点にないstrong-PF専用driverへ依存するためimportしない |

新しいrunnerが必要な理由は、旧applicability runnerが旧条件でSTOPする専用SOURCEであり、旧P1 runnerがK2固定・native correctness・旧予算へ結び付いているため。これらを無理に改変せず、mean-onlyの接続だけを追加した。

packageの科学moduleは数値入口の内部だけでimportする。runnerのimport、CLI help、mock評価で配列やoperatorを作らない。

## 4. 同じlawから平均・sector参照へ接続する方法

### 4.1 固定辞書のfull-space表現

保存された係数projectionの全signed I/Z/ZZ係数を使用する。各generic frameは既存diagonalizerと同じ構成とし、full Fockのexterior liftで完成させる。signed非identity atom和 `R_dict` はfull空間で構築する。

λ_Rと抽出identityは既存のfull辞書の値を維持する。sector内でmassを計算し直さず、非保存46件や微小係数をdropせず、確率を再最適化しない。保存coefficient projection hashと元のexecutable tail参照を別fieldで保持する。projectionを実行可能な回路preparationだと偽装しない。

物理的な平均は既存RTE関数により `(I−iθR_dict)/B0` をr回合成する。normalizationを戻した補正平均と物理平均を混同せず、B0^-qrを二重適用しない。

### 4.2 保存する解析参照と接続誤差

保存CAR fragmentから同じtailを形成し、数値上のHermitian参照 `R_ref` を明示する。これは固定辞書とは別のsampling法ではなく、追補§8に基づく誤差解析の参照。`R_dict−R_ref` はfull normとsectorからのcolumn normの両方で記録する。和の保存を辞書の一致だと断定しない。

headはgeneric frameの完成blockとしてfull空間で構築する。一方、解析参照は同じCAR fragmentの完成指数を使う。内部Gaussian gateや各Z/ZZを射影した積は作らない。forward→tail→reverse、scalar一回の順序を既存`compose_step`で維持する。reverseを逆時間にはしない。

完成macrostepを使い、次を評価する設計である。

`e_conn = ||S_dict J − J S_ref,S||_2`

左辺は256×36のcolumn matrixで、sector外への成分も残す。`||Π(S_dict−S_ref)Π||`だけで置換しない。追補§8.2のcontraction・理想参照保存の条件の下で、q e_connを接続項として使う。数値matrix normだけで厳密なcontraction／保存を証明したとはしない。

primary raw信号は、微小な再構成上のsector外成分を捨てないfull macrostepの反復二乗から、同じ元stateで取得する。保存する参照モデルの信号をsector36で別に評価し、両者の差も記録する。full-space operator照合は接続確認の範囲であり、別sectorのstate・gap・信号を研究するものではない。

### 4.3 Γ・位相／半径への接続

`R_ref,S`の一回のspectral分解からRθ・Γθを作り、同じstateでγとvΓを取得する。full辞書のλを使い、Γの負固有値を無断でclampしない。stateやtargetを別の基底状態へ置換しない。

動力学誤差の内訳は次の設計である。

| 項 | 比較対象 |
|---|---|
| T δ² C_split | CAR headと、Γを含む参照tailのStrang誤差 |
| T ||Rθ−R_ref|| | arctan補正 |
| q(η_forward+η_reverse) | CAR完成sweepと理想的なCAR head指数との差 |
| T η_model | HermitianなDF modelと明示した元H Hermitian参照との差 |
| q e_conn | generic辞書・完成headから、上のCAR完成macrostepへ接続する差。sector外成分を含む |
| raw参照への表現差 | raw H_physとH_solveの差による比較誤差を別記し、一回だけ含める |

head内部PF誤差とgeneric/CAR再構成差を同じ項として二重計上しない。raw H_phys、H_solve、DF model、保存stateの役割を別に保持し、H_solveへtargetを黙って変更しない。

finite-bit/nativeの包括的実装誤差はUNKNOWNのまま。未知項を0にして包括保証PASSを出さず、まず理想演算子モデルの条件付き式の数値評価を保存する。

## 5. 評価・資格・精度の予定

対象はm0とm17だけである。中間の反復二乗は計算内部処理であり、中間roundの信号や適格性は算出しない。

- primary raw平均信号、保存する解析参照信号、減衰参照信号、既知E0の位相比較参照を二点で記録する。既知Eでbranchを選ばない。
- 保存gapを用いた値はnumerical diagnostic。資格付きgap下界が未取得なら、保証側の基礎はT²/2のgap非依存式とする。
- d≥aならarcsinをclampせず、位相上界はnull、十分条件不成立として残す。
- operator norm等のbinary64値を、丸めまで囲んだ厳密上界へ昇格させない。moment・半径・位相・実装誤差の資格を別々に表示する。
- 既存80/120桁の検査は、保存する36次元参照stepの累乗演算の安定性に限定する。基本step構成やfull dictionary全体を高精度認証したとは扱わない。
- K2のm0/m17保存raw信号を参照し、同じ元H/state/DF/δ/r/prefixを結合する。K2を再simulationしない。
- shot数、全18round branch復号、価格・総費用は今回の出力にしない。

現在これらの数値はすべて未取得。新しい科学結果や科学的GOは出していない。

## 6. 技術確認

構文検査・CLI helpと、[非科学的mock/schema test](../../../tests/test_calpha_h4_step1_mean_contract.py) **8件PASS**を確認した。

1. import時に科学moduleをimportしない。
2. roundがm0/m17の二点以外・欠測・重複・順序違いなら拒否。
3. disk幾何の定義域外でarcsinをclampしない。実装誤差のUNKNOWNを保持。
4. mock callbackから二点だけを受け、JSON保存・mandatory STOPを維持。
5. 既存出力を上書きせず、callback前に拒否。
6. 科学仕様原本が未取得ならcallback前に拒否。
7. callbackの例外を保存し、retryしない。
8. callbackが追加roundを返した場合も拒否・記録。

テストはmockとscalar/schemaの技術確認のみ。private入力、人工行列、科学operator、simulation、旧46件の再計算は使っていない。科学評価の正しさ・位相保証・費用改善のPASSではない。

旧runner/run01/log/REPORTはarchiveともbyte一致し、旧STOPを保全した。新しい科学出力directoryはまだ存在しない。共有venv・site-packages・他ユーザーjobは不変更。commit/pushは未実施。

## 7. 次に求める科学実行範囲

**訂正後scopeの明示的な実行許可を待つ。**

許可を求める対象は、固定generic辞書を維持したH4/DF12/generation LD3/sector36、元H保存state、K0/r4/δ=.02、新C-α systematic .14 radのまま、上記の平均・完成block・参照生成子の接続確認、および未取得のΓ moment・m0/m17平均信号・上界評価だけである。原入力の構築は必要な量を取得するために既存loaderを使い、旧primitive検査を再実行しない。

出力先案は `artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected/`。旧run01は読み取り参照のみ。科学実行時には現SRCと追補原本のhashを結果に記録し、実行中に変更しない。

未実行のコマンド案（専用worktree内）：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/calpha-h4-step1-mpl PYTHONPATH=src \
  '/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python' -u \
  scripts/run_calpha_h4_step1_mean.py \
  --private-dir artifacts/calpha_h4_step1/2026-11-10/input_private \
  --sector-review '/home/abe/Project/Partially Randomized Trotter/track_a_calpha_sector_condition_review_20261110.md' \
  --output artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected
```

小さい8/36/256次元行列には既存BLAS・batched minors・反復二乗・分解の再利用を使う。上のsingle-thread設定はこの小行列workload内のBLAS overheadを避ける選択であり、共有資源の上限設定や速度倍率の主張ではない。科学実行前に利用可能資源を再確認し、必要な範囲で調整する。

平均接続そのものを正当化できない場合はその理由でSTOP。個別primitive非保存という既知の理由では再びSTOPしない。数値上界が弱くても取得できた理想lawの診断は保存し、包括実装保証と区別する。

二点の完了後は良否にかかわらずmandatory STOP。全18round、r/LD変更、別law、spin-aware辞書、sector70、新SCF/DF/ground solve、sampling、native build/compile、価格MC、旧runのretryへ進まない。

**次の担当：ユーザー**が訂正後Step 1残作業の科学実行を明示許可するか判断する。**Codex**は許可後に上記固定範囲だけを実行し、結果を報告する。通常の技術修正ごとに再承認を求めず、承認証明書・ID・grant等は追加しない。

根拠は今回のcomment.md §2：「前回の実行許可…を、STOP後の科学実行の許可へ読み替えない」「許可が未取得の場合は、科学実行の直前に一度だけ確認する」。今回、追加の科学実行許可はまだ受けていない。
