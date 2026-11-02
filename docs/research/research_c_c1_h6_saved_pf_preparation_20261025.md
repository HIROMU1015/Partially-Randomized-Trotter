# 研究C・C1：保存H6 rank14の一般K・ground/alias・S2 PF限定検証準備

2026-10-25。起点 `598aa4de622b32e76b0b98451c4efb1fdaac1ae0`。
Branch `codex/research-c-c1-h6-saved-pf-preparation-20261025`。

**技術準備完了。新規21＋既存64 tests、29 subtests PASS。実H6の数値decode・新評価・fit・回路生成は0。保存CDF14／XDF14の4候補×13 roundを、最大72 logical unitsで将来評価する専用経路を用意した。科学実行は別途の明示許可が必要。公開照合後mandatory STOP。**

[SOURCE・人工検証・環境・保全](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/technical/)／[未承認科学契約](research_c_c1_h6_saved_pf_contract_unapproved_20261025.json)／[直前の修正1 fit結果](research_c_c1_h6_corrected_fit_execution_20261025.md)。

## 追加した経路と既存資産の再利用

新 [c1_h6_saved_pf.py](../../src/trottertracks/c1_h6_saved_pf.py)、[専用CLI](../../scripts/run_research_c_c1_h6_saved_pf.py)、[人工test](../../tests/management/test_research_c_c1_h6_saved_pf.py)を追加した。旧 `c1_molecule_extension.run` は化学入力生成とrank探索から始まるため再起動しない。

数値処理には既存 `SectorCAR`、`generators`、`representation_gate`、一般K I/Z/ZZ展開、`PF_rounds`、`reference_kernels`、`Calls`、`resource_plan` を再利用する。新しいfitter、分解法、CAR、PF実装、eigensolverやcompilerは作っていない。ground処理だけは旧runner内のclosureを小さい関数へ抽出し、同じNumPy eigh・residual・base-time alias ruleを使う。

保存CDFは直前の修正1 fitの**返却された最終rank14**、XDFは元の保存selected rank14。初期値へのfallbackやrank変更を行わない。元input、J/U、paired sector row、optimizer metadataをhashで対応付ける。元g補正κ・scalar・元frame順を保持する。CDFの誤差とXDFの誤差は異なり、既存結果を同精度DFと読み替えない。

旧fitで評価済みのCDF最終sector norm／guardと、byte同一なXDF beforeのsector norm／guardを再利用する。新たなDF誤差normは再計算しない。今回の技術準備でも保存J/Uの数値decode・科学的再解析はしていない。

## 将来の限定実行順序

1. byte／JSON／version preflight、空き資源確認、fresh output。CLI defaultはここまで。
2. 元h/g/c、保存CDF14／XDF14、対応する保存DF誤差・guard・CDF optimizer metadataを読み込む。有限値、float64、rank14、Jの対称性、Uの直交性、保存DF予算を確認する。CDFの保存optimizer成功も確認する。
3. sector400のCAR contextを1回作り、各armのbase＋14 ordered framesを構築する。独立一般K展開とraw CARのsector operator同値性を両armで検証する。
4. 元physical Hamiltonianは疎表現からsector400だけを構築し、physical／CDF／XDFのground・aliasを確認する。元physicalはground/alias用であり、新しいDF誤差評価には使わない。DF固有分解を全PF roundのexact evolutionに再利用する。
5. CDF:s1／CDF:s4／XDF:s1／XDF:s4の順で4候補を登録し、独立候補を資源に応じて並列実行する。既存S2 kernel、反復二乗法、同じ経過時間のexact DF evolutionに対するsector spectral operator誤差を13 roundずつ評価する。
6. 各セルを完了時に保存し、4候補を登録十分条件で分類する。全結果・原本・call・資源・STOPを保存・公開しmandatory STOP。

将来の証拠は52セルJSON、候補4 summaries、一般K同値性2件、ground/alias3件、再利用可能なsector spectra／ground statesとPF kernels4件になる。全Fock4096／controlled8192の密行列、回路・compileは含まない。

## 固定する条件・判定の意味

対象は直鎖H6・1.00 Å・STO-3G、中性singlet、Nα=Nβ=3／sector400、両arm rank14、通常number・元g補正κ・scalar・元frame順。PFは全frameのordered symmetric S2、s=1,4、δ=0.1、m=0..12。PR splitは非適用。own-DF ground oracleを仮定し、状態準備は費用対象外。

CA=0.00159 Ha、DF reserve=CA/8=0.00019875 Haと既存`sector_operation_scale_guard8_proposal_v1`を維持する。一般K gateはmax-entry1e-10／rtol0、ground residualは1e-10、base aliasは `abs(E0*0.1)+pi/6 < pi`。既存PF十分条件は `PF operator error <= sin(pi/12)`。これは旧beta_total=pi/6／beta_stat=pi/12の既知部分のroomに対応する。統計failure配分やshotsは今回評価・変更しない。

- 全13 roundのPF十分条件を満たせば `POTENTIALLY_FEASIBLE`。DF reserveとPFの既知部分だけの数値条件であり、CA達成ではない。
- どれかを満たさなければ `INFEASIBLE_REGISTERED_PHASE_ROOM`。登録した十分条件に対する不適格で、物理的な不可能性ではない。他の登録セルも継続する。
- 合成誤差・厳密reference enclosure・最終CAは未資格のまま。未知誤差を0と置かない。決定論PFなのでattenuation=1／RTE discrepancy=0は構造上の値で、未知合成誤差の0認定とは異なる。
- SOURCE／input／schema／有限値／一般K同値性／ground・aliasのFAILでは依存する後続callを停止し、開始済み作業だけをdrainして部分証拠を残す。retry/resume、追加fit、条件変更は0。

## 科学callとSOURCE

| 処理 | 上限 |
|---|---:|
| 保存input/J/U/metadata load | 1 |
| 保存DF schema・予算gate | 2 |
| sector CAR context | 1 |
| raw sector generators | 2 |
| 一般K同値性 | 2 |
| original physical sector | 1 |
| ground/alias（physical/CDF/XDF） | 3 |
| S2 kernel reference | 4 |
| PF round誤差（4×13） | 52 |
| 候補summary | 4 |
| **合計** | **72** |

契約scope：`C1_H6_SAVED_RANK14_SECTOR_PF_V1`。
SOURCE manifest SHA256：`53d5dce52cf7843e518361f9ed753a00dda91b5bab84213ae111fde0bed94969`。
契約SHA256：`d37088d4012a7b9f99570386a2e13fcc1e9eb6e770a339be29300564cd16afcd`。

SOURCEは適合する旧H2/H6の33 filesを保全して新規3 filesを追加した36 files。保存入力・fit来歴12 filesと既存evaluation lockを固定した。ffsim／fit環境を起動する必要はない。旧grant・ledger・marker・STOPを変更せず、承認証明書・記録・新grantも要求しない。将来のfresh outputは `artifacts/research_c_c1_h6_saved_pf/2026-10-25/run01`。

## 人工検証と実行可能規模

| 試験 | 結果 |
|---|---:|
| 新：人工保存入力end-to-end、全52セル、独立expm照合、予算／一般K／ground／PF FAIL時STOP、fresh output、不正入力、passive preflight | 21 PASS |
| 旧H2/H6＋修正fit親の人工回帰 | 26 PASS |
| 旧一般K回帰（独立process） | 13 tests＋29 subtests PASS |
| 旧PF／reference／精度接続回帰（独立process） | 25 PASS |
| **合計** | **85 tests＋29 subtests PASS** |

SOURCE36／input12／依存環境のpreflightは、`np.load`を遮断してもPASS。[人工・JUnit・preflight原本](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/technical/)。旧テスト中の回路／化学処理は人工・mockであり、実分子の新科学callは0。今回のテスト成功はH6 PFの適格性を示さない。

技術試験開始時は32 CPU、idle約99.2%、利用可能RAM約52.7 GiB。小さい人工行列とworker間の過剰並列を避けるためtest processだけBLAS/OMP1とし、共有設定は変更しなかった。将来は4独立PF候補を既存resource_planで並列化し、固有分解を再利用、長い反復は二乗法を使う。1個の400×400 complex128 matrixは2.56 MBで、各arm15 generatorsは約38.4 MB。実際の一時領域と4 workers分の総RAMは実行前に確認する。恣意的なCPU/RAM/worker/wall capは設けない。

残る実装blockerはない。H6の一般K gate、ground/alias、PF52セルの数値結果は未取得であり、新たに計算するには別途明示的な科学実行許可が必要。最終CA・native費用・shots・総資源比較はその後も別範囲になる。

**次はユーザー／研究Cがこの固定SOURCE・契約の72-call限定科学実行を許可するか判断する。許可後にCodexがfresh run01で登録範囲だけ実行する。今回は公開照合後mandatory STOP。**
