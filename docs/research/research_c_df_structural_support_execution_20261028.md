# 研究C：tail J固定support CDF・限定1 fitの結果

2026-10-28 JST。**1 fit、21/21 logical units COMPLETE、correctness FAIL0、retry/resume0。** 固定P4-tail制約のCDF8は、登録した終了時DF予算を満たした。判定は`DF_ELIGIBLE_NOT_PERFORMANCE_GO`。Tail λは既存PR-aware CDFより大きく、回路価格・shots・総資源の改善は未評価である。

## 対象・固定SOURCE

保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、物理sector36。通常number CDF、outer rank8、元frame順0..7、LD2を維持した。Head frame0,1は一般対称J、tail frame2..7は対角＋edge(0,1),(1,2),(2,3)だけを許す。全Uは可変。元g補正κ・nuclear scalar一度・interleaved spin順を固定し、general-K/raw CARを用いた。

- 科学SOURCE commit：`59e186ad5132ffde66684a781a47a2bd11a77ed4`。実行前後でSOURCE34件無変更。
- 結果branch：`codex/research-c-DF-structural-execution-20261028`。独立worktreeで一度実行。
- [固定契約](research_c_df_structural_support_contract_unapproved_20261028.json) SHA256：`8a0c1c3387d9ced3970c44228699c7bdf76be0f45a79dcee3938a989c965bd66`。
- [SOURCE34](../../artifacts/research_c_df_structural_support/2026-10-28/technical/SOURCE.json) SHA256：`406b5ce36cf626501b8732da8b82c321e60c9f610854c0fa8b085f982f4e7f48`。
- 保存input packet SHA256：`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。元P-H4来歴：`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。
- 保存qualified x0 SHA256：`78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856`。共通scale SHA256：`78b337df677a622876c8bf6b4593e1f338d9de46bc7594e00fa74f8eeb68b1e8`。
- 新候補[最終J/U・raw compact optimizer出力](../../artifacts/research_c_df_structural_support/2026-10-28/run01/fit/final.npz) SHA256：`ec0dfe3d53e8209b343b294a103ab89405ac757d3e26164ae591344d4423b985`。
- [技術調査・人工60件＋29 subtests PASS](research_c_df_structural_design_20261028.md)、[評価preflight](../../artifacts/research_c_df_structural_support/2026-10-28/execution_preflight.json)、[fit preflight](../../artifacts/research_c_df_structural_support/2026-10-28/fit_preflight.json)。

チャット上の明示許可に従い、歴史的`unapproved`という契約名・metadataを変更せず実行した。承認証明・承認原文保存・新grantは作成していない。旧科学runnerや使用済み枠を再開していない。

## 再利用したfitと実行順序

既存`c_pr_aware_df`のtensor loss/JAX解析gradient、ffsim0.0.83のparameter unpackと既存修正codec接続、SciPy1.14.1 L-BFGS-B/options=Noneをそのまま使用した。目的は保存の`f/f* + 0.1(Mε/M*)²`、ε=1e-8。元216 tail係数位置のsmooth和と共通scaleを維持した。

追加済みの固定線形注入`x=Ey`だけで128→110変数へ制限した。Optimizerのraw compact x/jacと、full layoutのx/projected gradientを区別して保存した。初期・終了時のunrestricted gradientも既存hookで記録した。新solver、別初期値、rank/γ/mask探索は実施していない。

保存x0へのmask projection → 初期構造/tensor → 独立sector表現gate → 1 fit → 終了構造/tensor/sector → 対照との記録上の比較、の順に実行した。**初期projectionは開始Hamiltonianを変更する操作**であり、同じ分解を保つcodec変換ではない。登録契約通り初期DF予算は診断で、fitのadmission条件にはしなかった。

## 初期・終了の精度とgate

| 項目 | Projected initial | 終了値 |
|---|---:|---:|
| Tensor Frobenius残差 | 2.1438945633e-1 | 5.0585461233e-9 |
| Tensor max-entry残差 | 4.4448331485e-2 | 9.8400903799e-10 |
| Tensor目的 `.5 ||g−gθ||F²` | 2.2981419493e-2 | 1.2794444441e-17 |
| 物理sector operator 2-norm誤差 [Ha] | 3.9403103925e-1 | 7.4515615285e-9 |
| 登録numerical guard [Ha] | 3.3160388716e-9 | 3.2638787060e-9 |
| 誤差＋guard ≤ CA/8=0.00019875 Ha | NOT_MET（診断） | **PASS** |
| Raw CAR/general-K max-entry差 | 2.6784130469e-15 | 2.5257573810e-15 |
| 同Frobenius差 | 8.3438366149e-15 | 8.2992084270e-15 |
| 表現同値性（atol1e-10/rtol0） | PASS | PASS |
| Exact support/finite/一般対称J | PASS | PASS |

初期・終了ともactual outer rank8。各frameの数値inner rankは全て4（NumPy既定SVDによる診断値、pruningなし）。J上三角の非零数は`[10,10,7,7,7,7,7,7]`、nonidentity Pauli supportは`[36,36,24,24,24,24,24,24]`。禁止entryは最適化全体でexact0、frame/group順は不変である。Tailのactive係数位置は登録した144となったが、K0/r128のevent数を144/216倍に減らせるという意味ではない。

誤差は`sector_operation_scale_guard8_proposal_v1`を伴う**binary64 numerical pilot**。Rigorous enclosure、actual CA、実controlled回路の正しさを認定しない。今回の実分子表現gateはsector36であり、人工full256/full512 testを実回路検証の代用にしていない。

## Optimizerの終了状態

| 記録 | 値 |
|---|---|
| success / status | true / 0 |
| message | `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH` |
| nit / nfev / njev | 639 / 690 / 690 |
| 開始・終了combined objective | 10522805.326720698 → 0.047092990897471446 |
| 開始・終了projected gradient L2 | 220239348.98654518 → 1.5913948449784234 |
| 開始・終了unrestricted gradient L2 | 280934914.8130616 → 1.9967414894353157 |

成功判定は目的関数の相対減少条件であり、gradientが十分小さい停留点や大域最適性の証明ではない。Optimizer successとDF予算PASSを独立に記録し、双方を満たしたためDF段階のみ適格と判定した。追加iteration、再fit、初期値へのfallbackは0。

## 保存対照との比較

以下の通常CDF/旧PR-aware値は保存Stage A resultの引用であり、旧J/Uの再最適化・再評価ではない。分子・rank8・LD2・元のκ/scalar/DF予算は共通。新候補はprojected startのため、同一初期点で構造制約だけを変えた因果的ablationではない。

| 表現 | Exact tail λ [Ha] | 物理sector誤差 [Ha] | Tensor Frobenius | DF予算 |
|---|---:|---:|---:|---|
| 保存通常CDF γ0 | 2.2097618368 | 7.0188550388e-9 | 5.6079586983e-9 | PASS |
| 保存PR-aware CDF γ0.1 | 1.4424298856 | 9.0391618284e-9 | 6.2853010325e-9 | PASS |
| 今回tail-P4 CDF γ0.1 | 1.5161659324 | 7.4515615285e-9 | 5.0585461233e-9 | PASS |

新候補は固定した疎構造でDF予算を満たせた。一方、**λは現PR-awareより大きく、tail mass単独では改善していない。** Tensor/sector残差の小さな差を科学的優位とは扱わない。Gaussian/U・basis分布・head負担・PF/RTE/attenuation・shots・whole-kernel native価格・総資源がどう変わるかは未取得である。

特に、同じ6 tail basisと128 eventsを保つため、少ない係数位置だけから回路削減を主張できない。今回の結果は「登録した制約＋開始点＋solverで物理精度を達成できた」という実行可能性の証拠であり、新規性・性能GOではない。

## Call・資源・保全

| 段階 | Logical units |
|---|---:|
| Prepare：保存初期値・projection・構造・tensor | 4 |
| Fit：入力・単一solver・開始/終了診断・tensor・構造 | 6 |
| Evaluation：対照読込・snapshot2・sector/physical・sector2・表現2・gate・比較 | 11 |
| 合計 attempts / COMPLETE / FAIL | **21 / 21 / 0** |

新fit1、retry0、resume0。PF/RTE・sampling・shots・native build/compile・総費用・rank探索は全て0。[親・子call journalとraw](../../artifacts/research_c_df_structural_support/2026-10-28/run01/)28件/57,650 bytesを保存した。空のconsole logも保持する。

32 affinity CPU、利用可能RAM約52 GiB、低負荷を確認して実行。単一fitの逐次依存とsector36に適合する既存JAXx64/CPU vectorized contraction/ADを使用し、task-local venv/cache以外を変更していない。人工CPU/RAM/worker/wall capなし。[GNU time](../../artifacts/research_c_df_structural_support/2026-10-28/process_time.txt)は全runner＋子processでwall3.79秒、user17.19秒＋system6.40秒、最大RSS377,736 KiB、swap0、exit0。最大RSSはaggregate同時RAM上界ではない。

[Postflight](../../artifacts/research_c_df_structural_support/2026-10-28/execution_postflight.json)でSOURCE34件、保存input14件、旧保護1006件、契約の一致、call内訳、raw hashを照合した。固定SOURCE・入力・旧原本・STOPは変更していない。今回の契約名・metadataも保全した。

## 次の判断とmandatory STOP

**研究C／ユーザー**：DF適格性とλ悪化を踏まえ、係数support減少がwhole-kernel価格やPF/RTEに有用かを次に調べる価値があるか判断する。今回の結果だけでnative資源改善を採択しない。

**Codex（後続の依頼がある場合）**：この保存新J/Uへ固有identityを結合し、第三候補の一般K/atom/law/ground/alias接続と、s1/r128/K0・13 roundのPF/RTE限定契約を準備する。通常CDF γ0の同条件評価も未取得なので、旧Proposed結果で補完しない。Native価格校正は必要性を判断してさらに限定する。新候補自身の同J/U強決定論対照も、PR固有の優位を述べる前に必要。

下流は固定契約の`DESIGN_ONLY`のまま。今回の21-callから自動接続せず、別途固定SOURCE・call・ユーザーの実行許可を要する。重要GPT詳細科学レビューも今回開始していない。

SOURCE・結果・journal・STOPをcommit/pushし、remote SHA/tree・SOURCE・入力・成果物の一致確認後、**mandatory STOP**。
