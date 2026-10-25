# 研究C・C1：H6のDF／PF限定実行・DF予算未達でSTOP

2026-10-25 JST。科学SOURCE `33c380d8aa792e5119bb3adec5dd07d1e1377214`を無変更で使用。H2証拠を含むcheckoutは`cc0cdb52570993fa4f14f6eb199b583d0411d585`。
Branch `codex/research-c-c1-h6-execution-20261025`。

**61/129 logical unitsで登録のDF予算判定に到達し、`DF_BUDGET_NOT_MET`でSTOPした。XDFはrank14で合格、CDFはrequested rank1..21をすべて試みてもCA/8予算に届かなかった。optimizerは21/21成功、技術例外・correctness FAIL0。PFは未実行で、H6のCA適格性・回路費用は取得していない。公開照合後mandatory STOP。**

[実行要約](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6_execution_summary.json)／[全35rank評価](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6_rank_selection.csv)／[4候補の未実行理由](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6_candidate_summary.csv)／[gate状態](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6_gate_summary.csv)／[call表](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6_calls_summary.csv)／[原本・h/g・全fit・STOP](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/)／[固定契約](research_c_c1_h2_h6_contract_unapproved_20261025.json)／[技術準備](research_c_c1_h2_h6_preparation_20261025.md)。

## 固定条件と実行範囲

ユーザーがH6のDF・PF限定実行への進行を許可したため、独立worktree・fresh H6-run01で1回実行した。承認原文保存・承認証明書・新grantは作成していない。旧H2/H4の結果・SOURCE・grant・ledger・marker・STOPは保全し、固定契約の歴史的なunapproved表示を書き換えていない。

直鎖H6、結合距離1.00Å、STO-3G、neutral singlet、Nα=Nβ=3、物理sector400。PySCF2.7.0 RHF(conv_tol1e-12)とMO積分変換で新しい共通h/g/constantを生成し、fit前に保存。gはchemist ERI、元g補正κ・scalar・元frame順を保持した。H2/H4のrank・入力や古いH6のPF係数を転用していない。

通常number CDFはffsim0.0.83、tol1e-8、L-BFGS-B、Cholesky初期化、requested rank1..21のfirst-passingで各1fit。optimizerの停止条件は変更していない。XDFは同じffsimの非最適化modified-Choleskyを上限21で1回実行し、元順prefixを評価する。XDF方式はH4保存OpenFermion-eigh rank10とは異なる。CDF／XDF双方に同じphysical-sector normとDF reserve CA/8を用い、精度を無理に完全一致させない。

CA=0.00159 Ha、DF reserve=0.00019875 Ha。予定PFは全frameの決定論ordered S2、s=1／4、δ=0.1・m=0..12、PRのL_D splitは非適用。own-DF sector ground oracle・状態準備費用除外という参照条件を維持したが、今回ground／PFまで進んでいない。初回H6 scopeはsector DF/PFのみで、native build／compile・合成資格・shots・総費用は対象外である。

## DF予算の結果

| arm／診断対象 | requested／actual rank | sector norm Ha | engineering guard Ha | norm+guard Ha | CA/8 |
|---|---:|---:|---:|---:|---|
| XDF・first passing | 14／14 | 0.00010247385485275451 | 1.1384527781887395e-06 | 0.00010361230763094324 | PASS |
| CDF・登録内の最小観測値 | 18／18 | 0.00046795466223530292 | 1.4487327229419941e-06 | 0.00046940339495824489 | NOT_MET |

XDFのfull exportはrequested21／actual18で、prefix14が最初の合格だった。CDFはrequested1..21の21fitを実行し、すべてoptimizer success=true／status0だったが、どのnorm+guardも予算を満たさなかった。requested18..21はactual18で、export SHAも一致していた。これは新しいfitを呼ばなかったという意味ではなく、21回それぞれを実行・計上した結果である。

CDF rank18のoptimizer記録はnit159／nfev168、objective4.2855875026842845e-8、停止messageは`CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`。optimizerの目的関数値・successは物理sector normの合格を意味せず、独立したDF予算判定を維持した。rank18の値は保存された登録範囲の診断的な最小値であり、不合格のrankを採択したものではない。

guardは`sector_operation_scale_guard8_proposal_v1`、`8 eps64 (n^4+R_actual n^3)(d_sector+1) max(1,||Hphys||F+Σ||Hblock||F)`。厳密rounding enclosureではない。CDFはguardを除いたraw sector normでもCA/8を超えており、guardだけが判定を変えたのではない。物理DF誤差とadapter同値性を混同しない。

このSTOPは登録した十分条件を満たさなかったという結果で、一般的なCDFの不可能性やH6の化学的精度達成不能を示さない。rank・tol・optimizer条件・誤差配分を結果に合わせて変更せず、追加fit／retryも行っていない。

## 到達したgateと未取得事項

| gate | 状態 |
|---|---|
| SOURCE33・契約・環境・fresh input schema | PASS |
| XDF export／prefix14 DF予算 | PASS |
| CDF optimizer | 21/21成功 |
| CDF DF予算 | NOT_MET・STOP |
| selected一般K対raw CARの表現同値性 | NOT_RUN |
| physical／own-DF ground残差・alias | NOT_RUN |
| sector S2 kernels4件・PF52round | NOT_RUN |
| native／合成／shots／総費用 | OUT_OF_SCOPE・未実行 |

契約は両armのDF予算合格をadapter・ground・PFの前提としている。従ってXDFだけのPFへ継続することもせず、4候補すべてを`NOT_RUN_DEPENDENCY_STOP_CDF_DF_BUDGET`と記録した。PF誤差・attenuation・CA・shots・費用を0やPASSで補完していない。PF/RTEの潜在的適格性の分類もまだできない。

raw `STOP.json`の`failed=false`は例外／correctness failureフラグであり、DF予算合格を表さない。科学上の到達状態は`result.json`の`DF_BUDGET_NOT_MET`であることを要約・gate表にも明記した。旧raw STOPを書き換えていない。

## call・負荷・入力来歴

| logical unit | 実績 | 最大 |
|---|---:|---:|
| `scf` | 1 | 1 |
| `spatial_integrals` | 1 | 1 |
| `physical_sector` | 1 | 1 |
| `sector_context` | 1 | 1 |
| `explicit_XDF` | 1 | 1 |
| `CDF_fit` | 21 | 21 |
| `rank_assessment` | 35 | 42 |
| `representation_gate` | 0 | 2 |
| `ground_alias` | 0 | 3 |
| `sector_kernel_reference` | 0 | 4 |
| `PF_round_assessment` | 0 | 52 |

61 attempt／61 COMPLETE／FAIL0。CDF fit・optimizer21、非最適化XDF分解1、rank評価35。retry／resume0。CDFの予算未達に伴い、adapter2・ground3・sector kernel4・PF52と未使用XDF prefix7の計68単位を実行しなかった。H2の科学callとは別に計上している。

疎CARのsector bilinearとcomplete density-frameを使い、物理referenceは既存OpenFermionの疎行列からsector400へ制限してからdensifyした。dense4096／controlled8192、full CAR tensor、実QuantumCircuit／Operatorは作成していない。Gaussianの非保存な基本gateの途中へsector射影を挿入した近似でもない。

開始時affinity32 CPU・空きRAM約51GiB。人工400×400複素行列のSVDを1/4/8/16/32 threadsで確認し、値の一致を確かめた上で、実測に基づきこのjobの初期BLAS/OMPを16とした。独立PF候補4 workers／各BLAS1は既存planだが、PFへ到達しなかったので起動していない。rankはfirst-passing条件のため逐次。wall 35.212540秒、parent CPU 64.418889秒、parent peakRSS 434,712KiB／child peakRSS 434,712KiB。parent CPUはfit subprocessを含まず、journalの21fit wall合計は約26.251秒。未実行PFのtimingやspeedupをこの記録から主張しない。

人工14 tests PASS。SOURCE33と固定契約、eval/fitの各package版をpreflightし、既存専用venvはread-onlyで再利用、venv Python symlinkは保持した。task内cache/tempとprocess-local thread設定以外に環境を変更せず、人工CPU/RAM/wall capや共有設定・他ユーザーjobへの操作はない。

SOURCE manifest SHA256 `380bb20f69c8c0b92d00069a7e2ed33d42f28ff5d1bf500894a536e571cc2a0b`、契約SHA256 `967173d6cb8a2ac695a55e5fd7338776322cef929cb496a33802c33c65517537`。新H6 input SHA256 `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447`、XDF selected export SHA256 `559dc43bf923a421ccc62d05c4cefa0bdf39a48ba68824abf3c8f770c04642b1`、CDF requested18（未採択）export SHA256 `add5fbe58fbf052d6e5e8e86c556241ec1ba9353f14a7ec26b48460bc8d91543`。input/export hashはworker metadataと一致し、全SOURCE・全raw bytesを実行後にも確認した。postflightと各fit metadataに来歴を保存した。報告時はJSONの整形だけで、保存J/U decode・norm／fit／PFの再計算は0。

これらは固定SOURCEのlocal実行を公開した証拠で、CIや外部独立再現ではない。厳密CA・合成error enclosure・状態準備込み資源優位は未資格のままで、今回それらの詳細科学レビューは実施していない。

## 次の担当とmandatory STOP

ユーザー／研究Cが、CDFの初期化・optimizer停止条件と物理DF予算の関係を次に調査するか判断する。Codexは具体的な依頼後、既存SOURCE・保存optimizer metadataを用いた技術調査／人工検証を優先する。新しいfit条件・科学的数値再解析・PF実行は、必要な条件を固定し別途明示許可された場合だけ実施する。今回の未消化callを次の実行へ繰り越さず、XDFだけの後続評価やH6 nativeへ自動進行しない。

結果・証拠をcommit・push・remote SHA／SOURCE／成果物照合後、**mandatory STOP**。
