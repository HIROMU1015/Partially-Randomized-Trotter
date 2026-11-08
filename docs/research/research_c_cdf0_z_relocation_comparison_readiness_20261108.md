# 通常CDF＋線形Z再配置＋PR：比較調査・実装準備

2026-11-08 JST。**CASE B：既存データだけでは主比較が不足するが、保存候補・接続実装・人工試験・受動preflightを準備した。新科学実行前でSTOP。**

ENTRYの追加効果は、まだ判定できない。保存された通常CDFのPR費用は「再配置なし」、通常CDFの再配置後費用は「決定論PF」である。この二つを通常CDF＋再配置＋PRの結果として組み合わせることはできない。新fitを先行する必要はない。

## 1. 対象・固定資料・調査範囲

今回の主比較は、A＝通常CDF＋線形Z再配置＋PR、B＝保存ENTRY E2＋同じ再配置＋PR。主指標はshots加重総native RZ、他の5回路指標は別に保持する。独立新規性・論文化・研究継続の重要科学レビューは実施していない。

指定された[最終総括](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eccab64a2672813e479c113c2485219f8479885/docs/research/research_c_final_synthesis_20261107/REPORT.md)と同commitのEVIDENCE_INDEX.csvを確認。索引151レコードのE01〜E23と11月1〜7日の後続、SOURCE・入力・STOPの案内を探索した。索引は独立実験数ではない。今回の静的確認対象は[証拠一覧](../../artifacts/research_c_cdf0_relocated/2026-11-08/technical/evidence_inventory.json)に保存する。

| 固定資料 | Result／閲覧commit | 今回利用した内容 |
|---|---|---|
| 最終総括・E01〜E23索引 | `3eccab64a2672813e479c113c2485219f8479885` | 通常CDF、初期対照、後続結果と資格、過去STOP |
| ENTRY/PR統合 | `417fa459b823509be5c2010c5836e58955088282` | 有限fit、own-ground、selection/final、旧辞書の比較 |
| Z mass診断 | `d601e0ef4375c37185211aa75685ae0e582f3e64` | 36 endpointと別枠中立C0、再配置規約 |
| 再配置PR下流 | `3015d4fb342d4d98b58e41d57553e75539a07655` | P/E1/E2、r grid、独立final64、ideal条件付き費用 |
| 再配置強D | `91417fb3ef75e7eec0849ad39f0649d170754664` | 保存CDF8/XDF8、κ/scalar、full-H、強D価格とSTOP |
| 既知法との重複 | `fd0dd0704ff3172c88c598aab91d63dbe2729c62` | RC-DF/SCDF/UWC/O1の既存整理のみ再利用。外部文献調査を繰り返さない |

`PROJECT_MAP.md`、当初checkoutの研究概要全文、研究目的、対象系列の検証文書、VALIDATION_STATUSと中央manifestを参照した。正式運用規則`research_project_instructions_unified_2026-10-25.md`の§8.1/8.2/8.4/10/10.3を確認。新規grant、承認原文、証明書、詳細レビューを技術作業の前提に追加しない。

最終総括branchには後続moduleが揃っていないため、専用worktree `/tmp/research-c-cdf0-z-preparation-20261108` は再配置・強Dコードが揃う固定`91417fb…`を基点とした。branchは`codex/research-c-cdf0-z-relocation-comparison-preparation-20261108`。元mainのdirty/untracked作業は触らない。閲覧commitと科学SOURCEを区別する。

## 2. 「通常CDF」と保存候補の区別

通常CDFは、一般実対称signed Jとframe別実直交Uを用いるnumber-CDFのtensor-only（γ=0）fit。正則化なしでも非凸の有限fitであり、十分収束した大域最良CDFを意味しない。ENTRY、PR-aware、rank-one的なXDFと同一視しない。

| 候補 | 内容・保存identity | 利用状況 |
|---|---|---|
| **今回のCDF0** | Stage A V2のBaseline γ=0最終CDF8。`baseline/final.npz`、SHA256 `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` | 主候補。再配置なしPRの後続評価あり。新fit不要 |
| 初期研究Cの通常CDF8／強D B2 | 10/25 `fit_export.npz`、SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` | Stage Aの元初期export。10/26強Dおよび11/7再配置強Dの候補。Stage A終端とは別 |
| 中立C0参照 | Stage A `prepare/initial.npz`、SHA256 `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856` | 修正codec後の共通起点。11/6のJ-only mass診断1件。γ=0最終fitではない |
| CDF7 γ=0 | E16〜18の独自rank7初期値・有限fit | 保存DF予算PASS、旧辞書PR/nativeまで実施。rank・初期値が違う参考候補。今回追加実行しない |
| P1/P2 tensor-only restore | 統合実験の摂動start復元。C0とともにPR/ENTRYの初期化用 | γ=0操作を含むが、36件の正則化endpointを通常CDF終端と呼ばない。今回の候補集合へ追加しない |
| XDF8 | signed λ/G、元順prefix8の非最適化XDF | 参考強D。通常一般J-CDFではない |
| ENTRY E2/E1 | C0・64更新、strength .03/.1のCDF8 | 正則化済み。今回の通常CDFへ改名・流用しない |
| PR-aware P | C0・64更新、γ=.1のCDF8 | 正則化済みの参考候補。E01の旧Proposedやupdate005/064とも別 |

初期強D CDF8はffsim0.0.83、L-BFGS-B、nit68/nfev74、相対目的減少停止、tensor Frobenius `6.60902898358172e-5`。保存DF norm＋guardは`7.666215916301914e-5 Ha`。この初期候補の再配置済みD結果を、次節の高精度γ=0終端のPR評価に付け替えない。

## 3. Stage A γ=0終端の入力・精度・収束・再利用資格

以下の保存値は[Stage A原報告](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)の引用。条件は直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2/sector36、number CDF rank8、F0〜F7の元順・LD2。Stage AそのものにPF時間幅/r/Kは適用しない。後続未再配置PRの条件は§4に別記する。

| 項目 | 保存事実 |
|---|---|
| 正式な今回の候補ID | `CDF0_STAGE_A_X0_V2_GAMMA0_FINAL_R8` |
| 科学fit SOURCE | `e1085c95f5c01b68566e17d543ace5832cb94061`、実装`5a185964090f2cb8d19d445d41cf520c4d8ed2c7` |
| 保存場所 | `artifacts/research_c_pr_aware_df/2026-10-27/run01/baseline/` |
| J/U | `final.npz`に各shape(8,4,4)、binary64。x/jac各128も保存。今回はヘッダーとfile bytesのみ確認 |
| 初期値 | 上記共通`prepare/initial.npz`。`baseline/initial_parameters.npz` SHA256 `39a2859b0339e8fd7fbff88468460427be2796ab1538358c9fcab5941ee379df` |
| 終端parameters | `baseline/final_parameters.npz` SHA256 `80e005ecb1d15c8c32858421ff857e36d5e84e3e0c699d9893254f7871243ce1`。元fitの初期/終端保存を維持 |
| 元分子snapshot | P-H4 SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| h/g/constant入力 | `prepare/input.npz` SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。後続ENTRY再配置入力と同じ保存input identity |
| one-body/scalar | 元gから補正されたκを既存保存exportから受け取る。J/Uから再導出しない。nuclear scalarは一回。κとscalarの保存array identityも新proposalに固定 |
| tensor Frobenius／max-entry | `5.607958698344648e-9`／`9.689323562562511e-10`。物理sector誤差と別 |
| sector norm／engineering guard | `7.018855038843621e-9 Ha`／`3.278781198621627e-9 Ha` |
| 保存norm＋guard | `1.0297636237465247e-8 Ha`、DF reserve CA/8=`.00019875 Ha`をPASS。厳密rounding enclosureは未資格 |
| optimizer | ffsim0.0.83の既存L-BFGS-B、options=None、初期lossで正規化。success=true/status0/nit97/nfev=njev107 |
| 停止・gradient | `REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`。終端gradient L2=`3.104515097647238`、∞=`.7423484410886623`。一次/大域収束を認定しない |
| 保存raw／SOURCE／STOP | Stage AのSOURCE、契約、各arm初期/終端parameters、診断、tensor/sector、STOPあり。新proposalで必要ファイルをbyte binding |
| 再利用できるstate/reference | B1の`baseline_reference.npz`にgenerators/H、`ground_baseline.npz`にenergy/sector_state。保存B1契約が同じγ=0終端をbind |

B1 referenceにはJ/U自身がない。今回のadapterは、B1の固定契約による終端J/Uの来歴を確認し、そのJ/Uを旧schemaへ接続する。将来実行時には元9因子の独立CAR照合・scalar込み全H保存・ground/alias gateを省略しない。旧PF/corrected行列や旧shots/価格は新再配置結果に転用しない。

## 4. 既存の再配置＋PR検証の有無

探索対象は指定6commit、最終索引151レコード、通常CDF関連の原報告と対応するSOURCE/保存metadata。全remote branchや非公開rawの不存在を証明する調査ではない。

| 証拠 | γ=0 Stage A終端 | 中立C0起点 | 旧強D CDF8 | ENTRY E2／PR-aware P |
|---|---|---|---|---|
| 線形Z再配置 | 今回の実分子評価なし | J-only診断あり | 11/7 D用に実施済み | 実施済み |
| 再配置tail mass | 未取得 | 保存M_ZZ=`1.124347459961 Ha`（J-only） | Dの保存再配置でtail構造を確認。PR費用とは別 | 各保存値あり |
| scalar込みfull-H同値 | 新CDF0再配置は未評価 | J-onlyなので未評価 | 実施済み | 実施済み |
| 再配置PF/RTE | 未取得 | 未取得 | D PFのみ。PR/RTEなし | PRで実施済み |
| 再配置attenuation/shots | 未取得 | 未取得 | D shotsのみ | PRで実施済み |
| 再配置native価格／総RZ | 未取得 | 未取得 | Dのみ | PRで実施済み |
| 決定論対照 | 初期2 armの同J/U Dなどは元の政策で存在 | 旧初期CDF系列は別来歴 | 旧/再配置Dとも存在 | 同J/U Dと保存CDF/XDF Dあり |

中立C0のM_ZZはH4/1Å/STO-3G/CDF8/元順LD2の11/6 J-only保存診断で、PF δ/r/Kは対象外。これをStage A γ=0終端のM_ZZとして使わない。**今回主候補の再配置M_ZZ自体も新計算していない。**

[第三γ0 DFの既存評価](research_c_df_basis_third_repair_execution_20261029.md)は同じ最終exportを使用。再配置なし、partial S2/s1/δ=h=.1/r128/K0/m0..12、own-DF ground・状態準備除外、同P0の条件で493 calls、native32のfull512 correctness、PF/RTE、RPE、資源まで完了している。Ideal条件付き総RZ=`627,057,959,562.5`、保存λ=`2.2097618367788057 Ha`。これは既存値の引用で、新しい再集計ではない。新再配置費用や、最適rを表す値ではない。旧run01のmetadata STOPと修正run02を両方保全する。

## 5. 科学条件の対応と比較できる範囲

| 条件 | A：CDF0新計算案 | B：保存ENTRY E2 | 参考P／強D |
|---|---|---|---|
| 元物理H | 同じH4直鎖1Å/STO-3Gの保存h/g/c | 同じinput identity `1d6087…` | 同じ元P-H4。各input containerの来歴を維持 |
| DF近似H／rank | γ=0最終CDF8 | ENTRY .03/64更新CDF8 | P=.1/64更新CDF8、強D=CDF8/XDF8 |
| DF accuracy | 保存sector norm＋guardを再利用、共通CA/8 | 保存DF予算PASSを再利用 | 各自の保存値。等しい残差ではなく共通予算 |
| state／energy task | own-DF sector ground、prep除外 | 同左 | 同左。元H exact-ground taskへ無料置換しない |
| 再配置 | 元順LD2、F2〜F7の線形Zを元κへ。frame I/ZZ保持、scalar shift一回 | 同規約 | P同規約。Dは不利なら旧未再配置保存政策を参考にできる |
| PF／時間窓 | ordered partial S2、s1、δ=h=.1、m0..12 | 同左 | P同左。Dは全frame ordered S2/s1/h=.1 |
| r/K機会 | r={8,16,32,64,128}、K0のみ | 同じ5 r/K0からselection32でr16固定 | Pも同機会。今回rやKの追加探索なし |
| RPE条件 | CA=.00159Ha、βtotal=π/6、βstat=π/12、failure=.01、axis=.01/26 | 同左 | 同じ定義を保つ。比較全体の同時RPE成功保証とは別 |
| compiler／cost | P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、phase fitなし、fenced serial | 同左 | 強Dは各法の合法なrelative Gaussian融合を許容 |
| selection/final | 各適格rに32、選択固定後1 pairだけfresh64 | 既存独立final64、再計算・poolなし | P既存final。Dの固定回路を価格標本64とみなさない |

主会計は`W × mean_kernel + shell`。Wにrや校正標本数を重ねて掛けない。主RZ、RZ-depth、CX count/depth、total depth、circuit sizeを別々に保存する。

保存E2/r16の総RZ **75,621,395,471** はこのH4/CDF8/LD2/δ=.1/K0/13round/own-ground/P0/ideal条件付きモデル・final64での既存値。P/r16は**87,514,417,699**。これらと新CDF0を比較する数値再解析は今回していない。現時点でENTRYの通常CDFに対する追加利益を主張しない。

11/7再配置強Dは旧CDF8=`108,561,283,488`、XDF8=`101,780,388,260`。Dでは全frame S2/s1/h=.1/m0..12、同じ元順LD2で再配置している。一方、元の再配置なしD保存費用はCDF8=`102,056,444,569`、XDF8=`100,267,421,030`。今回の新比較にDを強制的に再配置する必要はない。これらを引用するだけで、新しいD順位計算や最適政策の認定はしない。

**限界：これは保存候補の性能比較を準備するもの。** CDF0は一つの有限tensor-only fit、ENTRYは複数strength等から選択された有限endpointで、最適化機会が完全に揃うpenalty単独ablationではない。r/K機会は対応させるが、正則化一般の追加効果や十分最適化した能力差の因果結論には届かない。比較を強めるための新fitを今回の必須作業へ追加しない。

## 6. 再利用と最小差分

| 機能 | 実際のSOURCEで確認したAPI・再利用 |
|---|---|
| CDF目的・ENTRY | `c_pr_aware_df.make_value_and_grad/fit`、`c_near_objective.entry_weights/mass`を静的確認。新fit/新optimizerは実装も実行もしない |
| number-CDF／κ | `c1_general_k_adapter.from_ffsim_number_arrays`、既存保存κ、SectorCARを継承 |
| Z再配置 | `c_h4_relocated_downstream.prepare/prepare_candidate/raw_generators/tail_gate`を無変更で呼ぶ |
| finite-RTE/PF | `c_integrated_retuning_adapter.reference_r`、`c1_stage3_reference`と既存law/normalization/cacheを継承 |
| shots／会計 | `c1_stage3_accuracy`、既存serial cost/calibration/affine helperを継承 |
| native | `native_context`、`c_integrated_retuning_adapter.native_cohort`、既存`StrongBuilder`、spawn/mmap backend、full512/実逆/X-Yを継承 |
| selection／統計 | 既存`select_r`と`c_near_r32_statistics.primary`。1主差、固定N64・nominal95%・登録診断。新統計を今回実行しない |
| STOP | 既存`run_fresh`のexclusive作成・失敗保存・retry/resume0・mandatory STOPを継承 |

新規[SOURCE adapter](../../src/trottertracks/c_cdf0_relocated.py)は、(1) B1 referenceにないJ/Uのschema接続、(2)通常CDFの独立IDと保存契約binding、(3)固定P/E1/E2 runnerを使わない1候補の薄いorchestration、(4)bytes/package/fresh-output preflightだけを追加する。既存の再配置、Hamiltonian生成、solver、RTE、shot式、compilerを作り直していない。

旧engineの`role='baseline'`は型のslotで、候補内容は固定γ=0 exportとそのhash。ENTRYを通常CDFへ偽装していない。統計helperの歴史的`PR`キーは新結果でCDF0側を指すことを外側ラベルへ明記する。保存E2区間をCDF0へ移し替えない。

[CLI](../../scripts/check_research_c_cdf0_relocated.py)は受動preflight専用で`--execute`を持たない。科学関数`execute()`は休止した実装としてのみ準備した。旧3候補runner・handoff・消費済みrunを呼び出さない。

## 7. 人工試験と受動preflight

[新規試験](../../tests/management/test_research_c_cdf0_relocated.py)と既存再配置testの人工部分を実行。CDF8形状のinvented J/U、非零κ/scalar、独立CAR、Hermiticity、全H保存、signed atom/identity/finite-law接続、candidate/SOURCE改変拒否、selection/final分離、無断handoff拒否、例外時STOP・再起動拒否を検査した。mock実行ではfinalを意図的に高値にしてもrを選び直さないことを確認。

さらに4mode人工回路だけで、既存P0の位相込みraw/compiled、実raw/compiled inverse、return、X/Y、leakage検査を再利用した。**33件PASS、実行対象外5件。** 実H4の回路build/compileではない。[JUnit原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/technical/artificial_tests.xml)と[技術検証記録](../../artifacts/research_c_cdf0_relocated/2026-11-08/technical/verification.json)を参照。人工PASSを実H4 correctnessや費用利益へ格上げしない。

既存venvの固定依存のうち`threadpoolctl==3.6.0`が不足していたため、専用`/tmp/research-c-cdf0-preparation-deps-20261108`にだけ追加した。共有venv・package・OS・他processは変更していない。専用環境を移す場合も同じ固定依存を利用する。

```bash
cd /tmp/research-c-cdf0-z-preparation-20261108
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/research-c-cdf0-preparation-deps-20261108:src \
/home/AbeHiromu/projects/partially-randomized-trotter/.venv/bin/python -B \
  scripts/check_research_c_cdf0_relocated.py
```

preflightはSOURCE closure、32保存input、保存DF資格metadata、元B1契約、package metadata、fresh output不在を確認する。NPZ/QPYの数値decode、H4 probe、旧資源の再集計は0。技術準備中の依存不足とSOURCE再固定前のテスト変更は解消後に再確認し、科学runのretryとして扱わない。

## 8. 新科学実行を行う場合の固定した最小範囲

[未実行proposal](research_c_cdf0_relocated_contract_proposal_20261108.json)、[新SOURCE manifest](../../artifacts/research_c_cdf0_relocated/2026-11-08/technical/SOURCE.json)が具体的な引渡し対象。今回は科学実行許可なし。後の会話でこの候補・範囲への明示実行許可を受けた場合、その指示自体が許可であり、証明書やgrant発行を要求しない。

1. CDF0一候補の保存J/U・κ/scalar・旧sector/groundをloadし、元9因子・全H・ground/aliasを再配置後の独立CARと照合する。DF fit/norm再計算は0。
2. 元順LD2で一度再配置。PF kernelを一度、13 exact evolutionsを一度ずつ作り、同候補内だけcacheする。5 rのfinite-RTE law、計65 reference round、既存RPE精度/半径/shotsを評価する。
3. 適格な各rだけnative selection32。最大160 complete kernel標本。全512入出力のraw/compiled/実逆/phase/return/X-Y検査を維持。不適格rはnullで保持し、価格0や敗北にしない。
4. selectionだけでrを固定し、選択1 pairに独立final64。最大新native224、build/compile各224、実逆build448、kernel full512 decode896。shellは保存6 QPYをdecodeして同じ規約を検査する。新ENTRY/P/強D評価は0。
5. 固定したCDF0 final64と既存E2/r16 final64について、主RZ差を既存affine/Welch helperで一度評価する。歴史的E2を再利用する延長比較で、今回両armを新規取得するprospective試験ではない。固定nominal95%、5%半幅/ENTRY、skew2/delete-one .25、追加標本・final再選択・新ratio区間なし。
6. `artifacts/research_c_cdf0_relocated/2026-11-08/science/run01`へexclusive作成するone-shot。retry/resume0。SOURCE/input/compiler/identity/phase等の不一致、例外、全r不適格、登録完了でSTOP。旧runの残枠を使用しない。

CPU/RAM/workerの恣意的capは追加せず、既存resource planner、候補内13round並列、spawn/mmap、immutable cacheを使う。実際の空き資源と非干渉を確認する。新CDF0の実行時間・利益は未測定で、過去ENTRYや強Dの秒数を実測予測として使わない。

## 9. 追加実行で判定できること・できないこと

許可後の最小実行が成功すれば、**この保存γ=0 CDF8を同じ再配置・同じr/K候補機会へ接続した場合、保存ENTRY E2より条件付きshots加重総RZが低いか、高いか、登録精度・分布診断で未解決か**を判定できる。通常CDFでも再配置＋PRの低費用を再現できるかという具体的な対照が得られる。

全rが登録条件で不適格なら、今回gridで費用比較不成立と報告する。新fitの必要性、別K/r/LD/δへの拡張を自動採択しない。入力不一致ならCASE C、保存精度条件不適合ならCASE Dへ切り分ける。

肯定・否定どちらでも、penaltyだけの因果効果、全通常CDFへの一般化、十分収束した最適能力、全law synthesis/implemented-law/ideal-law価格transfer、actual CA、状態準備・noise・hardware/FT、最終総コスト最適化の達成は認定できない。RPE統計誤差とDF/PF/RTEの決定論的誤差も分離する。

## 10. 今回行わなかった科学処理

新DF fit、optimizer、実H4 Hamiltonian/state/DF/PF/RTE評価、保存結果の新しい科学再解析・再ランキング、実H4回路build/compile/native計測、新RPE shots、量子shot抽選、停止科学runのretry/resume、H2/H6/1.15Å、重要科学レビューはすべて0。保存数値は原報告からの引用であり、今回の新科学結果ではない。

## 11. 保全・公開・MANDATORY STOP

旧科学SOURCE・契約・input・raw・結果・grant/ledger/marker/STOPは変更しない。今回のSOURCEは旧SOURCEを書き換えたものではなく、基点commit上の既存codeと新adapterを別namespaceでbindする。旧runの「STOP欠落」は欠落のまま、過去failed STOPは失敗のまま保つ。生成済み巨大rawの全再監査や外部独立再現を主張しない。

索引・研究概要・研究ノート・中央manifestには技術準備として追記し、過去の記録を保持する。commit/push後にremote SHA/tree、公開本文、worktree clean、旧SOURCE保全を確認する。科学outputはまだ存在しない。

**今回の完了判定はCASE B。公開照合後MANDATORY STOP。次の新科学計算・詳細科学レビューへ自動進行しない。**
