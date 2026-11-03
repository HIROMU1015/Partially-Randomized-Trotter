# 研究C：PR対weighted L1・共通r32下流評価の技術準備

2026-11-03 JST。今回のチャット指示は技術準備から固定範囲の一度の科学実行・公開までを許可している。承認証明や旧call残枠は使用しない。

対象は直鎖H4・1.00 Å・STO-3G、通常number CDF8、Nα=Nβ=2・sector36/full256/controlled512。比較IIの結果commit `2ceb0a8825785ebd4f21241d8ef0f20cb321bf69` のPR/ENTRY endpointを変更しない。両者は16更新の有限endpointでJ一次MET/U一次NOT_MET。通常の大域最適解比較ではない。

| 入力 | SHA256 |
|---|---|
| PR endpoint | `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| ENTRY endpoint | `345dd11a14a281bf716f42009565b2d1875fdcf9845b928a15c102164208cb3f` |
| 元h/g/scalar packet | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |

## 再利用と差分

`c_update064_r32_downstream.py`/`c_update064_strong_d.py`（`cca6b487c8e1ed3362f8825e95c937ebbcc87755`）のPF/RTE/RPE・native32・候補固有強Dを再利用する。`c_r32_price_statistics.py`（`3718d369b3db515b2dd13b2cdbb6c49d57880121`）の価格校正・一標本t区間・分布診断を再利用する。公開GitHubの固定tree/報告も確認した。

新しい `c_near_practical_r32.py` はendpointを束縛するadapterである。保存DF norm+guardは同一endpoint/J/U/評価規約を確認して再利用し、各候補のground・law・PF/RTE・shots・native価格・強Dは新しく評価する。元physical Hとsector basisのみ共有し、物理groundを一度だけ計算する。PR/Dは同じ候補の新しいeigenpairを共有する。旧005/064のground・誤差・shots・価格は入力にしない。

既存helperの変更はopt-in candidate label/identity、sampling引数、同じ分布診断の関数抽出に限る。旧runnerの既定条件・617-call定義は保つ。強Dの候補ラベルをshellループ変数から独立させた。ffsim・optimizer・共有venvは変更しない。

2群とも価格が確率変数なので、固定D用の一標本区間を主比較へ流用しない。新しい `c_near_r32_statistics.py` は既存affine会計と校正のwrapperで、2群区間自体はpinned SciPy `ttest_ind(equal_var=False).confidence_interval(.95)` を使用する。PF、shots、native builder、統計分布関数を再実装しない。

## 固定条件と進行規則

PRは元frame順LD2、partial ordered symmetric S2、s1/r32/K0、δ=h=.1、m0..12。CA=.00159 Ha・DF reserve CA/8。own-DF ground oracle/状態準備除外、元g補正κ・nuclear scalar一回・signed atom law・identity位相を維持する。強Dは各同じJ/Uのκ+全8frame、ordered S2/s1、同じδ/round/accuracy政策。

P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false。既存MPFR160 declared phase、relative Gaussian融合、同basis共有、fenced serial会計を両armへ同様に適用する。

1. Byte/metadata/runtime/fresh preflight。ここでは実分子NPZ/QPYをdecodeしない。
2. 保存DF適格性・exact J/U match・GeneralK/CAR9・候補固有ground/alias・law。correctness FAILは全依存処理STOP、alias未資格は該当armの依存処理を省略。
3. 両armのPR13roundを並列評価し、PF/RTE/A/complex signal、3laneのRPE/energyを保存。Evidenceの未知量はnull。
4. 各候補の強DはPRの適格性と独立に自身のPF13roundから判定し、適格ならnative/合成誤差数値pilot/費用を取得。
5. PRの全13roundとideal energyが適格なarmだけnative32。全raw/compiled/reference/actual adjoint/X-Y gateがPASSした場合だけ会計・統計へ進む。
6. 両PRが適格な場合のみ主比較。片側不適格を除外して勝者を認定しない。完了・FAILのいずれでも証拠を保存し公開後mandatory STOP。

Evidence laneは全law synthesis/implemented-law signal誤差が未資格でshots/総費用null。Idealはexact native/ideal finite-RTE IID law/finite-bit sampled-price proxyという反実条件。Assumedは未検証のsynthesis≤.001 A、law signal誤差≤.001 Aという感度仮定。未知誤差を実際に0と認定しない。強Dのfull512 norm-growth合成評価も数値pilotで、厳密rounding enclosureではない。

## 初回価格校正・統計政策

両arm各32 trajectory、各32 K0 event。一つの既存native32 batchを各候補で実行し、粗い初回価格差を評価する。256/640件は転用しない。追加標本・interim decision・外れ値除去・旧cohort混合は0。

Master `2026110304`、cohort `near_practical_PR_r32_fixed32_v1` / `near_practical_ENTRY_r32_fixed32_v1`、axisもarm固有。既存seed関数へmaster/cohort/新SOURCE digest/candidate/round=-1/axis/sample/occurrence=0を渡す。64 seedの衝突を検査する。hash分離はideal IID lawやprice transferの証明ではない。

主指標はideal laneのshell込み期待総RZのPR−ENTRY差。`W × mean(kernel) + S_shell`、Wは13roundのX/Y shots×2^mであり、r32や校正32件を重複乗算しない。主区間は固定N32/armの近似Welch95%差区間。ratioは点値だけで、ランダム分母へ固定D区間を適用しない。

登録精度目標は差区間halfwidth≤ENTRY総RZ点値の5%。単一初回batchの粗いスクリーニング精度であり、結果を見て変更しない。絶対moment skewness>2、delete-one mean shift>各cohortのt-halfwidthの25%、ゼロ観測分散、精度未達はUNRESOLVEDとする。診断PASSは未観測tail/IID/coverageの検証ではない。条件が揃った場合の `NOMINAL_MODEL_PR_LOWER` 等も名目的モデル内の順位に限る。

各armのPR/固定強Dは既存一標本95%区間・5% halfwidth目標による個別の副比較。多重比較での同時優位を主張しない。RZ depth/CX count/CX depth/total depth/sizeは記述的平均・SEのみ。観測min/maxはsupportではない。全law・actual CA・厳密保証・一般性・独立新規性は未資格。

## 科学call上限と実行規模

| 区分 | 共通 | armごと | 合計上限 |
|---|---:|---:|---:|
| 共有setup7 + arm A | 7 | 17 | 41 |
| PR PF/RTE/RPE B | 0 | 90 | 180 |
| native32・会計 C | 0 | 357 | 714 |
| 同じJ/U強D D | 0 | 141 | 282 |
| 価格統計E + 主比較 + 最終summary | 2 | 6 | 14 |
| 合計 | 9 | 611 | **1,231** |

C357=共通native35 +9種×32 +校正summary1 +会計校正1 +serial26 +aggregate3 +uncertainty3。D141は同じ候補の新groundを再利用する既存helper。E6=accounting1+rows1+個別区間2+secondary1+arm summary1。各kindとarmの上限は契約JSONとコードで固定する。内部のprimitive演算数・optimizer iteration数とは異なるlogical unitsであり、旧645/1629の未使用984を転用しない。

最大新eigensolve3、PR26round、D26round、PR complete kernel64本/P0 compile64件、D2本/compile2件、新QPY264件（PR4×64+D4×2）。6件の候補非依存shell QPYは原本のまま再利用・正しさ再確認する。新fit/optimizer/DF norm/実quantum shots/retry/resumeは0。

Fresh outputは `artifacts/research_c_near_practical_r32/2026-11-03/run01`。第一correctness FAILで新callを止め、開始済みthreadを回収する。条件・precision・seed・r・標本数の事後変更やfallbackは行わない。

## 人工検証と資源

[人工・回帰log](../../artifacts/research_c_near_practical_r32/2026-11-03/technical/artificial_tests.txt)：79 PASS。発明したJ/U/Hamiltonianによるend-to-end（成功、片側RPE不適格、alias未資格、correctness FAIL）、SOURCE/input/fresh fault、call上限、64 seed、r32 native plumbing、会計、Welch区間、ゼロ分散、既存native/retuning/statisticsの既定動作を確認した。実H4追加decode/演算を人工テストに含めない。人工PASSは科学的性能の証拠ではない。

Dominant workloadはfull512 raw/compiled/actual-adjoint decode×64 trajectory。旧1候補617-callは約970秒/peak2.4GiBだったが、今回の時間や速度比を保証しない。利用可能CPU/RAM/pressureを実行直前に観測し、既存resource_planを使う。PR/ENTRYの13roundは並列。Qiskitの一時global equivalence/adjoint処理のためcohort同士は順次、各32trajectoryは内部並列、build/compile/adjoint作成は既存mutex、Dの独立4decodeも並列。プロセス内BLAS1でoversubscriptionを避ける。CPU/RAM/worker/wallの追加上限は設けない。task-local cache/tmpだけを使用し、共有環境・他job・旧SOURCE/raw/STOPを変更しない。

実行時のSOURCE digest/contract/input/runtime/preflight・calls・raw・resources・STOPを保存する。終了後の結果は別の実行報告へ収録し、公開照合後停止する。今回の技術準備はformal科学レビューや研究方針の大域GO/NO-GOではない。
