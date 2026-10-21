# 固定S2候補の理論／compiled費用比較 — 2026-10-22

この検証は、保存済み最良候補を固定した有限時間信号の費用比較である。結果・図・条件別coverageの入口は
[図A/B packet](progress/track_a/figures/pr_pf_s2_recomputed_20261022/README.md)。
全RPE段の最終総費用評価、一般の最適性、PRの最終的な科学的優位性は主張しない。

## 条件と比較範囲

linear H2/H4/H6/H8、隣接1.00 Å、STO-3G、signed tol-only DF 1e-8、rank3/10/19/28、
decomposer generation order、係数cutoff0。T=0.8、複素信号精度ε=0.05/0.01/0.005/0.001/0.0005、
usual signed bias、経験的numerical allowance倍率1、axis failure allocation0.025、状態準備P=0。
cosine/sine両軸のfull Hadamard wrapperをshot数で重み付けした総RZ命令数で比較する。

外側はS2に限定する。旧最良PRおよびFull PF/B0/B0+Iの旧最良S2候補を固定し、理論側で再選択しない。
27種類のcandidate ID、L_D、q、r、Kと利用精度を
`artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22/candidates.json`へ保存した。
δ=T/qであり、各条件の値はfigure_dataに対応するqから確認できる。
全体のδ範囲は0.05–0.8だが、各系・候補は登録されたqだけで評価する。PF係数Cの新しいfitや外挿はない。

native DF、symmetric-directional、original_plus_tail_once、reference_rotated=false。
Qiskit1.3.0、Qiskit Nature0.7.2、rz/sx/x/cx、optimization1、seed17、topologyなし。
理論単価は既存`NATIVE_MIXED / explicit_rz_sx_x_cx_v1`を使用し、compiled単価モデルから推測しない。
一般identity位相でcounterが返す区間は保持し、点期待値へ勝手に置換しない。

## 元cohortと新cohort

H2/H4の再構築済み科学配列とH6元snapshotを回収した。H8は元snapshotが見つからず、
元のSCF/DF/weighted Hermitian projection/state solver方針で新規構築した。
H8のrankは28、Hamiltonian/state hashは元cohortと異なるため、新cohortで両費用を評価する。

新H2実行には失敗した単位があり、それらのgrant/marker/STOPは保全し、再投入していない。
一方、成功済みH2入力・理論counter結果について、元結果のHamiltonian/state/preparation hash、
有限RTE法則のsource bytes、candidate ID、元cosine/sine shot、compiler hash、
各replicaのpaired測定・seed・event digestを照合できた。
元H2のn=8 compiled標本は、その元cohortの結果としてのみ使用する。
新H2 MCが成功した、またはn=8が新cohortの精度目標を満たしたとは表示しない。

H4/H6/H8の新MCは元n=8とのpoolingをしない。新しいshot数と理論・compiled単価は、
同じ入力・preparation・candidate・scalarで会計する。新cohortでの候補最良性は主張しない。
cohort別の図と、cohort境界を表示した一覧図を併記する。

## 信号・shot・回路の技術検証

保存状態のnormとsector liftを照合し、production DF作用を独立occupation CSRと比較した。
各one-body/DF fragmentのspin-factorized作用をCSRの指数作用と比較する。
full-H referenceは直接時間と半時間2回の指数作用、Rayleigh残差と固有位相診断で確認する。
有限RTEのcorrected平均はHornerとforward polynomialで照合し、raw平均とのnormalization closureを検査する。
これらはbinary64の局所数値検証であり、厳密な数値certificateではない。

元providerの経験的allowance方針はsafety_factor10、floor1e-12、roundoff_multiplier64、
maximum_u_over_headroom0.01。factor1を使用し、biasから引いたaxis headroomとnormalizationから
Hoeffding十分条件の整数shotを算出する。PFの決定論biasとshot統計誤差を混同しない。

B0+Iの補正は、保存binary64行列係数から元の有理数算術で求めたfull-Fock Tail identity係数を
元scalarへ一度だけ加える。referenceを回転せず、scalarを二重加算しない。

compile検証では代表paired trajectoryのnative wrapperとcompiled wrapperの状態ベクトルを比較し、
閾値1e-7を維持する。出力状態からの位相合わせやstate fittingをしない。
Qiskit1.3のtemplate scalarと入力global phaseのbookkeepingは、固定gate/angle/wire streamを変えずに
process内で保持する。Aerのset_statevectorが回路初期global phaseを上書きするため、
回路に記録された数値phaseを初期ベクトルへ一度だけ適用する。観測結果から補正値を推定しない。
初回H2の失敗した検証を、後から成功へ変更していない。

## MC設計と希少event

compiled費用は、whole-trajectory最適化に依存する。native counterの有限状態DPをcompiled期待値の代用にはしない。
法則全列挙からcompiled母平均を厳密計算できる同等な方法は、今回確立していない。

新PR候補ごとに独立pilot128 trajectoryを使用し、対象精度のshot重み付き総費用の分散を求める。
本評価nはpilotから、全対象精度について名目95%相対半幅1%を目標として事前固定し、最低128とする。
本評価は別seed集合を使い、pilotを混ぜない。未達時の自動延長・追加sampling・retryはない。

区間はStudent-tを使った名目区間で、希少eventを含む母平均の厳密保証ではない。
全正次数の法則上の確率、trajectory中の発生確率、観測回数、観測費用寄与を保存する。
未観測次数のcompiled費用寄与はUNKNOWNとし、native費用をcompiled母平均の上界と呼ばない。
図Bの差全体をcompiler最適化だけに帰属しない。

## 実行単位と再現性

`cohort × system × candidate × stage`、精度別shot stageではεも含める。
共通入力はsystem単位で一度だけ構築し、その完了結果をhash照合して再利用する。
各単位のplan/grant/marker/result/STOP、SOURCE closure、入力hash、seed規則を保存する。
失敗を別ID・別seedで再投入せず、依存しない単位だけを続行する。

SOURCE初版は`5e3503d8247d9a8788668c04818b665693501292`、
未実行H4/H6/H8用の修正SOURCEは`0cd0adfb947f29d563ce65cd6c86afcad83c7696`。
counterの元実装と旧raw/STOPは変更しない。新しい詳細科学レビュー、merge、PR作成、公開pushは行わない。

実行adapterは`scripts/recomputed_s2_runtime*.py`、再利用libraryは`src/trotterlib/current_s2_theory.py`、
runnerは`scripts/run_recomputed_s2_comparison*.py`と`run_recomputed_s2_candidates.py`、
人工testは`tests/test_recomputed_s2_comparison*.py`。
`scripts/build_recomputed_s2_figures.py`は完成JSONだけを読む作図処理で、科学callを伴わない。
旧grantを使用して科学runnerを再実行してはならない。

大きいcontext.pklは元bytesを削除・変更せず、context.pkl.gzへ可逆圧縮した。
Gitには圧縮版を保存し、`context_compression_manifest.json`に復元後の元SHA-256を記録する。
元snapshot・JSONだけでも物理入力と費用結果を点検できる。localの主担当検証であり、immutable CIや外部再現ではない。

## 追加native物理対応検証と未解決の診断

追加SOURCE `4ec95cd461804cf6513efd83608445f06c0fadb4` のlibraryは
`src/trotterlib/recomputed_s2_physics_validation.py`、runnerは
`scripts/run_recomputed_s2_physics_validation.py`、人工testは
`tests/test_recomputed_s2_physics_validation.py`。完了済み共通入力を再利用した別stageであり、
失敗したMC/compile/input単位の再実行ではない。

全DF basisについて単体Gと実際のHamiltonian項lambda*G^2を保存状態上で検査した。
linear H-chain、1.00 Å、STO-3G、tol-only1e-8、H6 rank19、H8 rank28、generation order、
H6のhalf-step時間0.05/0.4、H8の0.025/0.4、phase alignmentなしの局所binary64検証。
費用候補のL_DはH6で4/7/11、H8で6/7/8/11/15に固定し、Tailを含む全basisを照合した。

補助的なunweighted Gの単体作用は、H6のblock18で最大3.375329763318796e-9、
H8のblock26/28で最大1.3074065209570388e-9となり、固定1e-9閾値を超えた。
一方、実際のHamiltonian項lambda*G^2の検査は、H6で最大1.542929280438198e-13、
H8で最大9.193399603899586e-15で、全項が閾値内だった。
この違いを理由に診断の閾値を緩和したり、失敗単位を再実行・PASSへ変更したりしない。

H6/H8の費用データは同一input/preparationで計数した値として保存するが、
追加診断不合格を注記した条件付き表示とし、科学的検証完了とは数えない。
unweighted-G診断の失敗をどの範囲でRTEの物理誤差へ接続すべきかは未解決であり、
今回の費用図だけで一般の物理精度保証や最終的優位性を確定しない。
詳細はpacketの `physical_validation_audit.json` と各raw result/STOPを参照する。

図のMC区間は、古典的にsampleした回路費用から期待総RZ費用を推定する不確かさであり、量子測定shotの信号精度やQPE/RPEの統計誤差ではない。同一trajectoryのcosine/sine費用をpairedとして重み付けし、その共分散を含む総費用の標本分散を使う。元H2 n=8と新しい本評価はpoolしない。
