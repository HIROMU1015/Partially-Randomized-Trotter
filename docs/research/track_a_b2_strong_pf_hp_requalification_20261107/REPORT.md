# B2 P1強PF：高精度構成による停止5 armの全18round再適格化

2026-11-07、SERVER 2 GPUA100-gleap02。

**停止5 armの全90点が新SOURCEで条件付き信号適格性PASS。**
対象はFull S2 s4/s8、標準Yoshida4 s2/s4/s8。各armでm=0..17を昇順評価し、不適格点は発生しなかった。
旧SOURCEの3 PASS／5 STOPと全raw結果は変更せず、今回の5 PASSを別SOURCEの新結果として保存した。
**branch・shot・価格へ進まず、公開確認後mandatory STOP。**

## SOURCE・入力・範囲

- 実行SOURCE：[`d401d82ab9e77ab43c6eec3483b3086633dcea6d`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/d401d82ab9e77ab43c6eec3483b3086633dcea6d)。人工接続5件PASS後、commit/tree/作業bytesを固定して実行。
- 精度adapter SOURCE：`75cb38e686ad8b82d00d787531d767e816e1331d`。元native/CAR科学SOURCE `a7940b58c6cb939d77e9f8e44d214b3787b7e101`、元8 arm結果 `2583ce4957ed44e914ce6571e3e676830439dfa7`、15点診断 `7dc2113e1325ef87b594b04f2d638fef641de9b7`を保全。
- P1 `7bc952772e451f27c1654dab6803192f2b204c9f`、既存強PF接続 `66a41ad9219f12377e7d325e272c1a90d66ab7e5`。B2の全18round `d7a55ac181de71cd2b2c49644810373cafd0a450`、shot `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`は参照のidentity照合のみ。
- [全結果JSON](../../../artifacts/track_a_b2_strong_pf_hp_requalification/2026-11-07/run01/result.json)、[90点CSV](../../../artifacts/track_a_b2_strong_pf_hp_requalification/2026-11-07/run01/rounds.csv)。新規[runner接続](../../../src/trotterlib/b2_strong_pf_hp_requalification.py)、[実行入口](../../../scripts/run_b2_strong_pf_hp_requalification.py)、[人工テスト](../../../tests/test_b2_strong_pf_hp_requalification.py)。

linear H4、1.00 Å、STO-3G、cutoff前H_phys target、保存DF12、Nα=Nβ=2/sector36、保存元H基底stateを固定。
厳密ground stateを研究上仮定し、状態準備費用を除外。stateをDF/PF固有stateと仮定しない。
Full S2/標準Yoshida4、既存signed factors・one-body・核scalar・factor identity・spin/JW/bit-order・stage順序・負係数・substepsを保持。
δ=.02 Ha^-1、Q=2^m、T=Qδ、m=0..17。新候補探索・再ランキングはしない。

P1はε_E=.00015936001019904 Ha、95%目標、β=.4 rad、事前区間[-128,128] Ha、shift0、exp(-itH)/φ=-δE。
model/PF/RTE/statistical/implementation/reserve=.02/.06/.04/.24/.02/.02 rad、各X/Y α=.05/36、u_z/u_radius/u_normalization=1e-8、u_phase=1e-8 rad、独立照合u/10=1e-9を維持。
決定論RTE disk=0、normalization=attenuation=1。未使用RTE予算.04をPFや統計へ再配分しない。
結果JSONの`P1_conditions`内L_D=3,r=4,K=2は比較元B2の来歴であり、今回のFull PFには全12 factorとone-bodyを用いる。

## 入力gate

既存`load_private_inputs`で元H/state/DF12、sector、spin/bit-order、核scalar、SOURCEを照合。
H_phys S0 hash=`bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2`、
state S0 hash=`1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db`、
DF Hamiltonian hash=`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
private state NPZ SHA256=`deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c`、DF NPZ SHA256=`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。

**one-bodyと全12 factorの13/13でexact Hermitian・exact interleaved spin liftがPASS**。sectorは完全なNα=Nβ=2集合で、元のQiskit index順を保持する。
対称化・spin projection・fallbackは不要で、実施していない。private原本は読取のみで実行後hashも一致。
入力照合の既存operator再構成以外に新SCF/DF/state solveは行わず、z_phys/z_DFの時間発展は再計算しない。

## 接続と数値検証

1. 既存高精度adapterを変更せず、同じbinary64入力値から80/120桁でそれぞれnative/CAR stepを構成。Fraction時間と既存Yoshida係数を再定義しない。
2. arm×precisionの独立10 taskで構成を並列化。stepのIPCはmpmath成分の整数tupleを用い、binary64やdecimalへの一時丸めを入れない。step配列はmemory内に留めて公開しない。
3. 構成が揃ったarmについてm昇順に整数二乗を繰り返し、各roundのnative120/CAR120信号と80桁参照を、同じ保存元stateから算出。初回不適格ならそのarmの後続信号を評価しない設計。
4. operatorと信号の80/120桁差、native/CARの複素差・半径差・円周位相差を**HPのまま**1e-9で検査。
5. 保存z_phys/z_DFと今回PF信号に対してmodel/PF/implementation位相、PF半径下限、総予約位相をHPで検査。その後scalarへの丸め差を検査し、既存`qualify`→`radius_phase_diagnostics`も同じallowanceで実行。両方の判定が成立する場合のみPASS。

model判定=円周位相差(z_DF,z_phys)+u_phase、PF判定=円周位相差(z_PF,z_DF)+u_phase、implementation判定=円周位相差(native,CAR)+u_phase。
PF/raw半径下限=|z_PF|-u_radius。総予約位相は既存PF .06+RTE .04+stat .24にmodel/implementation判定とreserve .02を加える。
旧結果を参照した位相合わせ・unitary projection・再正規化・固有値絶対値置換は行わない。

人工接続**5件PASS（2.67秒）**。値保持IPC、入力不適格の無補正STOP、固定5 arm/初回失格STOP、HP cap/精度検査、人工18roundとJSON接続を確認した。
既存46件の精度検証、3旧PASS arm、B2信号/branch/shotを無意味に再実行していない。

## 5 armの全18round結果

位相はrad。表の値は各armの全18round内の最大/最小で、各点の値はJSON/CSVに保存。

| arm | 新SOURCE判定 | 最大PF位相判定(.06 cap) | 最小PF/raw半径下限 | 最大native/CAR複素差 | 最大80/120 operator entry差 |
|---|---|---:|---:|---:|---:|
| Full_S2_s4 | 18/18 PASS | 0.0008759989261 | 0.999999989999330 | 2.43644e-114 | 7.66582e-74 |
| Full_S2_s8 | 18/18 PASS | 0.0002190070866 | 0.999999989999958 | 4.43324e-114 | 1.52444e-73 |
| Yoshida4_s2 | 18/18 PASS | 2.100751035e-7 | 0.999999990000000 | 3.83722e-114 | 1.14019e-73 |
| Yoshida4_s4 | 18/18 PASS | 2.250856419e-8 | 0.999999990000000 | 7.14679e-114 | 2.27142e-73 |
| Yoshida4_s8 | 18/18 PASS | 1.078551407e-8 | 0.999999990000000 | 1.47251e-113 | 4.53783e-73 |

全90点で独立複素差・半径差・位相差≤1e-9、80/120参照安定、HPと既存判定の両方が成立。
model位相判定の最大は約1.00096147e-8 rad、implementation判定は約1.00000000e-8 rad、総予約位相の最大は約.360000020009615 radで、成分capとβ=.4を満たした。
最小半径下限も正である。これらはP1のengineering allowanceに条件付く数値適格性で、厳密rounding enclosureではない。

## 旧証拠との関係・未評価事項

| 有限baseline集合 | 使用する信号証拠 |
|---|---|
| Full S2 s1/s2、Yoshida4 s1 | 旧SOURCE/結果の全18round PASSを再利用。今回は再計算しない。 |
| Full S2 s4/s8、Yoshida4 s2/s4/s8 | 今回の新HP SOURCEによる全18round PASS。旧5 STOPは履歴として残す。 |

同じ元H/state/DF12・P1条件の**限定8 armについて条件付き信号適格性の証拠が揃った**。
全決定論PFの大域最適や費用順位の主張ではない。B2の既存shot結果も変更していない。

今回のimplementation判定は独立数値経路の差と既存allowanceであり、native emitterの全gate・全support・長compiled回路の実装誤差証明ではない。
branch/energy復元、shot込み95%条件、実機95%成功、価格・総RPE費用・B2への費用優位は未評価/未認証。
実回路build/compile、sampling、価格MC、パラメータ最適化、GPT詳細科学レビューは行わない。

## 実行と次段階

約596.38秒、SERVER 2、CPU affinity64、開始前の利用可能RAM約414GiBを確認。
独立した5 arm×2 precisionの10 taskを使い、恣意的CPU/RAM/BLAS制限・共有設定変更は行っていない。
private state/HDF5/provenance、積分/MO、step matrix・state配列は公開せず、SOURCEとscalar結果のみを公開。

**次の担当**：ユーザーが必要に応じて限定8 armの保存raw=z_PFを用いるbranch/energy復元と条件付きshot会計を承認し、Codexが既存decoder/会計へ接続して実行する。価格測定・全RPE費用比較は、その範囲が別途明示承認されるまで開始しない。
既存SOURCE・raw・grant・marker・STOPは維持する。今回の90点評価と公開確認を終え、**mandatory STOP**。
