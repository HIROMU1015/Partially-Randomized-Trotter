# H2/H4/H6高次PF PR：新環境の独立確認・固定費用評価結果

2026-10-22 JST。**118 trajectory・236 compile軸の独立確認、固定費用評価、別経路の整数数値照合と終了監査を完了。mandatory STOP。**
3系とも固定されたS2に対する4次PRの費用改善方向を再現した。新しい独立seed namespaceによるlocal検証であり、外部再現・immutable CI・別担当の科学レビューではない。

## 条件と結果の範囲

linear H2/H4/H6、隣接1.0 Å、STO-3G、tol-only DF threshold 1e-8、rank3/10/19（最終rankの指定なし・返却順・signed weights・後段cutoff0）、決定論prefix L_D=1/3/5。T=.8、P=0、K=4、ABS_TIME_PER_OUTER=32、無条件finite-RTE。元のsigned PF係数、scalar-once、symmetric-directional control、状態準備なしfull Hadamard wrapperのcosine/sine対を維持した。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。

outer時間幅δ=T/qはH2 S2/q4=.2、Yoshida4/q2=.4、H4/H6 S2/q8=.1、m5_best/q1=.8。高次公式の各signed pieceは元SOURCEの係数・順序を保持する。この明示した固定3比較・ε=.0005の費用検証であり、全RPE roundの総費用最適化や全候補集合に対する最終科学結論ではない。

費用は `G_i=N_cos C_cos,i+N_sin C_sin,i` の平均（compiled RZ数）。cosine/sineは同一trajectory内の対であり、2n個の独立標本とは数えない。凍結済みshot6行をそのまま使用し、新shot算術は0。量子backendのshot実行やnoise評価は行っていない。

## 平均・不偏分散・SE

次表のSEはランダム回路費用のMonte Carlo標本誤差。PFの決定論的biasや測定shotの統計誤差とは別である。不偏分散と厳密有理数の平均・分散は機械可読JSON/CSVに保存した。

| 系 | 固定候補 | n | N_cos / N_sin | 平均総RZ費用 | SE |
|---|---|---:|---:|---:|---:|
| H2 | S2/q4 | 24 | 117441887 / 106313566 | 8.7674845e+11 | 1.13173e+10 |
| H2 | Yoshida4/q2 | 28 | 79644174 / 78371520 | 4.6896236e+11 | 4.84541e+09 |
| H4 | S2/q8 | 25 | 142588026 / 77606944 | 1.1321025e+13 | 1.22769e+11 |
| H4 | m5_best/q1 | 9 | 72730947 / 72534104 | 5.3085660e+12 | 5.20806e+10 |
| H6 | S2/q8 | 23 | 124124833 / 195407304 | 5.8182509e+13 | 4.61505e+11 |
| H6 | m5_best/q1 | 9 | 76367866 / 76495895 | 2.1454004e+13 | 1.09265e+11 |

## 改善方向の再現と統計的不確かさ

差は常にS2−高次PF。主要3比較family α=.05の両側Bonferroni正規近似（z≈2.39398）を適用した。全3比較でREPRODUCED、近似方向POSITIVE。比の保証区間は計算していない。探索と確認の標本は混合していない。

| 系 | 探索時の削減率 | 今回の削減率（点推定） | 費用差のSE | 差のBonferroni正規近似区間 |
|---|---:|---:|---:|---|
| H2 | 47.12% | **46.51%** | 1.23110e+10 | [3.7831383e+11, 4.3725834e+11] |
| H4 | 53.30% | **53.11%** | 1.33359e+11 | [5.6932006e+12, 6.3317172e+12] |
| H6 | 63.12% | **63.13%** | 4.74263e+11 | [3.5593128e+13, 3.7863882e+13] |

これは有限標本での方向再現と近似区間である。母平均順位の認証、正規性の証明、希少tailの保証にはしない。全6群でS4未観測。H2 Yoshida4群だけS2_singleが1本、他の5群は全てS0であった。未観測層の費用・母分散上界はnullであり、0補完・条件付き抽出・棄却は行っていない。

## 入力回収・再構築・新旧同一性

- H6原snapshotは過去commit f52787a2から回収し、旧SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`と一致。SCF/DF/state再生成0。
- H2/H4は元の固定生成関数37b1cf72/dfef38fcで各1回再生成。全科学配列のhash、Hamiltonian hash、state hash、基底エネルギーが旧保存値と完全一致した。NPZ容器SHAは異なり、旧snapshot原本の復元とは呼ばない。
- H2新snapshot SHA `8bb1aa35cf80266dd4aa1552cc65097770dfe67c690c2e292860923a5b4ae2f9`、H4新snapshot SHA `ed9b5a136e54a47047b6458ebb7ccb6c9b9ef699cdd534ae1f9f18da5310a520`。
- 欠落reference3件を元手法で再構築。欠落H2 signalは固定2候補だけ再計算し、旧bias/log_Bとの差は0。H4/H6は元公開signalとpreparation hashを照合し、新referenceとの整合を確認。凍結shotを継承した。
- private proposal02と約25.65万件の旧seed除外原資料は未回収。元契約を継承した新packを封印した。新846 seedはstage間および利用可能な旧提案・科学call0の初回技術起動seed1692個に非衝突だが、旧全seedとの完全非衝突は未検証・未保証。seed値はprivate packに保全し公開成果物へ載せない。

科学snapshot/referenceの小さな実体と元生成recipeは成果物に含め、入力SHA・provenanceを固定した。新rawの可逆archive708件、private grant/ledger/markerは専用private領域に保全し、branchにはSHA目録と固定118標本の整数RZ表を保存する。

## 科学call・技術監査・保全

- 入力生成は独立確認と別枠：H2/H4各SCF1・積分1・DF1・sector固有値計算1、H2 sector作用4列と独立occupation行列1件、H4 sector作用36列と独立occupation作用36列。reference3・欠落H2 signal2も別grant。shot再計算0。
- 独立確認はH2 52/104軸、H4 34/68軸、H6 32/64軸。trajectory118、piece sampling846、native piece合成846、RTE event17,728、compile236の固定予算と完全一致。TASKを含む取得ledgerは3,108件。
- 初回SOURCE84a301f1の技術起動はworker pickle identity不一致でSTOP。3系とも科学ledger0・trajectory/compile0、worker41全回収。旧grant/marker/STOPを保全し、使用済みgrantを再使用せず、科学call未消費gateを通して別SOURCE・別namespace・別新規packへ切り替えた。取得した118標本の科学retry/resumeは0。
- 成功した取得は各native exit0、全worker exit0/reap。708 archiveの原byte復元SHA・別archive SHA、cos/sin対、event/seed/signed schedule、compiler、全8数値gateと終了監査がPASS。新規専用raw一時fileのみ永続化・復元SHA確認後に解放し、旧rawは変更していない。
- 全6群の平均・不偏分散を非中心化した整数和/二乗和で再構成し、元の中心化有理数計算とexact一致。cos/sin共分散による別式もexact一致。差SE・近似区間・改善率・台帳の別経路照合もPASS。これは一次担当の数値照合であり、別担当の科学レビューではない。

compile前後vectorの最大raw差は3.984792e-10（CT01閾値1e-7未満）、他の検査の最大差は3.009961e-13（1e-9未満）。phase alignmentなし。これらは固定入力状態に対する検査であり、全演算子の誤差証明ではない。

## SOURCEと成果物

- 元基準：`c7c1d8229508253390212588ae2b25e504eb4915`、元科学SOURCE：`8825391ff73b531be5834bf852b9bf2479d1799b`。元SOURCE68ファイルの数値実装を保持。
- H2/H4生成SOURCE：`88cd0907e00151268e3f9b41a350af3efa2f1bb8`。reference復元SOURCE：`eb4a241116ba1a5fcc88e2c9e7c91911a7da63d1`。
- 成功取得・固定評価SOURCE：`c028f1c2885afdba65ef8c62c075738aadd7323d`。独立整数監査SOURCE：`7845c4f9c0b8c5c747d405cd3e1a1c319011f269`。
- frozen provider：`217c971dd537b1cc7fbc13742bc0f6c263ff7f45`。

[結果JSON](../../artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/fixed_evaluation/results.json)、[6群統計CSV](../../artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/fixed_evaluation/group_statistics_6.csv)、[118標本CSV](../../artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/fixed_evaluation/paired_samples_118.csv)、[新契約digest・予算](../../artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/stage_contracts.json)、[科学入力目録](../../artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/scientific_input_inventory.json)、[入力再構築終了監査](../../artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/reconstruction_end_audit.json)、[実行契約・技術履歴](track_a_h246_migration_execution_20261022.md)。

専用branch `h246-reconstructed-confirmation-20261022`へローカルcommitしmandatory STOPする。GitHub push・H4補助比較・候補拡張・標本追加・production・AD03採択・詳細科学レビュー・論文本文作成は実施していない。
