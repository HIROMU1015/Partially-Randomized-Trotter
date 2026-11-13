# H4 DF10：DF head＋Pauli tailのP0結果

承認されたLD1/LD2のoperator・辞書検証を、科学SOURCE **fd111f989b9cd2f052d7189e4b7b6fab10aa1a9f**でfresh run01に一回実行した。**81個の数値gateがすべて通過**した。両prefixで、同じ保存DF tailをPauli和で数値的に再構成でき、head/合計tailはsectorを保存する一方、個別Pauli labelの多くはsector外へ遷移することを確認した。

[result.json](run01/result.json)／[prefix要約CSV](PREFIX_SUMMARY.csv)／[HANDOFF](HANDOFF.md)／[固定数値経路](IMPLEMENTATION.md)。実行計画は75fb0ba1の[SCIENCE_PLAN P0](../df_head_pauli_tail_preparation_20261114/SCIENCE_PLAN.md)。P1・mean・N・価格の実行には進んでいない。

## 固定物理条件・再利用

Linear H4、隣接1.00 Å、STO-3G、保存tol-only1e−8 DF10、8mode/full256、sector(2,2)/36。保存weighted-Hermitian completion、signed weights、fragment順、state/snapshot、target=保存DF-Hを変更していない。S2/ΔPF=.1/K2、h=.1/M12、元exactκ/prior/decoder/誤差予算は元configurationのmetadataとして保持し、今回は時間発展やscheduleを新生成していない。P=0。

- H hash：`c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`
- state hash：`9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`
- snapshot：`12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`
- 元scalar c：`2.29310124732`。headはone-bodyと先頭LD個のweighted DF block、tailは残るweighted blockの和。tailのidentityをs_Rとして外へ出す規則を維持し、head自身のidentityを再抽出していない。

保存入力は155529e2のsave-only loaderを再利用し、receipt/配列/H/state hash、basis/liftを検証した。現AのCAR one-body作用と`h6_df19_rpe_d2.make_context`を同じ演算順でfull256へ拡張し、同じbuilderを保存sector36に適用したblockと照合した。旧meanや保存signalの再解析ではない。M1のone-body FermionOperator生成をprefix後のfactorへ限定し、`Σ weight*(a*a)`→JW→既存Pauli collectorで辞書を作った。裸の元分子H−headや旧LD0の演算子で代用していない。新しいHermitian化・固有値計算はない。

Qiskit labelは右端がqubit0、保存CAR basisはmode0がMSBである。bit reversalで行列の両軸を明示変換し、辞書にQiskit labelと保存tensor順labelを併記した。元のH/state/basisを変更して一致させていない。

## 辞書・λ・scalar

| prefix | 非identity Pauli項 | 保存DF λ_R | Pauli λ_P=Σabs(c_j) | Pauli identity s_P | 保存s_R | s_P−s_R |
|---|---:|---:|---:|---:|---:|---:|
| LD1 | 184 | 2.8629570945298686 | 3.3066052942233157 | 0.4177175120626943 | 0.4177175120626939 | +3.885780586188048e−16 |
| LD2 | 184 | 1.6030956277929231 | 2.0550998769244275 | 0.22237590941070173 | 0.2223759094107018 | −5.551115123125783e−17 |

[LD1辞書](run01/LD1_DICTIONARY.json)／[LD2辞書](run01/LD2_DICTIONARY.json)に全label・実係数・binary64 exact hex・符号・p_j・sector外の全basis遷移を保存した。p_jのfsumは両方1。λは各辞書の非identity係数に対する値で、元DF λの置換を本番採択したものではない。λ_Pが大きいこと自体は正常な結果で、計算FAILや費用劣位の判定にはしない。

| 辞書 | JSON payload SHA256（canonical candidate IDではない） |
|---|---|
| LD1 | `dbdde5a12791cd7b31b00d8ec6ae5074d56964b467389688163ece8bf186945c` |
| LD2 | `dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7` |

## operator・sector・stateの照合

以下はFrobenius残差、既定gateは1e−9。最大の保存gate値は7.110182658015774e−15だった。

| 検査 | LD1 | LD2 |
|---|---:|---:|
| full H−(cI+head+tail) | 6.364736188629584e−15 | 5.8053725919263836e−15 |
| Pauli非identity和+s_P I−DF tail/full256 | 7.110182658015774e−15 | 2.2506848936162844e−15 |
| 保存s_R使用時の同残差/full256 | 1.8376935306624233e−15 | 1.5140156440522419e−15 |
| Pauli tail対DF tail/sector36 | 2.7640121731740246e−15 | 8.316341585463472e−16 |
| 合計DF tailのsector外offblock | 0 | 0 |
| 合計Pauli非identity和のsector外offblock | 2.7249518739372675e−17 | 1.9128718763261737e−17 |
| 個別labelでsector外に出るもの | 136/184 | 136/184 |

one-body＋10weighted blockはすべてHermitianで、双方向のsector offblockは0。full256から36へのblock射影残差は最大1.3340012251170559e−15。両headのHermiticityとsector offblockも0。stateのfull/sector norm誤差とlift残差は0で、既定1e−12 gateを通過した。元3入力ファイルは実行後も同じhashである。

個別labelの分類は保存36basisからのbit像を全数調べた結果であり、event列やtrajectoryを生成したものではない。labelがsector外へ写すbasisがある場合、一般の非自明Pauli回転も漏れ得る。回転角が特殊な場合や複数eventの相殺は別であり、「全eventが必ず漏れる」という主張ではない。個々の射影Pauliを36次元involution/unitaryとして使うことはできない。

この結果は、**合計tail/headのideal平均を36で扱うためのoperator側の条件**を支持する。Pauli finite-K meanの本番受入、full/reduced mean一致、actual sampler/control回路、全roundのRMSEは今回検証していない。

## 数値資格と実行範囲

raw JW係数の虚部max/L1は両LDで0。追加のadapter cutoffは0、既存collectorの虚部許容1e−12、OpenFermion1.6.1の内部簡約/EQ_TOLERANCE=1e−8はそのままである。raw JW係数とfull operator残差を保存しており、上の残差はその実装を含むbinary64数値診断である。tiny termの厳密保持やrigorousな演算子同一性証明を主張しない。saved s_Rをs_Pへ上書きしていない。

Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、実行時間3.4756秒。新人工5＋既存契約4の計9件がPASS。人工値の2e−14を実分子gateへ移していない。[TESTS](TESTS.md)。

保存入力1組、full-Fock extension1組、Pauli辞書2件。**mean/reference/eigh/finite-time normalizer/sampling/build/compile/RPE/N/RMSE/C/Gはすべて0**。K/κ・H/state・数値gate・既存SOURCE/raw/結果/KEEP/STOPを変更せず、自動retryは0。

## 次の最小範囲とSTOP

ユーザーが別途承認する場合、元SCIENCE_PLANのP1（二prefix×m0/m6のPauli mean、必要時のLD1/m0 fixed-Rと同候補full256対照、最大6 mean/18complex・unique normalizer最大5・新reference0）を具体化できる。B管理の辞書付きcanonical identityとPauli λ/scalar/law受入を先に固定する。旧DF CID・raw/N・価格をPauli結果へ再ラベルしない。P0だけでP1や全13roundへ進まない。

- ユーザー：P1を行うか、対象prefixと有限実行範囲を判断・承認する。
- Codex A：承認後だけ既存providerへsource-bound Pauli辞書を接続し、限定meanとfull/36照合を実施する。
- Codex B/C/D：別の担当範囲でcanonical/risk受入、辞書と同lawの構造/価格接続を扱う。今回他担当のbranch/worktree/processは変更していない。

成果のcommit/push・remote SHA/tree/主要ファイル照合後、**mandatory STOP**。十分N/global RMSE/C/G/winnerは未取得/nullのままである。
