# DF HamiltonianのFull PF・Truncated PF・改良PR比較：H2/H4/H6

2026-11-08。Track Cの新しい独立研究。基点は`c5a3c102199fe97f4d5c21837c059ae2684f5ef2`。
旧研究・原本・STOPは変更しない。ユーザーのH2/H4/H6拡張指示の範囲で計算・実装・再検証を実施した。

## 比較結果

| 分子 | 最良Full PF（総RZ） | 最良Truncated PF（総RZ） | 最良改良PR（総RZ ± MC SE） | 低コスト方式 | 科学的資格 |
|---|---:|---:|---:|---|---|
| H2 | 8.277772e+09 | 7.851935e+09 | 5.009546e+09 ± 7.327e+05 | 改良PR | 固定元H・oracle ground・理想native・独立draws下で全18round資格あり |
| H4 | 5.647934e+11 | 3.337772e+11 | 1.303899e+12 ± 4.834e+09 | Truncated PF | 固定元H・oracle ground・理想native・独立draws下で全18round資格あり |
| H6 | 2.082026e+12 | 1.222907e+12 | 1.617754e+12 ± 2.001e+09 | Truncated PF | 固定元H・oracle ground・理想native・独立draws下で全18round資格あり |

「最良」は後述する有限候補集合・round別policyの中の値であり、大域最適の証明ではない。PRの値は独立holdoutによる期待価格である。科学的資格は固定保存Hamiltonian、元Hの厳密ground入力、理想native gate、独立な測定と256-bit drawsという条件付きである。

| 分子 | 方式 | 総RZ | 総CX | 総size | serial depth上界 | 十分trial数 | RZ MC SE |
|---|---|---:|---:|---:|---:|---:|---:|
| H2 | Full | 8.277771739e+09 | 2.176712270e+09 | 1.249852239e+10 | 6.349359243e+09 | 14,208 | 0.000e+00 |
| H2 | Truncated | 7.851934551e+09 | 1.836036870e+09 | 1.173200980e+10 | 5.767372489e+09 | 14,207 | 0.000e+00 |
| H2 | PR | 5.009545544e+09 | 1.714095870e+09 | 7.737398221e+09 | 4.179183735e+09 | 15,896 | 7.327e+05 |
| H4 | Full | 5.647934021e+11 | 1.012313194e+11 | 8.371684838e+11 | 2.088520659e+11 | 14,225 | 0.000e+00 |
| H4 | Truncated | 3.337772238e+11 | 5.830379151e+10 | 4.941682283e+11 | 1.188928069e+11 | 14,629 | 0.000e+00 |
| H4 | PR | 1.303898609e+12 | 1.824266292e+11 | 1.904338949e+12 | 4.992174341e+11 | 16,791 | 4.834e+09 |
| H6 | Full | 2.082026044e+12 | 3.657833912e+11 | 3.085775668e+12 | 5.917635392e+11 | 14,248 | 0.000e+00 |
| H6 | Truncated | 1.222907256e+12 | 2.106713327e+11 | 1.809978048e+12 | 3.347103693e+11 | 13,539 | 0.000e+00 |
| H6 | PR | 1.617753939e+12 | 2.708191202e+11 | 2.389309688e+12 | 4.422261561e+11 | 15,754 | 2.001e+09 |

全値は状態準備を除く18round interrogation費用。RZ/CX/sizeは実際に選んだ同じnative moduleの値であり、depthはそのserial連結上界。決定論的価格のMC SE=0はゲート数が確定する意味であり、物理・化学モデルの不確かさが0という意味ではない。

## 同じ推定課題と入力

各分子はlinear、隣接距離1.00 Å、STO-3G、RHF積分に由来する保存Hamiltonianを用いる。分子ごとに元Hが異なるため、分子間の絶対価格と同じ分子内の方式比較を区別する。

| 分子 | 固定DF・順序 | 信号計算sector | system + control qubits | 元H・状態 |
|---|---|---|---|---|
| H2 | 保存full XDF3、元frame順、one-body block別枠 | Nα=Nβ=1、4次元 | 4+1 | 保存積分から元Hを疎CARで構成、元Hのgroundを計算 |
| H4 | 保存DF12、generation order、PR LD3/218 signed components | N=4、70次元 | 8+1 | c5a3のpublic replayと元H groundを再利用 |
| H6 | 保存full XDF18、元frame順、one-body block別枠 | Nα=Nβ=3、400次元 | 12+1 | 保存積分から元Hを疎CARで構成、元Hのgroundを計算 |

H2/H6の原入力は`5385c6ef01879ee8a70734bfc0d5ae8999dd6186`の`research_c_c1_h2_h6_science/2026-10-25/H{2,6}-run01/{input,XDF_full}.npz`。新規SCF・DF fitは0件。H2の旧groundデータは存在するが、今回の元H residual certificateのため小さいsectorのeigensystemを新規計算した。H6の旧rank14 own-DF groundは元H groundに転用していない。

原行列の微小な反Hermitian丸め残差は保存したまま、最終対象をそのHermitian部分と定義する。H4既存保証と同じ規約である。H2/H4/H6のrawからのFrobenius差は約2.36e-16/1.55e-15/1.68e-14 Ha。DF再構成と元Hの数値operator差は約6.79e-16/5.67e-14/1.73e-9 Ha。これらの数値観測だけを成功証明にせず、元Hに対する信号を直接囲んだ。連続積分や未保存の実数Hamiltonianへの無条件包含は主張しない。

全方式・全分子で18round、δ0=.02、q=2^m（m=0..17）、energy tolerance=1.5936e-4 Ha、1回の方式実行の失敗上限=.05、位相予算=.4 rad。地平線からの十分energy boundは.4/(.02×2^17)=1.52587890625e-4 Haであり、精度を緩めていない。全9行が同時に成功する確率の.05保証ではない。

元Hの厳密groundを利用可能とするpromiseと状態準備費用除外を共通に適用した。保存stateと厳密groundの距離・normalization、ground energy区間をresidual/gapで囲む。E0は科学的資格・誤差評価にのみ使用し、decoderにはphase列とq列だけを渡した。既知E0によるenergy補正は行わない。

## 既存研究の再利用と重複確認

[DISCOVERY](DISCOVERY.md)に参照SOURCE、B0/B0I raw、版・条件差、再利用関数、他branchの確認範囲をまとめた。Pauli-JW B0/B0Iからはnested候補探索、constant保持、元H stateの直接信号評価、bias・十分trial・価格の同時評価という方法を再利用した。旧Pauli価格やT=.8の数値はDFの結果に用いない。

同一条件の3方式比較は、確認したGit refs/worktree metadata・公開branchから取得できなかった。非公開の他チャットや別サーバーの全実行状態を取得したことにはしない。旧研究CのCDF14/LD4、XDF8/9、旧H6のCA/13round/別compiler資源値を流用しない。旧Track Bや旧STOP研究は再開していない。

H4は最新c5a3の強Fullと最適化Closed P3を、科学条件・compiler・元H state・α/N・既存native/source保証・独立価格holdoutの対応を確認して再利用する。新しいFull探索が高ければ、既存の安いwhole policyを比較に残す。

## 切り捨て候補・探索範囲

保存DF演算子H_DF=cI+Σ_l B_l diag(d_l) B_l†からHkeepを構成し、元Hとの差Hdrop=H−Hkeepを定義する。Hdropには上記の微小なDF再構成差も含める。保存binary64 operandsからI/Z/ZZ係数をexact rationalで形成し、Hermitianな実係数の部分集合だけを除く。basisや保存DFを再fitしない。

1. **DF prefix**：one-body blockを保持し、元generation orderの最初k個のDF fragmentを保持する。k=0..L−1を全て試す。末尾を丸ごと除くI0と、各fragmentのidentityだけは残すI1を別候補とする。
2. **小係数項**：全blockの非identity係数をabs係数、block index、supportで並べ、小さい方から累積L1 budget以下を除く。budget/energy toleranceは[.01,.03,.1,.25,.5,1,2,4,8,16,64]。同一drop集合を重複評価しない。cと各identity係数を保持する。
3. **Full**：drop=0を基準として必ず含める。I1のscalarは既存係数であり、E0から作る補正ではない。

| 分子 | 重複除去後の候補数（Full含む） | stand-alone exact Hkeepの十分pilot screen内（Full含む） |
|---|---:|---:|
| H2 | 9 | 2 |
| H4 | 36 | 19 |
| H6 | 48 | 20 |

全93候補の保持数、drop定義、数値ノルム、ground差、dominant overlap、最小信号半径、screen理由、directed資格を得たround、最終採択roundは[CANDIDATES.csv](../../../artifacts/track_c_df_truncation/2026-11-08/ordering/CANDIDATES.csv)。実際のdrop supportとexact係数はrawのMENU/INPUTにある。

pilotでは元H stateでHkeepの全18round信号を計算し、固有モードのoverlapと非共鳴の位相十分上界を調べた。screen外は今回の有限探索から除外した結果であり、全PFとの相殺を含む不可能性の証明ではない。単に||Hdrop||が大きいだけで不可能とは判定しない。

screen内の候補へS2/Yoshida4を等しく与え、h=[.0025,.005,.01,.02,.04,.08,.16,.32,.64,1.28,2.56]で902セルを評価した。各roundの上位候補を実価格・directed保証へ接続した。初期の216 deterministic round点は全て最終条件でclosed。

さらに旧H4強Fullが作用順を最適化していたことに合わせ、全方式の凍結候補をanchorとし、m15..17について先頭最大4因子の全順列、同じ既存PF family、隣接h/2,h,2hを一度だけ探索した。H4 PRは既にc5a3で24順列等を最適化済みのため保全。1,188 pilotセルから分子・方式・round別上位2点（48点）を選び、**proxyを価格や資格に使わず**実native価格・directed保証を実施した。48/48 closed。元whole policyも候補に残した。後半のretention/split以外の無制限部分集合・新PF familyの探索は行わない。

「最良Truncated PF」はround別retention/h/formula/orderを持つpolicyであり、18round全てを一つのHkeepで発展させるという制限は課していない。全roundで同じ元Hを推定対象とし、decoderの共通branch条件を検証した。

## PRの分子別接続

H4のD内部BCH係数、LD3、218成分、625件catalogue、h/resolution/α、価格をH2/H6へコピーしない。H2はLD0/1/2、H6はLD3/9/14/17の有限menuを作った。identity tailを正確なcontrol scalarへ分離し、λとsigned component確率を各分子のexact係数から定義する。

同じ既存D ordered/D4と、分子・split固有のBCH rootが[-2,2]に存在する場合だけDtunedを評価した。H2 LD0/1はrootなし。LD2 v≈1.3537976452、H6 LD3/9/14/17は約1.5555565724/1.3535929071/1.3512071954/1.3512071920。finite-hの信号とnative価格は別途評価し、BCH相殺だけで改善を判定しない。初期209 probe、72 qualified round点を保存し、最終policyは有限menuから選んだ。

H2/H6で新たに作ったsingleton/pair共同isometryは1,200件。列/source/native誤差を囲み、未登録の大きなsupport unionはfull basisへ戻す。最大列誤差約2.06e-15、Q source誤差約3.29e-14、native basis誤差約3.24e-14。9,490局所loweringの追加検査と、complete-number-sector normによってspin leakageも会計した。

samplingは既存G7 Closed P3、256-bit proposal、eta=rho=1e-24であり、新sampling familyではない。符号・wordの作用順を保ち、平均演算子は同じ有限P3。推定量はY=(event weight積)×Hadamard outcomeであり、通常の無重みHadamard平均と同一視しない。十分range LでY/L∈[-1,1]へ正規化する。rejection/zero-fillは使わない。

有限P3 tail、proposal/root angle、πとphase、binary64へのweight変換・積・最後一回のnormalization、SOURCE/native、QR、数値poweringを別々に会計する。全prefix weightがnormal/finiteであること、Wstar≥1、domainを検査した。normalized signalのrhoと残り統計位相βからNを求める。normalization改善だけから総資源改善を結論しない。

## 信号・誤差・回路価格・独立評価

全選択roundの詳細は[ROUNDS.csv](../../../artifacts/track_c_df_truncation/2026-11-08/ordering/ROUNDS.csv)、[ERROR_BUDGET.csv](../../../artifacts/track_c_df_truncation/2026-11-08/ordering/ERROR_BUDGET.csv)、[VALIDATION.json](../../../artifacts/track_c_df_truncation/2026-11-08/ordering/VALIDATION.json)。

- **科学的資格**：元Hの理想信号に対して、切り捨て+PFのdirected複素信号をbinary poweringで囲む。192-bit interval、spectral residual、ground interval、state距離/norm、basis/exterior、全数sector対角L1、angle roundoff、scalar relative phase、native loweringを積算する。振幅下限rho>0、残りβ>0、alias条件を全18roundで確認した。
- **切り捨てとPFの区別**：Hkeepのexact信号のsigned phase、PFとの差、combined phaseを[TRUNCATION_MECHANISMS.csv](../../../artifacts/track_c_df_truncation/2026-11-08/ordering/TRUNCATION_MECHANISMS.csv)へ保存。最終保証はcombined signalを直接囲み、signed cancellationを隠さない。
- **測定回数**：各軸の十分式N=ceil[4 log(2/α)/(rho² sin²β)]をdirectedに評価し、36軸のα和をexact rationalで≤.05とする。既存continuous配分+整数局所交換を再利用。Nは十分上界であり、実機shotや実測成功率ではない。
- **native回路**：同じQiskit1.3.0/compiler A、rz/sx/x/cx。すでにcontrolを付けたjoint回路をlowerし、control branch間の相対位相を保持する。許容されるのはjoint全体の一つのglobal phaseだけ。lower後に新しいcontrolは付けない。B†→controlled diagonal→B、X/Y wrapperとscalarを含める。opaque unitary合成、近似commutation、angle cutoffは使わない。
- **公平な合成**：全方式にterm parity/GraySynth topologyの同じ正確な対角合成を与え、各moduleでRZ優先lexicographicに一つを選び、その同じ回路のCX/size/depthを使う。identity-only fragmentではGaussianを省略する。PRは同じD合成と、新しい資格付き共同isometryを使う。
- **modular長時間価格**：deterministicはfirst+(steps−1)middle+last、PRはD/R境界とcomplete R occurrenceを実compileして連結する実装。RZ/CX/sizeの加法性とsource/native actionを確認する。serial depth上界はmodule depth和。巨大m17回路を直接生成した推測価格ではない。
- **独立な計算経路**：選択・α/N凍結後、expm+complex Schurによる長round信号、full controlled statevectorのX/Y、1/2stepのD/R/scalar・非return word、finite-P3 mean、branch decoderを照合。H6は8192状態成分へnative gateを作用させ、途中のsector射影を使わない。「独立」は実装経路・データの分離であり、別AI/別研究者によるレビューではない。

初期PR価格のtrainingは39cell×2strata×32=2,496 trajectories、初期holdoutは30cell×2×64=3,840。追加の順序探索ではこれら両方をselection dataへ明示的に再分類し、新しい2cellの128 trajectoryを補った。最終policy凍結後の新しいholdoutは29cell×2×64=3,712 trajectories。SHA256 seed namespaceを分離し、最終holdoutとのseed交差は0。旧H4 PRはc5a3の独立holdoutとcovarianceをそのまま再利用する。

価格MCは同じ256-bit lawのroot/no-rootをexact整数で条件付け、完全なR occurrenceをcompileする。量子sampling lawは変えない。期待価格にはweightを掛けない。shared cellが複数roundで使われる場合は係数を合算してから共分散を計算し、独立と誤認しない。表のSEは価格MCのnominal SEであり、RPE失敗確率とは別である。正規近似SEから分布自由の価格信頼保証や大域最適性は主張しない。量子backend shotsは0。

## 切り捨て・分子サイズ・機構

| 分子 | m17の採択Truncated | 除去係数数 | 数値 norm(Hdrop) [Ha] | E0(Hkeep)−E0(H) [Ha] | trunc / PF / combined signed phase [rad] |
|---|---|---:|---:|---:|---|
| H2 | coeff_L1_0.01; Y4_order3021, h=0.32 | 11 | 6.890831e-16 | +0.000000e+00 | +0.000000 / +0.022755 / +0.022755 |
| H4 | coeff_L1_2; Y4_order1023, h=0.16 | 204 | 2.180416e-04 | +8.335886e-06 | -0.021852 / +0.021345 / -0.000507 |
| H6 | prefix13_I0; S2_order2013, h=0.08 | 395 | 4.632769e-04 | -1.067390e-04 | +0.279810 / -0.313816 / -0.034006 |
H2の除去L1は約3.33e-16 Haで、11個の極小係数を除いてもground差は保存精度で0。Full比の総RZ減少は5.14%だが、これは有意なDF fragment近似の成功ではない。最後のfragmentを落とすprefix2はI0/I1でもground差約−.06710/−.04119 Haとなり、十分screen外だった。

H4 m17のcoeff_L1_2は累積L1約3.13e-4 Ha、204係数を除き236非identity項を保持する。数値drop normはenergy toleranceより大きいが、元H stateのdirect signalで資格を満たす。truncation −.021852 radとPF +.021345 radの相殺を含む。全policyではFull比40.90%低RZで、十分trial数は14225→14629と増える一方、native単価が下がる。

H6 m17は18中13 DF fragmentとone-bodyを保持し、末尾5個をidentityごと除く。truncation +.279810 radとPF −.313816 radが相殺し、combined −.034006 rad。combined directed上界は.034589 radで、誤差を相殺した観測だけに依存していない。全policyの総RZはFull比41.26%減。誤差を絶対値で別々に加えるだけの判定とは異なる、固定入力・state・h/orderに対する直接保証である。

PRの総RZに占めるR部分はH2約9.5%、H4約73.6%、H6約8.8%。H4ではDだけで約3.44e11 RZに加えてR約9.60e11が必要。H6では最終PRはD主体でも、Truncatedの単価・位相余裕・Nの組合せが安い。H2ではtailを保持するPRが、fragmentを落とせない決定論候補より低価格になった。これを一般的なPR優越へ拡張しない。


H2では今回の元prefix menuで実fragmentを除くと大きなenergy/phase biasが残り、採択できた非Fullは極小係数の除去だけだった。この有限menuの負結果を保持し、「どんな切り捨ても不可能」とは一般化しない。

H4/H6では元DF構造の小係数・低寄与tailを除く余地がある。ただし||Hdrop||やHkeepのground差だけで勝ちとせず、元H stateの信号振幅、signed phase、必要N、実native単価の変化を併せて判定する。特にH6ではtruncationとPFのsigned phaseに相殺が見られる。相殺は選択された固定h/orderへの性質であり、別条件や未知のstateへ一般保証として移さない。

基底変換と対角合成の価格、tailを保持するPRのrange/RTE誤差・trial増加、切り捨てによるPF step/retentionの変化が競合する。分子サイズが増えるだけで一方向に優劣が変わるという結果ではない。3分子の有限結果から漸近スケーリング則は確定しない。

## 技術修正・未完了記録の保全

本研究の初期rawも削除せず、以下を後続結果と区別した。

1. H6 input完成前に開始した最初のpilotはFileNotFoundErrorで終了し、完成後のpilotを正本とした。
2. 初期directed qualificationのH2/H4計144行を保全。Hermitian target、全数sector norm、state normalization移送の明確化と、同じ演算を囲むcached/vectorized normで技術的に置換した。H6未完了をPASSとせず、本人の初期workerだけを停止した。修正後216点が完了した。`QUALIFICATION_V1_TECHNICAL_SUPERSESSION.json`参照。
3. identity-only blockのGaussianを省く正確な合成修正で再価格評価。旧scanと新scan_repricedを別に保存し、完了セルは再利用した。queue handoffでは割当済み科学callを中断せず、他者のprocessは変更していない。
4. 初期のsource欄は実行中に観測したHEADだった。`SOURCE.json`は実際にloadされたSOURCE commitと観測HEADを区別する。以後はimport時点に固定した。過去rawのsource欄を後から書換えない。
5. PR長weightのdecimal serializationがPython桁数制限に当たった4件は、exact hex表記へ変更して再検査。成功済み32件、凍結policy、holdoutを保全した。科学的な演算・weightは変えていない。
6. 人工modular testで最初にstrict joint matrix equalityを要求したが、compiler Aの既存契約はjoint共通global phaseまで。両control branch全体で一つのphaseを整合して修正し、**8 tests PASS**。相対位相を免除していない。失敗logも保存。
7. 初期ROUNDS.csvの旧H4 Closed range列はkey名の違いで0になっていた。元q/rho/β/Nは正しく、最終ordering/ROUNDS.csvでは実際の`RA_log_Wmax_upper`を出力する。旧CSVはraw履歴として保全。

8. 順序auditの出力名にLDがなく、H6 PR m16の異なる2 splitがaudit集約名だけを共有していた。distinctな元`pr_qualified`には48計算全てが残っている。再計算せず完全な48点viewを復元し、最終holdoutを読まずに元selection dataだけで選択を再確認した。全8新policyのkeys・α・N・training RZは完全一致し、元policy/holdoutを保全した。`QUALIFIED_COMPLETE.json`と`SHORTLIST_COMPLETENESS_CHECK.json`参照。

全実行はprimary agentで行い、AI subagent・詳細GPT科学レビューは使用していない。workerはaffinity 64とtask数に応じ並列化し、重なるstageは48+16や32+32で協調した。BLASは各worker1thread。共有OS/driver/library設定、他利用者process、旧SOURCE/input/grant/ledger/marker/STOPは変更しない。

## SOURCE・raw・再現

[SOURCE.json](../../../artifacts/track_c_df_truncation/2026-11-08/SOURCE.json)に各stageの実SOURCE、再利用ref、input identity、環境、主要file hashを保存する。[REPRODUCE](REPRODUCE.md)にraw復元と再現手順を記す。歴代SOURCE commitは`SOURCE_HISTORY.bundle`に収録し、基点c5a3への依存を明示する。公開snapshotのcommitは科学実行SOURCEと区別する。

[RAW_INDEX.json](../../../artifacts/track_c_df_truncation/2026-11-08/RAW_INDEX.json)は全raw memberのbytes/SHA256、[ARCHIVE_PARTS.json](../../../artifacts/track_c_df_truncation/2026-11-08/ARCHIVE_PARTS.json)はlossless tar.gzの分割順・part hash・全体hashを持つ。INPUT_ARRAYS/全候補/scan/interval証拠/価格trajectory/独立検査/失敗logも含める。小さい元H2/H6保存入力はbyte-identicalなcopyも同梱する。非公開原本pickleは保持し、公開しない。

public replayはH/state/B/sp/対角/coefficients/gate parametersが既存cacheと完全一致した。SCF/DF fit/ground solveを再実行せず再構築可能。原入力identityにある旧runの`native_allowed:false`等は原生成記録として保全し、今回の独立承認scopeと混同しない。

## 結論の範囲とSTOP

同じ元Hの推定課題・精度・失敗上限で、今回の有限menuは**H2：改良PR、H4/H6：Truncated PF**を支持する。H2 PRは最良Truncated比36.20%低RZ。H4 TruncatedはPR比74.40%、H6 TruncatedはPR比24.41%低RZ。H2/H6の最良対次点のgapはnominal MC SEのそれぞれ約3879/197倍であり、今回の順位は価格MC標準誤差より十分大きい。H4の最良対次点はどちらも決定論価格。探索選択biasは新しいholdoutで分離したが、大域最適性・hardware優越の保証ではない。

未確定なのは、より広いretention/order/split menuの最適値、状態準備・hardware noise・T合成込みの価格、oracle ground仮定を外した成功条件、別geometry/基底/分子への移送である。いずれも今回の比較完了とは分ける。

**本研究は3分子の実行可能な比較・検証・保存を終えてmandatory STOP。** 次の担当はユーザー：結果の採否・後続研究の要否を判断する。Codexは新しい科学計算、P5、新sampling family、H8以上、詳細GPT科学レビューを自動開始しない。
