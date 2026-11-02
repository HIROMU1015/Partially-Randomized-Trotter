# 研究C：局所step幅とreduced勾配精度の切り分け・技術準備

2026-11-02 JST。起点 `0bb25e3c0cd50b6794a78513925043d216607827`。対象は保存直鎖H4・隣接1.00Å・STO-3G、number CDF8/元順LD2、Nα=Nβ=2/sector36の既存結果。

## 1. 完了範囲と推奨

**保存JSON/JSONLの整理・人工検証・最小接続・未承認49-call契約を準備した。実H4の追加step/inner/outer/NPZ数値再解析は0。** 旧60-callのSOURCE・raw・STOPは保全し、公開照合後mandatory STOPする。

今回の技術的な結論は、①旧8点で局所降下をまだ見ていない可能性、②inexact innerによるreduced方向微分の誤差、③F差分の数値分解能を、同じ説明にまとめずに検証すべきということ。**実H4でどれが支配的かは未確定**で、solver変更・Joint移行・性能比較を行わない。

次案はArmijo trial数の追加ではなく、**同じ初期候補方向における3段階の±対称プローブ**。各U点でJを解き直し、初期の候補方向微分とreduced Fの差分・scale依存を比較する。U更新の受理は0で、FDの一致を真のreduced gradient資格へ昇格させない。

[未承認49-call契約](research_c_pr_local_direction_contract_unapproved_20261102.json)。

## 2. 保存値から見た旧step幅

[固定報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0bb25e3c0cd50b6794a78513925043d216607827/docs/research/research_c_pr_guarded_step_execution_20261102.md)と既存JSONを参照した。[保存scalarからの計算](../../artifacts/research_c_pr_local_step/2026-11-02/technical/saved_step_scale.json)は許可されたJSON分析のみで、J/U配列を新規decodeしていない。

F0=.012267644429815034、報告されたs=g0·d=-8.731877899141525。非負のFに対する一次モデルの尺度を`α_F=F0/|s|`と置くと、**α_F=.001404926245134638**。旧最小α=.002670284759424921でも`α|s|/F0=1.9006583219`で、旧8点すべての一次モデルF0+αsは負になる。

旧最小点の予測減少は約.0233でF0を超え、実測F=.0205435180との差（reported linear-model defect）は予測減少の約1.35493倍だった。ただしdefectには非線形性・inner値誤差・候補勾配誤差が混在する。**正確なΦのTaylor剰余や曲率/Hessianの測定ではない**。

α_Fは「一次予測がF0を使い切る尺度」であり、Lipschitz/curvature boundでも局所降下保証でもない。より小さいstepを調べる動機にはなるが、縮小だけで問題が解決すると結論しない。

## 3. Inner精度に関する保存事実

[既存inner診断の整理](../../artifacts/research_c_pr_local_step/2026-11-02/technical/saved_inner_observations.json)。全9点のいずれにもraw precision lossがあり、うち7点が数値資格PASS、2点がstability未資格。したがってraw status2だけでは両者を分けられない。

最終J基準は全9点MET。一方、初回solveではpoint3がmaxiter/status1・J未達、point4がprecision loss/status2・J未達で、登録された2 refinement後に最終J基準を満たした。最後のpairのFが同じ表示値でも、U差が1e-6を超える点があり、ΔJ normも同じU配列から保存されている。小さいJ残差・F変化・refinement一致のいずれも、真のinner argminへの誤差上界ではない。

実H4のquotient Hessian/H_UJ、gauge方向と非gauge方向のΔJ分解、正確なinner最小値・真のreduced gradient誤差は未取得。今回これらをNPZから再解析しない。

## 4. 人工問題で分離した機構

[人工結果](../../artifacts/research_c_pr_local_step/2026-11-02/technical/artificial_local_direction.json)。解析的なinner最小値・真のΦ・微分が分かるtoy、およびinvented DFを使った。保存H4のHamiltonian/J/Uは使っていない。

| ケース | 真の方向微分／候補s | 対称FDの観測 | 切り分け |
| --- | --- | --- | --- |
| 適度な曲率、正確なinner | -1 /-1 | 各scaleで-1、小stepでF減少 | 旧大stepが不成立でも局所では降下できる例 |
| 大きい曲率、正確なinner | -1 /-1 | 各scaleで-1、それでも全probeでF増加 | 勾配が正しくても選んだstepがまだ大きい例 |
| バイアスのあるinner | +1 /-1 | 各scaleで約+1、F増加 | J gateとrefinement一致だけではenvelope微分が正しいとは限らない |
| 同じtoyのinner誤差を小さくした対照 | 約-1 /約-1 | 差約2e-10、F減少 | 数値基準を緩めずinner精度の影響を確認。ただし実H4 solver修正ではない |
| 正確な微分だが高次項あり | -1 /-1 | -1.015625→-1.00097656→-1.00006104 | 中央差分のtruncationを勾配FAILと誤認してはいけない |
| F比較尺度/inner値誤差に埋もれる例 | -1e-12 /-1e-12 | FDがscale縮小で悪化、信号は比較尺度以下 | 小さくすれば必ず精度が上がるわけではない |

バイアスtoyは`F(j,u)=(j-a u)^2/2+φ(u)`、a=1e6、inner残差e=2e-6。J一次は元1e-5内、2 refinementが同じ解ならU差0だが、partial Uに-ae=-2の誤差が残る。固定JでのFDはこのpartialと一致しても、Jを最小化した真のΦのFDとは逆符号になり得る。これは**partialの正しさとenvelope精度の違い**を示す人工反例で、実H4の初期g0が誤っている証拠ではない。

解析toyでinner誤差を2e-16へ変えた対照では方向微分差が小さくなった。しかし実H4でinner精度が不足していると原因確定したわけではなく、既存Newton-CG/目的/閾値は変更しない。

## 5. 差分と数値誤差の扱い

近似Fを`F_tilde=Φ+e_inner+e_arithmetic`と書けば、中央差分には`(e_plus-e_minus)/(2h)`が入る。誤差が有界と分かる場合でもhを小さくしすぎると増幅される。滑らかな真のΦの中央差分には別にO(h²)のtruncationがある。**今回、実H4でその定数やinner値誤差の上界は分からない**。

新診断は次を別々に記録する。

- 近傍ごとにJを解き直したF(+h)/F(-h)、forward/backward/central差分、候補sとの差、scale間のFD変化。
- 元のvalue比較尺度`atol+rtol*max(|Fa|,|Fb|)`。
- 既存のrefinement F変化、extended/binary64 F差、独立AD F差。
- 上記の最大値を比較のengineering sensitivity scaleとして記録。中央FDではその尺度を2hで割った値も記録する。

これらは同じFの既知の観測尺度で、**正確なmin_J Fを包む区間ではない**。真のinnerバイアス、truncation、真の勾配誤差はnullのまま。未知誤差を0とせず、FDの一致/不一致を機械的な真の勾配PASS/FAILにしない。

F(+h)-F0が比較尺度より小さい/大きい/尺度内であることを`EMPIRICAL_DECREASE`/`EMPIRICAL_INCREASE`/`UNRESOLVED_VALUE_SCALE`として保存するが、これは診断ラベルでU受理ではない。片側でも未資格ならそのpairのFDはnull。登録された他の独立probeは実施できるが、correctness FAILでは即STOPする。

## 6. 実装・検証

新規は`c_pr_local_direction.py` / `c_pr_local_direction_pilot.py` / `run_research_c_pr_local_direction.py`と人工test2ファイル。旧160 SOURCEは無変更。旧guarded Oracle、同一decoded J評価器、Newton-CG/Hessian、3 solve/2 state/admission、rank/nullspace、input loader、Calls、raw/export/resources/STOPを再利用した。旧entryの科学scope/Armijo受理規則を変更しないため独立entryとした。

新しいoptimizer・fitter・Gaussian/CAR・数値計算libraryは作らない。FD組立は計算済みscalarだけを使い、新しいobjective/gradient問い合わせを起こさない。既定CLIはpassive preflightで、実H4実行を自動開始しない。

[テスト結果](../../artifacts/research_c_pr_local_step/2026-11-02/technical/test_results.json)：**269 unique items PASS（新18＋既存251）**。今回実行したfit257/evaluation10と、出力型・非有限slope処理を仕上げた後の新18を重複除外した件数。旧251は無変更の関連SOURCEに対して今回実行し、過去のPASSだけを転記していない。

人工genuine DFの全7点でJを解き直す経路、3対称pair、zero/nonfinite scale、片側/初期未資格、correctness FAIL、49-unit計上、fresh output/STOPを確認した。Study exportでNumPy bool_をJSONへ渡す不具合が出たため組立値をboolへ変換し、JSON回帰testを追加。失敗log/partialは技術履歴として保存し、完成studyは最終SOURCEで再現一致を確認した。CI/外部再現ではない。

## 7. 推奨する最小後続科学契約（未承認49 calls）

保存H4/1Å/STO-3G、number CDF8/LD2、Nα=Nβ2/sector36、元qualified J/U/x0、γ=.1/ε1e-8、f*=2.1839632053132894e-9/M*=2.2093719370722074、κ/scalar/spin/frameを維持。J/U一次1e-5＋元margin、gradient atol1e-8/rtol1e-7、value atol1e-10/rtol1e-8、stability1e-6、元rank/nullspaceを維持する。

Initial eta=0を新たに同じ手順で評価し、数値資格を通った候補g0からd=-g0、s=g0·d、α_F=F0/|s|を定義。**h=α_F×{1/8,1/32,1/128}**の順に、eta=+h dと-h dを各1点測る。元Jから初回solve＋2 refinementを全点で実施する。

保存scalarを代入した参考のhは約`1.75616e-4,4.39039e-5,1.09760e-5`、eta normは約`5.18940e-4,1.29735e-4,3.24338e-5`。**これは新H4の測定結果ではなく、登録scale規則への既存scalar代入値**。一次予測減少をF0の1/8以下へ抑える設計で、曲率boundや降下保証ではない。

| Logical units | 最大 |
| --- | ---: |
| saved_load / initial_identity / objective_setup | 各1＝3 |
| local_probe_driver | 1 |
| inner_J_solve（初期＋6側点、各3） | 21 |
| precision_state（各2） | 14 |
| reduced_admission（各1） | 7 |
| directional_diagnostics | 1 |
| classification / export | 各1＝2 |
| **合計** | **49** |

初期点込み7 distinct U点、inner21（refinement14）、**optimizer/受理U更新0**。旧8 trialの続行や単純な回数増加ではなく、微分・Fの分解能を調べる別契約。旧枠を転用せずfresh `artifacts/research_c_pr_local_step/2026-11-02/run01`、retry/resume0。

初期未資格/zero direction/非正F0ではprobeを始めず診断STOP。Sideの数値未資格はmissing pairとして保存し、既定の他probeだけ評価する。入力/SOURCE/非有限値/微分/value/直交性等のcorrectness FAILでは依存する後続を停止し部分証拠を保存。新しい混合Hessian・gauge分解・DF物理norm・PF/RTE/ground/native/shots/総資源・RC2/Jointは含めない。DF予算CA/8=.00019875Haは不変だが未測定/null。

SOURCE manifest SHA256：`8b22d88f22620879c45f1b3d78e377e2dfe471ce13377291f6935e249c577925`。契約SHA256：`02f6c7aa275ffe23f94e259a06c770190ab05be6299c18fe01f4c57a3fd6a51c`。

## 8. 保全・残る制約・次の担当

[passive preflight](../../artifacts/research_c_pr_local_step/2026-11-02/technical/passive_preflight.txt)はSOURCE165/input4・既存依存lock/ffsim SOURCE/versionでPASSし、実H4配列は開いていない。[保全postflight](../../artifacts/research_c_pr_local_step/2026-11-02/technical/preparation_postflight.json)で旧raw2495、旧SOURCE17組、旧60-call STOP/契約と科学記録135件を保全する。今回の追加は技術資料・索引のみで、科学結果をmanifestへ追加しない。

人工処理は既存NumPy/SciPy/JAX/Decimal等を再利用。小行列BLASを調整しfit/evaluationの独立試験を並列化した。逐次依存するJ/refinement/probeは順序を保ち、CPU/RAM/worker/wallの恣意的上限や共有環境/他job変更を追加しない。Speedupや資源優位は主張しない。

残るのは実H4の局所F/FDが未測定で、正確なinner最小値・true reduced-gradient誤差が未資格であること。小scaleでFDと候補sが整合しFが下がれば局所step説明が数値的に支持され得るが、証明ではない。尺度依存の不一致・値floor・未資格pairがあれば原因は引き続き未解決。勾配が正しくても選んだhが大きい例もあるため、結果を見て自動的に閾値・solver・probe数を変更しない。

次は**ユーザー／研究C**がこの49-call診断案を判断する。別途明示許可後のみ**Codex**が固定SOURCE/契約/入力を再照合し一度実行、公開STOPする。Jointへの全面移行、solver変更、PR/RC2性能比較は保留。承認証明書・承認記録・新grantは不要である。

Branch `codex/research-c-pr-local-step-preparation-20261102`。Commit/push後にremote SHA/tree・公開報告/SOURCE/契約/testと原本保全を照合し、**mandatory STOP。今回の作業で実H4の新しい計算を開始しない。**
