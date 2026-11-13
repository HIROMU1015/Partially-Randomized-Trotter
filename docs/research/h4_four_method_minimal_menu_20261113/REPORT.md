# H4 DF10：4方式の最小比較menuと不足証拠

2026-11-13 JST。**既存の4方式controller/plannerで足りる。新frameworkは不要。** PRは全13meanと条件付きS3、構造E[S]を持つが、適格な全law平均compiled E[C]とglobal RMSEは未取得。ベースラインは保存signal/価格と技術入口を再利用できるものの、同じRMSE契約で比較できる完成scheduleはまだない。**G/winner=null、menuは提案・未採択。**

[証拠matrix](BASELINE_EVIDENCE_MATRIX.md)／[最小menu](MINIMAL_COMPARISON_MENU.md)／[次の有限作業](NEXT_STEPS.md)／[HANDOFF](HANDOFF.md)／[読取metadata](EVIDENCE_METADATA.json)／[40件の既存回帰](REGRESSION_TESTS.txt)／[技術確認](TECHNICAL_CHECKS.txt)。今回は保存metadata・コード・資料の調査と人工回帰だけで、新signal/reference、保存科学値の再解析・ランキング、simulation/sampling/compile/価格認定/実Gは0。

## 比較の共通条件

Linear H4/1.00 Å/STO-3G、保存tol-only1e−8 DF10/weighted-Hermitian completion、sector(2,2)、同じ保存H/state/snapshot/fragment順を基準にする。「元H」はPF・finite-RTE・切断の前の保存completed H_DF。DF前の分子Hamiltonianへのtarget変更やDF誤差予算追加はしない。

共通目標εE=.00159360010199040 Ha、既定εqpe=εtrot=εE/2・minkowski/finite_B_diagnostic、h=.1、M12/T_m=h2^m、元prior/decoder/count caller・P0を保つ案。RPE時間は揃えるが、方式固有のPF幅・formula・保持prefix・K・event数は一致させない。将来h/Mを探索する場合は別menuである。

費用は`G=Σ_(m=0..12,a=X,Y) N_(m,a) E[C_(m,a)]`。Cはcontrolled時間発展＋scalar一回＋full Hadamard wrapper＋軸readout＋measurementを含むcompiled RZ/shot。決定論PFのCは同じ完成回路の直接値、ランダム方式のE[C]は宣言lawの期待値。D-only/uncontrolled/構造S/有限cohort平均を全law Cへ代用せず、normalizationや古典価格標本数をN×Cへ二重乗算しない。

同じcompiler/profile（rz/sx/x/cx、Qiskit1.3.0 opt1 seed17等）に加え、**lowering IDを価格contextとして一致**させる。D旧`CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424`と新`CONTROLLED_ZZ_CX_CRZ_CX_V1`は別scenario。旧価格を上書きしない。新loweringをPRだけへ適用し、他方式を未確認の旧実装に固定した比較をalgorithm差と呼ばない。

## 重要な再利用と不足

| 方式 | 保存成果の利用可能性 | 今回の主な不足 |
|---|---|---|
| PR | B2084791の13round S3、A169d/bbaのmean/物理契約、C9d95の全26構造recordはcanonicalと一致。D54b64の4 pathは新loweringの開発cohort | 全事象loss/population精度、proxy/数値/lawの資格、全law平均compiled C、同loweringの比較価格 |
| Full PF | D e7dcaの136候補中73でX/Y直接価格SOURCE一致。2nd/標準Yoshida4/M5の全H物理経路とV2対応あり | 13round scheduleの大半は保存grid外。RMSE型N/全体誤差契約なし。EXACT価格はRMSE資格ではない |
| Truncated PF | H4 B0 p7/S2/幅.1のsignal/referenceは旧N研究で全13round保存。B0I p7/Yoshida4/幅.4も短round保存あり | 元Hへの切断bias込み全RMSE、RMSE型shot政策、同scheduleの価格。旧HP型Nは流用しない |
| Full-H pure RTE | one-bodyをZ成分化して全DF tailへ追加する専用実装、κなしq/R/K layoutのV2接続、旧T=.8とT=2の保存研究 | 新13round layoutのmean/normalization/N/C、全Hのlawと平均価格。DF B3は一体項が決定論なので代用不可 |

A97146481の調査を採用して不足を重複開発しない。直接risk評価なら、PF/finite-K/stateの作用はrawに現れるため同じbiasを再加算しない。共通有効Hamiltonianの全面証明や全reference pairの再取得を初手の必須条件にしない。認証globalを求める場合は、target包含・raw誤差・law移送など、そのclaimに必要な追加証拠を別に扱う。

## 提案menuと判断順

初期は**11 schedule案**：PR保存1、Full PF 3公式×幅.1/.4の6、truncated B0 p7/S2/.1とB0I p7/Yoshida4/.4の2、Full-H RTE K2/K4の2。固定条件を正式採択した集合ではなく、最適性を主張する集合でもない。Full-Hの時間依存layoutは旧T2のK2 anchorを参考にした明示案で、旧価格の適用を仮定しない。

まず全meanを取り直さず、保存セルを受入れ、追加meanはm3/m6/m12の限定screenを候補とする（最大20 complex mean、別承認）。局所診断の良さだけで全RPE適格とせず、完成scheduleとN/RMSE契約を満たした候補だけを価格段階へ送る。幅.2・near-full prefix9・K0等は必要な場合の限定補間/境界probeとし、全格子へ自動拡張しない。

PRの固定nominalだけ最適化済みとは扱わず、ベースラインへ同じNを押し付けない。候補のN政策と価格精度・未探索範囲を対称に管理し、候補不足をもってPR優位やpure-RTE劣位を宣言しない。科学的保証を比較するなら全方式に同じclaim水準が必要。条件付き観測だけの比較はその限定を明記する。

## 技術確認と終了

既存method coverage14、Full baseline14、evidence planner12の**40件PASS**。人工N/Cによる既存積和・欠測/未資格/SOURCE拒否・4方式区分・非起動を確認した。8次を含む既存1methodは対象外にして実行していない。C26のfull identity/lawとnull資格、D新lowering/cohort、保存Full PFのmenu照合、B0全13のsource chainをmetadataだけで確認。科学数値の再集計やraw trial読取はしない。

新しい実装・科学テストは追加していない。B既存branch `codex/rpe-decoder-statistics-20261113`を再利用し、A/C/Dのworktree/process/cache/原本へ非介入。SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPを保全。十分な計画が得られたため追加調査を打ち切る。

**次：ユーザー**がclaim水準・menu・有限scopeを選ぶ。**Codex B**が採択後のmetadata接続、**A/B**が許可されたN/RMSE、**C/D**が同loweringの平均compiled価格をそれぞれ担当する。今回はcommit/push・remote SHA/tree/主要成果物照合後mandatory STOP。正式4方式比較や新科学計算へ自動進行しない。
