# 案M：1 slotの適格性とtarget対応

2026-11-08 JST／SERVER1。baseは[設計SOURCE 154c6e3][BASE]。今回の[adapter](../../../src/trottertracks/algorithm_codesign/m_label_average.py)は、人工gate記述に対する**十分条件による適格性判定**を実装する。一般のoperator等価性判定器や実H4へのbindingではない。

## 入力と保存する意味

`SlotLabel`はlabel、context η、slot、selected policy/lawのidentity、実際の無条件label質量、A/B/Vの明示Qiskit system circuit、system ordering、selected basis、signed Z/ZZ support、Product/Rotation、角φ、scalar/event phaseを持つ。support空はidentityであり、sign=−1も保持する。

callerが各記述を元selected target

\[
 U_{\eta,l}=e^{i\chi}A_\eta V_g^\dagger R_lV_gB_\eta
\]

へ結合することが前提。χは共通scalar＋event phaseであり、A/B自身のglobal phaseも別に保存する。R_l=D_lまたはexp(−iφD_l)、D_lはsigned対角involutionである。A/B/Vはsystem-only、作用順はB→V→R_l→V†→A。数値的global-phase fittingはしない。

`qualify_group`は**与えられた記述同士の一致**を調べる。偽のidentityやA/Bをcallerが渡しても、その由来を自動認証することはできない。実H4のselected policyから正しい記述を取り出す接口は未実装・未検証であり、既存のfull circuit builder/engineを実入力で起動して補完してはいない。

## 受理・拒否

| 条件 | 実装と拒否理由 |
|---|---|
| 同じη、1 slot、selected policy/law | `CONTEXT/SLOT/POLICY/LAW_*`。空identity、重複labelも拒否 |
| 同じA/B/V | gate型、parameterのbinary64表現、ordered wires、control数/state、circuit global phaseを構造比較。`A/B/V_SEQUENCE_MISMATCH` |
| 同じselected basis | IDと実V列をともに比較。original basis IDは参照情報に留める |
| 同じkind/φ | Productに角を与えない。Rotationは明示Angleが必要。`KIND/PHI_MISMATCH` |
| 同じscalar/event phase | 二項を個別に比較。channel一致・phaseの近似一致で共通化しない |
| 有効なsigned diagonal | distinctな0〜n−1のwire、長さ0/1/2、sign=±1。identityを落とさない |
| 同じsystem ordering | n個の一意な明示wire名と順序。ancillaは常にsystemの後のwire n |
| actual Qとの結合 | 既存`CategoricalLaw`とlaw.source、各row.actual_massの完全一致。元質量は非負で53-bit grid上、採用rowは正質量 |
| 量子入力の限定 | 既存A2の明示standard-unitary検査を再利用。測定/reset、unbound/custom oracle等を受け付けない |

回路の等価性が偶然成り立っても、gate列やphase記述が違えば受理しない。この保守的判定で全ての数学的に可能な群を発見できるとはしない。copyした回路はbuild前に署名を再検査し、資格取得後の変更を`QUALIFIED_GROUP_MUTATED`として拒否する。

`partition_supplied_rows`は同じslotの**明示的に与えられたrowsだけ**を共通署名で分ける。価格/行列/状態を使う探索は行わず、不正rowを捨てない。分割後も外側contextと群の質量はcallerが保持する。各群を一様に取り直す処理ではない。

## Actual qと群平均

元lawの質量をp_l、明示群gの質量をp_g=Σ_g p_lとし、q_l=p_l/p_gをFractionで正確に計算する。`EligibleGroup.mass`と`conditional`を区別して保存する。全lawをgへ勝手に再正規化しない。g外の正labelは元lawに残る。zero-grid labelをgroupへ誤投入したら`ZERO_ACTUAL_SUPPORT`として拒否し、law自体のzero記録は消さない。

\[
 r_g(z)=\sum_lq_l\,\operatorname{sign}_l(-1)^{\sum_{j\in support_l}z_j}
\]

はFractionとして評価する。q_l≥0、Σq_l=1から|r_g(z)|≤1。これはmetadata上の正確な性質であり、実H4のq抽出・selected target結合を認証したという意味ではない。回転角・phaseの`Angle`はradまたは正確な有理数π倍を明示する。π/2の整数倍に対する三角値だけを記号的に扱い、float π/2を許容差で0へsnapしない。

## 再利用と未取得

既存`a1_control_variates.CategoricalLaw`、`a2_lcu_access._check_system_circuit`、Hadamardのbit mappingをそのまま呼ぶ。DFの`DFRTEComponentCircuitSpec/DFRTEApplicationBasisChoice/DFRTEBasisPlan`、`df_rte_qiskit`のsigned support・inverse順序を接口の意味確認に使ったが、旧SOURCEは変更しない。現在の新規量子buildは人工1/2 system qubitだけで、H4幅のbuildは明示的に拒否する。metadata適格性の幅そのものに恣意的な2-qubit制限は設けない。

実H4での非自明群の存在、actual保存記述からA/B/V/phaseを切り出すbinding、正support全体の被覆、全wrapperの価格・誤差はUNKNOWN。[次のmetadata-only草案](NEXT_MINIMAL_SCIENCE_DRAFT.md)は提案に留める。

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/154c6e3cf1f48d27f558003b0e9bd74882414d9e/docs/research/o2_new_algorithm_candidate_exploration_20261108
