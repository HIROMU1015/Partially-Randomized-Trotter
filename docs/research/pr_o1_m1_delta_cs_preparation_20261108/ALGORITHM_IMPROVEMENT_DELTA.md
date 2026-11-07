# 改善差分：M1の重複排除とC-Sの同位置接口

2026-11-08 JST。既存の改善機構を接続する技術準備で、独立した新算法・資源優位性を認定しない。

## M1で変更しないもの

同物理H_D/H_Rを保って辞書を変える既知の機構は残る。しかし、global Pauli collect、DF回転基底Z/ZZ、S/P混在、既存selector/価格DP、nativeの合法共有を再び実装する追加価値は特定できなかった。[重複確認](EXISTING_EVIDENCE_AND_DUPLICATION_CHECK.md)の後続matched結果も再利用する。

同じ物理splitであっても、有限RTEの実装mean・normalization・CDF・角度・biasまで同じとは置かない。理想exact係数/同r,Kなら既存paired polynomial関係を参照できるが、その前提を実装lawの一般同値へ広げない。F2のN未資格、P3のengineering/条件付きstatusは維持する。これらの未資格を埋める作業は新しい科学評価であり、M1の新selectorを作る理由ではない。

## C-Sで追加する操作

既存[66553d94 C-S計画][plan]の「通常合成S0／eligible全体S1／D側のみS2」を、同じliteral rotation列へ接続する。対象primitiveは、**controlをlowerした後のjoint空間Pauli回転**

\[
R_P(\theta)=\exp(-i\theta P/2),\qquad
R_P(\theta)=R_P(\widetilde\theta)R_P(\theta-\widetilde\theta).
\]

粗角とresidualは同generator・同位置。同generatorなので理想2回転は可換だが、隣接する別generatorとの順序は変えられない。有限residualを使う実際の補正meanは、一般には右辺exact回転と異なる。

追加moduleは`same_position_residual.py`。`RotationSite`に元slot ID・joint Pauli・exact Angle・D/R/FIXED roleを保持し、`plan_sequence`でS0/S1/S2を一つの事前指定coarse表から作る。`splice_realization`はcoarse→提供されたresidual time wordを同slotに挿入し、隣接role・順序を保持する。全siteを中央tailへ移転したり、role間融合・角度周期化・価格後のslot選択を行わない。

### 再利用と接口の限界

- exact角度は既存`wrapper_sequence.Angle`（有理π＋有理radian）を再利用。
- finite operator coreは既存`CanonicalGenerator`のordinary、および`parent_allocation`のproposal/weight/phase処理を再利用する方針。人工testで既存`operator_target`/`event_statistics`へ接続した。新しいlaw・Parent配分・DPを作らない。
- `ResidualWord`はcoherent operator/time order/phase保持という入力schemaを要求する。**入力宣言は数値certificateではない**。provider実装、実generator対応、law/係数誤差、norm、strict合成誤差を自動認証しない。
- 新コードはlocal rotation sliceのplanner/spliceであり、H4全interrogationを読み込むrunnerではない。非回転gate・basis・wrapperを含むfull IRとの対応は、将来の入力adapterで固定する。
- original pathと追加residualの独立性、joint proposal、weight/range/normalization、誤差・confidence、価格は外部の既存provider/会計へ残す。勝手に確率を積にしたりUNKNOWNを0へ変えない。

## 既存S2/Parent/SPとの差

元のcanonical partial-S2/PFは変更しない。ParentのEuler恒等式やallocationを新方法と主張しない。SPのchannel PAIをcoherent residualとして流用せず、固定toyの8角度/23libraryやmask探索も使わない。既存Parent/S4/PS48/研究Cの停止familyへrunを戻さない。

新しい実装差分は**位置とroleに結び付けたcoherent residualの接続だけ**。物理的に新しい恒等式でも、未解決の統計最適化でもない。改善が残るかを問える条件は、既知の一様S1と通常S0にも同じcoarse catalogue・融合・合成精度と情報を与えた後、S2の配置が全費用に効くことである。

## 資源を削減し得る経路と相殺要因

| 経路 | 利益の可能性 | 相殺・UNKNOWN |
|---|---|---|
| coarse回転の低費用合成 | 通常の高精度回転を安いcoarseとresidualへ交換 | residual回路・追加phase/basis/control・合成情報取得費 |
| Dのみ配置 | 重い/低利益なR側への適用を避ける | 一様S1が同じ/より良い結果になる可能性 |
| 小residual | finite truncation・normalization増を抑え得る | 各slotのB/range増は反復で蓄積。角度が小さいだけでは総費用利益不明 |
| 同位置 | 中央R移転による追加の物理split/PF変更を避ける | finite/native実装meanは変わり、そのbias・半径を支払う |

主指標候補はstrict profile TのT/T†。CX・depth・古典費は別欄。旧Qiskit RZ単価へTを足し合わせない。RZ-only profileを使うならその限定目的を別途採択する。十分N×full-wrapper費用の比較まで行わない限り、単価が安いことを量子資源改善としない。

## 比較対象・未知・次の判断

S0は同じ物理H、同state、同PF/時刻の通常strict合成、S1は同coarse/residual方式の一様適用。S2だけに有利なfusion・低精度や真のsignalを与えない。元H ground、H_D/H_R、selected policyを保つことを将来の実入力binding条件とする。新しいstate solve/DF/CDFの生成を補完手段にしない。

現状は人工接口のcorrectnessだけ確認済み。H4のeligible inventory、coarse政策、strict native-provider、誤差予算、実full価格、N・総費用・相関・有望性はUNKNOWN。既存N1/N2/exact engineの認証を、変更後のresidual列へ自動継承できない。[次の最小契約草案](NEXT_MINIMAL_SCIENCE_CONTRACT_DRAFT.md)の別許可前に実入力を使わない。

[plan]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66553d948aa9df684987972789c6f3ee8a741312/docs/research/track_c_o3_pilot_plans_20261106/C_S_PILOT_PLAN.md
