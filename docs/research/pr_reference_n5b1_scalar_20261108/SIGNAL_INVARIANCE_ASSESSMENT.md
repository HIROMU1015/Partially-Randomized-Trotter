# N5-B1：信号不変性とstrict保証の境界

2026-11-08 JST。[REPORT](REPORT.md)。SOURCE命題は[JOINT_SCALAR_FACTORIZATION](JOINT_SCALAR_FACTORIZATION.md)、有限診断は[LOCAL_RUNTIME_CHECKS](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/LOCAL_RUNTIME_CHECKS.json)。両者の証拠の型を分ける。

## 誤差の帰属

| 項 | 今回支持する位置づけ | 保持する不足・制限 |
|---|---|---|
| Wのfloat具象化 | Control構成後の同じjoint named productに対する共通scalar | Logical wordのexact Wを削除しない。Strict operatorではscalar差を保持 |
| Local inverse／local→segmentのphase算術 | 同じnamed inverse／gate列なら共通scalar | 各位置のnamed列が同じであることが前提。Actual adjoint以外へ延長しない |
| Outer初期phase／segment→outer shell | Float fieldだけの処理は同じjointの共通scalar | 旧shell certificateを0へ更新しない。SOURCEのP角を含む旧誤差と区別 |
| Logical wordとRZ(theta) | 保存strict guardの対象 | Up_to_phase=False、W込み。新guard／再合成なし |
| Dyadic量子化／basic word和 | Logical targetへの差に残る | b36／内部10^-13の既存保証をm17へ移す資格は未完 |
| SOURCE scalar／event／vacuum P | Relative angleを実装するRZ等に残る | W具象化scalarと異なる。Actual binary64 literalを変えない |
| SOURCE→native／finite-RTE／finite-bit／vacuum算術 | 既存raw disk側に残る | Candidate Gを理由に控除・再会計しない |
| NumPy／Operatorの丸め | 局所技術診断の有限精度 | Named ideal gateの物理誤差や新strict boundへ移さない |

既存64u・辞書emitter項・shellには共通scalarの保守的評価が含まれる。信号不変性を研究上採用できれば、その**特定成分の信号寄与**を別扱いできる可能性がある。今回は既存strict certificateの値やN4-Cの会計を変更しておらず、新しい数値係数・raw diskを作っていない。

## 適用資格

固定consumerの許容入力クラスについて、任意の有限word列とwire embeddingで同じnamed productが残ることをSOURCEから支持した。このクラス内では共通scalar測定不変性は任意rho／Mに対する代数であり、特定ground状態の古典記述や未知E0／gapを使わない。

固定10件・4 prefix・正負・CX前後は、そのSOURCE解釈と実runtimeが整合する有限技術証拠である。Reset/readout／clbitの同一性も直接検査した。採用160組にrelative phaseやgate/wire不一致はなかった。一方、有限テストが任意m17入力のpolicy／domain／sinkを検査したわけではない。

実RPEのX/Yは同じjoint回路の観測量として扱う。同じomega／axisでの恒等式を同じcanonical lawで平均するため、trajectory依存のchi自体を揃える必要はない。Normalization／sampling law／range／shotを変更したり、uncontrolledな平均unitaryを置き換えたりしていない。

## M17に残る課題

背景はH4／1Å／STO-3G／DF12、Partial S2 LD3 h=.02 K2、q131072／r32／88 bits、B′626／Reference630 shots/axisのまま。[N5-B0](../pr_reference_n5b0_static_20261108/M17_ERROR_BUDGET_BLOCKERS.md)の保存証拠を引用し、今回再計算していない。

1. **Full consumer path**：全必要native位置で同じnamed gate意味、numeric field、inplace経路、後置controlなしを保つこと。別adapterや本番sinkへ無条件に移送しない。
2. **Logical strict誤差**：C0 logical guardだけなら保存m17 Y cap内のlibrary wordが11件あるが、全必要wordの資格ではない。43 basicの実guardだけの照合はB′0件／Reference1件で、選択chunk和を保証していない。Guard上端超過は実誤差の下限でも新規合成の必然性でもない。
3. **量子化**：b36の保存量子化保証はm7条件の仕様。Common scalar不変性はRZのrelative角の量子化を消さない。Actual m17 target集合との適合は未完で、新bを選んでいない。
4. **域**：固定Lはm17 vacuum包絡を覆わず、B′ scalarの保存literalは実際に域外。Eventの域外supersetとcanonical到達性を区別する。現providerはC0 lookup前に域を判定する。
5. **RPE／費用**：新joint error和、旧diskとの合算、半径／位相、保存N十分性、T価格が未評価。626／630の違いにより、共通frameの1-shot差取消しを総Tへそのまま使えない。

Source-supportedな因子分離はこのリストを解消しない。新規Clifford+T合成が必要かどうかも未確定であり、先に必要なlogical policy／precision／域の条件を判断する。**SIGNAL_FACTOR_QUALIFICATION_INCOMPLETE（m17への適用）、M17_RPE_UNQUALIFIEDを維持しmandatory STOP。**
