# 既知法との境界

網羅的新規性レビューではなく、候補を重複実装しないための限定技術確認。既存O2文献読解を再利用し、最重要一次文献の該当節のみ再確認した。

[Rowe et al., SE-QPE v2](https://arxiv.org/pdf/2604.14921v2) §IIは二register CSWAP、vacuum phase補正、共通固有基底の複数block保証、fanoutを扱う。§V Fig.8付近には、短いpowerでcontrolled、長いpowerでSEを使うmixed構成が明記されている。したがってround/powerによる切替そのものは既知であり、候補Aをその発明としない。論文の分子・hardware・資源削減率をTrack Dへ移植しない。

以下の「追加部分」は提案または未解決の適用課題であり、新規性の正式認定ではない。

| 構成・候補 | 既知/重複する部分 | PRに追加し得る部分 | 今回認定しないこと |
|---|---|---|---|
| 真空Reference単体 | SEと同じCSWAP干渉、二lane、位相補正 | ランダムwhole pathとselected/strongのphase・law・numerical資格への接続 | 同一gadgetを独立新アルゴリズムと呼ばない |
| A round-only | mixed controlled/SEの既知組合せ | PRの有限RTE条件と同一精度/成功率での実装選択 | 単なるq閾値に独立新機構があるとはしない |
| A descriptor-only path選択 | 標準の条件付き実装選択・cost prediction・cache利用 | 非可換ordered PRのrun/control/cut特徴で、strong対照と同じpath lawを保ち、compile前に費用を抑えて選べるか | 真の安い側を知るoracle、保存価格の事後min、新たな分散/shot改善 |
| B L/R cut | unitaryを二laneへ分ける一般のsplit evolution、load balancing | 合法step境界のclosure/fusion損失とPR pathのrun不均衡を同時に扱う | 単純max(lane)の加法価格定理、未知cutに対するcompiled同値性 |
| C target-preserving接口融合 | Gaussian相対変換、同basis合併、compiler cancellation | 未実装のhead-tail接口をcontrol mode/gaugeまで含め共同処理できるか | StrongBuilderや既存step融合の名称変更、全P0最適性 |
| C law/PF/event変更 | PR近似/確率構造自体の設計 | 変更後のbias/attenuation/radius/Nも含む別研究課題 | 元targetを保存する局所最適化としての扱い |

「元Hが粒子数を保存する」だけで、任意の分割・実装に必要なreference条件を満たすとはしない。Track D B6bのwhole-ray一般条件と、既存DF checkerのhalf-ray/最終真空条件を守る。coherent QPE全phase registerの分布は、一回Hadamardまたはfresh-state RPEの条件付き同値性から推定しない。

現時点で支持される位置付けは、既知SEのPRへの応用と、既知の選択/分割/融合をPRの実装条件に合わせる改善候補である。独立した数学的・アルゴリズム的新機構は未確立。重要科学レビューや正式な採択は今回行っていない。
