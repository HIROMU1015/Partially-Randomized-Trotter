# Figure 2A/2B更新案（描画・数値変更は未実施）

原本eb33de67の34ファイル（PNG/SVG/PDF、plot_values、data/render receipt等）をそのまま保持する。
Figure 2Aはlegacy H4 DF12/H6 DF19/H8 DF28の旧n32・探索n8を含む18比較、Figure 2BはH8独立n48の2比較。
原図116点の数値を変えず、新H4 DF10 Pへラベルを貼り替えない。

## 推奨配置案

- 2A：legacy/探索の機構分解図として保持。H4 DF12を明示し、新DF10独立確認の証拠から分離する。中間精度のPF modelはopen marker/注記を維持する。
- 2B：H8の固定独立n48比較を保持。既存Mechanism32内のH8_CONFIRM_1/3と同じ証拠であり、重複計数しない。
- 主表：新H4 DF10・H6・H8の独立確認6条件の保存平均・SE・PF費用・方向をmain_table_proposedに掲載する。全PF対照はこの6条件の直接実測に限定する。

## Caption追記案

「(A) 指定入力・登録比較集合における旧確認／探索標本の有限標本mechanism decomposition。H4はlegacy DF12で、本文主表のtol-only DF10独立確認とは異なる。中間精度の決定論modelは実測点と区別する。(B) H8 tol-only DF28・独立n48標本2条件に保存された分解。両panelはanalytic shots × compiled RZの指標であり、母平均の優位性、未観測希少eventの費用、最終energy/RPE総費用を保証しない。原図のA82未完了注記は描画時点の履歴で、説明因子は後続1ff6ac0bで既存32比較すべて保存済み。H8の詳細実行監査は公開receiptで再検証できていない。」

新H4/H6確認の機構説明が完成済みというcaptionにはしない。元画像内の履歴注記も今回は変更しない。

## 新しい確認図へ置換する場合に必要な別承認

H4 DF10 Pの2条件とH6独立確認2条件は、旧n8/n32の機構因子を転用できない。
最小入力は当該P cohortの保存axis別平均費用・保存shot/信号・同一PF対照のaxis値と、そのSOURCE/入力binding。
必要なら未保存集計の回収自体を独立承認対象とし、次にM_CORE/M_EXPLの比・norm・headroom・log因子等の数値解析を別承認で行う。
H4はG比・削減率すら新Pの保存値にないため今回空欄。新たな比、SE伝播、CI、誤差barを計算しない。
H8の因子は同一cohortの既存保存値を引用できるが、公開監査の不足は残る。
追加sampling・compileはこの案に含めず、必要性が生じた場合も別判断とする。

この数値作業は **REQUIRES_SEPARATE_NUMERICAL_AUTHORIZATION_NOT_RUN**。
配置案、入力手順差の明示、H8公開監査不足の扱いを利用者が判断するまで、原図の数値更新は行わない。mandatory STOP。
