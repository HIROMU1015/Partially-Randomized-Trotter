# Pauli B0停止後の最小条件（未実施・新実行未承認）

今回の1 attemptは終了し、run01/STOPを保全した。unused compile2件を次の許可と扱わない。実prepare/sample/compileを再び起動するには**ユーザーの別途明示承認**が必要である。

1. **同じ固定blobの展開を準備する。** `62ae1bd56718287de5d958cf0ed6411f5b7395c9:references/h46_m5_tol_pilot_inputs_v1.json`は存在する。sparse checkoutで欠けていただけなので、tracked referenceをそのSOURCEから展開するのが最小の技術対処。係数・H/scalar定義・compiler・gateを変更する理由はない。今回このmaterializationもしていない。
2. **実際のimport鎖を入力復元直前まで確認する。** fresh childで元install→provider SOURCE照合→`h46_m5_tol_native`等の実importを通す、数値ロード/prepareを呼ばない技術preflightを追加又は手順化する。関連するtracked reference依存が他にもあるかSOURCEから確認し、欠測で再び科学attemptを消費しない。既存metadata/toyテストだけで済ませない。private H/stateの再生成・科学的再診断は不要。
3. **新namespace/SOURCEを固定する。** caller62aeはpublic/privateともrun01固定で、現在そのpathは使用済み。再実行のためにrun01や結果を削除・上書きしない。新出力先（例：別のrun02 namespace）を受ける最小caller差分が必要なら、人工確認・新commitでSOURCEを固定し、ユーザーにその完全SHA・scopeを提示する。変更を黙って現科学許可へ含めない。実装/データ条件が同じでも、今回の失敗履歴を置き換えない。
4. **元の物理条件を維持する。** A714247/P04220、元configuration d905d219・physical RMSE:62a9a6…、T=.1/q1/R3/K2/184項、cohort/index0/seed14788159701244282917、route PR_PAULI_TAIL、D2 lowering、元scalar/control/compiler、X/Y個別というscopeを再指定する。B新Z1＋T1/Q3や別seed/roundへ代替しない。同cohort/seedを使う再attemptとして記録し、新独立標本とは呼ばない。
5. **次の一回の科学callを明示する。** 実H4復元/context＋native head、同normal1 path/3 event、full measured X/Y最大2 compile、独立full-2^n oracle・固定gate。失敗はpartial保存して再びSTOP。mean/CI/model/fit/N/G・B1や追加pathは含めない。

今回特定できたのはsparse checkout/import準備の不足だけであり、上記対処でその先の科学gateが通る保証はない。s_R実一致、scalar一回、native/compiled作用、価格C/metrics/fingerprintは次回の未取得証拠として残す。tolerance緩和、expected scalar置換、generic control/DF ZZへのfallbackはしない。

**ユーザー**が復旧技術作業と新しい有限実行の要否を判断する。**Codex D**は今回の結果commit/push・remote照合後mandatory STOPを維持する。
