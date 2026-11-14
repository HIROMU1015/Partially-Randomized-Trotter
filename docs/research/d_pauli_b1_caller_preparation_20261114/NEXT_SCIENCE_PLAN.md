# Pauli B1：採択前の有限実行案

**B1-W/V/CALの科学実行は未採択・未実行。mandatory STOP。** 以下はAの提案をcallerに対応付けたもので、承認済み予算ではない。範囲逸脱した保存P0演算子検査のPASSを科学的なpreflight証拠にしない。

| 候補段階 | 有限scopeの提案例 | 条件 |
|---|---|---|
| B1-W | LD2 m0 normal1 path、X/Y最大2 compile | canonical ID・LD2辞書/P0・scalar・SOURCE・seed/cohort・fresh outputの固定と明示許可 |
| B1-V（任意） | LD1 B0またはLD2 Wを元cohort/path0に事前宣言し、path1だけ追加、X/Y2 compile | 保存path0のSOURCE/ID/seed保全。独立n2新標本・holdout・較正として数えない |
| B1-CAL LD1 | Aの完全plan、最大24 compile | q1の3R各n2、独立q8 n2、pairedq2 n1、m5 n2/m6 n1。全CAL→C凍結→HOLDOUT |
| B1-CAL LD2 | 同じ構成、最大24 compile | 公開LD2 Wの配線証拠が必要。辞書はLD2専用 |

Aの順序提案は「W(LD2)→CAL(LD1)→資源観測→CAL(LD2)」、Vは任意。今回この順序や全48 compileを採択したわけではない。Wの成功1 pathで期待価格を認定しない。LD2の初回配線が必要なためWを提案しており、巨大m12のfull wrapperを必要条件にしていない。

実行前に決める内容：採択段階・candidate/physical ID・roundまたは非configuration cell・q/R/T/K・axes・独立/paired/path indices・cohort/step seed・実行SOURCE・input/P0辞書・新出力先・prepare/sampling/build/compile/oracleの科学call範囲・one-shot/再開条件。§10.5に従い既存table/catalog/alias、元trial/receipt、関連running jobを再照合する。B0/Wの未使用枠や旧grantを転用しない。

価格精度、confidence、許容誤差、freeze/holdout policy、law差/rare成分の扱いは未採択。ユーザーとA/C/Bの比較要求に応じて先に決める。callerのrank/domain検査は設計成立の技術条件であり、価格モデルの妥当性・長時間資格の代わりにはならない。Cのfit/係数凍結・holdout数値判定も許可範囲に明記する必要がある。

大R・長qでの実壁時計時間/RSSは未測定。B0のm0性能を新cellの実測と同一視しない。既存native/cacheを使い、独立cellを並列化できるが、親→paired・W→LD2較正・較正→freeze→holdoutの依存を守る。CPU/RAM/workerに恣意的な上限を置かず、共有設定・他process/fileを変更しない。

実行時はinput/scalar/source/gate不一致で部分結果を保全してSTOPし、別seed/round・追加path/compileへ進まない。成功後もfinite sampleのまま公開・C guard結合・STOPとする。実fit、m7〜m12外挿、E[C]/CI、N/Gは別途明示許可が必要。
