# Track A：論文化前の最終技術確認（2026-10-19）

**A〜Eの静的技術調査と証拠整理を完了し、科学的未解決を保持してmandatory STOPした。**
[A〜D報告・表・SOURCE hash・E Matrix](../../artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/README.md)。本報告は公開用projectionであり、原調査のprivate運用記録・原入力・回路は含めない。

|項目|確認結果|
|---|---|
|A 希少event|完全基底変換後・target/backend/couplingなしの固定profileでL1を確認。L2は固定置換mapの加法性を確認したが全native展開数とlibrary閉包は未確認。Tier Rの数値上界・母平均保証は認定しない|
|B domain境界|保存winner762注記と新A960 endpoint状態を整理。座標sliceとq/r scheduleを区別。中間精度でもp/Kや次数別prefixの端があり、一律内部判定は不成立|
|C 非Clifford|H8 PFの保存QPY6本は存在しSHA一致したがcompile前wrapper。調査対象metric/cacheにfinal角度列はなく、既存compiled非Clifford数の再集計coverageは未確立|
|D H6非相殺|旧B0I12候補の分解欠測を元V1・α60入力・新Aへ照合。旧Dがeligible:nullを未確定一覧に入れないため、有限値は利用可能subset比較に限定。全登録比は未確定|
|E Claim–Evidence|6主張候補へ固定commit、結果、SOURCE、条件、実測/model、探索/確認、未解決、証拠の強さを対応。科学的な正式採択ではない|

主panelはlinear H4/H6/H8、隣接1Å、STO-3G、T=.8、保存Hamiltonian/state、P=0。H4はlegacy固定DF12、H6/H8はtol-only1e-8 DF19/28。full Hadamard-wrapper compiled RZ、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。H8原profile file bytesは未確認の継承識別子として保持する。

L1のTは基底変換を完全に終えた直後のgate総数であり、native命令数や理論RZ数ではない。controlled RZZの実経路はCCX–CRZ–CCXで、理論モデルのCX–CRZ–CXの展開数を流用できない。公式Qiskit tag1.3.0は896d1c8b72888a52a46081a6d1f4de8c56db8aeaへ固定してPython/Rustを静的確認した。

既存証拠では、決定論対照強化による点利益の縮小、H8の新A探索による両端の保存点の符号変更、中間精度の保存点削減、回路単価と解析shot負担の結合、bias会計の影響を記述できる。新A探索n8と確認union n32をpoolingせず、H4旧n32は選択・推定用fixed anchorとして区別する。確認union内の保存winnerも選択後保証や母平均winner認定ではない。

H6の旧B0I p5/10/15×q1/2/4/8 S2は、非相殺分解がなくeligible:nullである。Dの強化subset・倍率1の保存削減率22.2791/32.2240/37.0663/36.2830/20.1071%はsubset引用に限る。全136候補の最良比に読み替えない。新A倍率10の保存shot欠測320 endpointは別問題である。

新しい科学sampling、回路生成/compile、signal/SCF/DF/state生成、科学的数値再集計、shot再計算、再ランキング、希少上界計算、非Clifford数集計、独立確認、新科学レビューは0。固定6commitは実取得しGitHub SHA一致を確認、担当間重複を含む245件のSOURCE file hash照合は一致。公開SOURCEの実取得範囲とprivate記録の限定観測を区別する。

**有限標本点比較に限定した論文化準備は、新しい数値処理なしに進められる。** 厳密rare保証、非Clifford指標での結論維持、H6全登録非相殺比は未確認。必要とする場合だけ、数値Tier R、対象を絞った回路復元/compile/角度集計、欠測成分取得と限定再会計、機構図用の比の算出を別承認で固定する。高次PR探索やH8確認設計を主panelの独立確認結果と混同しない。

次の担当：**ユーザー**が主張の限定と執筆開始、必要な追加保証を判断する。**Codex**は新科学stageへ自動進行しない。
