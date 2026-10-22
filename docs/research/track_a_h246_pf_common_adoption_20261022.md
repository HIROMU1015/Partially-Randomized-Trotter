# H2/H4/H6共通PF探索結果の正式採用・研究資料統合

2026-10-22 JST。**ユーザーの明示採択により、結果commit `90f1cd55a0d963953696bd0a27d9d041ff939bb7` の588候補・未解決0件を固定探索域の研究成果として採用。**
この変更は保存値の転記・研究資料・証拠索引の整備であり、新規科学計算・保存値再解析・再ランキング・sampling・compileは0。

## 研究データと証拠の接続

[採用bundle](../../artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/README.md)に、候補588件を科学入力identity、元signal、5精度endpoint2940行、取得来歴へ結合した。
公式S2/Yoshida4/m5_best、Full/B0/B0I、L_D、q、適格性、analytic shot、compiled RZ単価・総費用、SOURCE・終了監査は保存値のまま。
候補集合はH2 84/H4 228/H6 276、q≤8。全精度不適格候補と取得済み費用の未使用を明示する。

[v2補足](track_a_h246_pf_common_protocol_v2_supplement_20261022.md)を[v2末尾](track_a_pr_resource_evaluation_protocol_v2.md)へ接続し、正式条件や過去の契約は変更していない。
論文用には新しい[paper manifest](../../artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/paper_manifest.json)、
[CEM](../../artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/claim_evidence_matrix.md)、
[証拠台帳v4](../../artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/evidence_ledger_v4.csv)を追加する。
公開済み掲載資料commit `1923fce8f7461513d1bc47b5061e87cd70917e87` の旧CEM14行・旧台帳316行をbyte同一prefixとして保存し、新しい7主張・75証拠行を追記した。
旧paper manifest・1576行のpaper data manifest・CEM overlay・主表・Figure2全34fileは固定commit/SHAへの参照で引き継ぎ、既存版を書き換えない。
旧CEMの歴史的主張を新cohortへ自動適用しない。

## 採用する主張

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only DF1e-8の保存rank3/10/19、T=.8、P=0。
L_DはH2 B0/B0I0..2/Full3、H4 1..9/Full10、H6 5..15/Full19。
3公式、q1/2/4/8、δ=.8/.4/.2/.1、通常signed bias、共通経験的allowance、α_axis=.025、full cosine/sine wrapper、symmetric-directional、Qiskit1.3.0 rz/sx/x/cx opt1 seed17に限定する。

比較B（全公式）の保存点では、H2は全5精度でPF低費用。H4/H6はε=.001/.0005でPR低費用、それ以外の3精度でPF低費用。
H4の厳しい2精度での保存PR削減率は15.0068%/37.3736%、H6は31.1254%/40.9592%。
比較A（同一公式内）45行は別表で採用し、比較Bの15行と混同しない。
exact費用、保存平均・分散・SE・n、5精度全順位は[固定結果](track_a_h246_pf_common_final_20261022.md)と採用bundleの保存行が正本。

旧m5疎格子のみを分母とした比較から、全3公式のPF共通格子へ比較対象が変わった。
特にH6では、保存分解のm5 prefix拡張比は5精度とも1。S2/Yoshida4を含む公式拡張比は保存表示で.10742748/.33987932/.46896368/.87892018/1であり、緩い3精度ではPR低費用からPF低費用へ変わった。
費用や候補を再選択したのではなく、結果commitに保存済みの分解と順位変化をそのまま引用している。
高次PR対S2 PRの独立確認で示された削減率とも分母・cohortが異なる。

## 取得区分と監査上の限定

新規共通信号186、固有新価格378候補756軸、実compile予約累積776軸、同native保存価格の使用52pairは過去の実使用数。
今回の科学callは0。H2/H4の再生成容器とH6原snapshot復元を区別する。
H6不足63metadataは旧52signal維持・11raw新取得の別観測として記録する。
H6旧COSTは187pairを残して中断し、原parentのexit/reap・停止原因・正確な終了時刻は未確認。
保存完成pairの受入監査と、新grant22pairの正常終了を根拠とし、旧stageの正常終了を主張しない。旧STOP・grant・ledger・marker・原数値は保全した。

有限探索標本の選択効果は未補正。PRの保存n・平均・不偏分散・paired SEは独立確認と混合しない。
経験的allowanceは厳密certificateではない。未観測S4・希少eventのtail上界はnullであり0ではない。
母平均順位、大域最適性、PRの一般的優位、全energy/RPE総費用、実量子shot、noise/backend・FT資源を保証しない。

## 技術検証・残る判断

転記readerは既存のstdlib-only dataset helperを再利用し、固定Git blob/SHA、全588候補・5精度key、保存値の構造同一性、CEM/台帳prefix、プロトコル追記、リンク、旧証拠非変更を検査する。
DF12誤接続、候補/精度欠落、確認標本混合、旧終了状態の成功補完等を拒否する人工改変検査を行う。
[検証receipt](../../artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/technical_validation.json)は主担当によるローカル技術検査で、別担当の科学レビューやimmutable CIではない。

AD03の正式採択、既存主表・Figure2への掲載変更、最終evidence freeze、詳細科学レビューは次の別判断。
これらや追加科学実行は開始せず、専用branchへcommitし、確認済み公開資料と新しい採用文書の範囲だけを公開対象とする。
**完了後mandatory STOP。**
