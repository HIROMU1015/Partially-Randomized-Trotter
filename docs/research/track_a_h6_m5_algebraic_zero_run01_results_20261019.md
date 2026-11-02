# H6 m5：丸め0の代数的縮減を用いた一回実検証

2026-10-19。**固定した3 raw検査が全てPASSした。新しい代数的縮減を使ったH6 cosineの保存状態一つで、元の1e-9未満基準を維持できた。** 利用者の「もう一度実行して」に基づき、準備済みSOURCEでcosine一軸を新compile一回・保存状態作用一回だけ実行した。結果保存・終了監査後**mandatory STOP**、公開push0。

SOURCE：`0d47f59c351798434f7f29cdcbda993f60b5462c`。branch：`track-a-h6-m5-algebraic-zero-run01-20261019`。実行plan SHA256：`3e98c14338979155cd69b7dfab4a73bc4b3c12fff7c52f7034598731fa829267`。旧DRAFTと無効grantは上書きせず、新しい承認receipt・plan・grant・出力を作成した。準備DRAFTとのscope・SOURCE・入力・runtime・資源政策は一致し、承認結合と新しい保存先だけを更新した。

[準備・人工検証](track_a_m5_algebraic_oneq_readiness_20261019.md)、[summary](../../artifacts/resource_applicability/track_a_h6_m5_algebraic_zero_run01/2026-10-19/summary.json)、[raw検査](../../artifacts/resource_applicability/track_a_h6_m5_algebraic_zero_run01/2026-10-19/raw_gates.json)、[終了監査](../../artifacts/resource_applicability/track_a_h6_m5_algebraic_zero_run01/2026-10-19/end_audit.json)、[pass台帳](../../artifacts/resource_applicability/track_a_h6_m5_algebraic_zero_run01/2026-10-19/algebraic_passes.json)。

## 対象と固定条件

H6直鎖・隣接1 Å、STO-3G、保存tol-only1e-8・observed DF19。full `PF_M5_BEST_20D64C2_BINARY64_V1` p19/q2、T=.8、P=0、symmetric-directional control/full Hadamard cosine、13qubit/ancilla12、保存low状態一つ。Qiskit1.3.0・rz/sx/x/cx・preset opt1・seed17。新policyは`M5_OPT1_ALGEBRAIC_ONEQ_ZERO_ROUNDING_V1`であり、既定1qubit再合成の一箇所を正確なRZ和とSX/X恒等式へ置き換え、他のtaskとflow条件を維持した。

追加角度丸め予算は0。floatで正確に表せない角度和は元列を保持し、Euler再合成・小角度閾値・2π wrap・追加scalar生成を行わない。phase alignment・事後位相補正・状態の再normalizationは0。丸め予算0は浮動小数simulator誤差の保証ではないため、元のraw検査をそのまま実行した。

保存lowering QPY/vectorと初回PASSをSHA結合して再利用した。新lowering・変換後vectorの重複作用は0。旧runのSTOPは保持し、前回の再合成を無効にしたH6対照PASSも来歴へ結合した。

## raw精度の結果

| 検査 | 今回 | 保存raw差 | 判定 |
| --- | --- | ---: | --- |
| lowered_vs_saved_wrapper_vector | 再利用 | 2.326360070803798e-11 | PASS |
| compiled_vs_lowered | 新規 | 1.3066542295717234e-11 | PASS |
| compiled_vs_saved_wrapper_vector | 新規 | 1.5206807563089305e-11 | PASS |

全検査はfiniteかつ**厳密に1e-9未満**を要求する。これはこの保存状態・cosine一軸の結果であり、全状態・全演算・sine・他候補の検証ではない。過去の1qubit/Euler経路のどの処理がH6残差の唯一の原因だったかは認定しない。

## 回路縮減と実行証拠

入力および前回no-resynthesis対照の保存gate数はRZ2,855,001/SX696,962/X116,160/CX408,244/measure1。今回の診断gate数は`{'cx': 407894, 'measure': 1, 'rz': 1189845, 'sx': 679188, 'x': 107273}`である。これらは回路構造の診断値であり、受理済みの費用行やshot込み総費用ではない。

代数的passは2回、compiler callbackは24回。各passの正確な台帳で追加角度丸め0、global phase bit不変、Euler/閾値/wrapなしを確認した。作用前のguardも保存した。新policyの単価は、旧既定opt1・旧modelとの自動互換性を認めていない。

外部wallは448.232秒、peak RSSは4,054,940 KiB。launcher native exit=0、worker exit=0、全worker回収、自己signal0。段階別時間は[process metadata](../../artifacts/resource_applicability/track_a_h6_m5_algebraic_zero_run01/2026-10-19/stage_timing.json)に保存した。1compile→1作用の依存関係でproductive worker一つと独立監視を用い、記号的runのcacheを再利用した。host/上位cgroup PSIの一時上昇や他worktree更新で停止する仕組みは使用していない。

既存の人工30件PASSを再利用し、今回の起動前30項目と改変拒否12種、終了監査40項目を確認した。作者自身の検査であり、別AIによる独立レビューではない。監査は保存JSON/metadata/SHAの照合だけで、QPY/NPZ復元・vector再計算は0。

## 保全・採用範囲・次の担当

SOURCE/input/runtime 2016fileと、旧結果・grant・ledger・marker・STOP・準備案等2758fileをSHA照合した。既存58受理軸、V1/V2、他worktree・共有環境は変更していない。private QPY/vector・有効grantは公開していない。

新sampling・signal・SCF/DF/state/reference生成、sine、追加候補、fit、費用再会計・受理、自動retryは0。AD03のm5_best主比較採用は保留、共通プロトコル・他チャットの科学条件は変更していない。今回の一軸の結果だけで主比較の全受入を完了とはしない。

**ユーザー**：今回のraw精度と縮減結果を確認し、m5の残る検証・compiler政策の採否を判断する。

**Codex（別途承認後）**：判断された最小範囲だけについて、新SOURCE/入力・compiler identityと独立grantを準備する。追加軸・会計・主比較へ自動移行しない。

local commit後**MANDATORY STOP**。公開pushなし。
