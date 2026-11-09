# T1最小案：まずCX境界の保存人工trace比較（未実行）

2026-11-09 JST。**最初の対象はCX cancellationとその間のRZを含むnative界面とする。保存C7-2 traceを使う段階では新P0 compileは不要と見込む。** T1の採択・実行は今回行わない。担当ユーザー/GPTが範囲を判断し、指示後にCodexが実施する。

## 1. 最初に調べる接口と再利用証拠

[C7-2固定709d6ed9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/709d6ed9c234bf6d5a8848d3a30be69d88fd6476/artifacts/track_d_c7_2_toy/2026-11-07/run01)の `operation_traces.json` はcase別のraw/compiled、global_phase、ops、longest_path_operationsを持つ。今回読んだのはID/schemaだけで、T0 counterを保存traceへ適用していない。

初回T1は次の**3 case、各raw/compiledの6保存word**に限定する案：

- `cx_adjacent`：隣接inverse pair。
- `cx_commuting_control_rz`：間のcontrol側RZと実P0の扱い。
- `cx_noncommuting_target_rz`：target側RZと取消可否の違い。

rawが今回のnative domainに入ることを検査し、保存compiled側を同じT0 counterで数え、既存 `case_metrics.csv`／C7-2のcounterと照合する。保存opsは`[name, qubit_indices, parameters]`の三要素列である。宣言register幅をcase原SOURCE／保存metricsへbindし、空のcompiled wordから幅を推定しない。caseの科学的意味、既存P0の挙動とT0の「最適化しない」差を分類する。これはP0再実装ではなく、bridgeの最初の仕様化である。

次の候補は既存 `actual_inverse_pair`、`center_repeat_zz_controlled/physical`、`q2_same_reference_X`／`q2_same_shell_X`等。ただしrawには未展開gateが含まれ得る。未展開RZZ/CSWAP/Gaussianをnative一操作と数えず、T0対象外ならそのまま報告する。これらを初回3 caseへ自動追加しない。

## 2. 新P0が本当に必要になる場合

既存traceだけで境界問題を切り分けられる間は新compileをしない。固定frontierの連鎖相殺を新しいchunk配置で調べる必要があり、既存traceに同じ比較がなければ、次の**一種類・3 P0 call**を別途承認する最小案がある。

人工3qubitの確定native wordを

`CX(0,1), CX(1,2) | CX(1,2), CX(0,1)`

と固定し、全word一回と、縦棒で分けた二chunkを各一回compileする。**全word1＋chunk2＝3 call**で、軸・標本・分子条件を追加しない。既存P0／位相保持経路を再利用し、結果を予め0CX等へ固定しない。全compileと独立chunk連結の差を保存する。SOURCE・cut・precision・call数・STOPを実行前に定める。新P0の今回の実施数は0。

これはP0の局所処理を検討するための技術案で、同じP0との一般閉包や定理を3例で認定するものではない。追加family・本数を結果に応じて自動拡張しない。

## 3. 比較と不一致の分類

| 比較・原因 | 判定と扱い |
|---|---|
| 保存native wordのT0計数と既存metrics不一致 | T0/adapterのopcode・wire/classical mapping・size/filtered規約を確認。通常技術バグとして修正し、影響範囲だけ再試験 |
| 同一wordのchunk分割でT0値が変わる | transfer合成順・embedding・count会計の不具合。bridgeへ進まずSTOP |
| raw nativeと全P0の差 | 期待され得るoptimizer差。T0の誤計数と混同しない |
| 全P0と独立chunk P0連結の差 | 跨界面inverse cancellation、1q再合成、translation、反復での連鎖を分類。未解決残差を0にしない |
| Gaussian/parity/CSWAPを含む未対応raw | native展開が不足。勝手にlowering／価格補正を追加せずdomainを限定 |
| phase・SOURCE・公平なpolicy不整合 | 依存比較を止め、既存証拠を保全。閾値緩和やfixture交換をしない |

比較資源はCX/RZ count、各depth、total、size、幅を別々にする。三種類のarrivalを保持し、独立chunkのdepthを足すだけの方法を全P0との一致基準にしない。scalar/global phaseの意味は資源計数だけでは検証できず、必要な意味検証は別に明示する。

## 4. H0と長qへ進む条件

T1で人工CX界面が整理できても、H4 Gaussian・head/tail・step/global-cut・CSWAP/χの全P0 bridgeは残る。実H4へ主張を移す前には、別途H0の保存q≤64限定照合が必要となる見込みである。これは新しい科学的再解析なので、対象SOURCE、保存ケース、資源、実施回数を明示して承認する。新sampling／新P0が不要なら、それを優先する。

H0の有限一致も全law／全q証明ではない。native streaming emitterの意味、P0の有限interface閉包、絶対価格、長qの数値条件、law/joint-law、E[max]・bias・共分散が不足する。q128を機械的に次の地点にせず、必要な証拠から実測の要否を決める。A/F/S、N、全13-round会計、208−224qの未証明部分は変更しない。

**今回のT0は公開照合後mandatory STOP。次の担当はユーザー/GPT。CodexはT1／H0／新compile／長q検証／詳細レビューへ自動進行しない。**
