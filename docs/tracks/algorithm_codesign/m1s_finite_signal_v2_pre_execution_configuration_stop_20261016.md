# M1-S有限step pilot v2：実行前設定不具合によるtechnical STOP（2026-10-16）

**PRE_EXECUTION_TECHNICAL_STOP_SOURCE_CONFIGURATION_PARSE / NO_SCIENTIFIC_RUN / mandatory STOP。**
ユーザーは正式契約C=`c6254be7c41d800a8ee417e8dc124f03ba6b382d`の初回correctness＋通過時108行pilotを一回だけ承認した。CPU/runtime/SOURCEのmetadata照合と専用authorization草稿の準備中に、固定SOURCEの精度文字列parse不具合が判明したため、authorizationを固定・発行せず起動前に停止した。

## 確認した不具合

採択SOURCE S=`fbc2edea3816bf8f68508f62c6da595631df03b6` の [pilot.py](../../../src/trottertracks/algorithm_codesign/m1s_finite_signal_v2/pilot.py) は44、104、127行で精度指定をFractionへ直接渡す。
採択済み文字列`2^-24`、`2^-18`はPython fractions.Fractionの受理する表記ではなく、固定Python3.11.0rc1でそれぞれValueErrorとなる。`1/4`のfloor指定はparseできる。
これは設定文字列のsyntax確認であり、Hamiltonian／state／Pauli／信号／誤差算定式の実入力評価や模擬科学入力のcorrectness検証ではない。
以前のSOURCE静的683 PASS・採択91 PASSはこの経路の実行可能性を検査しておらず、不具合を見落としていた。過去のPASS記録は原本保全するが、数値correctnessや実行可能性の証明として用いない。

H4 1.00 Å/STO-3G/8 modes/DF rank12、G3/G6/G9、delta1/8・1/16 Ha^-1、R2/4、T1/4 Ha^-1、zero込み108行という採択条件は変更しない。旧v4 STOP/null、v5・独立one-shot結果、消費済markerと旧契約・SOURCEも保全する。

## 実施範囲と消費状態

[実行前停止記録](../../../artifacts/track_b_m1s_finite_signal_v2_preflight_stop/2026-10-16/v1/technical_preflight_stop.json)と[準備状態](../../../artifacts/track_b_m1s_finite_signal_v2_preflight_stop/2026-10-16/v1/execution_readiness.json)を参照。
実行authorizationは草稿のみで未発行、authorization commitなし。外部排他marker・result directoryは作成していない。scientific worker、初回数値correctness gate、108行pilotの起動はすべて0回、科学one-shotは未開始・未消費である。数値実行失敗後のretryを行ったものではない。
CPU32・GPU NVIDIA T1000の存在をmetadata確認したが、採択計画はCPU1 worker・5 thread変数各1・GPUなし・4 GiB・result/log各128 MiBのまま。新しい数値benchmarkや速度向上の実測はない。
[草稿archive](../../../artifacts/track_b_m1s_finite_signal_v2_preflight_stop/2026-10-16/v1/unissued_draft_manifest.json)は受領した実行承認発言と計画を保存する。UNISSUED_DRAFTファイル中のplanned pathは未発行計画の履歴であり有効なauthorizationの参照ではない。草稿authorityはSOURCEが要求する実行authority namespaceへ配置していない。

## 最小修正案と次の境界

[未適用patch候補](../../../artifacts/track_b_m1s_finite_signal_v2_preflight_stop/2026-10-16/v1/PROPOSED_NOT_APPLIED_precision_literal_fix.patch)は、採択文字列を限定してexact Fraction(1,1<<24)／Fraction(1,1<<18)へ対応させ、対象3 call siteだけを変更する。元SOURCEには適用していない。
科学的閾値やcoverage、grid、モデル差会計を変更する案ではない。ただし採択済みSOURCEのhashは変わるため、現在の固定契約を維持したまま修正済みworkerを実行することはできない。
次は修正版SOURCEの新規固定・source pin更新契約候補を準備する技術作業が必要である。新pinの受理と、それを対象とする実行承認の範囲を明確にするまで科学計算へ進まない。追加GPT科学レビューは開始していない。
**今回の科学計算・Pauli生成・科学的数値correctness test・有限信号・simulation/compile・実行authorization発行・grant・markerは0件。mandatory STOP。**
