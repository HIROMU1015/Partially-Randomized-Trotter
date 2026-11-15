# C向け受入規則案：Full-H κ2・ZZ置換対象0件のnormal path

**保存証拠のみでm0両軸の再判定を提案できる。ただし、coverage条件だけを適用外にする案では未保存parityが残る。独立な対象0件確認に加え、固定SOURCEでの非ZZ分岐同一性をparityの代替証拠として認めることを明示した案を推奨する。** 根拠はD1 `64624ea073dfc9afc32e2f09e5cd6b2eaadcc1d3` の [REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64624ea073dfc9afc32e2f09e5cd6b2eaadcc1d3/docs/research/d_full_h_m0_zz_evidence_20261115/REPORT.md) / [EVIDENCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64624ea073dfc9afc32e2f09e5cd6b2eaadcc1d3/docs/research/d_full_h_m0_zz_evidence_20261115/EVIDENCE.json)。旧runのparity・scalar/event・XY戻り値、runtime lowering marker、legacy QPYは未保存のまま。この案が採択されない場合、旧規則での受入0/2を維持する。

**これは提案であり正式採択・価格受入ではない。C_0X=C_0Y=null、全13round総額・部分額・割合・PFとの差もnullのまま。** 元観測X/Y各1062を単価へ登録せず、総額会計も起動しない。

## 開始点と役割

C `426b8c4f14b4a00869c73f7b2ae4ecdc638a0107` を開始点にする。同commitの受入0/2・保存24軸接続・総額null・人工/回帰18 test PASSは完了済みとして引用し、再実施しない。D1による今回の保存証拠照合を固定してから、本規則案を整理した。すべて主担当が実施したもので、別AI担当や独立reviewerによる審査を意味しない。

対象は元採択規則 `FULL_H_KAPPA2_M0_NORMAL_PATH_REFERENCE_V1`（`4a1cb0d7364fece475437749129fdce740fd8fc4`）。H4 linear1Å/STO-3G/保存tol-only DF10、one-body込みFull-H pure RTE・L_D0・finiteK2・κ2・P0、m0/T.1/q1/R9。新ZZ・symmetric-directional・測定付きfull Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を保持する。元SOURCE `7f85f20e38af4df1a649ac239e57bfe76d0899ee` と元結果 `b25d7da6dedebc4cd22e8d6f00e6de75d7072292` は変更しない。

## 提案する規則

提案ID `FULL_H_KAPPA2_NORMAL_PATH_ZERO_ZZ_ADMISSION_PROPOSAL_V1`。normal pathの経験的1 shot参照に限る。ZZ使用実績と、存在した操作の正しさを分離する。対象数Nは、置換前の全生成範囲について `controlled && nonidentity && role=rotation && support長2` から独立に確定する。置換counter、post-loweringのcrzz不在、Taylor orderのラベルだけでN=0と判定しない。

| 独立に確認した対象数 | 提案する扱い |
|---|---|
| N≥1 | 既存の全命令parity整合検査を適用する。PASS、置換数とNの一致、残存legacy crzzなし、その他の受入条件を要求する。不一致をcoverage不足へ読み替えない |
| N=0、全生成範囲を閉じられる | ZZ使用実績 `replaced_controlled_zz > 0` だけをNOT_APPLICABLEにする。下記の非ZZ整合性と残りの受入条件を維持する |
| N不明、範囲に未知head/opaque gate/不一致がある | 未受入・nullを維持する。counter0や数値gate PASSだけで補わない |

非ZZ整合性は、元の全命令parity PASSが保存されていればそれを用いる。未保存の場合、**次の条件をすべて満たすときに限り、固定SOURCEによる分岐同一性を代替証拠として認める案**とする。

1. 比較する固定legacy/new SOURCEの差がcontrolled ZZ分岐だけで、その他は同じ実装へ委譲される。実際のwrapperまでのcontext・入力・request・basis順・scalar・最適化patchの適用範囲を追跡できる。
2. 独立に確認したN=0により、その差分分岐が当該pathで一度も適用されないことを示せる。head/basis/外側wrapperから別の対象が追加されないことも確認する。
3. 保存full-wrapper QPYについて、未知の操作や制御経路がなく、scalar/event位相・X/Y同一U・global phase・測定軸/位置の読取照合が成立する。
4. 元の数値検査・SOURCE/physical/preparation/cohort/seed/compiler/fingerprint/軸別receiptの条件に欠測・不一致がない。runtime markerが未保存ならその事実を残し、固定SOURCEのactivation経路と保存wrapperの形による根拠を明示する。保存されなかったmarkerの実測値を創作しない。

これは**parity結果をPASSへ補完する規則ではなく、N=0に限定した代替証拠方式の提案**である。ZZ coverageの適用外化だけを採択する場合、この代替方式は自動で採択されない。

保持する残条件は、合法なnormal samplingと事前固定seed/order保持、元physical/H/state/preparation/component順、scalarを総Tで一回・event位相保持、control/P0/new loweringのSOURCE、X/Y別measurement/receipt/compile/profile、元index0の7数値検査、no-state-preparationの価格単位である。新規ZZ実装の正しさ一般や全law/母期待値/CIの資格を、この0対象pathから主張しない。残条件が不明なら該当軸をnullに保つ。

## 今回の保存pathへの適合状況（提案評価のみ）

| 条件 | D1の根拠と評価 |
|---|---|
| 対象0の独立根拠 | 全9 applicationがnonidentity/rotation/support1、head0、basisはsystemのみ、外側readoutはH/Sdg/H/measure。置換counterを根拠にしない |
| 非ZZ分岐同一性 | 01e068afのsubclassはsupport2だけ変更。今回の1-wireは元処理へ委譲。memoizationはnew/legacy双方に適用。同じrequests・prepからの処理で差分分岐がないというSOURCE根拠。実際のlegacy/new 2回路比較を再実施したものではない |
| scalar/event | 保存QPYを既存純粋関数で読む現在の技術照合が両軸PASS。ancilla CRZ9＋scalar P1、scalarはU末尾、event phaseは全+1 |
| X/Y | 現在の読取照合PASS。594命令のUとglobal phase0が一致、別のX/Y測定tailと各自のfingerprint/receiptを保持 |
| 元数値検査・input/compile binding | 保存7検査PASS・元context PASSを再利用。2 compiler JSON・4 QPYの元hash一致、元metricsを軸ごとに接続。再simulation・価格再計数なし |
| 未保存情報 | 元parity/scalar-event/XY戻り値・runtime marker・legacy QPYは欠落したまま。これらを元runのPASSとして記録しない |

以上より、**提案された代替証拠方式も正式に認めるなら、保存証拠だけで両軸の再判定を提案できる**。coverage条件だけの変更なら不足はparityの元記録/legacy QPYとruntime markerであり、現行未受入を維持する。新規build等による回収は今回の範囲外で、自動実行しない。

## 判断理由と適用限界

ZZ実装の使用実績は、ZZが存在する入力の処理確認には意味がある。一方、対象が存在しない合法なnormal drawに使用実績を要求すると、実装の誤りがなくても止まり、合法なpathの価格取得と実装coverageの目的が混ざる。対象0の独立根拠とその他の整合条件を残すことで、この二つを分離できる。

この判断は観測価格の大小やPF順位を条件にしない。本提案は観測後の明示的な規則変更案であり、観測前から採択されていたとは表示しない。ZZを含まない同種normal pathにも同じ条件を適用し、価格の大小による選別・再抽選を認めない。scope/head/SOURCE/未知gateが異なれば改めて範囲を閉じる必要がある。X/Yは同じ独立path1本であり2標本へ合算しない。

## 公開・引渡し・mandatory STOP

[PROPOSAL.json](PROPOSAL.json) に規則案・固定証拠locator・未保存情報・採択前のnull状態をまとめ、[HANDOFF.md](HANDOFF.md) を次の判断入口とする。正式規則やscientific caller、Cの会計readerは変更しない。旧STOPは有効な履歴のまま保持する。

今回のnew sampling/再抽選/build/compile/simulation/価格再計数/18 test再実施/会計runは0。元24軸・元F1総額nullはC開始点の記録を再利用する。C_0X/C_0Y・総額・m0部分額・割合・PF差をnullで維持し、正式採択・1062登録・全額集計へ進まず、commit/pushとremote照合後mandatory STOPする。ロードマップ本体は統合担当が更新する。
