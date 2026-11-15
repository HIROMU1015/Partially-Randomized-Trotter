# 固定scope・§10.5差分照合

対象は保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、B0I p7/S2、要求・実幅binary64 `.1`、P0除外、configuration `e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc` の26軸。今回ユーザーが採用した `C_ref(q)=5513q+307` を比較用価格として使う。係数のfit・再選定は行わない。

## 再利用する正本

- 価格：`e04a14797a56a586dcb72065a55f4d97bfa1479b:docs/research/d_pf_p7_s2_h_direct_prices_20261115/run01/AXIS_PRICES.json`。m0〜m4の10軸だけが受入済み直接価格。元receipt・SOURCE・cohortは同表経由で保持する。
- 配分・物理schedule：`bec032fc26093177f529754060df06d5375944b4:docs/research/p7_s2_d01_same_shots_q3_results_20261115/C_RESULT_JOIN.json`。元configuration、physical ID、q/T/実幅、axis整数配分を照合する。
- 引用Q3と比較先費用：同commit・同ディレクトリの `SUMMARY.json`。Q3は保存MSE区間・適合statusをそのまま引用する。435,481,090 RZは幅16h/7だけの保存達成費用である。

## 差分照合の範囲と判断

Dの既存幅coverage・直接10軸HANDOFFを引き継いだ。関連D/Bの固定研究索引・現在branch、価格規則を記したmd/py、同じ幅の保存Q3成果を読取照合した。確認時点のremote branch HEADは以下で、対応するlocal refと一致した。

| branch | 観測HEAD |
|---|---|
| `codex/d-price-records-c-bridge-20261114` | `e04a14797a56a586dcb72065a55f4d97bfa1479b` |
| `codex/rpe-total-dp-acceleration-20261114` | `bec032fc26093177f529754060df06d5375944b4` |
| `codex/b2-pf6-allround-cost-20261115` | `738df226232e539cc084dfad2aed2feae82f3073` |

同じconfiguration・5513q+307・この固定配分を結合した後続26軸表／総費用成果は、この範囲では確認されなかった。`C_ref` の別研究（Track A/Pauli等）や現行幅16h/7の閉包は条件が異なり、今回のhモデルへ流用しない。同ユーザーの関連Python caller名（`run_pf_p7_h`、`p7_s2_d01`、今回comparison名）のrunning jobは確認0。未参照branch、別host、非公開archive全体の不存在は主張しない。新しい実回路取得は不要かつ今回の範囲外。

## 実行と受入の境界

`build_comparison.py` はstdlibで固定Git JSONを3件読むだけの限定join・整数積算。既存の直接価格受入、元native/wrapper/Accounting/compiler、元Q3 evaluatorは変更・再実行しない。実compiler evidenceのない外挿にC guardのPASSやdirect receiptを発行しない。既存重い科学会計APIは新Q3/N資格処理を含み、この単純な固定整数積算には不要である。

人工fixtureでaxis/ID/q/幅/配分の取り違えと直接／外挿の区別を検査し、scriptをcommitで固定してから本番の固定入力join・積算を一回行う。26行の処理にworker起動は不要。新規build/compile、QPY監査、fit、Q3、新配分生成は0。元の直接価格表・Cの精度索引のnull欄・旧STOPを変更せず、別namespaceへ保存する。

成功・不整合のいずれでも結果と理由を保全して公開後mandatory STOP。モデル誤差の未知は0にせず、他の幅・公式・候補や全方式winnerへ拡張しない。
